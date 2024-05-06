"""PyTorch TTT model."""

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import ModelOutput, logging
from .configuration_ttt import TttConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TttConfig"


class TttCache:
    def __init__(self, config, batch_size, dtype=torch.float32, device=None):
        self.seqlen_offset = 0
        self.dtype = dtype
        self.inner_chunk_size = config.inner_net_chunk_size

        self.params_pytree = defaultdict(dict)
        self.param_names = ["W1", "W2", "b1", "b2"]

    def update(self, py_tree, layer_idx, seq_len):
        # print('update', seq_len, self.inner_chunk_size, self.seqlen_offset)
        if seq_len % self.inner_chunk_size == 0:
            for name in self.param_names:
                self.params_pytree[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                self.params_pytree[f"{name}_grad"][layer_idx].zero_()
            # print('update seq_len % self.inner_chunk_size == 0')
        elif seq_len < self.inner_chunk_size:
            if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.inner_chunk_size != 0:
                raise ValueError("fractional update not supported yet.")
            if (seq_len + self.seqlen_offset) % self.inner_chunk_size == 0:
                for name in self.param_names:
                    self.params_pytree[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                    self.params_pytree[f"{name}_grad"][layer_idx].zero_()
                # print('update seq_len + self.self.seqlen_offset % self.inner_chunk_size == 0')
            else:
                for name in self.param_names:
                    self.params_pytree[f"{name}_grad"][layer_idx].copy_(py_tree[f"{name}_grad"])
        else:
            raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")
        # elif seq_len < self.inner_chunk_size:
        #     if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.inner_chunk_size != 0:
        #         raise ValueError("fractional update not supported yet.")

        #     if (seq_len + self.seqlen_offset) % self.inner_chunk_size == 0:
        #         for name in self.param_names:
        #             self.params_pytree[f'{name}_states'][layer_idx] += py_tree[f'{name}_grad']
        #             self.params_pytree[f'{name}_grad'][layer_idx].zero_()
        #         print('update seq_len + self.self.seqlen_offset % self.inner_chunk_size == 0')
        #     else:
        #         for name in self.param_names:
        #             # cusum weight grad
        #             self.params_pytree[f'{name}_grad'][layer_idx] += py_tree[f'{name}_grad']
        #         print('cumsum')
        # else:
        #     raise ValueError(f'seq_len {seq_len} is a partial update not supported yet')

    # for vmap
    def to_pytree(self, layer_idx):
        return {name: self.params_pytree[name][layer_idx] for name in self.params_pytree}


class TttRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        TttRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(TttRMSNorm)


class TttMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def diff_gelu(x):
    """Derivative of the GELU activation function."""
    # https://github.com/NVIDIA/Megatron-LM/blob/cafda9529d9956578014d4cb89b69b741702b514/megatron/model/fused_bias_gelu.py#L24
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


# Function to unpack tensors along the first dimension
def unpack_tensors(tensor_dict):
    # Determine the number of items to unpack (length of first dimension)
    num_items = next(iter(tensor_dict.values())).shape[0]

    # Initialize a list to hold the unpacked dictionaries
    unpacked_list = []

    for i in range(num_items):
        # Create a new dictionary for each item, slicing each tensor along the first dimension
        item_dict = {key: tensor[i].clone() for key, tensor in tensor_dict.items()}
        unpacked_list.append(item_dict)

    return unpacked_list


def scan(f, init, xs, length=None):
    """Minic jax.lax.scan function."""
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    if isinstance(xs, dict):
        xs = unpack_tensors(xs)
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)


class TttModule(nn.Module):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.inner_chunk_size = config.inner_net_chunk_size

        token_idx = 1.0 / torch.arange(1, self.inner_chunk_size + 1, dtype=torch.float32)
        self.register_buffer("token_idx", token_idx, persistent=False)

        self.wq = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

        self.W1 = nn.Parameter(
            torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, config.inner_net_intermediate_size))
        )
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, config.inner_net_intermediate_size))
        self.W2 = nn.Parameter(
            torch.normal(
                0,
                0.02 / math.sqrt(2 * self.config.num_hidden_layers),
                size=(self.num_heads, config.inner_net_intermediate_size, self.head_dim),
            )
        )
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        self.decoder_ln_fn = partial(F.layer_norm, normalized_shape=[self.head_dim], eps=1e-6)
        # prepending head dim
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ln_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ln_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

        self.gate_ilr_fn = F.linear
        # prepending head dim
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.linear_weight = nn.Parameter(
            torch.stack([torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)], dim=0)
        )
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        # init bias to 0 following original JAX impl.
        self.linear_bias = nn.Parameter(
            torch.stack([torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)], dim=0)
        )

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[TttCache] = None,
        inner_chunk_size: Optional[int] = None,
        last_chunk_params_pytree: Optional[Dict[str, torch.Tensor]] = None,
        return_params: Optional[bool] = False,
    ):
        batch = hidden_states
        B, L, C = batch.shape
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        if cache_params is not None:
            inner_chunk_step_offset = cache_params.seqlen_offset % self.inner_chunk_size
            # print('inner_chunk_step_offset', inner_chunk_step_offset)
        else:
            inner_chunk_step_offset = 0

        n_chunk = L // inner_chunk_size
        # [B ,n_chunk, inner_chunk_size, C]
        X = batch.reshape(B, n_chunk, inner_chunk_size, self.width)

        XA, XB, XC = self.wq(batch), self.wk(batch), self.wv(batch)

        XA = XA.reshape(B, n_chunk, inner_chunk_size, self.num_heads, self.head_dim)
        # [B, num_heads, n_chunk, inner_chunk_size, head_dim]
        XA = XA.permute(0, 3, 1, 2, 4)

        XB = XB.reshape(B, n_chunk, inner_chunk_size, self.num_heads, self.head_dim)
        # [B, num_heads, n_chunk, inner_chunk_size, head_dim]
        XB = XB.permute(0, 3, 1, 2, 4)

        XC = XC.reshape(B, n_chunk, inner_chunk_size, self.num_heads, self.head_dim)
        # [B, num_heads, n_chunk, inner_chunk_size, head_dim]
        XC = XC.permute(0, 3, 1, 2, 4)

        # [B, num_heads, n_chunk, inner_chunk_size, 1]
        ilr_gated = torch.vmap(self.gate_ilr_fn, in_dims=(None, 0, 0), out_dims=1)(
            X, self.linear_weight, self.linear_bias
        )
        ilr_gated = F.sigmoid(ilr_gated)

        if inner_chunk_size % self.inner_chunk_size == 0:
            token_idx = self.token_idx
        else:
            token_idx = self.token_idx[inner_chunk_step_offset : inner_chunk_step_offset + L]
        coeff = (self.config.inner_net_lr * token_idx).reshape(1, 1, 1, -1, 1) * ilr_gated / self.head_dim

        @torch.vmap
        def update_embed(XA, XB, XC, coeff, last_params_pytree=None):
            if last_params_pytree is not None:
                params_pytree = last_params_pytree
            else:
                params_pytree = {
                    "W1_states": self.W1,
                    "W2_states": self.W2,
                    "b1_states": self.b1,
                    "b2_states": self.b2,
                }
                if cache_params is not None and inner_chunk_size % self.inner_chunk_size != 0:
                    params_pytree.upadte(
                        {
                            "W1_grad": torch.zeros_like(self.W1),
                            "W2_grad": torch.zeros_like(self.W2),
                            "b1_grad": torch.zeros_like(self.b1),
                            "b2_grad": torch.zeros_like(self.b2),
                        }
                    )

            @torch.vmap
            def parallelize_over_heads(XA, XB, XC, coeff, init_params_pytree, ln_weight, ln_bias):
                def compute_chunk(params_pytree, inputs):
                    # W_init_chunk: [C, C]
                    W1_init = params_pytree["W1_states"]
                    W2_init = params_pytree["W2_states"]
                    b1_init = params_pytree["b1_states"]
                    b2_init = params_pytree["b2_states"]

                    # [inner_chunk_size, C]
                    XA_chunk = inputs["XA"]
                    # [inner_chunk_size, C]
                    XB_chunk = inputs["XB"]
                    # [inner_chunk_size, C]
                    XC_chunk = inputs["XC"]
                    # [inner_chunk_size, 1]
                    coeff_chunk = inputs["coeff"]

                    X1 = XB_chunk
                    Z1 = X1 @ W1_init + b1_init
                    X2 = F.gelu(Z1, approximate="tanh")
                    Z2 = X2 @ W2_init + b2_init

                    DLN_out, LN_vjp = torch.func.vjp(
                        lambda z: self.decoder_ln_fn(z, weight=ln_weight, bias=ln_bias), Z2
                    )

                    grad_l_wrt_DLN_out = DLN_out - XA_chunk  # [K,f]
                    grad_l_wrt_Z2 = LN_vjp(grad_l_wrt_DLN_out)[0]  # [K,f]
                    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(1, 0) * diff_gelu(Z1)

                    # NOTE: generation mode
                    # [L, C, C]
                    # outprod1 = XB_chunk[..., None] @ grad_l_wrt_Z1[..., None]
                    # last_outprod2 = sum(outprod1, dim=0) CxC
                    # last_outprod2 = sum(grad_l_wrt_Z1, dim=0) 1xC
                    # outprod2 = XB_chunk[..., None] @ grad_l_wrt_Z2[..., None]
                    # last_outprod2 = sum(outprod2, dim=0) CxC

                    # Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(1,0))
                    # b1_bar = b1_init - coeff_chunk * torch.cumsum(grad_l_wrt_Z1 , dim=0)  # [K,f]
                    # Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar

                    ## Decoding of chunk size=1
                    # grad_W_token = XB_chunk.transpose(1,0) @ grad_l_wrt_Z1 # [C, C]
                    # grad_b_token = grad_l_wrt_Z1 # [1, C]
                    # W_token = W1_init - coeff_chunk * (grad_W_prefix_sum + grad_W_token)
                    # b_token = b1_init - coeff_chunk * (grad_b_prefix_sum + grad_b_token)
                    # output_token = XC @ W_token + b_token

                    # NOTE: fractional forward, caching the gradients, and cumsum from cache
                    if cache_params is not None and inner_chunk_size % self.inner_chunk_size != 0:
                        # [K, C, C]
                        grad_W1 = (
                            torch.cumsum(torch.einsum("bi,bj->bij", XB_chunk, grad_l_wrt_Z1), dim=0)
                            + params_pytree["W1_grad"]
                        )
                        # [K, C]
                        grad_b1 = torch.cumsum(grad_l_wrt_Z1, dim=0) + params_pytree["b1_grad"]
                        b1_bar = b1_init - (coeff_chunk * grad_b1)  # [K, C]
                        # [K, C]
                        Z1_bar = (XC_chunk.unsqueeze(1) @ (W1_init - coeff_chunk.unsqueeze(-1) * grad_W1)).squeeze(
                            1
                        ) + b1_bar

                        X2_bar = F.gelu(Z1_bar, approximate="tanh")
                        grad_W2 = (
                            torch.cumsum(torch.einsum("bi,bj->bij", X2, grad_l_wrt_Z2), dim=0)
                            + params_pytree["W2_grad"]
                        )
                        grad_b2 = torch.cumsum(grad_l_wrt_Z2, dim=0) + params_pytree["b2_grad"]
                        b2_bar = b2_init - coeff_chunk * grad_b2  # [K, C]
                        Z2_bar = (X2_bar.unsqueeze(1) @ (W2_init - coeff_chunk.unsqueeze(-1) * grad_W2)).squeeze(
                            1
                        ) + b2_bar
                        XCW_chunk = Z2_bar

                        grad_pytree = {
                            "W1_grad": grad_W1[-1],
                            "W2_grad": grad_W2[-1],
                            "b1_grad": grad_b1[-1:],
                            "b2_grad": grad_b2[-1:],
                        }

                        W1_last = W1_init - (coeff_chunk[-1] * grad_W1[-1])
                        W2_last = W2_init - (coeff_chunk[-1] * grad_W2[-1])
                        b1_last = b1_bar[-1:]
                        b2_last = b2_bar[-1:]

                    else:
                        Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(1, 0))
                        b1_bar = b1_init - coeff_chunk * torch.cumsum(grad_l_wrt_Z1, dim=0)  # [K,f]
                        Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar

                        X2_bar = F.gelu(Z1_bar, approximate="tanh")
                        Attn2 = torch.tril(X2_bar @ X2.transpose(1, 0))
                        b2_bar = b2_init - coeff_chunk * torch.cumsum(grad_l_wrt_Z2, dim=0)  # [K,f]
                        Z2_bar = X2_bar @ W2_init - (coeff_chunk * Attn2) @ grad_l_wrt_Z2 + b2_bar
                        XCW_chunk = Z2_bar

                        # no grad if we are doing dual form, param is updated
                        grad_pytree = {
                            "W1_grad": torch.zeros_like(W1_init),
                            "W2_grad": torch.zeros_like(W2_init),
                            "b1_grad": torch.zeros_like(b1_init),
                            "b2_grad": torch.zeros_like(b2_init),
                        }

                        W1_last = W1_init - (coeff_chunk[-1] * X1).transpose(1, 0) @ grad_l_wrt_Z1
                        W2_last = W2_init - (coeff_chunk[-1] * X2).transpose(1, 0) @ grad_l_wrt_Z2
                        b1_last = b1_bar[-1:]
                        b2_last = b2_bar[-1:]

                    # print(batch.shape)
                    # print('W1_init', W1_init.sum(), W1_init.norm())
                    # print('W1_last', W1_last.sum(), W1_last.norm())
                    last_param_pytree = {
                        "W1_states": W1_last,
                        "W2_states": W2_last,
                        "b1_states": b1_last,
                        "b2_states": b2_last,
                    }
                    last_param_pytree.update(grad_pytree)

                    return last_param_pytree, XCW_chunk

                inputs = {"XA": XA, "XB": XB, "XC": XC, "coeff": coeff}
                # [n_chunk, inner_chunk_size, head_dim]
                output_params_pytree, XCW = scan(compute_chunk, init_params_pytree, inputs)
                return XCW.reshape(-1, self.head_dim), output_params_pytree

            # [num_heads, L, C]
            return parallelize_over_heads(XA, XB, XC, coeff, params_pytree, self.ln_weight, self.ln_bias)

        # in this case, we are decoding
        if last_chunk_params_pytree is None and cache_params is not None:
            last_chunk_params_pytree = cache_params.to_pytree(self.layer_idx)

        if last_chunk_params_pytree is not None:
            XCW_batch, batch_params_pytree = update_embed(XA, XB, XC, coeff, last_chunk_params_pytree)
        else:
            XCW_batch, batch_params_pytree = update_embed(XA, XB, XC, coeff)
        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_pytree, self.layer_idx, L)

        # [B, L, C]
        XCW_batch = XCW_batch.permute(0, 2, 1, 3).reshape(B, L, C)
        z_batch = self.wo(XCW_batch)

        if return_params:
            return z_batch, batch_params_pytree
        else:
            return z_batch

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[TttCache] = None,
    ):
        L = hidden_states.shape[1]
        reminder_len = L % self.inner_chunk_size
        num_chunks = L // self.inner_chunk_size
        output_hidden_states = []
        last_chunk_params_pytree = None
        if num_chunks > 0:
            chunk_hidden_states, last_chunk_params_pytree = self.forward_chunk(
                hidden_states[:, : num_chunks * self.inner_chunk_size],
                attention_mask=attention_mask[:, : num_chunks * self.inner_chunk_size]
                if attention_mask is not None
                else None,
                cache_params=cache_params,
                return_params=True,
            )
            output_hidden_states.append(chunk_hidden_states)
        if reminder_len > 0:
            output_hidden_states.append(
                self.forward_chunk(
                    hidden_states[:, -reminder_len:],
                    attention_mask=attention_mask[:, -reminder_len:] if attention_mask is not None else None,
                    cache_params=cache_params,
                    inner_chunk_size=reminder_len,
                    last_chunk_params_pytree=last_chunk_params_pytree,
                )
            )

        output_hidden_states = torch.cat(output_hidden_states, dim=1)

        return output_hidden_states


class TttDecoderLayer(nn.Module):
    def __init__(self, config: TttConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # TODO(jiarui): rename `self_attn` to ttt related?
        self.self_attn = TttModule(config=config, layer_idx=layer_idx)

        self.mlp = TttMLP(config)
        self.input_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[TttCache] = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # TTT
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache_params=cache_params,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TttPreTrainedModel(PreTrainedModel):
    config_class = TttConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TttDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class TttOutput(ModelOutput):
    """
    Class for the TTT model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`TttCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[TttCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TttCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`TttCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[TttCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TttModel(TttPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TttDecoderLayer`]

    Args:
        config: TttConfig
    """

    def __init__(self, config: TttConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TttDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TttCache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_params is None and use_cache:
            cache_params = self.create_cache(inputs_embeds.size(0), inputs_embeds.device, inputs_embeds.dtype)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__, hidden_states, attention_mask, cache_params
                )
            else:
                hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask, cache_params=cache_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return TttOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )

    def create_cache(self, batch_size, device, dtype) -> TttCache:
        logger.info(f"Creating cache of size: {batch_size}")
        print("create_cache")
        cache = TttCache(self.config, batch_size, dtype=dtype, device=device)
        # TODO(jiaruixu): hardcode for noe
        cache.param_names = ["W1", "W2", "b1", "b2"]
        for layer_idx in range(self.config.num_hidden_layers):
            for name in cache.param_names:
                weight = getattr(self.layers[layer_idx].self_attn, name)
                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim())
                cache.params_pytree[f"{name}_states"][layer_idx] = tiled_weight
                cache.params_pytree[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)

        return cache


class TttForCausalLM(TttPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = TttModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, cache_params: Optional[TttCache] = None, inputs_embeds=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TttCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        *,
        output_attentions: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert output_attentions is False, "output_attentions is not available in TttForCausalLM"

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        # print('logits', logits)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TttCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )
