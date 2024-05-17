"""PyTorch TTT model."""

from dataclasses import dataclass
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import einops
from torch.utils._pytree import tree_map

from ...activations import ACT2FN as _ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import ModelOutput, logging
from ...utils.import_utils import is_causal_conv1d_available
from .configuration_ttt import TttConfig

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None


class Clamp(nn.Module):
    def __init__(self, max_val):
        super().__init__()
        self.max_val = max_val
    def forward(self, x):
        return x.clamp(min=0, max=self.max_val)

ACT2FN = {
    "softplus": nn.Softplus(),
    "softplus_clip_5": nn.Sequential(nn.Softplus(), Clamp(max_val=5)),
    **_ACT2FN,
}

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TttConfig"


def diff_gelu(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff

class TttCache:
    def __init__(self, model, batch_size):
        config = model.config
        self.seqlen_offset = 0
        self.inner_chunk_size = config.inner_net_chunk_size

        self.params_dic = defaultdict(dict)
        if 'm1' in config.inner_net_type:
            self.inner_param_names = ["W1", "b1"]
        elif 'm2' in config.inner_net_type:
            self.inner_param_names = ["W1", "b1", "W2", "b2"]
        else:
            raise ValueError(f"inner_net_type {config.inner_net_type} not supported yet")
        
        self.conv_states_dic = defaultdict(dict)
        logger.info(f"Creating cache of size: {batch_size}")
        for layer_idx in range(config.num_hidden_layers):
            for name in self.inner_param_names:
                weight = getattr(model.layers[layer_idx].self_attn, name)
                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(model.device)
                self.params_dic[f"{name}_states"][layer_idx] = tiled_weight
                self.params_dic[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)
            if config.conv_before_ttt:
                self.conv_states_dic["conv_before_ttt"][layer_idx] = torch.zeros(batch_size, config.hidden_size, config.conv_kernel, device=model.device)
            if config.use_mixer and config.share_qk:
                self.conv_states_dic["ttt_conv_q"][layer_idx] = torch.zeros(batch_size, config.hidden_size, config.conv_kernel, device=model.device)
                self.conv_states_dic["ttt_conv_k"][layer_idx] = torch.zeros(batch_size, config.hidden_size, config.conv_kernel, device=model.device)

    def update(self, py_tree, layer_idx, seq_len):
        # print('update', seq_len, self.inner_chunk_size, self.seqlen_offset)
        if seq_len % self.inner_chunk_size == 0:
            for name in self.inner_param_names:
                self.params_dic[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                self.params_dic[f"{name}_grad"][layer_idx].zero_()
            # print('update seq_len % self.inner_chunk_size == 0')
        elif seq_len < self.inner_chunk_size:
            if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.inner_chunk_size != 0:
                raise ValueError("fractional update not supported yet.")
            if (seq_len + self.seqlen_offset) % self.inner_chunk_size == 0:
                for name in self.inner_param_names:
                    self.params_dic[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                    self.params_dic[f"{name}_grad"][layer_idx].zero_()
                # print('update seq_len + self.self.seqlen_offset % self.inner_chunk_size == 0')
            else:
                for name in self.inner_param_names:
                    self.params_dic[f"{name}_grad"][layer_idx].copy_(py_tree[f"{name}_grad"])
        else:
            raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")
    # for vmap
    def to_dic(self, layer_idx):
        return {name: self.params_dic[name][layer_idx] for name in self.params_dic}

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

class TttRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class TttLinearScalingRotaryEmbedding(TttRotaryEmbedding):
    """TttRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class TttDynamicNTKScalingRotaryEmbedding(TttRotaryEmbedding):
    """TttRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# https://github.com/young-geng/EasyLM/blob/main/EasyLM/models/llama/convert_easylm_to_hf.py#L141
def permute_qk(q, k):
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, head_dim//2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, head_dim//2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k

def undo_permute_qk(q, k):
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, 2, head_dim//2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, 2, head_dim//2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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

class TttConv(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            kernel_size=config.conv_kernel,
            groups=config.hidden_size,
            padding=config.conv_kernel - 1,
        )

    def __call__(self, hidden_states, cache_params=None):
        seq_len = hidden_states.shape[1]
        hidden_states = self.norm(hidden_states)
        # [B, C, L]
        hidden_states = hidden_states.transpose(1, 2)


        if causal_conv1d_fn is None:

            if cache_params is not None:
                if cache_params.seqlen_offset > 0:
                    conv_state = cache_params.conv_states_dic['conv_before_ttt'][self.layer_idx]
                    conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                    conv_state[:, :, -1] = hidden_states[:, :, 0]
                    cache_params.conv_states_dic['conv_before_ttt'][self.layer_idx].copy_(conv_state)
                    hidden_states = torch.sum(conv_state * self.conv.weight[:, 0, :], dim=-1)
                    hidden_states += self.conv.bias
                    hidden_states = hidden_states.unsqueeze(-1)
                else:
                    conv_state = nn.functional.pad(
                        hidden_states,
                        (self.config.conv_kernel - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states_dic['conv_before_ttt'][self.layer_idx].copy_(conv_state)
                    hidden_states = self.conv(hidden_states)[..., :seq_len]
            else:
                hidden_states = self.conv(hidden_states)[..., :seq_len]
        else:
            conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states_dic['conv_before_ttt'][self.layer_idx],
                    conv_weights,
                    self.conv.bias,
                    None,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.config.conv_kernel - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states_dic['conv_before_ttt'][self.layer_idx].copy_(conv_states)
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv.bias, activation=None
                )

        # [B, L, C]
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states


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


class LayerNormFunction(torch.autograd.Function):
    generate_vmap_rule = True
    @staticmethod
    def forward(input, gamma, beta, eps):
        N, D = input.shape

        # Mean and variance computation
        mu = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)

        # Normalization
        std = torch.sqrt(var + eps)
        x_hat = (input - mu) / std

        # Scale and shift
        y = gamma * x_hat + beta

        # Save variables for backward pass
        # ctx.save_for_backward(x_hat, gamma, std)
        # ctx.eps = eps

        return y, x_hat, std
    
    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        x, gamma, beta, eps = inputs
        y, x_hat, std = output
        ctx.save_for_backward(x_hat, gamma, std)
        ctx.eps = eps

    @staticmethod
    def backward(ctx, grad_output, grad_x_hat_, grad_std_):
        x_hat, gamma, std = ctx.saved_tensors
        N, D = grad_output.shape

        # Gradients for gamma and beta
        grad_gamma = (grad_output * x_hat).sum(dim=0)
        grad_beta = grad_output.sum(dim=0)

        # Gradient w.r.t. normalized data
        grad_x_hat = grad_output * gamma

        # Backpropagation through normalization
        grad_input = (
            (1 / D)
            * (
                D * grad_x_hat
                - grad_x_hat.sum(dim=-1, keepdim=True)
                - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
            )
            / std
        )

        return grad_input, grad_gamma, grad_beta, None

def ttt_layer_norm(input, weight, bias, eps=1e-6):
    return LayerNormFunction.apply(input, weight, bias, eps)[0]


# Usage
class TttLayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(features))
        self.bias = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, input):
        return ttt_layer_norm(input, self.weight, self.bias, self.eps)


class TttBaseModule(nn.Module):
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
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.inner_chunk_size = config.inner_net_chunk_size

        token_idx = 1.0 / torch.arange(1, self.inner_chunk_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        if self.config.use_learnable_token_idx:
            self.learnable_token_idx = nn.Parameter(torch.zeros((self.inner_chunk_size, self.inner_chunk_size)))

        # self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        # self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        # self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.share_qk = config.share_qk
        self.conv_kernel = config.conv_kernel
        self._init_qkvo_proj()
        self._init_rope()

        self.decoder_ln_fn = partial(F.layer_norm, normalized_shape=[self.head_dim], eps=1e-6)
        # self.decoder_ln_fn = ttt_layer_norm
        # prepending head dim
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        # ln_weight_data = TttLayerNorm(self.head_dim).weight.data
        self.ln_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        # ln_bias_data = TttLayerNorm(self.head_dim).bias.data
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

        self.use_mixer = config.use_mixer
        if self.use_mixer:
            self.g_proj = nn.Linear(self.width, self.width, bias=False)

        self.inner_net_on_residual = config.inner_net_on_residual
        self.use_post_ln = config.use_post_ln
        self.use_vjp = config.use_vjp

    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        if not self.share_qk:
            self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

        if self.share_qk:
            self.conv_q = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                kernel_size=self.conv_kernel,
                groups=self.hidden_size,
                padding=self.conv_kernel - 1,
            )
            self.conv_k = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                kernel_size=self.conv_kernel,
                groups=self.hidden_size,
                padding=self.conv_kernel - 1,
            )

    def _init_rope(self):
        self.rope_theta = self.config.rope_theta
        if self.config.rope_scaling is None:
            self.rotary_emb = TttRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.inner_chunk_size,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = TttLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.inner_chunk_size,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = TttDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.inner_chunk_size,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def get_qkv_projections(self, hidden_states, cache_params: Optional[TttCache] = None):
        if self.share_qk:
            xq, XA = self.q_proj(hidden_states), self.v_proj(hidden_states)
            seq_len = xq.shape[1]
            xq = xq.transpose(1, 2)
            if causal_conv1d_fn is None:
                if cache_params is not None:
                    if cache_params.seqlen_offset > 0:
                        conv_q_state = cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx]
                        conv_q_state = torch.roll(conv_q_state, shifts=-1, dims=-1)
                        conv_q_state[:, :, -1] = xq[:, :, 0]
                        cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx].copy_(conv_q_state)
                        XC = torch.sum(conv_q_state * self.conv_q.weight[:, 0, :], dim=-1)
                        XC += self.conv_q.bias
                        XC = XC.unsqueeze(-1)

                        conv_k_state = cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx]
                        conv_k_state = torch.roll(conv_k_state, shifts=-1, dims=-1)
                        conv_k_state[:, :, -1] = xq[:, :, 0]
                        cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx].copy_(conv_k_state)
                        XB = torch.sum(conv_k_state * self.conv_k.weight[:, 0, :], dim=-1)
                        XB += self.conv_k.bias
                        XB = XB.unsqueeze(-1)
                    else:
                        conv_q_state = nn.functional.pad(
                            xq,
                            (self.config.conv_kernel - xq.shape[-1], 0)
                        )
                        cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx].copy_(conv_q_state)
                        XC = self.conv_q(xq)[..., :seq_len]
                        conv_k_state = nn.functional.pad(
                            xq,
                            (self.config.conv_kernel - xq.shape[-1], 0)
                        )
                        cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx].copy_(conv_k_state)
                        XB = self.conv_k(xq)[..., :seq_len]
                else:
                    XC = self.conv_q(xq)[..., :seq_len]
                    XB = self.conv_k(xq)[..., :seq_len]
            else:
                conv_q_weights = self.conv_q.weight.view(self.conv_q.weight.size(0), self.conv_q.weight.size(2))
                conv_k_weights = self.conv_k.weight.view(self.conv_k.weight.size(0), self.conv_k.weight.size(2))
                if cache_params is not None and cache_params.seqlen_offset > 0:
                    XC = causal_conv1d_update(
                        xq.squeeze(-1),
                        cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx],
                        conv_q_weights,
                        self.conv_q.bias,
                        None,
                    )
                    XC = XC.unsqueeze(-1)
                    XB = causal_conv1d_update(
                        xq.squeeze(-1),
                        cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx],
                        conv_k_weights,
                        self.conv_k.bias,
                        None,
                    )
                    XB = XB.unsqueeze(-1)
                else:
                    if cache_params is not None:
                        conv_q_states = nn.functional.pad(
                            xq, (self.config.conv_kernel - xq.shape[-1], 0)
                        )
                        cache_params.conv_states_dic['ttt_conv_q'][self.layer_idx].copy_(conv_q_states)
                        conv_k_states = nn.functional.pad(
                            xq, (self.config.conv_kernel - xq.shape[-1], 0)
                        )
                        cache_params.conv_states_dic['ttt_conv_k'][self.layer_idx].copy_(conv_k_states)
                    XC = causal_conv1d_fn(
                        xq, conv_q_weights, self.conv_q.bias, activation=None
                    )
                    XB = causal_conv1d_fn(
                        xq, conv_k_weights, self.conv_k.bias, activation=None
                    )


            XC = XC.transpose(1, 2)
            XB = XB.transpose(1, 2)
        else:
            XC, XB, XA = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        return XC, XB, XA

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _split_chunks(self, hidden_states, inner_chunk_size=None):
        B, N, num_head, head_dim = hidden_states.shape
        # @xinhao: 2 means two chunks as a group to use gradient checkpointing
        # T=2048, optimal ckpt num = sqrt(T) ~= 45
        # Since CS=16, when 4 chunks are grouped, ckpt num = 2048 / 64 = 32, which is closest to 45
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size
        hidden_states = hidden_states.reshape(B, -1, inner_chunk_size, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )  # [B,nh,n_chunk,K,f]
        return hidden_states

    def get_coeff(self, X, inner_chunk_step_offset, inner_chunk_size):
        # [B, num_heads, n_chunk, inner_chunk_size, 1]
        ilr_gated = torch.vmap(self.gate_ilr_fn, in_dims=(None, 0, 0), out_dims=1)(
            X, self.linear_weight, self.linear_bias
        )
        ilr_gated = ACT2FN[self.config.inner_net_gate_activation](ilr_gated)

        self.coeff_transposed = False
        # if self.layer_idx == 0:
        #     print('inner_chunk_step_offset', inner_chunk_step_offset, inner_chunk_size)
        if self.config.use_learnable_token_idx:
            # [B, L, L]
            token_idx = self.learnable_token_idx[
                inner_chunk_step_offset : inner_chunk_step_offset + inner_chunk_size,
                inner_chunk_step_offset : inner_chunk_step_offset + inner_chunk_size,
            ] + self.token_idx[:, None]
            ilr_gated = ilr_gated.permute(0, 1, 2, 4, 3)
            self.coeff_transposed = True
        else:
            # [B, L]
            token_idx = self.token_idx[
                inner_chunk_step_offset : inner_chunk_step_offset + inner_chunk_size
            ]
            # ilr_gated = ilr_gated.permute(0, 1, 2, 4, 3)
            # self.coeff_transposed = True

        coeff = (self.config.inner_net_lr * token_idx).reshape(1, 1, 1, inner_chunk_size, -1) * ilr_gated / self.head_dim

        return coeff

    def gate_with_mixer(self, hidden_states, ttt_output):
        y = self.g_proj(hidden_states)
        y = F.gelu(y, approximate='tanh')
        output = y * ttt_output
        return output

    def prepare_inner_loop_chunk_inputs(self, inputs, inner_chunk_size, cache_params):
        XC = inputs['XC']
        XB = inputs['XB']
        XA = inputs['XA']
        X = inputs['X']
        B, L, C = X.shape
        n_chunk = L // inner_chunk_size
        # [B ,n_chunk, inner_chunk_size, C]
        X = X.reshape(B, n_chunk, inner_chunk_size, self.width)

        XC = XC.reshape(B, self.num_heads, L//inner_chunk_size, inner_chunk_size, self.head_dim)
        XB = XB.reshape(B, self.num_heads, L//inner_chunk_size, inner_chunk_size, self.head_dim)
        XA = XA.reshape(B, self.num_heads, L//inner_chunk_size, inner_chunk_size, self.head_dim)

        if cache_params is not None:
            inner_chunk_step_offset = cache_params.seqlen_offset % self.inner_chunk_size
        else:
            inner_chunk_step_offset = 0
        coeff = self.get_coeff(X, inner_chunk_step_offset, inner_chunk_size)
        inputs = {'XC': XC, 'XB': XB, 'XA': XA, 'coeff': coeff}
        return inputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
    ):
        B, L = hidden_states.shape[:2]
        reminder_len = L % self.inner_chunk_size
        num_chunks = L // self.inner_chunk_size
        last_chunk_params_dic = None

        XC, XB, XA = self.get_qkv_projections(hidden_states, cache_params=cache_params)

        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        XC = XC.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XB = XB.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XA = XA.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(XA, position_ids % self.inner_chunk_size)

        # permute_qk and undo_permute_qk is just for aligning pytorch with jax pre-training
        XC, XB = permute_qk(XC, XB)
        XC, XB = apply_rotary_pos_emb(XC, XB, cos, sin)
        XC, XB = undo_permute_qk(XC, XB)

        output_XCW_batch = []
        if num_chunks > 0:
            inputs = {
                "XC": XC[:, :, : num_chunks * self.inner_chunk_size],
                "XB": XB[:, :, : num_chunks * self.inner_chunk_size],
                "XA": XA[:, :, : num_chunks * self.inner_chunk_size],
                "X": hidden_states[:, : num_chunks * self.inner_chunk_size],
            }
            XCW_batch_chunk, last_chunk_params_dic = self.process_inner_loop(
                self.prepare_inner_loop_chunk_inputs(inputs, self.inner_chunk_size, cache_params),
                inner_chunk_size=self.inner_chunk_size,
                last_chunk_params_dic=last_chunk_params_dic,
                cache_params=cache_params,
            )
            output_XCW_batch.append(XCW_batch_chunk)
        if reminder_len > 0:
            inputs = {
                "XC": XC[:, :, -reminder_len:],
                "XB": XB[:, :, -reminder_len:],
                "XA": XA[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            XCW_batch_chunk, last_chunk_params_dic = self.process_inner_loop(
                self.prepare_inner_loop_chunk_inputs(inputs, reminder_len, cache_params),
                inner_chunk_size=reminder_len,
                last_chunk_params_dic=last_chunk_params_dic,
                cache_params=cache_params,
            )
            output_XCW_batch.append(XCW_batch_chunk)

        XCW_batch = torch.cat(output_XCW_batch, dim=1)
        if self.use_mixer:
            XCW_batch = self.gate_with_mixer(hidden_states, XCW_batch)
        z_batch = self.o_proj(XCW_batch)

        return z_batch


def decoder_ln_bwd(input, label, gamma, beta, eps=1e-6):
    D = input.shape[-1]
    mu = input.mean(dim=1, keepdim=True)
    var = input.var(dim=1, keepdim=True, unbiased=False)

    std = torch.sqrt(var + eps)
    x_hat = (input - mu) / std
    y = gamma * x_hat + beta

    grad_output = y - label
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=1, keepdim=True)
        )
        / std
    )

    return z


class TttM1Module(TttBaseModule):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic, cache_params: Optional[TttCache]=None):
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        B = inputs['XA'].shape[0]  # [B, nh, NC, CS, f]
        L = inputs['XA'].shape[2] * inputs['XA'].shape[3]

        @torch.vmap
        def update_embed(inputs, last_params_dic=None):
            if last_params_dic is not None:
                params_dic = last_params_dic
            else:
                params_dic = {
                    "W1_states": self.W1,
                    "b1_states": self.b1,
                }
                if cache_params is not None:
                    params_dic.upadte(
                        {
                            "W1_grad": torch.zeros_like(self.W1),
                            "b1_grad": torch.zeros_like(self.b1),
                        }
                    )

            @torch.vmap
            def parallelize_over_heads(inputs, init_params_dic, ln_weight, ln_bias):
                def compute_chunk(params_dic, inputs):
                    # W_init_chunk: [C, C]
                    W1_init = params_dic["W1_states"]
                    b1_init = params_dic["b1_states"]

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

                    if self.config.inner_net_on_residual:
                        reconstruction_target = XA_chunk - XB_chunk
                    else:
                        reconstruction_target = XA_chunk

                    if self.config.use_vjp:
                        DLN_out, LN_vjp = torch.func.vjp(
                            lambda z: self.decoder_ln_fn(z, weight=ln_weight, bias=ln_bias), Z1
                        )
                        # DLN_out, LN_vjp = torch.func.vjp(
                        #     lambda z: self.decoder_ln_fn(z, ln_weight, ln_bias, 1e-6), Z1
                        # )
                        grad_l_wrt_DLN_out = DLN_out - reconstruction_target  # [K,f]
                        grad_l_wrt_Z1 = LN_vjp(grad_l_wrt_DLN_out)[0]  # [K,f]
                    else:
                        grad_l_wrt_Z1 = decoder_ln_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

                    # NOTE: fractional forward, caching the gradients, and cumsum from cache
                    if cache_params is not None and inner_chunk_size % self.inner_chunk_size != 0:
                        # [K, C, C]
                        grad_W1 = (
                            torch.cumsum(torch.einsum("bi,bj->bij", XB_chunk, grad_l_wrt_Z1), dim=0)
                            + params_dic["W1_grad"]
                        )
                        # [K, C]
                        grad_b1 = torch.cumsum(grad_l_wrt_Z1, dim=0) + params_dic["b1_grad"]
                        b1_bar = b1_init - (coeff_chunk * grad_b1)  # [K, C]
                        # [K, C]
                        Z1_bar = (XC_chunk.unsqueeze(1) @ (W1_init - coeff_chunk.unsqueeze(-1) * grad_W1)).squeeze(
                            1
                        ) + b1_bar

                        W1_last = W1_init - (coeff_chunk[-1] * grad_W1[-1])
                        b1_last = b1_bar[-1:]

                        last_param_dic = {
                            "W1_states": W1_last,
                            "b1_states": b1_last,
                            "W1_grad": grad_W1[-1],
                            "b1_grad": grad_b1[-1:],
                        }
                    else:

                        Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(1, 0))
                        # b1_bar = b1_init - coeff_chunk * torch.cumsum(grad_l_wrt_Z1, dim=0)  # [K,f]
                        b1_bar = b1_init - (coeff_chunk * torch.tril(torch.ones_like(Attn1))) @ grad_l_wrt_Z1  # [K,f]
                        Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar

                        # W1_last = W1_init - (coeff_chunk[-1] * X1).transpose(1, 0) @ grad_l_wrt_Z1
                        # b1_last = b1_bar[-1:]
                        last_coeff_chunk = coeff_chunk[-1][:, None]
                        W1_last = W1_init - (last_coeff_chunk * X1).transpose(1, 0) @ grad_l_wrt_Z1
                        b1_last = b1_init - torch.sum(last_coeff_chunk * grad_l_wrt_Z1, dim=0, keepdim=True)

                        last_param_dic = {
                            "W1_states": W1_last,
                            "b1_states": b1_last,
                            "W1_grad": torch.zeros_like(W1_init),
                            "b1_grad": torch.zeros_like(b1_init),
                        }
                    if self.config.use_post_ln:
                        # TODO: accuracy is slightly off, need to investigate later
                        Z1_bar = self.decoder_ln_fn(Z1_bar, weight=ln_weight, bias=ln_bias)

                    if self.config.inner_net_on_residual:
                        XCW_chunk = XC_chunk + Z1_bar
                    else:
                        XCW_chunk = Z1_bar

                    return last_param_dic, XCW_chunk

                # [n_chunk, inner_chunk_size, head_dim]
                output_params_dic, XCW = scan(compute_chunk, init_params_dic, inputs)
                return XCW.reshape(-1, self.head_dim), output_params_dic

            # [num_heads, L, C]
            return parallelize_over_heads(inputs, params_dic, self.ln_weight, self.ln_bias)

        # in this case, we are decoding
        if last_chunk_params_dic is None and cache_params is not None:
            last_chunk_params_dic = cache_params.to_dic(self.layer_idx)

        if last_chunk_params_dic is not None:
            XCW_batch, batch_params_dic = update_embed(inputs, last_chunk_params_dic)
        else:
            XCW_batch, batch_params_dic = update_embed(inputs)

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dic, self.layer_idx, L)

        # [B, L, C]
        XCW_batch = XCW_batch.permute(0, 2, 1, 3).reshape(B, L, -1)

        return XCW_batch, batch_params_dic

def dln_bwd(input, label, ln_weight, ln_bias, use_vjp=True):
    dim = input.shape[-1]
    decoder_ln_fn = partial(F.layer_norm, normalized_shape=[dim], eps=1e-6)
    if use_vjp:
        DLN_out, LN_vjp = torch.func.vjp(
            lambda z: decoder_ln_fn(z, weight=ln_weight, bias=ln_bias), input
        )
        grad_l_wrt_DLN_out = DLN_out - label  # [K,f]
        grad_l_wrt_inp = LN_vjp(grad_l_wrt_DLN_out)[0]  # [K,f]
    else:
        grad_l_wrt_inp = decoder_ln_bwd(input, label, ln_weight, ln_bias)
    
    return grad_l_wrt_inp

def ln_fwd(x, gamma, beta, eps=1e-6):

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y

def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z


class TttM1BMMModule(TttBaseModule):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic, cache_params: Optional[TttCache]=None):
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        # in this case, we are decoding
        if last_chunk_params_dic is None and cache_params is not None:
            last_chunk_params_dic = cache_params.to_dic(self.layer_idx)

        B = inputs['XA'].shape[0]  # [B, nh, NC/g, g*CS, f]
        L = inputs['XA'].shape[2] * inputs['XA'].shape[3]
        if cache_params is not None and inner_chunk_size % self.inner_chunk_size != 0:
            # @xinhao: decoding
            def compute_chunk(params_dic, inputs):
                W1_init = params_dic["W1_states"]  # [B,nh,f,f]
                b1_init = params_dic["b1_states"]  # [B,nh,1,f]

                XA_chunk = inputs["XA"]  # [B,nh,K=1,f]
                XB_chunk = inputs["XB"]
                XC_chunk = inputs["XC"]
                coeff_chunk = inputs["coeff"]  # [B,nh,K=1,1]

                X1 = XB_chunk
                Z1 = X1 @ W1_init + b1_init  # [B,nh,K=1,f] @ [B,nh,f,f] -> [B,nh,K=1,f]
                if self.inner_net_on_residual:
                    reconstruction_target = XA_chunk - XB_chunk
                else:
                    reconstruction_target = XA_chunk

                ln_weight = self.ln_weight.reshape(self.num_heads, 1, self.head_dim)
                ln_bias = self.ln_bias.reshape(self.num_heads, 1, self.head_dim)
                grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B,nh,K,f]
                if self.coeff_transposed:
                    # TODO(jiarui): this is buggy, the correct impl needs too keep track all the grad within the same chunk, too nasty to impl, skip for now
                    logger.warning_once(
                        "`use_cache=True` with coeff_transposed is not correctly implemented yet, please set `use_cache=False` for matching."
                    )
                    coeff_chunk = torch.broadcast_to(coeff_chunk, (*coeff_chunk.shape[:2], inner_chunk_size, inner_chunk_size))

                    # [B, nh, K, f, f]
                    grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                    grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(coeff_chunk), grad_W1) + params_dic["W1_grad"].unsqueeze(2)
                    # [B, nh, K, f]
                    grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(coeff_chunk), grad_l_wrt_Z1) + params_dic["b1_grad"]

                    W1_bar = W1_init.unsqueeze(2) - grad_W1
                    b1_bar = b1_init - grad_b1

                else:
                    # [B, nh, K, f, f]
                    grad_W1 = torch.cumsum(torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1), dim=2) + params_dic["W1_grad"].unsqueeze(2)
                    # [B, nh, K, f]
                    grad_b1 = torch.cumsum(grad_l_wrt_Z1, dim=2) + params_dic["b1_grad"]  # [B,nh,K=1,f]

                    W1_bar = W1_init.unsqueeze(2) - grad_W1 * coeff_chunk.unsqueeze(-1)
                    b1_bar = b1_init - grad_b1 * coeff_chunk

                # [B, nh, K, 1, f] @ [B, nh, K, f, f]
                Z1_bar = (XC_chunk.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                
                if self.use_post_ln:
                    Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

                # XCW_chunk = Z1_bar
                if self.inner_net_on_residual:
                    XCW_chunk = XC_chunk + Z1_bar
                else:
                    XCW_chunk = Z1_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]

                last_param_dic = {
                    "W1_states": W1_last,
                    "b1_states": b1_last,
                    "W1_grad": grad_W1_last,
                    "b1_grad": grad_b1_last,
                }
                return last_param_dic, XCW_chunk

        else:
            def compute_chunk(params_dic, inputs):
                W1_init = params_dic["W1_states"]  # [B,nh,f,f]
                b1_init = params_dic["b1_states"]  # [B,nh,1,f]

                XA_chunk = inputs["XA"]  # [B,nh,K,f]
                XB_chunk = inputs["XB"]
                XC_chunk = inputs["XC"]
                coeff_chunk = inputs["coeff"]  # [B,nh,K,K]
                coeff_chunk = torch.broadcast_to(coeff_chunk, (*coeff_chunk.shape[:2], inner_chunk_size, inner_chunk_size))

                X1 = XB_chunk
                Z1 = X1 @ W1_init + b1_init  # [B,nh,K,f] @ [1,nh,f,f] + [B,nh,1,f] -> [B,nh,K,f]
                if self.config.inner_net_on_residual:
                    reconstruction_target = XA_chunk - XB_chunk
                else:
                    reconstruction_target = XA_chunk

                ln_weight = self.ln_weight.reshape(self.num_heads, 1, self.head_dim)
                ln_bias = self.ln_bias.reshape(self.num_heads, 1, self.head_dim)

                grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B,nh,K,f]

                Attn1 = torch.tril(XC_chunk @ X1.transpose(-2,-1))  # [B,nh,K,K]
                b1_bar = b1_init - torch.tril(coeff_chunk) @ grad_l_wrt_Z1  # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar  # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]

                if self.use_post_ln:
                    Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)
                if self.inner_net_on_residual:
                    XCW_chunk = XC_chunk + Z1_bar
                else:
                    XCW_chunk = Z1_bar

                last_coeff_chunk = coeff_chunk[:, :, -1, :, None]
                W1_last = W1_init - (last_coeff_chunk * X1).transpose(-1,-2) @ grad_l_wrt_Z1  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                b1_last = b1_init - torch.sum(last_coeff_chunk * grad_l_wrt_Z1, dim=-2, keepdim=True)  # [B,nh,1,f]

                last_param_dic = {
                    "W1_states": W1_last,
                    "b1_states": b1_last,
                    "W1_grad": torch.zeros_like(W1_init),
                    "b1_grad": torch.zeros_like(b1_init),
                }
                return last_param_dic, XCW_chunk

        if last_chunk_params_dic is not None:
            init_params_dic = last_chunk_params_dic
        else:
            init_params_dic = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dic.update(W1_grad=torch.zeros_like(init_params_dic["W1_states"]))
            init_params_dic.update(b1_grad=torch.zeros_like(init_params_dic["b1_states"]))
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f]
        batch_params_dic, XCW_batch = scan(compute_chunk, init_params_dic, inputs)  # [NC,B,nh,CS,f]


        ######################

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dic, self.layer_idx, L)

        # XCW_batch = XCW_batch.permute(1, 2, 0, 3, 4).reshape(B, L, -1)  # [B,L,f]
        XCW_batch = einops.rearrange(XCW_batch, "nc b nh cs f -> b (nc cs) (nh f)")  # [B,L,f]
        return XCW_batch, batch_params_dic


class TttM2BMMModule(TttBaseModule):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic, cache_params: Optional[TttCache]=None):
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        # in this case, we are decoding
        if last_chunk_params_dic is None and cache_params is not None:
            last_chunk_params_dic = cache_params.to_dic(self.layer_idx)

        B = inputs['XA'].shape[0]  # [B, nh, NC/g, g*CS, f]
        L = inputs['XA'].shape[2] * inputs['XA'].shape[3]
        if cache_params is not None and inner_chunk_size % self.inner_chunk_size != 0:
            # @xinhao: decoding
            def compute_chunk(params_dic, inputs):
                W1_init = params_dic["W1_states"]  # [B,nh,f,f]
                b1_init = params_dic["b1_states"]  # [B,nh,1,f]
                W2_init = params_dic["W2_states"]  # [B,nh,f,f]
                b2_init = params_dic["b2_states"]  # [B,nh,1,f]

                XA_chunk = inputs["XA"]  # [B,nh,K=1,f]
                XB_chunk = inputs["XB"]
                XC_chunk = inputs["XC"]
                coeff_chunk = inputs["coeff"]  # [B,nh,K=1,1]

                X1 = XB_chunk
                Z1 = X1 @ W1_init + b1_init  # [B,nh,K=1,f] @ [B,nh,f,f] -> [B,nh,K=1,f]
                X2 = F.gelu(Z1, approximate='tanh')
                Z2 = X2 @ W2_init + b2_init
                if self.inner_net_on_residual:
                    reconstruction_target = XA_chunk - XB_chunk
                else:
                    reconstruction_target = XA_chunk

                ln_weight = self.ln_weight.reshape(self.num_heads, 1, self.head_dim)
                ln_bias = self.ln_bias.reshape(self.num_heads, 1, self.head_dim)
                grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)  # [B,nh,K,f]
                # [B, nh, K, f]
                grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2,-1) * diff_gelu(Z1)
                if self.coeff_transposed:
                    # TODO(jiarui): this is buggy, the correct impl needs too keep track all the grad within the same chunk, too nasty to impl, skip for now
                    logger.warning_once(
                        "`use_cache=True` with coeff_transposed is not correctly implemented yet, please set `use_cache=False` for matching."
                    )
                    coeff_chunk = torch.broadcast_to(coeff_chunk, (*coeff_chunk.shape[:2], inner_chunk_size, inner_chunk_size))

                    # [B, nh, K, f, f]
                    grad_W2 = torch.einsum("bhki,bhkj->bhkij", X2, grad_l_wrt_Z2)
                    grad_W2 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(coeff_chunk), grad_W2) + params_dic["W2_grad"].unsqueeze(2)
                    # [B, nh, K, f]
                    grad_b2 = torch.einsum("bhnk,bhki->bhni", torch.tril(coeff_chunk), grad_l_wrt_Z2) + params_dic["b2_grad"]

                    # [B, nh, K, f, f]
                    grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                    grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(coeff_chunk), grad_W1) + params_dic["W1_grad"].unsqueeze(2)
                    # [B, nh, K, f]
                    grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(coeff_chunk), grad_l_wrt_Z1) + params_dic["b1_grad"]

                    W1_bar = W1_init.unsqueeze(2) - grad_W1
                    b1_bar = b1_init - grad_b1
                    W2_bar = W2_init.unsqueeze(2) - grad_W2
                    b2_bar = b2_init - grad_b2

                else:
                    # [B, nh, K, f, f]
                    grad_W2 = torch.cumsum(torch.einsum("bhki,bhkj->bhkij", X2, grad_l_wrt_Z2), dim=2) + params_dic["W2_grad"].unsqueeze(2)
                    # [B, nh, K, f]
                    grad_b2 = torch.cumsum(grad_l_wrt_Z2, dim=2) + params_dic["b2_grad"]  # [B,nh,K=1,f]

                    # [B, nh, K, f, f]
                    grad_W1 = torch.cumsum(torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1), dim=2) + params_dic["W1_grad"].unsqueeze(2)
                    # [B, nh, K, f]
                    grad_b1 = torch.cumsum(grad_l_wrt_Z1, dim=2) + params_dic["b1_grad"]  # [B,nh,K=1,f]

                    W1_bar = W1_init.unsqueeze(2) - grad_W1 * coeff_chunk.unsqueeze(-1)
                    b1_bar = b1_init - grad_b1 * coeff_chunk
                    W2_bar = W2_init.unsqueeze(2) - grad_W2 * coeff_chunk.unsqueeze(-1)
                    b2_bar = b2_init - grad_b2 * coeff_chunk

                # [B, nh, K, 1, f] @ [B, nh, K, f, f]
                Z1_bar = (XC_chunk.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate='tanh')
                Z2_bar = (X2_bar.unsqueeze(3) @ W2_bar).squeeze(3) + b2_bar
                
                if self.use_post_ln:
                    Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)

                # XCW_chunk = Z2_bar
                if self.inner_net_on_residual:
                    XCW_chunk = XC_chunk + Z2_bar
                else:
                    XCW_chunk = Z2_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                W2_last = W2_bar[:, :, -1]
                b2_last = b2_bar[:, :, -1:]
                # last_coeff_chunk = coeff_chunk[:, :, -1, :, None]
                # W1_last = W1_init - (last_coeff_chunk * X1).transpose(-1,-2) @ grad_l_wrt_Z1  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                # b1_last = b1_init - torch.sum(last_coeff_chunk * grad_l_wrt_Z1, dim=-2, keepdim=True)  # [B,nh,1,f]
                # W2_last = W2_init - (last_coeff_chunk * X2).transpose(-1,-2) @ grad_l_wrt_Z2  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                # b2_last = b2_init - torch.sum(last_coeff_chunk * grad_l_wrt_Z2, dim=-2, keepdim=True)  # [B,nh,1,f]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]
                grad_W2_last = grad_W2[:, :, -1]
                grad_b2_last = grad_b2[:, :, -1:]
                # if self.layer_idx==0:
                #     # print('decode W1_last', self.layer_idx, W1_last.shape, W1_last.sum())
                #     # print('decode W2_last', self.layer_idx, W2_last.shape, W2_last.sum())
                #     # print('decode b1_last', self.layer_idx, b1_last.shape, b1_last.sum())
                #     # print('decode b2_last', self.layer_idx, b2_last.shape, b2_last.sum())
                    # print('decode Z2_bar', Z2_bar.shape, Z2_bar[:, :, -1:].sum())
                    # print('decode W1_init', W1_init.shape, W1_init.sum())
                    # print('decode W2_init', W2_init.shape, W2_init.sum())
                #     print('decode Z1', Z1.shape, Z1.sum())
                #     print('decode Z2', Z2.shape, Z2.sum())
                #     # print('deocde grad_l_wrt_Z1', grad_l_wrt_Z1.shape, grad_l_wrt_Z1.sum())
                #     # print('decode grad_l_wrt_Z2', grad_l_wrt_Z2.shape, grad_l_wrt_Z2.sum())
                #     # print('decode coeff', coeff_chunk.shape, coeff_chunk.sum())

                last_param_dic = {
                    "W1_states": W1_last,
                    "b1_states": b1_last,
                    "W2_states": W2_last,
                    "b2_states": b2_last,
                    "W1_grad": grad_W1_last,
                    "b1_grad": grad_b1_last,
                    "W2_grad": grad_W2_last,
                    "b2_grad": grad_b2_last,
                }
                return last_param_dic, XCW_chunk

        else:
            def compute_chunk(params_dic, inputs):
                W1_init = params_dic["W1_states"]  # [B,nh,f,f]
                b1_init = params_dic["b1_states"]  # [B,nh,1,f]
                W2_init = params_dic["W2_states"]  # [B,nh,f,f]
                b2_init = params_dic["b2_states"]  # [B,nh,1,f]

                XA_chunk = inputs["XA"]  # [B,nh,K,f]
                XB_chunk = inputs["XB"]
                XC_chunk = inputs["XC"]
                coeff_chunk = inputs["coeff"]  # [B,nh,K,K]
                coeff_chunk = torch.broadcast_to(coeff_chunk, (*coeff_chunk.shape[:2], inner_chunk_size, inner_chunk_size))

                X1 = XB_chunk
                Z1 = X1 @ W1_init + b1_init  # [B,nh,K,f] @ [1,nh,f,f] + [B,nh,1,f] -> [B,nh,K,f]
                X2 = F.gelu(Z1, approximate='tanh')
                Z2 = X2 @ W2_init + b2_init
                if self.config.inner_net_on_residual:
                    reconstruction_target = XA_chunk - XB_chunk
                else:
                    reconstruction_target = XA_chunk

                # grad_l_wrt_Z2 = Z2 - XA_chunk  # [B,nh,K,f]
                ln_weight = self.ln_weight.reshape(self.num_heads, 1, self.head_dim)
                ln_bias = self.ln_bias.reshape(self.num_heads, 1, self.head_dim)

                grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)  # [B,nh,K,f]
                grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2,-1) * diff_gelu(Z1)

                Attn1 = torch.tril(XC_chunk @ X1.transpose(-2,-1))  # [B,nh,K,K]
                # b1_bar = b1_init - coeff_chunk * torch.cumsum(grad_l_wrt_Z1, dim=-2)  # [B,nh,1,f] - [B,nh,K,1] * [B,nh,K,f] -> [B,nh,K,f]
                # b1_bar = b1_init - (coeff_chunk * torch.tril(torch.ones_like(Attn1))) @ grad_l_wrt_Z1  # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                b1_bar = b1_init - torch.tril(coeff_chunk) @ grad_l_wrt_Z1  # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar  # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                X2_bar = F.gelu(Z1_bar, approximate='tanh')

                Attn2 = torch.tril(X2_bar @ X2.transpose(-2,-1))  # [B,nh,K,K]
                # b2_bar = b2_init - coeff_chunk * torch.cumsum(grad_l_wrt_Z2, dim=-2)  # [B,nh,1,f] - [B,nh,K,1] * [B,nh,K,f] -> [B,nh,K,f]
                # b2_bar = b2_init - (coeff_chunk * torch.tril(torch.ones_like(Attn2))) @ grad_l_wrt_Z2  # [B,nh,1,f] - [B,nh,K,1] * [B,nh,K,f] -> [B,nh,K,f]
                b2_bar = b2_init - torch.tril(coeff_chunk) @ grad_l_wrt_Z2  # [B,nh,1,f] - [B,nh,K,1] * [B,nh,K,f] -> [B,nh,K,f]
                Z2_bar = X2_bar @ W2_init - (coeff_chunk * Attn2) @ grad_l_wrt_Z2 + b2_bar  # [B,nh,K,f] @ [1,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                if self.use_post_ln:
                    Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)
                # XCW_chunk = Z2_bar  # [B,nh,K,f]
                if self.inner_net_on_residual:
                    XCW_chunk = XC_chunk + Z2_bar
                else:
                    XCW_chunk = Z2_bar

                # W1_last = W1_init - (coeff_chunk[:,:,-1:] * X1).transpose(-1,-2) @ grad_l_wrt_Z1  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                # b1_last = b1_bar[:,:,-1:]  # [B,nh,1,f]
                # W2_last = W2_init - (coeff_chunk[:, :, -1:] * X2).transpose(-1,-2) @ grad_l_wrt_Z2  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                # b2_last = b2_bar[:, :, -1:]  # [B,nh,1,f]
                last_coeff_chunk = coeff_chunk[:, :, -1, :, None]
                W1_last = W1_init - (last_coeff_chunk * X1).transpose(-1,-2) @ grad_l_wrt_Z1  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                b1_last = b1_init - torch.sum(last_coeff_chunk * grad_l_wrt_Z1, dim=-2, keepdim=True)  # [B,nh,1,f]
                W2_last = W2_init - (last_coeff_chunk * X2).transpose(-1,-2) @ grad_l_wrt_Z2  # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                b2_last = b2_init - torch.sum(last_coeff_chunk * grad_l_wrt_Z2, dim=-2, keepdim=True)  # [B,nh,1,f]
                # if self.layer_idx==0:
                #     # print('prefill W1_last', self.layer_idx, W1_last.shape, W1_last.sum())
                #     # print('prefill W2_last', self.layer_idx, W2_last.shape, W2_last.sum())
                #     # print('prefill b1_last', self.layer_idx, b1_last.shape, b1_last.sum())
                #     # print('prefill b2_last', self.layer_idx, b2_last.shape, b2_last.sum())
                    # print('prefill Z2_bar', Z2_bar.shape, Z2_bar[:, :, -1:].sum())
                    # print('prefill W1_init', W1_init.shape, W1_init.sum())
                    # print('prefill W2_init', W2_init.shape, W2_init.sum())
                #     print('prefill Z1', Z1.shape, Z1.sum())
                #     print('prefill Z2', Z2.shape, Z2.sum())
                #     # print('prefill grad_l_wrt_Z1', grad_l_wrt_Z1.shape, grad_l_wrt_Z1.sum())
                #     # print('prefill grad_l_wrt_Z2', grad_l_wrt_Z2.shape, grad_l_wrt_Z2.sum())
                #     # print('prefill coeff', coeff_chunk.shape, coeff_chunk.sum())

                last_param_dic = {
                    "W1_states": W1_last,
                    "b1_states": b1_last,
                    "W2_states": W2_last,
                    "b2_states": b2_last,
                    "W1_grad": torch.zeros_like(W1_init),
                    "b1_grad": torch.zeros_like(b1_init),
                    "W2_grad": torch.zeros_like(W2_init),
                    "b2_grad": torch.zeros_like(b2_init),
                }
                return last_param_dic, XCW_chunk

        if last_chunk_params_dic is not None:
            init_params_dic = last_chunk_params_dic
        else:
            init_params_dic = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dic.update(W1_grad=torch.zeros_like(init_params_dic["W1_states"]))
            init_params_dic.update(b1_grad=torch.zeros_like(init_params_dic["b1_states"]))
            init_params_dic.update(W2_grad=torch.zeros_like(init_params_dic["W2_states"]))
            init_params_dic.update(b2_grad=torch.zeros_like(init_params_dic["b2_states"]))
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f]
        batch_params_dic, XCW_batch = scan(compute_chunk, init_params_dic, inputs)  # [NC,B,nh,CS,f]


        ######################

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dic, self.layer_idx, L)

        # XCW_batch = XCW_batch.permute(1, 2, 0, 3, 4).reshape(B, L, -1)  # [B,L,f]
        XCW_batch = einops.rearrange(XCW_batch, "nc b nh cs f -> b (nc cs) (nh f)")  # [B,L,f]
        return XCW_batch, batch_params_dic


class TttDecoderLayer(nn.Module):
    def __init__(self, config: TttConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.conv_before_ttt = config.conv_before_ttt

        # TODO: rename self_attn to ttt
        if config.inner_net_type == "m1":
            ttt_module = TttM1Module
        elif config.inner_net_type == "m1_bmm":
            ttt_module = TttM1BMMModule
        elif config.inner_net_type == "m2":
            ttt_module = TttM2BMMModule
        elif config.inner_net_type == "m2_bmm":
            ttt_module = TttM2BMMModule
        else:
            raise ValueError(f"Invalid inner_net_type: {config.inner_net_type}")

        self.self_attn = ttt_module(config=config, layer_idx=layer_idx)

        self.mlp = TttMLP(config)
        if self.conv_before_ttt:
            self.conv = TttConv(config, layer_idx)

        self.input_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
    ):
        if self.conv_before_ttt:
            residual = hidden_states
            hidden_states = self.conv(hidden_states, cache_params=cache_params)
            hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # TTT
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        position_ids: Optional[torch.LongTensor] = None,
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
            cache_params = TttCache(self, inputs_embeds.size(0))

        seqlen_offset = 0
        if cache_params is not None:
            seqlen_offset = cache_params.seqlen_offset
        position_ids = torch.arange(
            seqlen_offset, seqlen_offset+ inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
        ).unsqueeze(0)

        # print('input_ids', input_ids.shape, 'inputs_embeds', inputs_embeds.shape, 'position_ids', position_ids.shape, 'attention_mask', attention_mask.shape)
        # embed positions
        hidden_states = inputs_embeds

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cache_params,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_params=cache_params,
                )

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
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, cache_params: Optional[TttCache] = None, inputs_embeds=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            attention_mask = attention_mask[:, -1].unsqueeze(-1) if attention_mask is not None else None

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
        assert not output_attentions, "output_attentions is not available in TttForCausalLM"

        # print('input_ids', input_ids.shape)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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


if __name__ == "__main__":
    from .configuration_ttt import TTT_STANDARD_CONFIGS
    # 125M
    ttt_config = TttConfig(**TTT_STANDARD_CONFIGS["125m"])
    ttt_model = TttForCausalLM(ttt_config)
    print(ttt_model(torch.ones((1, 2048), dtype=torch.long)))
    
    # 1.3B
    ttt_config = TttConfig(**TTT_STANDARD_CONFIGS["1b"])
    ttt_model = TttForCausalLM(ttt_config)
    print(ttt_model(torch.ones((1, 2048), dtype=torch.long)))
