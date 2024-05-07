"""PyTorch TTT model."""
import pdb
from dataclasses import dataclass
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import triton
import triton.language as tl

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import ModelOutput, logging
from .configuration_ttt import TttConfig

from transformers.models.ttt_benchmark_cg.generation import GenerationMixin, TttCache


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TttConfig"


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


def tree_map(fn, inputs):
    if isinstance(inputs, dict):
        out = {}
        for k, val in inputs.items():
            out[k] = fn(val)
    elif isinstance(inputs, list):
        out = []
        for val in inputs:
            out.append(fn(val))
    else:
        out = fn(inputs)
    return out


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
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.inner_chunk_size = config.inner_net_chunk_size

        token_idx = 1.0 / torch.arange(1, self.inner_chunk_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)

        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

        self.decoder_ln_fn = partial(F.layer_norm, normalized_shape=[self.head_dim], eps=1e-6)
        # prepending head dim
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ln_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ln_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

        self.get_ilr = nn.Linear(self.width, self.num_heads, bias=True)

        if config.use_compile:
            self.get_inner_loop_inputs = torch.compile(self._get_inner_loop_inputs)
        else:
            self.get_inner_loop_inputs = self._get_inner_loop_inputs

    def _split_chunks_heads(self, hidden_states, inner_chunk_size=None):
        B, N, F = hidden_states.shape
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size
        hidden_states = hidden_states.reshape(B, -1, inner_chunk_size, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )  # [B,nh,n_chunk,K,f]
        return hidden_states

    def _get_inner_loop_inputs(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        cache_params: Optional[TttCache] = None,
        inner_chunk_size: Optional[int] = None,
    ):
        batch = hidden_states
        B, L, C = batch.shape
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        if cache_params is not None:
            # TODO: keep recompiling when decoding
            inner_chunk_step_offset = cache_params.seqlen_offset % self.inner_chunk_size
            # print('inner_chunk_step_offset', inner_chunk_step_offset)
        else:
            inner_chunk_step_offset = 0

        n_chunk = L // inner_chunk_size
        X = batch.reshape(B, n_chunk, inner_chunk_size, self.width)  # [B ,n_chunk, inner_chunk_size, C]

        XC, XB, XA = self.q_proj(batch), self.k_proj(batch), self.v_proj(batch)  # [B,N,F]

        XC = self._split_chunks_heads(XC, inner_chunk_size)  # [B,nh,NC / g, g * CS,f]
        XB = self._split_chunks_heads(XB, inner_chunk_size)
        XA = self._split_chunks_heads(XA, inner_chunk_size)

        ilr_gated = self.get_ilr(X).permute(0,3,1,2).unsqueeze(-1)  # [B,NC,CS,nh] -> [B,nh,NC,CS,1]
        ilr_gated = F.sigmoid(ilr_gated)
        token_idx = self.token_idx[inner_chunk_step_offset : inner_chunk_step_offset + inner_chunk_size]  # [B, L]
        coeff = (self.config.inner_net_lr * token_idx).reshape(1, 1, 1, -1, 1) * ilr_gated / self.head_dim  # [B,nh,NC,CS,1]
        return XC, XB, XA, coeff

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
        inner_chunk_size: Optional[int] = None,
        last_chunk_params_dic: Optional[Dict[str, torch.Tensor]] = None,
        return_params: Optional[bool] = False,
        is_prefill: Optional[bool] = None,
        is_last_in_chunk: Optional[bool] = None,
    ):
        if cache_params is None:
            XC, XB, XA, coeff = self.get_inner_loop_inputs(
                hidden_states, position_ids=position_ids, cache_params=cache_params, inner_chunk_size=inner_chunk_size
            )
        else:
            # @xinhao: decoding time should not compile `get_inner_loop_inputs`. Otherwise will recompile every step.
            XC, XB, XA, coeff = self._get_inner_loop_inputs(
                hidden_states, position_ids=position_ids, cache_params=cache_params, inner_chunk_size=inner_chunk_size
            )
        inputs = {'XC': XC, 'XB': XB, 'XA': XA, 'coeff': coeff}
        XCW_batch, batch_params_dict = self.process_inner_loop(
            inputs,
            inner_chunk_size=inner_chunk_size,
            last_chunk_params_dic=last_chunk_params_dic,
            cache_params=cache_params,
            is_prefill=is_prefill, is_last_in_chunk=is_last_in_chunk,
        )
        # XCW_batch = XA + XB + XC; batch_params_dict = None  # @xinhao: for QKVO-MLP Only
        z_batch = self.project_inner_loop_outputs(XCW_batch)

        if return_params:
            return z_batch, batch_params_dict
        else:
            return z_batch

    def project_inner_loop_outputs(self, XCW_batch):
        """
        Inputs
            XCW_batch: [B,N,F]
        Outputs
            z_batch: [B,N,F]
        """
        z_batch = self.o_proj(XCW_batch)
        return z_batch

    def process_inner_loop(self,
                           inputs,
                           inner_chunk_size, last_chunk_params_dic,
                           is_prefill, is_last_in_chunk, cache_params=None,):
        """
        Inputs:
            XA, XB, XC: [B, n_chunk, chunk_size, F] or [B, n_chunk // 4, 4 * chunk_size, F]
            coeff: [B, n_chunk, chunk_size, 1] or [B,nh, n_chunk / 4, 4 * K, 1]
        Outputs:
            [B,N,F]
        """
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
        is_prefill: Optional[bool] = None,
        is_last_in_chunk: Optional[bool] = None,
    ):
        L = hidden_states.shape[1]
        reminder_len = L % self.inner_chunk_size
        num_chunks = L // self.inner_chunk_size
        output_hidden_states = []
        last_chunk_params_dic = None
        # @xinhao: decoding from a prompt of length 1 will not activate this
        # @xinhao: prefilling should only activate this
        if num_chunks > 0:
            chunk_hidden_states, last_chunk_params_dic = self.forward_chunk(
                hidden_states[:, : num_chunks * self.inner_chunk_size],
                position_ids=position_ids[:, : num_chunks * self.inner_chunk_size]
                if position_ids is not None
                else None,
                cache_params=cache_params,
                return_params=True,
                is_prefill=is_prefill,
                is_last_in_chunk=is_last_in_chunk,
            )
            output_hidden_states.append(chunk_hidden_states)

        # @xinhao: decoding from a prompt of length 1 will activate this
        # @xinhao: prefilling should not activate this
        if reminder_len > 0:
            output_hidden_states.append(
                self.forward_chunk(
                    hidden_states[:, -reminder_len:],
                    position_ids=position_ids[:, -reminder_len:] if position_ids is not None else None,
                    cache_params=cache_params,
                    inner_chunk_size=reminder_len,
                    last_chunk_params_dic=last_chunk_params_dic,
                    is_prefill=is_prefill,
                    is_last_in_chunk=is_last_in_chunk,
                )
            )

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        return output_hidden_states


def m1_prefill_chunk(W1_init, XA_chunk, XB_chunk, XC_chunk,
                     coeff_chunk, coeff_chunk_last):
    ###
    ## Legible logic
    ###
    # Z1 = XB_chunk @ W1_init
    # grad_l_wrt_Z1 = Z1 - XA_chunk
    # Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))
    # Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1
    # W1_last = W1_init - (coeff_chunk_last * XB_chunk).transpose(-1, -2) @ grad_l_wrt_Z1

    ###
    ## Compact logic
    ###
    Z1 = (XB_chunk @ W1_init).sub_(XA_chunk)  # [B*nh,K,f] @ [B*nh,f,f] -> [B*nh,K,f]
    Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))  # [B*nh,K,K]
    Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ Z1  # [B*nh,K,f] @ [B*nh,f,f] - ([B*nh,K,1] * [B*nh,K,K]) @ [B*nh,K,f]
    W1_init.sub_((coeff_chunk_last * XB_chunk).transpose(-1, -2) @ Z1)
    return W1_init, Z1_bar


def m1_decode_one_token(states, XA_chunk, XB_chunk, XC_chunk, coeff_chunk):
    ###
    ## Legible logic
    ###
    # X1 = XB_chunk
    # Z1 = X1 @ W1_init  # [B,nh,K=1,f] @ [B,nh,f,f] -> [B,nh,K=1,f]
    # grad_l_wrt_Z1 = Z1 - XA_chunk
    # grad_W1 = XB_chunk.transpose(-2, -1) @ grad_l_wrt_Z1 + params_dic["W1_grad"]  # [B,nh,f,f]
    # Z1_bar = XC_chunk @ (W1_init - coeff_chunk * grad_W1)  # [B,nh,K=1,f]
    # XCW_chunk = Z1_bar
    # W1_last = W1_init - coeff_chunk * grad_W1

    ###
    ## Compact logic
    ###
    W1_init = states['W1_states']
    W1_grad = states['W1_grad']
    Z1 = (XB_chunk @ W1_init).sub_(XA_chunk)  # [B*nh,1,f] @ [B*nh,f,f] -> [B*nh,1,f]
    W1_grad.add_(torch.einsum('bij,bik->bjk', XB_chunk, Z1))  # [B*nh,f,1] @ [B*nh,1,f] -> [B*nh,f,f]
    W1_init.sub_(coeff_chunk * W1_grad)
    Z1 = XC_chunk @ W1_init
    states['W1_states'] = W1_init
    states['W1_grad'] = W1_grad
    return states, Z1


class TttM1BMMModule(TttBaseModule):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))

        if self.config.use_compile:
            self.prefill_chunk = torch.compile(m1_prefill_chunk)
            self.decode_chunk = torch.compile(m1_decode_one_token)
        else:
            self.prefill_chunk = m1_prefill_chunk
            self.decode_chunk = m1_decode_one_token

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        B, NH, NC, CS, HF = inputs['XA'].shape
        input_dtype = inputs['XA'].dtype
        L = NC * CS
        inputs.update(coeff_chunk_last=inputs['coeff'][...,-1:])  # [B,nh,NC,CS,1] -> [B,nh,NC,1,1]

        if is_prefill:

            def for_loop(W1_init, inputs):
                output_tensor = torch.empty(size=(NC, B * NH, CS, HF), device=W1_init.device, dtype=input_dtype)
                for i in range(NC):
                    # TODO: select is slow
                    XA_chunk = inputs["XA"][i]  # [B*nh,K,f]
                    XB_chunk = inputs["XB"][i]
                    XC_chunk = inputs["XC"][i]
                    coeff_chunk = inputs["coeff"][i]  # [B*nh,K,1]
                    coeff_chunk_last = inputs["coeff_chunk_last"][i]  # [B*nh,1,1]

                    W1_init, Z1_bar = self.prefill_chunk(W1_init,
                                                         XA_chunk, XB_chunk, XC_chunk,
                                                         coeff_chunk, coeff_chunk_last)
                    output_tensor[i] = Z1_bar
                return W1_init, output_tensor  # [NC, B*nh, K, f]

            inputs = tree_map(lambda x: x.permute(2,0,1,3,4).reshape(NC, B * NH, CS, -1), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f] -> [NC,B*nh,CS,f]
            batch_params_dict, XCW_batch = for_loop(
                torch.tile(self.W1, dims=(B,1,1)),  # [B*nh,f,f], cloned from W1, safe for in-place op
                inputs,                             # [NC,B,nh,CS,f]
            )

        else:

            def decode_one_token(states, inputs):
                XA_chunk = inputs["XA"]        # [B*nh,K,f]
                XB_chunk = inputs["XB"]
                XC_chunk = inputs["XC"]
                coeff_chunk = inputs["coeff"]  # [B*nh,K,1]

                states, Z1 = self.decode_chunk(states, XA_chunk, XB_chunk, XC_chunk, coeff_chunk)
                return states, Z1  # [B*nh, f]

            states = {
                "W1_states": cache_params.params_dict["W1_states"][self.layer_idx],
                "W1_grad": cache_params.params_dict["W1_grad"][self.layer_idx],
            }
            inputs = tree_map(lambda x: einops.rearrange(x, 'b nh nc cs f -> nc (b nh) cs f')[0], inputs)
            batch_params_dict, XCW_batch = decode_one_token(
                states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                inputs,  # [B*nh,f]
            )
            XCW_batch = XCW_batch.unsqueeze(0)  # [NC=1,B*nh,CS=1,f]

        if cache_params is not None:
            # @xinhao: can skip this for model() forward test by setting cache_params=None
            # As for prefill in .generate(), will not skip the below
            if is_last_in_chunk:
                cache_params.update_last_in_chunk(batch_params_dict, self.layer_idx)
            else:
                cache_params.update_non_last_in_chunk(batch_params_dict, self.layer_idx)

        XCW_batch = einops.rearrange(XCW_batch, "nc (b nh) cs f -> b (nc cs) (nh f)", b=B, nh=NH)  # [B,L,f]

        return XCW_batch, batch_params_dict


def m2_prefill_chunk(states, XA_chunk, XB_chunk, XC_chunk,
                     coeff_chunk, coeff_chunk_last):
    ###
    ## Legible logic
    ###
    # Z1 = XB_chunk @ W1_init
    # Z2 = Z1 @ W2_init
    # grad_l_wrt_Z2 = Z2 - XA_chunk
    # grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-1, -2)
    #
    # Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))
    # Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1
    # W1_last = W1_init - (coeff_chunk_last * XB_chunk).transpose(-1, -2) @ grad_l_wrt_Z1
    #
    # Attn2 = torch.tril(Z1_bar @ Z1.transpose(-1, -2))
    # Z2_bar = Z1_bar @ W2_init - (coeff_chunk * Attn2) @ grad_l_wrt_Z2
    # W2_last = W2_init - (coeff_chunk_last * Z1).transpose(-1, -2) @ grad_l_wrt_Z2

    ###
    ## Compact logic
    ###
    W1_init = states['W1_states']
    W2_init = states['W2_states']

    Z1 = XB_chunk @ W1_init
    Z2 = Z1 @ W2_init
    grad_l_wrt_Z2 = Z2 - XA_chunk
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-1, -2)

    Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))
    Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1
    W1_init.sub_((coeff_chunk_last * XB_chunk).transpose(-1, -2) @ grad_l_wrt_Z1)

    Attn2 = torch.tril(Z1_bar @ Z1.transpose(-1, -2))
    Z2_bar = Z1_bar @ W2_init - (coeff_chunk * Attn2) @ grad_l_wrt_Z2
    W2_init.sub_((coeff_chunk_last * Z1).transpose(-1, -2) @ grad_l_wrt_Z2)

    states['W1_states'] = W1_init
    states['W2_states'] = W2_init
    return states, Z2_bar


def m2_decode_one_token(states, XA_chunk, XB_chunk, XC_chunk, coeff_chunk):
    ###
    ## Legible logic
    ###
    # X1 = XB_chunk
    # Z1 = X1 @ W1_init  # [B,nh,K=1,f] @ [B,nh,f,f] -> [B,nh,K=1,f]
    # grad_l_wrt_Z1 = Z1 - XA_chunk
    # grad_W1 = XB_chunk.transpose(-2, -1) @ grad_l_wrt_Z1 + params_dic["W1_grad"]  # [B,nh,f,f]
    # Z1_bar = XC_chunk @ (W1_init - coeff_chunk * grad_W1)  # [B,nh,K=1,f]
    # XCW_chunk = Z1_bar
    # W1_last = W1_init - coeff_chunk * grad_W1

    ###
    ## Compact logic
    ###
    W1_init = states['W1_states']
    W1_grad = states['W1_grad']
    W2_init = states['W2_states']
    W2_grad = states['W2_grad']

    Z1 = XB_chunk @ W1_init  # [B,nh,K=1,f] @ [B,nh,f,f] -> [B,nh,K=1,f]
    Z2 = Z1 @ W2_init

    grad_l_wrt_Z2 = Z2 - XA_chunk
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-1,-2)

    W1_grad = XB_chunk.transpose(-1,-2) @ grad_l_wrt_Z1 + W1_grad  # [B,nh,f,f]
    Z1_bar = XC_chunk @ (W1_init - coeff_chunk * W1_grad)  # [B,nh,K=1,f]

    W2_grad = Z1.transpose(-1,-2) @ grad_l_wrt_Z2 + W2_grad  # [B,nh,f,f]
    Z2_bar = Z1_bar @ (W2_init - coeff_chunk * W2_grad)  # [B,nh,K=1,f]

    W1_init.sub_(coeff_chunk * W1_grad)
    W2_init.sub_(coeff_chunk * W2_grad)

    states['W1_states'] = W1_init
    states['W1_grad'] = W1_grad
    states['W2_states'] = W2_init
    states['W2_grad'] = W2_grad
    return states, Z2_bar


class TttM2BMMModule(TttBaseModule):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))

        if self.config.use_compile:
            self.prefill_chunk = torch.compile(m2_prefill_chunk)
            self.decode_chunk = torch.compile(m2_decode_one_token)
        else:
            self.prefill_chunk = m2_prefill_chunk
            self.decode_chunk = m2_decode_one_token

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        B, NH, NC, CS, HF = inputs['XA'].shape
        input_dtype = inputs['XA'].dtype
        input_device = inputs['XA'].device
        L = NC * CS
        inputs.update(coeff_chunk_last=inputs['coeff'][...,-1:])  # [B,nh,NC,CS,1] -> [B,nh,NC,1,1]

        if is_prefill:

            def for_loop(states, inputs):
                output_tensor = torch.empty(size=(NC, B * NH, CS, HF), device=input_device, dtype=input_dtype)
                for i in range(NC):
                    # TODO: select is slow
                    XA_chunk = inputs["XA"][i]  # [B*nh,K,f]
                    XB_chunk = inputs["XB"][i]
                    XC_chunk = inputs["XC"][i]
                    coeff_chunk = inputs["coeff"][i]  # [B*nh,K,1]
                    coeff_chunk_last = inputs["coeff_chunk_last"][i]  # [B*nh,1,1]

                    states, Z1_bar = self.prefill_chunk(states,
                                                        XA_chunk, XB_chunk, XC_chunk,
                                                        coeff_chunk, coeff_chunk_last)
                    output_tensor[i] = Z1_bar
                return states, output_tensor  # [NC, B*nh, K, f]

            inputs = tree_map(lambda x: x.permute(2,0,1,3,4).reshape(NC, B * NH, CS, -1), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f] -> [NC,B*nh,CS,f]
            states = {
                'W1_states': torch.tile(self.W1, dims=(B,1,1)),
                'W2_states': torch.tile(self.W2, dims=(B,1,1)),
            }
            batch_params_dict, XCW_batch = for_loop(
                states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                inputs,  # [NC,B,nh,CS,f]
            )

        else:

            def decode_one_token(states, inputs):
                XA_chunk = inputs["XA"]        # [B*nh,K,f]
                XB_chunk = inputs["XB"]
                XC_chunk = inputs["XC"]
                coeff_chunk = inputs["coeff"]  # [B*nh,K,1]

                states, Z1 = self.decode_chunk(states, XA_chunk, XB_chunk, XC_chunk, coeff_chunk)
                return states, Z1  # [B*nh, f]

            states = {
                "W1_states": cache_params.params_dict["W1_states"][self.layer_idx],
                "W2_states": cache_params.params_dict["W2_states"][self.layer_idx],
                "W1_grad": cache_params.params_dict["W1_grad"][self.layer_idx],
                "W2_grad": cache_params.params_dict["W2_grad"][self.layer_idx],
            }
            inputs = tree_map(lambda x: einops.rearrange(x, 'b nh nc cs f -> nc (b nh) cs f')[0], inputs)
            batch_params_dict, XCW_batch = decode_one_token(
                states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                inputs,  # [B*nh,f]
            )
            XCW_batch = XCW_batch.unsqueeze(0)  # [NC=1,B*nh,CS=1,f]

        if cache_params is not None:
            # @xinhao: can skip this for model() forward test by setting cache_params=None
            # As for prefill in .generate(), will not skip the below
            if is_last_in_chunk:
                cache_params.update_last_in_chunk(batch_params_dict, self.layer_idx)
            else:
                cache_params.update_non_last_in_chunk(batch_params_dict, self.layer_idx)

        XCW_batch = einops.rearrange(XCW_batch, "nc (b nh) cs f -> b (nc cs) (nh f)", b=B, nh=NH)  # [B,L,f]

        return XCW_batch, batch_params_dict


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=7, num_warps=8),
        triton.Config({}, num_stages=6, num_warps=8),
        triton.Config({}, num_stages=5, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=6, num_warps=4),
    ],
    key=['N_CHUNK'],
)
@triton.jit
def _m1_kernel(W1, XA, XB, XC, coeff_last, coeff, Out,
               stride_ab, stride_ah, stride_an, stride_ac, stride_af,
               stride_eb, stride_eh, stride_en, stride_ec,
               stride_pb, stride_ph, stride_pn,
               stride_wb, stride_wh, stride_wf, stride_wd,
               CS: tl.constexpr, HF: tl.constexpr,
               N_CHUNK: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    abco_offset = batch * stride_ab + head * stride_ah
    w_offset = batch * stride_wb + head * stride_wh
    coeff_offset = batch * stride_eb + head * stride_eh
    coeff_last_offset = batch * stride_pb + head * stride_ph

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)
    XA = XA + abco_offset
    XB = XB + abco_offset
    XC = XC + abco_offset
    Out = Out + abco_offset
    W1_data = tl.load(W1 + w_offset + rf[:, None] * stride_wf + rf[None, :] * stride_wd)
    coeff = coeff + coeff_offset
    coeff_last = coeff_last + coeff_last_offset
    for i in range(N_CHUNK):
        local_abco_offset = i * stride_an
        local_coeff_offset = i * stride_en
        local_coeff_last_offset = i * stride_pn
        XA_chunk = tl.load(XA + local_abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af))
        XB_chunk = tl.load(XB + local_abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af))
        XC_chunk = tl.load(XC + local_abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af))
        coeff_chunk = tl.load(coeff + local_coeff_offset + rc * stride_ec)
        coeff_chunk_last = tl.load(coeff_last + local_coeff_last_offset)

        Z1 = tl.dot(XB_chunk, W1_data) - XA_chunk
        mask = rc[:, None] >= rc[None, :]
        Attn1_full = tl.dot(XC_chunk, tl.trans(XB_chunk))
        Attn1 = tl.where(mask, Attn1_full, 0)
        Z1_bar = tl.dot(XC_chunk, W1_data) - tl.dot((coeff_chunk[:, None] * Attn1), Z1)
        W1_data -= tl.dot(tl.trans(coeff_chunk_last * XB_chunk).to(Z1.dtype), Z1).to(W1.type.element_ty)
        Out_chunk = Out + local_abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af)
        tl.store(Out_chunk, Z1_bar.to(Out.type.element_ty))


class TttM1BMMTritonModule(TttBaseModule):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))

        if self.config.use_compile:
            self.prefill_chunk = torch.compile(m1_prefill_chunk)
            self.decode_chunk = torch.compile(m1_decode_one_token)
        else:
            self.prefill_chunk = m1_prefill_chunk
            self.decode_chunk = m1_decode_one_token

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        B, NH, NC, CS, HF = inputs['XA'].shape
        input_dtype = inputs['XA'].dtype
        L = NC * CS
        inputs.update(coeff_chunk_last=inputs['coeff'][...,-1:])  # [B,nh,NC,CS,1] -> [B,nh,NC,1,1]

        if is_prefill:

            grid = (B, NH, 1)
            output = torch.empty(size=(B, NH, NC, CS, HF), device=self.W1.device, dtype=input_dtype)
            W1_expand = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))  # [B,nh,f,f]
            XA, XB, XC, coeff, coeff_last = inputs['XA'], inputs['XB'], inputs['XC'], \
                                            inputs['coeff'], inputs['coeff_chunk_last']  # [B,nh,NC,CS,f]
            _m1_kernel[grid](W1_expand,  # [B,nh,f,f], cloned from W1, safe for in-place op
                             XA, XB, XC, coeff_last, coeff, output,
                             NH * NC * CS * HF, NC * CS * HF, CS * HF, HF, 1,  # strides for A,B,C,O
                             NH * NC * CS, NC * CS, CS, 1,  # strides for E
                             NH * NC, NC, 1,  # strides for last coeff
                             NH * HF * HF, HF * HF, HF, 1,  # strides for W1
                             CS, HF,
                             NC
                             )
            XCW_batch = einops.rearrange(output, "b nh nc cs f -> nc (b nh) cs f")
            batch_params_dict = einops.rearrange(W1_expand, "b nh f d -> (b nh) f d")

        else:

            def decode_one_token(states, inputs):
                XA_chunk = inputs["XA"]        # [B*nh,K,f]
                XB_chunk = inputs["XB"]
                XC_chunk = inputs["XC"]
                coeff_chunk = inputs["coeff"]  # [B*nh,K,1]

                states, Z1 = self.decode_chunk(states, XA_chunk, XB_chunk, XC_chunk, coeff_chunk)
                return states, Z1  # [B*nh, f]

            states = {
                "W1_states": cache_params.params_dict["W1_states"][self.layer_idx],
                "W1_grad": cache_params.params_dict["W1_grad"][self.layer_idx],
            }
            inputs = tree_map(lambda x: einops.rearrange(x, 'b nh nc cs f -> nc (b nh) cs f')[0], inputs)
            batch_params_dict, XCW_batch = decode_one_token(
                states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                inputs,  # [B*nh,f]
            )
            XCW_batch = XCW_batch.unsqueeze(0)  # [NC=1,B*nh,CS=1,f]

        if cache_params is not None:
            # @xinhao: can skip this for model() forward test by setting cache_params=None
            # As for prefill in .generate(), will not skip the below
            if is_last_in_chunk:
                cache_params.update_last_in_chunk(batch_params_dict, self.layer_idx)
            else:
                cache_params.update_non_last_in_chunk(batch_params_dict, self.layer_idx)

        XCW_batch = einops.rearrange(XCW_batch, "nc (b nh) cs f -> b (nc cs) (nh f)", b=B, nh=NH)  # [B,L,f]

        return XCW_batch, batch_params_dict


class TttDecoderLayer(nn.Module):
    def __init__(self, config: TttConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # self.self_attn = TttM1Module(config=config, layer_idx=layer_idx)  # @xinhao: M1 vmap module
        if config.inner_net == 'mlp_1_dual':
            self.self_attn = TttM1BMMModule(config=config, layer_idx=layer_idx)
        elif config.inner_net == 'mlp_1_dual_triton':
            self.self_attn = TttM1BMMTritonModule(config=config, layer_idx=layer_idx)
        elif config.inner_net == 'mlp_2_dual':
            self.self_attn = TttM2BMMModule(config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError(f"Inner {config.inner_net} Not Implemented!")

        self.mlp = TttMLP(config)
        self.input_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

        if config.use_compile:
            self.mlp_forward = torch.compile(self._mlp_forward)
        else:
            self.mlp_forward = self._mlp_forward

    def _mlp_forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
        is_prefill: Optional[bool] = None,
        is_last_in_chunk: Optional[bool] = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # TTT
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
            is_prefill=is_prefill,
            is_last_in_chunk=is_last_in_chunk,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        hidden_states = self.mlp_forward(hidden_states)

        return hidden_states


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


class TttPreTrainedModel(nn.Module, GenerationMixin):
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


class TttModel(TttPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TttDecoderLayer`]

    Args:
        config: TttConfig
    """

    def __init__(self, config: TttConfig):
        # super().__init__(config)
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TttDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.num_hidden_layers = config.num_hidden_layers
        self.inner_net_chunk_size = config.inner_net_chunk_size
        self.config = config

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def create_cache(self, batch_size, device, dtype) -> TttCache:
        logger.info(f"Creating cache of size: {batch_size}")
        print("create_cache")
        cache = TttCache(self.config, batch_size, dtype=dtype, device=device)
        for layer_idx in range(self.config.num_hidden_layers):
            for name in cache.param_names:
                weight = getattr(self.layers[layer_idx].self_attn, name)
                # tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim())
                tiled_weight = torch.tile(weight, (batch_size,) + (1,) * (weight.dim() - 1))  # [B*nh,f,f]
                cache.params_dict[f"{name}_states"][layer_idx] = tiled_weight
                cache.params_dict[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)
        return cache

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TttCache] = None,  # @xinhao: must pass in non-none cache_params from generation.py
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        is_prefill: Optional[bool] = None,
        is_last_in_chunk: Optional[bool] = None,
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

        # if cache_params is None and use_cache:
        #     cache_params = self.create_cache(inputs_embeds.size(0), inputs_embeds.device, inputs_embeds.dtype)

        seqlen_offset = 0
        if cache_params is not None:
            seqlen_offset = cache_params.seqlen_offset
        position_ids = torch.arange(
            seqlen_offset, seqlen_offset+ inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
        ).unsqueeze(0)

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
                    is_prefill=is_prefill,
                    is_last_in_chunk=is_last_in_chunk,
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
        # super().__init__(config)
        super().__init__()
        self.model = TttModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.use_compile:
            self.get_output_logits = torch.compile(self._get_output_logits)
        else:
            self.get_output_logits = self._get_output_logits

            # Initialize weights and apply final processing
        # self.post_init()
        self.config = config

    def _get_output_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        return logits

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
        is_prefill: Optional[bool] = None,
        is_last_in_chunk: Optional[bool] = None,
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
            is_prefill=is_prefill,
            is_last_in_chunk=is_last_in_chunk,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.get_output_logits(hidden_states)
            # logits = self.lm_head(hidden_states)

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