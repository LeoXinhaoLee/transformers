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

from transformers.models.ttt_full_decode_optimize.generation import GenerationMixin, TttCache
from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TttConfig"

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


def diff_gelu(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


def ln_fwd(x, gamma, beta, eps=1e-6):
    """
    Args:
        x: [B*nh,N,f]
        gamma: [1,nh,1,f]
        beta: [1,nh,1,f]
        eps:

    Returns:
        z: [B*nh,N,f]

    """
    B_nh, N, HF = x.shape
    nh = gamma.shape[1]
    x = x.reshape(-1, nh, N ,HF)

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta
    y = y.reshape(-1, N, HF)
    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    """
    Args:
        x: [B*nh,N=1,f]
        l2_target: [B*nh,N=1,f]
        gamma: [1,nh,1,f]
        beta: [1,nh,1,f]
        eps:

    Returns:
        grad_l_x: [B*nh,N=1,f]
    """
    B_nh, N, HF = x.shape
    nh = gamma.shape[1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)  # [B*nh,N=1,1]
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std  # [B*nh,N=1,f]

    # Scale and shift
    y = gamma * x_hat.reshape(-1, nh, N, HF) + beta  #[1,nh,N=1,f] * [B,nh,N=1,f] + [1,nh,N=1,f]

    grad_output = y - l2_target.reshape(-1, nh, N, HF)  # [B,nh,N=1,f]
    grad_x_hat = (grad_output * gamma).reshape(B_nh, N, HF)  # [B,nh,N=1,f]
    z = (
        (1.0 / HF)
        * (
            HF * grad_x_hat  # [B*nh,1,f]
            - grad_x_hat.sum(dim=-1, keepdim=True)  # [B*nh,1,1]
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)  # [B*nh,N=1,f] * [B*nh,N=1,1]
        )
        / std
    )
    return z


class TttMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_gate_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate, up = torch.split(self.up_gate_proj(x), split_size_or_sections=self.intermediate_size, dim=-1)
        down_proj = self.down_proj(self.act_fn(gate) * up)
        return down_proj


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

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.inner_chunk_size = config.inner_net_chunk_size
        self.conv_kernel = config.conv_kernel
        
        token_idx = (self.config.inner_net_lr / self.head_dim) / torch.arange(1, self.inner_chunk_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size + self.num_heads, bias=False)  # share QK so can add Gate
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

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

        self.decoder_ln_fn = partial(F.layer_norm, normalized_shape=[self.head_dim], eps=1e-6)
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ln_weight = nn.Parameter(torch.tile(ln_weight_data.reshape(1, 1, 1, -1), (1, self.num_heads, 1, 1)))  # [1,h,1,f]
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ln_bias = nn.Parameter(torch.tile(ln_bias_data.reshape(1, 1, 1, -1), (1, self.num_heads, 1, 1)))  # [1,h,1,f]
        
        if config.use_compile:
            self.get_inner_loop_inputs = torch.compile(self._get_inner_loop_inputs)
        else:
            self.get_inner_loop_inputs = self._get_inner_loop_inputs

    def conv_qk(
        self,
        XCB,
        cache_params: Optional[TttCache] = None,
        is_prefill = False,
    ):
        '''
        Args:
            XCB: (1) prefill: [B,N,F]; (2) decode: [B,1,F];
            cache_params: [B,3,F]

        Returns:
            XC: [B,N,F]
            XB: [B,N,F]
            in-place update cache_params
        '''
        B, N, D = XCB.shape
        XCB = XCB.transpose(-1, -2)  # [B,F,N]
        if is_prefill:
            XC = self.conv_q(XCB)[..., :N].transpose(-1, -2)
            XB = self.conv_k(XCB)[..., :N].transpose(-1, -2)
            if cache_params is not None:
                conv_states = F.pad(XCB, (self.conv_kernel - N, 0))
                cache_params.params_dict["conv_states"][self.layer_idx].copy_(conv_states)  # [B,F,KS]
        else:
            if cache_params is not None:
                conv_states = cache_params.params_dict["conv_states"][self.layer_idx]
                conv_states = torch.roll(conv_states, shifts=-1, dims=-1)
                conv_states[:, :, -1] = XCB[:, :, 0]  # [B,F,KS]
                cache_params.params_dict["conv_states"][self.layer_idx].copy_(conv_states)
            else:
                conv_states = nn.functional.pad(XCB, (self.conv_kernel - N, 0))  # [B,F,KS]

            XC = torch.sum(conv_states * self.conv_q.weight[:, 0, :], dim=-1) + self.conv_q.bias  # ([B,F,KS] * [F,1,KS][:,0,:]).sum(-1) -> [B,F] + [F,]
            XB = torch.sum(conv_states * self.conv_k.weight[:, 0, :], dim=-1) + self.conv_k.bias
            XC = XC.unsqueeze(1)  # [B,N=1,F]
            XB = XB.unsqueeze(1)

        return XC, XB

    def conv_qk_fused(
        self,
        XCB,
        cache_params: Optional[TttCache] = None,
        is_prefill: bool = False,
    ):
        '''
        Args:
            XCB: (1) prefill: [B,N,F]; (2) decode: [B,1,F];
            cache_params: [B,3,F]

        Returns:
            XC: [B,N,F]
            XB: [B,N,F]
            in-place update cache_params
        '''
        B, N, D = XCB.shape
        conv_q_weights = self.conv_q.weight.view(self.conv_q.weight.size(0), self.conv_q.weight.size(2))
        conv_k_weights = self.conv_k.weight.view(self.conv_k.weight.size(0), self.conv_k.weight.size(2))
        if is_prefill:
            XCB = XCB.transpose(-1, -2)  # [B,F,N]
            XC = causal_conv1d_fn(
                XCB, conv_q_weights, self.conv_q.bias, activation=None
            )
            XB = causal_conv1d_fn(
                XCB, conv_k_weights, self.conv_k.bias, activation=None
            )
            if cache_params is not None:
                conv_states = F.pad(XCB, (self.conv_kernel - N, 0))
                cache_params.params_dict["conv_states"][self.layer_idx].copy_(conv_states)  # [B,F,KS]

        else:
            assert cache_params is not None
            XCB = XCB[:,0,:]  # [B,F]
            XC = causal_conv1d_update(
                XCB,
                cache_params.params_dict['conv_states'][self.layer_idx].clone(),  # avoid being updated twice
                conv_q_weights,
                self.conv_q.bias,
                None,
            )
            XB = causal_conv1d_update(
                XCB, # [B,F]
                cache_params.params_dict['conv_states'][self.layer_idx],  # [B,F,KS]
                conv_q_weights,
                self.conv_q.bias,
                None,
            )
            XC = XC.unsqueeze(1)  # [B,N=1,F]
            XB = XB.unsqueeze(1)

        return XC, XB


    def _get_inner_loop_inputs(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        cache_params: Optional[TttCache] = None,
        inner_chunk_size: Optional[int] = None,
        is_prefill: bool = False,
    ):
        B, L, D = hidden_states.shape
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        if cache_params is not None:
            # TODO: keeps recompiling when decoding, so cannot torch.compile this function when decode
            inner_chunk_step_offset = cache_params.seqlen_offset % self.inner_chunk_size
        else:
            inner_chunk_step_offset = 0

        XCBA_gilr = self.qkv_proj(hidden_states)  # [B,N,3*F + nh]

        XCBA, ilr_gated = torch.split(XCBA_gilr, split_size_or_sections=[3*D, self.num_heads], dim=-1)

        XCB, XGate_XA = torch.split(XCBA, split_size_or_sections=[self.hidden_size, 2 * self.hidden_size], dim=-1)
        XGate, XA = torch.split(
            XGate_XA.reshape(B, L, self.num_heads, 2 * self.head_dim).permute(0,2,1,3).reshape(-1, L, 2 * self.head_dim),
            split_size_or_sections=self.head_dim, dim=-1
        )  # [B*nh,N=1,f] x2

        # XC, XB = self.conv_qk(XCB, cache_params, is_prefill)  # [B,N,F] -> conv1: [B,N,F], conv2: [B,N,F]
        XC, XB = self.conv_qk_fused(XCB, cache_params, is_prefill)  # [B,N,F] -> conv1: [B,N,F], conv2: [B,N,F]

        # @xinhao: test no inner-loop
        # XC = XB = XCB

        XC = XC.reshape(B, L, self.num_heads, self.head_dim).permute(0,2,1,3).reshape(-1, L, self.head_dim)  # [B*nh,N,f]
        XB = XB.reshape(B, L, self.num_heads, self.head_dim).permute(0,2,1,3).reshape(-1, L, self.head_dim)

        ilr_gated = F.sigmoid(ilr_gated.permute(0,2,1).reshape(-1,1,1))  # [B,N=1,nh] -> [B,nh,N=1] -> [B*nh,N=1,1]
        coeff = self.token_idx[inner_chunk_step_offset] * ilr_gated   # [B*nh,N=1,1]
        return XC, XB, XA, coeff, XGate

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
            XC, XB, XA, coeff, XGate = self.get_inner_loop_inputs(
                hidden_states, position_ids=position_ids,
                cache_params=cache_params, inner_chunk_size=inner_chunk_size, is_prefill=is_prefill
            )
        else:
            # @xinhao: decoding time should not compile `get_inner_loop_inputs`. Otherwise will recompile every step.
            XC, XB, XA, coeff, XGate = self._get_inner_loop_inputs(
                hidden_states, position_ids=position_ids,
                cache_params=cache_params, inner_chunk_size=inner_chunk_size, is_prefill=is_prefill
            )
        B_mul_NH, N, HF = XA.shape
        B = B_mul_NH // self.num_heads
        inputs = {'XC': XC, 'XB': XB, 'XA': XA, 'coeff': coeff}
        XCW_batch, batch_params_dict = self.process_inner_loop(
            inputs,
            inner_chunk_size=inner_chunk_size,
            last_chunk_params_dic=last_chunk_params_dic,
            cache_params=cache_params,
            is_prefill=is_prefill, is_last_in_chunk=is_last_in_chunk,
        )

        # @xinhao: for QKVO-MLP Only
        # B, N = hidden_states.shape[:2]
        # XC = XB = XA = XGate = hidden_states.reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3).reshape(-1, N, self.head_dim)
        # XCW_batch = XA + XB + XC; batch_params_dict = None

        XCW_batch = F.gelu(XGate, approximate='tanh') * XCW_batch  # [B*nh,N,f]
        XCW_batch = XCW_batch.reshape(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N,- 1)
        z_batch = self.project_inner_loop_outputs(XCW_batch)  # [B,N,F]
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


####### M1 Decode Module #######
def m1_decode_one_token_last_in_chunk(states, inputs, ln_weight, ln_bias):
    W1_init = states['W1_states']
    b1_init = states['b1_states']
    W1_grad = states['W1_grad']
    b1_grad = states['b1_grad']

    XA_chunk, XB_chunk, \
    XC_chunk, coeff_chunk = inputs['XA'], inputs['XB'], inputs['XC'], inputs['coeff']  # [B*nh,f,f], [B*nh,N=1,f/1]

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ [B*nh,f,f] + [B*nh,1,f] -> [B*nh,K=1,f]
    reconstruction_target = XA_chunk - XB_chunk
    grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B*nh,K=1,f]

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B*nh,1,f].t @ [B*nh,1,f] + [B*nh,f,f]
    b1_grad.add_(grad_l_wrt_Z1)

    W1_init.sub_(coeff_chunk * W1_grad)  # [B*nh,f,f] - [B*nh,N=1,1] * [B*nh,f,f]
    b1_init.sub_(coeff_chunk * b1_grad)  # [B*nh,1,f] - [B*nh,N=1,1] * [B*nh,1,f]
    Z1_bar = XC_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ ([B*nh,f,f] - [B*nh,1,1] * [B*nh,f,f])
    Z1_bar = XC_chunk + ln_fwd(Z1_bar, ln_weight, ln_bias)
    W1_grad._zero()
    b1_grad._zero()
    return Z1_bar

def m1_decode_one_token_non_last_in_chunk(states, inputs, ln_weight, ln_bias):
    W1_init = states['W1_states']
    b1_init = states['b1_states']
    W1_grad = states['W1_grad']
    b1_grad = states['b1_grad']

    XA_chunk, XB_chunk, \
    XC_chunk, coeff_chunk = inputs['XA'], inputs['XB'], inputs['XC'], inputs['coeff']  # [B*nh,f,f], [B*nh,N=1,f/1]

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ [B*nh,f,f] + [B*nh,1,f] -> [B*nh,K=1,f]
    reconstruction_target = XA_chunk - XB_chunk
    grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B*nh,K=1,f]

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B*nh,1,f].t @ [B*nh,1,f] + [B*nh,f,f]
    b1_grad.add_(grad_l_wrt_Z1)

    W1_last = W1_init - (coeff_chunk * W1_grad)  # [B*nh,f,f] - [B*nh,N=1,1] * [B*nh,f,f]
    b1_last = b1_init - (coeff_chunk * b1_grad)  # [B*nh,1,f] - [B*nh,N=1,1] * [B*nh,1,f]
    Z1_bar = XC_chunk @ W1_last + b1_last  # [B*nh,K=1,f] @ ([B*nh,f,f] - [B*nh,1,1] * [B*nh,f,f])
    Z1_bar = XC_chunk + ln_fwd(Z1_bar, ln_weight, ln_bias)
    return Z1_bar

class TttM1BMMModule(TttBaseModule):

    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.ones(size=(self.num_heads, 1, self.head_dim)))

        if self.config.use_compile:
            self.decode_token_last_in_chunk = torch.compile(m1_decode_one_token_last_in_chunk)
            self.decode_token_non_last_in_chunk = torch.compile(m1_decode_one_token_non_last_in_chunk)
        else:
            self.decode_token_last_in_chunk = m1_decode_one_token_last_in_chunk
            self.decode_token_non_last_in_chunk = m1_decode_one_token_non_last_in_chunk

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        B_mul_NH, N, HF = inputs['XA'].shape  # [B*nh,N=1,f]
        B = B_mul_NH // self.num_heads

        states = {
            "W1_states": cache_params.params_dict["W1_states"][self.layer_idx],
            "b1_states": cache_params.params_dict["b1_states"][self.layer_idx],
            "W1_grad": cache_params.params_dict["W1_grad"][self.layer_idx],
            "b1_grad": cache_params.params_dict["b1_grad"][self.layer_idx],
        }

        if is_last_in_chunk:
            XCW_batch = self.decode_token_last_in_chunk(
                states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                inputs,  # [B*nh,f]
                self.ln_weight,  # [nh,1,f]
                self.ln_bias, # [nh,1,f]
            )  # ret: [B*nh,N=1,f]
        else:
            XCW_batch = self.decode_token_non_last_in_chunk(
                states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                inputs,  # [B*nh,f]
                self.ln_weight,  # [nh,1,f]
                self.ln_bias,  # [nh,1,f]
            )  # ret: [B*nh,N=1,f]

        return XCW_batch, None

##########################################


####### M2 Decode Module #######

def m2_decode_one_token_last_in_chunk(states, inputs, ln_weight, ln_bias):
    W1_init = states['W1_states']
    W1_grad = states['W1_grad']
    b1_init = states['b1_states']
    b1_grad = states['b1_grad']
    W2_init = states['W2_states']
    W2_grad = states['W2_grad']
    b2_init = states['b2_states']
    b2_grad = states['b2_grad']

    XA_chunk, XB_chunk, \
    XC_chunk, coeff_chunk = inputs['XA'], inputs['XB'], inputs['XC'], inputs['coeff']

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ [B*nh,f,f] -> [B*nh,K=1,f]
    X2 = F.gelu(Z1, approximate='tanh')
    Z2 = X2 @ W2_init + b2_init

    reconstruction_target = XA_chunk - XB_chunk
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)  # [B*nh,K=1,f]
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-1, -2) * diff_gelu(Z1)

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B*nh,1,f].t @ [B*nh,1,f] + [B*nh,f,f]
    b1_grad.add_(grad_l_wrt_Z1)
    W1_init.sub_(coeff_chunk * W1_grad)
    b1_init.sub_(coeff_chunk * b1_grad)
    Z1_bar = XC_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ ([B*nh,f,f] - [B*nh,1,1] * [B*nh,f,f])
    X2_bar = F.gelu(Z1_bar, approximate='tanh')

    W2_grad.add_(X2.transpose(-1, -2) @ grad_l_wrt_Z2)  # [B*nh,f,f]
    b2_grad.add_(grad_l_wrt_Z2)
    W2_init.sub_(coeff_chunk * W2_grad)
    b2_init.sub_(coeff_chunk * b2_grad)
    Z2_bar = X2_bar @ W2_init + b2_init  # [B*nh,K=1,f]
    Z2_bar = XC_chunk + ln_fwd(Z2_bar, ln_weight, ln_bias)

    W1_grad._zero()
    b1_grad._zero()
    W2_grad._zero()
    b2_grad._zero()

    return Z2_bar

def m2_decode_one_token_non_last_in_chunk(states, inputs, ln_weight, ln_bias):
    W1_init = states['W1_states']
    W1_grad = states['W1_grad']
    b1_init = states['b1_states']
    b1_grad = states['b1_grad']
    W2_init = states['W2_states']
    W2_grad = states['W2_grad']
    b2_init = states['b2_states']
    b2_grad = states['b2_grad']

    XA_chunk, XB_chunk, \
    XC_chunk, coeff_chunk = inputs['XA'], inputs['XB'], inputs['XC'], inputs['coeff']

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ [B*nh,f,f] + [B*nh,1,f] -> [B*nh,K=1,f]
    X2 = F.gelu(Z1, approximate='tanh')
    Z2 = X2 @ W2_init + b2_init

    reconstruction_target = XA_chunk - XB_chunk
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)  # [B*nh,K=1,f]
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-1, -2) * diff_gelu(Z1)

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B*nh,1,f].t @ [B*nh,1,f] + [B*nh,f,f]
    b1_grad.add_(grad_l_wrt_Z1)
    W1_last = W1_init - (coeff_chunk * W1_grad)
    b1_last = b1_init - (coeff_chunk * b1_grad)
    Z1_bar = XC_chunk @ W1_last + b1_last  # [B*nh,K=1,f] @ ([B*nh,f,f] - [B*nh,1,1] * [B*nh,f,f])
    X2_bar = F.gelu(Z1_bar, approximate='tanh')

    W2_grad.add_(X2.transpose(-1, -2) @ grad_l_wrt_Z2)  # [B*nh,f,f]
    b2_grad.add_(grad_l_wrt_Z2)
    W2_last = W2_init - (coeff_chunk * W2_grad)
    b2_last = b2_init - (coeff_chunk * b2_grad)
    Z2_bar = X2_bar @ W2_last + b2_last  # [B*nh,K=1,f]
    Z2_bar = XC_chunk + ln_fwd(Z2_bar, ln_weight, ln_bias)

    return Z2_bar


class TttM2BMMModule(TttBaseModule):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 1, 4 * self.head_dim)))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 1, self.head_dim)))

        if self.config.use_compile:
            self.decode_token_last_in_chunk = torch.compile(m2_decode_one_token_last_in_chunk)
            self.decode_token_non_last_in_chunk = torch.compile(m2_decode_one_token_non_last_in_chunk)
        else:
            self.decode_token_last_in_chunk = m2_decode_one_token_last_in_chunk
            self.decode_token_non_last_in_chunk = m2_decode_one_token_non_last_in_chunk

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):
        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        states = {
            "W1_states": cache_params.params_dict["W1_states"][self.layer_idx],
            "W1_grad": cache_params.params_dict["W1_grad"][self.layer_idx],
            "b1_states": cache_params.params_dict["b1_states"][self.layer_idx],
            "b1_grad": cache_params.params_dict["b1_grad"][self.layer_idx],
            "W2_states": cache_params.params_dict["W2_states"][self.layer_idx],
            "W2_grad": cache_params.params_dict["W2_grad"][self.layer_idx],
            "b2_states": cache_params.params_dict["b2_states"][self.layer_idx],
            "b2_grad": cache_params.params_dict["b2_grad"][self.layer_idx],
        }

        if is_last_in_chunk:
            XCW_batch = self.decode_token_last_in_chunk(
                states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                inputs,  # [B*nh,f]
                self.ln_weight,  # [nh,1,f]
                self.ln_bias,  # [nh,1,f]
            )  # ret: [B*nh,N=1,f]
        else:
            XCW_batch = self.decode_token_non_last_in_chunk(
                states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                inputs,  # [B*nh,f]
                self.ln_weight,  # [nh,1,f]
                self.ln_bias,  # [nh,1,f]
            )  # ret: [B*nh,N=1,f]

        return XCW_batch, None

##########################################


####### M1 Triton Decode Module #######
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
    key=['HF'],
    restore_value=['W1_init', 'W1_grad']
)
@triton.jit
def _m1_decode_kernel(W1_init, W1_grad, XA, XB, XC, coeff, Out,
                      stride_wb, stride_wh, stride_wf, stride_wd,
                      stride_ab, stride_ah, stride_ac, stride_af,
                      stride_cb, stride_ch, stride_cn, stride_cc,
                      CS: tl.constexpr, HF: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)

    W_dtype = W1_init.type.element_ty
    O_dtype = Out.type.element_ty

    abco_offset = batch * stride_ab + head * stride_ah
    w_offset = batch * stride_wb + head * stride_wh
    coeff_offset = batch * stride_cb + head * stride_ch

    XA = XA + abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af)
    XB = XB + abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af)
    XC = XC + abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af)
    Out = Out + abco_offset
    W1_init = W1_init + w_offset + (rf[:, None] * stride_wf + rf[None, :] * stride_wd)
    W1_grad = W1_grad + w_offset + (rf[:, None] * stride_wf + rf[None, :] * stride_wd)
    coeff = coeff + coeff_offset + rc * stride_cc
    Out_chunk = Out + (rf * stride_af)

    XA_chunk = tl.load(XA)
    XB_chunk = tl.load(XB)
    XC_chunk = tl.load(XC)
    coeff_chunk = tl.load(coeff)
    W1_init_data = tl.load(W1_init)
    W1_grad_data = tl.load(W1_grad)

    Z1 = tl.sum(tl.trans(XB_chunk) * W1_init_data, 0) - XA_chunk
    # W1_grad_data += tl.sum(XB_chunk * Z1, 0)  # @xinhao: bug, but doesn't affect speed too much
    W1_grad_data += tl.trans(XB_chunk) * Z1
    W1_init_data -= coeff_chunk * W1_grad_data
    Z1_bar = tl.sum(tl.trans(XC_chunk) * W1_init_data, 0)

    tl.store(Out_chunk, Z1_bar.to(Out.type.element_ty))
    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))

class TttM1BMMTritonModule(TttBaseModule):

    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):

        # XA = torch.randn(BS, NH, N, HF, device='cuda', dtype=input_dtype)  # reference shape

        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        B_mul_NH, CS, HF = inputs['XA'].shape  # [B*nh,N=1,f]
        B = B_mul_NH // self.num_heads
        NH = self.num_heads
        input_dtype = inputs['XA'].dtype
        input_device = inputs['XA'].device

        inputs = tree_map(lambda x: x.reshape(B, NH, CS, -1), inputs) # [B*nh,N=1,f], [B*nh,N=1,1] -> [BS,nh,N=1,f/1]
        XA, XB, XC, coeff = inputs['XA'], inputs['XB'], inputs['XC'], inputs['coeff']

        W1_init = cache_params.params_dict["W1_states"][self.layer_idx].reshape(B, NH, HF, HF)
        W1_grad = cache_params.params_dict["W1_grad"][self.layer_idx].reshape(B, NH, HF, HF)

        output = torch.empty(size=(B, NH, CS, HF), device=input_device, dtype=torch.float16)  # TODO FIX DTYPE
        grid = (B, NH, 1)
        _m1_decode_kernel[grid](W1_init, W1_grad, XA, XB, XC, coeff, output,

                                NH * HF * HF,
                                HF * HF,
                                HF,
                                1,  # strides for W

                                NH * CS * HF,
                                CS * HF,
                                HF,
                                1,  # strides for ABCO, output

                                NH * CS,
                                CS,
                                1,
                                1,  # strides for coeff

                                CS, HF)  # TODO: differentiate last_in_chunk and non_last_in_chunk

        output = output.permute(0, 2, 1, 3).reshape(B, CS, -1)
        return output, None

##########################################


####### M2 Triton Decode Module #######

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
    key=['HF', 'HF_prime'],
    restore_value=['W1_init', 'W1_grad', 'W2_init', 'W2_grad']
)
@triton.jit
def _m2_decode_kernel(W1_init, W1_grad, W2_init, W2_grad,
                      XA, XB, XC, coeff,
                      Out,
                      stride_w1b, stride_w1h, stride_w1f, stride_w1d,
                      stride_w2b, stride_w2h, stride_w2f, stride_w2d,
                      stride_ab, stride_ah, stride_ac, stride_af,
                      stride_cb, stride_ch, stride_cn, stride_cc,
                      CS: tl.constexpr, HF: tl.constexpr, HF_prime: tl.constexpr):

    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)
    rf_prime = tl.arange(0, HF_prime)

    W_dtype = W1_init.type.element_ty
    O_dtype = Out.type.element_ty

    w1_offset = batch * stride_w1b + head * stride_w1h
    w2_offset = batch * stride_w2b + head * stride_w2h

    abco_offset = batch * stride_ab + head * stride_ah
    coeff_offset = batch * stride_cb + head * stride_ch

    W1_init = W1_init + w1_offset + (rf[:, None] * stride_w1f + rf_prime[None, :] * stride_w1d)  # [HF, HF_prime]
    W1_grad = W1_grad + w1_offset + (rf[:, None] * stride_w1f + rf_prime[None, :] * stride_w1d)

    W2_init = W2_init + w2_offset + (rf_prime[:, None] * stride_w2f + rf[None, :] * stride_w2d)  # [HF_prime, HF]
    W2_grad = W2_grad + w2_offset + (rf_prime[:, None] * stride_w2f + rf[None, :] * stride_w2d)

    XA = XA + abco_offset + rf[None, :] * stride_af  # [1,HF]
    XB = XB + abco_offset + rf[None, :] * stride_af
    XC = XC + abco_offset + rf[None, :] * stride_af
    coeff = coeff + coeff_offset  # [1,1]
    Out_chunk = Out + abco_offset + rf * stride_af  # [1,HF]

    XA_chunk = tl.load(XA)
    XB_chunk = tl.load(XB)
    XC_chunk = tl.load(XC)
    coeff_chunk = tl.load(coeff)
    W1_init_data = tl.load(W1_init)
    W1_grad_data = tl.load(W1_grad)
    W2_init_data = tl.load(W2_init)
    W2_grad_data = tl.load(W2_grad)

    Z1 = tl.sum(tl.trans(XB_chunk) * W1_init_data, 0)[None,:]  # [1,HF_prime]
    Z2 = tl.sum(tl.trans(Z1) * W2_init_data, 0)[None,:]  # [1,HF]

    grad_l_wrt_Z2 = Z2 - XA_chunk  # [1,HF]
    grad_l_wrt_Z1 = tl.sum(grad_l_wrt_Z2 * W2_init_data, 1)[None,:]  # [1,HF] * [HF_p, HF] -> [HF_p,] -> [1,HF_p]

    W1_grad_data += tl.trans(XB_chunk) * grad_l_wrt_Z1  # [1,HF].t * [1,HF_p] -> [HF,HF_p]
    W1_init_data -= coeff_chunk * W1_grad_data
    Z1_bar = tl.sum(tl.trans(XC_chunk) * W1_init_data, 0)[None,:]

    W2_grad_data += tl.trans(Z1) * grad_l_wrt_Z2  # [1,HF_p].t * [1,HF] -> [HF_p,HF]
    W2_init_data -= coeff_chunk * W2_grad_data
    Z2_bar = tl.sum(tl.trans(Z1_bar) * W2_init_data, 0)

    tl.store(Out_chunk, Z2_bar.to(Out.type.element_ty))
    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))
    tl.store(W2_init, W2_init_data.to(W_dtype))
    tl.store(W2_grad, W2_grad_data.to(W_dtype))


class TttM2BMMTritonModule(TttBaseModule):

    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):

        # XA = torch.randn(BS, NH, N, HF, device='cuda', dtype=input_dtype)  # reference shape

        # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
        B_mul_NH, CS, HF = inputs['XA'].shape  # [B*nh,N=1,f]
        HF_prime = self.W1.shape[-1]
        B = B_mul_NH // self.num_heads
        NH = self.num_heads
        input_dtype = inputs['XA'].dtype
        input_device = inputs['XA'].device

        inputs = tree_map(lambda x: x.reshape(B, NH, CS, -1), inputs) # [B*nh,N=1,f], [B*nh,N=1,1] -> [BS,nh,N=1,f/1]
        XA, XB, XC, coeff = inputs['XA'], inputs['XB'], inputs['XC'], inputs['coeff']

        W1_init = cache_params.params_dict["W1_states"][self.layer_idx].reshape(B, NH, HF, 4 * HF)
        W1_grad = cache_params.params_dict["W1_grad"][self.layer_idx].reshape(B, NH, HF, 4 * HF)
        W2_init = cache_params.params_dict["W2_states"][self.layer_idx].reshape(B, NH, 4 * HF, HF)
        W2_grad = cache_params.params_dict["W2_grad"][self.layer_idx].reshape(B, NH, 4 * HF, HF)

        output = torch.empty(size=(B, NH, CS, HF), device=input_device, dtype=torch.float16)  # TODO FIX DTYPE
        grid = (B, NH, 1)
        _m2_decode_kernel[grid](W1_init, W1_grad, W2_init, W2_grad,
                                XA, XB, XC, coeff,
                                output,
                                NH * HF * HF_prime, HF * HF_prime, HF_prime, 1,  # strides for W1: [B,NH,HF,HF_prime]

                                NH * HF_prime * HF, HF_prime * HF, HF, 1,  # strides for W2

                                NH * CS * HF, CS * HF, HF, 1,  # strides for ABCO, output

                                NH * CS * 1, CS * 1, 1, 1,  # strides for coeff

                                CS=CS, HF=HF, HF_prime=HF_prime)

        output = output.permute(0, 2, 1, 3).reshape(B, CS, -1)
        return output, None

##########################################


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
        elif config.inner_net == 'mlp_2_dual_triton':
            self.self_attn = TttM2BMMTritonModule(config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError(f"Inner {config.inner_net} Not Implemented!")

        self.mlp = TttMLP(config)
        # self.input_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.residual_in_fp32
        self.input_layernorm = RMSNorm(hidden_size=self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size=self.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx

        if config.use_compile:
            self.mlp_forward = torch.compile(self._mlp_forward)
        else:
            self.mlp_forward = self._mlp_forward

    def _mlp_forward(self, hidden_states: torch.Tensor):
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
        is_prefill: Optional[bool] = None,
        is_last_in_chunk: Optional[bool] = None,
    ):
        # residual = hidden_states
        # hidden_states = self.input_layernorm(hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.input_layernorm(residual.to(dtype=self.input_layernorm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.input_layernorm.weight,
                self.input_layernorm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.input_layernorm.eps,
            )
        # TTT
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
            is_prefill=is_prefill,
            is_last_in_chunk=is_last_in_chunk,
        )
        # hidden_states = residual + hidden_states  # @xinhao: rms and add are fused in the next step

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.post_attention_layernorm(residual.to(dtype=self.post_attention_layernorm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.post_attention_layernorm.eps,
            )
        # Fully Connected
        hidden_states = self.mlp_forward(hidden_states)

        return hidden_states, residual


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
        # self.norm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.num_hidden_layers = config.num_hidden_layers
        self.inner_net_chunk_size = config.inner_net_chunk_size
        self.config = config

        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        residual = None
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
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    residual=residual,
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

        # hidden_states = self.norm(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

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