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

from transformers.models.ttt_full_prefill_decode_optimize.generation import GenerationMixin, TttCache
from transformers.models.mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

try:
    import tk_m1_prefill
except:
    tk_m1_prefill = None

try:
    import tk_m2_prefill
except:
    tk_m2_prefill = None

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


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


# from xformers impl.
@triton.jit
def gelu(x):
    return 0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))


@triton.jit
def diff_gelu_tl(x):
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['F'],
)
@triton.jit
def _fuse_gate_ln_kernel(__XGate, __X, __Out,
                         __ln_weight, __ln_bias,
                         stride_x_batch,
                         F: tl.constexpr):
    batch = tl.program_id(0)

    rf = tl.arange(0, F)

    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch

    x_inner_offset = rf[None, :]
    ln_inner_offset = rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    ln_offset = ln_inner_offset

    _XGate = __XGate + x_offset
    _X = __X + x_offset
    _Out = __Out + x_offset
    _ln_weight = __ln_weight + ln_offset
    _ln_bias = __ln_bias + ln_offset

    XGate = tl.load(_XGate)
    X = tl.load(_X)
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    ## LN(X)
    mu = tl.sum(X, 1) / F
    var = tl.sum((X - mu) * (X - mu), 1) / F
    std = tl.sqrt(var + 1e-6)
    X_hat = (X - mu) / std  # [1,f]
    LN_X = ln_weight * X_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]

    ## gelu(XGate)
    XGate_activated = gelu(XGate)

    output = XGate_activated * LN_X
    tl.store(_Out, output.to(O_dtype))


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
        
        token_idx = (self.config.inner_net_lr / self.head_dim) / torch.arange(1, self.inner_chunk_size + 1)  # [CS,]
        token_idx = token_idx.reshape(1, 1, -1, 1)  # [1,1,CS,1]
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

        # @xinhao: ln_weight and _bias must be normal tensor instead of nn.Parameter, otherwise will have triton error
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        # self.ln_weight = nn.Parameter(torch.tile(ln_weight_data.reshape(1, 1, 1, -1), (1, self.num_heads, 1, 1)))  # [1,h,1,f]
        self.ln_weight = torch.tile(ln_weight_data.reshape(1, 1, 1, -1), (1, self.num_heads, 1, 1)).to(self.config.dtype).to('cuda') * 1.
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        # self.ln_bias = nn.Parameter(torch.tile(ln_bias_data.reshape(1, 1, 1, -1), (1, self.num_heads, 1, 1)))  # [1,h,1,f]
        self.ln_bias = torch.tile(ln_bias_data.reshape(1, 1, 1, -1), (1, self.num_heads, 1, 1)).to(self.config.dtype).to('cuda') * 1.

        self.out_norm = nn.LayerNorm(self.hidden_size)
        out_ln_weight_data = self.out_norm.weight.data
        out_ln_bias_data = self.out_norm.bias.data
        self.out_ln_weight_data = out_ln_weight_data.to(self.config.dtype).to('cuda')  # [1,F]
        self.out_ln_bias_data = out_ln_bias_data.to(self.config.dtype).to('cuda')

        if self.config.use_compile:
            self.residual_add_post_LN = torch.compile(self._residual_add_post_LN)
            self.gate_out_norm = torch.compile(self._gate_out_norm)
        else:
            self.residual_add_post_LN = self._residual_add_post_LN
            self.gate_out_norm = self._gate_out_norm

        # self.decode_gate_out_norm = self._gate_out_norm
        self.decode_gate_out_norm = self._decode_gate_out_norm

    def _residual_add_post_LN(self, XC, XCW_batch):
        XCW_batch = XC + ln_fwd(XCW_batch, self.ln_weight, self.ln_bias)  # post LN
        return XCW_batch

    def _gate_out_norm(self, B, N, XGate, XCW_batch):
        XGate = XGate.reshape(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, -1)
        XCW_batch = XCW_batch.reshape(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, -1)
        XCW_batch = F.gelu(XGate, approximate='tanh') * self.out_norm(XCW_batch)  # [B*nh,N,f] *  [B*nh,N,f]
        return XCW_batch.contiguous()

    def _decode_gate_out_norm(self, B, N, XGate, XCW_batch):
        XGate = XGate.reshape(B, self.num_heads,
                              N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, -1).contiguous()
        XCW_batch = XCW_batch.reshape(B, self.num_heads,
                                      N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, -1).contiguous()
        output = torch.empty_like(XCW_batch)  # [B,N,F]
        grid = (B, 1, 1)
        _fuse_gate_ln_kernel[grid](XGate, XCW_batch, output,
                                   self.out_ln_weight_data, self.out_ln_bias_data,
                                   XCW_batch.stride(0),
                                   self.hidden_size)
        return output

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
            XCB = XCB.transpose(-1, -2).contiguous()  # [B,F,N]
            XC = causal_conv1d_fn(
                XCB, conv_q_weights, self.conv_q.bias, activation=None
            ).transpose(-1,-2).contiguous()
            XB = causal_conv1d_fn(
                XCB, conv_k_weights, self.conv_k.bias, activation=None
            ).transpose(-1,-2).contiguous()
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


    def get_inner_loop_inputs(
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

        if is_prefill:
            inner_chunk_step_offset = 0
            token_idx = self.token_idx  # [1,1,CS,1]
        else:
            # TODO: keeps recompiling when decoding, so cannot torch.compile this function when decode
            inner_chunk_step_offset = cache_params.seqlen_offset % self.inner_chunk_size
            token_idx = self.token_idx[:, :, inner_chunk_step_offset, :]  # [1,1,CS,1] -> [1,1,1]

        XCBA_gilr = self.qkv_proj(hidden_states)  # [B,N, 3*F + nh]

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
        ilr_gated = F.sigmoid(ilr_gated.permute(0,2,1).reshape(-1,L,1))  # [B,N,nh] -> [B,nh,N] -> [B*nh,N,1]

        XA = XA.contiguous()
        XB = XB.contiguous()
        XC = XC.contiguous()
        XGate = XGate.contiguous()
        ilr_gated = ilr_gated.contiguous()

        return XC, XB, XA, token_idx, ilr_gated, XGate

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
        inner_chunk_size: Optional[int] = None,
        last_chunk_params_dic: Optional[Dict[str, torch.Tensor]] = None,
        return_params: Optional[bool] = False,
        is_prefill: Optional[bool] = False,
        is_last_in_chunk: Optional[bool] = None,
    ):
        # XC, XB, XA, token_idx, ilr_gated, XGate = self.get_inner_loop_inputs(
        #     hidden_states, position_ids=position_ids,
        #     cache_params=cache_params, inner_chunk_size=inner_chunk_size, is_prefill=is_prefill
        # )
        XC  = hidden_states
        token_idx = self.token_idx
        B, L, D = XC.shape

        XC = XC.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(-1, L, self.head_dim)  # [B*nh,N,f]
        XB = XC
        XA = XC

        B_mul_NH, N, HF = XA.shape
        B = B_mul_NH // self.num_heads

        ilr_gated = torch.zeros((B_mul_NH, N, 1), dtype=XA.dtype, device=XA.device)

        if is_prefill:
            inputs = {'XC': XC, 'XB': XB, 'XA': XA, 'ilr_gated': ilr_gated}
        else:
            inputs = {'XC': XC, 'XB': XB, 'XA': XA, 'token_idx': token_idx, 'ilr_gated': ilr_gated}

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

        # if is_prefill:
        #     XCW_batch = self.gate_out_norm(B, N, XGate, XCW_batch)
        # else:
        #     XCW_batch = self.decode_gate_out_norm(B, N, XGate, XCW_batch)

        # z_batch = self.project_inner_loop_outputs(XCW_batch)  # [B,N,F]

        z_batch = XCW_batch.reshape(B, N, -1)

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
            XA, XB, XC: [B*nh, N, f]
            token_idx: [1,]
            ilr_gated: [B*nh, N, 1]
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
                inner_chunk_size=self.inner_chunk_size,
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
# With LN
def m1_prefill_chunk(states, inputs, i, ln_weight, ln_bias, Attn_b):
    W1_init = states['W1_states']
    b1_init = states['b1_states']
    XA_chunk, XB_chunk, XC_chunk, \
    coeff_chunk, coeff_chunk_last = inputs['XA'][i], inputs['XB'][i], inputs['XC'][i], \
                                    inputs['coeff'][i], inputs['coeff_last'][i]  # [B*nh,CS,CS], [B*nh,1,CS]

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K,f] @ [B*nh,f,f] -> [B*nh,K,f]
    reconstruction_target = XA_chunk - XB_chunk
    grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B*nh,K=1,f]: torch.compile makes it a lot faster

    b1_bar = b1_init - (coeff_chunk * Attn_b) @ grad_l_wrt_Z1

    Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))  # [B*nh,K,K]
    Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar  # [B*nh,K,f] @ [B*nh,f,f] - ([B*nh,K,1] * [B*nh,K,K]) @ [B*nh,K,f]

    W1_init.sub_((coeff_chunk_last * XB_chunk.transpose(-1, -2)) @ grad_l_wrt_Z1)  # in-place update: [B*nh,f,f] - ([B*nh,1,K] * [B*nh,K,f].t) @ [B*nh,K,f]
    b1_init.copy_(b1_bar[:,-1:])

    return Z1_bar


def m1_decode_end_chunk(states, inputs, ln_weight, ln_bias):
    W1_init = states['W1_states']  # [B*nh,f,f]
    b1_init = states['b1_states']
    W1_grad = states['W1_grad']
    b1_grad = states['b1_grad']

    XA_chunk, XB_chunk, \
    XC_chunk, token_idx, ilr_gated = inputs['XA'], inputs['XB'], inputs['XC'], \
                                     inputs['token_idx'], inputs['ilr_gated']  # [B*nh,N=1,f], [1,1,1], [B*nh,N=1,1]

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ [B*nh,f,f] + [B*nh,1,f] -> [B*nh,K=1,f]
    reconstruction_target = XA_chunk - XB_chunk
    grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B*nh,K=1,f]

    grad_l_wrt_Z1 = ilr_gated * grad_l_wrt_Z1  # [B*nh,K=1,f]

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B*nh,1,f].t @ [B*nh,1,f] + [B*nh,f,f]
    b1_grad.add_(grad_l_wrt_Z1)

    W1_init.sub_(token_idx * W1_grad)  # [B*nh,f,f] - [1,N=1,1] * [B*nh,f,f]
    b1_init.sub_(token_idx * b1_grad)  # [B*nh,1,f] - [1,N=1,1] * [B*nh,1,f]
    Z1_bar = XC_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ ([B*nh,f,f] - [B*nh,1,1] * [B*nh,f,f])
    W1_grad.zero_()
    b1_grad.zero_()

    return Z1_bar


def m1_decode(states, inputs, ln_weight, ln_bias):
    W1_init = states['W1_states']  # [B*nh,f,f]
    b1_init = states['b1_states']
    W1_grad = states['W1_grad']
    b1_grad = states['b1_grad']

    XA_chunk, XB_chunk, \
    XC_chunk, token_idx, ilr_gated = inputs['XA'], inputs['XB'], inputs['XC'], \
                                     inputs['token_idx'], inputs['ilr_gated']  # [B*nh,N=1,f], [1,1,1], [B*nh,N=1,1]

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ [B*nh,f,f] + [B*nh,1,f] -> [B*nh,K=1,f]
    reconstruction_target = XA_chunk - XB_chunk
    grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B*nh,K=1,f]

    grad_l_wrt_Z1 = ilr_gated * grad_l_wrt_Z1  # [B*nh,K=1,f]

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B*nh,1,f].t @ [B*nh,1,f] + [B*nh,f,f]
    b1_grad.add_(grad_l_wrt_Z1)

    W1_last = W1_init - (token_idx * W1_grad)  # [B*nh,f,f] - [1,N=1,1] * [B*nh,f,f]
    b1_last = b1_init - (token_idx * b1_grad)  # [B*nh,1,f] - [1,N=1,1] * [B*nh,1,f]
    Z1_bar = XC_chunk @ W1_last + b1_last      # [B*nh,K=1,f] @ [B*nh,f,f]

    return Z1_bar


class TttM1BMMModule(TttBaseModule):

    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.ones(size=(self.num_heads, 1, self.head_dim)))

        if self.config.use_compile:
            self.prefill_chunk = torch.compile(m1_prefill_chunk)  # TODO: this compile speeds up from 39k to 49k, but in micro-bench it seems not helpful
                                                                  # TODO: maybe because micro compiles the whole for loop, which is too large
            # self.prefill_chunk = m1_prefill_chunk
            self.decode_end_chunk = torch.compile(m1_decode_end_chunk)
            self.decode = torch.compile(m1_decode)
        else:
            self.prefill_chunk = m1_prefill_chunk
            self.decode_end_chunk = m1_decode_end_chunk
            self.decode = m1_decode

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):
        states = {
            "W1_states": cache_params.params_dict["W1_states"][self.layer_idx],
            "b1_states": cache_params.params_dict["b1_states"][self.layer_idx],
            "W1_grad": cache_params.params_dict["W1_grad"][self.layer_idx],
            "b1_grad": cache_params.params_dict["b1_grad"][self.layer_idx],
        }
        XC_residual = inputs['XC']

        if is_prefill:
            B_mul_NH, N, HF = inputs['XA'].shape  # [B*nh,N,f]
            NC = N // inner_chunk_size
            inputs = tree_map(lambda x: x.reshape(B_mul_NH, NC, inner_chunk_size, -1).transpose(1,0).contiguous(),
                              inputs)  # [B*nh,N,f] -> [B*nh,NC,CS,f] -> [NC,B*nh,CS,f]
            ilr_gated = inputs.pop('ilr_gated').transpose(-1,-2)  # [NC,B*nh,1,CS]
            inputs['coeff'] = self.token_idx * ilr_gated  # [1,1,CS,1] * [NC,B*nh,1,CS] -> [NC,B*nh,CS,CS]
            inputs['coeff_last'] = inputs['coeff'][...,-1:,:]  # pre-sclice: [NC,B*nh,1,CS]

            Attn_b = torch.tril(torch.ones(inner_chunk_size, inner_chunk_size,
                                           dtype=ilr_gated.dtype, device=ilr_gated.device))  # [CS,CS]

            def for_loop(states, inputs):
                output_tensor = torch.empty_like(inputs['XA'])
                for i in range(NC):
                    Z1_bar = self.prefill_chunk(states, inputs, i, self.ln_weight, self.ln_bias, Attn_b)
                    output_tensor[i] = Z1_bar
                return output_tensor  # [NC, B*nh, K, f]

            XCW_batch = for_loop(
                states,  # [B*nh,f,f]
                inputs,  # [NC,B*nh,CS,f]
            )
            XCW_batch = XCW_batch.transpose(1,0).reshape(B_mul_NH, -1, HF).contiguous()  # [B*h,N,f]

        else:
            # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
            B_mul_NH, N, HF = inputs['XA'].shape  # [B*nh,N=1,f]
            # inputs['ilr_gated']: [B*nh,N=1,1]
            # inputs['token_idx']: [1,1,1]

            if is_last_in_chunk:
                XCW_batch = self.decode_end_chunk(
                    states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                    inputs,  # [B*nh,f]
                    self.ln_weight,  # [nh,1,f]
                    self.ln_bias, # [nh,1,f]
                )  # ret: [B*nh,N=1,f]
            else:
                XCW_batch = self.decode(
                    states,  # [B*nh,f,f], cloned from W1, safe for in-place op
                    inputs,  # [B*nh,f]
                    self.ln_weight,  # [nh,1,f]
                    self.ln_bias,  # [nh,1,f]
                )  # ret: [B*nh,N=1,f]

        XCW_batch = self.residual_add_post_LN(XC_residual, XCW_batch)

        return XCW_batch, None

##########################################


####### M2 Decode Module #######
## With LN
def m2_prefill_chunk(states, inputs, i, ln_weight, ln_bias, Attn_b):
    W1_init = states['W1_states']
    b1_init = states['b1_states']
    W2_init = states['W2_states']
    b2_init = states['b2_states']
    XA_chunk, XB_chunk, XC_chunk, \
    coeff_chunk, coeff_chunk_last = inputs['XA'][i], inputs['XB'][i], inputs['XC'][i], \
                                    inputs['coeff'][i], inputs['coeff_last'][i]  # [B*nh,CS,CS], [B*nh,1,CS]

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K,f] @ [B*nh,f,4f] + [B*nh,1,4f] -> [B*nh,K,4f]
    X2 = F.gelu(Z1, approximate='tanh')
    Z2 = X2 @ W2_init + b2_init

    l2_target = XA_chunk - XB_chunk
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, l2_target, ln_weight, ln_bias)  # [B*nh,K=1,f]
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-1, -2) * diff_gelu(Z1)

    b1_bar = b1_init - (coeff_chunk * Attn_b) @ grad_l_wrt_Z1  # [B*nh,1,4f] - ([B*nh,K,K] * [K,K]) @ [B*nh,K,4f]
    Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))  # [B*nh,K,K]
    Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar  # [B*nh,K,f] @ [B*nh,f,4f] - ([K,K] * [B*nh,K,K]) @ [B*nh,K,4f] + [B*nh,K,4f]

    X2_bar = F.gelu(Z1_bar, approximate='tanh')

    b2_bar = b2_init - (coeff_chunk * Attn_b) @ grad_l_wrt_Z2  # [B*nh,1,4f] - ([B*nh,K,K] * [K,K]) @ [B*nh,K,4f]
    Attn2 = torch.tril(X2_bar @ X2.transpose(-1, -2))  # [B*nh,K,K]
    Z2_bar = X2_bar @ W2_init - (coeff_chunk * Attn2) @ grad_l_wrt_Z2 + b2_bar

    W1_init.sub_((coeff_chunk_last * XB_chunk.transpose(-1, -2)) @ grad_l_wrt_Z1)  # in-place update: [B*nh,f,4f] - ([B*nh,1,K] * [B*nh,K,f].t) @ [B*nh,K,4f]
    b1_init.copy_(b1_bar[:,-1:])
    W2_init.sub_((coeff_chunk_last * X2.transpose(-1, -2)) @ grad_l_wrt_Z2)  # in-place update: [B*nh,4f,f] - ([B*nh,1,K] * [B*nh,K,4f].t) @ [B*nh,K,f]
    b2_init.copy_(b2_bar[:, -1:])

    return Z2_bar

def m2_decode_end_chunk(states, inputs, ln_weight, ln_bias):
    W1_init = states['W1_states']
    W1_grad = states['W1_grad']
    b1_init = states['b1_states']
    b1_grad = states['b1_grad']
    W2_init = states['W2_states']
    W2_grad = states['W2_grad']
    b2_init = states['b2_states']
    b2_grad = states['b2_grad']

    XA_chunk, XB_chunk, \
    XC_chunk, token_idx, ilr_gated = inputs['XA'], inputs['XB'], inputs['XC'], \
                                     inputs['token_idx'], inputs['ilr_gated']  # [B*nh,N=1,f], [1,1,1], [B*nh,N=1,1]

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ [B*nh,f,f] + [B*nh,1,f] -> [B*nh,K=1,f]
    X2 = F.gelu(Z1, approximate='tanh')
    Z2 = X2 @ W2_init + b2_init

    l2_target = XA_chunk - XB_chunk
    grad_l_wrt_Z2 = ilr_gated * ln_fused_l2_bwd(Z2, l2_target, ln_weight, ln_bias)  # [B*nh,K=1,f]
    grad_l_wrt_Z1 = ilr_gated * (grad_l_wrt_Z2 @ W2_init.transpose(-1, -2) * diff_gelu(Z1))  # [B*nh,K=1,4f]

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B*nh,1,f].t @ [B*nh,1,4f]
    b1_grad.add_(grad_l_wrt_Z1)
    W1_init.sub_(token_idx * W1_grad)
    b1_init.sub_(token_idx * b1_grad)
    Z1_bar = XC_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ ([B*nh,f,f] - [B*nh,1,1] * [B*nh,f,f])

    X2_bar = F.gelu(Z1_bar, approximate='tanh')

    W2_grad.add_(X2.transpose(-1, -2) @ grad_l_wrt_Z2)  # [B*nh,K,4f].t @ [B*nh,K,f]
    b2_grad.add_(grad_l_wrt_Z2)
    W2_init.sub_(token_idx * W2_grad)
    b2_init.sub_(token_idx * b2_grad)
    Z2_bar = X2_bar @ W2_init + b2_init  # [B*nh,K=1,f]

    W1_grad.zero_()
    b1_grad.zero_()
    W2_grad.zero_()
    b2_grad.zero_()

    return Z2_bar

def m2_decode(states, inputs, ln_weight, ln_bias):
    W1_init = states['W1_states']
    W1_grad = states['W1_grad']
    b1_init = states['b1_states']
    b1_grad = states['b1_grad']
    W2_init = states['W2_states']
    W2_grad = states['W2_grad']
    b2_init = states['b2_states']
    b2_grad = states['b2_grad']

    XA_chunk, XB_chunk, \
    XC_chunk, token_idx, ilr_gated = inputs['XA'], inputs['XB'], inputs['XC'], \
                                     inputs['token_idx'], inputs['ilr_gated']  # [B*nh,N=1,f], [1,1,1], [B*nh,N=1,1]

    Z1 = XB_chunk @ W1_init + b1_init  # [B*nh,K=1,f] @ [B*nh,f,f] + [B*nh,1,f] -> [B*nh,K=1,f]
    X2 = F.gelu(Z1, approximate='tanh')
    Z2 = X2 @ W2_init + b2_init

    l2_target = XA_chunk - XB_chunk
    grad_l_wrt_Z2 = ilr_gated * ln_fused_l2_bwd(Z2, l2_target, ln_weight, ln_bias)  # [B*nh,K=1,f]
    grad_l_wrt_Z1 = ilr_gated * (grad_l_wrt_Z2 @ W2_init.transpose(-1, -2) * diff_gelu(Z1))  # [B*nh,K=1,4f]

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B*nh,1,f].t @ [B*nh,1,4f]
    b1_grad.add_(grad_l_wrt_Z1)
    W1_last = W1_init - token_idx * W1_grad
    b1_last = b1_init - token_idx * b1_grad
    Z1_bar = XC_chunk @ W1_last + b1_last  # [B*nh,K=1,f] @ ([B*nh,f,f] - [B*nh,1,1] * [B*nh,f,f])

    X2_bar = F.gelu(Z1_bar, approximate='tanh')

    W2_grad.add_(X2.transpose(-1, -2) @ grad_l_wrt_Z2)  # [B*nh,K,4f].t @ [B*nh,K,f]
    b2_grad.add_(grad_l_wrt_Z2)
    W2_last = W2_init - token_idx * W2_grad
    b2_last = b2_init - token_idx * b2_grad
    Z2_bar = X2_bar @ W2_last + b2_last  # [B*nh,K=1,f]

    return Z2_bar


class TttM2BMMModule(TttBaseModule):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 1, 4 * self.head_dim)))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 1, self.head_dim)))

        if self.config.use_compile:
            self.prefill_chunk = torch.compile(m2_prefill_chunk)
            self.decode_end_chunk = torch.compile(m2_decode_end_chunk)
            self.decode = torch.compile(m2_decode)
        else:
            self.prefill_chunk = m2_prefill_chunk
            self.decode_end_chunk = m2_decode_end_chunk
            self.decode = m2_decode

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):
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
        XC_residual = inputs['XC']

        if is_prefill:
            B_mul_NH, N, HF = inputs['XA'].shape  # [B*nh,N,f]
            NC = N // inner_chunk_size
            inputs = tree_map(lambda x: x.reshape(B_mul_NH, NC, inner_chunk_size, -1).transpose(1, 0).contiguous(),
                              inputs)  # [B*nh,N,f] -> [B*nh,NC,CS,f] -> [NC,B*nh,CS,f]
            ilr_gated = inputs.pop('ilr_gated').transpose(-1, -2)  # [NC,B*nh,1,CS]
            inputs['coeff'] = self.token_idx * ilr_gated  # [1,1,CS,1] * [NC,B*nh,1,CS] -> [NC,B*nh,CS,CS]
            inputs['coeff_last'] = inputs['coeff'][..., -1:, :]  # pre-sclice: [NC,B*nh,1,CS]

            Attn_b = torch.tril(torch.ones(inner_chunk_size, inner_chunk_size,
                                           dtype=ilr_gated.dtype, device=ilr_gated.device))  # [CS,CS]

            def for_loop(states, inputs):
                output_tensor = torch.empty_like(inputs['XA'])
                for i in range(NC):
                    Z2_bar = self.prefill_chunk(states, inputs, i,
                                                self.ln_weight, self.ln_bias, Attn_b)
                    output_tensor[i] = Z2_bar
                return output_tensor  # [NC, B*nh, K, f]

            XCW_batch = for_loop(
                states,  # [B*nh,f,f]
                inputs,  # [NC,B*nh,CS,f]
            )
            XCW_batch = XCW_batch.transpose(1, 0).reshape(B_mul_NH, -1, HF).contiguous()  # [B*h,N,f]

        else:
            # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
            B_mul_NH, N, HF = inputs['XA'].shape  # [B*nh,N=1,f]

            if is_last_in_chunk:
                XCW_batch = self.decode_end_chunk(
                    states,  # [B*nh,f,f]
                    inputs,  # [B*nh,f]
                    self.ln_weight,  # [nh,1,f]
                    self.ln_bias,  # [nh,1,f]
                )  # ret: [B*nh,N=1,f]
            else:
                XCW_batch = self.decode(
                    states,  # [B*nh,f,f]
                    inputs,  # [B*nh,f]
                    self.ln_weight,  # [nh,1,f]
                    self.ln_bias,  # [nh,1,f]
                )  # ret: [B*nh,N=1,f]

        XCW_batch = self.residual_add_post_LN(XC_residual, XCW_batch)

        return XCW_batch, None


##########################################


####### M1 Triton Decode Module #######

####### Decode #######
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['HF'],
    restore_value=['__W1', '__b1', '__W1_grad', '__b1_grad']
)
@triton.jit
def _m1_decode_kernel(__W1, __W1_grad, __b1, __b1_grad,
                      __XA, __XB, __XC,
                      __ln_weight, __ln_bias,
                      __ilr_gated, __token_idx, __Out,
                      stride_w_batch, stride_w_head, stride_w_fin,
                      stride_b_batch, stride_b_head, stride_b_f,
                      stride_x_batch, stride_x_head, stride_x_n,
                      stride_ln_head, stride_ln_f,
                      stride_ilr_batch, stride_ilr_head,
                      CS: tl.constexpr, HF: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)

    W_dtype = __W1.type.element_ty
    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch + head * stride_x_head
    w_block_offset = batch * stride_w_batch + head * stride_w_head
    b_block_offset = batch * stride_b_batch + head * stride_b_head
    ln_block_offset = head * stride_ln_head
    ilr_block_offset = batch * stride_ilr_batch + head * stride_ilr_head

    x_inner_offset = rc[:, None] * stride_x_n + rf[None, :]
    w_inner_offset = rf[:, None] * stride_w_fin + rf[None, :]
    b_inner_offset = rc[:, None] * stride_b_f + rf[None, :]
    ln_inner_offset = rc[:, None] * stride_ln_f + rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    w_offset = w_block_offset + w_inner_offset
    b_offset = b_block_offset + b_inner_offset
    ln_offset = ln_block_offset + ln_inner_offset
    ilr_offset = ilr_block_offset

    _XA = __XA + x_offset
    _XB = __XB + x_offset
    _XC = __XC + x_offset
    _Out = __Out + x_offset
    _W1 = __W1 + w_offset
    _W1_grad = __W1_grad + w_offset
    _b1 = __b1 + b_offset
    _b1_grad = __b1_grad + b_offset
    _ln_weight = __ln_weight + ln_offset
    _ln_bias = __ln_bias + ln_offset
    _ilr_gated = __ilr_gated + ilr_offset
    _token_idx = __token_idx

    XA = tl.load(_XA)
    XB = tl.load(_XB)
    XC = tl.load(_XC)
    token_idx = tl.load(_token_idx)
    ilr_gated = tl.load(_ilr_gated)
    W1 = tl.load(_W1)
    W1_grad = tl.load(_W1_grad)
    b1 = tl.load(_b1)
    b1_grad = tl.load(_b1_grad)
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    Z1 = tl.sum(tl.trans(XB) * W1, axis=0)[None, :] + b1  # [1,f] @ [f,f] + [1,f]
    l2_target = XA - XB

    mu = tl.sum(Z1, 1) / HF
    var = tl.sum((Z1 - mu) * (Z1 - mu), 1) / HF

    std = tl.sqrt(var + 1e-6)
    Z1_hat = (Z1 - mu) / std  # [1,f]

    # Scale and shift
    LN_out = ln_weight * Z1_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]

    dl_dLN_out = LN_out - l2_target  # [1,f]

    dl_dZ1_hat = dl_dLN_out * ln_weight  # [1,f]

    dl_dZ1_term_1 = HF * dl_dZ1_hat
    dl_dZ1_term_2 = tl.sum(dl_dZ1_hat, 1)
    dl_dZ1_term_3 = Z1_hat * tl.sum(dl_dZ1_hat * Z1_hat, 1)
    dl_dZ1_sum = dl_dZ1_term_1 + dl_dZ1_term_2 + dl_dZ1_term_3
    dl_dZ1 = dl_dZ1_sum / (std * HF * 100)
    dl_dZ1 = dl_dZ1 * 100.

    ilr_mul_dl_dZ1 = ilr_gated * dl_dZ1  # [K=1,1] * [K=1,f]

    ##
    W1_grad += tl.trans(XB) * ilr_mul_dl_dZ1
    b1_grad += ilr_mul_dl_dZ1
    tl.store(_W1_grad, W1_grad.to(W_dtype))
    tl.store(_b1_grad, b1_grad.to(W_dtype))

    W1_bar = W1 - token_idx * W1_grad
    b1_bar = b1 - token_idx * b1_grad

    Z1_bar = tl.sum(tl.trans(XC) * W1_bar, axis=0)[None, :] + b1_bar

    ## Residual + Post LN
    mu_bar = tl.sum(Z1_bar, 1) / HF
    var_bar = tl.sum((Z1_bar - mu_bar) * (Z1_bar - mu_bar), 1) / HF
    std_bar = tl.sqrt(var_bar + 1e-6)
    Z1_bar_hat = (Z1_bar - mu_bar) / std_bar  # [1,f]
    LN_out_bar = ln_weight * Z1_bar_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    Z1_bar = XC + LN_out_bar

    tl.store(_Out, Z1_bar.to(O_dtype))

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['HF'],
    restore_value=['__W1', '__b1', '__W1_grad', '__b1_grad']
)
@triton.jit
def _m1_decode_end_chunk_kernel(__W1, __W1_grad, __b1, __b1_grad,
                                __XA, __XB, __XC,
                                __ln_weight, __ln_bias,
                                __ilr_gated, __token_idx, __Out,
                                stride_w_batch, stride_w_head, stride_w_fin,
                                stride_b_batch, stride_b_head, stride_b_f,
                                stride_x_batch, stride_x_head, stride_x_n,
                                stride_ln_head, stride_ln_f,
                                stride_ilr_batch, stride_ilr_head,
                                CS: tl.constexpr, HF: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)

    W_dtype = __W1.type.element_ty
    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch + head * stride_x_head
    w_block_offset = batch * stride_w_batch + head * stride_w_head
    b_block_offset = batch * stride_b_batch + head * stride_b_head
    ln_block_offset = head * stride_ln_head
    ilr_block_offset = batch * stride_ilr_batch + head * stride_ilr_head

    x_inner_offset = rc[:, None] * stride_x_n + rf[None, :]
    w_inner_offset = rf[:, None] * stride_w_fin + rf[None, :]
    b_inner_offset = rc[:, None] * stride_b_f + rf[None, :]
    ln_inner_offset = rc[:, None] * stride_ln_f + rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    w_offset = w_block_offset + w_inner_offset
    b_offset = b_block_offset + b_inner_offset
    ln_offset = ln_block_offset + ln_inner_offset
    ilr_offset = ilr_block_offset

    _XA = __XA + x_offset
    _XB = __XB + x_offset
    _XC = __XC + x_offset
    _Out = __Out + x_offset
    _W1 = __W1 + w_offset
    _W1_grad = __W1_grad + w_offset
    _b1 = __b1 + b_offset
    _b1_grad = __b1_grad + b_offset
    _ln_weight = __ln_weight + ln_offset
    _ln_bias = __ln_bias + ln_offset
    _ilr_gated = __ilr_gated + ilr_offset
    _token_idx = __token_idx

    XA = tl.load(_XA)
    XB = tl.load(_XB)
    XC = tl.load(_XC)
    token_idx = tl.load(_token_idx)
    ilr_gated = tl.load(_ilr_gated)
    W1 = tl.load(_W1)
    W1_grad = tl.load(_W1_grad)
    b1 = tl.load(_b1)
    b1_grad = tl.load(_b1_grad)
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    Z1 = tl.sum(tl.trans(XB) * W1, axis=0)[None, :] + b1  # [1,f] @ [f,f] + [1,f]
    l2_target = XA - XB

    mu = tl.sum(Z1, 1) / HF
    var = tl.sum((Z1 - mu) * (Z1 - mu), 1) / HF

    std = tl.sqrt(var + 1e-6)
    Z1_hat = (Z1 - mu) / std  # [1,f]

    # Scale and shift
    LN_out = ln_weight * Z1_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]

    dl_dLN_out = LN_out - l2_target  # [1,f]

    dl_dZ1_hat = dl_dLN_out * ln_weight  # [1,f]

    dl_dZ1_term_1 = HF * dl_dZ1_hat
    dl_dZ1_term_2 = tl.sum(dl_dZ1_hat, 1)
    dl_dZ1_term_3 = Z1_hat * tl.sum(dl_dZ1_hat * Z1_hat, 1)
    dl_dZ1_sum = dl_dZ1_term_1 + dl_dZ1_term_2 + dl_dZ1_term_3
    dl_dZ1 = dl_dZ1_sum / (std * HF * 100)
    dl_dZ1 = dl_dZ1 * 100.

    ilr_mul_dl_dZ1 = ilr_gated * dl_dZ1  # [K=1,1] * [K=1,f]

    ##
    W1_grad += tl.trans(XB) * ilr_mul_dl_dZ1
    b1_grad += ilr_mul_dl_dZ1
    tl.store(_W1_grad, tl.zeros_like(W1_grad).to(W_dtype))
    tl.store(_b1_grad, tl.zeros_like(b1_grad).to(W_dtype))

    W1_bar = W1 - token_idx * W1_grad
    b1_bar = b1 - token_idx * b1_grad
    tl.store(_W1, W1_bar.to(W_dtype))
    tl.store(_b1, b1_bar.to(W_dtype))

    Z1_bar = tl.sum(tl.trans(XC) * W1_bar, axis=0)[None, :] + b1_bar

    ## Residual + Post LN
    mu_bar = tl.sum(Z1_bar, 1) / HF
    var_bar = tl.sum((Z1_bar - mu_bar) * (Z1_bar - mu_bar), 1) / HF
    std_bar = tl.sqrt(var_bar + 1e-6)
    Z1_bar_hat = (Z1_bar - mu_bar) / std_bar  # [1,f]
    LN_out_bar = ln_weight * Z1_bar_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    Z1_bar = XC + LN_out_bar

    tl.store(_Out, Z1_bar.to(O_dtype))


class TttM1BMMTKModule(TttBaseModule):

    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.ones(size=(self.num_heads, 1, self.head_dim)))

        if self.config.use_compile:
            # Use PyTorch for decode now
            self.decode_end_chunk = torch.compile(m1_decode_end_chunk)
            self.decode = torch.compile(m1_decode)
        else:
            self.decode_end_chunk = m1_decode_end_chunk
            self.decode = m1_decode

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):

        B_mul_NH, N, HF = inputs['XA'].shape  # [B*nh,N,f]
        B = B_mul_NH // self.num_heads
        NH = self.num_heads
        XC_residual = inputs['XC']

        if is_prefill:
            W1_init = cache_params.params_dict["W1_states"][self.layer_idx].reshape(B, NH, HF, HF)
            b1_init = cache_params.params_dict["b1_states"][self.layer_idx].reshape(B, NH, 1, HF)

            CS = inner_chunk_size
            NC = N // CS
            inputs = tree_map(lambda x: x.reshape(B, NH, NC, CS, -1).contiguous(), inputs)  # [B*nh,N,f/1] -> [B,nh,nc,cs,f/1]
            ilr_gated = inputs.pop('ilr_gated').transpose(-1, -2)  # [B,nh,nc,1,cs]

            inputs['coeff'] = self.token_idx[None,:] * ilr_gated  # [1,1,1,cs,1] * [B,nh,nc,1,cs] -> [B,nh,nc,CS,CS]

            XA, XB, XC, coeff = inputs['XA'], inputs['XB'], inputs['XC'], inputs['coeff']  # [B,nh,nc,cs,f/1]
            input_device = XA.device
            input_dtype = XA.dtype
            output = torch.empty_like(XA)

            ln_weight = self.ln_weight.squeeze(0).expand(-1, CS, -1).contiguous()
            ln_bias = self.ln_bias.squeeze(0).expand(-1, CS, -1).contiguous()
            b1_init = b1_init.expand(-1, -1, CS, -1).contiguous()
            cumsum_matrix = torch.tril(torch.ones(CS, CS, dtype=input_dtype, device=input_device))
            make_last_b_matrix = torch.zeros(CS, CS, dtype=input_dtype, device=input_device)
            make_last_coeff_1_matrix = torch.zeros(CS, HF, dtype=input_dtype, device=input_device)
            make_last_b_matrix[:,-1] = 1.
            make_last_coeff_1_matrix[-1,:] = 1.

            # tk_m1_prefill.prefill_whole_loop_LN_bias_fp16(
            #     W1_init, b1_init, ln_weight, ln_bias,
            #     cumsum_matrix, make_last_b_matrix, make_last_coeff_1_matrix,
            #     XA, XB, XC, coeff, output
            # )
            tk_m1_prefill.prefill_whole_loop_LN_bias_res_PLN_fp16(
                W1_init, b1_init, ln_weight, ln_bias,
                cumsum_matrix, make_last_b_matrix, make_last_coeff_1_matrix,
                XA, XB, XC, coeff, output
            )
            b1_init = b1_init[:,:,-1:,:].reshape(B_mul_NH, 1, -1)
            cache_params.params_dict["b1_states"][self.layer_idx].copy_(b1_init)

            output = output.reshape(B_mul_NH, N, HF)
            # output = self.residual_add_post_LN(XC_residual, output)  # residual + postLN

        else:
            W1 = cache_params.params_dict["W1_states"][self.layer_idx].reshape(B, NH, HF, HF)
            b1 = cache_params.params_dict["b1_states"][self.layer_idx].reshape(B, NH, 1, HF)
            W1_grad = cache_params.params_dict["W1_grad"][self.layer_idx].reshape(B, NH, HF, HF)
            b1_grad = cache_params.params_dict["b1_grad"][self.layer_idx].reshape(B, NH, 1, HF)

            token_idx = inputs.pop('token_idx')  # [1,1,1]

            # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
            inputs = tree_map(lambda x: x.reshape(B, NH, N, -1), inputs)  # [B*nh,N=1,f], [B*nh,N=1,1] -> [BS,nh,N=1,f/1]
            XA, XB, XC, ilr_gated = inputs['XA'], inputs['XB'], inputs['XC'], inputs['ilr_gated']  # [B,nh,N=1,f/1]

            output = torch.empty_like(XA)  # [B,nh,N,f]
            grid = (B, NH, 1)
            CS = 1

            if is_last_in_chunk:
                _m1_decode_end_chunk_kernel[grid](W1, W1_grad, b1, b1_grad,
                                                  XA, XB, XC,
                                                  self.ln_weight, self.ln_bias,
                                                  ilr_gated, token_idx, output,
                                                  W1.stride(0), W1.stride(1), W1.stride(2),
                                                  b1.stride(0), b1.stride(1), b1.stride(2),
                                                  XA.stride(0), XA.stride(1), XA.stride(2),
                                                  self.ln_weight.stride(1), self.ln_weight.stride(2),
                                                  ilr_gated.stride(0), ilr_gated.stride(1),
                                                  CS, HF)
            else:
                # @xinhao: XB and XC have the same stride, but different from XA's due to different way of getting XBC and XA
                # @xinhao: Either (1) .contiguous() in get_inner_input(), or (2) use differnt strides at triton kernel
                _m1_decode_kernel[grid](W1, W1_grad, b1, b1_grad,
                                        XA, XB, XC,
                                        self.ln_weight, self.ln_bias,
                                        ilr_gated, token_idx, output,
                                        W1.stride(0), W1.stride(1), W1.stride(2),
                                        b1.stride(0), b1.stride(1), b1.stride(2),
                                        XA.stride(0), XA.stride(1), XA.stride(2),
                                        self.ln_weight.stride(1), self.ln_weight.stride(2),
                                        ilr_gated.stride(0), ilr_gated.stride(1),
                                        CS, HF)

            output = output.reshape(B_mul_NH, N, HF)

        return output, None

##########################################


####### M2 Triton Decode Module #######
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['HF', 'HF_prime'],
    restore_value=['__W1', '__b1', '__W1_grad', '__b1_grad',
                   '__W2', '__b2', '__W2_grad', '__b2_grad']
)
@triton.jit
def _m2_decode_kernel(__W1, __W1_grad, __b1, __b1_grad,
                      __W2, __W2_grad, __b2, __b2_grad,
                      __XA, __XB, __XC,
                      __ln_weight, __ln_bias,
                      __ilr_gated, __token_idx, __Out,
                      stride_w1_batch, stride_w1_head, stride_w1_fin,
                      stride_b1_batch, stride_b1_head, stride_b1_f,
                      stride_w2_batch, stride_w2_head, stride_w2_fin,
                      stride_b2_batch, stride_b2_head, stride_b2_f,
                      stride_x_batch, stride_x_head, stride_x_n,
                      stride_ln_head, stride_ln_f,
                      stride_ilr_batch, stride_ilr_head,
                      CS: tl.constexpr, HF: tl.constexpr, HF_prime: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)
    rf_prime = tl.arange(0, HF_prime)

    W_dtype = __W1.type.element_ty
    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch + head * stride_x_head
    w1_block_offset = batch * stride_w1_batch + head * stride_w1_head
    b1_block_offset = batch * stride_b1_batch + head * stride_b1_head
    w2_block_offset = batch * stride_w2_batch + head * stride_w2_head
    b2_block_offset = batch * stride_b2_batch + head * stride_b2_head
    ln_block_offset = head * stride_ln_head
    ilr_block_offset = batch * stride_ilr_batch + head * stride_ilr_head

    x_inner_offset = rc[:, None] * stride_x_n + rf[None, :]
    w1_inner_offset = rf[:, None] * stride_w1_fin + rf_prime[None, :]
    b1_inner_offset = rc[:, None] * stride_b1_f + rf_prime[None, :]
    w2_inner_offset = rf_prime[:, None] * stride_w2_fin + rf[None, :]
    b2_inner_offset = rc[:, None] * stride_b2_f + rf[None, :]
    ln_inner_offset = rc[:, None] * stride_ln_f + rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    w1_offset = w1_block_offset + w1_inner_offset
    b1_offset = b1_block_offset + b1_inner_offset
    w2_offset = w2_block_offset + w2_inner_offset
    b2_offset = b2_block_offset + b2_inner_offset
    ln_offset = ln_block_offset + ln_inner_offset
    ilr_offset = ilr_block_offset

    _XA = __XA + x_offset
    _XB = __XB + x_offset
    _XC = __XC + x_offset
    _Out = __Out + x_offset
    _W1 = __W1 + w1_offset
    _W1_grad = __W1_grad + w1_offset
    _b1 = __b1 + b1_offset
    _b1_grad = __b1_grad + b1_offset
    _W2 = __W2 + w2_offset
    _W2_grad = __W2_grad + w2_offset
    _b2 = __b2 + b2_offset
    _b2_grad = __b2_grad + b2_offset
    _ln_weight = __ln_weight + ln_offset
    _ln_bias = __ln_bias + ln_offset
    _ilr_gated = __ilr_gated + ilr_offset
    _token_idx = __token_idx

    XA = tl.load(_XA)
    XB = tl.load(_XB)
    XC = tl.load(_XC)
    token_idx = tl.load(_token_idx)
    ilr_gated = tl.load(_ilr_gated)
    W1 = tl.load(_W1)
    W1_grad = tl.load(_W1_grad)
    b1 = tl.load(_b1)
    b1_grad = tl.load(_b1_grad)
    W2 = tl.load(_W2)
    W2_grad = tl.load(_W2_grad)
    b2 = tl.load(_b2)
    b2_grad = tl.load(_b2_grad)
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    Z1 = tl.sum(tl.trans(XB) * W1, axis=0)[None,:] + b1  # [1,f] @ [f,4f] + [1,4f]
    X2 = gelu(Z1)
    Z2 = tl.sum(tl.trans(X2) * W2, axis=0)[None,:] + b2  # [1,4f] @ [4f,f] + [1,f]

    l2_target = XA - XB

    mu = tl.sum(Z2, 1) / HF
    var = tl.sum((Z2 - mu) * (Z2 - mu), 1) / HF
    std = tl.sqrt(var + 1e-6)
    Z2_hat = (Z2 - mu) / std  # [1,f]
    LN_out = ln_weight * Z2_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    dl_dLN_out = LN_out - l2_target  # [1,f]
    dl_dZ2_hat = dl_dLN_out * ln_weight  # [1,f]

    dl_dZ2_term_1 = HF * dl_dZ2_hat
    dl_dZ2_term_2 = tl.sum(dl_dZ2_hat, 1)
    dl_dZ2_term_3 = Z2_hat * tl.sum(dl_dZ2_hat * Z2_hat, 1)
    dl_dZ2_sum = dl_dZ2_term_1 - dl_dZ2_term_2 - dl_dZ2_term_3
    dl_dZ2 = dl_dZ2_sum / (std * HF * 100)
    dl_dZ2 = dl_dZ2 * 100.

    dl_dZ1 = tl.sum(tl.trans(dl_dZ2) * tl.trans(W2), axis=0)[None,:] * diff_gelu_tl(Z1)  # [1,f] @ [4f,f].t

    ilr_mul_dl_dZ2 = ilr_gated * dl_dZ2  # [K=1,1] * [K=1,f]
    ilr_mul_dl_dZ1 = ilr_gated * dl_dZ1  # [K=1,1] * [K=1,f]

    ##
    W1_grad += tl.trans(XB) * ilr_mul_dl_dZ1
    b1_grad += ilr_mul_dl_dZ1
    tl.store(_W1_grad, W1_grad.to(W_dtype))
    tl.store(_b1_grad, b1_grad.to(W_dtype))
    W1_bar = W1 - token_idx * W1_grad
    b1_bar = b1 - token_idx * b1_grad
    Z1_bar = tl.sum(tl.trans(XC) * W1_bar, axis=0)[None,:] + b1_bar

    X2_bar = gelu(Z1_bar)

    ##
    W2_grad += tl.trans(X2) * ilr_mul_dl_dZ2
    b2_grad += ilr_mul_dl_dZ2
    tl.store(_W2_grad, W2_grad.to(W_dtype))
    tl.store(_b2_grad, b2_grad.to(W_dtype))
    W2_bar = W2 - token_idx * W2_grad
    b2_bar = b2 - token_idx * b2_grad
    Z2_bar = tl.sum(tl.trans(X2_bar) * W2_bar, axis=0)[None, :] + b2_bar

    ## residual + postln
    mu_bar = tl.sum(Z2_bar, 1) / HF
    var_bar = tl.sum((Z2_bar - mu_bar) * (Z2_bar - mu_bar), 1) / HF
    std_bar = tl.sqrt(var_bar + 1e-6)
    Z2_bar_hat = (Z2_bar - mu_bar) / std_bar  # [1,f]
    LN_out_bar = ln_weight * Z2_bar_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    Z2_bar = XC + LN_out_bar

    tl.store(_Out, Z2_bar.to(O_dtype))

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['HF', 'HF_prime'],
    restore_value=['__W1', '__b1', '__W1_grad', '__b1_grad',
                   '__W2', '__b2', '__W2_grad', '__b2_grad']
)
@triton.jit
def _m2_decode_end_chunk_kernel(__W1, __W1_grad, __b1, __b1_grad,
                                __W2, __W2_grad, __b2, __b2_grad,
                                __XA, __XB, __XC,
                                __ln_weight, __ln_bias,
                                __ilr_gated, __token_idx, __Out,
                                stride_w1_batch, stride_w1_head, stride_w1_fin,
                                stride_b1_batch, stride_b1_head, stride_b1_f,
                                stride_w2_batch, stride_w2_head, stride_w2_fin,
                                stride_b2_batch, stride_b2_head, stride_b2_f,
                                stride_x_batch, stride_x_head, stride_x_n,
                                stride_ln_head, stride_ln_f,
                                stride_ilr_batch, stride_ilr_head,
                                CS: tl.constexpr, HF: tl.constexpr, HF_prime: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)
    rf_prime = tl.arange(0, HF_prime)

    W_dtype = __W1.type.element_ty
    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch + head * stride_x_head
    w1_block_offset = batch * stride_w1_batch + head * stride_w1_head
    b1_block_offset = batch * stride_b1_batch + head * stride_b1_head
    w2_block_offset = batch * stride_w2_batch + head * stride_w2_head
    b2_block_offset = batch * stride_b2_batch + head * stride_b2_head
    ln_block_offset = head * stride_ln_head
    ilr_block_offset = batch * stride_ilr_batch + head * stride_ilr_head

    x_inner_offset = rc[:, None] * stride_x_n + rf[None, :]
    w1_inner_offset = rf[:, None] * stride_w1_fin + rf_prime[None, :]
    b1_inner_offset = rc[:, None] * stride_b1_f + rf_prime[None, :]
    w2_inner_offset = rf_prime[:, None] * stride_w2_fin + rf[None, :]
    b2_inner_offset = rc[:, None] * stride_b2_f + rf[None, :]
    ln_inner_offset = rc[:, None] * stride_ln_f + rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    w1_offset = w1_block_offset + w1_inner_offset
    b1_offset = b1_block_offset + b1_inner_offset
    w2_offset = w2_block_offset + w2_inner_offset
    b2_offset = b2_block_offset + b2_inner_offset
    ln_offset = ln_block_offset + ln_inner_offset
    ilr_offset = ilr_block_offset

    _XA = __XA + x_offset
    _XB = __XB + x_offset
    _XC = __XC + x_offset
    _Out = __Out + x_offset
    _W1 = __W1 + w1_offset
    _W1_grad = __W1_grad + w1_offset
    _b1 = __b1 + b1_offset
    _b1_grad = __b1_grad + b1_offset
    _W2 = __W2 + w2_offset
    _W2_grad = __W2_grad + w2_offset
    _b2 = __b2 + b2_offset
    _b2_grad = __b2_grad + b2_offset
    _ln_weight = __ln_weight + ln_offset
    _ln_bias = __ln_bias + ln_offset
    _ilr_gated = __ilr_gated + ilr_offset
    _token_idx = __token_idx

    XA = tl.load(_XA)
    XB = tl.load(_XB)
    XC = tl.load(_XC)
    token_idx = tl.load(_token_idx)
    ilr_gated = tl.load(_ilr_gated)
    W1 = tl.load(_W1)
    W1_grad = tl.load(_W1_grad)
    b1 = tl.load(_b1)
    b1_grad = tl.load(_b1_grad)
    W2 = tl.load(_W2)
    W2_grad = tl.load(_W2_grad)
    b2 = tl.load(_b2)
    b2_grad = tl.load(_b2_grad)
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    Z1 = tl.sum(tl.trans(XB) * W1, axis=0)[None,:] + b1  # [1,f] @ [f,4f] + [1,4f]
    X2 = gelu(Z1)
    Z2 = tl.sum(tl.trans(X2) * W2, axis=0)[None,:] + b2  # [1,4f] @ [4f,f] + [1,f]

    l2_target = XA - XB

    mu = tl.sum(Z2, 1) / HF
    var = tl.sum((Z2 - mu) * (Z2 - mu), 1) / HF
    std = tl.sqrt(var + 1e-6)
    Z2_hat = (Z2 - mu) / std  # [1,f]
    LN_out = ln_weight * Z2_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    dl_dLN_out = LN_out - l2_target  # [1,f]
    dl_dZ2_hat = dl_dLN_out * ln_weight  # [1,f]

    dl_dZ2_term_1 = HF * dl_dZ2_hat
    dl_dZ2_term_2 = tl.sum(dl_dZ2_hat, 1)
    dl_dZ2_term_3 = Z2_hat * tl.sum(dl_dZ2_hat * Z2_hat, 1)
    dl_dZ2_sum = dl_dZ2_term_1 - dl_dZ2_term_2 - dl_dZ2_term_3
    dl_dZ2 = dl_dZ2_sum / (std * HF * 100)
    dl_dZ2 = dl_dZ2 * 100.

    dl_dZ1 = tl.sum(tl.trans(dl_dZ2) * tl.trans(W2), axis=0)[None,:] * diff_gelu_tl(Z1)  # [1,f] @ [4f,f].t

    ilr_mul_dl_dZ2 = ilr_gated * dl_dZ2  # [K=1,1] * [K=1,f]
    ilr_mul_dl_dZ1 = ilr_gated * dl_dZ1  # [K=1,1] * [K=1,f]

    ##
    W1_grad += tl.trans(XB) * ilr_mul_dl_dZ1
    b1_grad += ilr_mul_dl_dZ1
    tl.store(_W1_grad, tl.zeros_like(W1_grad).to(W_dtype))
    tl.store(_b1_grad, tl.zeros_like(b1_grad).to(W_dtype))
    W1_bar = W1 - token_idx * W1_grad
    b1_bar = b1 - token_idx * b1_grad
    tl.store(_W1, W1_bar.to(W_dtype))
    tl.store(_b1, b1_bar.to(W_dtype))
    Z1_bar = tl.sum(tl.trans(XC) * W1_bar, axis=0)[None,:] + b1_bar

    X2_bar = gelu(Z1_bar)

    ##
    W2_grad += tl.trans(X2) * ilr_mul_dl_dZ2
    b2_grad += ilr_mul_dl_dZ2
    tl.store(_W2_grad, tl.zeros_like(W2_grad).to(W_dtype))
    tl.store(_b2_grad, tl.zeros_like(b2_grad).to(W_dtype))
    W2_bar = W2 - token_idx * W2_grad
    b2_bar = b2 - token_idx * b2_grad
    tl.store(_W2, W2_bar.to(W_dtype))
    tl.store(_b2, b2_bar.to(W_dtype))
    Z2_bar = tl.sum(tl.trans(X2_bar) * W2_bar, axis=0)[None, :] + b2_bar

    ## residual + postln
    mu_bar = tl.sum(Z2_bar, 1) / HF
    var_bar = tl.sum((Z2_bar - mu_bar) * (Z2_bar - mu_bar), 1) / HF
    std_bar = tl.sqrt(var_bar + 1e-6)
    Z2_bar_hat = (Z2_bar - mu_bar) / std_bar  # [1,f]
    LN_out_bar = ln_weight * Z2_bar_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    Z2_bar = XC + LN_out_bar

    tl.store(_Out, Z2_bar.to(O_dtype))


class TttM2BMMTKModule(TttBaseModule):

    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 1, 4 * self.head_dim)))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 1, self.head_dim)))

        if self.config.use_compile:
            self.decode_end_chunk = torch.compile(m2_decode_end_chunk)
            self.decode = torch.compile(m2_decode)
        else:
            self.decode_end_chunk = m2_decode_end_chunk
            self.decode = m2_decode

    def process_inner_loop(self, inputs, inner_chunk_size, last_chunk_params_dic,
                           is_prefill=False,
                           is_last_in_chunk=False,
                           cache_params=None):

        B_mul_NH, N, HF = inputs['XA'].shape  # [B*nh,N,f]
        HF_prime = 4 * HF
        B = B_mul_NH // self.num_heads
        NH = self.num_heads
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
        XC_residual = inputs['XC']

        if is_prefill:
            HF_prime = 4 * HF
            W1_init = cache_params.params_dict["W1_states"][self.layer_idx].reshape(B, NH, HF, HF_prime)
            b1_init = cache_params.params_dict["b1_states"][self.layer_idx].reshape(B, NH, 1, HF_prime)
            W2_init = cache_params.params_dict["W2_states"][self.layer_idx].reshape(B, NH, HF_prime, HF)
            b2_init = cache_params.params_dict["b2_states"][self.layer_idx].reshape(B, NH, 1, HF)

            CS = inner_chunk_size
            NC = N // CS
            inputs = tree_map(lambda x: x.reshape(B, NH, NC, CS, -1).contiguous(), inputs)  # [B*nh,N,f/1] -> [B,nh,nc,cs,f/1]
            ilr_gated = inputs.pop('ilr_gated').transpose(-1, -2)  # [B,nh,nc,1,cs]
            inputs['coeff'] = self.token_idx[None, :] * ilr_gated  # [1,1,1,cs,1] * [B,nh,nc,1,cs] -> [B,nh,nc,CS,CS]

            XA, XB, XC, coeff = inputs['XA'], inputs['XB'], inputs['XC'], inputs['coeff']  # [B,nh,nc,cs,f/1]
            input_device = XA.device
            input_dtype = XA.dtype
            output = torch.empty_like(XA)

            ln_weight = self.ln_weight.squeeze(0).expand(-1, CS, -1).contiguous()
            ln_bias = self.ln_bias.squeeze(0).expand(-1, CS, -1).contiguous()
            b1_init = b1_init.expand(-1, -1, CS, -1).contiguous()
            b2_init = b2_init.expand(-1, -1, CS, -1).contiguous()
            cumsum_matrix = torch.tril(torch.ones(CS, CS, dtype=input_dtype, device=input_device))
            make_last_b_matrix = torch.zeros(CS, CS, dtype=input_dtype, device=input_device)
            make_last_coeff_1_matrix = torch.zeros(CS, HF, dtype=input_dtype, device=input_device)
            make_last_coeff_2_matrix = torch.zeros(CS, HF_prime, dtype=input_dtype, device=input_device)
            make_last_b_matrix[:, -1] = 1.
            make_last_coeff_1_matrix[-1, :] = 1.
            make_last_coeff_2_matrix[-1, :] = 1.

            # tk_m2_prefill.prefill_whole_loop_gelu_coeff_bias_LN_fp16(W1_init, W2_init, b1_init, b2_init,
            #                                                          ln_weight, ln_bias,
            #                                                          cumsum_matrix,
            #                                                          make_last_b_matrix,
            #                                                          make_last_coeff_1_matrix, make_last_coeff_2_matrix,
            #                                                          XA, XB, XC, coeff,
            #                                                          output)
            tk_m2_prefill.prefill_whole_loop_gelu_coeff_bias_LN_res_PLN_fp16(W1_init, W2_init, b1_init, b2_init,
                                                                     ln_weight, ln_bias,
                                                                     cumsum_matrix,
                                                                     make_last_b_matrix,
                                                                     make_last_coeff_1_matrix, make_last_coeff_2_matrix,
                                                                     XA, XB, XC, coeff,
                                                                     output)
            b1_init = b1_init[:, :, -1:, :].reshape(B_mul_NH, 1, -1)
            b2_init = b2_init[:, :, -1:, :].reshape(B_mul_NH, 1, -1)
            cache_params.params_dict["b1_states"][self.layer_idx].copy_(b1_init)
            cache_params.params_dict["b2_states"][self.layer_idx].copy_(b2_init)

            output = output.reshape(B_mul_NH, N, HF)
            # output = self.residual_add_post_LN(XC_residual, output)  # residual + postLN

        else:
            W1 = cache_params.params_dict["W1_states"][self.layer_idx].reshape(B, NH, HF, HF_prime)
            b1 = cache_params.params_dict["b1_states"][self.layer_idx].reshape(B, NH, 1, HF_prime)
            W1_grad = cache_params.params_dict["W1_grad"][self.layer_idx].reshape(B, NH, HF, HF_prime)
            b1_grad = cache_params.params_dict["b1_grad"][self.layer_idx].reshape(B, NH, 1, HF_prime)

            W2 = cache_params.params_dict["W2_states"][self.layer_idx].reshape(B, NH, HF_prime, HF)
            b2 = cache_params.params_dict["b2_states"][self.layer_idx].reshape(B, NH, 1, HF)
            W2_grad = cache_params.params_dict["W2_grad"][self.layer_idx].reshape(B, NH, HF_prime, HF)
            b2_grad = cache_params.params_dict["b2_grad"][self.layer_idx].reshape(B, NH, 1, HF)

            token_idx = inputs.pop('token_idx')  # [1,1,1]

            # @xinhao: decoding from a prompt of length 1 will always have `inner_chunk_size=remainder=1`
            inputs = tree_map(lambda x: x.reshape(B, NH, N, -1), inputs)  # [B*nh,N=1,f], [B*nh,N=1,1] -> [BS,nh,N=1,f/1]
            XA, XB, XC, ilr_gated = inputs['XA'], inputs['XB'], inputs['XC'], inputs['ilr_gated']  # [B,nh,N=1,f/1]

            output = torch.empty_like(XA)  # [B,nh,N,f]
            grid = (B, NH, 1)
            CS = 1

            if is_last_in_chunk:
                _m2_decode_end_chunk_kernel[grid](W1, W1_grad, b1, b1_grad,
                                                  W2, W2_grad, b2, b2_grad,
                                                  XA, XB, XC,
                                                  self.ln_weight, self.ln_bias,
                                                  ilr_gated, token_idx, output,
                                                  W1.stride(0), W1.stride(1), W1.stride(2),
                                                  b1.stride(0), b1.stride(1), b1.stride(2),
                                                  W2.stride(0), W2.stride(1), W2.stride(2),
                                                  b2.stride(0), b2.stride(1), b2.stride(2),
                                                  XA.stride(0), XA.stride(1), XA.stride(2),
                                                  self.ln_weight.stride(1), self.ln_weight.stride(2),
                                                  ilr_gated.stride(0), ilr_gated.stride(1),
                                                  CS, HF, HF_prime)
            else:
                _m2_decode_kernel[grid](W1, W1_grad, b1, b1_grad,
                                        W2, W2_grad, b2, b2_grad,
                                        XA, XB, XC,
                                        self.ln_weight, self.ln_bias,
                                        ilr_gated, token_idx, output,
                                        W1.stride(0), W1.stride(1), W1.stride(2),
                                        b1.stride(0), b1.stride(1), b1.stride(2),
                                        W2.stride(0), W2.stride(1), W2.stride(2),
                                        b2.stride(0), b2.stride(1), b2.stride(2),
                                        XA.stride(0), XA.stride(1), XA.stride(2),
                                        self.ln_weight.stride(1), self.ln_weight.stride(2),
                                        ilr_gated.stride(0), ilr_gated.stride(1),
                                        CS, HF, HF_prime)

            output = output.reshape(B_mul_NH, N, HF)

        return output, None

##########################################


class TttDecoderLayer(nn.Module):
    def __init__(self, config: TttConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # self.self_attn = TttM1Module(config=config, layer_idx=layer_idx)  # @xinhao: M1 vmap module
        if config.inner_net == 'mlp_1_dual':
            self.self_attn = TttM1BMMModule(config=config, layer_idx=layer_idx)
        elif config.inner_net == 'mlp_1_dual_tk':
            self.self_attn = TttM1BMMTKModule(config=config, layer_idx=layer_idx)
        elif config.inner_net == 'mlp_2_dual':
            self.self_attn = TttM2BMMModule(config=config, layer_idx=layer_idx)
        elif config.inner_net == 'mlp_2_dual_tk':
            self.self_attn = TttM2BMMTKModule(config=config, layer_idx=layer_idx)
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

        # if not self.fused_add_norm:
        #     residual = (hidden_states + residual) if residual is not None else hidden_states
        #     hidden_states = self.input_layernorm(residual.to(dtype=self.input_layernorm.weight.dtype))
        #     if self.residual_in_fp32:
        #         residual = residual.to(torch.float32)
        # else:
        #     fused_add_norm_fn = rms_norm_fn
        #     hidden_states, residual = fused_add_norm_fn(
        #         hidden_states,
        #         self.input_layernorm.weight,
        #         self.input_layernorm.bias,
        #         residual=residual,
        #         prenorm=True,
        #         residual_in_fp32=self.residual_in_fp32,
        #         eps=self.input_layernorm.eps,
        #     )
        # TTT
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
            is_prefill=is_prefill,
            is_last_in_chunk=is_last_in_chunk,
        )
        # hidden_states = (residual + hidden_states).to(hidden_states.dtype)  # @xinhao: rms and add are fused in the next step

        # if not self.fused_add_norm:
        #     residual = (hidden_states + residual) if residual is not None else hidden_states
        #     hidden_states = self.post_attention_layernorm(residual.to(dtype=self.post_attention_layernorm.weight.dtype))
        #     if self.residual_in_fp32:
        #         residual = residual.to(torch.float32)
        # else:
        #     fused_add_norm_fn = rms_norm_fn
        #     hidden_states, residual = fused_add_norm_fn(
        #         hidden_states,
        #         self.post_attention_layernorm.weight,
        #         self.post_attention_layernorm.bias,
        #         residual=residual,
        #         prenorm=True,
        #         residual_in_fp32=self.residual_in_fp32,
        #         eps=self.post_attention_layernorm.eps,
        #     )

        # Fully Connected
        # hidden_states = self.mlp_forward(hidden_states)

        return hidden_states, residual


@dataclass
class TttOutput(ModelOutput):

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[TttCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TttCausalLMOutput(ModelOutput):

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
        # if not self.fused_add_norm:
        #     residual = (hidden_states + residual) if residual is not None else hidden_states
        #     hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        # else:
        #     # Set prenorm=False here since we don't need the residual
        #     fused_add_norm_fn = rms_norm_fn
        #     hidden_states = fused_add_norm_fn(
        #         hidden_states,
        #         self.norm_f.weight,
        #         self.norm_f.bias,
        #         eps=self.norm_f.eps,
        #         residual=residual,
        #         prenorm=False,
        #         residual_in_fp32=self.residual_in_fp32,
        #     )

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
        self.get_output_logits = self._get_output_logits

        # Initialize weights and apply final processing
        # self.post_init()
        self.config = config

    def _get_output_logits(self, hidden_states):
        # logits = self.lm_head(hidden_states)
        logits = torch.zeros((hidden_states.shape[0], 1, self.vocab_size),
                             device=hidden_states.device, dtype=hidden_states.dtype)
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
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            is_prefill=is_prefill,
            is_last_in_chunk=is_last_in_chunk,
        )

        # hidden_states = outputs[0][:,-1:,:]  # [BS,N,F] -> [BS,1,F] to avoid OOM when prefilling
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.get_output_logits(hidden_states)
            # logits = self.lm_head(hidden_states)
        # logits = logits.float()

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