import copy

import torch
import torch.nn as nn
import einops
import triton
import triton.language as tl
import os
from transformers.models.ttt_full_prefill_decode_optimize.micro_decode_modules_PT import ttt_m1_decode


########## Triton ##########
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=2),
    ],
    key=['HF'],
    restore_value=['W1_init', 'W1_grad', 'b1_init', 'b1_grad'],
)
@triton.jit
def _m1_decode_kernel(
        W1_init, W1_grad, b1_init, b1_grad,
        XA, XB, XC, coeff, Out,
        ln_weight, ln_bias,

        stride_w_batch, stride_w_head, stride_w_in, stride_w_out,
        stride_b_batch, stride_b_head, stride_b_in, stride_b_out,
        stride_ln_head, stride_ln_f,
        stride_x_batch, stride_x_head, stride_x_n, stride_x_f,
        stride_coeff_batch, stride_coeff_head,
        CS: tl.constexpr, HF: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)

    W_dtype = W1_init.type.element_ty
    O_dtype = Out.type.element_ty

    abco_offset = batch * stride_x_batch + head * stride_x_head
    coeff_offset = batch * stride_coeff_batch + head * stride_coeff_head

    w_offset = batch * stride_w_batch + head * stride_w_head
    b_offset = batch * stride_b_batch + head * stride_b_head
    ln_offset = head * stride_ln_head

    XA = XA + abco_offset + (rc[:, None] * stride_x_n + rf[None, :] * stride_x_f)
    XB = XB + abco_offset + (rc[:, None] * stride_x_n + rf[None, :] * stride_x_f)
    XC = XC + abco_offset + (rc[:, None] * stride_x_n + rf[None, :] * stride_x_f)
    Out = Out + abco_offset + (rf[None, :] * stride_x_f)
    coeff = coeff + coeff_offset

    W1_init = W1_init + w_offset + (rf[:, None] * stride_w_in + rf[None, :] * stride_w_out)  # [f,f]
    W1_grad = W1_grad + w_offset + (rf[:, None] * stride_w_in + rf[None, :] * stride_w_out)

    b1_init = b1_init + b_offset + rf[None, :] * stride_b_out  # [1,f]
    b1_grad = b1_grad + b_offset + rf[None, :] * stride_b_out

    ln_weight = ln_weight + ln_offset + rf[None, :] * stride_ln_f  # [1,f]
    ln_bias = ln_bias + ln_offset + rf[None, :] * stride_ln_f

    XA_data = tl.load(XA)
    XB_data = tl.load(XB)
    XC_data = tl.load(XC)
    coeff_data = tl.load(coeff)
    W1_init_data = tl.load(W1_init)
    W1_grad_data = tl.load(W1_grad)
    b1_init_data = tl.load(b1_init)
    b1_grad_data = tl.load(b1_grad)
    ln_weight_data = tl.load(ln_weight)
    ln_bias_data = tl.load(ln_bias)

    Z1 = tl.sum(tl.trans(XB_data) * W1_init_data, 0) + b1_init_data  # [1,HF]: same as PT
    reconstruction_target = XA_data - XB_data  # [1,HF]

    mu = tl.sum(Z1) / HF  # [1,]
    var = tl.sum((Z1 - mu) * (Z1 - mu)) / HF  # [1,]
    std = tl.sqrt(var + 1e-6)
    # std = 1.
    x_hat = (Z1 - mu) / std  # [1,f], err from PT: 4e-3
    y = ln_weight_data * x_hat + ln_bias_data  # [1,f], err: 4.54

    grad_output = y - reconstruction_target  # [1,f]
    grad_x_hat = ln_weight_data * grad_output  # [1,f]
    grad_l_Z1 = (
            (1.0 / HF)
            * (
                HF * grad_x_hat  # [1,f]
                - tl.sum(grad_x_hat)  #[1,]
                - x_hat * tl.sum(grad_x_hat * x_hat)  # [1,f] * [1,]
            )
            / std
    )  # grad_l_Z1: [1,f]

    W1_grad_data += (tl.trans(XB_data) * grad_l_Z1)  # @xinhao: tl.dot(tl.trans(XB_chunk), Z1) doesn't support [HF,1] @ [1,HF]
    b1_grad_data += grad_l_Z1
    W1_init_data -= coeff_data * W1_grad_data
    b1_init_data -= coeff_data * b1_grad_data
    Z1_bar = tl.sum(tl.trans(XC_data) * W1_init_data, 0) + b1_init_data  # Z1_bar

    # Post LN
    mu = tl.sum(Z1_bar) / HF  # [1,]
    var = tl.sum((Z1_bar - mu) * (Z1_bar - mu)) / HF  # [1,]
    std = tl.sqrt(var + 1e-6)
    # std = 1.
    x_hat_bar = (Z1_bar - mu) / std  # [1,f]
    Z1_bar = XC_data + (ln_weight_data * x_hat_bar + ln_bias_data)  # XC + LN(Z1_bar): [1,f]

    # tl.store(Out, x_hat.to(O_dtype))
    # tl.store(Out, y.to(O_dtype))
    tl.store(Out, Z1_bar.to(O_dtype))

    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))
    tl.store(b1_init, b1_init_data.to(W_dtype))
    tl.store(b1_grad, b1_grad_data.to(W_dtype))


def ttt_m1_triton_decode(
        XA, XB, XC, coeff,
        W1_init, W1_grad, b1_init, b1_grad,
        ln_weight, ln_bias
):
    B, NH, CS, HF = XA.shape
    assert CS == 1
    assert XA.shape == XB.shape == XC.shape
    assert coeff.shape == (B, NH, CS, 1)

    output = torch.empty_like(XA)
    grid = (B, NH, 1)

    _m1_decode_kernel[grid](W1_init, W1_grad, b1_init, b1_grad,
                            XA, XB, XC, coeff, output,
                            ln_weight, ln_bias,

                            W1_init.stride(0), W1_init.stride(1), W1_init.stride(2), W1_init.stride(3),
                            b1_init.stride(0), b1_init.stride(1), b1_init.stride(2), b1_init.stride(3),
                            ln_weight.stride(1), ln_weight.stride(2),

                            XA.stride(0), XA.stride(1), XA.stride(2), XA.stride(3),
                            coeff.stride(0), coeff.stride(1),

                            CS, HF)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    input_dtype = torch.float16
    # input_dtype = torch.float64

    BS, NH, CS, HF = 64, 32, 1, 64

    original_state_dict = {
        'W1': torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02,
        'W1_grad': torch.zeros(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02,
        'b1': torch.randn(BS, NH, 1, HF, device='cuda', dtype=input_dtype) * 0.02,
        'b1_grad': torch.zeros(BS, NH, 1, HF, device='cuda', dtype=input_dtype) * 0.02,
        'ln_weight': torch.randn(1, NH, 1, HF, device='cuda', dtype=input_dtype) * 0.02,
        'ln_bias': torch.randn(1, NH, 1, HF, device='cuda', dtype=input_dtype) * 0.02,
    }
    original_input_dict = {
        'XA': torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'XB': torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'XC': torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'coeff': torch.randn(BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02,
    }

    pt_state_dict = copy.deepcopy(original_state_dict)
    pt_input_dict = copy.deepcopy(original_input_dict)
    XCW_batch_pt = ttt_m1_decode(pt_input_dict['XA'], pt_input_dict['XB'],
                                pt_input_dict['XC'], pt_input_dict['coeff'],
                                pt_state_dict['W1'], pt_state_dict['W1_grad'],
                                pt_state_dict['b1'], pt_state_dict['b1_grad'],
                                pt_state_dict['ln_weight'], pt_state_dict['ln_bias'])
    # print(XCW_batch_pt.shape)

    triton_state_dict = copy.deepcopy(original_state_dict)
    triton_input_dict = copy.deepcopy(original_input_dict)
    XCW_batch_triton = ttt_m1_triton_decode(triton_input_dict['XA'], triton_input_dict['XB'],
                                        triton_input_dict['XC'], triton_input_dict['coeff'],
                                        triton_state_dict['W1'], triton_state_dict['W1_grad'],
                                        triton_state_dict['b1'], triton_state_dict['b1_grad'],
                                        triton_state_dict['ln_weight'], triton_state_dict['ln_bias'])
    # print(XCW_batch_triton.shape)

    print('=== PyTorch v.s Triton ===')
    for k in original_state_dict.keys():
        diff = pt_state_dict[k] - triton_state_dict[k]
        print(f'{k} diff: max={diff.max():.5f}, median={diff.median():.5f}')

    diff = torch.abs(XCW_batch_pt - XCW_batch_triton)
    print(f'Output diff: max={diff.max():.5f}, median={diff.median():.5f}')


