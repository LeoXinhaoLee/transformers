import pdb

import torch
import einops
import triton
import triton.language as tl
import os


########## Pytorch ##########

###
## M1
###
def ttt_m1_decode(XA, XB, XC, coeff, W1_init, W1_grad):
    Z1 = (XB @ W1_init).sub_(XA)
    W1_grad.add_(XB.transpose(-1, -2) @ Z1)
    W1_init.sub_(coeff * W1_grad)
    Z1 = XC @ W1_init
    return W1_init, W1_grad, Z1


########## Triton ##########

###
## M1
###
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
    restore_value=['W1_init', 'W1_grad'],
)
@triton.jit
def _m1_decode_kernel(W1_init, W1_grad,
                      XA, XB, XC, coeff, Out,
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

    # '''
    Z1 = tl.sum(tl.trans(XB_chunk) * W1_init_data, 0) - XA_chunk # [HF,] - [1,HF] - > [1,HF]
    W1_grad_data += tl.trans(XB_chunk) * Z1  # @xinhao: tl.dot(tl.trans(XB_chunk), Z1) doesn't support [HF,1] @ [1,HF]
    W1_init_data -= coeff_chunk * W1_grad_data
    Z1_bar = tl.sum(tl.trans(XC_chunk) * W1_init_data, 0)
    tl.store(Out_chunk, Z1_bar.to(O_dtype))
    # '''

    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))

def ttt_m1_triton_decode(XA, XB, XC, coeff, W1_init, W1_grad):
    B, NH, CS, HF = XA.shape
    assert CS == 1
    assert W1_init.shape == W1_grad.shape == (B, NH, HF, HF)
    assert XA.shape == XB.shape == XC.shape
    assert coeff.shape == (B, NH, CS, 1)
    output = torch.empty(size=(B, NH, CS, HF), device=W1_init.device, dtype=torch.float16)  # TODO FIX DTYPE
    grid = (B, NH, 1)
    _m1_decode_kernel[grid](W1_init, W1_grad, XA, XB, XC, coeff, output,
                            NH * HF * HF, HF * HF, HF, 1,  # strides for W
                            NH * CS * HF, CS * HF, HF, 1,  # strides for ABCO, output
                            NH * CS, CS, 1, 1,  # strides for coeff
                            CS, HF)
    return W1_init, W1_grad, output


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_stages=7, num_warps=8),
#         triton.Config({}, num_stages=6, num_warps=8),
#         triton.Config({}, num_stages=5, num_warps=8),
#         triton.Config({}, num_stages=4, num_warps=8),
#         triton.Config({}, num_stages=3, num_warps=8),
#         triton.Config({}, num_stages=3, num_warps=4),
#         triton.Config({}, num_stages=4, num_warps=4),
#         triton.Config({}, num_stages=6, num_warps=4),
#     ],
#     key=['HF', 'HF_shard'],
#     restore_value=['W1_init', 'W1_grad'],
# )
# @triton.jit
# def _m1_sharded_decode_kernel(W1_init, W1_grad,
#                               XA, XB, XC, coeff, Out,
#                               stride_w_batch, stride_w_head, stride_w_shard, stride_w_in, stride_w_out,
#
#                               stride_a_batch, stride_a_head, stride_a_shard, stride_a_n, stride_a_f,  # XA, output
#
#                               stride_b_batch, stride_b_head, stride_b_n, stride_b_f,  # XB, XC
#
#                               stride_coeff_batch, stride_coeff_head, stride_coeff_n, stride_coeff_f,
#
#                               CS: tl.constexpr, HF: tl.constexpr, HF_shard: tl.constexpr):
#     batch = tl.program_id(0)
#     head = tl.program_id(1)
#     shard = tl.program_id(2)
#
#     rc = tl.arange(0, CS)
#     rf = tl.arange(0, HF)
#     rf_shard = tl.arange(0, HF_shard)
#
#     W_dtype = W1_init.type.element_ty
#     O_dtype = Out.type.element_ty
#
#     w_offset = batch * stride_w_batch + head * stride_w_head + shard * stride_w_shard
#     ao_offset = batch * stride_a_batch + head * stride_a_head + shard * stride_a_shard
#     bc_offset = batch * stride_b_batch + head * stride_b_head
#     coeff_offset = batch * stride_coeff_batch + head * stride_coeff_head
#
#     coeff = coeff + coeff_offset
#     XB = XB + bc_offset + (rc[:, None] * stride_b_n + rf[None, :] * stride_b_f)
#     XC = XC + bc_offset + (rc[:, None] * stride_b_n + rf[None, :] * stride_b_f)
#
#     XA = XA + ao_offset + (rc[:, None] * stride_a_n + rf_shard[None, :] * stride_a_f)
#     Out = Out + ao_offset + (rc[:, None] * stride_a_n + rf_shard[None, :] * stride_a_f)
#
#     W1_init = W1_init + w_offset + (rf[:, None] * stride_w_in + rf_shard[None, :] * stride_w_out)
#     W1_grad = W1_grad + w_offset + (rf[:, None] * stride_w_in + rf_shard[None, :] * stride_w_out)
#
#     XA_chunk = tl.load(XA)
#     XB_chunk = tl.load(XB)
#     XC_chunk = tl.load(XC)
#     coeff_chunk = tl.load(coeff)
#     W1_init_data = tl.load(W1_init)
#     W1_grad_data = tl.load(W1_grad)
#
#     # '''
#     Z1 = tl.sum(tl.trans(XB_chunk) * W1_init_data, 0) - XA_chunk # [HF,] - [1,HF] - > [1,HF]
#     W1_grad_data += tl.trans(XB_chunk) * Z1  # @xinhao: tl.dot(tl.trans(XB_chunk), Z1) doesn't support [HF,1] @ [1,HF]
#     W1_init_data -= coeff_chunk * W1_grad_data
#     Z1_bar = tl.sum(tl.trans(XC_chunk) * W1_init_data, 0)[None,:]
#     tl.store(Out, Z1_bar.to(O_dtype))
#     # '''
#
#     tl.store(W1_init, W1_init_data.to(W_dtype))
#     tl.store(W1_grad, W1_grad_data.to(W_dtype))
#
#
# def ttt_m1_triton_sharded_decode(XA, XB, XC, coeff, W1_init, W1_grad):
#     B, NH, M, CS, HF_div_M = XA.shape
#     HF = XB.shape[-1]
#     assert CS == 1
#     assert W1_init.shape == W1_grad.shape == (B, NH, M, HF, HF_div_M)
#     assert XB.shape == XC.shape == (B, NH, CS, HF)
#     assert coeff.shape == (B, NH, CS, 1)
#
#     # M = 1
#     # W1_init = W1_init.reshape(B, NH, HF, M, HF // M).permute(0, 1, 3, 2, 4)  # [BS,NH,M,HF,HF//M]
#     # W1_grad = W1_grad.reshape(B, NH, HF, M, HF // M).permute(0, 1, 3, 2, 4)
#     # XA = XA.reshape(B, NH, CS, M, HF // M).permute(0, 1, 3, 2, 4)  # [BS,NH,M,CS,HF//M]
#     output = torch.empty(size=(B, NH, M, CS, HF_div_M), device=W1_init.device, dtype=torch.float16)
#     grid = (B, NH, M)
#
#     _m1_sharded_decode_kernel[grid](W1_init, W1_grad, XA, XB, XC, coeff, output,
#
#                                     W1_init.stride(0), W1_init.stride(1), W1_init.stride(2), W1_init.stride(3), W1_init.stride(4),  # strides for W
#
#                                     XA.stride(0), XA.stride(1), XA.stride(2), XA.stride(3), XA.stride(4),  # strides for XA, output
#
#                                     XB.stride(0), XB.stride(1), XB.stride(2), XB.stride(3),  # strides for XB, XC
#
#                                     coeff.stride(0), coeff.stride(1), coeff.stride(2), coeff.stride(3),  # strides for coeff
#
#                                     CS, HF, HF // M)
#
#     # output = output.permute(0, 1, 3, 2, 4).reshape(B, NH, CS, HF)
#     # W1_init = W1_init.permute(0, 1, 3, 2, 4).reshape(B, NH, HF, HF)
#     # W1_grad = W1_grad.permute(0, 1, 3, 2, 4).reshape(B, NH, HF, HF)
#     return W1_init, W1_grad, output
#
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_stages=7, num_warps=8),
#         triton.Config({}, num_stages=6, num_warps=8),
#         triton.Config({}, num_stages=5, num_warps=8),
#         triton.Config({}, num_stages=4, num_warps=8),
#         triton.Config({}, num_stages=3, num_warps=8),
#         triton.Config({}, num_stages=3, num_warps=4),
#         triton.Config({}, num_stages=4, num_warps=4),
#         triton.Config({}, num_stages=6, num_warps=4),
#     ],
#     key=['HF', 'HF_shard'],
#     restore_value=['W1_init', 'W1_grad'],
# )
# @triton.jit
# def _m1_sharded_decode_kernel(W1_init, W1_grad,
#                               XA, XB, XC, coeff, Out,
#                               stride_w_batch_head, stride_w_shard, stride_w_in, stride_w_out,
#
#                               stride_a_batch_head, stride_a_shard, stride_a_n, stride_a_f,  # XA, output
#
#                               stride_b_batch_head, stride_b_n, stride_b_f,  # XB, XC
#
#                               stride_coeff_batch_head, stride_coeff_n, stride_coeff_f,
#
#                               CS: tl.constexpr, HF: tl.constexpr, HF_shard: tl.constexpr):
#     batch_head = tl.program_id(0)
#     shard = tl.program_id(1)
#
#     rc = tl.arange(0, CS)
#     rf = tl.arange(0, HF)
#     rf_shard = tl.arange(0, HF_shard)
#
#     W_dtype = W1_init.type.element_ty
#     O_dtype = Out.type.element_ty
#
#     w_offset = batch_head * stride_w_batch_head + shard * stride_w_shard
#     ao_offset = batch_head * stride_a_batch_head + shard * stride_a_shard
#     bc_offset = batch_head * stride_b_batch_head
#     coeff_offset = batch_head * stride_coeff_batch_head
#
#     coeff = coeff + coeff_offset
#     XB = XB + bc_offset + (rc[:, None] * stride_b_n + rf[None, :] * stride_b_f)
#     XC = XC + bc_offset + (rc[:, None] * stride_b_n + rf[None, :] * stride_b_f)
#
#     XA = XA + ao_offset + (rc[:, None] * stride_a_n + rf_shard[None, :] * stride_a_f)
#     Out = Out + ao_offset + (rc[:, None] * stride_a_n + rf_shard[None, :] * stride_a_f)
#
#     W1_init = W1_init + w_offset + (rf[:, None] * stride_w_in + rf_shard[None, :] * stride_w_out)
#     W1_grad = W1_grad + w_offset + (rf[:, None] * stride_w_in + rf_shard[None, :] * stride_w_out)
#
#     XA_chunk = tl.load(XA)
#     XB_chunk = tl.load(XB)
#     XC_chunk = tl.load(XC)
#     coeff_chunk = tl.load(coeff)
#     W1_init_data = tl.load(W1_init)
#     W1_grad_data = tl.load(W1_grad)
#
#     # '''
#     Z1 = tl.sum(tl.trans(XB_chunk) * W1_init_data, 0) - XA_chunk # [HF,] - [1,HF] - > [1,HF]
#     W1_grad_data += tl.trans(XB_chunk) * Z1  # @xinhao: tl.dot(tl.trans(XB_chunk), Z1) doesn't support [HF,1] @ [1,HF]
#     W1_init_data -= coeff_chunk * W1_grad_data
#     Z1_bar = tl.sum(tl.trans(XC_chunk) * W1_init_data, 0)[None,:]
#     tl.store(Out, Z1_bar.to(O_dtype))
#     # '''
#
#     tl.store(W1_init, W1_init_data.to(W_dtype))
#     tl.store(W1_grad, W1_grad_data.to(W_dtype))
#
#
# def ttt_m1_triton_sharded_decode(XA, XB, XC, coeff, W1_init, W1_grad):
#     B_mul_NH, M, CS, HF_div_M = XA.shape
#     HF = XB.shape[-1]
#     assert CS == 1
#     assert W1_init.shape == W1_grad.shape == (B_mul_NH, M, HF, HF_div_M)
#     assert XB.shape == XC.shape == (B_mul_NH, CS, HF)
#     assert coeff.shape == (B_mul_NH, CS, 1)
#
#     output = torch.empty(size=(B_mul_NH, M, CS, HF_div_M), device=W1_init.device, dtype=torch.float16)
#     grid = (B_mul_NH, M, 1)
#
#     _m1_sharded_decode_kernel[grid](W1_init, W1_grad, XA, XB, XC, coeff, output,
#
#                                     W1_init.stride(0), W1_init.stride(1), W1_init.stride(2), W1_init.stride(3),  # strides for W
#
#                                     XA.stride(0), XA.stride(1), XA.stride(2), XA.stride(3),  # strides for XA, output
#
#                                     XB.stride(0), XB.stride(1), XB.stride(2),  # strides for XB, XC
#
#                                     coeff.stride(0), coeff.stride(1), coeff.stride(2),  # strides for coeff
#
#                                     CS, HF, HF_div_M)
#
#     return W1_init, W1_grad, output


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
    key=['HF', 'HF_shard'],
    restore_value=['W1_init', 'W1_grad'],
)
@triton.jit
def _m1_sharded_decode_kernel(W1_init, W1_grad,
                              XA, XB, XC, coeff, Out,
                              stride_w_batch_head, stride_w_shard, stride_w_in, stride_w_out,

                              stride_a_batch_head, stride_a_shard, stride_a_n, stride_a_f,  # XA, output

                              stride_b_batch_head, stride_b_n, stride_b_f,  # XB, XC

                              stride_coeff_batch_head, stride_coeff_n, stride_coeff_f,

                              CS: tl.constexpr, HF: tl.constexpr, HF_shard: tl.constexpr):
    batch_head = tl.program_id(0)
    shard = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)
    rf_shard = tl.arange(0, HF_shard)

    W_dtype = W1_init.type.element_ty
    O_dtype = Out.type.element_ty

    w_offset = batch_head * stride_w_batch_head + shard * stride_w_shard
    ao_offset = batch_head * stride_a_batch_head + shard * stride_a_shard
    bc_offset = batch_head * stride_b_batch_head
    coeff_offset = batch_head * stride_coeff_batch_head

    coeff = coeff + coeff_offset
    XB = XB + bc_offset + (rc[:, None] * stride_b_n + rf[None, :] * stride_b_f)
    XC = XC + bc_offset + (rc[:, None] * stride_b_n + rf[None, :] * stride_b_f)

    XA = XA + ao_offset + (rc[:, None] * stride_a_n + rf_shard[None, :] * stride_a_f)
    Out = Out + ao_offset + (rc[:, None] * stride_a_n + rf_shard[None, :] * stride_a_f)

    W1_init = W1_init + w_offset + (rf[:, None] * stride_w_in + rf_shard[None, :] * stride_w_out)
    W1_grad = W1_grad + w_offset + (rf[:, None] * stride_w_in + rf_shard[None, :] * stride_w_out)

    XA_chunk = tl.load(XA)
    XB_chunk = tl.load(XB)
    XC_chunk = tl.load(XC)
    coeff_chunk = tl.load(coeff)
    W1_init_data = tl.load(W1_init)
    W1_grad_data = tl.load(W1_grad)

    # '''
    Z1 = tl.sum(tl.trans(XB_chunk) * W1_init_data, 0) - XA_chunk # [HF,] - [1,HF] - > [1,HF]
    W1_grad_data += tl.trans(XB_chunk) * Z1  # @xinhao: tl.dot(tl.trans(XB_chunk), Z1) doesn't support [HF,1] @ [1,HF]
    W1_init_data -= coeff_chunk * W1_grad_data
    Z1_bar = tl.sum(tl.trans(XC_chunk) * W1_init_data, 0)[None,:]
    tl.store(Out, Z1_bar.to(O_dtype))
    # '''

    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))


def ttt_m1_triton_sharded_decode(XA, XB, XC, coeff, W1_init, W1_grad):
    B, NH_mul_M, CS, HF_div_M = XA.shape
    HF = XB.shape[-1]
    assert CS == 1
    assert W1_init.shape == W1_grad.shape == (B, NH_mul_M, HF, HF_div_M)
    assert XB.shape == XC.shape == (B, NH, CS, HF)
    assert coeff.shape == (B_mul_NH, CS, 1)

    output = torch.empty(size=(B_mul_NH, M, CS, HF_div_M), device=W1_init.device, dtype=torch.float16)
    grid = (B_mul_NH, M, 1)

    _m1_sharded_decode_kernel[grid](W1_init, W1_grad, XA, XB, XC, coeff, output,

                                    W1_init.stride(0), W1_init.stride(1), W1_init.stride(2), W1_init.stride(3),  # strides for W

                                    XA.stride(0), XA.stride(1), XA.stride(2), XA.stride(3),  # strides for XA, output

                                    XB.stride(0), XB.stride(1), XB.stride(2),  # strides for XB, XC

                                    coeff.stride(0), coeff.stride(1), coeff.stride(2),  # strides for coeff

                                    CS, HF, HF_div_M)

    return W1_init, W1_grad, output



if __name__ == "__main__":
    os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

    input_dtype = torch.float16

    ############### M1 Matching outputs abs diff ###############

    BS, NH, CS, HF = 64, 32, 1, 64
    M = 2
    HF_div_M = HF // M
    W1_pt = torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
    W1_grad_pt = torch.randn_like(W1_pt) * 0.02
    W1_triton = W1_pt.clone().reshape(BS * NH, HF, M, HF_div_M).permute(0,2,1,3)
    W1_grad_triton = W1_grad_pt.clone().reshape(BS * NH, HF, M, HF_div_M).permute(0,2,1,3)

    XA = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02

    XA_triton = XA.clone().reshape(BS * NH, CS, M, HF_div_M).permute(0,2,1,3)
    XB_triton = XB.clone().reshape(BS * NH, CS, HF)
    XC_triton = XC.clone().reshape(BS * NH, CS, HF)
    coeff_triton = coeff.clone().reshape(BS* NH, CS, 1)

    W1, W1_grad, \
    XCW_batch = ttt_m1_decode(XA, XB, XC, coeff, W1_pt, W1_grad_pt)

    # W1_triton, W1_grad_triton, \
    # XCW_batch_triton = ttt_m1_triton_decode(XA, XB, XC, coeff, W1_triton, W1_grad_triton)

    W1_triton, W1_grad_triton, \
    XCW_batch_triton = ttt_m1_triton_sharded_decode(XA_triton, XB_triton, XC_triton, coeff_triton,
                                                    W1_triton, W1_grad_triton)
    W1_triton = einops.rearrange(W1_triton, "(b h) m f f_m -> b h f (m f_m)", b=BS, h=NH)
    W1_grad_triton = einops.rearrange(W1_grad_triton, "(b h) m f f_m -> b h f (m f_m)", b=BS, h=NH)
    XCW_batch_triton = einops.rearrange(XCW_batch_triton, "(b h) m cs f_m -> b h cs (m f_m)", b=BS, h=NH)

    print('========== M1 Matching outputs abs diff ==========')
    print('W1 diff: ' + str(torch.abs(W1 - W1_triton).max()))
    print('W1_grad diff: ' + str(torch.abs(W1_grad - W1_grad_triton).max()))
    print('Output diff: ' + str(torch.abs(XCW_batch - XCW_batch_triton).max()))

