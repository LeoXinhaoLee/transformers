import torch
import einops
import triton
import triton.language as tl
import os


########## Pytorch ##########

###
## M2
###
def ttt_m2_decode(XA_chunk, XB_chunk, XC_chunk, coeff_chunk, W1_init, W1_grad, W2_init, W2_grad):
    """
    Args:
        XA, XB, XC: [B,NH,CS,HF]
        coeff: [B,NH,CS,1]
        W1_init, W1_grad: [B,NH,HF,HF_prime]
        W2_init, W2_grad: [B,NH,HF_prime,HF]

    Returns:
        Z2_bar: [B,NH,CS,HF]
    """
    Z1 = XB_chunk @ W1_init  # [B,nh,K=1,f] @ [B,nh,f,f] -> [B,nh,K=1,f]
    Z2 = Z1 @ W2_init

    Z2.sub_(XA_chunk)  # grad_l_wrt_Z2 = Z2 - XA_chunk
    grad_l_wrt_Z1 = Z2 @ W2_init.transpose(-1, -2)

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)
    W1_init.sub_(coeff_chunk * W1_grad)
    Z1_bar = XC_chunk @ W1_init

    W2_grad.add_(Z1.transpose(-1, -2) @ Z2)
    W2_init.sub_(coeff_chunk * W2_grad)
    Z2_bar = Z1_bar @ W2_init
    return W1_init, W1_grad, W2_init, W2_grad, Z2_bar

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
## M2
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
    key=['HF', 'HF_prime'],  # the two above configs will be evaluated anytime the value of key changes
    restore_value=['W1_init', 'W1_grad', 'W2_init', 'W2_grad'],
)
@triton.jit
def _m2_decode_kernel(W1_init, W1_grad,
                      W2_init, W2_grad,
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

    # '''
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
    tl.store(Out_chunk, Z2_bar.to(O_dtype))
    # '''

    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))
    tl.store(W2_init, W2_init_data.to(W_dtype))
    tl.store(W2_grad, W2_grad_data.to(W_dtype))

def ttt_m2_triton_decode(XA, XB, XC, coeff, W1_init, W1_grad, W2_init, W2_grad):
    B, NH, CS, HF = XA.shape
    HF_prime = W1_init.shape[-1]
    assert CS == 1
    assert W1_init.shape == W1_grad.shape == (B, NH, HF, HF_prime)
    assert W2_init.shape == W2_grad.shape == (B, NH, HF_prime, HF)
    assert XA.shape == XB.shape == XC.shape
    assert coeff.shape == (B, NH, CS, 1)

    output = torch.empty(size=(B, NH, CS, HF), device=W1_init.device, dtype=torch.float16)  # TODO FIX DTYPE
    grid = (B, NH, 1)

    _m2_decode_kernel[grid](W1_init, W1_grad, W2_init, W2_grad,
                            XA, XB, XC, coeff,
                            output,

                            NH * HF * HF_prime,     HF * HF_prime,     HF_prime,    1,  # strides for W1: [B,NH,HF,HF_prime]

                            NH * HF_prime * HF,     HF_prime * HF,     HF,          1,  # strides for W2

                            NH * CS * HF,           CS * HF,           HF,          1,  # strides for ABCO, output

                            NH * CS * 1,            CS * 1,            1,           1,  # strides for coeff

                            CS=CS, HF=HF, HF_prime=HF_prime)

    return W1_init, W1_grad, W2_init, W2_grad, output


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


###
## M2 Block (slow, not used by default)
###
def get_cuda_autotune_block_config():
    return [
        triton.Config({'BLOCK_SIZE': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 4}, num_stages=3, num_warps=8),

        triton.Config({'BLOCK_SIZE': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 4}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 1}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE': 4}, num_stages=5, num_warps=2),
    ]

def get_cuda_autotune_non_block_config():
    return [
        triton.Config({}, num_stages=7, num_warps=8),
        triton.Config({}, num_stages=6, num_warps=8),
        triton.Config({}, num_stages=5, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=6, num_warps=4),
    ]

@triton.autotune(
    # configs=get_cuda_autotune_block_config(),
    configs=get_cuda_autotune_non_block_config(),
    key=['HF', 'HF_prime'],  # the two above configs will be evaluated anytime the value of key changes
    restore_value=['W1_init', 'W1_grad', 'W2_init', 'W2_grad'],
)
@triton.jit
def _m2_decode_block_kernel(W1_init, W1_grad,
                      W2_init, W2_grad,
                      XA, XB, XC, coeff,
                      Out,
                      stride_w1bh, stride_w1fi, stride_w1fo,  # [BS*nh,f,f]
                      stride_w2bh, stride_w2fi, stride_w2fo,
                      stride_abh, stride_an, stride_af,  # [B*nh,1,f]
                      CS: tl.constexpr, HF: tl.constexpr, HF_prime: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):

    b_h = tl.program_id(0)

    rb = tl.arange(0, BLOCK_SIZE)
    rf = tl.arange(0, HF)
    rf_prime = tl.arange(0, HF_prime)

    W_dtype = W1_init.type.element_ty
    O_dtype = Out.type.element_ty

    w1_offset = b_h * BLOCK_SIZE * stride_w1bh
    w2_offset = b_h * BLOCK_SIZE * stride_w2bh
    abco_offset = b_h * BLOCK_SIZE * stride_abh
    coeff_offset = b_h * BLOCK_SIZE * 1

    W1_current_blk_offset = rb[:, None, None] * stride_w1bh + \
                            (rf[:, None] * stride_w1fi + rf_prime[None, :] * stride_w1fo)[None, :]
    W1_init = W1_init + w1_offset + W1_current_blk_offset # [bs, HF, HF_prime]
    W1_grad = W1_grad + w1_offset + W1_current_blk_offset

    W2_current_blk_offset = rb[:, None, None] * stride_w2bh + \
                            (rf_prime[:, None] * stride_w2fi + rf[None, :] * stride_w2fo)[None, :]
    W2_init = W2_init + w2_offset + W2_current_blk_offset  # [bs, HF, HF_prime]
    W2_grad = W2_grad + w2_offset + W2_current_blk_offset

    coeff = coeff + coeff_offset + rb[:, None, None]  # [bs,1,1]

    X_current_blk_offset = rb[:, None, None] * stride_abh + (rf[None, :] * stride_an)[None, :]  # [bs,1,HF]
    XA = XA + abco_offset + X_current_blk_offset    # [bs,1,HF]
    XB = XB + abco_offset + X_current_blk_offset
    XC = XC + abco_offset + X_current_blk_offset
    Out_chunk = Out + abco_offset + X_current_blk_offset  # [bs,1,HF]

    XA_chunk = tl.load(XA)
    XB_chunk = tl.load(XB)
    XC_chunk = tl.load(XC)
    coeff_chunk = tl.load(coeff)
    W1_init_data = tl.load(W1_init)
    W1_grad_data = tl.load(W1_grad)
    W2_init_data = tl.load(W2_init)
    W2_grad_data = tl.load(W2_grad)

    Z1 = tl.sum(tl.reshape(XB_chunk, shape=(BLOCK_SIZE, HF, CS)) * W1_init_data, 1)[:, None, :]  # [bs,1,HF] @ [bs,HF,HF_prime] -> [bs,1,HF_prime]
    Z2 = tl.sum(tl.reshape(Z1, shape=(BLOCK_SIZE, HF_prime, CS)) * W2_init_data, 1)[:, None, :]  # [bs,1,HF_prime].t @ [bs,HF_prime,HF]

    grad_l_wrt_Z2 = Z2 - XA_chunk  # [bs,1,HF]
    grad_l_wrt_Z1 = tl.sum(grad_l_wrt_Z2 * W2_init_data, 2)[:, None, :]  # [bs,1,HF] * [bs,HF_p,HF] -> [bs,HF_p] -> [bs,1,HF_p]

    W1_grad_data += tl.reshape(XB_chunk, shape=(BLOCK_SIZE, HF, CS)) * grad_l_wrt_Z1  # [bs,1,HF].t * [bs,1,HF_p] -> [bs,HF,HF_p]
    W1_init_data -= coeff_chunk * W1_grad_data
    Z1_bar = tl.sum(tl.reshape(XC_chunk, shape=(BLOCK_SIZE, HF, CS)) * W1_init_data, 1)[:, None, :]  # [bs,1,HF_p]

    W2_grad_data += tl.reshape(Z1, shape=(BLOCK_SIZE, HF_prime, CS)) * grad_l_wrt_Z2  # [bs,1,HF_p].t * [bs,1,HF] -> [bs,HF_p,HF]
    W2_init_data -= coeff_chunk * W2_grad_data
    Z2_bar = tl.sum(tl.reshape(Z1_bar, shape=(BLOCK_SIZE, HF_prime, CS)) * W2_init_data, 1)[:, None, :]  # [bs,1,HF]
    tl.store(Out_chunk, Z2_bar.to(O_dtype))

    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))
    tl.store(W2_init, W2_init_data.to(W_dtype))
    tl.store(W2_grad, W2_grad_data.to(W_dtype))


def ttt_m2_triton_block_decode(XA, XB, XC, coeff, W1_init, W1_grad, W2_init, W2_grad):
    B_mul_NH, CS, HF = XA.shape
    HF_prime = W1_init.shape[-1]
    assert CS == 1
    assert W1_init.shape == W1_grad.shape == (B_mul_NH, HF, HF_prime)
    assert W2_init.shape == W2_grad.shape == (B_mul_NH, HF_prime, HF)
    assert XA.shape == XB.shape == XC.shape
    assert coeff.shape == (B_mul_NH, CS, 1)

    output = torch.empty(size=(B_mul_NH, CS, HF), device=W1_init.device, dtype=torch.float16)  # TODO FIX DTYPE

    # grid = (B * NH,)
    grid = lambda META: (triton.cdiv(B_mul_NH, META['BLOCK_SIZE']),)

    _m2_decode_block_kernel[grid](W1_init, W1_grad, W2_init, W2_grad,
                            XA, XB, XC, coeff,
                            output,

                            HF * HF_prime,     HF_prime,     1,  # strides for W1: [B*NH,HF,HF_prime]

                            HF_prime * HF,     HF,           1,  # strides for W2: [B*NH,HF_prime,HF]

                            CS * HF,           HF,           1,  # strides for ABCO, output: [B*NH,CS,HF]

                            CS=CS, HF=HF, HF_prime=HF_prime,

                            BLOCK_SIZE=1,  # Make sure that you don't re-define auto-tuned symbols
                            )

    return W1_init, W1_grad, W2_init, W2_grad, output
