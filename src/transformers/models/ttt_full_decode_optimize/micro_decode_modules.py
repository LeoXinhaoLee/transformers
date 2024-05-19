import torch
import torch.nn as nn
import einops
import triton
import triton.language as tl
import os


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
    B, NH, N, HF = x.shape

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
    """
    Args:
        x: [B,NH,N=1,f]
        l2_target: [B,NH,N=1,f]
        gamma: [1,nh,1,f]
        beta: [1,nh,1,f]
        eps:

    Returns:
        grad_l_x: [B,nh,N=1,f]
    """
    B, NH, N, HF = x.shape

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)  # [B,nh,N=1,1]
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std  # [B,nh,N=1,f]

    # Scale and shift
    y = gamma * x_hat + beta  #[1,nh,N=1,f] * [B,nh,N=1,f] + [1,nh,N=1,f]

    grad_output = y - l2_target  # [B,nh,N=1,f]
    grad_x_hat = grad_output * gamma  # [B,nh,N=1,f]
    z = (
        (1.0 / HF)
        * (
            HF * grad_x_hat  # [B,nh,1,f]
            - grad_x_hat.sum(dim=-1, keepdim=True)  # [B,nh,1,1]
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)  # [B,nh,N=1,f] * [B,nh,N=1,1]
        )
        / std
    )
    return z

########## Pytorch ##########

###
## M2
###
def ttt_m2_decode(XA, XB, XC, coeff,
                  W1_init, W1_grad, b1_init, b1_grad,
                  W2_init, W2_grad, b2_init, b2_grad,
                  ln_weight, ln_bias):
    """
    Args:
        XA, XB, XC: [B,NH,CS,HF]
        coeff: [B,NH,CS,1]
        W1_init, W1_grad: [B,NH,HF,HF_prime]
        b1_init, b1_grad: [B,NH,1,HF_prime]
        W2_init, W2_grad: [B,NH,HF_prime,HF]
        b2_init, b2_grad: [B,NH,1,HF]

    Returns:
        Z2_bar: [B,NH,CS,HF]
    """

    Z1 = XB @ W1_init + b1_init  # [B,nh,K=1,f] @ [B,nh,f,f] + [B,nh,1,f] -> [B,nh,K=1,f]
    X2 = nn.functional.gelu(Z1, approximate='tanh')
    Z2 = X2 @ W2_init + b2_init

    reconstruction_target = XA - XB
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)  # [B,nh,K=1,f]
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-1, -2) * diff_gelu(Z1)

    W1_grad.add_(XB.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B,nh,1,f].t @ [B,nh,1,f] + [B,nh,f,f]
    b1_grad.add_(grad_l_wrt_Z1)
    W1_init.sub_(coeff * W1_grad)
    b1_init.sub_(coeff * b1_grad)
    Z1_bar = XC @ W1_init + b1_init  # [B,nh,K=1,f] @ ([B,nh,f,f] - [B,nh,1,1] * [B,nh,f,f])
    X2_bar = nn.functional.gelu(Z1_bar, approximate='tanh')

    W2_grad.add_(X2.transpose(-1, -2) @ grad_l_wrt_Z2)  # [B,nh,f,f]
    b2_grad.add_(grad_l_wrt_Z2)
    W2_init.sub_(coeff * W2_grad)
    b2_init.sub_(coeff * b2_grad)
    Z2_bar = X2_bar @ W2_init + b2_init  # [B,nh,K=1,f]
    Z2_bar = XC + ln_fwd(Z2_bar, ln_weight, ln_bias)

    return W1_init, W1_grad, b1_init, b1_grad, \
           W2_init, W2_grad, b2_init, b2_grad, \
           Z2_bar

###
## M1
###
def ttt_m1_decode(XA, XB, XC, coeff,
                  W1_init, W1_grad, b1_init, b1_grad,
                  ln_weight, ln_bias):
    """
    Args:
        XA XB XC: [B,NH,N=1,f]
        coeff: [B,NH,N=1,1]
        W1_init W1_grad: [B,NH,f,f]
        b1_init b1_grad: [B,NH,1,f]
        ln_weight ln_bias: [1,NH,1,f]

    Returns:

    """

    Z1 = XB @ W1_init + b1_init  # [B,nh,K=1,f] @ [B,nh,f,f] + [B,nh,1,f] -> [B,nh,K=1,f]
    reconstruction_target = XA - XB
    # grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B,nh,K=1,f]
    # Z1_bar = grad_l_wrt_Z1

    mu = Z1.mean(dim=-1, keepdim=True)  # [B,nh,N=1,1]
    var = Z1.var(dim=-1, keepdim=True, unbiased=False)
    # std = torch.sqrt(var + 1e-6)
    std = 1.
    x_hat = (Z1 - mu) / std  # [B,nh,N=1,f]
    y = ln_weight * x_hat + ln_bias  # [1,nh,N=1,f] * [B,nh,N=1,f] + [1,nh,N=1,f]

    # grad_output = y - reconstruction_target  # [B,nh,N=1,f]
    # grad_x_hat = grad_output * ln_weight  # [B,nh,N=1,f]
    # Z1_bar = (
    #         (1.0 / HF)
    #         * (
    #                 HF * grad_x_hat  # [B,nh,1,f]
    #                 - grad_x_hat.sum(dim=-1, keepdim=True)  # [B,nh,1,1]
    #                 - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)  # [B,nh,N=1,f] * [B,nh,N=1,1]
    #         )
    #         / std
    # )
    Z1_bar = y


    # W1_grad.add_(XB.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B,nh,1,f].t @ [B,nh,1,f] + [B,nh,f,f]
    # b1_grad.add_(grad_l_wrt_Z1)
    #
    # W1_init.sub_(coeff * W1_grad)  # [B,nh,f,f] - [B,nh,N=1,1] * [B,nh,f,f]
    # b1_init.sub_(coeff * b1_grad)  # [B,nh,1,f] - [B,nh,N=1,1] * [B,nh,1,f]
    # Z1_bar = XC @ W1_init + b1_init  # [B,nh,K=1,f] @ [B,nh,f,f] - [B,nh,1,1] * [B,nh,f,f]
    # Z1_bar = XC + ln_fwd(Z1_bar, ln_weight, ln_bias)

    # Z1_bar = ln_weight * XC + ln_bias  # testing
    # Z1_bar = XC
    return W1_init, W1_grad, b1_init, b1_grad, Z1_bar



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

                            CS=CS, HF=HF, HF_prime=HF_prime,

                            # num_warps=8
                            )

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
    restore_value=['W1_init', 'W1_grad', 'b1_init', 'b1_grad'],
)
@triton.jit
def _m1_decode_kernel(W1_init, W1_grad, b1_init, b1_grad,
                      XA, XB, XC, coeff, Out,
                      ln_weight, ln_bias,

                      stride_w_batch, stride_w_head, stride_w_in, stride_w_out,
                      stride_b_batch, stride_b_head, stride_b_in, stride_b_out,
                      stride_ln_head, stride_ln_f,

                      stride_x_batch, stride_x_head, stride_x_n, stride_x_f,
                      stride_coeff_batch, stride_coeff_head,

                      CS: tl.constexpr, HF: tl.constexpr):
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

    reconstruction_target = XA_data - XB_data  # [1,HF]
    Z1 = tl.sum(tl.trans(XB_data) * W1_init_data, 0) + b1_init_data  # [1,HF]

    mu = tl.sum(Z1) / HF  # [1,]
    var = tl.sum((Z1 - mu) * (Z1 - mu)) / HF  # [1,]
    # std = tl.sqrt(var + 1e-6)
    std = 1.
    x_hat = (Z1 - mu) / std  # [1,f]
    y = ln_weight_data * x_hat + ln_bias_data  # [1,f]

    # grad_output = y - reconstruction_target  # [1,f]
    # grad_x_hat = ln_weight_data * grad_output  # [1,f]
    # grad_l_Z1 = (
    #         (1.0 / HF)
    #         * (
    #             HF * grad_x_hat  # [1,f]
    #             - tl.sum(grad_x_hat)  #[1,]
    #             - x_hat * tl.sum(grad_x_hat * x_hat)  # [1,f] * [1,]
    #         )
    #         / std
    # )  # grad_l_Z1: [1,f]
    grad_l_Z1 = y

    # if batch == 0 and head == 0:
    #     tmp = tl.tensor(grad_l_Z1.shape, type=tl.float32)
    #     tl.device_print('grad_l_Z1: ', tmp)

    # W1_grad_data += (tl.trans(XB_data) * grad_l_Z1)  # @xinhao: tl.dot(tl.trans(XB_chunk), Z1) doesn't support [HF,1] @ [1,HF]
    # b1_grad_data += grad_l_Z1
    # W1_init_data -= coeff_data * W1_grad_data
    # b1_init_data -= coeff_data * b1_grad_data
    # Z1 = tl.sum(tl.trans(XC_data) * W1_init_data, 0) + b1_init_data  # Z1_bar

    # Post LN
    # mu = tl.sum(Z1) / HF  # [1,]
    # var = tl.sum((Z1 - mu) * (Z1 - mu)) / HF  # [1,]
    # std = tl.sqrt(var + 1e-6)
    # x_hat = (Z1 - mu) / std  # [1,f]
    # Z1 = XC_data + (ln_weight_data * x_hat + ln_bias_data)  # XC + LN(Z1_bar): [1,f]

    # Z1 = ln_weight_data * XC  #  testing
    # if batch == 0 and head == 0:
    #     # tmp = tl.tensor(Z1.shape, type=tl.float32)
    #     # tl.device_print('Z1: ', tmp)
    #     tmp = tl.tensor(ln_bias_data.shape, type=tl.float32)
    #     tl.device_print('ln_bias_data: ', tmp)

    # Z1 = Z1 + ln_bias_data
    # tl.store(Out, Z1.to(O_dtype))
    tl.store(Out, grad_l_Z1.to(O_dtype))
    # tl.store(Out, XC_data)
    # tl.store(XA, XA_data)

    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))
    tl.store(b1_init, b1_init_data.to(W_dtype))
    tl.store(b1_grad, b1_grad_data.to(W_dtype))
    # tl.store(ln_weight, ln_weight_data.to(W_dtype))
    # tl.store(ln_bias, ln_bias_data.to(W_dtype))


def ttt_m1_triton_decode(XA, XB, XC, coeff,
                         W1_init, W1_grad, b1_init, b1_grad,
                         ln_weight, ln_bias):
    B, NH, CS, HF = XA.shape
    assert CS == 1
    assert W1_init.shape == W1_grad.shape == (B, NH, HF, HF)
    assert b1_init.shape == b1_grad.shape == (B, NH, 1, HF)
    assert ln_weight.shape == ln_bias.shape == (1, NH, 1, HF)
    assert XA.shape == XB.shape == XC.shape
    assert coeff.shape == (B, NH, CS, 1)
    output = torch.empty(size=(B, NH, CS, HF), device=W1_init.device, dtype=torch.float16)  # TODO FIX DTYPE
    grid = (B, NH, 1)
    _m1_decode_kernel[grid](W1_init, W1_grad, b1_init, b1_grad,
                            XA, XB, XC, coeff, output,
                            ln_weight, ln_bias,

                            W1_init.stride(0), W1_init.stride(1), W1_init.stride(2), W1_init.stride(3),  # strides for W
                            b1_init.stride(0), b1_init.stride(1), b1_init.stride(2), b1_init.stride(3),  # strides for b
                            ln_weight.stride(1), ln_weight.stride(2),

                            XA.stride(0), XA.stride(1), XA.stride(2), XA.stride(3),  # strides for ABCO, output
                            coeff.stride(0), coeff.stride(1), # strides for coeff

                            CS, HF)

    return W1_init, W1_grad, b1_init, b1_grad, output



if __name__ == "__main__":
    os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
    torch.manual_seed(0)
    # input_dtype = torch.float16
    input_dtype = torch.float64

    ############### M2 Matching outputs abs diff ###############

    # BS, NH, CS, HF, HF_prime = 64, 32, 1, 64, 4 * 64
    #
    # W1_pt = torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 1.0
    # W1_grad_pt = torch.randn_like(W1_pt) * 1.0
    # W1_triton = W1_pt.clone()
    # W1_grad_triton = W1_grad_pt.clone()
    #
    # b1_pt = torch.randn(BS, NH, 1, HF_prime, device='cuda', dtype=input_dtype) * 1.0
    # b1_grad_pt = torch.randn_like(b1_pt) * 1.0
    # b1_triton = b1_pt.clone()
    # b1_grad_triton = b1_grad_pt.clone()
    #
    # W2_pt = torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 1.0
    # W2_grad_pt = torch.randn_like(W2_pt) * 1.0
    # W2_triton = W2_pt.clone()
    # W2_grad_triton = W2_grad_pt.clone()
    #
    # b2_pt = torch.randn(BS, NH, 1, HF, device='cuda', dtype=input_dtype) * 1.0
    # b2_grad_pt = torch.randn_like(b2_pt) * 1.0
    # b2_triton = b2_pt.clone()
    # b2_grad_triton = b2_grad_pt.clone()
    #
    # ln_weight = torch.randn(BS, NH, 1, HF, device='cuda', dtype=input_dtype)
    # ln_bias = torch.randn(BS, NH, 1, HF, device='cuda', dtype=input_dtype)
    #
    # XA = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 1.0
    # XB = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 1.0
    # XC = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 1.0
    # coeff = torch.randn(BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 1.0
    #
    # W1_pt, W1_grad_pt, b1_pt, b1_grad_pt, \
    # W2_pt, W2_grad_pt, b2_pt, b2_grad_pt, \
    # XCW_batch = ttt_m2_decode(XA, XB, XC, coeff,
    #                           W1_pt, W1_grad_pt, b1_pt, b1_grad_pt,
    #                           W2_pt, W2_grad_pt, b2_pt, b2_grad_pt,
    #                           ln_weight, ln_bias)
    # print(XCW_batch.shape)

    # W1_triton, W1_grad_triton, \
    # W2_triton, W2_grad_triton, \
    # XCW_batch_triton = ttt_m2_triton_decode(XA, XB, XC, coeff,
    #                                         W1_original, W1_grad_original, W2_original, W2_grad_original)
    #
    # print('========== M2 Matching outputs abs diff ==========')
    # print('W1 diff: ' + str(torch.abs(W1 - W1_triton).max()))
    # print('W1_grad diff: ' + str(torch.abs(W1_grad - W1_grad_triton).max()))
    # print('W2 diff: ' + str(torch.abs(W2 - W2_triton).max()))
    # print('W2_grad diff: ' + str(torch.abs(W2_grad - W2_grad_triton).max()))
    # print('Output diff: ' + str(torch.abs(XCW_batch - XCW_batch_triton).max()))

    ############### M1 Matching outputs abs diff ###############

    BS, NH, CS, HF = 64, 32, 1, 64

    W1_pt = torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 1.0
    W1_grad_pt = torch.randn_like(W1_pt) * 1.0
    b1_pt = torch.randn(BS, NH, 1, HF, device='cuda', dtype=input_dtype) * 1.0
    b1_grad_pt = torch.randn_like(b1_pt) * 1.0

    W1_triton = W1_pt.clone()
    W1_grad_triton = W1_grad_pt.clone()
    b1_triton = b1_pt.clone()
    b1_grad_triton = b1_grad_pt.clone()

    # ln_weight = torch.randn(1, NH, 1, HF, device='cuda', dtype=input_dtype)
    # ln_bias = torch.randn(1, NH, 1, HF, device='cuda', dtype=input_dtype)
    ln_weight = torch.ones(1, NH, 1, HF, device='cuda', dtype=input_dtype)
    ln_bias = torch.randn(1, NH, 1, HF, device='cuda', dtype=input_dtype) * 1.0

    ln_weight_triton = ln_weight.clone()
    ln_bias_triton = ln_bias.clone()

    XA = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 1.0
    XB = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 1.0
    XC = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 1.0
    coeff = torch.randn(BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 1.0

    XA_triton = XA.clone()
    XB_triton = XB.clone()
    XC_triton = XC.clone()

    W1_pt, W1_grad_pt, b1_pt, b1_grad_pt, \
    XCW_batch_pt = ttt_m1_decode(XA, XB, XC, coeff,
                                 W1_pt, W1_grad_pt, b1_pt, b1_grad_pt,
                                 ln_weight, ln_bias)
    print(XCW_batch_pt.shape)

    W1_triton, W1_grad_triton, b1_triton, b1_grad_triton, \
    XCW_batch_triton = ttt_m1_triton_decode(XA_triton, XB_triton, XC_triton, coeff,
                                            W1_triton, W1_grad_triton, b1_triton, b1_grad_triton,
                                            ln_weight_triton, ln_bias_triton)
    print(XCW_batch_triton.shape)

    print('========== M1 Matching outputs abs diff ==========')
    print('W1 diff: ' + str(torch.abs(W1_pt - W1_triton).max()))
    print('W1_grad diff: ' + str(torch.abs(W1_grad_pt - W1_grad_triton).max()))
    print('b1 diff: ' + str(torch.abs(b1_pt - b1_triton).max()))
    print('b1_grad diff: ' + str(torch.abs(b1_grad_pt - b1_grad_triton).max()))
    print('Output diff: ' + str(torch.abs(XCW_batch_pt - XCW_batch_triton).max()))
    print('LN weight diff: ' + str(torch.abs(ln_weight - ln_weight_triton).max()))
    print('LN bias diff: ' + str(torch.abs(ln_bias - ln_bias_triton).max()))
    print('XA diff: ' + str(torch.abs(XA - XA_triton).max()))


