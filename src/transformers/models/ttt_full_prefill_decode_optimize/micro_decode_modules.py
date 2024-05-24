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
    restore_value=['W1_init', 'b1_init'],
)
@triton.jit
def _m1_prefill_chunk_triton(
        W1_init, b1_init,
        XA, XB, XC, coeff, coeff_last, Out,
        ln_weight, ln_bias,

        stride_w_batch, stride_w_head, stride_w_in, stride_w_out,
        stride_b_batch, stride_b_head, stride_b_in, stride_b_out,
        stride_ln_head, stride_ln_f,
        stride_x_batch, stride_x_head, stride_x_nc, stride_x_cs, stride_x_f,
        stride_coeff_batch, stride_coeff_head, stride_coeff_nc, stride_coeff_cs,

        CS: tl.constexpr,
        HF: tl.constexpr,
        N_CHUNK: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    abco_offset = batch * stride_x_batch + head * stride_x_head
    coeff_offset = batch * stride_coeff_batch + head * stride_coeff_head
    coeff_last_offset = coeff_offset
    w_offset = batch * stride_w_batch + head * stride_w_head
    b_offset = batch * stride_b_batch + head * stride_b_head
    ln_offset = head * stride_ln_head

    W_dtype = W1_init.type.element_ty
    O_dtype = Out.type.element_ty

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)

    XA_blk_ptr = XA + abco_offset
    XB_blk_ptr = XB + abco_offset
    XC_blk_ptr = XC + abco_offset
    Out_blk_ptr = Out + abco_offset

    coeff_blk_ptr = coeff + coeff_offset
    coeff_last_blk_ptr = coeff_last + coeff_last_offset

    W1_data_blk_ptr = W1_init + w_offset + (rf[:, None] * stride_w_in + rf[None, :] * stride_w_out)
    b1_data_blk_ptr = b1_init + b_offset + rf[None, :] * stride_b_out  # [1,f]
    W1_data = tl.load(W1_data_blk_ptr)  # [f,f]
    b1_data = tl.load(b1_data_blk_ptr)  # [1,f]

    ln_blk_shape = ln_offset + rf[None, :] * stride_ln_f  # [1,f]
    ln_weight_data = tl.load(ln_weight + ln_blk_shape)
    ln_bias_data = tl.load(ln_bias + ln_blk_shape)

    for i in range(N_CHUNK):
        local_abco_offset = i * stride_x_nc + (rc[:, None] * stride_x_cs + rf[None, :] * stride_x_f)
        local_coeff_offset = i * stride_coeff_nc
        local_coeff_last_offset = local_coeff_offset

        XA_chunk = tl.load(XA_blk_ptr + local_abco_offset)
        XB_chunk = tl.load(XB_blk_ptr + local_abco_offset)
        XC_chunk = tl.load(XC_blk_ptr + local_abco_offset)  # [CS,f]
        coeff_chunk = tl.load(coeff_blk_ptr + local_coeff_offset + rc * stride_coeff_cs)[:,None]  # [CS,1]
        coeff_chunk_last = tl.load(coeff_last_blk_ptr + local_coeff_last_offset)  # scaler

        Z1 = tl.dot(XB_chunk, W1_data) + b1_data  # [CS,f] @ [f,f] -> [CS,f]
        # tl.device_print(Z1.type)

        # tl.device_print(W1_init.type.element_ty)
        reconstruction_target = XA_chunk - XB_chunk  # [CS,f]

        # if batch == 0 and head == 0:
            # tmp = tl.tensor(Z1.shape, type=tl.float32)
            # tl.device_print('Z1: ', tmp)
            # tmp = tl.tensor(reconstruction_target.shape, type=tl.float32)
            # tl.device_print('recons: ', tmp)
            # tmp = tl.tensor(b1_data.shape, type=tl.float32)
            # tl.device_print('b1: ', tmp)
            # tmp = tl.tensor(mu.shape, type=tl.float32)
            # tl.device_print('mu: ', tmp)

        mu = tl.sum(Z1, 1) / HF  # [CS,]
        mu = mu[:,None]  # [CS,1]
        # if batch == 0 and head == 0:
        #     tmp = tl.tensor(mu.shape, type=tl.float32)
        #     tl.device_print('mu: ', tmp)

        var = tl.sum((Z1 - mu) * (Z1 - mu), 1) / HF  # [CS,]
        var = var[:,None]
        std = tl.sqrt(var + 1e-6)
        x_hat = (Z1 - mu) / std  # [CS,f]
        y = ln_weight_data * x_hat + ln_bias_data  # [CS,f]

        grad_output = y - reconstruction_target  # [CS,f]
        grad_x_hat = ln_weight_data * grad_output  # [CS,f]

        tmp = grad_x_hat * x_hat
        # if batch == 0 and head == 0:
            # tmp = tl.tensor(grad_x_hat.shape, type=tl.float32)
            # tl.device_print('grad_x_hat: ', tmp)
            # tmp = tl.tensor(x_hat.shape, type=tl.float32)
            # tl.device_print('x_hat: ', tmp)
        grad_l_Z1 = (
                (1.0 / HF)
                * (
                        HF * grad_x_hat  # [CS,f]
                        - tl.sum(grad_x_hat, 1)[:,None]  # [CS,]
                        - x_hat * (tl.sum(grad_x_hat * x_hat, 1))[:,None]  # [CS,f] * [CS,]
                )
                / std
        ).to(W_dtype)  # grad_l_Z1: [CS,f]
        #
        mask = rc[:, None] >= rc[None, :]
        Attn1_full = tl.dot(XC_chunk, tl.trans(XB_chunk))
        Attn1 = tl.where(mask, Attn1_full, 0).to(W_dtype)  # [CS,CS]

        # b1_bar = (b1_data - coeff_chunk * tl.cumsum(grad_l_Z1, 0)).to(W_dtype)   # [CS,f] - [CS,1] * [CS,f]
        # b1_bar = - coeff_chunk * tl.cumsum(grad_l_Z1, axis=0).to(W_dtype)
        # if batch == 0 and head == 0:
        #     tmp = tl.tensor(b1_bar.shape, type=tl.float32)
        #     tl.device_print('b1 bar: ', tmp)
        b1_bar = b1_data - coeff_chunk * grad_l_Z1

        Z1_bar = tl.dot(XC_chunk, W1_data) - tl.dot((coeff_chunk * Attn1), grad_l_Z1) + b1_bar  # [CS,f]
        # Z1_bar = tl.dot(XC_chunk, W1_data) - tl.dot((coeff_chunk * Attn1), grad_l_Z1)  # [CS,f]
        # if batch == 0 and head == 0:
        #     tl.device_print("=========================== ", i)
        #     tmp = tl.tensor(grad_x_hat.shape, type=tl.float32)
        #     tl.device_print('grad_x_hat: ', tmp)
        tl.store(Out_blk_ptr + local_abco_offset, Z1_bar.to(O_dtype))  #.to(Output_chunk_ptr.type.element_ty)

        W1_data -= tl.dot(tl.trans(coeff_chunk_last * XB_chunk), grad_l_Z1).to(W_dtype)
        # if batch == 0 and head == 0:
        #     tl.device_print(W1_data.type)

        # Output_chunk_ptr = Out_blk_ptr + local_abco_offset
        # if batch == 0 and head == 0:
            # tmp = tl.tensor(Output_chunk_ptr.shape, type=tl.float32)
            # tl.device_print('Output_chunk_ptr', tmp)
            # tl.device_print('type:', Output_chunk_ptr.type.element_ty)
        # tl.store(Output_chunk_ptr, Z1_bar.to(O_dtype))  #.to(Output_chunk_ptr.type.element_ty)
        # tl.store(Output_chunk_ptr, grad_l_Z1.to(O_dtype))  #.to(Output_chunk_ptr.type.element_ty)

    tl.store(W1_data_blk_ptr, W1_data)
    tl.store(b1_data_blk_ptr, b1_data)


if __name__ == "__main__":
    os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
    torch.manual_seed(0)
    input_dtype = torch.float16
    # input_dtype = torch.float64

    BS, NH, NC, CS, HF = 64, 32, 2, 16, 64

    W1_pt = torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 1.0
    b1_pt = torch.randn(BS, NH, 1, HF, device='cuda', dtype=input_dtype) * 1.0

    W1_triton = W1_pt.clone()
    b1_triton = b1_pt.clone()

    ln_weight = torch.ones(1, NH, 1, HF, device='cuda', dtype=input_dtype)
    ln_bias = torch.randn(1, NH, 1, HF, device='cuda', dtype=input_dtype) * 1.0

    ln_weight_triton = ln_weight.clone()
    ln_bias_triton = ln_bias.clone()

    XA = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 1.0
    XB = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 1.0
    XC = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 1.0
    coeff = torch.randn(BS, NH, NC, CS, 1, device='cuda', dtype=input_dtype) * 1.0
    coeff_last = torch.randn(BS, NH, NC, 1, 1, device='cuda', dtype=input_dtype) * 1.0

    XA_triton = XA.clone()
    XB_triton = XB.clone()
    XC_triton = XC.clone()

    # W1_pt, W1_grad_pt, b1_pt, b1_grad_pt, \
    # XCW_batch_pt = ttt_m1_decode(XA, XB, XC, coeff,
    #                              W1_pt, W1_grad_pt, b1_pt, b1_grad_pt,
    #                              ln_weight, ln_bias)
    # print(XCW_batch_pt.shape)

    grid = (BS, NH, 1)
    output = torch.empty_like(XA)
    _m1_prefill_chunk_triton[grid](W1_triton, b1_triton,
                                   XA, XB, XC, coeff, coeff_last, output,
                                   ln_weight, ln_bias,

                                   W1_triton.stride(0), W1_triton.stride(1), W1_triton.stride(2), W1_triton.stride(3),
                                   b1_triton.stride(0), b1_triton.stride(1), b1_triton.stride(2), b1_triton.stride(3),
                                   ln_weight.stride(1), ln_weight.stride(2),

                                   XA.stride(0), XA.stride(1), XA.stride(2), XA.stride(3), XA.stride(4),
                                   coeff.stride(0), coeff.stride(1), coeff.stride(2), coeff.stride(3),

                                   CS, HF, NC)
    print(output.shape)

    # print('========== M1 Matching outputs abs diff ==========')
    # print('W1 diff: ' + str(torch.abs(W1_pt - W1_triton).max()))
    # print('W1_grad diff: ' + str(torch.abs(W1_grad_pt - W1_grad_triton).max()))
    # print('b1 diff: ' + str(torch.abs(b1_pt - b1_triton).max()))
    # print('b1_grad diff: ' + str(torch.abs(b1_grad_pt - b1_grad_triton).max()))
    # print('Output diff: ' + str(torch.abs(XCW_batch_pt - XCW_batch_triton).max()))
    # print('LN weight diff: ' + str(torch.abs(ln_weight - ln_weight_triton).max()))
    # print('LN bias diff: ' + str(torch.abs(ln_bias - ln_bias_triton).max()))
    # print('XA diff: ' + str(torch.abs(XA - XA_triton).max()))


