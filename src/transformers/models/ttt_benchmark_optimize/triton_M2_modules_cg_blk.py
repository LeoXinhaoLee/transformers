import sys
import pdb

import torch
import einops
import triton
import triton.language as tl
import os

EXPAND = 4

def clean_tensor(tensor):
    tensor[torch.isnan(tensor) | (tensor == float('inf')) | (tensor == float('-inf'))] = 0
    return tensor


########## Pytorch ##########
###
## M2
###
def ttt_m2_decode(XA_chunk, XB_chunk, XC_chunk, coeff_chunk, W1_init, W1_grad, W2_init, W2_grad):
    """
    Args:
        XA, XB, XC: [B*NH,CS,HF]
        coeff: [B*NH,CS,1]
        W1_init, W1_grad: [B*NH,HF,HF_prime]
        W2_init, W2_grad: [B*NH,HF_prime,HF]

    Returns:
        Z2_bar: [B*NH,CS,HF]
    """
    Z1 = XB_chunk @ W1_init  # [B*nh,K=1,f] @ [B*nh,f,f] -> [B*nh,K=1,f]
    Z2 = Z1 @ W2_init

    Z2.sub_(XA_chunk)
    grad_l_wrt_Z1 = Z2 @ W2_init.transpose(-1, -2)  # [b*nh,1,f] @ [b*nh,f_p,f].t

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)  # [b*nh,1,f].t @ [b*nh,1,f_p] -> [b*nh,f,f_p]
    W1_init.sub_(coeff_chunk * W1_grad)  # [b*nh,1,1] * [b*nh,f,f_p]
    Z1_bar = XC_chunk @ W1_init

    W2_grad.add_(Z1.transpose(-1, -2) @ Z2)
    W2_init.sub_(coeff_chunk * W2_grad)
    Z2_bar = Z1_bar @ W2_init

    return W1_init, W1_grad, W2_init, W2_grad, Z2_bar



########## Triton ##########

###
## M2
###

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 4}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 16}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=8),

        triton.Config({'BLOCK_SIZE': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 16}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 1}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE': 4}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE': 8}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE': 16}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE': 32}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE': 64}, num_stages=5, num_warps=2),
    ]

def get_cuda_autotune_config_no_blk():
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
    # configs=get_cuda_autotune_config(),
    configs=get_cuda_autotune_config_no_blk(),
    key=['HF', 'HF_prime'],  # the two above configs will be evaluated anytime the value of key changes
    restore_value=['W1_init', 'W1_grad', 'W2_init', 'W2_grad'],
)
@triton.jit
def _m2_decode_kernel(W1_init, W1_grad,
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
    # if b_h == 0:
        # tmp = tl.tensor(Z1.shape, type=tl.float32)  # [bs,1,HF_prime]
        # tl.device_print('Z1: ', tmp)
        # tmp = tl.tensor(W1_init_data.shape, type=tl.float32)  # [bs,HF,HF_prime]
        # tl.device_print('W1_init_data: ', tmp)
        # tmp = tl.tensor(W2_init_data.shape, type=tl.float32)  # [bs,HF_prime,HF]
        # tl.device_print('W2_init_data ', tmp)

    Z2 = tl.sum(tl.reshape(Z1, shape=(BLOCK_SIZE, HF_prime, CS)) * W2_init_data, 1)[:, None, :]  # [bs,1,HF_prime].t @ [bs,HF_prime,HF]

    # if b_h == 0:
    #     tmp = tl.tensor(Z2.shape, type=tl.float32)  # [256, 1, 64]? should be [bs,1,64]
    #     tl.device_print('Z2', tmp)

    grad_l_wrt_Z2 = Z2 - XA_chunk  # [bs,1,HF]

    # if b_h == 0:
    #     tmp = tl.tensor(grad_l_wrt_Z2.shape, type=tl.float32)
    #     tl.device_print('grad_l_wrt_Z2', tmp)  # [bs, 1, 64]

    grad_l_wrt_Z1 = tl.sum(grad_l_wrt_Z2 * W2_init_data, 2)[:, None, :]  # [bs,1,HF] * [bs,HF_p,HF] -> [bs,HF_p] -> [bs,1,HF_p]

    # if b_h == 0:
        # tmp = tl.tensor(grad_l_wrt_Z1.shape, type=tl.float32)
        # tl.device_print('grad_l_wrt_Z1', tmp)  # [bs, 1, 256]
        # tmp = tl.reshape(XB_chunk, shape=(BLOCK_SIZE, HF, CS)) * grad_l_wrt_Z1
        # tmp = tl.tensor(tmp.shape, type=tl.float32)
        # tl.device_print('tmp', tmp)  # [bs, 64, 256]
        # tmp = tl.tensor(coeff_chunk.shape, type=tl.float32)  # [1,1,2]? shoudl be [bs,1,1]
        # tl.device_print('coeff', tmp)

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


def ttt_m2_triton_decode(XA, XB, XC, coeff, W1_init, W1_grad, W2_init, W2_grad):
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

    _m2_decode_kernel[grid](W1_init, W1_grad, W2_init, W2_grad,
                            XA, XB, XC, coeff,
                            output,

                            HF * HF_prime,     HF_prime,     1,  # strides for W1: [B*NH,HF,HF_prime]

                            HF_prime * HF,     HF,           1,  # strides for W2: [B*NH,HF_prime,HF]

                            CS * HF,           HF,           1,  # strides for ABCO, output: [B*NH,CS,HF]

                            CS=CS, HF=HF, HF_prime=HF_prime,

                            BLOCK_SIZE=1,  # Make sure that you don't re-define auto-tuned symbols
                            )

    return W1_init, W1_grad, W2_init, W2_grad, output



###
# M2 non-block
###
@triton.autotune(
    # configs=get_cuda_autotune_config(),
    configs=get_cuda_autotune_config_no_blk(),
    key=['HF', 'HF_prime'],  # the two above configs will be evaluated anytime the value of key changes
    restore_value=['W1_init', 'W1_grad', 'W2_init', 'W2_grad'],
)
@triton.jit
def _m2_decode_kernel_NB(W1_init, W1_grad,
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


    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))
    tl.store(W2_init, W2_init_data.to(W_dtype))
    tl.store(W2_grad, W2_grad_data.to(W_dtype))


def ttt_m2_triton_decode_NB(XA, XB, XC, coeff, W1_init, W1_grad, W2_init, W2_grad):
    B, NH, CS, HF = XA.shape
    HF_prime = W1_init.shape[-1]
    assert CS == 1
    assert W1_init.shape == W1_grad.shape == (B, NH, HF, HF_prime)
    assert W2_init.shape == W2_grad.shape == (B, NH, HF_prime, HF)
    assert XA.shape == XB.shape == XC.shape
    assert coeff.shape == (B, NH, CS, 1)

    output = torch.empty(size=(B, NH, CS, HF), device=W1_init.device, dtype=torch.float16)  # TODO FIX DTYPE
    grid = (B, NH, 1)

    _m2_decode_kernel_NB[grid](W1_init, W1_grad, W2_init, W2_grad,
                            XA, XB, XC, coeff,
                            output,

                            NH * HF * HF_prime,     HF * HF_prime,     HF_prime,    1,  # strides for W1: [B,NH,HF,HF_prime]

                            NH * HF_prime * HF,     HF_prime * HF,     HF,          1,  # strides for W2

                            NH * CS * HF,           CS * HF,           HF,          1,  # strides for ABCO, output

                            NH * CS * 1,            CS * 1,            1,           1,  # strides for coeff

                            CS=CS, HF=HF, HF_prime=HF_prime)

    return W1_init, W1_grad, W2_init, W2_grad, output



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        # x_vals=[2 ** i for i in range(5, 8)],  # different possible values for `x_name`
        x_vals=[1,2],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'M2 triton',
            # 'M1 triton',
            # 'M2 triton CG',
            # 'M1 triton CG',
        ],  # possible values for `line_arg``
        line_names=[
            'M2 triton',
            # 'M1 triton',
            # 'M2 triton CG',
            # 'M1 triton CG',
        ],  # label name for the lines
        styles=[('blue', '--'), ('green', '--'), ('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="decode time",  # name for the plot. Used also as a file name for saving the plot.
        args={'BS': 64, 'NH': 32, 'CS': 1, 'HF': 64, 'HF_prime': int(EXPAND * 64)},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_decode_cg_or_not(N, BS, NH, CS, HF, HF_prime, provider):
    assert CS == 1

    quantiles = [0.5, 0.2, 0.8]
    n_warmups = 2

    input_dtype = torch.float16

    XA = torch.randn(N, BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(N, BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(N, BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(N, BS * NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02

    XA_holder = torch.zeros_like(XA[0])
    XB_holder = torch.zeros_like(XB[0])
    XC_holder = torch.zeros_like(XC[0])
    coeff_holder = torch.zeros_like(coeff[0])

    if provider == 'M2 triton':
        W1 = torch.randn(BS * NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
        W1_grad = torch.randn_like(W1) * 0.02

        W2 = torch.randn(BS * NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
        W2_grad = torch.randn_like(W2) * 0.02

        def loop(decode, W1, W1_grad, W2, W2_grad):
            for i in range(N):
                W1, W1_grad, W2, W2_grad, _ = decode(XA[i], XB[i], XC[i], coeff[i], W1, W1_grad, W2, W2_grad)

        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(ttt_m2_triton_decode,
                                                                  W1, W1_grad, W2, W2_grad),
                                                     quantiles=quantiles)

    elif provider == 'M2 triton CG':
        W1 = torch.randn(BS * NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
        W1_grad = torch.randn_like(W1) * 0.02

        W2 = torch.randn(BS * NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
        W2_grad = torch.randn_like(W2) * 0.02

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                W1_tmp, W1_grad_tmp, \
                W2_tmp, W2_grad_tmp, Z_tmp = ttt_m2_triton_decode(XA_holder, XB_holder, XC_holder, coeff_holder,
                                                                  W1, W1_grad, W2, W2_grad)
            s.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        mempool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(graph, pool=mempool):
            W1_tmp, W1_grad_tmp, \
            W2_tmp, W2_grad_tmp, Z_tmp = ttt_m2_triton_decode(XA_holder, XB_holder, XC_holder, coeff_holder,
                                                              W1, W1_grad, W2, W2_grad)

        def run(new_XA, new_XB, new_XC, new_coeff):
            XA_holder.copy_(new_XA)
            XB_holder.copy_(new_XB)
            XC_holder.copy_(new_XC)
            coeff_holder.copy_(new_coeff)
            graph.replay()
            return Z_tmp.clone()

        def loop():
            for i in range(N):
                Z_tmp = run(XA[i], XB[i], XC[i], coeff[i])

        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(), quantiles=quantiles)


    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[1,7,],
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            # 'M2 triton Non-BLOCK',
            'M2 triton BLOCK',
        ],  # possible values for `line_arg``
        line_names=[
            # 'M2 triton Non-BLOCK',
            'M2 triton BLOCK',
        ],  # label name for the lines
        styles=[('blue', '--'), ('blue', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="decode time",  # name for the plot. Used also as a file name for saving the plot.
        args={'BS': 64, 'NH': 32, 'CS': 1, 'HF': 64, 'HF_prime': int(EXPAND * 64)},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_decode_blk_or_not(N, BS, NH, CS, HF, HF_prime, provider):
    assert CS == 1

    quantiles = [0.5, 0.2, 0.8]

    input_dtype = torch.float16

    XA = torch.randn(N, BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(N, BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(N, BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(N, BS * NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02

    W1 = torch.randn(BS * NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
    W1_grad = torch.randn_like(W1) * 0.02

    W2 = torch.randn(BS * NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
    W2_grad = torch.randn_like(W2) * 0.02

    def loop(decode, W1, W1_grad, W2, W2_grad):
        for i in range(N):
            W1, W1_grad, W2, W2_grad, _ = decode(XA[i], XB[i], XC[i], coeff[i], W1, W1_grad, W2, W2_grad)

    if provider == 'M2 triton Non-BLOCK':
        XA = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        XB = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        XC = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        coeff = torch.randn(N, BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02

        W1 = torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
        W1_grad = torch.randn_like(W1) * 0.02

        W2 = torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
        W2_grad = torch.randn_like(W2) * 0.02

        def loop(decode, W1, W1_grad, W2, W2_grad):
            for i in range(N):
                W1, W1_grad, W2, W2_grad, _ = decode(XA[i], XB[i], XC[i], coeff[i], W1, W1_grad, W2, W2_grad)

        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(ttt_m2_triton_decode_NB,
                                                                  W1, W1_grad, W2, W2_grad),
                                                     quantiles=quantiles)

    elif provider == 'M2 triton BLOCK':

        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(ttt_m2_triton_decode,
                                                                  W1, W1_grad, W2, W2_grad),
                                                     quantiles=quantiles)
    else:
        raise NotImplementedError

    return ms, min_ms, max_ms




if __name__ == "__main__":
    # os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

    print('========== Timing ==========')
    # benchmark_decode_cg_or_not.run(show_plots=False, print_data=True)
    benchmark_decode_blk_or_not.run(show_plots=False, print_data=True)

    '''
    print('========== M2 Matching outputs abs diff ==========')
    input_dtype = torch.float16
    BS, NH, CS, HF, HF_prime = 64, 32, 1, 64, 4 * 64
    W1 = torch.randn(BS * NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
    W1_grad = torch.randn_like(W1) * 0.02
    W1_original = W1.clone()
    W1_grad_original = W1_grad.clone()
    W1_nb = W1.clone().reshape(BS, NH, *W1.shape[1:])
    W1_grad_nb = W1_grad.clone().reshape(BS, NH, *W1_grad.shape[1:])

    W2 = torch.randn(BS * NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
    W2_grad = torch.randn_like(W2) * 0.02
    W2_original = W2.clone()
    W2_grad_original = W2_grad.clone()
    W2_nb = W2.clone().reshape(BS, NH, *W2.shape[1:])
    W2_grad_nb = W2_grad.clone().reshape(BS, NH, *W2_grad.shape[1:])

    XA = torch.randn(BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(BS * NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(BS * NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02

    W1, W1_grad, \
    W2, W2_grad, \
    XCW_batch = ttt_m2_decode(XA, XB, XC, coeff, W1, W1_grad, W2, W2_grad)

    W1_triton_nb, W1_grad_triton_nb, \
    W2_triton_nb, W2_grad_triton_nb, \
    XCW_batch_triton_nb = ttt_m2_triton_decode_NB(XA.reshape(BS, NH, CS, HF),
                                                  XB.reshape(BS, NH, CS, HF),
                                                  XC.reshape(BS, NH, CS, HF),
                                                  coeff.reshape(BS, NH, CS, 1),
                                                  W1_nb, W1_grad_nb, W2_nb, W2_grad_nb)
    W1_triton_nb = W1_triton_nb.reshape(-1, *W1_triton_nb.shape[2:])
    W1_grad_triton_nb = W1_grad_triton_nb.reshape(-1, *W1_grad_triton_nb.shape[2:])
    W2_triton_nb = W2_triton_nb.reshape(-1, *W2_triton_nb.shape[2:])
    W2_grad_triton_nb = W2_grad_triton_nb.reshape(-1, *W2_grad_triton_nb.shape[2:])
    XCW_batch_triton_nb = XCW_batch_triton_nb.reshape(-1, *XCW_batch_triton_nb.shape[2:])


    W1_triton, W1_grad_triton, \
    W2_triton, W2_grad_triton, \
    XCW_batch_triton = ttt_m2_triton_decode(XA, XB, XC, coeff,
                                            W1_original, W1_grad_original, W2_original, W2_grad_original)


    print('============= Triton Block v.s Pytorch =============')
    print('W1 diff: ' + str(torch.abs(W1 - W1_triton).max()))
    print('W1_grad diff: ' + str(torch.abs(W1_grad - W1_grad_triton).max()))
    print('W2 diff: ' + str(torch.abs(W2 - W2_triton).max()))
    print('W2_grad diff: ' + str(torch.abs(W2_grad - W2_grad_triton).max()))
    print('Output diff: ' + str(torch.abs(XCW_batch - XCW_batch_triton).max()))

    # print('============= Triton Block v.s Triton Non-Block =============')
    # print('W1 diff: ' + str(torch.abs(W1_triton_nb - W1_triton).max()))
    # print('W1_grad diff: ' + str(torch.abs(W1_grad_triton_nb - W1_grad_triton).max()))
    # print('W2 diff: ' + str(torch.abs(W2_triton_nb - W2_triton).max()))
    # print('W2_grad diff: ' + str(torch.abs(W2_grad_triton_nb - W2_grad_triton).max()))
    # print('Output diff: ' + str(torch.abs(XCW_batch_triton_nb - XCW_batch_triton).max()))

    # print('============= Triton Non-Block v.s Pytorch =============')
    # print('W1 diff: ' + str(torch.abs(W1 - W1_triton_nb).max()))
    # print('W1_grad diff: ' + str(torch.abs(W1_grad - W1_grad_triton_nb).max()))
    # print('W2 diff: ' + str(torch.abs(W2 - W2_triton_nb).max()))
    # print('W2_grad diff: ' + str(torch.abs(W2_grad - W2_grad_triton_nb).max()))
    # print('Output diff: ' + str(torch.abs(XCW_batch - XCW_batch_triton_nb).max()))
    '''