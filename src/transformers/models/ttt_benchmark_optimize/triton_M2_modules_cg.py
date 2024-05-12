import sys
import pdb

import torch
import einops
import triton
import triton.language as tl
import os

EXPAND = 1

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
        XA, XB, XC: [B,NH,CS,HF]
        coeff: [B,NH,CS,1]
        W1_init, W1_grad: [B,NH,HF,HF_prime]
        W2_init, W2_grad: [B,NH,HF_prime,HF]

    Returns:
        Z2_bar: [B,NH,CS,HF]
    """
    Z1 = XB_chunk @ W1_init  # [B,nh,K=1,f] @ [B,nh,f,f] -> [B,nh,K=1,f]
    Z2 = Z1 @ W2_init

    # grad_l_wrt_Z2 = Z2 - XA_chunk
    Z2.sub_(XA_chunk)  # grad_l_wrt_Z2
    # grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-1, -2)
    grad_l_wrt_Z1 = Z2 @ W2_init.transpose(-1, -2)

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)
    W1_init.sub_(coeff_chunk * W1_grad)
    Z1_bar = XC_chunk @ W1_init

    # W2_grad.add_(Z1.transpose(-1, -2) @ grad_l_wrt_Z2)
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

    Z1 = tl.sum(tl.trans(XB_chunk) * W1_init_data, 0) - XA_chunk # [HF,] - [1,HF] - > [1,HF]
    W1_grad_data += tl.trans(XB_chunk) * Z1  # @xinhao: tl.dot(tl.trans(XB_chunk), Z1) doesn't support [HF,1] @ [1,HF]
    W1_init_data -= coeff_chunk * W1_grad_data
    Z1_bar = tl.sum(tl.trans(XC_chunk) * W1_init_data, 0)

    tl.store(Out_chunk, Z1_bar.to(O_dtype))
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


################### Benchmark M2 v.s M1, Pytorch v.s Triton ###########################
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(0, 12)],  # different possible values for `x_name`
        # x_vals=[128, 256],
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'M2 torch-native CG',
            'M1 torch-native CG',
            'M2 triton CG',
            'M1 triton CG',
        ],  # possible values for `line_arg``
        line_names=[
            'M2 torch-native CG',
            'M1 torch-native CG',
            'M2 triton CG',
            'M1 triton CG',
        ],  # label name for the lines
        styles=[('blue', '--'), ('green', '--'), ('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="decode time",  # name for the plot. Used also as a file name for saving the plot.
        args={'BS': 64, 'NH': 32, 'CS': 1, 'HF': 64, 'HF_prime': int(EXPAND * 64)},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_decode_cg(N, BS, NH, CS, HF, HF_prime, provider):
    assert CS == 1

    quantiles = [0.5, 0.2, 0.8]
    n_warmups = 2

    input_dtype = torch.float16

    XA = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(N, BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02

    XA_holder = torch.zeros_like(XA[0])
    XB_holder = torch.zeros_like(XB[0])
    XC_holder = torch.zeros_like(XC[0])
    coeff_holder = torch.zeros_like(coeff[0])

    if 'M2' in provider:
        if 'triton' in provider:
            decode_fn = ttt_m2_triton_decode
        else:
            decode_fn = ttt_m2_decode

        W1 = torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
        W1_grad = torch.randn_like(W1) * 0.02

        W2 = torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
        W2_grad = torch.randn_like(W2) * 0.02

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                W1_tmp, W1_grad_tmp, \
                W2_tmp, W2_grad_tmp, Z_tmp = decode_fn(XA_holder, XB_holder, XC_holder, coeff_holder,
                                                       W1, W1_grad, W2, W2_grad)
            s.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        mempool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(graph, pool=mempool):
            W1_tmp, W1_grad_tmp, \
            W2_tmp, W2_grad_tmp, Z_tmp = decode_fn(XA_holder, XB_holder, XC_holder, coeff_holder,
                                                   W1, W1_grad, W2, W2_grad)

    else:
        if 'triton' in provider:
            decode_fn = ttt_m1_triton_decode
        else:
            decode_fn = ttt_m1_decode

        W1 = torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
        W1_grad = torch.randn_like(W1) * 0.02

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                W1_tmp, W1_grad_tmp, Z_tmp = decode_fn(XA_holder, XB_holder, XC_holder, coeff_holder,
                                                       W1, W1_grad)
            s.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        mempool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(graph, pool=mempool):
            W1_tmp, W1_grad_tmp, Z_tmp = decode_fn(XA_holder, XB_holder, XC_holder, coeff_holder,
                                                   W1, W1_grad)

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
        x_vals=[2 ** i for i in range(0, 12)],  # different possible values for `x_name`
        # x_vals=[128, 256],
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'M2 torch-native',
            'M1 torch-native',
            'M2 triton',
            'M1 triton',
        ],  # possible values for `line_arg``
        line_names=[
            'M2 torch-native',
            'M1 torch-native',
            'M2 triton',
            'M1 triton',
        ],  # label name for the lines
        styles=[('blue', '--'), ('green', '--'), ('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="decode time",  # name for the plot. Used also as a file name for saving the plot.
        args={'BS': 64, 'NH': 32, 'CS': 1, 'HF': 64, 'HF_prime': int(EXPAND * 64)},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_decode(N, BS, NH, CS, HF, HF_prime, provider):
    assert CS == 1

    quantiles = [0.5, 0.2, 0.8]

    input_dtype = torch.float16

    XA = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(N, BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02

    if 'M2' in provider:
        W1 = torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
        W1_grad = torch.randn_like(W1) * 0.02

        W2 = torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
        W2_grad = torch.randn_like(W2) * 0.02

        def loop(decode, W1, W1_grad, W2, W2_grad):
            for i in range(N):
                W1, W1_grad, W2, W2_grad, _ = decode(XA[i], XB[i], XC[i], coeff[i], W1, W1_grad, W2, W2_grad)
    else:
        W1 = torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
        W1_grad = torch.randn_like(W1) * 0.02

        def loop(decode, W1, W1_grad):
            for i in range(N):
                W1, W1_grad, _ = decode(XA[i], XB[i], XC[i], coeff[i], W1, W1_grad)

    # if provider == 'M2 torch-native':
    #     ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: loop(ttt_m2_decode, W1, W1_grad, W2, W2_grad))
    # elif provider == 'M1 torch-native':
    #     ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: loop(ttt_m1_decode, W1, W1_grad))
    # elif provider == 'M2 triton':
    #     ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: loop(ttt_m2_triton_decode, W1, W1_grad, W2, W2_grad))
    # elif provider == 'M1 triton':
    #     ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: loop(ttt_m1_triton_decode, W1, W1_grad))
    # else:
    #     raise NotImplementedError

    if provider == 'M2 torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(ttt_m2_decode, W1, W1_grad, W2, W2_grad),
                                                     quantiles=quantiles)
    elif provider == 'M1 torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(ttt_m1_decode, W1, W1_grad),
                                                     quantiles=quantiles)
    elif provider == 'M2 triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(ttt_m2_triton_decode, W1, W1_grad, W2, W2_grad),
                                                     quantiles=quantiles)
    elif provider == 'M1 triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(ttt_m1_triton_decode, W1, W1_grad),
                                                     quantiles=quantiles)
    else:
        raise NotImplementedError

    return ms, min_ms, max_ms



if __name__ == "__main__":
    # os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

    print('========== Timing ==========')
    # benchmark_decode.run(show_plots=False, print_data=True)
    benchmark_decode_cg.run(show_plots=False, print_data=True)
