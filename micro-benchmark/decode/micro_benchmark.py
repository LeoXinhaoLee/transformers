import pdb

import sys
import os

import argparse
import torch
import triton
import triton.language as tl
from nvtx_do_bench import do_bench  # @xinhao: support nvtx tag
from micro_decode_modules import pt_m2_decode_non_end_chunk, triton_m2_decode_non_end_chunk

import m2_decode_cpp


EXPAND = 4

parser = argparse.ArgumentParser(description="Micro benchmarking")
parser.add_argument("--profile", action='store_true')  # @xinhao: launching by `nsys profile` will affect time
args = parser.parse_args()                             # @xinhao: but if not, even if specify --profile, time won't be affected


################### Benchmark M2 v.s M1, Pytorch v.s Triton v.s Triton CG ###########################

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        # x_vals=[16, 64],  # different possible values for `x_name`
        x_vals=[4, ],  # Nsight compute
        # x_vals=[16,],  # real
        line_arg='provider',
        line_vals=[
            # 'M2 torch-native',
            # 'M2 triton',
            # 'M2 triton CG',
            'M2 cuda',
            'M2 cuda CG',
        ],
        line_names=[
            # 'M2 torch-native',
            # 'M2 triton',
            # 'M2 triton CG',
            'M2 cuda',
            'M2 cuda CG',
        ],
        styles=[('blue', '-'), ('blue', '--'), ('blue', ':'),
                ('green', '--'), ('green', ':')],
        ylabel="ms",
        plot_name="decode N token time",
        args={'BS': 512, 'NH': 32, 'CS': 1, 'HF': 64, 'HF_prime': int(EXPAND * 64), 'cuda_transpose': ''},
    )
)
def benchmark_decode(N, BS, NH, CS, HF, HF_prime, cuda_transpose, provider):
    assert CS == 1

    quantiles = [0.5, 0.2, 0.8]
    n_warmups = 2

    input_dtype = torch.float16

    W1 = torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
    W1_grad = torch.randn_like(W1) * 0.02
    W2 = torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
    W2_grad = torch.randn_like(W2) * 0.02

    XA = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(N, BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02
    XA_holder, XB_holder, XC_holder, coeff_holder = map(lambda x: x[0], [XA, XB, XC, coeff])

    if 'cuda' in provider:
        XA, XB, XC, coeff = map(lambda x: x.squeeze(3), [XA, XB, XC, coeff])
        XA_holder, XB_holder, XC_holder, coeff_holder = map(lambda x: x.squeeze(2),
                                                            [XA_holder, XB_holder, XC_holder, coeff_holder])
        # @xinhao: transpose such that cuda read W along row
        if cuda_transpose == 'W12':
            W1 = W1.transpose(-1,-2).contiguous()
            W1_grad = W1_grad.transpose(-1,-2).contiguous()
            W2 = W2.transpose(-1,-2).contiguous()
            W2_grad = W2_grad.transpose(-1,-2).contiguous()
        elif cuda_transpose == 'W1':
            W1 = W1.transpose(-1,-2).contiguous()
            W1_grad = W1_grad.transpose(-1,-2).contiguous()


    if provider == 'M2 torch-native' or provider == 'M2 triton' or provider == 'M2 cuda':
        def loop(decode_fn):
            for i in range(N):
                Z_tmp = decode_fn(XA[i], XB[i], XC[i], coeff[i], W1, W1_grad, W2, W2_grad)

    elif 'CG' in provider:
        if provider == 'M2 triton CG':
            decode_fn = triton_m2_decode_non_end_chunk
        elif provider == 'M2 cuda CG':
            decode_fn = m2_decode_cpp.decode
        else:
            raise NotImplementedError

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                Z_tmp = decode_fn(XA_holder, XB_holder, XC_holder, coeff_holder, W1, W1_grad, W2, W2_grad)
            s.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        mempool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(graph, pool=mempool):
            Z_tmp = decode_fn(XA_holder, XB_holder, XC_holder, coeff_holder, W1, W1_grad, W2, W2_grad)

        def run(new_XA, new_XB, new_XC, new_coeff):
            XA_holder.copy_(new_XA)
            XB_holder.copy_(new_XB)
            XC_holder.copy_(new_XC)
            coeff_holder.copy_(new_coeff)
            graph.replay()
            return Z_tmp.clone()

        def loop_m2_cg():
            for i in range(N):
                Z_tmp = run(XA[i], XB[i], XC[i], coeff[i])

    else:
        NotImplementedError


    if provider == 'M2 torch-native':
        ms, min_ms, max_ms = do_bench(lambda: loop(pt_m2_decode_non_end_chunk),
                                      quantiles=quantiles, profile=args.profile)

    elif provider == 'M2 triton':
        ms, min_ms, max_ms = do_bench(lambda: loop(triton_m2_decode_non_end_chunk),
                                      quantiles=quantiles, profile=args.profile)

    elif provider == 'M2 cuda':
        ms, min_ms, max_ms = do_bench(lambda: loop(m2_decode_cpp.decode),  # non_end_chunk
                                      quantiles=quantiles, profile=args.profile)

    elif provider == 'M2 triton CG' or provider == 'M2 cuda CG':
        ms, min_ms, max_ms = do_bench(lambda: loop_m2_cg(),
                                      quantiles=quantiles, profile=args.profile)

    else:
        raise NotImplementedError

    return ms, min_ms, max_ms


if __name__ == "__main__":
    print('========== Timing ==========')
    benchmark_decode.run(show_plots=False, print_data=True)
