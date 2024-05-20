import pdb

import sys
import os

import argparse
import torch
import triton
import triton.language as tl
# from triton.testing import do_bench
from transformers.models.ttt_benchmark_decode_optimize.nvtx_do_bench import do_bench  # @xinhao: support nvtx tag

from transformers.models.ttt_benchmark_decode_cuda.micro_decode_modules import pt_m2_decode_non_end_chunk, \
    triton_m2_decode_non_end_chunk

import m2_decode_cpp


EXPAND = 4

parser = argparse.ArgumentParser(description="Micro benchmarking")
parser.add_argument("--profile", action='store_true')  # @xinhao: launching by `nsys profile` will affect time
args = parser.parse_args()                             # @xinhao: but if not, even if specify --profile, time won't be affected


################### Benchmark M2 v.s M1, Pytorch v.s Triton v.s Triton CG ###########################

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        # x_vals=[2 ** i for i in range(0, 12)],  # different possible values for `x_name`
        x_vals=[16, 64, 128],
        line_arg='provider',
        line_vals=[
            'M2 torch-native',
            'M2 triton',
            'M2 triton CG',
            'M2 cuda',
            'M2 cuda CG',
        ],
        line_names=[
            'M2 torch-native',
            'M2 triton',
            'M2 triton CG',
            'M2 cuda',
            'M2 cuda CG',
        ],
        styles=[('blue', '-'), ('blue', '--'), ('blue', ':'),
                ('green', '-'), ('green', '--')],
        ylabel="ms",
        plot_name="decode N token time",
        args={'BS': 64, 'NH': 32, 'CS': 1, 'HF': 64, 'HF_prime': int(EXPAND * 64)},
    )
)
def benchmark_decode(N, BS, NH, CS, HF, HF_prime, provider):
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
    # os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

    ############### M2 Matching outputs abs diff ###############

    # BS, NH, CS, HF, HF_prime = 64, 32, 1, 64, 4 * 64
    # W1 = torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
    # W1_grad = torch.randn_like(W1) * 0.02
    # W1_original = W1.clone()
    # W1_grad_original = W1_grad.clone()
    #
    # W2 = torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
    # W2_grad = torch.randn_like(W2) * 0.02
    # W2_original = W2.clone()
    # W2_grad_original = W2_grad.clone()
    #
    # XA = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    # XB = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    # XC = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    # coeff = torch.randn(BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02
    #
    # W1, W1_grad, \
    # W2, W2_grad, \
    # XCW_batch = ttt_m2_decode(XA, XB, XC, coeff, W1, W1_grad, W2, W2_grad)
    #
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

    # BS, NH, CS, HF = 64, 32, 1, 64
    # W1 = torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
    # W1_grad = torch.randn_like(W1) * 0.02
    # W1_original = W1.clone()
    # W1_grad_original = W1_grad.clone()
    #
    # XA = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    # XB = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    # XC = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    # coeff = torch.randn(BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02
    #
    # W1, W1_grad, \
    # XCW_batch = ttt_m1_decode(XA, XB, XC, coeff, W1, W1_grad)
    #
    # W1_triton, W1_grad_triton, \
    # XCW_batch_triton = ttt_m1_triton_decode(XA, XB, XC, coeff, W1_original, W1_grad_original)
    #
    # print('========== M1 Matching outputs abs diff ==========')
    # print('W1 diff: ' + str(torch.abs(W1 - W1_triton).max()))
    # print('W1_grad diff: ' + str(torch.abs(W1_grad - W1_grad_triton).max()))
    # print('Output diff: ' + str(torch.abs(XCW_batch - XCW_batch_triton).max()))

    print('========== Timing ==========')
    benchmark_decode.run(show_plots=False, print_data=True)
