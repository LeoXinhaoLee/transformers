import copy
import pdb

import sys
import os

import argparse
import torch
import einops
import triton
import triton.language as tl
from nvtx_do_bench import do_bench  # @xinhao: support nvtx tag
from micro_decode_modules import pt_m1_prefill, pt_m2_prefill, triton_m1_prefill, triton_m2_prefill

EXPAND = 4

parser = argparse.ArgumentParser(description="Micro benchmarking")
parser.add_argument("--profile", action='store_true')  # @xinhao: launching by `nsys profile` will affect time
args = parser.parse_args()                             # @xinhao: but if not, even if specify --profile, time won't be affected


################### Benchmark M2 v.s M1, Pytorch v.s Triton v.s Triton CG ###########################

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['NC'],  # number of chunks
        x_vals=[64, ],  # different possible values for `x_name`
        # x_vals=[4, ],  # Nsight compute
        line_arg='provider',
        line_vals=[
            'M1 torch-native',
            'M1 triton',
            'M1 triton CG',
            'M2 torch-native',
            'M2 triton',
            'M2 triton CG',
        ],
        line_names=[
            'M1 torch-native',
            'M1 triton',
            'M1 triton CG',
            'M2 torch-native',
            'M2 triton',
            'M2 triton CG',
        ],
        styles=[('blue', '-'), ('blue', '--'), ('blue', ':'),
                ('green', '-'), ('green', '--'), ('green', ':')],
        ylabel="ms",
        plot_name="Prefill NC chunks time",
        args={'BS': 128, 'NH': 32, 'CS': 16, 'HF': 64, 'HF_prime': int(EXPAND * 64)},
    )
)
def benchmark_decode(NC, BS, NH, CS, HF, HF_prime, provider):
    assert CS == 16

    quantiles = [0.5, 0.2, 0.8]
    n_warmups = 2

    input_dtype = torch.float16

    if 'M1' in provider:
        HF_prime = HF

    state_dict = {
        'W1': torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02,
        'W2': torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02,
    }
    input_dict = {
        'XA': torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'XB': torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'XC': torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'coeff': torch.randn(BS, NH, NC, CS, 1, device='cuda', dtype=input_dtype) * 0.02,
        'coeff_last': torch.randn(BS, NH, NC, 1, 1, device='cuda', dtype=input_dtype) * 0.02,
    }

    if 'torch' in provider:
        for k in state_dict.keys():
            state_dict[k] = einops.rearrange(state_dict[k], 'b nh f1 f2 -> (b nh) f1 f2')
        for k in input_dict.keys():
            input_dict[k] = einops.rearrange(input_dict[k], 'b nh nc cs f -> nc (b nh) cs f')

    state_dict_holder = copy.deepcopy(state_dict)
    input_dict_holder = copy.deepcopy(input_dict)

    if provider == 'M1 torch-native':
        prefill_fn = pt_m1_prefill
    elif provider == 'M2 torch-native':
        prefill_fn = pt_m2_prefill
    elif provider == 'M1 triton':
        prefill_fn = triton_m1_prefill
    elif provider == 'M2 triton':
        prefill_fn = triton_m2_prefill

    elif 'CG' in provider:
        if provider == 'M1 triton CG':
            base_fn = triton_m1_prefill
        elif provider == 'M2 triton CG':
            base_fn = triton_m2_prefill
        else:
            raise NotImplementedError

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                Z_tmp = base_fn(**state_dict_holder, **input_dict_holder)
            s.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        mempool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(graph, pool=mempool):
            Z_tmp = base_fn(**state_dict_holder, **input_dict_holder)

        def run(XA, XB, XC, coeff, coeff_last, **kwargs):
            input_dict_holder['XA'].copy_(XA)
            input_dict_holder['XB'].copy_(XB)
            input_dict_holder['XC'].copy_(XC)
            input_dict_holder['coeff'].copy_(coeff)
            input_dict_holder['coeff_last'].copy_(coeff_last)
            graph.replay()
            return Z_tmp.clone()

        prefill_fn = run

    else:
        raise NotImplementedError


    ms, min_ms, max_ms = do_bench(lambda: prefill_fn(**state_dict, **input_dict),
                                  quantiles=quantiles, profile=args.profile)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    print('========== Timing ==========')
    benchmark_decode.run(show_plots=False, print_data=True)
