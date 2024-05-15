import pdb

import torch
import einops
import triton
import triton.language as tl


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
)
@triton.jit
def _m1_kernel(W1, XA, XB, XC, coeff_last, coeff, Out,
               stride_ab, stride_ah, stride_an, stride_ac, stride_af,
               stride_eb, stride_eh, stride_en, stride_ec,
               stride_pb, stride_ph, stride_pn,
               stride_wb, stride_wh, stride_wf, stride_wd,
               CS: tl.constexpr, HF: tl.constexpr,
               N_CHUNK: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    abco_offset = batch * stride_ab + head * stride_ah
    w_offset = batch * stride_wb + head * stride_wh
    coeff_offset = batch * stride_eb + head * stride_eh
    coeff_last_offset = batch * stride_pb + head * stride_ph

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)
    XA = XA + abco_offset
    XB = XB + abco_offset
    XC = XC + abco_offset
    Out = Out + abco_offset
    W1_data = tl.load(W1 + w_offset + rf[:, None] * stride_wf + rf[None, :] * stride_wd)
    coeff = coeff + coeff_offset
    coeff_last = coeff_last + coeff_last_offset
    for i in range(N_CHUNK):
        local_abco_offset = i * stride_an
        local_coeff_offset = i * stride_en
        local_coeff_last_offset = i * stride_pn
        XA_chunk = tl.load(XA + local_abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af))
        XB_chunk = tl.load(XB + local_abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af))
        XC_chunk = tl.load(XC + local_abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af))
        coeff_chunk = tl.load(coeff + local_coeff_offset + rc * stride_ec)
        coeff_chunk_last = tl.load(coeff_last + local_coeff_last_offset)

        Z1 = tl.dot(XB_chunk, W1_data) - XA_chunk
        mask = rc[:, None] >= rc[None, :]
        Attn1_full = tl.dot(XC_chunk, tl.trans(XB_chunk))
        Attn1 = tl.where(mask, Attn1_full, 0)
        Z1_bar = tl.dot(XC_chunk, W1_data) - tl.dot((coeff_chunk[:, None] * Attn1), Z1)
        W1_data -= tl.dot(tl.trans(coeff_chunk_last * XB_chunk).to(Z1.dtype), Z1).to(W1.type.element_ty)
        Out_chunk = Out + local_abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af)
        tl.store(Out_chunk, Z1_bar.to(Out.type.element_ty))


def m1_forward(W1, XA, XB, XC, coeff_last, coeff):
    B, NH, NC, CS, HF = XA.shape
    assert coeff.shape == (B, NH, NC, CS)
    coeff_last = coeff[..., -1:].squeeze(-1)
    grid = (B, NH, 1)
    output = torch.empty(size=(B, NH, NC, CS, HF), device=W1.device, dtype=input_dtype)
    W1_expand = torch.tile(W1, dims=(B, 1, 1, 1))
    _m1_kernel[grid](W1_expand,  # [B,nh,f,f], cloned from W1, safe for in-place op
                     XA, XB, XC, coeff_last, coeff, output,
                     NH * NC * CS * HF, NC * CS * HF, CS * HF, HF, 1,  # strides for A,B,C,O
                     NH * NC * CS, NC * CS, CS, 1,  # strides for E
                     NH * NC, NC, 1,  # strides for last coeff
                     NH * HF * HF, HF * HF, HF, 1,  # strides for W1
                     CS, HF,
                     NC
                     )
    return W1_expand, output


def ttt_m1_triton_forward(XA, XB, XC, coeff, W1):
    B, NH, NC, CS, HF = XA.shape
    coeff = coeff.squeeze(-1)
    coeff_last = coeff[..., -1:]
    W1, output = m1_forward(W1, XA, XB, XC, coeff_last, coeff)
    output = einops.rearrange(output, "b nh nc cs f -> b (nc cs) (nh f)")
    W1 = einops.rearrange(W1, "b nh f d -> (b nh) f d")
    return W1, output


def for_loop(W1, XA, XB, XC, coeff_last, coeff):
    NC, B_times_NH, CS, HF = XA.shape
    output_tensor = torch.empty(size=(NC, B_times_NH, CS, HF), device=W1.device, dtype=input_dtype)
    W1_init = W1
    for i in range(NC):
        XA_chunk = XA[i]
        XB_chunk = XB[i]
        XC_chunk = XC[i]
        coeff_chunk_last = coeff_last[i]
        coeff_chunk = coeff[i]
        Z1 = (XB_chunk @ W1_init).sub_(XA_chunk)  # [B*nh,K,f] @ [B*nh,f,f] -> [B*nh,K,f]
        Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))  # [B*nh,K,K]
        Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ Z1  # [B*nh,K,f] @ [B*nh,f,f] - ([B*nh,K,1] * [B*nh,K,K]) @ [B*nh,K,f]
        W1_init.sub_((coeff_chunk_last * XB_chunk).transpose(-1, -2) @ Z1)
        output_tensor[i] = Z1_bar
    return W1_init, output_tensor


def ttt_m1_forward(XA, XB, XC, coeff, W1):
    B, NH, NC, CS, HF = XA.shape
    coeff_last = coeff[..., -1:, :]
    XA = XA.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, CS, -1)
    XB = XB.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, CS, -1)
    XC = XC.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, CS, -1)
    coeff = coeff.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, CS, -1)
    coeff_last = coeff_last.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, 1, 1)
    W1, XCW_batch = for_loop(
        torch.tile(W1, dims=(B, 1, 1)),  # [B*nh,f,f], cloned from W1, safe for in-place op
        XA, XB, XC, coeff_last, coeff
    )
    XCW_batch = einops.rearrange(XCW_batch, "nc (b nh) cs f -> b (nc cs) (nh f)", b=B, nh=NH)
    return W1, XCW_batch


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(10, 15)],  # different possible values for `x_name`
        # x_vals=[1024, 2048, 4096],
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'torch-native',
            # 'torch-jit',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch (native)",
            # "Torch (jit)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="M1 forward time",  # name for the plot. Used also as a file name for saving the plot.
        args={'BS': 64, 'NH': 32, 'CS': 16, 'HF': 64},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(BS, N, CS, NH, HF, provider):

    input_dtype = torch.float16
    W1 = torch.randn(NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
    W1_original = W1.clone()
    NC = N // CS
    XA = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(BS, NH, NC, CS, 1, device='cuda', dtype=input_dtype) * 0.02

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ttt_m1_forward(XA, XB, XC, coeff, W1),
                                                     quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ttt_m1_triton_forward(XA, XB, XC, coeff, W1_original),
                                                     quantiles=quantiles)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    L = 1024
    BS, NH, NC, CS, HF = 16, 32, L // 16, 16, 64
    input_dtype = torch.float16
    W1 = torch.randn(NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
    W1_original = W1.clone()
    XA = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(BS, NH, NC, CS, 1, device='cuda', dtype=input_dtype) * 0.02
    W1, XCW_batch = ttt_m1_forward(XA, XB, XC, coeff, W1)
    print(W1.shape, XCW_batch.shape)
    W1_triton, XCW_batch_triton = ttt_m1_triton_forward(XA, XB, XC, coeff, W1_original)
    print(W1_triton.shape, XCW_batch_triton.shape)

    print('========== Matching outputs abs diff ==========')
    print('W1 diff: ' + str(torch.abs(W1 - W1_triton).max()))
    print('Output diff: ' + str(torch.abs(XCW_batch - XCW_batch_triton).max()))

    print('========== Timing ==========')
    benchmark.run(show_plots=False, print_data=True)

