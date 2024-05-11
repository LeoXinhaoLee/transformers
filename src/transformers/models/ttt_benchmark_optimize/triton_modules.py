import sys
import pdb

import torch
import einops
import triton
import triton.language as tl


def clean_tensor(tensor):
    tensor[torch.isnan(tensor) | (tensor == float('inf')) | (tensor == float('-inf'))] = 0
    return tensor


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
#     key=['N_CHUNK'],
# )
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
    # if batch == 0 and head == 0:
    #     # temp = W1_data.shape
    #     # temp = XB_chunk.shape
    #     temp = tl.tensor(W1_data.shape, type=tl.float32)
    #     tl.device_print("w1: ", temp)
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

        # if batch == 0 and head == 0:
        #     temp = tl.tensor(W1_data.shape, type=tl.float32)
        #     tl.device_print("gg: ", temp)

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
        Z1_bar = XC_chunk @ W1_init - (
                    coeff_chunk * Attn1) @ Z1  # [B*nh,K,f] @ [B*nh,f,f] - ([B*nh,K,1] * [B*nh,K,K]) @ [B*nh,K,f]
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
#         triton.Config({}, num_warps=8),
#         triton.Config({}, num_warps=4),
#     ],
#     key=['HF'],
# )
@triton.jit
def _m1_decode_kernel(W1_init, W1_grad, XA, XB, XC, coeff, Out,
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

    # if batch == 0 and head == 0:
    #     XA_chunk_shape = tl.tensor(XA_chunk.shape, type=tl.float32)
    #     tl.device_print("x: ", XA_chunk_shape)

    Z1 = tl.sum(tl.trans(XB_chunk) * W1_init_data, 0) - XA_chunk # [1,HF]
    # if batch == 0 and head == 0:
    #     # XB_chunk_shape = tl.tensor(XB_chunk.shape, type=tl.float32)
    #     # tl.device_print("x: ", XB_chunk_shape)
    #     # Z1_shape = tl.tensor(Z1.shape, type=tl.float32)
    #     # tl.device_print("x: ", Z1_shape)
    #     # temp = tl.sum(XB_chunk * Z1, 0)
    #     # temp = tl.tensor(temp.shape, type=tl.float32)
    #     # temp = tl.tensor(W1_grad_data.shape, type=tl.float32)
    #     # temp = tl.tensor(tl.dot(tl.trans(XB_chunk), Z1).shape, type=tl.float32)
    #     temp = tl.tensor((tl.trans(XB_chunk) * Z1).shape, type=tl.float32)
    #     tl.device_print("x: ", temp)

    W1_grad_data += tl.trans(XB_chunk) * Z1  # tl.dot(tl.trans(XB_chunk), Z1)
    W1_init_data -= coeff_chunk * W1_grad_data
    Z1_bar = tl.sum(tl.trans(XC_chunk) * W1_init_data, 0)

    tl.store(Out_chunk, Z1_bar.to(Out.type.element_ty))
    tl.store(W1_init, W1_init_data.to(W_dtype))
    tl.store(W1_grad, W1_grad_data.to(W_dtype))


def ttt_m1_decode(XA, XB, XC, coeff, W1_init, W1_grad):
    Z1 = (XB @ W1_init).sub_(XA)
    W1_grad.add_(XB.transpose(-1, -2) @ Z1)
    W1_init.sub_(coeff * W1_grad)
    Z1 = XC @ W1_init
    return W1_init, W1_grad, Z1


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(10, 15)],  # different possible values for `x_name`
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
def benchmark_forward(BS, N, CS, NH, HF, provider):
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
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ttt_m1_triton_forward(XA, XB, XC, coeff, W1_original),
                                                     quantiles=quantiles)
    else:
        raise NotImplementedError

    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(0, 12)],  # different possible values for `x_name`
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
        plot_name="M1 decode time",  # name for the plot. Used also as a file name for saving the plot.
        args={'BS': 64, 'NH': 32, 'CS': 1, 'HF': 64},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_decode(N, BS, NH, CS, HF, provider):
    assert CS == 1

    input_dtype = torch.float16

    W1 = torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
    W1_grad = torch.randn_like(W1) * 0.02
    XA = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(N, BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(N, BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02

    def loop(decode, W1, W1_grad):
        for i in range(N):
            W1, W1_grad, _ = decode(XA[i], XB[i], XC[i], coeff[i], W1, W1_grad)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(ttt_m1_decode, W1, W1_grad),
                                                     quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: loop(ttt_m1_triton_decode, W1, W1_grad),
                                                     quantiles=quantiles)
    else:
        raise NotImplementedError

    return ms, min_ms, max_ms


if __name__ == "__main__":
    if sys.argv[1] == 'prefill':
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
        print('  mean ratio: ' + str(torch.abs(clean_tensor((W1 - W1_triton) / W1)).mean()))
        print('  median ratio: ' + str(torch.abs(clean_tensor((W1 - W1_triton) / W1)).median()))
        print('  max ratio: ' + str(torch.abs(clean_tensor((W1 - W1_triton) / W1)).max()))
        print(
            '  torch value @ max ratio: ' + str(W1.flatten()[torch.abs(clean_tensor((W1 - W1_triton) / W1)).argmax()]))
        print('  triton value @ max ratio: ' + str(
            W1_triton.flatten()[torch.abs(clean_tensor((W1 - W1_triton) / W1)).argmax()]))
        print('Output diff: ' + str(torch.abs(XCW_batch - XCW_batch_triton).max()))
        print('  mean ratio: ' + str(torch.abs(clean_tensor((XCW_batch - XCW_batch_triton) / XCW_batch)).mean()))
        print('  median ratio: ' + str(torch.abs(clean_tensor((XCW_batch - XCW_batch_triton) / XCW_batch)).median()))
        print('  max ratio: ' + str(torch.abs(clean_tensor((XCW_batch - XCW_batch_triton) / XCW_batch)).max()))
        print('  torch value @ max ratio: ' + str(
            XCW_batch.flatten()[torch.abs(clean_tensor((XCW_batch - XCW_batch_triton) / XCW_batch)).argmax()]))
        print('  triton value @ max ratio: ' + str(
            XCW_batch_triton.flatten()[torch.abs(clean_tensor((XCW_batch - XCW_batch_triton) / XCW_batch)).argmax()]))
        print('  index @ max ratio:', torch.abs(clean_tensor((XCW_batch - XCW_batch_triton) / XCW_batch)).argmax())
        print('========== Timing ==========')
        benchmark_forward.run(show_plots=False, print_data=True)
    elif sys.argv[1] == 'decode':
        BS, NH, CS, HF = 16, 32, 1, 64
        # BS, NH, CS, HF = 1, 1, 1, 4
        input_dtype = torch.float16
        W1 = torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
        W1_grad = torch.randn_like(W1) * 0.02
        W1_original = W1.clone()
        W1_grad_original = W1_grad.clone()
        XA = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        XB = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        XC = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        coeff = torch.randn(BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02
        W1, W1_grad, XCW_batch = ttt_m1_decode(XA, XB, XC, coeff, W1, W1_grad)
        W1_triton, W1_grad_triton, XCW_batch_triton = ttt_m1_triton_decode(XA, XB, XC, coeff, W1_original,
                                                                           W1_grad_original)

        print('W1 = \n', W1_grad, '\n\nW1_triton = \n', W1_grad_triton)

        print('========== Matching outputs abs diff ==========')
        print('W1 diff: ' + str(torch.abs(W1 - W1_triton).max()))
        print('W1_grad diff: ' + str(torch.abs(W1_grad - W1_grad_triton).max()))
        print('Output diff: ' + str(torch.abs(XCW_batch - XCW_batch_triton).max()))
        print('========== Timing ==========')
        benchmark_decode.run(show_plots=False, print_data=True)
    else:
        # # raise NotImplementedError
        BS, NH, CS, HF = 16, 32, 1, 64
        # BS, NH, CS, HF = 1, 1, 1, 4
        input_dtype = torch.float16
        W1 = torch.randn(BS, NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
        W1_grad = torch.randn_like(W1) * 0.02
        W1_original = W1.clone()
        W1_grad_original = W1_grad.clone()
        XA = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        XB = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        XC = torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        coeff = torch.randn(BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02

        W1_triton, W1_grad_triton, XCW_batch_triton = ttt_m1_triton_decode(XA, XB, XC, coeff, W1_original,
                                                                           W1_grad_original)

        # L = 16
        # BS, NH, NC, CS, HF = 16, 32, L // 16, 16, 64
        # input_dtype = torch.float16
        # W1 = torch.randn(NH, HF, HF, device='cuda', dtype=input_dtype) * 0.02
        # W1_original = W1.clone()
        # XA = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        # XB = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        # XC = torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
        # coeff = torch.randn(BS, NH, NC, CS, 1, device='cuda', dtype=input_dtype) * 0.02
        #
        # W1_triton, XCW_batch_triton = ttt_m1_triton_forward(XA, XB, XC, coeff, W1_original)
