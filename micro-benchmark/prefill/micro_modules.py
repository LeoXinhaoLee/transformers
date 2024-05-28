import copy
import pdb

import torch
import einops
import triton
import triton.language as tl


########## Pytorch ##########
def pt_m2_prefill(W1, W2, XA, XB, XC, coeff, coeff_last):
    '''
    Args:
        W1: [B*NH,H,H_prime]
        W2:
        XA: [B*NH,NC,CS,HF]
        XB:
        XC:
        coeff: [B*NH,NC,CS,1]

    Returns:

    '''
    NC, B_mul_NH, CS, HF = XA.shape
    output_tensor = torch.empty_like(XA)

    for i in range(NC):
        XA_chunk = XA[i]
        XB_chunk = XB[i]
        XC_chunk = XC[i]
        coeff_chunk_last = coeff_last[i]
        coeff_chunk = coeff[i]

        Z1 = (XB_chunk @ W1)  # [B*nh,K,f] @ [B*nh,f,f] -> [B*nh,K,f]
        grad_l_wrt_Z2 = Z1 @ W2 - XA_chunk
        grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2.transpose(-1, -2)

        Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))  # [B*nh,K,K]
        Z1_bar = XC_chunk @ W1 - \
                 (coeff_chunk * Attn1) @ grad_l_wrt_Z1  # [B*nh,K,f] @ [B*nh,f,f] - ([B*nh,K,1] * [B*nh,K,K]) @ [B*nh,K,f]

        Attn2 = torch.tril(Z1_bar @ Z1.transpose(-2, -1))
        Z2_bar = Z1_bar @ W2 - (coeff_chunk * Attn2) @ grad_l_wrt_Z2

        W1.sub_((coeff_chunk_last * XB_chunk).transpose(-1, -2) @ grad_l_wrt_Z1)
        W2.sub_((coeff_chunk_last * Z1).transpose(-1, -2) @ grad_l_wrt_Z2)
        output_tensor[i] = Z2_bar

    return output_tensor


def pt_m1_prefill(W1, XA, XB, XC, coeff, coeff_last, **kwargs):
    '''
    Args:
        W1: [B*NH,HF,HF]
        XA: [B*NH,NC,CS,HF]
        XB:
        XC:
        coeff: [B*NH,NC,CS,1]

    Returns:

    '''
    NC, B_mul_NH, CS, HF = XA.shape
    output_tensor = torch.empty_like(XA)

    for i in range(NC):
        XA_chunk = XA[i]
        XB_chunk = XB[i]
        XC_chunk = XC[i]
        coeff_chunk_last = coeff_last[i]
        coeff_chunk = coeff[i]

        grad_l_wrt_Z1 = XB_chunk @ W1 - XA_chunk
        Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))  # [B*nh,K,K]
        Z1_bar = XC_chunk @ W1 - (coeff_chunk * Attn1) @ grad_l_wrt_Z1
        W1.sub_((coeff_chunk_last * XB_chunk).transpose(-1, -2) @ grad_l_wrt_Z1)
        output_tensor[i] = Z1_bar

    return output_tensor


########## Triton ##########
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=1),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=1),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=1),
        triton.Config({}, num_stages=4, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=8),
    ],
    key=['N_CHUNK'],
    restore_value=['W1', 'W2'],
)
@triton.jit
def _m2_prefill_kernel(
        W1, W2, XA, XB, XC, coeff_last, coeff, Out,
        stride_ab, stride_ah, stride_an, stride_ac, stride_af,
        stride_eb, stride_eh, stride_en, stride_ec,
        stride_pb, stride_ph, stride_pn,
        stride_wb, stride_wh, stride_wf, stride_wd,
        stride_wf_prime,
        CS: tl.constexpr, HF: tl.constexpr, HF_prime: tl.constexpr,
        N_CHUNK: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    abco_offset = batch * stride_ab + head * stride_ah
    w_offset = batch * stride_wb + head * stride_wh
    coeff_offset = batch * stride_eb + head * stride_eh
    coeff_last_offset = batch * stride_pb + head * stride_ph

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)
    rf_prime = tl.arange(0, HF_prime)
    XA = XA + abco_offset
    XB = XB + abco_offset
    XC = XC + abco_offset
    Out = Out + abco_offset
    W1_ptr = W1 + w_offset + rf[:, None] * stride_wf + rf_prime[None, :] * stride_wd
    W1_data = tl.load(W1_ptr)
    W2_ptr = W2 + w_offset + rf_prime[:, None] * stride_wf_prime + rf[None, :] * stride_wd
    W2_data = tl.load(W2_ptr)
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

        Z1 = tl.dot(XB_chunk, W1_data, out_dtype=tl.float16)
        grad_l_wrt_Z2 = tl.dot(Z1, W2_data, out_dtype=tl.float16) - XA_chunk
        grad_l_wrt_Z1 = tl.dot(grad_l_wrt_Z2, tl.trans(W2_data), out_dtype=tl.float16)

        mask = rc[:, None] >= rc[None, :]
        Attn1_full = tl.dot(XC_chunk, tl.trans(XB_chunk), out_dtype=tl.float16)
        Attn1 = tl.where(mask, Attn1_full, 0)
        Z1_bar = tl.dot(XC_chunk, W1_data, out_dtype=tl.float16) - tl.dot((coeff_chunk[:, None] * Attn1), grad_l_wrt_Z1,
                                                                          out_dtype=tl.float16)

        Attn2_full = tl.dot(Z1_bar, tl.trans(Z1), out_dtype=tl.float16)
        Attn2 = tl.where(mask, Attn2_full, 0)
        Z2_bar = tl.dot(Z1_bar, W2_data, out_dtype=tl.float16) - tl.dot((coeff_chunk[:, None] * Attn2), grad_l_wrt_Z2,
                                                                        out_dtype=tl.float16)

        W1_data -= tl.dot(tl.trans(coeff_chunk_last * XB_chunk).to(tl.float16), grad_l_wrt_Z1, out_dtype=tl.float16)
        W2_data -= tl.dot(tl.trans(coeff_chunk_last * Z1).to(tl.float16), grad_l_wrt_Z2, out_dtype=tl.float16)

        Out_chunk = Out + local_abco_offset + (rc[:, None] * stride_ac + rf[None, :] * stride_af)
        tl.store(Out_chunk, Z2_bar.to(tl.float16))
    tl.store(W1_ptr, W1_data.to(W1.type.element_ty))
    tl.store(W2_ptr, W2_data.to(W2.type.element_ty))


def triton_m2_prefill(W1, W2, XA, XB, XC, coeff, coeff_last):
    B, NH, NC, CS, HF = XA.shape
    B, NH, HF, HF_prime = W1.shape
    grid = (B, NH, 1)
    output = torch.empty_like(XA)
    _m2_prefill_kernel[grid](
        W1,
        W2,
        XA, XB, XC, coeff_last, coeff, output,
        NH * NC * CS * HF, NC * CS * HF, CS * HF, HF, 1,  # strides for A,B,C,O
        NH * NC * CS, NC * CS, CS, 1,  # strides for E
        NH * NC, NC, 1,  # strides for last coeff
        NH * HF * HF_prime, HF * HF_prime, HF_prime, 1,  # strides for W1
        HF,  # stride for W2
        CS, HF, HF_prime,
        NC
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=1),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=1),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=1),
        triton.Config({}, num_stages=4, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=8),
    ],
    key=['N_CHUNK'],
    restore_value=['W1'],
)
@triton.jit
def _m1_prefill_kernel(
        W1, XA, XB, XC, coeff_last, coeff, Out,
        stride_ab, stride_ah, stride_an, stride_ac, stride_af,
        stride_eb, stride_eh, stride_en, stride_ec,
        stride_pb, stride_ph, stride_pn,
        stride_wb, stride_wh, stride_wf, stride_wd,
        CS: tl.constexpr, HF: tl.constexpr,
        N_CHUNK: tl.constexpr
):
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
    W1_ptr = W1 + w_offset + rf[:, None] * stride_wf + rf[None, :] * stride_wd
    W1_data = tl.load(W1_ptr)
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
    tl.store(W1_ptr, W1_data.to(W1.type.element_ty))


def triton_m1_prefill(W1, XA, XB, XC, coeff, coeff_last, **kwargs):
    B, NH, NC, CS, HF = XA.shape
    grid = (B, NH, 1)
    output = torch.empty_like(XA)
    _m1_prefill_kernel[grid](
        W1,
        XA, XB, XC, coeff_last, coeff, output,
        NH * NC * CS * HF, NC * CS * HF, CS * HF, HF, 1,  # strides for A,B,C,O
        NH * NC * CS, NC * CS, CS, 1,  # strides for E
        NH * NC, NC, 1,  # strides for last coeff
        NH * HF * HF, HF * HF, HF, 1,  # strides for W1
        CS, HF,
        NC
    )
    return output


if __name__ == "__main__":

    match_module = 'M1'  # 'M1' | 'M2'

    if match_module == 'M2':
        pt_prefill = pt_m2_prefill
        triton_prefill = triton_m2_prefill
    elif match_module == 'M1':
        pt_prefill = pt_m1_prefill
        triton_prefill = triton_m1_prefill
    else:
        raise NotImplementedError

    input_dtype = torch.float16
    BS, NH, L, HF, = 64, 32, 128, 64
    CS = 16
    NC = L // CS

    HF_prime = 4 * HF if match_module == 'M2' else HF

    original_state_dict = {
        'W1': torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02,
        'W2': torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02,
    }
    original_input_dict = {
        'XA': torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'XB': torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'XC': torch.randn(BS, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'coeff': torch.randn(BS, NH, NC, CS, 1, device='cuda', dtype=input_dtype) * 0.02,
        'coeff_last': torch.randn(BS, NH, NC, 1, 1, device='cuda', dtype=input_dtype) * 0.02,
    }

    ########## PyTorch ##########
    pt_state_dict = copy.deepcopy(original_state_dict)
    pt_input_dict = copy.deepcopy(original_input_dict)
    for k in pt_state_dict.keys():
        pt_state_dict[k] =  einops.rearrange(pt_state_dict[k], 'b nh f1 f2 -> (b nh) f1 f2')
    for k in pt_input_dict.keys():
        pt_input_dict[k] = einops.rearrange(pt_input_dict[k], 'b nh nc cs f -> nc (b nh) cs f')

    pt_output = pt_prefill(**pt_state_dict, **pt_input_dict)
    pt_output = einops.rearrange(pt_output, 'nc (b nh) cs f -> b nh nc cs f', b=BS)

    for k in pt_state_dict.keys():
        pt_state_dict[k] = einops.rearrange(pt_state_dict[k], '(b nh) cs f -> b nh cs f', b=BS)
    for k in pt_input_dict.keys():
        pt_input_dict[k] = einops.rearrange(pt_input_dict[k], 'nc (b nh) cs f -> b nh nc cs f', b=BS)
    ##############################

    ########## Triton ##########
    triton_state_dict = copy.deepcopy(original_state_dict)
    triton_input_dict = copy.deepcopy(original_input_dict)
    triton_output = triton_prefill(**triton_state_dict, **triton_input_dict)
    ##############################

    ########## Triton  CG ##########
    triton_cg_state_dict = copy.deepcopy(original_state_dict)
    triton_cg_input_dict = copy.deepcopy(original_input_dict)

    n_warmups = 2
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            Z_tmp = triton_prefill(**triton_cg_state_dict, **triton_cg_input_dict)
        s.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)

    graph_triton = torch.cuda.CUDAGraph()
    mempool_triton = torch.cuda.graphs.graph_pool_handle()
    with torch.cuda.graph(graph_triton, pool=mempool_triton):
        Z_tmp = triton_prefill(**triton_cg_state_dict, **triton_cg_input_dict)

    def run_triton():
        graph_triton.replay()
        return Z_tmp.clone()

    for k in triton_state_dict.keys():
        triton_cg_state_dict[k].copy_(original_state_dict[k])
    for k in triton_input_dict.keys():
        triton_cg_input_dict[k].copy_(original_input_dict[k])
    triton_cg_output = run_triton()
    ##############################

    print(f'\n========== {match_module} Matching  ============')

    print('Pytorch v.s Triton')
    for k in original_state_dict.keys():
        diff = torch.abs(pt_state_dict[k] - triton_state_dict[k])
        print(f'{k} diff: max={diff.max()}, median={diff.median()}')
    diff = torch.abs(pt_output - triton_output)
    print(f'Output diff: max={diff.max()}, median={diff.median()}\n')

    print('Pytorch v.s Triton CG')
    for k in original_state_dict.keys():
        diff = torch.abs(pt_state_dict[k] - triton_cg_state_dict[k])
        print(f'{k} diff: max={diff.max()}, median={diff.median()}')
    diff = torch.abs(pt_output - triton_cg_output)
    print(f'Output diff: max={diff.max()}, median={diff.median()}')
