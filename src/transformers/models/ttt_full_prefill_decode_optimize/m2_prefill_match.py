import torch
import einops
import triton
import triton.language as tl


########## Triton ##########
@triton.jit
def _m2_kernel(W1, W2, XA, XB, XC, coeff_last, coeff, Out,
               stride_ab, stride_ah, stride_an, stride_ac, stride_af,
               stride_eb, stride_eh, stride_en, stride_ec,
               stride_pb, stride_ph, stride_pn,
               stride_wb, stride_wh, stride_wf, stride_wd,
               stride_wf_prime,
               CS: tl.constexpr, HF: tl.constexpr, HF_prime: tl.constexpr,
               N_CHUNK: tl.constexpr):
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
    W1_data = tl.load(W1 + w_offset + rf[:, None] * stride_wf + rf_prime[None, :] * stride_wd)
    W2_data = tl.load(W2 + w_offset + rf_prime[:, None] * stride_wf_prime + rf[None, :] * stride_wd)
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


def m2_forward(W1, W2, XA, XB, XC, coeff_last, coeff):
    B, NH, NC, CS, HF = XA.shape
    NH, HF, HF_prime = W1.shape
    assert coeff.shape == (B, NH, NC, CS)
    coeff_last = coeff[..., -1:].squeeze(-1)
    grid = (B, NH, 1)
    output = torch.empty(size=(B, NH, NC, CS, HF), device=W1.device, dtype=input_dtype)
    W1_expand = torch.tile(W1, dims=(B, 1, 1, 1))
    W2_expand = torch.tile(W2, dims=(B, 1, 1, 1))
    _m2_kernel[grid](W1_expand,  # [B,nh,f,f], cloned from W1, safe for in-place op
                     W2_expand,
                     XA, XB, XC, coeff_last, coeff, output,
                     NH * NC * CS * HF, NC * CS * HF, CS * HF, HF, 1,  # strides for A,B,C,O
                     NH * NC * CS, NC * CS, CS, 1,  # strides for E
                     NH * NC, NC, 1,  # strides for last coeff
                     NH * HF * HF_prime, HF * HF_prime, HF_prime, 1,  # strides for W1
                     HF,  # stride for W2
                     CS, HF, HF_prime,
                     NC
                     )
    return W1_expand, W2_expand, output


########## Pytorch ##########

def for_loop(W1, W2, XA, XB, XC, coeff_last, coeff):
    output_tensor = torch.empty(size=(NC, B * NH, CS, HF), device=W1.device, dtype=input_dtype)
    W1_init = W1
    W2_init = W2
    for i in range(NC):
        XA_chunk = XA[i]
        XB_chunk = XB[i]
        XC_chunk = XC[i]
        coeff_chunk_last = coeff_last[i]
        coeff_chunk = coeff[i]

        Z1 = (XB_chunk @ W1_init)  # [B*nh,K,f] @ [B*nh,f,f] -> [B*nh,K,f]
        grad_l_wrt_Z2 = Z1 @ W2_init - XA_chunk
        grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-1, -2)

        Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(-1, -2))  # [B*nh,K,K]
        Z1_bar = XC_chunk @ W1_init - (
                    coeff_chunk * Attn1) @ grad_l_wrt_Z1  # [B*nh,K,f] @ [B*nh,f,f] - ([B*nh,K,1] * [B*nh,K,K]) @ [B*nh,K,f]

        Attn2 = torch.tril(Z1_bar @ Z1.transpose(-2, -1))
        Z2_bar = Z1_bar @ W2_init - (coeff_chunk * Attn2) @ grad_l_wrt_Z2

        W1_init.sub_((coeff_chunk_last * XB_chunk).transpose(-1, -2) @ grad_l_wrt_Z1)
        W2_init.sub_((coeff_chunk_last * Z1).transpose(-1, -2) @ grad_l_wrt_Z2)
        output_tensor[i] = Z2_bar

    return W1_init, W2_init, output_tensor


def ttt_m2_prefill(XA, XB, XC, coeff, W1, W2):
    B, NH, NC, CS, HF = XA.shape
    coeff_last = coeff[..., -1:, :]
    XA = XA.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, CS, -1)
    XB = XB.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, CS, -1)
    XC = XC.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, CS, -1)
    coeff = coeff.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, CS, -1)
    coeff_last = coeff_last.permute(2, 0, 1, 3, 4).reshape(NC, B * NH, 1, 1)
    W1, W2, XCW_batch = for_loop(
        torch.tile(W1, dims=(B, 1, 1)),  # [B*nh,f,f], cloned from W1, safe for in-place op
        torch.tile(W2, dims=(B, 1, 1)),
        XA, XB, XC, coeff_last, coeff
    )
    XCW_batch = einops.rearrange(XCW_batch, "nc (b nh) cs f -> b (nc cs) (nh f)", b=B, nh=NH)
    return W1, W2, XCW_batch


def ttt_m2_triton_prefill(XA, XB, XC, coeff, W1, W2):
    B, NH, NC, CS, HF = XA.shape
    coeff = coeff.squeeze(-1)
    coeff_last = coeff[..., -1:]
    W1, W2, output = m2_forward(W1, W2, XA, XB, XC, coeff_last, coeff)
    output = einops.rearrange(output, "b nh nc cs f -> b (nc cs) (nh f)")
    W1 = einops.rearrange(W1, "b nh f d -> (b nh) f d")
    W2 = einops.rearrange(W2, "b nh f d -> (b nh) f d")
    return W1, W2, output


if __name__ == "__main__":
    L = 512
    B, NH, NC, CS, HF, HF_prime = 3, 7, L // 16, 16, 64, 256
    input_dtype = torch.float16
    W1 = torch.randn(NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02
    W1_original = W1.clone()
    W2 = torch.randn(NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02
    W2_original = W2.clone()
    XA = torch.randn(B, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XB = torch.randn(B, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    XC = torch.randn(B, NH, NC, CS, HF, device='cuda', dtype=input_dtype) * 0.02
    coeff = torch.randn(B, NH, NC, CS, 1, device='cuda', dtype=input_dtype) * 0.02
    W1, W2, XCW_batch = ttt_m2_prefill(XA, XB, XC, coeff, W1, W2)
    print(W1.shape, W2.shape, XCW_batch.shape)
    W1_triton, W2_triton, XCW_batch_triton = ttt_m2_triton_prefill(XA, XB, XC, coeff, W1_original, W2_original)
    print(W1_triton.shape, W2_triton.shape, XCW_batch_triton.shape)

    print('========== M2 Matching Pytorch v.s Triton ============')
    print('W1 diff: ' + str(torch.abs(W1 - W1_triton).max()))
    print('W2 diff: ' + str(torch.abs(W2 - W2_triton).max()))
    print('Output diff: ' + str(torch.abs(XCW_batch - XCW_batch_triton).max()))
