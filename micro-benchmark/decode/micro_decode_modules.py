import copy
import pdb

import torch
import einops
import triton
import triton.language as tl
import os

import m2_decode_cpp

###
## Pytorch M2
###
def pt_m2_decode_non_end_chunk(XA_chunk, XB_chunk, XC_chunk, coeff_chunk, W1_init, W1_grad, W2_init, W2_grad):
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

    Z2.sub_(XA_chunk)  # grad_l_wrt_Z2 = Z2 - XA_chunk
    grad_l_wrt_Z1 = Z2 @ W2_init.transpose(-1, -2)

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)
    W1_last = W1_init - coeff_chunk * W1_grad
    Z1_bar = XC_chunk @ W1_last

    W2_grad.add_(Z1.transpose(-1, -2) @ Z2)
    W2_last = W2_init - coeff_chunk * W2_grad
    Z2_bar = Z1_bar @ W2_last

    return Z2_bar


def pt_m2_decode_end_chunk(XA_chunk, XB_chunk, XC_chunk, coeff_chunk, W1_init, W1_grad, W2_init, W2_grad):
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

    Z2.sub_(XA_chunk)  # grad_l_wrt_Z2 = Z2 - XA_chunk
    grad_l_wrt_Z1 = Z2 @ W2_init.transpose(-1, -2)

    W1_grad.add_(XB_chunk.transpose(-1, -2) @ grad_l_wrt_Z1)
    W1_init.sub_(coeff_chunk * W1_grad)
    Z1_bar = XC_chunk @ W1_init

    W2_grad.add_(Z1.transpose(-1, -2) @ Z2)
    W2_init.sub_(coeff_chunk * W2_grad)
    Z2_bar = Z1_bar @ W2_init

    W1_grad.zero_()
    W2_grad.zero_()

    return Z2_bar


###
## Triton M2
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
def _m2_decode_non_end_chunk_kernel(W1_init, W1_grad,
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

    tl.store(W1_grad, W1_grad_data.to(W_dtype))
    tl.store(W2_grad, W2_grad_data.to(W_dtype))

def triton_m2_decode_non_end_chunk(XA, XB, XC, coeff, W1_init, W1_grad, W2_init, W2_grad):
    B, NH, CS, HF = XA.shape
    HF_prime = W1_init.shape[-1]
    assert CS == 1
    assert W1_init.shape == W1_grad.shape == (B, NH, HF, HF_prime)
    assert W2_init.shape == W2_grad.shape == (B, NH, HF_prime, HF)
    assert XA.shape == XB.shape == XC.shape
    assert coeff.shape == (B, NH, CS, 1)

    output = torch.empty(size=(B, NH, CS, HF), device=W1_init.device, dtype=torch.float16)  # TODO FIX DTYPE
    grid = (B, NH, 1)

    _m2_decode_non_end_chunk_kernel[grid](W1_init, W1_grad, W2_init, W2_grad,
                            XA, XB, XC, coeff,
                            output,
                            NH * HF * HF_prime,     HF * HF_prime,     HF_prime,    1,  # strides for W1: [B,NH,HF,HF_prime]
                            NH * HF_prime * HF,     HF_prime * HF,     HF,          1,  # strides for W2
                            NH * CS * HF,           CS * HF,           HF,          1,  # strides for ABCO, output
                            NH * CS * 1,            CS * 1,            1,           1,  # strides for coeff
                            CS=CS, HF=HF, HF_prime=HF_prime,
                            # num_warps=8
                            )
    return output


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
def _m2_decode_end_chunk_kernel(W1_init, W1_grad,
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
    tl.store(W2_init, W2_init_data.to(W_dtype))
    tl.store(W1_grad, tl.zeros_like(W1_grad_data).to(W_dtype))
    tl.store(W2_grad, tl.zeros_like(W2_grad_data).to(W_dtype))

def triton_m2_decode_end_chunk(XA, XB, XC, coeff, W1_init, W1_grad, W2_init, W2_grad):
    B, NH, CS, HF = XA.shape
    HF_prime = W1_init.shape[-1]
    assert CS == 1
    assert W1_init.shape == W1_grad.shape == (B, NH, HF, HF_prime)
    assert W2_init.shape == W2_grad.shape == (B, NH, HF_prime, HF)
    assert XA.shape == XB.shape == XC.shape
    assert coeff.shape == (B, NH, CS, 1)

    output = torch.empty(size=(B, NH, CS, HF), device=W1_init.device, dtype=torch.float16)  # TODO FIX DTYPE
    grid = (B, NH, 1)

    _m2_decode_end_chunk_kernel[grid](W1_init, W1_grad, W2_init, W2_grad,
                            XA, XB, XC, coeff,
                            output,
                            NH * HF * HF_prime,     HF * HF_prime,     HF_prime,    1,  # strides for W1: [B,NH,HF,HF_prime]
                            NH * HF_prime * HF,     HF_prime * HF,     HF,          1,  # strides for W2
                            NH * CS * HF,           CS * HF,           HF,          1,  # strides for ABCO, output
                            NH * CS * 1,            CS * 1,            1,           1,  # strides for coeff
                            CS=CS, HF=HF, HF_prime=HF_prime,
                            # num_warps=8
                            )
    return output


if __name__ == "__main__":
    '''
    @xinhao
    Matching outputs of Triton, and CUDA to PyTorch to ensure correctness
        test_cg: if True, use CG
        end_chunk: if True, test kernel for last token in chunk (recommend to use False to focus on non-last-in-chunk kernel first)
        cuda_transpose ['' | 'W1' | 'W12']: no transpose or transpose W1 or transpose both W1 and W2
    '''

    torch.manual_seed(42)
    input_dtype = torch.float16

    test_cg = False
    end_chunk = False
    cuda_transpose = ''

    ############### M2 Matching outputs abs diff ###############

    if end_chunk:
        pt_decode_fn = pt_m2_decode_end_chunk
        triton_decode_fn = triton_m2_decode_end_chunk
        cuda_decode_fn = m2_decode_cpp.decode_end_chunk  # last-in-chunk: update W_init, zero out W_grad
    else:
        pt_decode_fn = pt_m2_decode_non_end_chunk
        triton_decode_fn = triton_m2_decode_non_end_chunk
        cuda_decode_fn = m2_decode_cpp.decode  # non-last-in-chunk: only update W_grad

    BS, NH, CS, HF, HF_prime = 512, 32, 1, 64, 4 * 64

    original_state_dict = {
        'W1': torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02,
        'W1_grad': torch.randn(BS, NH, HF, HF_prime, device='cuda', dtype=input_dtype) * 0.02,
        'W2': torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02,
        'W2_grad': torch.randn(BS, NH, HF_prime, HF, device='cuda', dtype=input_dtype) * 0.02,
    }
    original_input_dict = {
        'XA': torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'XB': torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'XC': torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.02,
        'coeff': torch.randn(BS, NH, CS, 1, device='cuda', dtype=input_dtype) * 0.02,
    }

    ### PT
    pt_state_dict = copy.deepcopy(original_state_dict)
    pt_input_dict = copy.deepcopy(original_input_dict)
    XCW_batch_pt = pt_decode_fn(pt_input_dict['XA'], pt_input_dict['XB'],
                                pt_input_dict['XC'], pt_input_dict['coeff'],
                                pt_state_dict['W1'], pt_state_dict['W1_grad'],
                                pt_state_dict['W2'], pt_state_dict['W2_grad'])

    if not test_cg:
        ### Triton
        triton_state_dict = copy.deepcopy(original_state_dict)
        triton_input_dict = copy.deepcopy(original_input_dict)
        XCW_batch_triton = triton_decode_fn(triton_input_dict['XA'], triton_input_dict['XB'],
                                            triton_input_dict['XC'], triton_input_dict['coeff'],
                                            triton_state_dict['W1'], triton_state_dict['W1_grad'],
                                            triton_state_dict['W2'], triton_state_dict['W2_grad'])

        ## CUDA
        cuda_state_dict = copy.deepcopy(original_state_dict)
        cuda_input_dict = copy.deepcopy(original_input_dict)
        # @xinhao: transpose W1, W2
        for k in cuda_state_dict.keys():
            for k in cuda_state_dict.keys():
                if cuda_transpose == '':
                    flag = False
                elif cuda_transpose == 'W1':
                    flag = 'W1' in k
                elif cuda_transpose == 'W12':
                    flag = True
                else:
                    raise NotImplementedError
            if flag:
                cuda_state_dict[k] = cuda_state_dict[k].transpose(-1, -2).contiguous()
        for k in cuda_input_dict.keys():
            cuda_input_dict[k] = cuda_input_dict[k].squeeze(2)
        XCW_batch_cuda = cuda_decode_fn(cuda_input_dict['XA'], cuda_input_dict['XB'],
                                        cuda_input_dict['XC'], cuda_input_dict['coeff'],
                                        cuda_state_dict['W1'], cuda_state_dict['W1_grad'],
                                        cuda_state_dict['W2'], cuda_state_dict['W2_grad'])
        XCW_batch_cuda = XCW_batch_cuda.unsqueeze(2)
    else:
        ### Triton CG
        triton_state_dict = copy.deepcopy(original_state_dict)
        triton_input_dict = copy.deepcopy(original_input_dict)

        n_warmups = 4
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                Z_tmp = triton_decode_fn(triton_input_dict['XA'], triton_input_dict['XB'],
                                         triton_input_dict['XC'], triton_input_dict['coeff'],
                                         triton_state_dict['W1'], triton_state_dict['W1_grad'],
                                         triton_state_dict['W2'], triton_state_dict['W2_grad'])
            s.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        torch.cuda.current_stream().wait_stream(s)

        graph_triton = torch.cuda.CUDAGraph()
        mempool_triton = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(graph_triton, pool=mempool_triton):
            Z_tmp = triton_decode_fn(triton_input_dict['XA'], triton_input_dict['XB'],
                                     triton_input_dict['XC'], triton_input_dict['coeff'],
                                     triton_state_dict['W1'], triton_state_dict['W1_grad'],
                                     triton_state_dict['W2'], triton_state_dict['W2_grad'])
        def run_triton():
            graph_triton.replay()
            return Z_tmp.clone()

        for k in triton_state_dict.keys():
            triton_state_dict[k].copy_(original_state_dict[k])
        for k in triton_input_dict.keys():
            triton_input_dict[k].copy_(original_input_dict[k])
        XCW_batch_triton = run_triton()


        ### CUDA CG
        cuda_state_dict = copy.deepcopy(original_state_dict)
        cuda_input_dict = copy.deepcopy(original_input_dict)
        # @xinhao: transpose W1, W2
        for k in cuda_state_dict.keys():
            for k in cuda_state_dict.keys():
                if cuda_transpose == '':
                    flag = False
                elif cuda_transpose == 'W1':
                    flag = 'W1' in k
                elif cuda_transpose == 'W12':
                    flag = True
                else:
                    raise NotImplementedError
            if flag:
                cuda_state_dict[k] = cuda_state_dict[k].transpose(-1, -2).contiguous()
        for k in cuda_input_dict.keys():
            cuda_input_dict[k] = cuda_input_dict[k].squeeze(2)

        n_warmups = 4
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                Z_tmp = cuda_decode_fn(cuda_input_dict['XA'], cuda_input_dict['XB'],
                                       cuda_input_dict['XC'], cuda_input_dict['coeff'],
                                       cuda_state_dict['W1'], cuda_state_dict['W1_grad'],
                                       cuda_state_dict['W2'], cuda_state_dict['W2_grad'])
            s.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        torch.cuda.current_stream().wait_stream(s)

        Z_tmp_before = Z_tmp.clone()
        graph_cuda = torch.cuda.CUDAGraph()
        mempool_cuda = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(graph_cuda, pool=mempool_cuda):
            Z_tmp = cuda_decode_fn(cuda_input_dict['XA'], cuda_input_dict['XB'],
                                   cuda_input_dict['XC'], cuda_input_dict['coeff'],
                                   cuda_state_dict['W1'], cuda_state_dict['W1_grad'],
                                   cuda_state_dict['W2'], cuda_state_dict['W2_grad'])

        Z_tmp_before = Z_tmp.clone()
        def run():
            graph_cuda.replay()
            return Z_tmp.clone()

        for k in cuda_state_dict.keys():
            cuda_state_dict[k].copy_(original_state_dict[k])
        for k in cuda_input_dict.keys():
            cuda_input_dict[k].copy_(original_input_dict[k].squeeze(2))
        XCW_batch_cuda = run()
        XCW_batch_cuda = XCW_batch_cuda.unsqueeze(2)


    # print('========== M2 Matching ==========')
    print(f'Last in chunk: {end_chunk}')
    print(f'Use CG: {test_cg}')

    print('=== PyTorch v.s CUDA ===')
    # @xinhao: transpose back
    for k in cuda_state_dict.keys():
        for k in cuda_state_dict.keys():
            if cuda_transpose == '':
                flag = False
            elif cuda_transpose == 'W1':
                flag = 'W1' in k
            elif cuda_transpose == 'W12':
                flag = True
            else:
                raise NotImplementedError
        if flag:
            cuda_state_dict[k] = cuda_state_dict[k].transpose(-1, -2).contiguous()
    for k in original_state_dict.keys():
        diff = pt_state_dict[k] - cuda_state_dict[k]
        print(f'{k} diff: max={diff.max()}')

    print('Output diff: max={}'.format(torch.abs(XCW_batch_pt - XCW_batch_cuda).max()))

    print('=== PyTorch v.s Triton ===')
    for k in original_state_dict.keys():
        diff = pt_state_dict[k] - triton_state_dict[k]
        print(f'{k} diff: max={diff.max()}')

    print('Output diff: max={}'.format(torch.abs(XCW_batch_pt - XCW_batch_triton).max()))
