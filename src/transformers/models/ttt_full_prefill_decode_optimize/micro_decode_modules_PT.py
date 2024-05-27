import copy
import pdb

import torch
import torch.nn as nn
import einops
import triton
import triton.language as tl
import os


def diff_gelu(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff

def ln_fwd(x, gamma, beta, eps=1e-6):
    """
    Args:
        x: [B*nh,N,f]
        gamma: [1,nh,1,f]
        beta: [1,nh,1,f]
        eps:

    Returns:
        z: [B*nh,N,f]

    """
    B, NH, N, HF = x.shape

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta
    return y

def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    """
    Args:
        x: [B,NH,N=1,f]
        l2_target: [B,NH,N=1,f]
        gamma: [1,nh,1,f]
        beta: [1,nh,1,f]
        eps:

    Returns:
        grad_l_x: [B,nh,N=1,f]
    """
    B, NH, N, HF = x.shape

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)  # [B,nh,N=1,1]
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std  # [B,nh,N=1,f]

    # Scale and shift
    y = gamma * x_hat + beta  #[1,nh,N=1,f] * [B,nh,N=1,f] + [1,nh,N=1,f]

    grad_output = y - l2_target  # [B,nh,N=1,f]
    grad_x_hat = grad_output * gamma  # [B,nh,N=1,f]
    z = (
        (1.0 / HF)
        * (
            HF * grad_x_hat  # [B,nh,1,f]
            - grad_x_hat.sum(dim=-1, keepdim=True)  # [B,nh,1,1]
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)  # [B,nh,N=1,f] * [B,nh,N=1,1]
        )
        / std
    )
    return z

########## Pytorch ##########

###
## M1
###
# def ttt_m1_decode(XA, XB, XC, coeff,
#                  W1_init, W1_grad, b1_init, b1_grad,
#                  ln_weight, ln_bias):
#     """
#     Args:
#         XA XB XC: [B,NH,N=1,f]
#         coeff: [B,NH,N=1,1]
#         W1_init W1_grad: [B,NH,f,f]
#         b1_init b1_grad: [B,NH,1,f]
#         ln_weight ln_bias: [1,NH,1,f]
#
#     Returns:
#
#     """
#     B, NH, N, HF = XA.shape
#
#     Z1 = XB @ W1_init + b1_init  # [B,nh,K=1,f] @ [B,nh,f,f] + [B,nh,1,f] -> [B,nh,K=1,f]
#     reconstruction_target = XA - XB
#     # grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # [B,nh,K=1,f]  # replaced by below explicitly
#
#     mu = Z1.mean(dim=-1, keepdim=True)  # [B,nh,N=1,1]
#     var = Z1.var(dim=-1, keepdim=True, unbiased=False)
#     std = torch.sqrt(var + 1e-6)
#     # std = 1.
#     x_hat = (Z1 - mu) / std  # [B,nh,N=1,f]
#     y = ln_weight * x_hat + ln_bias  # [1,nh,N=1,f] * [B,nh,N=1,f] + [1,nh,N=1,f]
#
#     grad_output = y - reconstruction_target  # [B,nh,N=1,f]
#     grad_x_hat = ln_weight * grad_output  # [B,nh,N=1,f]
#     grad_l_wrt_Z1 = (
#             (1.0 / HF)
#             * (
#                     HF * grad_x_hat  # [B,nh,1,f]
#                     - grad_x_hat.sum(dim=-1, keepdim=True)  # [B,nh,1,1]
#                     - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)  # [B,nh,N=1,f] * [B,nh,N=1,1]
#             )
#             / std
#     )
#     # grad_l_wrt_Z1 = y # test
#
#     W1_grad.add_(XB.transpose(-1, -2) @ grad_l_wrt_Z1)  # [B,nh,1,f].t @ [B,nh,1,f] + [B,nh,f,f]
#     b1_grad.add_(grad_l_wrt_Z1)
#
#     W1_init.sub_(coeff * W1_grad)  # [B,nh,f,f] - [B,nh,N=1,1] * [B,nh,f,f]
#     b1_init.sub_(coeff * b1_grad)  # [B,nh,1,f] - [B,nh,N=1,1] * [B,nh,1,f]
#
#     Z1_bar = XC @ W1_init + b1_init  # [B,nh,K=1,f] @ [B,nh,f,f] - [B,nh,1,1] * [B,nh,f,f]
#     # Z1_bar = XC + ln_fwd(Z1_bar, ln_weight, ln_bias)  # replace by below explicitly
#
#     mu = Z1_bar.mean(dim=-1, keepdim=True)  # [B,nh,N=1,1]
#     var = Z1_bar.var(dim=-1, keepdim=True, unbiased=False)
#     std = torch.sqrt(var + 1e-6)
#     # std = 1.
#     x_hat_bar = (Z1_bar - mu) / std  # [B,nh,N=1,f]
#     Z1_bar = XC + (ln_weight * x_hat_bar + ln_bias)
#
#     return Z1_bar


def ttt_m1_decode(XA, XB, XC, coeff,
                 W1_init, W1_grad, b1_init, b1_grad,
                 ln_weight, ln_bias):
    Z1 = (XB @ W1_init).sub_(XA)
    W1_grad.add_(XB.transpose(-1, -2) @ Z1)
    W1_init.sub_(coeff * W1_grad)
    Z1 = XC @ W1_init
    return Z1

