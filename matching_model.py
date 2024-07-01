import pdb

import copy
import gc

import argparse
import os
import os.path as osp
import time
import json
import logging

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PretrainedConfig
from transformers.models.ttt.configuration_ttt import TTT_STANDARD_CONFIGS, TttConfig  # 125m and 1b config
from transformers.models.ttt_full_prefill_decode_optimize.modeling_ttt import TttForCausalLM

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--logdir", type=str, default="./exp/clean")
parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
parser.add_argument("--promptlen", type=int, default=512)
parser.add_argument("--genlen", type=int, default=0)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--inner_net", type=str, default='mlp_1_dual', choices=['mlp_1_dual', 'mlp_2_dual'])
parser.add_argument("--use_compile", action='store_true')
parser.add_argument("--no_cg", action='store_true')    # @xinhao: currently only implemented for Mamba and TTT
args = parser.parse_args()

torch.random.manual_seed(0)
device = "cuda"
dtype = torch.float16

ttt_size = '1b'

ttt_config_pt = TttConfig(**TTT_STANDARD_CONFIGS[ttt_size], vocab_size=32000)
ttt_config_pt.inner_net = args.inner_net
ttt_config_pt.use_compile = args.use_compile
ttt_config_pt.dtype = dtype
ttt_config_pt.fused_add_norm = True
ttt_config_pt.residual_in_fp32 = True

ttt_config_tk = copy.deepcopy(ttt_config_pt)
ttt_config_tk.inner_net = args.inner_net + '_tk'

print('Inner Net: ', args.inner_net)

model_pt = TttForCausalLM(ttt_config_pt).to(device=device, dtype=dtype)
model_pt.eval()
print(f"Number of parameters (PT): {sum(p.numel() for p in model_pt.parameters() if p.requires_grad)}")

model_tk = TttForCausalLM(ttt_config_tk).to(device=device, dtype=dtype)
model_tk.eval()
print(f"Number of parameters (TK): {sum(p.numel() for p in model_tk.parameters() if p.requires_grad)}")


input_ids = torch.randint(1, 32000, (args.batch, args.promptlen), dtype=torch.long, device=device)
max_length = input_ids.shape[1] + args.genlen

fn_pt = lambda: model_pt.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=(not args.no_cg),
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        temperature=1.0,
        top_k=1,  # @xinhao: mamba src code: shortcut for greedy
        top_p=0)

fn_tk = lambda: model_tk.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=(not args.no_cg),
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        temperature=1.0,
        top_k=1,  # @xinhao: mamba src code: shortcut for greedy
        top_p=0)

out_pt = fn_pt()  # capture graph if cg=True, will not be timed
# out_tk = fn_tk()  # capture graph if cg=True, will not be timed
# del out_pt, out_tk

out_pt = fn_pt()
print(out_pt[0].shape)

