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
from transformers.models.ttt_clean.configuration_ttt import TTT_STANDARD_CONFIGS, TTTConfig  # 125m and 1b config
from transformers.models.ttt_clean.modeling_ttt import TTTForCausalLM

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

device = "cuda"
dtype = torch.float16

ttt_size = '1b'

ttt_config_pt = TTTConfig(**TTT_STANDARD_CONFIGS[ttt_size], vocab_size=32000)
ttt_config_pt.inner_net = args.inner_net
ttt_config_pt.use_compile = args.use_compile
ttt_config_pt.dtype = dtype
ttt_config_pt.fused_add_norm = True
ttt_config_pt.residual_in_fp32 = True
ttt_config_pt.conv_before_ttt = True

ttt_config_tk = copy.deepcopy(ttt_config_pt)
ttt_config_tk.inner_net = args.inner_net + '_tk'

print('Inner Net: ', args.inner_net)

torch.random.manual_seed(0)
model_pt = TTTForCausalLM(ttt_config_pt).to(device=device, dtype=dtype)
model_pt.eval()
print(f"Number of parameters (PT): {sum(p.numel() for p in model_pt.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
model_tk = TTTForCausalLM(ttt_config_tk).to(device=device, dtype=dtype)
model_tk.eval()
print(f"Number of parameters (TK): {sum(p.numel() for p in model_tk.parameters() if p.requires_grad)}")

def compare_model_params(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(param1, param2):
            raise ValueError("PT and TK model's params are different!")

compare_model_params(model_pt, model_tk)

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
        top_k=1,
        top_p=0)

fn_tk = lambda: model_tk.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=(not args.no_cg),
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        temperature=1.0,
        top_k=1,
        top_p=0)

out_pt = fn_pt()  # capture graph if cg=True, will not be timed
out_tk = fn_tk()  # capture graph if cg=True, will not be timed
print("Succeeded.")
print('Out Len: ', len(out_pt.sequences[0]))
print('In Len: ', len(input_ids[0]))
del out_pt, out_tk

out_pt = fn_pt()
sequence_pt, probs_list_pt = out_pt.sequences, out_pt.logits  # [BS,prompt_len+1+gen_len], [[BS,prompt_len,V], [BS,V], [BS,V], ...]

out_tk = fn_tk()
sequence_tk, probs_list_tk = out_tk.sequences, out_tk.logits  # [BS,prompt_len+1+gen_len], [[BS,prompt_len,V], [BS,V], [BS,V], ...]

### Prefill ###
prompt_probs_avg_max_diff = []
prompt_probs_avg_mean_diff = []
prompt_token_diff = []
if args.promptlen > 1:
    prompt_probs_pt = probs_list_pt[0]  # [BS, prompt_len, V]
    prompt_probs_tk = probs_list_tk[0]
    prompt_probs_avg_max_diff = torch.abs(prompt_probs_pt - prompt_probs_tk).max(dim=2)[0].mean(dim=0).cpu().numpy()  # [prompt_len,]
    prompt_probs_avg_mean_diff = torch.abs(prompt_probs_pt - prompt_probs_tk).mean(dim=(0,2)).cpu().numpy()  # [prompt_len,]
    prompt_token_diff = torch.sum(prompt_probs_pt.argmax(dim=-1) != prompt_probs_tk.argmax(dim=-1), axis=0).cpu().numpy()  # [prompt_len,]

### Decode ###
decode_probs_avg_max_diff = []
decode_probs_avg_mean_diff = []
decode_token_diff = []
if args.genlen > 0:
    assert len(probs_list_pt) > 1 and len(probs_list_tk) > 1 and  len(probs_list_pt) ==  len(probs_list_tk)
    if args.promptlen == 1:
        # no prefill, all resulting tokens are decoded
        decode_st = 0
    else:
        # 1st: prompt; 2nd: 1st generated token, produced by prefill but not decode
        decode_st = 2
    for i in range(decode_st, len(probs_list_pt)):
        decode_probs_pt = probs_list_pt[i]
        decode_probs_tk = probs_list_tk[i]
        decode_probs_avg_max_diff.append(
            torch.abs(decode_probs_pt - decode_probs_tk).max(dim=1)[0].mean()
        )
        decode_probs_avg_mean_diff.append(
            torch.abs(decode_probs_pt - decode_probs_tk).mean()
        )
        decode_token_diff.append(
            torch.sum(decode_probs_pt.argmax(dim=-1) != decode_probs_tk.argmax(dim=-1))
        )
decode_probs_avg_max_diff = torch.tensor(decode_probs_avg_max_diff).cpu().numpy()
decode_probs_avg_mean_diff = torch.tensor(decode_probs_avg_mean_diff).cpu().numpy()
decode_token_diff = torch.tensor(decode_token_diff).cpu().numpy()

all_stats = {
    'prompt_probs_avg_max_diff': prompt_probs_avg_max_diff,
    'prompt_probs_avg_mean_diff': prompt_probs_avg_mean_diff,
    'prompt_token_diff': prompt_token_diff,
    'decode_probs_avg_max_diff': decode_probs_avg_max_diff,
    'decode_probs_avg_mean_diff': decode_probs_avg_mean_diff,
    'decode_token_diff': decode_token_diff,
}
os.makedirs(args.logdir, exist_ok=True)
torch.save(all_stats, os.path.join(args.logdir, 'all_stats.pth'))
