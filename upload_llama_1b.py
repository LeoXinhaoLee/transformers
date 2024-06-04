import gc
import pdb

import argparse
import os
import os.path as osp
import time
import json
import logging
from distutils.util import strtobool

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PretrainedConfig
from transformers import LlamaForCausalLM, LlamaConfig

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--logdir", type=str, default="./exp/clean")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-1.4b")
# state-spaces/mamba-130m | meta-llama/Llama-2-7b | state-spaces/mamba-1.4b | ttt-125m | ttt-1b | ttt-profile
parser.add_argument("--mode", type=str, default="prefill", choices=["prefill", "decode"])
parser.add_argument("--promptlen", type=int, default=1)
parser.add_argument("--genlen", type=int, default=128)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--attn_impl", type=str, default='flash_attention_2', choices=['eager', 'flash_attention_2'])
parser.add_argument("--inner_net", type=str, default='mlp_2_dual', choices=['mlp_1_dual', 'mlp_2_dual', 'mlp_1_dual_triton', 'mlp_2_dual_triton'])
parser.add_argument("--use_compile", action='store_true')
parser.add_argument("--no_cg", action='store_true')    # @xinhao: currently only implemented for Mamba and TTT
parser.add_argument("--profile", action='store_true')  # @xinhao: pytorch profiler, different from nsys in micro-benchmark
args = parser.parse_args()


torch.random.manual_seed(0)  # @xinhao: make sure model init is fixed

repeats = 3
device = "cuda"
dtype = torch.float16  # @xinhao: follow mamba benchmark

config = LlamaConfig.from_json_file('./llama_config/config_1b.json')  # 1B llama config
config._attn_implementation = args.attn_impl  # @xinhao: llama config use `_attn_implementation` to select attn
model = LlamaForCausalLM(config).to(dtype=dtype)
model.push_to_hub("LeoLearntoCode/llama-1.3b")
#
# config = LlamaConfig.from_json_file('./llama_config/config_125m.json')  # 1B llama config
# config._attn_implementation = args.attn_impl  # @xinhao: llama config use `_attn_implementation` to select attn
# model = LlamaForCausalLM(config).to(dtype=dtype)
# model.push_to_hub("LeoLearntoCode/llama-125m")
#
# config = LlamaConfig.from_json_file('./llama_config/config_350m.json')  # 1B llama config
# config._attn_implementation = args.attn_impl  # @xinhao: llama config use `_attn_implementation` to select attn
# model = LlamaForCausalLM(config).to(dtype=dtype)
# model.push_to_hub("LeoLearntoCode/llama-350m")
#
# config = LlamaConfig.from_json_file('./llama_config/config_760m.json')  # 1B llama config
# config._attn_implementation = args.attn_impl  # @xinhao: llama config use `_attn_implementation` to select attn
# model = LlamaForCausalLM(config).to(dtype=dtype)
# model.push_to_hub("LeoLearntoCode/llama-760m")

# model = AutoModelForCausalLM.from_pretrained('LeoLearntoCode/llama-1.3b', device_map={"": device}, torch_dtype=dtype)
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# model = AutoModelForCausalLM.from_pretrained('LeoLearntoCode/llama-125m', device_map={"": device}, torch_dtype=dtype)
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# model = AutoModelForCausalLM.from_pretrained('LeoLearntoCode/llama-760m', device_map={"": device}, torch_dtype=dtype)
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")