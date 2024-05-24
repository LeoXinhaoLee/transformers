import pdb

import argparse
import os
import os.path as osp
import time
import json
import logging

import torch
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--logdir", type=str, default="./exp/clean")
parser.add_argument("--mode", type=str, default="decode", choices=["prefill", "decode"])
parser.add_argument("--model-name", type=str, default="LeoLearntoCode/llama-1.3b")
parser.add_argument("--promptlen", type=int, default=1)
parser.add_argument("--genlen", type=int, default=128)
parser.add_argument("--batch", type=int, default=16)
args = parser.parse_args()

def print_args(args):
    s = "Arguments: \n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

os.makedirs(args.logdir, mode=0o777, exist_ok=True)

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to the lowest level to catch all messages
file_handler = logging.FileHandler(osp.join(args.logdir, "logging.log"), mode='a')
console_handler = logging.StreamHandler()
# Create a formatter and set it on handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Test messages
logger.info(f"\n============== Model Name: {args.model_name} ==============")
config_str = print_args(args)
logger.info(config_str)

torch.random.manual_seed(0)  # @xinhao: make sure model init is fixed

repeats = 3
device = "cuda"
dtype = torch.float16  # @xinhao: follow mamba benchmark
logger.info("dtype: " + str(dtype))

in_len = args.promptlen
out_len = args.promptlen + args.genlen

# sampling_params = SamplingParams(top_p=1.0, ignore_eos=True, max_tokens=args.genlen, min_tokens=args.genlen)
sampling_params = SamplingParams(top_p=1.0, ignore_eos=True, max_tokens=out_len, min_tokens=out_len)
llm = LLM(model=args.model_name, skip_tokenizer_init=True)
prompts = [torch.randint(low=10, high=100, size=(args.promptlen,)).tolist() for _ in range(args.batch)]

outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
del outputs
logger.info(f"Succeeded.")

torch.cuda.synchronize()
start = time.time()
for i in range(repeats):
    llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
torch.cuda.synchronize()
avg_time = (time.time() - start) / repeats

logger.info(f"Mode: {args.mode}")
logger.info(f"Prompt length: {in_len}, generation length: {out_len - in_len}")
logger.info(f"{args.model_name} prompt processing + decoding time: {avg_time * 1000:.0f}ms")
logger.info(f"Throughput (total tok = prefill + decode): {args.batch * out_len / avg_time:.3f} tokens / s")
logger.info(f"Throughput (total tok = decode): {args.batch * (out_len - in_len) / avg_time:.3f} tokens / s")
logger.info("==================================")
