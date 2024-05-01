# Copyright (c) 2023, Tri Dao, Albert Gu.
import pdb

import argparse
import os
import os.path as osp
import time
import json
import logging

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers.models.ttt.modeling_ttt import TttConfig, TttForCausalLM
from transformers.models.ttt.configuration_ttt import TTT_STANDARD_CONFIGS


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--logdir", type=str, default="./exp/clean")
parser.add_argument("--model-name", type=str, default="openai-community/gpt2")
# state-spaces/mamba-130m | openai-community/gpt2-xl | state-spaces/mamba-1.4b
parser.add_argument("--promptlen", type=int, default=1)
parser.add_argument("--genlen", type=int, default=128)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--attn_impl", type=str, default='eager')  # 'eager' | 'sdpa' | 'flash_attention_2'
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

repeats = 3
device = "cuda"
dtype = torch.float16
logger.info("dtype: " + str(dtype))

logger.info(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba-")
is_ttt = args.model_name.startswith("ttt")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
    # model = MambaForCausalLM.from_pretrained(args.model_name)
    # model = model.to(device=device, dtype=dtype)
elif is_ttt:
    ttt_size = args.model_name.split('-')[-1]
    if ttt_size not in TTT_STANDARD_CONFIGS.keys():
        raise NotImplementedError(f"TTT Config {args.model_name} Not Implemented!")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    ttt_config = TttConfig(tie_word_embeddings=True, **TTT_STANDARD_CONFIGS[ttt_size])
    model = TttForCausalLM(ttt_config).to(device=device, dtype=dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 attn_implementation=args.attn_impl,
                                                 device_map={"": device},
                                                 torch_dtype=dtype)
model.eval()
logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")

max_length = input_ids.shape[1] + args.genlen

if is_mamba:
    # fn = lambda: model.generate(
    #     input_ids=input_ids,
    #     attention_mask=attn_mask,
    #     max_length=max_length,
    #     min_length=max_length,
    #     return_dict_in_generate=True,
    #     pad_token_id=tokenizer.eos_token_id,
    #     do_sample=False,
    #     num_beams=1,
    #     temperature=1.0,
    #     top_k=0,
    #     top_p=0,
    # )
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=True,
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        temperature=1.0,  # @xinhao: mamba src code: shortcut for greedy
        top_k=0,
        top_p=0,
    )
else:
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        min_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
out = fn()
logger.info(f"output.sequences.shape: {out.sequences.shape}")

torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()
avg_time = (time.time() - start) / repeats

logger.info(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
logger.info(f"{args.model_name} prompt processing + decoding time: {avg_time * 1000:.0f}ms")
logger.info(f"Throughput: {args.batch * len(out.sequences[0]) / avg_time:.3f} tokens / s")
logger.info("==================================")
