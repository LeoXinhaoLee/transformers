# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import pdb
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers.models.ttt.modeling_ttt import TttConfig, TttForCausalLM
from transformers.models.ttt.configuration_ttt import TTT_STANDARD_CONFIGS


parser = argparse.ArgumentParser(description="Generation benchmarking")
# parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--model-name", type=str, default="openai-community/gpt2")
# parser.add_argument("--model-name", type=str, default="ttt-125m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=1)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--attn_impl", type=str, default='eager')  # 'eager' | 'sdpa' | 'flash_attention_2'
args = parser.parse_args()

repeats = 3
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba-")
is_ttt = args.model_name.startswith("ttt")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
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
                                                 # attn_implementation=args.attn_impl,
                                                 device_map={"": device},
                                                 torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(1, 100, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)

max_length = input_ids.shape[1] + args.genlen
if is_mamba:
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        # min_length=max_length,  # @xinhao: Make sure an output has length=prompt_len+gen_len. Then fix seed for benchmarking speed.
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
    )
else:
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        min_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
    )
out = fn()
print(out.sequences.shape)
if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))

torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()
avg_time = (time.time() - start) / repeats
print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")
print(f"Throughput (total tok = prefill + decode): {args.batch * len(out.sequences[0]) / avg_time:.3f} tokens / s")
print(f"Throughput (total tok = decode): {args.batch * (len(out.sequences[0])  - len(input_ids[0])) / avg_time:.3f} tokens / s")
