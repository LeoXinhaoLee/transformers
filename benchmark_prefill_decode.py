'''Benchmarking Prefilling and Decoding
E.g.,
# Prefilling
python benchmark_prefill_decode.py --logdir ./exp/prefill_ttt_125m \
                                   --mode prefill \
                                   --model-name ttt-1b \
                                   --inner_net mlp_1_dual_triton \
                                   --batch 64 \
                                   --promptlen 512 \
                                   --genlen 0 \
                                   --use_compile

# Decoding
python benchmark_prefill_decode.py --logdir ./exp/decode_ttt_125m \
                                   --mode decode \
                                   --model-name ttt-1b \
                                   --inner_net mlp_2_dual_triton \
                                   --batch 64 \
                                   --promptlen 1 \
                                   --genlen 512 \
                                   --use_compile
'''
import gc
import pdb

import argparse
import os
import os.path as osp
import time
import json
import logging

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PretrainedConfig
from transformers import LlamaForCausalLM, LlamaConfig

from transformers.models.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # copy from mamba repo, modify generation to avoid OOM

from transformers.models.ttt_clean.configuration_ttt import TTT_STANDARD_CONFIGS, TTTConfig
from transformers.models.ttt_clean import TTTForCausalLM

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--logdir", type=str, default="./exp/clean")
parser.add_argument("--model-name", type=str, default="ttt-1b")
# state-spaces/mamba-130m | meta-llama/Llama-2-7b | state-spaces/mamba-1.4b | ttt-125m | ttt-1b | ttt-profile
parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
parser.add_argument("--promptlen", type=int, default=1)
parser.add_argument("--genlen", type=int, default=128)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--attn_impl", type=str, default='flash_attention_2', choices=['eager', 'flash_attention_2'])
parser.add_argument("--inner_net", type=str, default='mlp_2_dual', choices=['mlp_1_dual', 'mlp_2_dual',
                                                                            'mlp_1_dual_tk', 'mlp_2_dual_tk'])
parser.add_argument("--use_compile", action='store_true')
parser.add_argument("--no_cg", action='store_true')    # @xinhao: currently only implemented for Mamba and TTT
parser.add_argument("--profile", action='store_true')  # @xinhao: pytorch profiler, different from nsys in micro-benchmark
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
if args.dtype == 'fp16':
    dtype = torch.float16  # @xinhao: follow mamba benchmark
elif args.dtype == 'bf16':
    dtype = torch.bfloat16
else:
    raise NotImplementedError
logger.info("dtype: " + str(dtype))

logger.info(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba-")
is_ttt = args.model_name.startswith("ttt")
if is_mamba:
    assert not args.use_compile, "Mamba does not support torch.compile!"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
    ## 1.4B
    config = {
        "d_model": 2048,
        "n_layer": 48,
        "vocab_size": 32000,  # llama2 tokenizer's vocab size
        "ssm_cfg": {},
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 8
    }
    model = MambaLMHeadModel(**config, device=device, dtype=dtype)
elif is_ttt:
    ttt_size = args.model_name.split('-')[-1]
    if ttt_size == 'profile':
        ttt_size = '1b'
    elif ttt_size not in TTT_STANDARD_CONFIGS.keys():
        raise NotImplementedError(f"TTT Config {args.model_name} Not Implemented!")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    ttt_config = TTTConfig(**TTT_STANDARD_CONFIGS[ttt_size], vocab_size=32000)
    ttt_config.inner_net = args.inner_net
    ttt_config.use_compile = args.use_compile
    ttt_config.dtype = dtype
    # @xinhao: follow mamba-1.4b
    ttt_config.fused_add_norm = True
    ttt_config.residual_in_fp32 = True
    if args.model_name.split('-')[-1] == 'profile':
        ttt_config.num_hidden_layers = 1
    model = TTTForCausalLM(ttt_config).to(device=device, dtype=dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # meta-llama/Llama-2-7b
    config = LlamaConfig.from_json_file('./llama_config/config.json')  # 1B llama config, vocab size=32000
    config._attn_implementation = args.attn_impl  # @xinhao: llama config use `_attn_implementation` to select attn
    config.dtype = dtype
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)

model.eval()
logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device=device)
attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
max_length = input_ids.shape[1] + args.genlen

if is_mamba:
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=(not args.no_cg),
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        temperature=1.0,
        top_k=1, # @xinhao: mamba src code: shortcut for greedy
        top_p=0,
    )
elif is_ttt:
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=(not args.no_cg),
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        temperature=1.0,
        top_k=1,  # @xinhao: mamba src code: shortcut for greedy
        top_p=0,
    )
else:
    if args.use_compile:
        model = torch.compile(model)  # @xinhao: can compile the whole Transformer for decode, though doesn't help
    if not args.no_cg:
        logger.info(f"CUDA Graph Not Implemented for Transformers")
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        min_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )


out = fn()  # capture graph if cg=True, will not be timed
logger.info("Succeeded.")
out_len = len(out.sequences[0])
in_len = len(input_ids[0])
del out

if args.profile:

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True, use_cuda=True,
                 with_flops=True, with_stack=True, with_modules=True
        ) as prof:
           fn()
    prof.export_chrome_trace(osp.join(args.logdir, f"trace.json"))
    prof.export_stacks(osp.join(args.logdir, f"cuda_flamedata.txt"), "self_cuda_time_total")
    prof.export_stacks(osp.join(args.logdir, f"cpu_flamedata.txt"), "self_cpu_time_total")
    torch.cuda.synchronize()

    logger.info(f"Prompt length: {in_len}, generation length: {out_len - in_len}")
    logger.info(f"SUCCESS: RECORDED TRACE TO {args.logdir}/trace.json")
    logger.info(f"SUCCESS: RECORDED FLAME DATA TO {args.logdir}/[cuda,cpu]_flamedata.txt")
    logger.info("==================================")

else:

    torch.cuda.synchronize()
    start = time.time()
    for i in range(repeats):
        fn()
    torch.cuda.synchronize()
    avg_time = (time.time() - start) / repeats

    logger.info(f"Prompt length: {in_len}, generation length: {out_len - in_len}")
    logger.info(f"Prompt processing + Decoding time: {avg_time * 1000:.0f}ms")
    logger.info(f"Throughput (total tok = prefill): {args.batch * in_len / avg_time:.3f} tokens / s")
    logger.info(f"Throughput (total tok = prefill + decode): {args.batch * out_len / avg_time:.3f} tokens / s")
    logger.info(f"Throughput (total tok = decode): {args.batch * (out_len - in_len) / avg_time:.3f} tokens / s")
    logger.info("==================================")
