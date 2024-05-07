'''Benchmarking Prefilling and Decoding
E.g.,
# Decoding
python benchmark_prefill_decode.py --logdir ./exp/decode_ttt_125m \
                                   --mode decode \
                                   --model-name ttt-125m \
                                   --inner_net mlp_2_dual \
                                   --batch 64 \
                                   --promptlen 1 \
                                   --genlen 512
# Prefilling
python benchmark_prefill_decode.py --logdir ./exp/prefill_ttt_125m \
                                   --mode prefill \
                                   --model-name ttt-125m \
                                   --inner_net mlp_2_dual \
                                   --batch 64 \
                                   --promptlen 512 \
                                   --genlen 0
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
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PretrainedConfig
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import MambaForCausalLM  # HF Mamaba (slow)

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # Mamba Package (fast)
# from transformers.models.ttt.modeling_ttt import TttConfig, TttForCausalLM
# from transformers.models.ttt.configuration_ttt import TTT_STANDARD_CONFIGS
# from transformers.models.ttt_benchmark.modeling_ttt import TttConfig, TttForCausalLM
# from transformers.models.ttt_benchmark.configuration_ttt import TTT_STANDARD_CONFIGS
from transformers.models.ttt_benchmark_cg.modeling_ttt import TttConfig, TttForCausalLM
from transformers.models.ttt_benchmark_cg.configuration_ttt import TTT_STANDARD_CONFIGS

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--logdir", type=str, default="./exp/clean")
parser.add_argument("--model-name", type=str, default="openai-community/gpt2")
# state-spaces/mamba-130m | meta-llama/Llama-2-7b | state-spaces/mamba-1.4b | ttt-125m | ttt-1b | ttt-profile
parser.add_argument("--mode", type=str, default="prefill", choices=["prefill", "decode"])
parser.add_argument("--promptlen", type=int, default=1)
parser.add_argument("--genlen", type=int, default=128)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--attn_impl", type=str, default='flash_attention_2', choices=['eager', 'flash_attention_2'])
parser.add_argument("--inner_net", type=str, default='mlp_2_dual', choices=['mlp_1_dual', 'mlp_2_dual', 'mlp_1_dual_triton'])
parser.add_argument("--use_compile", action='store_true')
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
if torch.cuda.is_available():
    device = "cuda"
else:
    device = 'cpu'
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
    if ttt_size == 'debug':
        ttt_size = '1b'
    elif ttt_size not in TTT_STANDARD_CONFIGS.keys():
        raise NotImplementedError(f"TTT Config {args.model_name} Not Implemented!")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    ttt_config = TttConfig(**TTT_STANDARD_CONFIGS[ttt_size])
    ttt_config.inner_net = args.inner_net
    ttt_config.use_compile = args.use_compile
    ttt_config.dtype = dtype
    if args.model_name.split('-')[-1] == 'debug':
        ttt_config.num_hidden_layers = 1
    model = TttForCausalLM(ttt_config).to(device=device, dtype=dtype)
else:
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # config = AutoConfig.from_pretrained(args.model_name)
    # config.n_positions = 4096
    # config.attn_implementation = args.attn_impl
    # model = GPT2LMHeadModel(config).to(device=device, dtype=dtype)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name,
    #                                              attn_implementation=args.attn_impl,
    #                                              device_map={"": device},
    #                                              torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # meta-llama/Llama-2-7b
    config = LlamaConfig.from_json_file('./llama_config/config.json')  # 1B llama config
    config._attn_implementation = args.attn_impl  # @xinhao: llama config use `_attn_implementation` to select attn
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)

model.eval()
logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device=device)
attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
max_length = input_ids.shape[1] + args.genlen

assert is_ttt, "profile only TTT"

if args.mode == 'decode':
    if is_mamba:
        # model = torch.compile(model)
        # fn = lambda: model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attn_mask,
        #     max_length=max_length,
        #     min_length=max_length,
        #     return_dict_in_generate=True,
        #     pad_token_id=tokenizer.eos_token_id,
        #     do_sample=False,
        #     use_cache=True,
        # )
        fn = lambda i: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=False,
            enable_timing=False,
            temperature=1.0,
            top_k=1, # @xinhao: mamba src code: shortcut for greedy
            top_p=0,
        )
    elif is_ttt:
        # model = torch.compile(model)
        # fn = lambda: model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attn_mask,
        #     use_cache=True,  # @xinhao: efficient decoding
        #     max_length=max_length,
        #     min_length=max_length,
        #     return_dict_in_generate=True,
        #     pad_token_id=tokenizer.eos_token_id,
        #     do_sample=False,
        # )
        fn = lambda i: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=False,
            enable_timing=False,
            temperature=1.0,
            top_k=1,  # @xinhao: mamba src code: shortcut for greedy
            top_p=0,
            i=i  # @xinhao: for debug output
        )
    else:
        # model = torch.compile(model)
        fn = lambda i: model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=max_length,
            min_length=max_length,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
elif args.mode == 'prefill':
    # model = torch.compile(model)
    if is_ttt:
        kwargs = {'input_ids': input_ids, 'is_prefill': True}
    else:
        kwargs = {'input_ids': input_ids}
    @torch.inference_mode()
    def fn(*args):
        model(**kwargs)
        return
else:
    raise NotImplementedError(f"Invalid Mode {args.mode}!")

out = fn(0)  # capture graph if cg=True, will not be timed
if args.mode == 'decode':
    logger.info(f"Decode succeeds. output.sequences.shape: {out.sequences.shape}")
    out_len = len(out.sequences[0])
else:
    logger.info("Prefill succeeds.")
    out_len = len(input_ids[0])
in_len = len(input_ids[0])
if args.mode == 'decode':
    print(tokenizer.batch_decode(out.sequences[0].tolist()[:5]))
    print('=================')
del out
torch.cuda.empty_cache()
gc.collect()

torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True, use_cuda=True,
             with_flops=True, with_stack=True, with_modules=True
    ) as prof:
       fn(0)
prof.export_chrome_trace(osp.join(args.logdir, f"{args.mode}_trace.json"))
prof.export_stacks(osp.join(args.logdir, f"{args.mode}_cuda_flamedata.txt"), "self_cuda_time_total")
prof.export_stacks(osp.join(args.logdir, f"{args.mode}_cpu_flamedata.txt"), "self_cpu_time_total")
torch.cuda.synchronize()

logger.info(f"Mode: {args.mode}")
logger.info(f"Prompt length: {in_len}, generation length: {out_len - in_len}")
logger.info(f"SUCCESS: RECORDED TRACE TO {args.logdir}/{args.mode}_trace.json")
logger.info(f"SUCCESS: RECORDED FLAME DATA TO {args.logdir}/{args.mode}_[cuda,cpu]_flamedata.txt")
logger.info("==================================")
