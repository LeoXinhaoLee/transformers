import argparse
import os
import os.path as osp
import time
import logging

import torch

from transformers.models.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # copy from mamba repo, modify generation to avoid OOM
from transformers.models.ttt_clean.configuration_ttt import TTT_STANDARD_CONFIGS, TTTConfig
from transformers.models.ttt_clean import TTTForCausalLM


parser = argparse.ArgumentParser(description="Speed Benchmark")
parser.add_argument("--logdir", type=str, default="./exp/clean")
parser.add_argument("--model-name", type=str, default="ttt", choices=['ttt-1.3b', 'mamba-1.3b'])
parser.add_argument("--promptlen", type=int, default=1)
parser.add_argument("--genlen", type=int, default=128)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--seq_modeling_block", type=str,
                    default='mlp_2_dual', choices=['ttt_linear', 'ttt_mlp', 'ttt_linear_fast', 'ttt_mlp_fast'])
parser.add_argument("--use_compile", action='store_true')
parser.add_argument("--no_cg", action='store_true')
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

torch.random.manual_seed(0)
# follow mamba's benchmark: https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/benchmarks/benchmark_generation_mamba_simple.py#L32
repeats = 3
device = "cuda"
dtype = torch.float16
logger.info("dtype: " + str(dtype))

is_mamba = args.model_name.startswith("mamba")
is_ttt = args.model_name.startswith("ttt")
if is_mamba:
    assert not args.use_compile, "Mamba does not support torch.compile!"
    # Copied from https://huggingface.co/state-spaces/mamba-1.4b/blob/main/config.json
    # except changing vocab size to llama2 tokenizer's vocab size
    config = {
        "d_model": 2048,
        "n_layer": 48,
        "vocab_size": 32000,
        "ssm_cfg": {},
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 8
    }
    model = MambaLMHeadModel(**config, device=device, dtype=dtype)
elif is_ttt:
    ttt_config = TTTConfig(**TTT_STANDARD_CONFIGS['1b'], vocab_size=32000)
    ttt_config.seq_modeling_block = args.seq_modeling_block
    ttt_config.use_compile = args.use_compile
    ttt_config.dtype = dtype
    ttt_config.fused_add_norm = True
    ttt_config.residual_in_fp32 = True
    model = TTTForCausalLM(ttt_config).to(device=device, dtype=dtype)

model.eval()
logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device=device)
attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
max_length = input_ids.shape[1] + args.genlen

fn = lambda: model.generate(
    input_ids=input_ids,
    max_length=max_length,
    cg=(not args.no_cg),
)

out = fn()  # capture graph if cg=True, will not be timed
logger.info("Succeeded.")
out_len = len(out.sequences[0])
in_len = len(input_ids[0])
del out

torch.cuda.synchronize()
start = time.time()
for i in range(repeats):
    fn()
torch.cuda.synchronize()
avg_time = (time.time() - start) / repeats

logger.info(f"Prompt length: {in_len}, generation length: {out_len - in_len}")
logger.info(f"Prompt processing + Decoding time: {avg_time * 1000:.0f}ms")
logger.info(f"Throughput (total tok = prefill): {args.batch * in_len / avg_time:.3f} tokens / s")
logger.info(f"Throughput (total tok = decode): {args.batch * (out_len - in_len) / avg_time:.3f} tokens / s")
logger.info("==================================")
