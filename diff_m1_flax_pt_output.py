"""Compare the output of Flax and PyTorch LLaMA models.

# disable cuda for precision
CUDA_VISIBLE_DEVICES= python diff_m1_flax_pt_output.py

"""

import torch

torch.set_printoptions(precision=8)
import numpy as np
import jax
import jax.numpy as jnp
import flax
from addict import Dict
from transformers import AutoTokenizer
from transformers.models.ttt import TttForCausalLM

from EasyLM.infra.checkpoint import StreamingCheckpointer
from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLM
from EasyLM.jax_utils import JaxRNG, next_rng, set_random_seed

use_post_ln = True
inner_net_on_residual = False

flax_args = Dict()

flax_args.input_length = 1024
flax_args.seq_length = 2048
# flax_args.input_length = 16
# flax_args.seq_length = 32
flax_args.seed = 42
flax_args.llama_config_update = dict(
    inner_net="mlp_1_dual",
    ilr=1.0,
    max_sequence_length=flax_args.seq_length,
    remat_chunk_group_size=1,
    post_LN=use_post_ln,
    inner_net_on_residual=inner_net_on_residual,
)

pt_args = Dict()

model_size = "1b-TTT"
flax_args.weight_path = "trainstate_params::/nlp/scr/yusun/data/jiarui/easylm_ckpts/04_11_D_300B_ctx_2048_BS_512_M1_Dual_lr_1e-3_ilr_1.0/streaming_train_state_120000"
pt_args.weight_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts/04_11_D_300B_ctx_2048_BS_512_M1_Dual_lr_1e-3_ilr_1.0/hf_120000"



def forward_flax_token(input_tokens, input_mask):
    set_random_seed(flax_args.seed)
    sharded_rng = next_rng()

    llama_config = LLaMAConfig.load_config(model_size)
    llama_config.update(dict(vocab_size=50277))
    llama_config.update(dict(flax_args.llama_config_update))
    _, params = StreamingCheckpointer.load_trainstate_checkpoint(
        flax_args.weight_path, disallow_trainstate=True
    )
    params = jax.tree_map(lambda x: x.astype(jnp.float32), params)
    flax_hf_model = FlaxLLaMAForCausalLM(
        llama_config,
        input_shape=(1, flax_args.seq_length),
        seed=flax_args.seed,
        _do_init=False,
    )
    rng_generator = JaxRNG(sharded_rng)
    logits = flax_hf_model.module.apply(
        params,
        input_tokens,
        attention_mask=input_mask,
        deterministic=True,
        rngs=rng_generator(llama_config.rng_keys()),
    ).logits

    logits = jax.device_get(logits)
    return logits


@torch.no_grad()
def forward_pt_token(input_tokens, input_mask):
    model = TttForCausalLM.from_pretrained(
        pt_args.weight_path,
        torch_dtype=torch.float32,
        device_map="auto",
        use_post_ln=use_post_ln,
        inner_net_on_residual=inner_net_on_residual,
    )
    input_tokens = torch.from_numpy(input_tokens).to(model.device)
    input_mask = torch.from_numpy(input_mask).to(model.device)
    logits = model(input_tokens, attention_mask=input_mask).logits

    return logits.detach().cpu().numpy()


if __name__ == "__main__":
    prefix_text = "The correct answer is I am a student:"
    text = "42 and 42."

    prefix_tokenizer = AutoTokenizer.from_pretrained(
        pt_args.weight_path, truncation_side="left", padding_side="left"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pt_args.weight_path, truncation_side="right", padding_side="right"
    )
    prefix_tokenizer.pad_token_id = prefix_tokenizer.eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix = prefix_tokenizer(
        prefix_text,
        padding="max_length",
        truncation=True,
        max_length=flax_args.input_length,
        return_tensors="np",
    )
    print("prefix", prefix)
    # inputs = tokenizer(text, return_tensors="np")
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=flax_args.seq_length - flax_args.input_length,
        return_tensors="np",
    )
    print("inputs", inputs)

    input_tokens = np.concatenate([prefix.input_ids, inputs.input_ids], axis=1)
    input_mask = np.concatenate([prefix.attention_mask, inputs.attention_mask], axis=1)
    print("input_tokens", input_tokens)
    print("input_mask", input_mask)

    flax_logits = forward_flax_token(input_tokens, input_mask)
    pt_logits = forward_pt_token(input_tokens, input_mask)
    print("err", np.abs(flax_logits - pt_logits).max())
    print("all close:", np.allclose(flax_logits, pt_logits))
    print("apply masking")
    masked_flax_logits = flax_logits * input_mask[..., None]
    masked_pt_logits = pt_logits * input_mask[..., None]
    print("err", np.abs(masked_flax_logits - masked_pt_logits).max())
    print("all close:", np.allclose(masked_flax_logits, masked_pt_logits))
    import ipdb

    ipdb.set_trace()
    pass