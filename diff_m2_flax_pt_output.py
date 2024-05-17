"""Compare the output of Flax and PyTorch LLaMA models.

# disable cuda for precision
CUDA_VISIBLE_DEVICES= python diff_m2_flax_pt_output.py

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
inner_net_on_residual = True

flax_args = Dict()

flax_args.input_length = 1024
flax_args.seq_length = 2048
# flax_args.input_length = 16
# flax_args.seq_length = 32
flax_args.seed = 42
flax_args.llama_config_update = dict(
    inner_net="mlp_2_mixer_linear_dual",
    # inner_net="mlp_2_dual",
    ilr=0.1,
    max_sequence_length=flax_args.seq_length,
    remat_chunk_group_size=1,
    use_rotary_emb="chunk",
    # tie_word_embeddings=False,
    # num_hidden_layers=22,
    # num_attention_heads=16,
    post_LN=use_post_ln,
    inner_net_on_residual=inner_net_on_residual,
    # use_learnable_token_idx="learnable_transpose",
    # ilr_gate_act="sigmoid",
    conv1d_before_attn=True,
    # qk_conv1d_width=4,
    # conv1d_width=4,
    use_bmm=True,
    use_learnable_token_idx="fix",
)

pt_args = Dict()

model_size = "125m-TTT"
# flax_args.weight_path = "trainstate_params::/nlp/scr/yusun/data/jiarui/easylm_ckpts/LLAMA-125M/05_07_Tok_llama2_D_2.5B_ctx_2048_BS_256_c1d_k4_M2MixerLinear_Dual_share_qk_qk4_token_idx_learnable_transpose_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/streaming_train_state_4800"
# pt_args.weight_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts/LLAMA-125M/05_07_Tok_llama2_D_2.5B_ctx_2048_BS_256_c1d_k4_M2MixerLinear_Dual_share_qk_qk4_token_idx_learnable_transpose_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/hf_4800"
# flax_args.weight_path = "trainstate_params::/nlp/scr/yusun/data/jiarui/easylm_ckpts/LLAMA-125M/05_02_Tok_llama2_D_2.5B_ctx_2048_BS_256_M2_Dual_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/streaming_train_state_4800"
# pt_args.weight_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts/LLAMA-125M/05_02_Tok_llama2_D_2.5B_ctx_2048_BS_256_M2_Dual_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/hf_4800"
flax_args.weight_path = "trainstate_params::/nlp/scr/yusun/data/jiarui/easylm_ckpts/LLAMA-125M/05_15_Tok_llama2_D_2.5B_ctx_2048_BS_256_c1d_k4_M2MixerLinear_Dual_bmm_share_qk_qk4_token_idx_fix_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/streaming_train_state_4800"
pt_args.weight_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts/LLAMA-125M/05_15_Tok_llama2_D_2.5B_ctx_2048_BS_256_c1d_k4_M2MixerLinear_Dual_bmm_share_qk_qk4_token_idx_fix_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/hf_4800"
pt_args.model_args = dict(
    inner_net_type="m2",
    inner_net_on_residual=inner_net_on_residual,
    use_post_ln=use_post_ln,
)
# pt_args.model_args = dict(inner_net_type='m1_bmm', inner_net_on_residual=inner_net_on_residual, use_post_ln=use_post_ln)


def forward_flax_token(input_tokens, input_mask):
    set_random_seed(flax_args.seed)
    sharded_rng = next_rng()

    llama_config = LLaMAConfig.load_config(model_size)
    # llama_config.update(dict(vocab_size=50277))
    update_dic = flax_args.llama_config_update
    # update_dic has to overlap with llama_config
    update_keys = set(update_dic.keys())
    original_keys = set(llama_config.__dict__.keys())
    assert update_keys.issubset(
        original_keys
    ), f"Update keys {update_keys-original_keys} not in llama_config"
    llama_config.update(update_dic)
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
    print('forward_pt_token')
    model = TttForCausalLM.from_pretrained(
        pt_args.weight_path,
        torch_dtype=torch.float32,
        device_map="auto",
        **pt_args.model_args,
    )
    print('model', model)
    print('model.config', model.config)
    input_tokens = torch.from_numpy(input_tokens).to(model.device)
    input_mask = torch.from_numpy(input_mask).to(model.device)
    logits = model(input_tokens, attention_mask=input_mask).logits

    return logits.detach().cpu().numpy()


if __name__ == "__main__":
    prefix_text = "The correct answer is I am a student:" * 1000
    text = "42 and 42." * 10000

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

    pt_logits = forward_pt_token(input_tokens, input_mask)
    flax_logits = forward_flax_token(input_tokens, input_mask)
    print('flax_logits', np.abs(flax_logits).max())
    print('pt_logits', np.abs(pt_logits).max())
    print("err", np.abs(flax_logits - pt_logits).max())
    print("all close:", np.allclose(flax_logits, pt_logits))
    print("apply masking")
    masked_flax_logits = flax_logits * input_mask[..., None]
    masked_pt_logits = pt_logits * input_mask[..., None]
    print("err", np.abs(masked_flax_logits - masked_pt_logits).max())
    print("all close:", np.allclose(masked_flax_logits, masked_pt_logits))

    print("apply argmax")
    argmax_flax_logits = flax_logits.argmax(-1)
    argmax_pt_logits = pt_logits.argmax(-1)
    print("err", np.abs(argmax_flax_logits - argmax_pt_logits).max())
    print("all close:", np.allclose(argmax_flax_logits, argmax_pt_logits))

    import ipdb

    ipdb.set_trace()
    pass
