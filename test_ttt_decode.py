import os
import torch
import time
from transformers.models.ttt.modeling_ttt import TttConfig, TttForCausalLM
from transformers import AutoTokenizer, GenerationConfig
TTT_STANDARD_CONFIGS = {
    'debug': { # A small model for debugging
        'vocab_size': 32000,
        'hidden_size': 128,
        'intermediate_size': 256,
        'num_hidden_layers': 1,
        'num_attention_heads': 4,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
        'inner_net_lr': 1.,
        'inner_net_chunk_size': 16,
        'inner_net_intermediate_size': 128,
    },
}

# model_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts/04_11_D_300B_ctx_2048_BS_512_M1_Dual_lr_1e-3_ilr_1.0/hf_307200"
# model_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts/LLAMA-125M/05_02_Tok_llama2_D_2.5B_ctx_2048_BS_256_M2_Dual_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/hf_4800"
model_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts/LLAMA-125M/05_15_Tok_llama2_D_2.5B_ctx_2048_BS_256_c1d_k4_M2MixerLinear_Dual_bmm_share_qk_qk4_token_idx_fix_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/hf_4800"

if __name__ == '__main__':
    # config = TttConfig.from_dict(TTT_STANDARD_CONFIGS['debug'])
    # model = TttForCausalLM(config)
    model = TttForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to('cuda')
    
    if tokenizer.pad_token:
        print('pad token:', tokenizer.pad_token)
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        print('pad token to unk token', tokenizer.pad_token_id)
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print('pad token to eos token', tokenizer.pad_token_id)
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = 'right'
    print('pad_token_id:', tokenizer.pad_token_id)
    # prompt = ["Hey, are you conscious? Can you talk to me?", "I am Jerry."]
    prompt = "Hey, are you conscious? Can you talk to me?"
    # inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=30, truncation=True).to('cuda')
    # test_length = 100
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=12, truncation=True).to('cuda')
    test_length = 100
    # inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).to('cuda')
    # test_length = 100
    print('inputs:', inputs.input_ids.shape)
    generation_config=GenerationConfig(pad_token_id=0)

    # Generate using cache
    start_time = time.time()
    cache_generate_ids = model.generate(**inputs, max_length=test_length, min_length=test_length, do_sample=False, use_cache=True, generation_config=generation_config)
    print('Time taken to generate using cache:', time.time() - start_time)

    # Generate without using cache
    start_time = time.time()
    no_cache_generate_ids = model.generate(**inputs, max_length=test_length, min_length=test_length, do_sample=False, use_cache=False, generation_config=generation_config)
    print('Time taken to generate without cache:', time.time() - start_time)


    print('no_cache_generate_ids:', no_cache_generate_ids.shape)
    print(tokenizer.batch_decode(no_cache_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    print('cache_generate_ids:', cache_generate_ids.shape)
    print(tokenizer.batch_decode(cache_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    print(cache_generate_ids - no_cache_generate_ids)
    print(cache_generate_ids[0, inputs.input_ids.shape[1]:] - no_cache_generate_ids[0, inputs.input_ids.shape[1]:])
