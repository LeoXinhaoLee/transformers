"""Compare the output of PyTorch TTT w/ and w/o VJP

python test_m1_bmm.py

"""
import time
import torch
torch.set_printoptions(precision=8)
import numpy as np
from addict import Dict
from transformers import AutoTokenizer
from transformers.models.ttt import TttForCausalLM
import tqdm

pt_args = Dict()
pt_args.input_length = 1024
pt_args.seq_length = 2048
pt_args.weight_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts/04_11_D_300B_ctx_2048_BS_512_M1_Dual_lr_1e-3_ilr_1.0/hf_120000"


@torch.no_grad()
def forward_pt_token(input_tokens, input_mask, **kwargs):
    model = TttForCausalLM.from_pretrained(
        pt_args.weight_path, torch_dtype=torch.float32, device_map='auto',
        **kwargs
    )
    model.to('cuda')
    input_tokens = torch.from_numpy(input_tokens).to(model.device)
    input_mask = torch.from_numpy(input_mask).to(model.device)
    start_time = time.time()
    for _ in tqdm.tqdm(range(1)):
    # for _ in range(100):
        logits = model(input_tokens, attention_mask=input_mask).logits
    print(f'pt time {kwargs}', time.time() - start_time)

    return logits.detach().cpu().numpy()


if __name__ == '__main__':
    prefix_text = 'The correct answer is I am a student:'
    text = '42 and 42.'
    # prefix_text = ""
    # text = ""

    # flax_logits = forward_flax(prefix_text, text)
    # pt_logits = forward_pt(prefix_text, text)
    # print('err', np.abs(flax_logits - pt_logits).max())
    # print('all close:', np.allclose(flax_logits, pt_logits))

    prefix_tokenizer = AutoTokenizer.from_pretrained(pt_args.weight_path, truncation_side='left', padding_side='left')
    tokenizer = AutoTokenizer.from_pretrained(pt_args.weight_path, truncation_side='right', padding_side='right')
    prefix_tokenizer.pad_token_id = prefix_tokenizer.eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # prefix = prefix_tokenizer(prefix_text, return_tensors="np")
    prefix = prefix_tokenizer(
        prefix_text,
        padding='max_length',
        truncation=True,
        max_length=pt_args.input_length,
        return_tensors='np',
    )
    print('prefix', prefix)
    # inputs = tokenizer(text, return_tensors="np")
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=pt_args.seq_length - pt_args.input_length,
        return_tensors='np',
    )
    print('inputs', inputs)

    input_tokens = np.concatenate([prefix.input_ids, inputs.input_ids], axis=1)
    input_mask = np.concatenate(
        [prefix.attention_mask, inputs.attention_mask], axis=1
    )
    print('input_tokens', input_tokens)
    print('input_mask', input_mask)

    pt_logits_a = forward_pt_token(input_tokens, input_mask, inner_net_type='m1')
    pt_logits_b = forward_pt_token(input_tokens, input_mask, inner_net_type='m1_bmm')
    print('pt_logits_a', np.abs(pt_logits_a).max())
    print('pt_logits_b', np.abs(pt_logits_b).max())
    print('err', np.abs(pt_logits_a - pt_logits_b).max())
    print('all close:', np.allclose(pt_logits_a, pt_logits_b))

    pt_logits_a = forward_pt_token(input_tokens, input_mask, inner_net_type='m1', inner_net_on_residual=True, use_post_ln=True)
    pt_logits_b = forward_pt_token(input_tokens, input_mask, inner_net_type='m1_bmm', inner_net_on_residual=True, use_post_ln=True)
    print('pt_logits_a', np.abs(pt_logits_a).max())
    print('pt_logits_b', np.abs(pt_logits_b).max())
    print('err', np.abs(pt_logits_a - pt_logits_b).max())
    print('all close:', np.allclose(pt_logits_a, pt_logits_b))
    import ipdb; ipdb.set_trace()
    pass
