import pdb

import numpy as np
import torch
import matplotlib.pyplot as plt

# folder = '07_01_M1_1B_prefill_match_BS_32_plen_512'
# folder = '07_02_M1_1B_decode_match_BS_32_glen_512'
# folder = '07_02_M1_1B_prefill_match_BS_32_plen_512_runpod'

# folder = '07_02_M2_1B_prefill_match_BS_32_plen_512'
# folder = '07_02_M2_1B_decode_match_BS_32_glen_512_no_cg'
# folder = '07_02_M2_1B_decode_match_BS_32_glen_512'

# folder = '07_03_M1_1B_decode_match_BS_32_glen_512_conv_no_cg'
folder = '07_03_M1_1B_decode_match_BS_32_glen_512_conv'

if 'prefill' in folder:
    mode = 'prefill'
else:
    mode = 'decode'

all_stats = torch.load(f'./exp/{folder}/all_stats.pth')

prompt_logits_max_diff = all_stats['prompt_logits_max_diff']  # [prompt_len,]
prompt_logits_median_diff = all_stats['prompt_logits_median_diff']  # [prompt_len,]
prompt_logits_mse_diff = all_stats['prompt_logits_mse_diff']
prompt_token_diff = all_stats['prompt_token_diff']
decode_logits_max_diff = all_stats['decode_logits_max_diff']  # [gen_len,]
decode_logits_median_diff = all_stats['decode_logits_median_diff']  # [gen_len,]
decode_logits_mse_diff = all_stats['decode_logits_mse_diff']
decode_token_diff = all_stats['decode_token_diff']


if mode == 'prefill':
    fig, ax = plt.subplots(1,4, figsize=(20,5))
    x_axis = np.arange(len(prompt_logits_max_diff))

    ax[0].plot(x_axis, prompt_logits_max_diff)
    ax[0].set_xlabel('T')
    ax[0].set_title('Prompt Logits Max Diff')

    ax[1].plot(x_axis, prompt_logits_median_diff)
    ax[1].set_xlabel('T')
    ax[1].set_title('Prompt Logits Median Diff')

    ax[2].plot(x_axis, prompt_logits_mse_diff)
    ax[2].set_xlabel('T')
    ax[2].set_title('Prompt Logits MSE Diff')

    ax[3].plot(x_axis, prompt_token_diff)
    ax[3].set_xlabel('T')
    ax[3].set_title('Prompt Token Diff')

    plt.show()

elif mode == 'decode':
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    x_axis = np.arange(len(decode_logits_max_diff))

    ax[0].plot(x_axis, decode_logits_max_diff)
    ax[0].set_xlabel('T')
    ax[0].set_title('Decode Logits Max Diff')

    ax[1].plot(x_axis, decode_logits_median_diff)
    ax[1].set_xlabel('T')
    ax[1].set_title('Decode Logits Median Diff')

    ax[2].plot(x_axis, decode_logits_mse_diff)
    ax[2].set_xlabel('T')
    ax[2].set_title('Decode Logits MSE Diff')

    ax[3].plot(x_axis, decode_token_diff)
    ax[3].set_xlabel('T')
    ax[3].set_title('Decode Token Diff')

    plt.show()
