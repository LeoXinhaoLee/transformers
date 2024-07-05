import pdb
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

dir = './exp/A_matching'

### 07/04
## M1
# name = '07_04_M1_1B_decode_no_cg_no_compile'
# name = '07_04_M1_1B_decode_no_cg'
# name = '07_04_M1_1B_decode_no_compile'
# name = '07_04_M1_1B_decode'

# name = '07_04_M1_1B_prefill_no_compile'
# name = '07_04_M1_1B_prefill'

## M2
# name = '07_04_M2_1B_decode_no_compile_no_cg'
# name = '07_04_M2_1B_decode'

# name = '07_04_M2_no_gelu_1B_prefill_no_compile'
name = '07_04_M2_no_diff_gelu_1B_prefill_no_compile'
# name = '07_04_M2_1B_prefill_no_compile'
# name = '07_04_M2_1B_prefill'

folder = os.path.join(dir, name)
if 'prefill' in folder:
    mode = 'prefill'
else:
    mode = 'decode'

all_stats = torch.load(f'{folder}/all_stats.pth')

prompt_probs_avg_max_diff = all_stats['prompt_probs_avg_max_diff']  # [prompt_len,]
# prompt_probs_avg_mean_diff = all_stats['prompt_probs_avg_mean_diff']  # [prompt_len,]
prompt_token_diff = all_stats['prompt_token_diff']

decode_probs_avg_max_diff = all_stats['decode_probs_avg_max_diff']  # [gen_len,]
# decode_probs_avg_mean_diff = all_stats['decode_probs_avg_mean_diff']  # [gen_len,]
decode_token_diff = all_stats['decode_token_diff']


if mode == 'prefill':
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    x_axis = np.arange(len(prompt_probs_avg_max_diff))

    ax[0].plot(x_axis, prompt_probs_avg_max_diff)
    ax[0].set_xlabel('T')
    ax[0].set_title('Prompt Probs Avg Max Diff')

    ax[1].plot(x_axis, prompt_token_diff)
    ax[1].set_xlabel('T')
    ax[1].set_title('Prompt Token Diff')

    plt.show()

elif mode == 'decode':
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    x_axis = np.arange(len(decode_probs_avg_max_diff))

    ax[0].plot(x_axis, decode_probs_avg_max_diff)
    ax[0].set_xlabel('T')
    ax[0].set_title('Decode Probs Avg Max Diff')

    ax[1].plot(x_axis, decode_token_diff)
    ax[1].set_xlabel('T')
    ax[1].set_title('Decode Token Diff')

    plt.show()
