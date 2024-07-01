import torch
import matplotlib.pyplot as plt
import numpy as np

time_stats = torch.load('./exp/matmul_time/06_30_matmul_time_list.pth')

time_list = np.asarray(time_stats) * 1000  # ms
log_time_list = np.log2(time_list)

token_num_list = np.asarray([2**i for i in range(10, 22)])
log_token_num_list = np.log2(token_num_list)

ratio_list = time_list / token_num_list
log_ratio_list = log_time_list / log_token_num_list

i=3
fig, ax2 = plt.subplots(figsize=(8,6))
ax2.plot(log_token_num_list[i:], ratio_list[i:])
ax2.set_ylabel('ms/token', fontsize=15)
ax2.set_xticks(log_token_num_list[i:], token_num_list[i:])
ax2.set_xlabel('Token Num (log scale)', fontsize=15)

ax2.set_title('1 matmul in PyTorch: [N,2048] @ [2048,2048]')

plt.show()
