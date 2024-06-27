import torch
import matplotlib.pyplot as plt
import numpy as np

time_stats = torch.load('./exp/matmul_time/time_list.pth')

time_list = np.asarray(time_stats) * 1000  # ms
log_time_list = np.log2(time_list)

token_num_list = np.asarray([2**i for i in range(10, 21)])
log_token_num_list = np.log2(token_num_list)

ratio_list = time_list / token_num_list
log_ratio_list = log_time_list / log_token_num_list

# fig, ax1 = plt.subplots()
# ax1.plot(log_token_num_list, log_time_list)
# ax1.set_xticklabels(token_num_list)
# ax1.set_xlabel('N (log scale)')
# ax1.set_ylabel('Time (ms, log scale)')  # ms
# ax1.tick_params(axis='y', labelcolor='b')
#
# ax2 = ax1.twinx()
# ax2.plot(log_token_num_list, log_ratio_list, 'r')
# ax2.set_ylabel('ms/token', color='r')
# ax2.tick_params(axis='y', labelcolor='r')

i=3
fig, ax2 = plt.subplots(figsize=(8,6))
ax2.plot(log_token_num_list[i:], ratio_list[i:])
# ax2.plot(log_token_num_list[i:], log_ratio_list[i:])
ax2.set_ylabel('ms/token', fontsize=15)
ax2.set_xticks(log_token_num_list[i:], token_num_list[i:])
ax2.set_xlabel('Token Num (log scale)', fontsize=15)

ax2.set_title('1 matmul in PyTorch: [N,2048] @ [2048,2048]')

plt.show()
