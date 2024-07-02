import numpy as np
import matplotlib.pyplot as plt

### Whole Model Decode ###
# m1 = [31536.435, 31584.523, 31687.193, 31621.184, 31640.295]
# m2 = [4681.657, 4689.174, 4684.998, 4690.487, 4686.465]
# mamba = [29611.113, 31277.532, 31924.04, 32599.265, 32817.916]

# m1 = [1685309.255, 1690814.776, 1693492.523, 1695379.812, 1695562.402, 1696025.261, 1696386.371, 1696386.371, 1695699.992]
# m2 = [125087.263, 125265.416, 125391.421, 125482.684, 125462.628, 125524.102, 125483.874, 125503.408]
# mamba = [1683773.002, 1681407.654, 1674452.888, 1657590.697, 1648460.108, 1633065.265, 1616301.516, 1618167.486, 1619191.282]
# x_ticks_lb = ['512', '1k', '2k', '4k', '8k', '16k', '24k', '32k', '40k']

### 1B whole model ###
# mode = 'Prefill'
# # y_mode = 'abs'
# y_mode = 'norm'
# m1 = [57467.817, 57016.971, 56175.784, 55157.172, 55062.676]
# m2 = [32196.391, 32267.017, 31664.331, 31232.128, 31170.997]
# mamba = [52280.759, 51990.344, 51762.822, 51276.272, 50915.416]
# x_ticks_lb = ['2k', '4k', '8k', '16k', '32k']

mode = 'Decode'
# y_mode = 'abs'
y_mode = 'norm'
m1 = [31536.435, 31584.523, 31687.193, 31621.184, 31640.295]
m2 = [4681.657, 4689.174, 4684.998, 4690.487, 4686.465]
mamba = [33185.693, 33106.481, 33065.511, 33058.662, 33066.166]
x_ticks_lb = ['512', '1k', '2k', '4k', '8k']

x_axis = np.arange(len(m1)) + 1
fig, ax = plt.subplots()
if y_mode == 'norm':
    ax.plot(np.arange(len(m1)) + 1, np.asarray(m1) / m1[0] * 100, label='M1')
    ax.plot(np.arange(len(m2)) + 1, np.asarray(m2) / m2[0] * 100, label='M2')
    ax.plot(np.arange(len(mamba)) + 1, np.asarray(mamba) / mamba[0] * 100, label='Mamba')
    ax.set_ylabel('token/sec (normalized, %)', fontsize=15)  # token/sec
    ax.set_title(f'{mode} Throughput (normalized by throughput at N=512)')
else:
    ax.plot(np.arange(len(m1)) + 1, np.asarray(m1), label='M1')
    ax.plot(np.arange(len(m2)) + 1, np.asarray(m2), label='M2')
    ax.plot(np.arange(len(mamba)) + 1, np.asarray(mamba), label='Mamba')
    ax.set_ylabel('token/sec', fontsize=15)  # token/sec
    ax.set_title(f'{mode} Throughput')

ax.set_xticks(x_axis, x_ticks_lb)
ax.set_xlabel(f'{mode} Length', fontsize=15)
ax.legend()

plt.show()
