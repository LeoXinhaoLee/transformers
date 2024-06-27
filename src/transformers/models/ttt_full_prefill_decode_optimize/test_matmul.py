from time import perf_counter
import torch

W = torch.randn((2048, 2048), dtype=torch.float16, device='cuda')
batch_list = [2**i for i in range(10, 21)]
time_list = []

repeats = 3

for bs in batch_list:
    X = torch.randn((bs, 2048), dtype=W.dtype, device=W.device)
    torch.cuda.synchronize()
    start = perf_counter()
    for i in range(repeats):
        X @ W
    torch.cuda.synchronize()
    avg_time = (perf_counter() - start) / repeats
    time_list.append(avg_time)

torch.save(time_list, '/workspace/transformers/exp/matmul_time/time_list.pth')
