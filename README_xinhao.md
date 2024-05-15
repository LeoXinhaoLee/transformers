<h1>Whole Model Decode Throughput</h1>

<h2>Latest results table</h2>
Please see Prefill=1, Decode=2048, vocab=50277 [here](https://docs.google.com/spreadsheets/d/1rhl2RNSzORNz5rZLFQV_LTeEsVhbeKDYCdziRjUOIsw/edit#gid=520284762).
Please ignore the gray cells.

<h2>Python environments</h2>
pytorch 2.2 + triton 2.2 (needed by Mamba and components borrowed from it)

<h2>Commands for reproducing results</h2>
Note that Mamba and TTT have constant decode throughput given any generation length, so only need to test 2K for example.
```
#!/bin/bash

set -x
DATE=''

for glen in 2048
do
 for bs in 128 256 512 1024 2048 4096 8192
 do
   python benchmark_prefill_decode.py --logdir ./exp/${DATE}_mamba_decode_2k \
                                      --model-name state-spaces/mamba-1.4b \
                                      --mode decode \
                                      --batch ${bs} \
                                      --promptlen 1 \
                                      --genlen ${glen}
 done
done

for glen in 2048
do
 for bs in 128 256 512 1024 2048 4096 8192
 do
   python benchmark_prefill_decode.py --logdir ./exp/${DATE}_M1_triton_compile_cg_decode_2k \
                                      --model-name ttt-1b \
                                      --mode decode \
                                      --batch ${bs} \
                                      --promptlen 1 \
                                      --genlen ${glen} \
                                      --inner_net mlp_1_dual_triton \
                                      --use_compile
 done
done

for glen in 2048
do
 for bs in 128 256 512 1024 2048 4096 8192
 do
   python benchmark_prefill_decode.py --logdir ./exp/${DATE}_M2_triton_compile_cg_decode_2k \
                                      --model-name ttt-1b \
                                      --mode decode \
                                      --batch ${bs} \
                                      --promptlen 1 \
                                      --genlen ${glen} \
                                      --inner_net mlp_2_dual_triton \
                                      --use_compile
 done

```

To generate pytorch profiler .json file, follow this example:
```

python benchmark_prefill_decode.py --logdir ./exp/05_15_M2_decode_profile \
                                   --model-name ttt-profile \
                                   --mode decode \
                                   --batch 32 \
                                   --promptlen 1 \
                                   --genlen 4 \
                                   --inner_net mlp_2_dual \
                                   --no_cg \
                                   --profile

```
Then go to https://ui.perfetto.dev/ to open .json trace.


<h1>Micro-Benchmark Decode Kernel</h1>

<h2>Commands</h2>

To benchmark clock-time:
```
python src/models/ttt_benchmark_decode_optimize/micro_benchmark.py
```
Note that to select which kernel function to benchmark, you can comment out the unneeded ones in `line_vals` in 
`@triton.testing.perf_report`.

To generate Nvidia Nsight System record:
```
nsys profile -f true -o OUTPUT_PATH python src/models/ttt_benchmark_decode_optimize/micro_benchmark.py --profile
```
