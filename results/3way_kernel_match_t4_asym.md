# 3-Way Head-to-Head Kernel Benchmark Report (Tesla T4 - Asymmetric INT4)

- **Hardware Device**: Tesla T4 (sm_75, 16GB GDDR6)
- **Timing Config**: 20 warmup iterations, 50 timed iterations (CUDA events)
- **Quantization**: Asymmetric INT4 / NF4 / FP4 with group_size=128
- **Baselines**: cuBLAS FP16, bitsandbytes NF4, bitsandbytes FP4, Marlin (Req sm_80+)

### [Qwen2.5-7B Full] Qwen-7B Gate/Up Proj (K=3584, N=18944)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 579.0 us | 280.6 us | 763.5 us | 761.2 us | Req sm_80+ | 2.06x | 2.72x |
| M=4 | 625.8 us | 679.9 us | 931.5 us | 913.3 us | Req sm_80+ | 0.92x | 1.37x |
| M=16 | 628.7 us | 1205.4 us | 760.7 us | 722.5 us | Req sm_80+ | 0.52x | 0.63x |
| M=64 | 762.7 us | 2050.1 us | 1012.9 us | 1012.2 us | Req sm_80+ | 0.37x | 0.49x |
| M=256 | 1793.8 us | 7718.0 us | 2827.7 us | 2798.6 us | Req sm_80+ | 0.23x | 0.37x |
### [Qwen2.5-7B Full] Qwen-7B Down Proj (K=18944, N=3584)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 587.8 us | 297.0 us | 718.8 us | 458.8 us | Req sm_80+ | 1.98x | 2.42x |
| M=4 | 630.2 us | 661.3 us | 1016.0 us | 819.2 us | Req sm_80+ | 0.95x | 1.54x |
| M=16 | 651.3 us | 1345.3 us | 825.4 us | 885.0 us | Req sm_80+ | 0.48x | 0.61x |
| M=64 | 790.5 us | 2164.6 us | 1203.6 us | 1191.9 us | Req sm_80+ | 0.37x | 0.56x |
| M=256 | 2118.4 us | 7915.0 us | 3224.1 us | 3334.8 us | Req sm_80+ | 0.27x | 0.41x |
### [Qwen2.5-7B Full] Qwen-7B QKV / Q-Proj (K=3584, N=3584)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 135.3 us | 44.0 us | 139.9 us | 139.0 us | Req sm_80+ | 3.07x | 3.18x |
| M=4 | 150.4 us | 103.7 us | 207.2 us | 210.8 us | Req sm_80+ | 1.45x | 2.00x |
| M=16 | 153.1 us | 256.4 us | 216.3 us | 214.0 us | Req sm_80+ | 0.60x | 0.84x |
| M=64 | 190.6 us | 450.6 us | 280.5 us | 282.8 us | Req sm_80+ | 0.42x | 0.62x |
| M=256 | 342.6 us | 1572.9 us | 622.8 us | 636.7 us | Req sm_80+ | 0.22x | 0.40x |
### [Qwen2.5-7B Full] Qwen-7B KV-Proj (K=3584, N=512)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 51.2 us | 23.1 us | 108.7 us | 107.9 us | Req sm_80+ | 2.21x | 4.70x |
| M=4 | 63.5 us | 26.8 us | 116.7 us | 118.9 us | Req sm_80+ | 2.36x | 4.35x |
| M=16 | 59.5 us | 151.5 us | 159.6 us | 155.6 us | Req sm_80+ | 0.39x | 1.05x |
| M=64 | 69.2 us | 206.4 us | 159.4 us | 149.1 us | Req sm_80+ | 0.34x | 0.77x |
| M=256 | 86.3 us | 280.8 us | 233.7 us | 232.4 us | Req sm_80+ | 0.31x | 0.83x |
### [Qwen2.5-7B TP Sharded] TP Gate/Up (Col-Parallel) (K=3584, N=9472)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 296.3 us | 106.7 us | 233.2 us | 236.4 us | Req sm_80+ | 2.78x | 2.19x |
| M=4 | 329.3 us | 245.1 us | 463.9 us | 426.5 us | Req sm_80+ | 1.34x | 1.89x |
| M=16 | 333.8 us | 663.6 us | 388.4 us | 407.0 us | Req sm_80+ | 0.50x | 0.59x |
| M=64 | 403.5 us | 1147.2 us | 505.9 us | 543.1 us | Req sm_80+ | 0.35x | 0.44x |
| M=256 | 701.4 us | 3900.3 us | 1290.8 us | 1458.2 us | Req sm_80+ | 0.18x | 0.33x |
### [Qwen2.5-7B TP Sharded] TP Down (Row-Parallel) (K=9472, N=3584)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 307.2 us | 103.2 us | 268.3 us | 262.1 us | Req sm_80+ | 2.98x | 2.60x |
| M=4 | 323.6 us | 315.0 us | 522.2 us | 516.6 us | Req sm_80+ | 1.03x | 1.66x |
| M=16 | 337.9 us | 769.1 us | 435.8 us | 460.6 us | Req sm_80+ | 0.44x | 0.57x |
| M=64 | 418.1 us | 1209.0 us | 604.2 us | 632.1 us | Req sm_80+ | 0.35x | 0.50x |
| M=256 | 833.5 us | 4014.1 us | 1531.4 us | 1674.6 us | Req sm_80+ | 0.21x | 0.38x |
### [Llama-3-8B Full] Llama-3-8B Gate/Up (K=4096, N=14336)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 491.8 us | 174.1 us | 490.2 us | 488.9 us | Req sm_80+ | 2.82x | 2.82x |
| M=4 | 545.3 us | 619.3 us | 858.1 us | 661.1 us | Req sm_80+ | 0.88x | 1.39x |
| M=16 | 550.4 us | 1061.5 us | 657.4 us | 689.8 us | Req sm_80+ | 0.52x | 0.62x |
| M=64 | 652.0 us | 1818.5 us | 801.4 us | 866.7 us | Req sm_80+ | 0.36x | 0.44x |
| M=256 | 1618.9 us | 6804.3 us | 2500.4 us | 2498.5 us | Req sm_80+ | 0.24x | 0.37x |
### [Llama-3-8B Full] Llama-3-8B Down Proj (K=14336, N=4096)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 494.8 us | 233.6 us | 486.4 us | 479.3 us | Req sm_80+ | 2.12x | 2.08x |
| M=4 | 579.6 us | 800.8 us | 768.1 us | 749.6 us | Req sm_80+ | 0.72x | 0.96x |
| M=16 | 577.5 us | 1175.3 us | 761.9 us | 783.5 us | Req sm_80+ | 0.49x | 0.65x |
| M=64 | 680.9 us | 2009.1 us | 949.4 us | 978.8 us | Req sm_80+ | 0.34x | 0.47x |
| M=256 | 1935.4 us | 7033.0 us | 2867.1 us | 2816.3 us | Req sm_80+ | 0.28x | 0.41x |

