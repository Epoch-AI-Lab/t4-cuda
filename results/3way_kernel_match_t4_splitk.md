# 3-Way Head-to-Head Kernel Benchmark Report (Tesla T4, Split-K GEMV)

- **Hardware Device**: Tesla T4 (sm_75, 16GB GDDR6)
- **Timing Config**: 20 warmup iterations, 50 timed iterations (CUDA events)
- **Quantization**: Symmetric INT4 / NF4 with group_size=128
- **Baselines**: cuBLAS FP16, bitsandbytes NF4, bitsandbytes FP4, Marlin (Req sm_80+)
- **Kernel**: Split-K W4A16 GEMV (grid tiles N and K, T4_GEMV_SPLITK override available)

### [Qwen2.5-7B Full] Qwen-7B Gate/Up Proj (K=3584, N=18944)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 579.4 us | 280.6 us | 761.9 us | 758.4 us | Req sm_80+ | 2.06x | 2.72x |
| M=4 | 635.6 us | 749.6 us | 1218.3 us | 741.0 us | Req sm_80+ | 0.85x | 1.63x |
| M=16 | 628.7 us | 1206.3 us | 727.0 us | 752.6 us | Req sm_80+ | 0.52x | 0.60x |
| M=64 | 753.7 us | 2050.0 us | 985.1 us | 1005.5 us | Req sm_80+ | 0.37x | 0.48x |
| M=256 | 1663.5 us | 7792.6 us | 2605.1 us | 2496.5 us | Req sm_80+ | 0.21x | 0.33x |
### [Qwen2.5-7B Full] Qwen-7B Down Proj (K=18944, N=3584)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 580.6 us | 286.5 us | 687.4 us | 688.1 us | Req sm_80+ | 2.03x | 2.40x |
| M=4 | 657.4 us | 658.3 us | 846.7 us | 833.3 us | Req sm_80+ | 1.00x | 1.29x |
| M=16 | 645.1 us | 1329.0 us | 792.6 us | 922.5 us | Req sm_80+ | 0.49x | 0.60x |
| M=64 | 788.9 us | 2180.4 us | 1214.9 us | 1192.9 us | Req sm_80+ | 0.36x | 0.56x |
| M=256 | 1525.8 us | 8015.9 us | 3354.5 us | 2949.1 us | Req sm_80+ | 0.19x | 0.42x |
### [Qwen2.5-7B Full] Qwen-7B QKV / Q-Proj (K=3584, N=3584)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 131.7 us | 47.0 us | 150.4 us | 139.2 us | Req sm_80+ | 2.80x | 3.20x |
| M=4 | 145.0 us | 117.6 us | 222.2 us | 219.5 us | Req sm_80+ | 1.23x | 1.89x |
| M=16 | 147.5 us | 299.6 us | 224.3 us | 225.6 us | Req sm_80+ | 0.49x | 0.75x |
| M=64 | 190.6 us | 520.4 us | 311.3 us | 305.9 us | Req sm_80+ | 0.37x | 0.60x |
| M=256 | 342.0 us | 1603.6 us | 616.4 us | 624.6 us | Req sm_80+ | 0.21x | 0.38x |
### [Qwen2.5-7B Full] Qwen-7B KV-Proj (K=3584, N=512)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 47.6 us | 23.7 us | 104.6 us | 100.9 us | Req sm_80+ | 2.01x | 4.42x |
| M=4 | 57.2 us | 25.8 us | 100.4 us | 99.3 us | Req sm_80+ | 2.22x | 3.89x |
| M=16 | 62.2 us | 149.2 us | 151.6 us | 150.1 us | Req sm_80+ | 0.42x | 1.02x |
| M=64 | 63.7 us | 204.7 us | 141.2 us | 141.3 us | Req sm_80+ | 0.31x | 0.69x |
| M=256 | 90.2 us | 285.5 us | 210.1 us | 213.0 us | Req sm_80+ | 0.32x | 0.74x |
### [Qwen2.5-7B TP Sharded] TP Gate/Up (Col-Parallel) (K=3584, N=9472)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 292.9 us | 107.5 us | 227.7 us | 229.9 us | Req sm_80+ | 2.73x | 2.12x |
| M=4 | 326.2 us | 248.1 us | 444.8 us | 431.5 us | Req sm_80+ | 1.31x | 1.79x |
| M=16 | 332.8 us | 653.3 us | 388.6 us | 382.8 us | Req sm_80+ | 0.51x | 0.59x |
| M=64 | 395.0 us | 1156.6 us | 505.5 us | 518.1 us | Req sm_80+ | 0.34x | 0.44x |
| M=256 | 534.7 us | 3933.3 us | 1312.9 us | 1358.6 us | Req sm_80+ | 0.14x | 0.33x |
### [Qwen2.5-7B TP Sharded] TP Down (Row-Parallel) (K=9472, N=3584)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 299.0 us | 94.0 us | 237.6 us | 232.3 us | Req sm_80+ | 3.18x | 2.53x |
| M=4 | 322.2 us | 280.6 us | 491.4 us | 455.1 us | Req sm_80+ | 1.15x | 1.75x |
| M=16 | 335.0 us | 721.8 us | 432.1 us | 454.7 us | Req sm_80+ | 0.46x | 0.60x |
| M=64 | 415.7 us | 1171.5 us | 587.7 us | 630.2 us | Req sm_80+ | 0.35x | 0.50x |
| M=256 | 800.8 us | 4023.8 us | 1445.9 us | 1495.3 us | Req sm_80+ | 0.20x | 0.36x |
### [Llama-3-8B Full] Llama-3-8B Gate/Up (K=4096, N=14336)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 493.8 us | 253.9 us | 612.1 us | 612.4 us | Req sm_80+ | 1.95x | 2.41x |
| M=4 | 563.9 us | 505.3 us | 757.4 us | 663.6 us | Req sm_80+ | 1.12x | 1.50x |
| M=16 | 552.2 us | 1056.0 us | 629.7 us | 654.8 us | Req sm_80+ | 0.52x | 0.60x |
| M=64 | 650.8 us | 1798.3 us | 847.3 us | 862.2 us | Req sm_80+ | 0.36x | 0.47x |
| M=256 | 1473.1 us | 6812.0 us | 2488.5 us | 2262.4 us | Req sm_80+ | 0.22x | 0.37x |
### [Llama-3-8B Full] Llama-3-8B Down Proj (K=14336, N=4096)

| Batch M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Marlin | Speedup (T4/cuB) | Speedup (T4/BNB) |
|---|---|---|---|---|---|---|---|
| M=1 | 488.1 us | 172.0 us | 370.7 us | 348.7 us | Req sm_80+ | 2.84x | 2.15x |
| M=4 | 562.0 us | 571.4 us | 790.5 us | 765.2 us | Req sm_80+ | 0.98x | 1.38x |
| M=16 | 577.6 us | 1164.2 us | 722.9 us | 734.5 us | Req sm_80+ | 0.50x | 0.62x |
| M=64 | 672.8 us | 2019.3 us | 952.3 us | 987.6 us | Req sm_80+ | 0.33x | 0.47x |
| M=256 | 1777.7 us | 7079.9 us | 2844.3 us | 2668.0 us | Req sm_80+ | 0.25x | 0.40x |
