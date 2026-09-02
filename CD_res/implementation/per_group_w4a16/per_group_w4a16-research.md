# Implementation Research: Per-Group INT4 W4A16 GEMM (group=128)

## The Task
Implement per-group (group=128 in K dimension) INT4 W4A16 GEMV/GEMM on NVIDIA Turing T4 (TU104, sm_75) in CUDA C++ and PyTorch bindings. Replace the lossy per-channel INT4 setup that showed 8-9% relative activation error on Qwen2.5-0.5B, targeting <= 0.1% relative quantization error vs fp16 and bit-exact reconstruction vs Python reference.

## 1. Common Gotchas
- **Group index alignment along inner K**: With group=128, each packed `uint32_t` holds 8 weights. Thus, 1 group = 128 / 8 = 16 `uint32_t`s. In the grid/thread mapping, threads step with `stride_k = 64` (which spans 4 groups per stride). If scale/zero-point indexing assumes consecutive threads work on the same group or that a single thread stays in one group, the scale lookup corrupts.
  - *Source:* `src/kernels/fused_w4a16_gemm.cu` thread stride geometry; GPTQ per-group format (Frantar et al., 2022).
  - *Mitigation:* Explicitly compute group index `g = k_idx >> 4` per packed word iteration or per weight.
- **Scale tensor memory layout**: If scales are stored as `(num_groups, N)` with row-major order, memory accesses for adjacent threads in a warp (`col = blockIdx.x * 4 + threadIdx.x % 4`) access adjacent columns within the same row `g`. This yields coalesced memory transactions. Storing scales as `(N, num_groups)` would force strided uncoalesced memory reads across the warp.
  - *Source:* CUDA C++ Best Practices Guide (Section 5.2.1: Coalesced Access to Global Memory).
  - *Mitigation:* Use shape `(num_groups, N)` where `num_groups = K / group_size`.
- **Catastrophic fp16 cancellation**: In the original u4 implementation, unpacking via LOP3 yielded `1024 + q` in fp16. Multiplying in fp16 by activations at magnitude ~1024 rounded away bottom bits before subtracting the bias, leaving severe noise.
  - *Source:* `research-log.md` [2026-09-02] entry and commit d0ae530.
  - *Mitigation:* Convert `1024 + q` directly to fp32, subtract `1024.0f` to recover exact integer `q \in [0, 15]`, compute `w = (q - z) * s` in fp32, and accumulate products in fp32.

## 2. Best Practices
- **In-loop direct dequantization vs multi-accumulator**: For per-channel quantization, factoring out scale `s` and zero-point `z` across the entire K reduction enabled a two-accumulator formulation. But for per-group quantization, group boundaries cross thread strides. Subtracting 1024.0f in fp32 immediately after LOP3 unpack is bit-exact (integers in [1024, 1039] minus 1024 have exact IEEE-754 float representation). Performing `(q - z) * s * a` in fp32 in the loop requires only one fp32 accumulator per column, eliminating complex inter-group epilogues.
  - *Source:* IEEE-754 standard (Sterbenz lemma & exact representability of integers < 2^24 in single precision).
- **Vectorized activation loads**: Keep activation loads paired with the LOP3 unpack structure (stride-4 paired halves or half2) to match memory throughput to compute.
  - *Source:* Turing Tuning Guide (Memory Hierarchy & Instruction Mix).

## 3. Pitfalls & Language Quirks
- **Integer division latency in CUDA**: `k_idx / 16` should use bitshift `k_idx >> 4` or compiler strength reduction so it does not emit expensive `S32.DIV` instructions in the hot loop.
  - *Source:* PTX ISA 8.x Guide (Integer Arithmetic Instructions).
- **Scale and zp clamping during quantization**: GPTQ and standard asymmetric min-max quantization can produce `max == min`, resulting in division by zero if scale is clamped to 0. Scale must be clamped to minimum epsilon `1e-10` or `1e-7`.
  - *Source:* PyTorch quantization reference & `benchmarks/debug_qproj.py`.

## 4. Differentiation
- **Industry Standard (e.g., AutoGPTQ, Marlin, bitsandbytes)**: General GEMM kernels tile along M, N, and K with large shared memory tiles (e.g., 64x64 or 128x128), optimized for M >= 16 batch sizes, but suffer high launch overhead and register spilling on small decode batch sizes (M=1..8).
- **Our Approach**: Specialized small-batch GEMV kernel (M=1..8). Keeps weights in registers via single-cycle LOP3 0xEA bitwise dequant, reduces along K across warp threads with `__shfl_xor_sync`, uses shared memory only for inter-warp reduction across 4 columns, and directly dequantizes per-group scale/zp in-register.
- **Usefulness**: For GRPO rollouts (autoregressive token generation with M=1), decode is 85.4% of step time. A specialized GEMV that minimizes latency beats heavy tiled GEMM kernels on T4.

## Recommendation
1. Update `fused_w4a16_gemm_u4` kernel to accept scales and zero-points shaped `(num_groups, N)` where `group_size = 128`.
2. Compute `g = k_idx >> 4` in the inner loop, load `s = scale[g * N + col]` and `z = zero_point[g * N + col]`.
3. Subtract `1024.0f` from the LOP3 unpacked half-floats in FP32, scale by `(q - z) * s`, and accumulate in FP32.
4. Support both per-channel (`num_groups == 1`) and per-group (`num_groups > 1`) cleanly by checking `group_size`.
5. Implement python quantization and dequantization functions for group=128 in `tests/test_fused_gemm_correctness.py` and verify bit-exact and relative error envelopes.

## Sources
- NVIDIA CUDA C++ Programming Guide & Turing Tuning Guide (sm_75)
- Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (arXiv:2210.17323)
- PyTorch C++ Extension API documentation (torch::Tensor accessors and CUDA dispatch)
- Research Log `research-log.md` [2026-09-02]

## Adversarial Verification
- Check 1: Does per-group with group=128 handle non-multiple of 128 K? (Qwen2.5-0.5B has K=896 = 7 * 128, exact multiple). If K is not multiple of 128, guard bounds or assert K % 128 == 0.
- Check 2: Scale/zp memory access overhead: does loading scale/zp every 16 uint32s slow down the inner loop? In our thread mapping, `stride_k = 64`, so a thread visits a new group every iteration. We load 2 half values per iteration (4 bytes) while loading 4 bytes of weights and 16 bytes of activations. The L1 cache hit rate on scales across the warp is high because 4 adjacent columns share the same K group.
