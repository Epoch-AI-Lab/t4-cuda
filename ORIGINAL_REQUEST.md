# Original User Request

## Initial Request — 2026-09-05T21:01:09+05:30

Benchmark and implement multi-GPU sharding for custom INT4 and CP-Hybrid CUDA kernels on dual Tesla T4 GPUs (32 GB total VRAM), trying intra-layer tensor parallelism first and comparing against pipeline parallelism for Qwen2.5-Math-7B training and rollout decode.

Working directory: /home/kriday/epoch_website/t4-cuda
Integrity mode: development

## Requirements

### R1. Multi-Device CUDAGuard and Stream Safety
Ensure all kernel entry points in `src/bindings.cpp` (including `fused_w4a16_gemm_u4_cuda`, `fused_w4a16_gemm_s4_cuda`, and dequantization routines) enforce `at::cuda::CUDAGuard` and bind to the input tensor's device stream. Kernels must execute on any valid CUDA device without cross-device pointer faults or illegal memory access errors.

### R2. Pipeline vs Tensor Parallelism (Attempt Tensor Parallelism First)
Investigate, attempt, and benchmark intra-layer Tensor Parallelism (TP) for the custom INT4 MLP and FP16 Attention blocks across dual T4 GPUs.
- Attempt column-parallel (gate_proj, up_proj) and row-parallel (down_proj) sharding with an All-Reduce over dual-GPU PCIe Gen3.
- Discuss and measure the direct trade-off between Tensor Parallelism (intra-layer communication per token) and Pipeline Parallelism (inter-layer communication at layer boundaries).
- If Tensor Parallelism is viable and maintains high decode tok/s, adopt it; if PCIe latency makes decode too slow, document the empirical numbers and deploy Pipeline Parallelism.

### R3. Dual-GPU Validation on 7B Model
Validate the sharded execution on `Qwen/Qwen2.5-Math-7B` with 2048-token context length. The solution must demonstrate zero OOMs, maintain peak VRAM below 14.0 GB per GPU, and keep numerical relative error within <= 0.1% of the FP16 baseline.

## Acceptance Criteria

### Correctness & Numerics
- [ ] Every kernel binding in `src/bindings.cpp` guards the device with `at::cuda::CUDAGuard`.
- [ ] Multi-device unit tests pass with tensors allocated on both `cuda:0` and `cuda:1`.
- [ ] Quantized decode output has relative error <= 0.1% compared to FP16 reference.

### Performance & Architectural Comparison
- [ ] Concrete empirical comparison of Tensor Parallelism vs Pipeline Parallelism on dual T4 GPUs (measuring tok/s, kernel execution time, and PCIe All-Reduce transfer latency).
- [ ] Working implementation of the winning sharding architecture (or both) integrated into `CPHybridInferenceScope` and model runners.
- [ ] Peak VRAM per GPU stays strictly under 14.0 GB during 7B forward and decode passes.
- [ ] Zero CUDA illegal memory access, device mismatch, or OOM crashes across 100 consecutive decode steps.

## Follow-up — 2026-09-06T06:47:59Z

Quota has reset and server restarted. Resume the teamwork execution from the current workspace state. Milestone 1 (src/bindings.cpp CUDAGuard and stream safety) is implemented and verified (85 tests passing). Proceed to Milestone 2 (Intra-Layer Tensor Parallelism in src/sharding/tp.py and comm.py), Milestone 3 (Pipeline Parallelism and TP vs PP benchmark), and Milestone 4 (CPMultiGPUInferenceScope on 7B). Keep executing until all acceptance criteria are verified.

## Follow-up — 2026-09-06T08:58:34Z

Benchmark our custom W4A16 GEMV and WMMA kernels head-to-head against bitsandbytes (NF4) and Marlin on dual Tesla T4 GPUs across real Qwen2.5-7B shapes, re-tiling the GEMV memory access to eliminate the 87.5% memory bus waste under a strict zero-tolerance protocol against hardcoded or faked benchmark numbers.

Working directory: /home/kriday/epoch_website/t4-cuda
Integrity mode: benchmark

## Requirements

### R1. Authentic 3-Way Kernel Benchmark Harness
Execute an authentic, synchronized benchmark comparing our custom kernel (`t4_kernels`), bitsandbytes (`Linear4bit` NF4), and Marlin on identical Qwen2.5-7B projection dimensions ($K=3584, N=18944$ Gate/Up; $K=18944, N=3584$ Down; $K=3584, N=3584$ QKV; and TP rank $K=3584, N=9472$).
- Measure across decode ($M=1, 4$) and prefill ($M=16, 64, 256$) regimes.
- Use CUDA events with `torch.cuda.synchronize()` and at least 20 warmup iterations and 50 timed iterations.
- Record median latency, P95 latency, effective memory bandwidth (GB/s against T4 320 GB/s peak), and compute throughput (TFLOP/s against T4 65 TFLOP/s peak).

### R2. Re-tile GEMV Kernel for 100% Memory Coalescing
Re-architect `fused_w4a16_gemv_u4_kernel` in `src/kernels/fused_w4a16_gemm.cu` so that threads within a warp access contiguous memory addresses along row dimensions ($N$-contiguous) instead of striding across rows separated by $75.7\text{ KB}$.
- Replace scalar 32-bit loads with vectorized 128-bit (`uint4` / `float4`) loads.
- Ensure 100% bus utilization (eliminating the discarded 112 bytes per 128-byte cache line).
- Preserve exact FP16/INT4 numerical output and parity with zero regression in accuracy.

### R3. Dual-GPU Sharding Execution Profile
Evaluate and document multi-GPU generation throughput under both Pipeline Parallelism (`PP=2`) and Tensor Parallelism (`TP=2`) on Kaggle dual T4s:
- Measure real PCIe transfer latencies (All-Reduce vs Boundary P2P transfer).
- Enforce that token generation uses the optimal sharding path to bypass the 56 All-Reduce PCIe bottleneck per token.

### R4. Strict Zero-Tolerance Integrity Verification
Strictly enforce the zero-tolerance policy against faking, mock passes, and hardcoded values:
- Every reported latency, throughput, and bandwidth figure must be calculated dynamically from measured elapsed time during runtime.
- If a dependency, extension, or GPU is missing, the test must explicitly mark `[SKIP]`. Never emit `PASSED` for unexecuted code or CPU dummies.
- Output raw machine-readable JSON logs containing individual iteration times, device info, and timestamped run metadata.

## Acceptance Criteria

### Benchmark Authenticity & Rigor
- [ ] Benchmark script (`benchmarks/benchmark_3way_kernels.py`) runs with real CUDA synchronization (`torch.cuda.Event` and `torch.cuda.synchronize()`).
- [ ] No hardcoded numbers, constants, or mock assertions in any benchmark script or test.
- [ ] Script outputs verified JSON results with raw timings, calculated bandwidth, and TFLOP/s.

### Kernel Performance & Memory Coalescing
- [ ] Re-tiled GEMV kernel achieves contiguous memory coalescing verified by instruction trace / nsight profile or bandwidth analysis.
- [ ] Measured effective memory bandwidth on T4 increases from ~40 GB/s toward hardware ceiling (>180 GB/s).
- [ ] Numerical parity test passes with cosine similarity >= 0.999 vs unquantized FP16 baseline.

### Multi-GPU Sharding Validation
- [ ] Verified dual T4 generation throughput (tok/s) under PP=2 and TP=2.
- [ ] Peak VRAM stays strictly below 14.0 GB per GPU across 100 consecutive decode steps.
- [ ] All 140+ unit and integration tests in `tests/` pass with zero failures.

