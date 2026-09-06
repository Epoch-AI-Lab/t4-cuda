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

## 2026-09-06T13:04:20Z

SEND A TEAM TO AUDIT AND FIX ALL TESTS AND BENCHMARKS

Audit every test in `tests/` and every benchmark in `benchmarks/` to uncover and eradicate all hardcoded numbers, fake metrics, tautological assertions, and dummy passes, replacing them with authentic dynamic execution under strict zero tolerance.

Working directory: /home/kriday/epoch_website/t4-cuda
Integrity mode: benchmark

## Requirements

### R1. Complete Adversarial Codebase Audit
Scan every file in `tests/`, `benchmarks/`, and `src/` for fake or hardcoded shortcuts:
- Hardcoded timing or throughput constants (e.g. fixed `compute_per_layer_ms = 0.45`, scaling multipliers like `0.65`, or dividing constants in python).
- Tautological assertions (e.g. `assert 1 == 1`, asserting hardcoded numbers against themselves, or mocking execution to claim passes).
- Baselines that are crippled or given asymmetric execution paths.
- Output a comprehensive audit report detailing every violation found, file by file and line by line.

### R2. Replace Hardcoded Benchmarks with Live Dynamic Execution
Fix all benchmark scripts (including `benchmarks/benchmark_tp_vs_pp.py`, `benchmarks/benchmark_e2e_serving.py`, `benchmarks/benchmark_3way_kernels.py`):
- Eliminate all hardcoded compute estimates. Measure real layer forward passes through instantiated modules (`TPColumnParallelLinear`, `TPRowParallelLinear`, `PipelineParallelQwen2`, `HybridLinear`) using synchronized CUDA events.
- In `benchmark_tp_vs_pp.py`, instantiate real transformer layer blocks on `devices[0]` and `devices[1]`, execute actual forward passes, and calculate `tok_s` from measured kernel time plus measured PCIe transfer time.
- Ensure all metric calculations ($TFLOP/s$, $GB/s$, $ms/token$) derive solely from live measured time, parameter shapes, and bytes transferred.

### R3. Fix and Harden Test Suite Integrity
Refactor and harden tests in `tests/`:
- Eliminate any tautological assertions or dummy tests.
- When GPU hardware or custom extensions are unavailable, mark tests explicitly as `[SKIP]`. Never emit `PASSED` for unexecuted or mocked CUDA code.
- Tests must fail loudly with non-zero exit codes if assertions fail or numerical parity drifts beyond tolerance ($\text{cosine similarity} < 0.999$ or max absolute error threshold).

### R4. Comprehensive Verification Run
Execute the full test and benchmark suite locally and verify:
- Zero syntax errors, zero missing imports, zero hardcoded constants.
- Programmatic AST verification confirming absence of banned patterns (e.g. hardcoded timing constants, tautological assertions).
- All unit and integration tests pass cleanly or skip explicitly with clear reasons.

## Acceptance Criteria

### Audit & Elimination of Fake Logic
- [ ] No hardcoded timing, compute, or bandwidth constants in any file under `benchmarks/` or `tests/`.
- [ ] `benchmarks/benchmark_tp_vs_pp.py` executes real forward passes through instantiated sharded blocks rather than multiplying fixed millisecond constants.
- [ ] Zero tautological assertions (`assert True`, `assert x == x`, or trivial identity assertions) across the entire test suite.

### Benchmark Authenticity
- [ ] All reported latencies, throughputs, and memory metrics in `outputs/` and `results/` are calculated dynamically from measured runtimes (`torch.cuda.Event` elapsed time or `time.perf_counter()` on CPU).
- [ ] Dual-GPU benchmarks run live physical forward passes and communications when dual CUDA devices are present, and cleanly `[SKIP]` with zero synthetic numbers when absent.

### Test Suite Execution
- [ ] Automated AST scanner script passes with 0 violations found across all `.py` files in `src/`, `tests/`, and `benchmarks/`.
- [ ] Pytest suite (`pytest tests/`) runs with 0 failures, with all tests either authentically passing or explicitly skipping.


