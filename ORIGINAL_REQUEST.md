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
