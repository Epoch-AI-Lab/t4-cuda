# Project: Dual Tesla T4 Multi-GPU Sharding (INT4 & CP-Hybrid)

## Architecture
This project implements multi-GPU sharding across dual NVIDIA Tesla T4 GPUs (32 GB total VRAM over PCIe Gen3) for custom INT4 and CP-Hybrid CUDA kernels on `Qwen/Qwen2.5-Math-7B` (and 1.5B).
The system provides two coordinated tracks:
1. **Implementation Track**:
   - Device Safety Layer: `at::cuda::CUDAGuard` and device stream binding across all entry points in `src/bindings.cpp`, with cross-device pointer validation.
   - Tensor Parallelism (TP=2): Intra-layer sharding with column-parallel projections (`gate_proj`, `up_proj`, `q_proj`, `k_proj`, `v_proj`) and row-parallel projections (`down_proj`, `o_proj`) with peer All-Reduce over PCIe Gen3.
   - Pipeline Parallelism (PP=2): Inter-layer partitioning (Layers 0-13 on `cuda:0`, Layers 14-27 on `cuda:1`) with point-to-point boundary activations.
   - Unified Scope Integration: `CPMultiGPUInferenceScope` extending `CPHybridInferenceScope` supporting dynamic TP and PP execution.
2. **E2E Testing Track**:
   - Independent opaque-box test suite across 4 tiers verifying multi-device safety, correctness vs FP16 baseline (relative error <= 0.1%), zero illegal memory access across 100 decode steps, and peak VRAM < 14.0 GB per GPU.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | CUDAGuard on `dequantize_u4/s4/s3/fp8` | Set device context to input tensor device before stream query and launch | M1 | Survey (Explorer 1) |
| 2 | CUDAGuard on `fused_w4a16_gemm_u4/s4` | Set device context to input tensor device before stream query and launch | M1 | Survey (Explorer 1) |
| 3 | Input Stream Binding | Query `c10::cuda::getCurrentCUDAStream(tensor.device().index())` in all entry points | M1 | Survey (Explorer 1) |
| 4 | Cross-Device Argument Check | Validate that all tensor parameters belong to identical device index | M1 | Survey (Explorer 1) |
| 5 | Contiguity Validation | Enforce `.is_contiguous()` checks on all inputs before raw pointer access | M1 | Survey (Explorer 1) |
| 6 | Multi-Device C++ Extension Build | Update `src/setup.py` and compile verification on dual devices | M1 | Survey (Explorer 1) |
| 7 | Column-Parallel INT4 MLP Linear | Slices `gate_proj` and `up_proj` column-wise with zero communication | M2 | Survey (Explorer 2) |
| 8 | Row-Parallel INT4 MLP Linear | Slices `down_proj` row-wise with partial output accumulation | M2 | Survey (Explorer 2) |
| 9 | Fast Dual-T4 Peer All-Reduce | Low-latency All-Reduce over PCIe Gen3 (P2P IPC / NCCL) | M2 | Survey (Explorer 2, 3) |
| 10 | Column/Row Parallel FP16 Attention | Slices Q/K/V column-wise and O row-wise with attention All-Reduce | M2 | Survey (Explorer 2) |
| 11 | Sharded Static KV Cache | Halves KV cache memory per GPU (2 KV heads per GPU) | M2 | Survey (Explorer 2, 3) |
| 12 | Pipeline Parallel Partitioning | Partitions 28 layers (14 layers per GPU) with boundary tensor transfer | M3 | Survey (Explorer 2, 3) |
| 13 | Empirical TP vs PP Benchmark | Measures tok/s, kernel execution time, PCIe latency, prefill vs decode | M3 | Survey (Explorer 3) |
| 14 | Architectural Comparison Report | Empirical report documenting trade-offs between TP and PP | M3 | Survey (Explorer 3) |
| 15 | `CPMultiGPUInferenceScope` | Unified multi-GPU scope supporting TP and PP execution modes | M4 | Survey (Explorer 2) |
| 16 | 7B Model Dual-GPU Rollout Runner | Autoregressive generation script for Qwen2.5-Math-7B on dual T4s | M4 | Survey (Explorer 2, 3) |
| 17 | 100-Step Decode Stability Test | Stress test verifying zero OOMs, illegal accesses, or memory leaks | M4 | Survey (Explorer 3) |
| 18 | Relative Error <= 0.1% Gate | Verifies sharded decode output against FP16 baseline | M4 | Survey (Explorer 2) |
| 19 | Peak VRAM < 14.0 GB Validation | Confirms peak VRAM stays well under 14.0 GB per GPU during 2048-token runs | M4 | Survey (Explorer 2, 3) |
| 20 | E2E Multi-Device Test Harness | Independent test runner executing Tiers 1-4 with pass/fail exit codes | E2E | Survey (Explorer 3) |
| 21 | Tier 1 Feature Coverage Tests | >=5 tests per feature covering kernel safety, TP slicing, and PP transfer | E2E | Survey (Explorer 3) |
| 22 | Tier 2 Boundary & Corner Tests | Tests single-token, max context (2048), batch sizes, and edge shapes | E2E | Survey (Explorer 3) |
| 23 | Tier 3 Cross-Feature Tests | Tests interaction between TP/PP sharding, KV cache, and INT4 GEMM | E2E | Survey (Explorer 3) |
| 24 | Tier 4 Real-World 7B Math Scenarios | End-to-end mathematical reasoning rollouts on AMC/AIME problems | E2E | Survey (Explorer 3) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: CUDAGuard & Stream Safety | Fix bindings.cpp entry points with CUDAGuard, stream binding, cross-device checks, and multi-device tests | none | PLANNED |
| 2 | M2: Intra-Layer Tensor Parallelism | Column/row INT4 MLP + FP16 Attention sharding and fast PCIe All-Reduce | M1 | PLANNED |
| 3 | M3: Pipeline Parallelism & Empirical Benchmark | PP partitioning, empirical comparison of TP vs PP (tok/s, latency), decision gate | M1, M2 | PLANNED |
| 4 | M4: CPMultiGPU Scope & 7B Model Validation | Integrate winning sharding into CPMultiGPUInferenceScope, validate Qwen2.5-Math-7B 2048 ctx, VRAM < 14GB, 100 decode steps | M1, M2, M3 | PLANNED |
| 5 | E2E: Independent Multi-GPU Test Suite | Design and build opaque-box 4-tier test suite, publish TEST_READY.md | M1 | PLANNED |

## Interface Contracts
### C++/CUDA Bindings (`t4_kernels`)
- All entry points require:
  - Input tensors on valid CUDA device (checked via `A.is_cuda()`).
  - Device guard: `const at::cuda::CUDAGuard device_guard(A.device());`.
  - Stream: `cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index()).stream();`.
  - All input tensors must reside on `A.device()`. If mismatched, throw `c10::Error`.
  - Contiguity: check `tensor.is_contiguous()`.

### Tensor Parallelism (`src/sharding/tp.py`)
- `TPColumnParallelLinear`:
  - Input: activation $X \in \mathbb{R}^{B \times M \times K}$ replicated across GPUs.
  - Weights: $W \in \mathbb{R}^{(N/2) \times K}$ (or packed INT4) residing locally on `cuda:i`.
  - Output: local slice $Y^{(i)} \in \mathbb{R}^{B \times M \times (N/2)}$. No communication.
- `TPRowParallelLinear`:
  - Input: local slice $Z^{(i)} \in \mathbb{R}^{B \times M \times (K/2)}$.
  - Weights: $W \in \mathbb{R}^{N \times (K/2)}$ residing locally on `cuda:i`.
  - Output: reduced tensor $O = \sum_i Z^{(i)} W^T \in \mathbb{R}^{B \times M \times N}$ via peer All-Reduce.

### Pipeline Parallelism (`src/sharding/pp.py`)
- `PipelineParallelQwen2`:
  - Stage 0 (`cuda:0`): Embeddings + Layers 0..13.
  - Stage 1 (`cuda:1`): Layers 14..27 + Final Norm + LM Head.
  - Boundary: Asynchronous P2P tensor copy of activation $[B, M, D]$ at layer 13/14 boundary.

### Inference Scope (`src/sharding/scope.py`)
- `CPMultiGPUInferenceScope(model, mode='tp' | 'pp', devices=['cuda:0', 'cuda:1'])`:
  - Context manager that shards model upon `__enter__` and restores original state upon `__exit__`.

## Code Layout
- `src/bindings.cpp`: C++/CUDA entry points and device guards.
- `src/setup.py`: PyTorch C++ extension build configuration.
- `src/sharding/`:
  - `__init__.py`: Package exports.
  - `comm.py`: Dual-GPU All-Reduce, P2P transfer, and synchronization primitives.
  - `tp.py`: Intra-layer Tensor Parallel linear layers (INT4 MLP and FP16 Attention).
  - `pp.py`: Inter-layer Pipeline Parallel partitioning and boundary transfer.
  - `scope.py`: `CPMultiGPUInferenceScope` context manager.
- `benchmarks/benchmark_tp_vs_pp.py`: Empirical comparison harness measuring tok/s, kernel execution, PCIe latency, and VRAM.
- `tests/test_multi_gpu_guard.py`: Multi-device unit tests for `t4_kernels`.
- `tests/test_sharding_tp_correctness.py`: Correctness and numerics tests for TP=2.
- `tests/test_sharding_pp_correctness.py`: Correctness and numerics tests for PP=2.
- `tests/test_e2e_multigpu_suite.py`: Comprehensive 4-tier opaque-box test suite.
