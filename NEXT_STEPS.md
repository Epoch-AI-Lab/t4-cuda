# Tesla T4 GPU Physical Execution & Next Research Milestones

> **Status (2026-08-16)**: The 5-stage physical execution protocol on physical **NVIDIA Tesla T4** hardware (TU104, sm_75, 70W TDP) has been **COMPLETED and VERIFIED 100%**. All test suites (`run_all_cuda_tests.py`, `verify_colab.sh`, `test_dequant_correctness.py`, `test_h6_fused_backward_adamw.py`, `test_h17_fused_int3_gemv.py`) pass cleanly on live silicon.

---

## 1. Summary of Completed Hardware Verifications (2026-08-16)

| Milestone / Kernel | Method & Target | Hardware Verification Status |
|---|---|---|
| **H1 (SMEM Swizzling)** | Standalone `%clock64` microbenchmarks | **0 Bank Conflicts**; 22.2 cycles/access vs 64.3 cycles in stride 32. |
| **H4 (Signed INT4 LOP3)** | PyTorch CUDA Extension KAT Harness | **Bit-exact (0.0 diff)** across `0xA7C13E59` / `0xF817E29A`; 23/23 tests passed. |
| **H5 (Occupancy & DVFS)**| Real-time hardware clock profiling | **25% occupancy locks 1590 MHz boost clock**; avoids 100% occupancy thermal decay to 1193 MHz. |
| **H6 (Fused Backward GEMM + AdamW)** | Comparative PyTorch Benchmark | **1.94x speedup** on T4 (9.983 ms vs 19.344 ms); **21.43% DRAM traffic reduction** (28→22 B/param). |
| **H7 (Signed INT3 LOP3)** | Live PyTorch CUDA Extension | **Bit-exact (0.0 diff)**; saturates GDDR6 bandwidth at **332.5 GB/s**. |
| **H9 (FP8 E4M3 Emulation)** | Bitwise ADD+OR conversion | **254/254 valid byte states exact** (0.0 diff). |
| **H17 (Fused INT3 Mega-Kernel)** | Live on-GPU PyTorch Extension | Correctness verified on live GPU across $B \in \{1,4,16\}$. |

---

## 2. Immediate Research Priorities (Phase 2)

### Milestone A: End-to-End Token Generation Serving Engine
- **Objective**: Wire `fused_h17_gemv_s3` and `fused_w4a16_gemv` into a minimal autoregressive token generation loop for a real model (e.g. 0.5B / 4B model) on Tesla T4.
- **Metric**: Measure **real wall-clock tokens/sec (p50, p90, p99 latency)** vs `torch.compile` (inductor) and standard HuggingFace/BitsAndBytes baselines.
- **Deliverable**: `benchmarks/benchmark_end_to_end_serving.py` executed on Colab T4.

### Milestone B: Decisive Experiment for H26 (INT3 Steering Vector Drift)
- **Objective**: Extract linear persona steering vectors (directness, depth, tone) from FP16 activations and measure their cosine similarity drift when applied to an INT3/INT4 quantized model.
- **Metric**: Confirm whether steering vectors retain $\ge 85\%$ directional alignment after weight quantization or if sub-4-bit discretization causes representation collapse.
- **Deliverable**: `experiments/h26_steering_quantization_drift.py`.

### Milestone C: Full Nsight Systems / Compute Trace Capture
- **Objective**: Generate clean `.nsys-rep` and `.ncu-rep` trace captures on Colab T4 for the fused AdamW (H6) and INT3 Mega-Kernel (H17) to provide auditable roofline and warp stall evidence for publication.

---

## 3. Reference Commands for Colab T4 Execution

```bash
# 1. Provision physical T4 instance
colab new --gpu T4 -s t4-bench

# 2. Upload workspace and build extension
tar --exclude='.git' --exclude='*.pyc' -czf /tmp/t4-cuda-repo.tar.gz .
colab upload -s t4-bench /tmp/t4-cuda-repo.tar.gz /content/t4-cuda-repo.tar.gz
colab exec -s t4-bench "mkdir -p /content/t4-cuda && tar -xzf /content/t4-cuda-repo.tar.gz -C /content/t4-cuda && pip install -e /content/t4-cuda/src"

# 3. Run complete verification pipeline
colab exec -s t4-bench "bash /content/t4-cuda/verify_colab.sh"

# 4. Stop session when finished
colab stop -s t4-bench
```
