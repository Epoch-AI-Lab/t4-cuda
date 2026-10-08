# Tesla T4 GPU Physical Execution & Next Research Milestones

> **Status (2026-09-23)**: Phase 2 physical execution on **NVIDIA Tesla T4** (TU104, sm_75, 70W TDP) is **COMPLETE and VERIFIED**. Milestones A–C below are done: end-to-end serving engine (1.48x wall-clock speedup), H26 steering-vector experiments superseded, and Nsight `.nsys-rep`/`.ncu-rep` traces captured (Milestone C, see `results/traces/`). The RL track has also advanced: Baby-Chalk (1.5B) cold-start SFT + GRPO RL post-training passed its victory audit (`docs/VICTORY_AUDIT_BABY_CHALK_FIX.md`).

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

## 2. Immediate Research Priorities (Phase 3 — as of 2026-09-23)

### ~~Milestone A~~ ✅ DONE: End-to-End Token Generation Serving Engine
- Unified speculative serving engine verified at **1.48x net wall-clock speedup** (`src/unified_speculative_engine.py`, `results/t4_speculative_benchmark_report.json`).

### ~~Milestone B~~ ✅ SUPERSEDED: H26 Steering Vector Drift
- Superseded by the RL/post-training track; hypothesis retired from the critical path.

### ~~Milestone C~~ ✅ DONE: Full Nsight Systems / Compute Trace Capture
- `.nsys-rep` / `.ncu-rep` traces captured for H6 fused AdamW and H17 mega-kernel (`results/traces/`); AdamW kernel identified as 95% of GPU time, starved by shared memory.

### Milestone D (NEW): Big-Chalk (Qwen2.5-Math-7B) RL Post-Training
- **Objective**: Scale the verified Baby-Chalk RL pipeline (cursor-style GRPO + Rottweiler verifier + strict scaffold enforcement + calibrated abstention) to `Qwen2.5-Math-7B` on dual Tesla T4 (Kaggle, 2×16 GB) using the PP=2 decode / TP=2 training sharding decided in `TODO.md` §5.
- **Metric**: Contest Pass@1 ≥ 56.2% and Grounding Sanity ≥ 80.0% (match or beat Baby-Chalk), 0 malformed tag artifacts, peak VRAM within dual-T4 budget.
- **Deliverables**: `benchmarks/train_big_chalk_rl.py` (does not exist yet — only `train_baby_chalk_rl.py` + `scripts/run_kaggle_7b_sft.sh` SFT runner exist), Kaggle runner notebook, eval artifact under `results/benchmarks/`.

### Milestone E (NEW): Conjecture Loop (stretch — `TODO.md` §6)
- Old-model proposes lemmas over post-cutoff mathlib/Lean-Workbook; Lean 4 checks truth; embeddings check novelty. Only after Milestone D lands.

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
