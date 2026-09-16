# NOTION DUMP - T4 Profiling Session 2026-09-09

Everything from today's Milestone A fix + Milestone C trace capture + trace analysis.
Paste-ready. Source of truth lives in the repo (paths at the bottom).

Repo: t4-cuda @ branch feat-big-chalk-math-7b

## 1. Milestone A - end-to-end serving on real T4 (SOLVED, 3rd attempt)

Failures were mine, both dumb, both logged:
- Attempt 1: `device` never defined -> NameError in CUDA path. Fixed in 3f6cf81.
- Attempt 2: `device = torch.device("cuda")` never equals the string "cuda",
  so the `if device == "cuda"` guards silently skipped engines 2 and 3.
  Fixed in 102c32d (`device = "cuda"`). Lesson: string compare vs object.

Final numbers (26 requests, CUDA-event timed, proper sync, same shapes):

| Engine | tok/s | p50 | p99 | Speedup |
|---|---|---|---|---|
| PyTorch FP16 Eager | 57.44 | 15.76 ms | 29.58 ms | 1.00x |
| torch.compile (Inductor) | 62.36 | 15.55 ms | 20.40 ms | 1.09x |
| T4 Fused INT4 (ours) | 123.66 | 7.79 ms | 12.83 ms | 2.15x |

Caveat: Inductor cudagraphs auto-disabled (mutated inputs), so 1.09x is plain
Inductor codegen, not cudagraph-accelerated. 2.15x headline is clean.

## 2. Milestone C - traces captured on Colab T4

- 2x NCU reports (--set basic, 9-pass replay): h6_fused_adamw_t4.ncu-rep,
  h17_w4a16_gemv_t4.ncu-rep
- 1x NSYS timeline (repo's first): t4_kernels_timeline.nsys-rep
- Capture log (fail-loud, verified non-empty before DONE):
  trace_capture_20260909.log
- nsys binary on Colab hides at:
  /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys (not on PATH)
- Workload used the canonical quantizer imported from
  tests/test_h17_fused_int3_gemv.py (no hand-rolled re-implementation)

## 3. Trace analysis (the important part)

Method: fresh Colab T4, `ncu --import --csv --page details|raw` exported there,
CSVs downloaded, VM killed (~6 min VM time). No profilers on the local box.

### Headline: H6 fused backward+AdamW kernel is 95.1% of all GPU time

nsys cuda_gpu_kern_sum: adamw kernel 44.6 ms of 46.9 ms total (3 invocations,
~14.9 ms each). H17 GEMV is ~0.4% of GPU time. All optimization effort belongs
on AdamW.

### H6 fused bwd+AdamW: latency-bound, occupancy-starved by shared memory

| Metric | Value |
|---|---|
| Duration | 16.70 ms (avg of 3 replays) |
| Compute (SM) throughput | 18.4% |
| DRAM throughput | 33.3% (~107 GB/s of ~320 spec) |
| L1/TEX throughput | 41.7% |
| L2 throughput | 15.1% |
| Achieved occupancy | 24.85% (theoretical 25%) |
| Block limiter | shared memory: 2 blocks/SM |
| Static smem/block | 26.62 KB |
| Grid x block | 4096 x 128 (524,288 threads) |
| Waves per SM | 51.2 (tail effect negligible) |

NCU's own OPT rules on this kernel:
- "Est. Local Speedup: 75% ... theoretical occupancy (25.0%) is limited by the
  required amount of shared memory" (2 theoretical warps/scheduler vs hw max 8)
- "low compute throughput and memory bandwidth utilization ... typically
  indicate latency issues" (both roofs under 60%)

Reading: neither roof near saturation; only 7.95 resident warps of 64 possible,
so nothing hides latency. Achieved == theoretical, so fix the ceiling itself.
~1.8 GB moved per 16.7 ms call: fit 4+ blocks/SM (halve smem) and push DRAM
toward ~300 GB/s.

ELI5: kernel rents a warehouse floor to store one pallet. Everyone else waits
outside. Give workers lockers instead of floors, 4x the workers fit inside.

### H17 W4A16 GEMV: healthy, do not touch (at these shapes)

| Metric | Value |
|---|---|
| Duration | 124.96 us |
| Compute (SM) throughput | 50.6% |
| DRAM throughput | 38.0% (~122 GB/s) |
| L1/TEX throughput | 85.6% (busiest unit, good for dequant-in-L1 GEMV) |
| Achieved occupancy | 91.1% (theoretical 100%) |
| Block limiter | warps (4 blocks/SM, at hardware max; smem NOT a limiter) |
| Grid x block | 512 x 256 (warp-specialized) |
| Waves per SM | 3.2 (3 full waves + partial wave of 33 blocks) |

Only nit: small grid at capture shape -> NCU estimates up to ~25% tail waste.
Shrinks toward zero as production grids grow. Revisit only if grid stays ~512
in production or shapes change a lot.

## 4. Next steps (expected-value order)

1. AdamW: cut smem per block so >2 blocks fit/SM (target >=4 warps/scheduler)
2. AdamW: restructure tile shape/loops for more concurrent warps
3. Re-profile after each change (same fail-loud capture procedure)
4. H17: leave alone; H26+ queue unaffected
5. Optional: recapture with --set full to name exact stall reasons
   (basic set doesn't collect smsp stall ratios; SOL+occupancy+OPT rules
   already pin the conclusion, stall names would just add color)

## 5. Caveats (honesty section)

- --set basic: no per-stall-reason breakdown collected. Latency-bound verdict
  stands on SOL % + occupancy + NCU OPT rules.
- SM clock at capture: 585 MHz (T4 boost ~1590). Absolute times conservative;
  percentage verdicts unaffected.
- DRAM bytes/call derived from dram__cycles_active % x spec peak x duration
  (per-sector byte counts not in basic set).
- AdamW ~1.8 GB/call is training-shaped traffic (FP32 master states m,v +
  FP32 grads). Not comparable to decode-kernel numbers.

## 6. File inventory (repo paths, t4-cuda)

results/milestone_a_e2e_serving.log          - raw Milestone A benchmark output
results/traces/h6_fused_adamw_t4.ncu-rep     - NCU, AdamW kernel
results/traces/h17_w4a16_gemv_t4.ncu-rep     - NCU, H17 GEMV
results/traces/t4_kernels_timeline.nsys-rep  - NSYS timeline
results/traces/trace_capture_20260909.log    - capture log (fail-loud)
results/traces/analysis/TRACE_ANALYSIS_20260909.md - full written analysis
results/traces/analysis/local_{adamw,h17}_{details,raw}.csv - raw NCU CSVs

Commits: 3f6cf81 (NameError fix), 102c32d (device compare fix),
3cfd407 (A numbers + C traces). Analysis dir NOT yet committed.

## 7. Colab CLI gotchas learned (save for next time)

- `colab exec -f` reads a LOCAL file; uploading then pointing -f at the remote
  path fails with FileNotFoundError.
- Uploads >~10 MB can exceed the 30 s tool timeout: run with
  setsid nohup ... & then poll `colab ls`.
- Colab VMs have ncu at /usr/local/cuda/bin/ncu; nsys only inside the
  nsight-compute host dir (see path above).
- Parse NCU CSV locally with a real csv parser: kernel names contain commas.
- NCU report has a units row after the header: skip rows[1] when averaging.
