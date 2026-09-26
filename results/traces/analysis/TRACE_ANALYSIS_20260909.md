# T4 Trace Analysis - 2026-09-09

Traces captured on Colab Tesla T4 (CC 7.5, Nsight Compute 2025.1.1, `--set basic`).
Exported via `ncu --import --csv --page details|raw`, filtered to the two fused kernels.
Raw CSVs in this dir (`local_*.csv`); SM clock at capture: 585 MHz, DRAM 4.99 GHz.

## TL;DR

| | H6 fused bwd+AdamW | H17 W4A16 GEMV |
|---|---|---|
| Duration (avg of 3 replays) | 16.70 ms | 124.96 us |
| Compute (SM) throughput | 18.4% | 50.6% |
| DRAM throughput | 33.3% (~107 GB/s) | 38.0% (~122 GB/s) |
| Achieved occupancy | 24.85% (theoretical 25%) | 91.1% (theoretical 100%) |
| Limiter | **shared memory** (2 blocks/SM) | warps (at hardware max) |
| Verdict | latency-bound, fixable | near-optimal for this shape |

The AdamW kernel is 95.1% of GPU time in the timeline (nsys `cuda_gpu_kern_sum`).
The GEMV is ~0.4% of GPU time. Optimization effort belongs on AdamW.

## H6 fused backward+AdamW kernel - why it's slow

- Launch: grid (64,64,1)=4096 blocks x 128 threads, 524,288 threads.
- **Theoretical occupancy capped at 25%: Block Limit Shared Mem = 2 blocks/SM.**
  (Static smem 26.62 KB/block -> 2 blocks = ~53 KB of T4's 64 KB/SM.)
  NCU OPT rule: "Est. Local Speedup: 75% ... theoretical occupancy (25.0%) is
  limited by the required amount of shared memory" (2 theoretical warps/scheduler
  vs hardware max 8).
- Achieved ≈ theoretical (24.85%), so no scheduling waste on top - the kernel is
  simply issued starved: 7.95 active warps/SM out of 64 possible.
- SM 18.4%, DRAM 33.3%: **neither roof saturated -> latency-bound** (NCU OPT:
  "low compute throughput and memory bandwidth utilization ... typically indicate
  latency issues").
- With ~1.8 GB moved per 16.7 ms call (~107 GB/s), pushing DRAM toward its
  ~300 GB/s roof by raising occupancy is the primary lever.
- Tail effect: 51.2 waves/SM, negligible (partial-wave penalty <2%).

## H17 W4A16 GEMV - healthy

- Launch: 512 blocks x 256 threads (warp-specialized), 131,072 threads.
- Occupancy 91.1% achieved against 100% theoretical; Block Limit Warps = 4 blocks
  is the binding constraint at hardware max, smem is NOT a limiter (128 blocks).
- L1/TEX throughput 85.6% - the cache hierarchy is the busiest unit, consistent
  with a dequant-in-L1 GEMV. DRAM 38% (122 GB/s) - it streams weights well but
  is not DRAM-bound at this shape.
- SM 50.6%: mixed int8/half-pipe math keeps SM below 60%.
- Only flag: 3 full waves + partial wave of 33 blocks (grid 512 vs 3.2 waves/SM)
  -> NCU estimates up to ~25% tail waste **at this small grid**; at production
  sizes (larger grids) this shrinks toward zero.

## Recommended next steps (in expected-value order)

1. **AdamW: cut shared memory per block** so >2 blocks fit per SM (halve smem ->
   4 blocks -> ~50% occupancy ceiling; target >=4 warps/scheduler).
2. **AdamW: restructure tile shape/loops for more concurrent warps per SM.**
3. Re-profile after each change (same capture procedure, fail-loud script in
   git history: `102c32d..3cfd407`).
4. H17: leave it alone at these shapes; revisit only if grid stays ~512 in
   production (tail rule above), or if K/N grows enough to shift its mix.

## Method notes / caveats

- `--set basic` = no stall-reason breakdown in the report (metrics like
  `smsp__average_warps_issue_stalled_*` were not collected). The
  latency-bound-vs-roof-bound conclusion stands on SOL + occupancy + NCU's own
  OPT rules; a `--set full` rerun would name the exact stall reasons.
- SM clock 585 MHz is well under T4 boost (~1590): absolute times are
  conservative; percentage verdicts unaffected.
- DRAM bytes/call derived from `dram__cycles_active` % x spec peak BW x
  duration (per-sector byte counts not in the basic set).
- AdamW bytes ~1.8 GB/call at the capture shape is dominated by FP32 master
  states (m, v) + FP32 grad/proj traffic; it is a training-shaped kernel, not a
  decode-shaped one.
