# Figure Specification: fig8_w4a16_speedup_regimes

## 1. Metadata
- **Figure Key:** `fig8_w4a16_speedup_regimes`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig8_w4a16_speedup_regimes.*`)
- **Generator:** `paper/generate_figures.py::fig8_w4a16_speedup_regimes`
- **LaTeX Reference:** `Figure~\ref{fig:w4a16_speedup}`

## 2. Empirical Provenance
- **Data Source:** `results/benchmarks/fused_3b_benchmark.json` and Table 4 (`tab:w4a16_regimes`)
- **Hardware:** Physical Tesla T4 GPU (Driver 580.82.07, CUDA 13.0)
- **Points:**
  - M=1 (Attn 896x896): cuBLAS 18.9 us vs W4A16 9.2 us (2.06x speedup, err 0.0625)
  - M=1 (MLP 896x4864): cuBLAS 45.0 us vs W4A16 26.9 us (1.67x speedup, err 0.0625)
  - M=4: cuBLAS 21.7 us vs W4A16 21.1 us (1.03x speedup)
  - M=8: cuBLAS 19.4 us vs W4A16 61.4 us (0.32x speedup)
  - M=16: cuBLAS 23.5 us vs W4A16 63.3 us (0.37x speedup)
  - M=32 (MLP): cuBLAS 58.8 us vs W4A16 156.3 us (0.38x speedup)
  - M=64 (MLP): cuBLAS 60.9 us vs W4A16 233.1 us (0.26x speedup)

## 3. Visual Encodings
- **Panel A:** Speedup Factor vs Batch Size M with shaded green CP-Hybrid regime ($M \le 4$) and dashed 1.00x parity reference line.
- **Panel B:** Raw execution latency (us) at decode ($M=1$) comparing attention and MLP projections.
- **Color Palette:** Okabe-Ito Blue (`#2563eb`), Amber (`#d97706`), Emerald (`#059669`).

## 4. Takeaway
Demarcates the exact physical CP-Hybrid dispatch boundary: low-precision fused GEMV provides up to 2.06x speedup for $M \le 4$, after which compute-bound tensor cores dominate.
