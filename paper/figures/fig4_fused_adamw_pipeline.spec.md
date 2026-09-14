# Figure Specification: fig4_fused_adamw_pipeline

## 1. Metadata
- **Figure Key:** `fig4_fused_adamw_pipeline`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig4_fused_adamw_pipeline.*`)
- **Generator:** `paper/generate_figures.py::fig4_fused_adamw_pipeline`
- **LaTeX Reference:** `Figure~\ref{fig:fused_adamw_arch}`

## 2. Microarchitectural Provenance
- **Hardware:** NVIDIA Tesla T4 Silicon (Turing TU104, CC 7.5)
- **Data Source:** `results/logs/t4_h6_empirical_run_20260813.log` and Lemma 3 (`lem:dram_traffic`)
- **Traffic Reduction:**
  - Unfused Baseline: $28\text{ B/param}$ (gradient write/read roundtrip + optimizer states)
  - Fused H6 Pipeline: $22\text{ B/param}$ via in-register gradient accumulation and inline FP16 downcast
  - Net DRAM Traffic Saved: $21.43\%$
  - Measured End-to-End Speedup: $1.94\times$ ($14.2\text{ ms}$ vs $27.5\text{ ms}$)

## 3. Visual Encodings
- **Comparative Flowchart:** Unfused pipeline (separate Backward GEMM, DRAM gradient staging, and AdamW kernel) vs Fused pipeline (in-register fused GEMM + AdamW update with inline weight downcast).
- **Color Palette:** Rose (`#f43f5e`), Emerald (`#10b981`), Blue (`#3b82f6`), Slate (`#1e293b`).

## 4. Takeaway
Fusing the backward gradient GEMM with AdamW state updates in registers eliminates the global DRAM gradient roundtrip, cutting memory traffic by 21.43% and accelerating training iterations by 1.94x.\n