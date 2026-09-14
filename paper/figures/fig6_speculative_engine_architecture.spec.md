# Figure Specification: fig6_speculative_engine_architecture

## 1. Metadata
- **Figure Key:** `fig6_speculative_engine_architecture`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig6_speculative_engine_architecture.*`)
- **Generator:** `paper/generate_figures.py::fig6_speculative_engine_architecture`
- **LaTeX Reference:** `Figure~\ref{fig:spec_engine_arch}`

## 2. Architectural Provenance
- **Hardware:** NVIDIA Tesla T4 Silicon (Turing TU104, CC 7.5)
- **Data Source:** `results/benchmarks/m1_kv_cache_benchmark.json` and `results/benchmarks/t4_speculative_benchmark_report.json`
- **Components:**
  - Unified Draft Engine: Zero-weight prompt lookup + compact draft head
  - Pre-allocated `StaticKVCache`: Fixed $O(1)$ pointer rollbacks ($0.51\ \mu\text{s}$ vs $7.35\text{ ms}$ dynamic reallocation)
  - CUDA Graph Capture: Eliminates host dispatch latency and kernel launch overhead
  - Target Verification: Fused parallel scoring of draft candidate tokens
- **Measured End-to-End Speedup:** $1.48\times$ ($39.34\text{ tok/s}$ vs $26.59\text{ tok/s}$)

## 3. Visual Encodings
- **Architecture Diagram:** Drafting phase, CUDA graph execution, static KV-cache pointer rewind, and verification engine.
- **Color Palette:** Navy (`#1e3a8a`), Emerald (`#059669`), Cyan (`#0891b2`), Slate (`#334155`).

## 4. Takeaway
Pre-allocating KV-cache memory with O(1) pointer rollbacks and CUDA Graph capture eliminates host dispatch overhead, enabling speculative decoding to achieve a 1.48x net speedup on legacy T4 silicon.\n