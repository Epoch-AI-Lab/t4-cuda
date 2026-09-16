# Figure Specification: fig10_speculative_throughput_and_kv

## 1. Metadata
- **Figure Key:** `fig10_speculative_throughput_and_kv`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig10_speculative_throughput_and_kv.*`)
- **Generator:** `paper/generate_figures.py::fig10_speculative_throughput_and_kv`
- **LaTeX Reference:** `Figure~\ref{fig:spec_throughput_kv}`

## 2. Empirical Provenance
- **Data Source:** `results/benchmarks/t4_speculative_benchmark_report.json` & `results/benchmarks/m1_kv_cache_benchmark.json`
- **Hardware:** Physical Tesla T4 GPU (16 GB GDDR6)
- **Metrics:**
  - Autoregressive Baseline: 26.59 tok/s
  - Python-driven Speculative (K=2): 18.00 tok/s (host launch tax)
  - Python-driven Speculative (K=4): 17.50 tok/s (host launch tax)
  - Unified Speculative Serving Engine (Ours): 39.34 tok/s (1.48x net wall-clock speedup)
  - Rollback Latency: Dynamic `torch.cat` 7353.6 us vs `StaticKVCache` 0.51 us (14,388x faster)
  - DRAM Memory Traffic: Dynamic 120,344 KB/tok vs Static 56 KB/tok (2,149x cut)

## 3. Visual Encodings
- **Panel A:** Serving generation throughput bar chart comparing baseline, Python-taxed speculative, and unified engine with dashed baseline reference line.
- **Panel B:** Microbenchmark metric cards detailing rollback latency and DRAM memory traffic reduction.
- **Color Palette:** Emerald Green (`#059669`) for unified speedup, Slate Gray (`#94a3b8`) for baseline, Light Coral (`#fca5a5`) for host tax.

## 4. Takeaway
Overcomes the Python host dispatch tax on legacy GPUs, delivering a verified 1.48x net wall-clock throughput speedup via static KV-cache pointer rollback.
