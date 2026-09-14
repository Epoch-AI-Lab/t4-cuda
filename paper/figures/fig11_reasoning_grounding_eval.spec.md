# Figure Specification: fig11_reasoning_grounding_eval

## 1. Metadata
- **Figure Key:** `fig11_reasoning_grounding_eval`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig11_reasoning_grounding_eval.*`)
- **Generator:** `paper/generate_figures.py::fig11_reasoning_grounding_eval`
- **LaTeX Reference:** `Figure~\ref{fig:reasoning_eval}`

## 2. Empirical Provenance
- **Data Source:** `results/benchmarks/quarantined_benchmark.json`, `results/benchmarks/external_eval_results.json`, `results/benchmarks/base_eval_results.json`, and Table 6 (`tab:sft_eval`)
- **Hardware:** Physical Tesla T4 GPU (Fine-tuning Qwen2.5-Math-1.5B with LoRA r=32, alpha=64)
- **Evaluation Benchmark:** 150 quarantined zero-contamination questions & competition math problems
- **Metrics:**
  - Base Model (1.5B): Grounding Sanity 10.0% (1/10), Honest Abstention 6.7%, Hallucinations 93.3%, Factual QA 76.7%, Schema Adherence 0.0%, Repetition Loops 38.2%
  - SFT Tuned Model (Ours): Grounding Sanity 60.0% (6/10, 6.0x boost), Honest Abstention 53.3% (8.0x boost), Hallucinations 46.7% (-50% cut), Factual QA 73.3%, Schema Adherence 100.0%, Repetition Loops 0.0%

## 3. Visual Encodings
- **Panel A:** Abstention & Grounding on Traps (%) comparing Grounding (10.0% vs 60.0%), Honest Abstention (6.7% vs 53.3%), and Hallucination rates (93.3% vs 46.7%) between Base and SFT models.
- **Panel B:** Accuracy & 5-Tag Schema Adherence showing Factual QA preservation (76.7% vs 73.3%), Schema Adherence (0% -> 100%), and Degenerate Repetition elimination (38.2% -> 0.0%).
- **Color Palette:** Emerald Green (`#059669`) for improvements, Coral Red (`#f87171`) for error baseline, Blue (`#2563eb`) for schema adherence.

## 4. Takeaway
5-tag procedural reasoning conditioning via prompt loss masking on legacy T4 silicon yields a 6-fold grounding improvement and an 8-fold honest abstention boost on unanswerable traps while maintaining factual QA accuracy.\n