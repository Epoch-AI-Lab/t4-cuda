# Figure Specification: fig5_sft_reasoning_flowchart

## 1. Metadata
- **Figure Key:** `fig5_sft_reasoning_flowchart`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig5_sft_reasoning_flowchart.*`)
- **Generator:** `paper/generate_figures.py::fig5_sft_reasoning_flowchart`
- **LaTeX Reference:** `Figure~\ref{fig:sft_schema}`

## 2. Empirical Provenance
- **Model:** `Qwen2.5-Math-1.5B` fine-tuned on physical Tesla T4 GPU (LoRA $r=32, \alpha=64$)
- **Dataset:** 605 certified reasoning traces (`chalk_seeds_500.jsonl`, 445k tokens)
- **5-Tag Schema:**
  - `<explore>`: Hypothesis generation and search space mapping
  - `<conjecture>`: Formal invariant and claim formulation
  - `<test_edge_cases>`: Adversarial boundary condition probing
  - `<lemma_isolate>`: Deconstruction into provable sub-claims
  - `<formal_proof>`: Rigorous proof derivation and final synthesis
- **Measured Impact:** 6x boost in grounding sanity pass rate (60.0% vs 10.0%) and complete suppression of repetition loops.

## 3. Visual Encodings
- **Structured Schema Workflow:** Sequence of 5 procedural reasoning stages with condition branching, fallback loops, and verification gates.
- **Color Palette:** Emerald (`#059669`), Blue (`#2563eb`), Amber (`#d97706`), Purple (`#7c3aed`), Teal (`#0d9488`).

## 4. Takeaway
A 5-tag scientific discovery reasoning schema enforced via prompt loss masking structures model reasoning into disciplined verification phases, boosting grounding pass rate 6-fold.\n