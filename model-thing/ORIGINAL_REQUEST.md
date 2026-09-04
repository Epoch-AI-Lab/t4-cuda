# Original User Request

## Initial Request — 2026-09-04T10:44:07+05:30

Investigate open-weights base foundation models for mathematics without relying on raw leaderboard benchmarks, identify standard and radical training paradigms for mathematical reasoning, and survey pure mathematics literature on how mathematicians structure final results versus messy real-time discovery.

Working directory: ~/teamwork_projects/chock_math_research
Integrity mode: development

Requested team: Research team of sub-agents: one analyzing established training methods, multiple sub-agents brainstorming bold innovative paradigms without artificial caps, alongside base model auditors and pure mathematics literature researchers.

## Requirements

### R1. Independent Mathematical Base Model Audit
Identify and audit the strongest open-weights base foundation models for mathematics across both edge scales (0.5B to 8B) and larger foundation scales (14B to 70B+). 
- Do not rely on self-reported leaderboard scores (e.g. standard GSM8K/MATH benchmarks), as benchmarks can be gamed or contaminated.
- Evaluate models based on architectural suitability, pre-training corpus composition, tokenization efficiency on LaTeX and symbolic math, and documented community replication.
- Detail failure modes and behavioral stability under zero-shot and temperature perturbations.

### R2. Training Methodology: Standard Practices and Radical Innovations
Investigate training methodologies divided across two distinct research tracks:
- Track A (Standard Best Practices): Document industry-proven workflows for mathematical reasoning (such as rejection sampling fine-tuning, process-supervised reward models, verifiable reinforcement learning like RLVR, synthetic step-by-step trace generation, and formal assistant integration like Lean 4 / Isabelle).
- Track B (Radical Innovation): Ideas are cheap, implementation is not. Brainstorm as many bold, innovative, and non-traditional training approaches as possible without artificial caps (whether 10, 20, or 50). The sole criterion is that each idea aims to meaningfully build a better mathematical model. Propose concrete mechanisms, risk profiles, and testable hypotheses.

### R3. Pure Mathematics Literature: Final Structure vs. Messy Exploration
Survey foundational and modern pure mathematics literature (outside computer science and machine learning) with a sharp dual focus:
- Final Result Structure: Analyze how real human mathematicians structure their final, published results (canonical definitions, lemma-theorem-corollary hierarchies, clean proof architecture, and concise notation).
- Real-Time Discovery vs. Final Proof: Contrast the clean final output against the reality that real-time mathematical discovery is messy, iterative, and non-linear (scratchpad trials, dead-end discarding, counterexample probing, and intuitive leaps).
- Extract concrete algorithmic analogies on how models can maintain chaotic exploration in reasoning scratchpads while converging on disciplined, clean final mathematical structures.

### R4. Structured Research Dossier Deliverable
Compile all findings into a structured markdown dossier in ~/teamwork_projects/chock_math_research containing:
- An audit ledger of examined base models with licensing and architectural notes.
- Detailed hypothesis cards for training methods (both established standards and radical ideas), highlighting proposed mechanisms and falsification criteria.
- A literature synthesis index analyzing how human mathematical proofs are discovered versus how they are formally presented.

## Acceptance Criteria

### Base Model Audit
- At least 4 open-weights base models covering small (0.5B-8B) and larger (14B-70B+) tiers are evaluated with explicit tokenization and architectural analyses.
- No recommendation is justified solely by raw benchmark leaderboard rankings; every model recommendation includes independent evidence or documented reproduction.

### Methodology Hypotheses
- Track A covers data curation, step-level verification, and reward design with specific codebase and literature citations.
- Track B provides an expansive set of innovative training hypotheses without an artificial ceiling, each detailing a proposed mechanism and how it could be tested or falsified.

### Mathematics Literature Synthesis
- Pure mathematics works are analyzed for both real-time discovery dynamics (messy scratchpads, heuristics) and canonical final proof architecture.
- Concrete recommendations are provided for how a model can separate messy internal exploration from polished, rigorous final outputs.

### Deliverable Completeness
- All reports, hypothesis cards, and source indices are organized and written to ~/teamwork_projects/chock_math_research.
