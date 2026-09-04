# Project: Mathematical Foundation Models, Training Paradigms, and Pure Mathematics Research

## Architecture
The project is a multi-track research and synthesis investigation delivering an authoritative markdown research dossier on mathematical AI:
- Track 1 (R1): Independent Base Model Audit (0.5B-8B and 14B-70B+ tiers, tokenizer analysis, pre-training corpus, stability, failure modes).
- Track 2 (R2 Track A): Standard Training Paradigms (rejection sampling, PRMs, RLVR, synthetic traces, Lean 4 / Isabelle formal theorem proving).
- Track 3 (R2 Track B): Radical Training Innovations (uncapped collection of bold paradigms, mechanisms, risk profiles, testable hypotheses).
- Track 4 (R3): Pure Mathematics Literature (discovery heuristics vs. canonical proof presentation, algorithmic scratchpad analogies).
- Track 5 (R4): Dossier Compilation & Master Synthesis.
- Track 6: Quality, Rigor & Integrity Verification.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Base Model Audit: Edge Tier (0.5B-8B) | Architectural suitability, tokenizer analysis on LaTeX, corpus composition, stability, failure modes | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Base Model Audit: Foundation Tier (14B-70B+) | DeepSeek-Math, Qwen2.5-Math, Llama-3.1, Yi-Math; tokenization efficiency, community replication, failure modes | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Independent Validation Criteria | Elimination of contaminated/gamed leaderboard metrics; rigorous empirical reproduction checks | M1 | ORIGINAL_REQUEST §R1 |
| 4 | Track A: Data Curation & Synthetic Traces | Web extraction, synthetic CoT, question-answer filtering, deductive step generation | M2 | ORIGINAL_REQUEST §R2 |
| 5 | Track A: Step-Level Verification & PRMs | Process supervision, Monte Carlo tree search scoring, Math-Shepherd, token/step credit assignment | M2 | ORIGINAL_REQUEST §R2 |
| 6 | Track A: RL with Verifiable Rewards (RLVR) | Rule-based verifiers, GRPO/PPO, DeepSeek-R1-Zero dynamics, reward hacking mitigation | M2 | ORIGINAL_REQUEST §R2 |
| 7 | Track A: Formal Assistants & Proof Checkers | Lean 4, Isabelle/HOL, autoformalization, premise selection, tactic generation | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Track B: Neuro-Symbolic & Architectural Frontiers | Continuous-discrete latent manifolds, co-execution compilers, algebraic inductive bias | M3 | ORIGINAL_REQUEST §R2 |
| 9 | Track B: Cognitive Search & Dynamic Exploration | Chaotic exploration, counterexample co-evolution, dual-system fast/slow reflection, lemma distillation | M3 | ORIGINAL_REQUEST §R2 |
| 10 | Pure Math: Canonical Final Proof Structure | Bourbaki architecture, definition-lemma-theorem-corollary hierarchy, concise notation, rigor | M4 | ORIGINAL_REQUEST §R3 |
| 11 | Pure Math: Real-Time Discovery Dynamics | Poincaré, Hadamard, Polya, Lakatos, Grothendieck, Thurston, Tao; messy scratchpads, dead-ends | M4 | ORIGINAL_REQUEST §R3 |
| 12 | Algorithmic Analogies for Models | Translating messy discovery vs. clean output into concrete model scratchpad & reasoning architectures | M4 | ORIGINAL_REQUEST §R3 |
| 13 | Dossier: Audit Ledger | Comprehensive markdown ledger of examined base models with licensing and architectural notes | M5 | ORIGINAL_REQUEST §R4 |
| 14 | Dossier: Methodology Hypothesis Cards | Detailed hypothesis cards for Track A & B with mechanisms, risks, and falsification criteria | M5 | ORIGINAL_REQUEST §R4 |
| 15 | Dossier: Literature Synthesis Index | Full index analyzing human discovery vs. formal presentation with model analogies | M5 | ORIGINAL_REQUEST §R4 |
| 16 | Master Research Dossier | Executive summary and integrated blueprint synthesizing all four research streams | M5 | ORIGINAL_REQUEST §R4 |
| 17 | Adversarial Quality & Rigor Audit | Independent review verifying unslop compliance, no benchmark faking, and complete requirement coverage | M6 | Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Base Model Audit | Independent audit of 0.5B-8B and 14B-70B+ base models (tokenization, architecture, corpus, failure modes) | none | DONE |
| M2 | Standard Training Paradigms (Track A) | Analysis of proven industry workflows (rejection sampling, PRMs, RLVR, synthetic traces, Lean 4 / Isabelle) | none | DONE |
| M3 | Radical Training Innovations (Track B) | Expansive set of bold, non-traditional training hypotheses with mechanisms, risks, and falsification tests | none | DONE |
| M4 | Pure Mathematics Literature Synthesis | Real-time discovery vs canonical proof structure (Poincaré, Polya, Lakatos, Grothendieck, Thurston, Tao) | none | DONE |
| M5 | Dossier Compilation & Master Synthesis | Full markdown deliverables in project target directory (AUDIT_LEDGER, METHODOLOGY_TRACK_A, METHODOLOGY_TRACK_B, PURE_MATH_LITERATURE_SYNTHESIS, RESEARCH_DOSSIER) | M1, M2, M3, M4 | DONE |
| M6 | Adversarial Audit & Quality Gate | Independent verification of all deliverables against acceptance criteria, unslop style, and technical rigor | M5 | DONE |

## Code Layout & Deliverables
Target Directory: `/home/kriday/teamwork_projects/chock_math_research`
- `AUDIT_LEDGER.md`: R1 deliverable: Independent base model audit ledger
- `METHODOLOGY_TRACK_A.md`: R2 Track A deliverable: Standard best practices
- `METHODOLOGY_TRACK_B.md`: R2 Track B deliverable: Radical innovations & hypothesis cards
- `PURE_MATH_LITERATURE_SYNTHESIS.md`: R3 deliverable: Pure mathematics epistemological & algorithmic synthesis
- `RESEARCH_DOSSIER.md`: R4 deliverable: Master synthesis and actionable executive blueprint

Agent Working Directories (`.agents/`):
- `.agents/teamwork_preview_orchestrator_1/`: Orchestrator metadata, plan, progress, briefing
- `.agents/teamwork_preview_explorer_base_models/`: Track 1 working directory
- `.agents/teamwork_preview_explorer_track_a/`: Track 2 working directory
- `.agents/teamwork_preview_explorer_track_b1/`: Track 3A working directory
- `.agents/teamwork_preview_explorer_track_b2/`: Track 3B working directory
- `.agents/teamwork_preview_explorer_pure_math/`: Track 4 working directory
- `.agents/teamwork_preview_worker_synthesizer/`: Track 5 working directory
- `.agents/teamwork_preview_critic_reviewer/`: Track 6 working directory
