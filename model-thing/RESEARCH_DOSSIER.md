# Master research dossier: Next-generation mathematical foundation models

## 1. Executive summary

Current mathematical language models face a structural barrier. Simply scaling autoregressive next-token prediction on textbook and web data yields diminishing returns on competition-level and research-level mathematics. Standard models exhibit arithmetic carry breakdown, intermediate algebraic hallucinations, reward hacking against learned reward models, and catastrophic branching collapse during long proof search.

This master research dossier provides an integrated, empirical, and epistemologically grounded blueprint for building frontier mathematical foundation models. Synthesizing four distinct investigation tracks, this dossier establishes five core architectural principles:

1. Independent Base Model Foundation: Raw leaderboard benchmarks (GSM8K, MATH) are extensively contaminated. Rigorous model selection depends on tokenizer mechanics (single-digit splitting vs multi-digit merging, dedicated LaTeX tokens), pre-training corpus exposure to code and formal literature, grouped query attention efficiency, and policy gradient stability. Qwen2.5-Math-7B and DeepSeek-Math-7B represent the optimal open 7B backbones for verifiable reinforcement learning, while Qwen2.5-Math-72B and DeepSeek-V3 provide the strongest source distributions for synthetic distillation.
2. Disciplined Standard Methodology (Track A): Outcome-only supervision reinforces compensating errors and false-positive logic. State-of-the-art training requires a five-stage pipeline: two-tier decontamination (13-gram hash and MinHash LSH 8-gram Jaccard >= 0.7), tool-integrated reasoning (ToRA / SymPy sandbox execution), bounded rejection sampling fine-tuning (RFT plateauing after 2 rounds), Group Relative Policy Optimization (GRPO eliminating the critic network to save 50% VRAM), and formal assistant kernel checking (Lean 4 and Mathlib4).
3. Bold Training and Architectural Frontiers (Track B): Twenty-four distinct, testable hypotheses expand mathematical capabilities beyond discrete autoregression. These include Riemannian geodesic flow matching on continuous proof manifolds, SRAM-resident differentiable Gröbner basis and SAT co-execution kernels, dynamic entropy annealing via phased temperature schedules, Lakatosian minimax counterexample co-evolution, and metamorphic invariant-preserving regularization. Each hypothesis is backed by an iso-compute baseline and an explicit Gate 1 falsification protocol.
4. Epistemological Separation of Discovery from Presentation (Pure Mathematics Literature): Classical and modern masters of pure mathematics (Poincaré, Hadamard, Pólya, Lakatos, Grothendieck, Thurston, Tao, Bourbaki) demonstrate that published mathematical proofs represent an inverted, sanitized illusion. Human discovery is chaotic, heuristic, numerical, adversarial, and non-linear. AI models fail when forced to generate exploration and Bourbaki-style formal presentation in a single autoregressive token stream.
5. Actionable Engineering Recipe: A concrete five-phase production recipe guides the training of a frontier mathematical foundation model, combining tagged semantic reasoning scratchpads (`<explore>`, `<conjecture>`, `<counterexample_test>`, `<lemma_isolate>`, `<formal_synthesis>`), asymmetric reinforcement learning rewards, and automated compilation into formal theorem provers.

---

## 2. Integrated comparative synthesis matrices

### 2.1 Base model audit matrix

The audited models span edge scales (0.5B to 12B) and foundation scales (14B to 671B MoE). Raw leaderboards are discarded in favor of structural metrics.

| Model | Active / Total params | Attention & RoPE theta | Context window | Tokenizer & math handling | Community verification status | Primary failure modes | VRAM & deployment footprint | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-Math-1.5B** | 1.54B Dense | GQA (12Q/2KV), RoPE 10k | 4k | BPE 152k, single digits, LaTeX split | High independent validation | Zero-shot drift, hallucination at T>=0.7 | 3.2 GB BF16, single consumer GPU | Apache 2.0 |
| **Qwen2.5-Math-7B** | 7.61B Dense | GQA (28Q/4KV), RoPE 10k | 4k | BPE 152k, single digits, LaTeX split | SOTA 7B pure math baseline | Prompt format sensitivity, sign errors at T>0 | 15.2 GB BF16, single 24GB GPU | Apache 2.0 |
| **DeepSeek-Math-7B** | 6.9B Dense | MHA (32Q/32KV), RoPE 10k | 4k | Byte BPE 100k, single digits, whitespace | Verified RLVR/GRPO standard | Large KV cache (MHA), loop repetition on fail | 13.8 GB BF16, single 24GB GPU | DeepSeek Open |
| **Llama-3.1-8B** | 8.03B Dense | GQA (32Q/8KV), RoPE 500k | 128k | Tiktoken 128k, 3-digit chunking | Universal community baseline | Multi-digit arithmetic errors, LaTeX split | 16.1 GB BF16, single 24GB GPU | Llama 3.1 Community |
| **Mistral-NeMo-12B** | 12.2B Dense | GQA (32Q/8KV), RoPE 1M | 128k | Tekken 131k, single digits, dedicated LaTeX | Strong code/structured reasoning | Generalist base, skips proof steps | 24.5 GB BF16, fits in FP8 on 24GB | Apache 2.0 |
| **Mistral-7B-v0.3** | 7.25B Dense | GQA (32Q/8KV), RoPE 1M | 32k | BPE 32k, single digits, LaTeX split | Stable general base | Lower pure math token exposure | 14.5 GB BF16, single 24GB GPU | Apache 2.0 |
| **Gemma-2-2B** | 2.61B Dense | GQA (8Q/4KV), soft-capping | 8k (sliding) | SentencePiece 256k, GeGLU | Distilled compact reasoning | Soft-capping divergence during SFT | 5.4 GB BF16, single consumer GPU | Gemma Terms |
| **Gemma-2-9B** | 9.24B Dense | GQA (16Q/8KV), soft-capping | 8k (sliding) | SentencePiece 256k, GeGLU | Distilled pretraining efficiency | Loss spikes without soft-capping kernel | 18.5 GB BF16, single 24GB GPU | Gemma Terms |
| **Yi-1.5-9B** | 8.8B Dense | GQA (32Q/4KV), RoPE 5M | 4k (32k) | BPE 64k, single digits, GQA | High math continual pretraining | Small vocab, formula expansion | 17.6 GB BF16, single 24GB GPU | Apache 2.0 |
| **Yi-1.5-34B** | 34.3B Dense | GQA (56Q/8KV), RoPE 5M | 4k (200k YaRN) | BPE 64k, single digits, GQA | Sweet spot between 8B and 70B | Long derivation context drift | 68 GB BF16, single 80GB GPU | Apache 2.0 |
| **Qwen2.5-Math-72B** | 72.7B Dense | GQA (64Q/8KV), RoPE 10k | 4k | BPE 152k, single digits, LaTeX split | SOTA open math foundation | 4k context cap, over-confident lemmas | 145 GB BF16, 2x 80GB GPUs | Qianwen License |
| **Llama-3.1-70B** | 70.6B Dense | GQA (64Q/8KV), RoPE 500k | 128k | Tiktoken 128k, 3-digit chunking | Generalist 70B standard | Lower math density than Qwen2.5-Math | 141 GB BF16, 2x 80GB GPUs | Llama 3.1 Community |
| **DeepSeek-V3** | 37B / 671B MoE | MLA (128 heads), RoPE 10k YaRN | 160k | Byte BPE 129k, MTP, fine experts | Matches top proprietary models | High infra barrier (8x H100 cluster) | 680 GB FP8, 8x H100 SXM | DeepSeek Open |

### 2.2 Standard training methodology matrix (Track A)

| Training paradigm | Compute cost (training) | Verification latency | False positive risk | Primary failure mode | Recommended deployment role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Supervised CoT (SFT)** | Low ($1\times$) | Zero | High | Hallucinated reasoning steps, memorization of solution templates | Initial instruction tuning, establishing base formatting syntax |
| **Rejection sampling (RFT)** | Moderate ($2\times - 5\times$) | Low (offline batch) | Moderate | Mode collapse, failure on hard problems ($p(x) \to 0$), diminishing returns after round 2 | Post-SFT policy bootstrapping, expanding accessible solution space |
| **Process supervision (PRMs)** | High ($10\times$ for rollouts) | Moderate (evaluating $K$ step tokens) | Moderate | Reward hacking via filler phrases, length decay under product metric | Test-time search reranking (Best-of-N), tree search guidance |
| **RLVR (GRPO)** | High ($5\times - 10\times$) | Low to moderate (rule checks) | Low | Repetition loops, format degradation, extreme token length expansion | Frontier capability scaling, inducing autonomous self-correction |
| **Formal provers (Lean 4)** | Extreme ($20\times - 50\times$) | High (0.5s - 5.0s per tactic step) | Zero (kernel-level) | Autoformalization failures, search space explosion, REPL timeout bottlenecks | Rigorous Olympiad mathematics, verified theorem synthesis, kernel grounding |

### 2.3 Radical innovations taxonomy (Track B)

The 24 hypotheses span architectural latent priors and evolutionary dynamic search:

| ID Range | Frontier | Primary Target Bottleneck | Core Mechanisms | Decisive Validation Metric |
| :--- | :--- | :--- | :--- | :--- |
| **H01 - H06** | Continuous Latent Spaces and Flow Matching | Combinatorial branching collapse in discrete token search; lack of metric continuity | Riemannian geodesic flow matching, continuous-to-discrete homotopy continuation, bidirectional latent search | Formal pass@1 gains on MiniF2F, proof search depth reduction >= 40% |
| **H02, H09, H12** | Embedded Neuro-Symbolic Execution | Tool call serialization latency; lack of gradient flow through verification engines | Differentiable Gröbner basis and SAT SRAM kernels, compiler IR co-optimization, confluent Knuth-Bendix rewriting | Zero word-problem loop errors, elaboration error drops from 68% to <= 22% |
| **H03, H04, H07, H10** | Algebraic and Category-Theoretic Priors | Opacity to algebraic symmetries; failure of attention to model transitive deduction | Group-equivariant attention (Lie/permutation), monoidal category attention, mixed-curvature Lorentz graphs | 5x token efficiency on isomorphic algebra, 98% transitive deduction at depth 10 |
| **H13 - H15, H18** | Cognitive Search and Dialectical Dynamics | Premature entropy collapse; static sampling temperatures; irreversible autoregressive mistakes | Dynamic entropy annealing, Lakatosian minimax counterexample co-evolution, dual-system proposer-verifier, branch rollback | Pass@1 gains on AIME, 50% drop in false lemmas, 3.5x token throughput |
| **H16, H17, H21** | Concept Abstraction and Continual Learning | Repetitive subproof re-derivation; static training data saturation; context bloating | Automated lemma distillation (MDL / Sequitur), autotelic conjecture curriculum, sleep-phase offline consolidation | 40% proof length compression, non-zero out-of-distribution learning slope |
| **H19, H22 - H24** | Topological, Evolutionary, and Invariant Frontiers | Metric collapse in proof space; non-convex proof valleys; sensitivity to trivial re-labelings | Simplicial complex navigation, Socratic multi-agent RL, genetic AST code search, metamorphic invariance | Multi-premise Recall@10 gains, solving >= 20% of monolithic dead ends |

### 2.4 Epistemological pure mathematics matrix

| Thinker & Work | Core epistemological concept | Real-time discovery mechanism | Presentation artifact | Algorithmic analog for AI |
| :--- | :--- | :--- | :--- | :--- |
| **Henri Poincaré** (1908) | Unconscious incubation and aesthetic sieve | Chaotic preparation followed by subliminal stochastic idea recombination | Formal deductive verification paper | Speculative MCTS search with learned aesthetic value functions; early pruning |
| **Jacques Hadamard** (1945) | Non-verbal thought and mental imagery | Multimodal, spatial, kinetic, and geometric visualization | Linear algebraic notation and text | Decouple internal continuous latent search from external discrete token output |
| **George Pólya** (1945, 1954) | Heuristics and plausible reasoning (Analysis vs Synthesis) | Specialization (small n), analogy, working backward from goal to premises | Synthetic forward proof hiding backward analysis | Tagged scratchpad with small case testing, Python REPL, backward goal decomposition |
| **Imre Lakatos** (1976) | Proofs, refutations, and proof-generated concepts | Local and global counterexamples; monster-barring and lemma incorporation | Sanitized theorem statement with hidden lemmas built in | Adversarial counterexample generator; reward for finding bugs; dynamic lemma repair |
| **Alexander Grothendieck** (1986) | The Rising Sea and conceptual dissolution | Construct broad universal properties and functorial categories | Abstract scheme definitions, commutative diagrams | Modular lemma abstraction; identify universal invariants making proofs trivial |
| **William Thurston** (1994) | Human understanding vs formal logic strings | Multimodal communication, mental models, colloquial debate | Machine-checkable strings of formal axioms | Multi-level proof compiler; high-level conceptual sketch lowered into Lean 4 tactics |
| **Terence Tao** (2008) | Three cognitive stages (pre-rigorous, rigorous, post-rigorous) | Free movement between post-rigorous structural vision and micro-inequalities | Rigorous formal paper with all technical estimates verified | Three-tier inference engine: Post-rigorous plan -> Rigorous inequalities -> Formal check |
| **Nicolas Bourbaki** (1939) | Extreme structuralism and deductive linearization | Completely concealed; treated as irrelevant to mathematical truth | Definition -> Lemma -> Theorem -> Cor. deductive cascade | Deterministic Phase 2 proof compiler; token-efficiency penalty; strict ordering |

---

## 3. Unified foundation model architecture blueprint

To synthesize these insights, the diagram below illustrates the unified system flow for a next-generation mathematical foundation model. The system strictly separates divergent, stochastic discovery from disciplined, deterministic proof compilation.

```
+===================================================================================================+
|                        UNIFIED MATHEMATICAL FOUNDATION MODEL ARCHITECTURE                         |
+===================================================================================================+

                                    [ PROBLEM INPUT STATEMENT ]
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| PRE-PROCESSING & INVARIANT MAPPING (H24: Metamorphic Normalizer & Suffix Decontaminator)           |
| - Strips formatting noise, canonicalizes algebraic variables, verifies benchmark non-overlap      |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| PHASE 1: STOCHASTIC DISCOVERY SCRATCHPAD (Poincaré / Hadamard / Pólya / Lakatos Engine)           |
|                                                                                                   |
| Dynamic Entropy Schedule (H13): T in [1.2, 1.6] -> T in [0.05, 0.2] via Progress Head             |
|                                                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | <explore>                                                                                   |  |
|  |   - Multi-branch associative generation; dimensional analysis; structural symmetry checks    |  |
|  |   - Continuous Geodesic Flow Guidance (H01) in latent proposition manifold M                |  |
|  | </explore>                                                                                  |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                |                                                  |
|                                                v                                                  |
|  +---------------------------------------------------------------------------------------------+  |
|  | <conjecture>                                                                                |  |
|  |   - Precise, falsifiable intermediate lemma candidate statements                            |  |
|  | </conjecture>                                                                               |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                |                                                  |
|                                                v                                                  |
|  +---------------------------------------------------------------------------------------------+  |
|  | <counterexample_test> (Adversarial Refuter H14 + Sandboxed Tool Co-Execution H02)          |  |
|  |   - Small-number checks (n = 0, 1, 2, -1), degenerate geometry cases, parity conditions     |  |
|  |   - Embedded Python REPL / SymPy CAS / SRAM Gröbner basis execution                         |  |
|  |   - Refutation Witness Found? -> Trigger <refutation_analysis> (Monster-Barring)            |  |
|  |   - Counterexample Passed?    -> Proceed to Lemma Isolation                                 |  |
|  | </counterexample_test>                                                                      |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                |                                                  |
|                                                v                                                  |
|  +---------------------------------------------------------------------------------------------+  |
|  | <lemma_isolate> (Grothendieckian Conceptual Distillation H16)                               |  |
|  |   - Extract minimal sufficient invariants barring all discovered boundary failures          |  |
|  |   - Structure-preserving lemma subgraph compression (MDL Sequitur mining)                   |  |
|  | </lemma_isolate>                                                                            |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 | Surviving Minimal Lemma Graph & Invariants
                                                 v
+---------------------------------------------------------------------------------------------------+
| COMPILATION & CONSOLIDATION FILTER (Tao Post-Rigorous Translation & H21 Sleep-Phase Refactoring)  |
| - Discards all failed search branches, dead ends, arithmetic scrapings, and exploratory tokens    |
| - Topologically sorts verified lemma dependencies into a linear directed acyclic graph (DAG)      |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| PHASE 2: DETERMINISTIC PROOF COMPILER (Bourbaki / Hilbert Formal Synthesis Engine)                 |
|                                                                                                   |
| Low Temperature Greedy Decoding (T = 0.0) with Grammar and Tactic Constraints                     |
|                                                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | <formal_synthesis>                                                                          |  |
|  |   - Tier 1: Post-Rigorous Plan (High-level conceptual roadmap and invariant definitions)    |  |
|  |   - Tier 2: Rigorous Deductive Cascade (Definition -> Lemma -> Theorem -> Corollaries)      |  |
|  |   - Tier 3: Formal Kernel Translation (Lean 4 / Isabelle tactic script via REPL daemon)    |  |
|  | </formal_synthesis>                                                                         |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| MECHANICAL VERIFICATION KERNEL (Lean 4 Mathlib4 / Isabelle / SymPy Typechecker)                   |
| - Kernel Type Check Passes (Q.E.D.) -> Emit Verified Proof Artifact                               |
| - Kernel Exception Encountered     -> Feedback precise tactic error to Tier 2 for local repair    |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Actionable five-phase training recipe and engineering roadmap

To build an open-weights mathematical foundation model that competes with leading closed-source systems, engineering teams must execute a disciplined five-phase roadmap:

### Phase 1: Base pre-training and vocabulary optimization
1. Architectural Backbone Selection:
   - For edge scale (single node): Select Qwen2.5-Math-7B-Base or DeepSeek-Math-7B-Base as the initialization point. Both possess single-digit tokenizer splitting and high baseline mathematical token density.
   - For foundation scale (multi-node cluster): Select Qwen2.5-Math-72B-Base (dense) or DeepSeek-V3 (MoE with Multi-Head Latent Attention).
2. Tokenizer Surgery:
   - Verify that all decimal digits (`0` through `9`) are isolated tokens.
   - Inject dedicated single tokens for high-frequency LaTeX macros (`\mathbb`, `\times`, `\neq`, `\cdot`, `\binom`, `\frac`) to eliminate macro fragmentation and reduce sequence length on formulas by 15% to 20%.
3. Pre-training Mix & Continual Pre-training:
   - Continually pre-train on 150 Billion tokens: 40% curated LaTeX web mathematics (OpenWebMath, MathWeb), 30% academic literature (arXiv math, physics, computer science), 20% formal code (Lean 4 Mathlib, Isabelle AFP, Python SymPy), and 10% high-quality natural language reasoning text.
   - Execute strict two-tier decontamination: scan against GSM8K, MATH, OlympiadBench, AMC, AIME, PutnamBench, and MiniF2F using exact 13-gram hash matching and MinHash LSH (k=8 shingles, 128 hash functions, Jaccard threshold 0.7).

### Phase 2: Tool-integrated cold-start SFT with tagged reasoning semantics
1. Dataset Construction (300,000 High-Fidelity Traces):
   - Synthesize problem-solution pairs using backward reasoning (generating parameters and answers first, then inverting equations to guarantee solvable problems).
   - Augment with Evol-Instruct to scale problem difficulty across Olympiad and undergraduate mathematics.
   - Format every training trace into explicit semantic tags: `<explore>`, `<conjecture>`, `<counterexample_test>`, `<refutation_analysis>`, `<lemma_isolate>`, and `<formal_synthesis>`.
   - Embed executable Python / SymPy code blocks into the `<counterexample_test>` section, training the model to write verification scripts that test small cases ($n=0, 1, 2$) and boundary conditions.
2. Fine-Tuning Execution:
   - Train for 3 epochs using standard cross-entropy loss with learning rate $1 \times 10^{-5}$, cosine decay, and sequence packing up to 8,192 tokens.
   - Ensure that the model learns strict tag closure and delimiter discipline.

### Phase 3: Bounded rejection sampling fine-tuning (RFT)
1. Offline Trajectory Sampling:
   - Sample $N = 16$ independent trajectories per problem at temperature $T = 0.7$ with top-$p = 0.95$.
   - Execute deterministic verification: run embedded SymPy / Python blocks in a 5-second CPU sandbox and check final answers against canonicalized targets.
2. Two-Round Iteration Cap:
   - Compile the verified positive traces into $\mathcal{D}_{\text{RFT}}$ and update the policy via supervised cross-entropy loss.
   - Execute exactly two iterative rounds. Terminate after round 2 to avoid mode collapse, distribution shift toward verbose wandering paths, and the plateau of diminishing returns.

### Phase 4: Reinforcement learning with verifiable rewards (RLVR via GRPO)
1. Policy Optimization Configuration:
   - Deploy Group Relative Policy Optimization (GRPO). Completely eliminate the Critic value network, saving 50% of GPU cluster VRAM and avoiding value-function estimation instability over long derivations.
   - Sample groups of $G = 8$ completions per training prompt.
   - Compute group-normalized advantages: $A_i = (r_i - \mu_r) / (\sigma_r + 10^{-4})$.
2. Asymmetric Reward Design:
   - In the exploration scratchpad: provide auxiliary rewards $R_{\text{refute}} = +0.5$ when the model successfully discovers a counterexample that invalidates a proposed conjecture, and $R_{\text{diversity}} = +0.2$ for testing boundary cases.
   - In the formal synthesis section: provide primary accuracy reward $r_{\text{acc}} \in \{0, 1\}$ evaluated by deterministic CAS / Lean type checker.
   - Enforce strict formatting constraints ($r_{\text{format}} = -1.0$ if tags are unclosed or text follows `</answer>`).
   - Apply 4-gram repetition penalties ($r_{\text{rep}} = -1.0$ if duplicate 4-gram frequency exceeds 0.15) to terminate degenerate token loops.
   - Apply length normalization to the surrogate loss to prevent trajectory lengthening from dominating gradient updates.
3. Emergent Cognitive Horizon:
   - Allow rollout length to expand up to 16,384 tokens, enabling the autonomous emergence of self-correction, recalculation, and backtracking loops.

### Phase 5: Test-time search and formal assistant compilation
1. Decoupled Inference Architecture:
   - Allocate 80% of test-time compute budget to Phase 1: run parallel stochastic exploration (temperature $T = 0.7$ to $1.0$, or speculative tree search with branch rollback H18) to discover invariants and verify boundary stability.
   - Allocate 20% of compute budget to Phase 2: compile the surviving lemma graph at temperature $T = 0.0$ into a concise, linear Bourbaki proof.
2. Formal Kernel Integration:
   - For applications requiring zero logical defect (formal theorem proving), connect the Phase 2 output to a Lean 4 REPL server daemon via JSON-RPC.
   - Use dense bi-encoder premise retrieval over Mathlib4 to suggest tactics.
   - Wrap all interactive tactic executions in 5-second operating system timeouts and recycle REPL daemon worker processes every 500 rollouts to prevent memory exhaustion.

---

## 5. Strategic conclusion and falsification ledger

Building a frontier mathematical foundation model is not an exercise in adding compute to naive autoregression. It requires aligning model architectures with the true epistemological structure of mathematical cognition:

- Tokenization matters: Single-digit tokenization and dedicated LaTeX vocabulary eliminate artificial arithmetic barriers before learning begins.
- Supervision must be structured: Process supervision and rule-based verification must replace outcome-only reward hacking.
- Exploration must be separated from compilation: Forcing models to think in the same linear token stream they use to report proofs cripples discovery.
- Mathematical rigor is evolutionary: As Lakatos demonstrated, theorems are forged through the dialectic of proof proposals and counterexample discoveries.

By combining the empirical rigor of Track A, the architectural frontiers of Track B, and the cognitive wisdom of the pure mathematics literature, this dossier provides an authoritative roadmap for creating systems that do not merely mimic mathematical syntax, but genuinely understand, discover, and verify new mathematics.
