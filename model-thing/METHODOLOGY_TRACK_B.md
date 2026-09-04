# Radical training innovations and non-traditional hypotheses for mathematical foundation models
## 1. Executive summary
Standard autoregressive language models approach mathematics through discrete token prediction over flattened text representations. This paradigm imposes severe structural bottlenecks:
1. Combinatorial branching collapse during multi-step proof search, where a single local error corrupts the full sequence.
2. Inherent opacity to algebraic symmetries, forcing models to memorize operations like commutativity and associativity through sheer parameter scale.
3. Decoupled tool usage, where symbolic engines and formal proof checkers exist only as external text APIs, denying internal transformer layers direct algebraic signals.
4. Premature entropy collapse in search dynamics, locking models into shallow heuristics rather than allowing chaotic exploration and principled backtracking.
5. Inability to extract durable conceptual abstractions or discover reusable lemmas during reinforcement learning.

Track B formulates 24 distinct, bold, non-traditional training hypotheses designed to surpass these limitations. The catalog spans two complementary research frontiers:
- Part 1 (Hypotheses 01 to 12): Architectural, latent space, and neuro-symbolic frontiers, embedding continuous manifolds, differentiable SMT and computer algebra kernels, topological routers, and category-theoretic attention.
- Part 2 (Hypotheses 13 to 24): Dynamic search, cognitive architectures, and evolutionary frontiers, introducing phased entropy annealing, Lakatosian minimax counterexample co-evolution, dual-system dialectics, automated lemma distillation, and metamorphic invariance.

Every hypothesis card adheres strictly to the Gate Protocol of ML research rigor: each specifies a concrete mathematical mechanism, training loss formulation, failure modes, an iso-compute baseline, and a falsification test with registered kill conditions.
## 2. Master inventory of 24 radical training hypotheses
The table below catalogs all 24 hypotheses across their core paradigm, target bottleneck, parameter scale, and primary falsification metric.

| # | Hypothesis title | Research frontier | Core mechanism | Decisive falsification metric |
|---|---|---|---|---|
| 01 | Continuous geodesic trajectory planning in latent proof manifolds | Part 1: Latent Space | Riemannian flow matching on proof manifolds | Lean 4 pass@1 on MiniF2F-test gain >= 8.0 percentage points |
| 02 | Intra-layer differentiable SMT and computer algebra co-execution kernels | Part 1: Neuro-Symbolic | Differentiable Gröbner basis and SAT kernel in transformer layers | MATH-Algebra accuracy gain >= 12.0 percentage points at <= 50 ms/token |
| 03 | Dynamic computation graphs via homological topological routing | Part 1: Architecture | Persistent homology Betti number routing across 16 domain experts | Putnam / Olympiad multi-domain gain >= 6.0 percentage points vs top-2 MoE |
| 04 | Group-equivariant attention preserving algebraic symmetries | Part 1: Architecture | Lie group and finite group equivariant attention projections | 5x token efficiency on isomorphic algebra problems |
| 05 | Bidirectional forward-backward latent proof search | Part 1: Latent Space | Forward hypothesis and backward goal streams meeting in latent space | Proof search depth reduction >= 40% on Lean 4 PutnamBench |
| 06 | Continuous-to-discrete homotopy continuation training | Part 1: Latent Space | Continuous deformation of solved theorems into hard conjectures | Pass rate >= 18% on Olympiad problems where RLVR scores 0% |
| 07 | Mixed-curvature geometric dependency graph attention over Mathlib | Part 1: Architecture | Product manifold Lorentz attention over Mathlib dependency DAGs | Premise selection Recall@10 gain >= 15.0 percentage points on MiniF2F |
| 08 | Dual-clock multi-resolution hierarchical attention | Part 1: Architecture | Asynchronous macro-planner (1/16 clock) and micro-executor (full clock) | Proof validity maintenance >= 85% on sequences > 4000 tokens |
| 09 | Formal verification compiler loss via differentiable IR co-optimization | Part 1: Neuro-Symbolic | Surrogate tree-edit and type-checking loss on Lean elaboration states | Lean 4 elaboration and type error drop from 68% to <= 22% |
| 10 | Monoidal category compositionality priors in attention kernels | Part 1: Architecture | Morphism compositionality and identity constraints on attention weights | Multi-step transitive deduction accuracy >= 98% at chain length >= 10 |
| 11 | Differentiable hyperbolic curvature adaptation for proof tree navigation | Part 1: Latent Space | Dynamic Ricci curvature scaling in Poincare ball based on entropy | 35% lower perplexity and >= 8.0 points higher recall at 40% fewer params |
| 12 | Differentiable Knuth-Bendix term rewriting attention modules | Part 1: Neuro-Symbolic | Learned confluent term rewriting inside attention blocks | 100% accuracy on length-30 equational word problems vs < 25% baseline |
| 13 | Dynamic entropy annealing via phased scratchpad temperature modulation | Part 2: Search Dynamics | Three-phase entropy schedule with learned progress estimator | Pass@1 on MATH-500 / AIME gain >= 4.5 points with >= 98% valid syntax |
| 14 | Lakatosian minimax counterexample co-evolution | Part 2: Adversarial Search | Prover-Refuter zero-sum game with monster-barring and lemma incorporation | 50% drop in subtle false lemmas, >= 40% discovery on degenerate cases |
| 15 | Asynchronous dual-system dialectic proposer-verifier architecture | Part 2: Cognitive Search | Speculative proposer streaming to formal verifier via shared KV cache | 3.5x token throughput on formal proofs at >= 92% verification yield |
| 16 | Automated lemma distillation and concept abstraction during RL | Part 2: Concept Formation | Subgraph mining and MDL extraction of recurrent proof motifs | 40% proof length compression, >= 6.0 points pass@1 gain on AMC/AIME |
| 17 | Autotelic curriculum entropy via self-generated conjectures | Part 2: Self-Directed RL | Frontier conjecture generation maximizing policy validation entropy | Sustained non-zero learning slope on out-of-distribution math |
| 18 | Speculative multi-branch scratchpad with branch-and-bound rollback | Part 2: Search Dynamics | Tree-structured scratchpad with checkpointing and state rewinding | Recovery yield >= 30% on dead-end branches, 2.8x faster than MCTS |
| 19 | Simplicial complex and hypergraph proof-space navigation | Part 2: Topological Search | Higher-order multi-premise relations embedded via nerve complexes | Recall@10 gain >= 9.0 percentage points on multi-premise formal goals |
| 20 | Negative trace distillation from productive dead ends | Part 2: Contrastive Learning | Contrastive learning on error pivot tokens and recovery trajectories | 45% reduction in repeated derivation errors, +5.5 points pass@1 |
| 21 | Sleep-phase offline proof consolidation and refactoring | Part 2: Continual Learning | Offline wake-sleep refactoring compressing sprawling exploration | 60% context reduction without accuracy loss; zero catastrophic forgetting |
| 22 | Socratic dialectical multi-agent RL for collaborative theorem discovery | Part 2: Multi-Agent Search | Role-specialized agents (algebraist, geometer, critic) in turn-based dialogue | Solves >= 20% of problems unsolvable by monolithic 32B models |
| 23 | Evolutionary proof-space code search with semantic mutation operators | Part 2: Genetic Algorithms | Genetic programming in Lean 4 AST space using semantic crossover | Closes >= 15% of open goals failing under 1000-sample MCTS |
| 24 | Metamorphic invariant-preserving proof perturbation regularization | Part 2: Invariance | Contrastive loss preserving proof topology under isomorphic transformations | Cross-isomorphism consistency >= 96% (reducing 14% gap to < 2%) |
---
## Part 1: Architectural, latent space, and neuro-symbolic frontiers (cards 01 to 12)
## Hypothesis card 01: Continuous geodesic trajectory planning in latent proof manifolds
### 1. Title and core concept
Continuous geodesic trajectory planning in latent proof manifolds. 
Rather than generating proofs as sequences of discrete text tokens, this approach models proof generation as finding a minimum-energy geodesic path on a continuous Riemannian manifold of mathematical propositions. Discrete proof tactics are extracted only as boundary projections at trajectory waypoints.

### 2. Target LLM bottleneck in current models
Current autoregressive models treat theorem proving as an open-ended token generation problem. Each step requires sampling from vocabulary $|V| \approx 32{,}000$ to $128{,}000$. A single invalid token choice or erroneous premise reference creates catastrophic compounding errors. Search algorithms like Monte Carlo Tree Search in discrete token space suffer from severe combinatorial branching factors (often exceeding $10^2$ viable tactic prefixes per state). Discrete token representations also prevent smooth interpolation between distinct mathematical concepts or lemmas.

### 3. Concrete mechanism and mathematical formulation
Let $\mathcal{M}$ be a smooth $d$-dimensional Riemannian manifold with metric tensor $g$. A mathematical assertion $A$ (theorem statement, premise, or intermediate state) is mapped into $\mathcal{M}$ via an encoder $\mathcal{E}: \mathcal{T} \to \mathcal{M}$.
Given premises $P = \{p_1, \dots, p_k\}$ and target goal $T$, let $z_0 = \mathcal{E}(P)$ and $z_1 = \mathcal{E}(T)$.
A proof is defined as a smooth curve $\gamma: [0, 1] \to \mathcal{M}$ with boundary conditions $\gamma(0) = z_0$ and $\gamma(1) = z_1$.
The curve minimizes the action functional:
$$S(\gamma) = \int_0^1 \left[ g_{\gamma(t)}(\dot{\gamma}(t), \dot{\gamma}(t)) + V(\gamma(t)) \right] dt$$
where $V: \mathcal{M} \to \mathbb{R}_{\ge 0}$ is a learned potential energy function that penalizes regions of mathematically inconsistent or unprovable propositions.

We train a continuous-time vector field $v_\theta(z_t, t)$ using Riemannian flow matching. Given pairs of adjacent states in verified formal proofs $(s_i, s_{i+1})$, the ground-truth trajectory is defined by the geodesic segment connecting their latent embeddings. The vector field loss is:
$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, q(z_0, z_1), z_t} \left[ \| v_\theta(z_t, t) - \dot{\psi}_t(z_0, z_1) \|_g^2 \right]$$
where $\psi_t(z_0, z_1)$ is the geodesic interpolation under metric $g$.
Once the continuous trajectory $\hat{\gamma}(t)$ is integrated via a numerical ODE solver, a discrete decoder $\mathcal{D}$ autoregressively predicts formal Lean 4 tactic blocks conditioned on latent anchor points $z_{t_k} = \hat{\gamma}(k/K)$ via cross-attention.

```
[Premises P] ---> Encoder E ---> z_0                                       \  Continuous Geodesic Flow v_theta(z_t, t)
                                       +=========================================> Decoder D ---> [Formal Proof Steps]
                                      /  (Minimizing Riemannian Action S(gamma))
[Goal T]     ---> Encoder E ---> z_1 /
```

### 4. Training protocol and loss formulation
1. Dataset: 120,000 formal proofs extracted from Lean 4 Mathlib, Isabelle AFP, and Coq libraries. Each proof is broken into discrete proof states $s_0, s_1, \dots, s_N$.
2. Stage 1 (Metric alignment): Train encoder $\mathcal{E}$ with contrastive metric learning such that the geodesic distance $d_g(\mathcal{E}(s_i), \mathcal{E}(s_j))$ directly correlates with the minimum number of formal tactic steps separating $s_i$ and $s_j$.
3. Stage 2 (Flow matching): Train the 12-layer continuous-time Diffusion Transformer (DiT) vector field $v_\theta$ on geodesic paths between premise states and goal states.
4. Stage 3 (Discrete tactic decoding): Freeze $v_\theta$ and train the autoregressive tactic decoder on Lean 4 tactic sequences conditioned on sampled latent trajectory points.

#### Architectural details and compute budget tradeoffs
- Encoder: 7B parameter bidirectional transformer outputting latent vector $z \in \mathbb{R}^{4096}$.
- Trajectory generator: 12-layer DiT, hidden dimension 2048, 16 heads, 500M parameters.
- Tactic decoder: 7B parameter autoregressive transformer with cross-attention over trajectory tokens.
- Compute requirements: Stage 1 requires 16x H100 GPUs for 4 days (approx 1,500 GPU hours). Stage 2 requires 32x H100 GPUs for 6 days (approx 4,600 GPU hours). Stage 3 requires 16x H100 GPUs for 5 days.
- Inference profile: Solving a conjecture executes 32 steps of an adaptive Runge-Kutta (RK4) ODE solver over the 500M DiT, followed by 7B autoregressive generation over 8 anchor points. Per-token latency increases by 1.8x, but search iterations drop by 10x relative to MCTS.

### 5. Risk profile and potential failure modes
1. Trajectory drift: The continuous path may traverse valid regions of $\mathcal{M}$ that correspond to no realizable discrete mathematical tactic.
2. Decoder hallucination: The discrete decoder may fail to invert continuous latent states back to syntactically valid Lean 4 code.
3. Metric collapse: Contrastive metric learning may collapse distinct mathematical branches into degenerate low-dimensional submanifolds.

### 6. Concrete falsification test
- Method: Geodesic Flow Prover (7B base + 500M DiT) vs baseline DeepSeek-Math-7B fine-tuned on identical Lean 4 Mathlib traces.
- Setting: MiniF2F-test benchmark (488 formal competition problems) evaluated on 8x H100 GPUs with budget of 64 sampled trajectories / rollouts per problem.
- Primary metric: Formal verification pass@1 rate in Lean 4 environment.
- Claim: Geodesic Flow Prover improves pass@1 by >= 8.0 percentage points over DeepSeek-Math-7B at equal total inference FLOP budget.
- Falsifier: If pass@1 improvement $\Delta < 3.0$ percentage points, or if >= 40% of generated latent trajectories fail to decode into syntactically valid Lean 4 tactics, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will argue that continuous paths smooth out non-convex logical contradictions. If the potential $V(z)$ fails to produce high barriers around false statements, trajectory planning will cut across mathematically impossible shortcuts.

---

---
## Hypothesis card 02: Intra-layer differentiable SMT and computer algebra co-execution kernels
### 1. Title and core concept
Intra-layer differentiable SMT and computer algebra co-execution kernels. 
Embed differentiable Computer Algebra System (CAS) and Satisfiability Modulo Theories (SMT) solver kernels directly into transformer hidden layers. Rather than calling external tools via text input/output, intermediate representations are pruned and normalized algebraically during the forward pass.

### 2. Target LLM bottleneck in current models
Current tool-augmented LLMs treat calculators, SymPy, and Z3 as external text APIs called during generation. This approach has three fatal flaws:
1. Massive round-trip serialization overhead: expressions must be decoded into text strings, passed over IPC sockets, parsed by Python, solved, and serialized back.
2. Zero gradient flow: the model cannot optimize representations through symbolic verification.
3. No intermediate pruning: standard layers cannot detect that an equation system has an empty solution set until hundreds of tokens have been generated downstream.

### 3. Concrete mechanism and mathematical formulation
At layers 8, 16, 24, and 32 of a 36-layer transformer, insert a Differentiable Symbolic Block (DSB).
The DSB consists of two parallel GPU kernels running in SRAM:
1. Differentiable polynomial reduction (Gröbner basis kernel): Projects hidden states into polynomial coefficients $c \in \mathbb{R}^{m \times k}$ over variables $x_1, \dots, x_n$. It executes an unrolled, temperature-softened Buchberger algorithm. S-polynomial formation and reduction are softened using smooth minimums:
$$\text{soft-rem}(f, g) = f - \sum_i \sigma_\tau(\langle \text{LT}(f), \text{LT}(g_i) \rangle) \cdot \frac{\text{LT}(f)}{\text{LT}(g_i)} g_i$$
where $\sigma_\tau$ is a temperature-scaled sigmoid and $\text{LT}$ extracts leading terms.
2. Differentiable SAT solver: Projects attention activations into a continuous boolean satisfiability problem represented as a factor graph. It runs $T=5$ iterations of neural belief propagation to compute marginal clause satisfiability.

Let $h_l \in \mathbb{R}^{B \times L \times d}$ be the hidden state at layer $l$. The DSB computes:
$$z_{sym} = \text{LayerNorm}(\mathbf{W}_{proj} h_l)$$
$$z_{reduced}, s_{sat} = \text{Kernel}_{SRAM}(z_{sym})$$
$$h_{l+1} = h_l + \alpha_l \cdot \mathbf{W}_{out} [z_{reduced} \mathbin{\Vert} s_{sat}]$$
where $\alpha_l = \text{sigmoid}(\mathbf{w}_g^T h_l)$ is a dynamic gating scalar. Gradients backpropagate through the unrolled symbolic solver using the Implicit Function Theorem:
$$\frac{\partial z^*}{\partial h_l} = - \left[ \nabla_z F(z^*, h_l) \right]^{-1} \nabla_{h_l} F(z^*, h_l)$$
avoiding the need to store intermediate unrolled solver steps in activation memory.

```
[Layer l Hidden State h_l] ----------------------------------------(+)-----> [Layer l+1]
         |                                                          ^
         v                                                          |
  Linear Projection W_proj                                    Gated Output
         |                                                    alpha_l * W_out
         v                                                          |
  +-------------------------------------------------------------+   |
  | GPU SRAM Symbolic Co-Execution Kernel                       |   |
  | 1. Soft Buchberger Polynomial Reduction -> z_reduced        |---+
  | 2. Neural Belief Propagation SAT Kernel -> s_sat            |
  +-------------------------------------------------------------+
```

### 4. Training protocol and loss formulation
1. Synthetic algebra and logic dataset: 50 million pairs of polynomial systems with known Gröbner bases and SAT instances from SATLib with known satisfiability certificates.
2. Mathlib and GSM-Symbolic: Natural language mathematical proofs parsed into symbolic equations and deduction clauses.
3. Loss function:
$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \| z_{reduced} - z_{true\_basis} \|_2^2 + \lambda_2 \text{BCE}(s_{sat}, y_{sat})$$
where $\lambda_1 = 0.1$ and $\lambda_2 = 0.05$.

#### Architectural details and compute budget tradeoffs
- Base model: 8B parameter dense transformer (36 layers, $d=4096$).
- DSB blocks: Added at 4 layers. Each block adds 35M parameters in projection and gating matrices (140M parameters total, < 2% parameter increase).
- Kernel implementation: Written in Triton and CUDA C++, keeping all polynomial coefficient arrays and factor graph messages strictly in 96 KB SM shared memory (SRAM) per streaming multiprocessor to avoid HBM roundtrips.
- Compute overhead: Training wall-clock increases by 18%. Inference latency increases by 12 ms per token (total generation latency <= 48 ms/token on an NVIDIA A100-SXM4-80GB).

### 5. Risk profile and potential failure modes
1. Numerical divergence: Soft polynomial division can produce gradient explosions when leading coefficients approach zero.
2. Degree explosion: High-degree polynomial reductions cannot fit within fixed SRAM bounds, requiring fallback truncation that drops higher-order terms.
3. Over-regularization: The network may learn to suppress the gating factor $\alpha_l \to 0$ to avoid the loss penalty of symbolic inconsistencies.

### 6. Concrete falsification test
- Method: 8B Transformer with Intra-Layer SMT Co-Execution vs baseline 8B Transformer equipped with external Python SymPy/Z3 tool-use via standard text generation.
- Setting: MATH dataset (Algebra and Number Theory subsets) and GSM-Hard benchmark, evaluated on 4x A100-80GB GPUs with identical inference time budget (<= 50 ms/token).
- Primary metric: Solve accuracy on MATH-Algebra and GSM-Hard.
- Claim: Intra-layer co-execution improves solve accuracy by >= 12.0 percentage points over the tool-use baseline at matched inference latency.
- Falsifier: If accuracy gain $\Delta < 4.0$ percentage points, or if inference latency exceeds 65 ms/token, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will claim the gains stem purely from extra parameters in the projection heads. The ablation removing the SRAM symbolic kernel while keeping projection parameters must show no significant gain over baseline.

---

---
## Hypothesis card 03: Dynamic computation graphs via homological topological routing
### 1. Title and core concept
Dynamic computation graphs via homological topological routing. 
Replace heuristic token-level Mixture-of-Experts (MoE) routing with a mathematically grounded router based on persistent homology. The router extracts topological invariants from problem syntax trees and routes representations to specialized mathematical domain experts (algebra, topology, analysis, combinatorics).

### 2. Target LLM bottleneck in current models
Current MoE models (such as Mixtral or DeepSeek-V3) route individual tokens using simple dot-product gating against expert centroids. This causes mathematical tokens to scatter across disparate experts without semantic coherence. An algebraic variable $x$ in a group theory proof may be routed to an analysis expert, destroying domain-specific deductive consistency. Mathematical branches possess fundamentally distinct inferential structures (for example, discrete combinatorial proofs rely on bijectivity and induction, while real analysis relies on epsilon-delta bounds and compactness). Forcing them into shared generalist weights induces severe gradient interference.

### 3. Concrete mechanism and mathematical formulation
Let the input mathematical text sequence be $X = (x_1, \dots, x_N)$.
We construct an abstract syntax graph $G_X = (V_X, E_X)$ using a lightweight dependency parser.
We compute the Vietoris-Rips filtration on the graph distance metric space of $G_X$. A 50M parameter graph neural network extracts the persistent homology signature:
$$\mathbf{b}(X) = [b_0(\epsilon_1), b_0(\epsilon_2), \dots, b_1(\epsilon_k), \dots, b_2(\epsilon_m)] \in \mathbb{R}^{128}$$
where $b_k(\epsilon)$ are the Betti numbers at filtration radius $\epsilon$, capturing connected components, cycles, and higher-dimensional voids in the mathematical reasoning graph.

The router computes expert assignment probabilities for 16 specialized domain experts:
$$p_i(X) = \text{softmax}\left( \frac{\mathbf{W}_{route} [\bar{h}_X \mathbin{\Vert} \mathbf{b}(X)]}{\tau} \right)_i, \quad i \in \{1, \dots, 16\}$$
where $\bar{h}_X$ is the pooled sequence embedding, and $\tau$ is a learned temperature.
To prevent parameter interference, we enforce structural orthogonality across expert weight matrices:
$$\mathcal{L}_{orth} = \sum_{i=1}^{16} \sum_{j \neq i}^{16} \frac{\text{Tr}(\mathbf{W}_i^T \mathbf{W}_j)^2}{\|\mathbf{W}_i\|_F^2 \|\mathbf{W}_j\|_F^2}$$

```
[Input Math Sequence X] ---> Parser ---> Graph G_X ---> Vietoris-Rips Filtration
                                                              |
                                                              v
[Pooled Embedding h_X] -------------------------> Router (Homological Signature b(X))
                                                              |
             +--------------------+---------------------------+--------------------+
             v                    v                           v                    v
      [Expert 01: Algebra]  [Expert 02: Analysis]      [Expert 03: Topology]  [Expert 04: Combinatorics]
```

### 4. Training protocol and loss formulation
1. Pre-training corpus: 300 billion tokens of mathematical literature categorized by ArXiv math subject classifications (math.AG, math.CA, math.CO, math.PR, etc.).
2. Two-stage training:
   - Stage 1: Train individual branch experts independently on domain-pure subsets for 50 billion tokens to establish specialized feature representations.
   - Stage 2: Joint training on mixed mathematical text using homological routing with load-balancing loss and the weight orthogonality penalty $\mathcal{L}_{orth}$.

#### Architectural details and compute budget tradeoffs
- Total parameters: 14B parameters across 16 experts.
- Active parameters: 3.2B parameters per token (top-2 expert activation).
- Router cost: Persistent homology extraction and GNN forward pass take 1.8 ms per sequence of 2048 tokens, consuming < 4% of forward pass FLOPs.
- Memory: Requires 32 GB VRAM per GPU across 8x A100-80GB GPUs with tensor parallelism.

### 5. Risk profile and potential failure modes
1. Interdisciplinary failure: Problems spanning multiple disciplines (such as algebraic topology or analytic number theory) may suffer from routing instability.
2. Cold start collapse: If the homological signature fails to separate early in training, the router collapses to a uniform distribution, reducing to a standard poorly-balanced MoE.
3. GNN latency bottleneck: On very long proofs (> 8192 tokens), graph construction and filtration calculation may become compute-bound.

### 6. Concrete falsification test
- Method: Homological MoE (14B total, 3.2B active) vs standard top-2 token-choice MoE (14B total, 3.2B active) trained on identical 200B token math corpus.
- Setting: Multi-domain benchmark comprising Putnam Competition problems (1995 to 2024) and International Mathematical Olympiad (IMO) Shortlists.
- Primary metric: Solve accuracy across multi-step interdisciplinary proofs.
- Claim: Homological MoE improves multi-domain solve accuracy by >= 6.0 percentage points over standard MoE at identical active FLOPs.
- Falsifier: If accuracy gain $\Delta < 2.0$ percentage points, or if single-domain accuracy degrades by > 3.0 percentage points due to expert isolation, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will argue that token-level routing naturally discovers domain clusters without explicit topological parsing. The test must ablate the topological signature $\mathbf{b}(X)$ to prove it outperforms standard token-level clustering.

---

---
## Hypothesis card 04: Group-equivariant attention preserving algebraic symmetries
### 1. Title and core concept
Group-equivariant attention preserving algebraic symmetries. 
Design self-attention mechanisms that are strictly equivariant under algebraic group actions. If an equation or mathematical expression is transformed by a group operation (such as variable renaming, commutative operand swapping, or basis transformations), the attention representations transform equivariantly by construction.

### 2. Target LLM bottleneck in current models
Standard transformer attention is permutation-equivariant with respect to sequence position indices when positional encodings are removed, but completely oblivious to algebraic symmetries. For example, the relations $a + b = c$, $b + a = c$, and $x + y = z$ are syntactically distinct token sequences. Standard transformers must spend vast parameter capacity and billions of training tokens memorizing that mathematical truth is invariant under variable renaming and commutative reordering. When presented with unfamiliar variable names or permuted expressions at test time, standard models exhibit severe performance drops.

### 3. Concrete mechanism and mathematical formulation
Let $G$ be a symmetry group acting on mathematical expressions. Specifically, let $G = S_k \times (\mathbb{Z}_2)^m$, representing permutations of $k$ independent variables and $m$ commutative binary operators.
Let the token feature vectors transform under a linear representation $\rho: G \to GL(d)$.
We define Group-Equivariant Multi-Head Attention (GEMHA). For queries, keys, and values:
$$Q(g \cdot x) = \rho_Q(g) Q(x), \quad K(g \cdot x) = \rho_K(g) K(x), \quad V(g \cdot x) = \rho_V(g) V(x)$$
To guarantee that the attention matrix $A_{ij}$ is invariant under group transformations $g \in G$:
$$A_{ij} = \frac{1}{\sqrt{d}} \int_G \langle \rho_Q(g) Q_i, \rho_K(g) K_j \rangle dg$$
For compact finite groups, the Haar integral reduces to a finite sum over group elements:
$$A_{ij} = \frac{1}{|G| \sqrt{d}} \sum_{g \in G} Q_i^T \rho_Q(g)^T \rho_K(g) K_j$$
We parameterize the projection matrices $W_Q, W_K, W_V$ as linear combinations of Clebsch-Gordan harmonic basis tensors:
$$W_Q = \sum_{J} c_J \mathbf{C}_J$$
where $\mathbf{C}_J$ are fixed projection tensors satisfying the Wigner-Eckart theorem for group $G$, and $c_J$ are learnable scalar coefficients. This guarantees $f(g \cdot X) = \rho_{out}(g) f(X)$ for every layer $f$.

```
[Input Tokens X] ---> Projection via Clebsch-Gordan Basis C_J ---> Equivariant Q, K, V
                                                                          |
                                                                          v
[Group Action g in G] ---> Permutes Variables ---> Attention Invariant: A_{ij}(gX) = A_{ij}(X)
                                                                          |
                                                                          v
[Output Representation] <--- Equivariant Value Mixing <-------------------+
```

### 4. Training protocol and loss formulation
1. Data pipeline: Formal algebra expressions from SymPy, Abstract Algebra theorem corpora, and Lean 4 identities. Automated parser identifies the automorphism group $\text{Aut}(E)$ for each mathematical expression $E$.
2. Equivariance regularization: In addition to standard language modeling loss, add an equivariance penalty on intermediate layer representations:
$$\mathcal{L}_{equiv} = \mathbb{E}_{g \sim G, X} \left[ \| f(\rho_{in}(g) X) - \rho_{out}(g) f(X) \|_2^2 \right]$$
3. Curriculum: Train initially on symmetric polynomial identities and linear algebra invariants before moving to general mathematical proofs.

#### Architectural details and compute budget tradeoffs
- Model size: 3B parameter dense transformer.
- Parameter count: GEMHA projection layers require 12% more parameters due to multi-basis decomposition, but zero increase in KV cache memory during inference.
- Attention kernel: Fast group-orbit summation implemented via a custom Triton kernel. Compute latency is 1.25x standard FlashAttention-2 for group $|G| \le 8$.
- Total compute budget: 8x H100 GPUs for 14 days on 100 billion math tokens (approx 2,700 GPU hours).

### 5. Risk profile and potential failure modes
1. Over-constrained expressivity: Imposing strict equivariance may impair the model on non-commutative algebraic structures (such as quaternion multiplication, free groups, or matrix multiplication).
2. Group identification failure: Incorrectly inferring the symmetry group of an informal natural language sentence will apply the wrong equivariance constraint.
3. Computational scaling with group size: For large permutation groups $S_k$ with $k > 4$, summing over $|G| = k!$ elements becomes intractable without Monte Carlo subgroup sampling.

### 6. Concrete falsification test
- Method: 3B GEMHA Transformer vs 3B baseline Transformer with standard Rotary Position Embeddings (RoPE), both trained on 50B tokens of algebraic problems.
- Setting: Zero-shot evaluation on the Isomorphic Algebra Evaluation Suite (10,000 polynomial simplification and group theory problems tested under random variable permutations and symbol renamings).
- Primary metric: Token sample efficiency to reach 80% accuracy, and accuracy drop under variable renaming.
- Claim: GEMHA Transformer achieves 80% accuracy with >= 5x fewer training tokens than baseline, and maintains zero accuracy drop (delta <= 0.5%) under variable renaming where baseline drops by >= 15.0%.
- Falsifier: If sample efficiency gain is < 2.5x, or if accuracy on non-commutative algebra degrades by >= 5.0 percentage points, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will argue that data augmentation with permuted variable names achieves the same result without custom kernels. The baseline must include 4x data augmentation; GEMHA must beat it decisively in token efficiency.

---

---
## Hypothesis card 05: Bidirectional forward-backward latent proof search
### 1. Title and core concept
Bidirectional forward-backward latent proof search. 
Replace unidirectional autoregressive proof generation with a dual-stream architecture that simultaneously plans forward from hypotheses and backward from target goals, meeting at an intermediate bridging lemma in continuous latent space.

### 2. Target LLM bottleneck in current models
Current theorem-proving models generate proofs unidirectionally: either forward from axioms (which suffers from an exponential combinatorial explosion of irrelevant deductions) or backward via goal reduction (which frequently fails when proofs require introducing auxiliary constructions or unmentioned lemmas, such as constructing an auxiliary line in geometry or defining a clever helper function). Human mathematicians rarely generate proofs in a single forward or backward sweep; they work simultaneously from both ends until the gap closes.

### 3. Concrete mechanism and mathematical formulation
The architecture comprises two coupled transformer streams sharing 60% of lower-layer parameters:
1. Forward Stream $S_f$: Encodes premises $H$ and unrolls forward deduction states $z_f(t) \in \mathbb{R}^d$.
2. Backward Stream $S_b$: Encodes goal theorem $G$ and unrolls backward reduction states $z_b(t) \in \mathbb{R}^d$.
At each transformer block, bidirectional cross-attention layers exchange representations between forward and backward streams.

We define a continuous compatibility tensor $\mathbf{M} \in \mathbb{R}^{d \times d \times k}$. The latent gap between forward state $z_f$ and backward state $z_b$ is scored by:
$$\text{Gap}(z_f, z_b) = \sigma\left( z_f^T \mathbf{W}_{bridge} z_b \right)$$
A meeting-in-the-middle event occurs when $\text{Gap}(z_f, z_b) > 1 - \epsilon$.
When this condition is met, a Bridge Decoder $\mathcal{D}_{bridge}$ takes the concatenated latent vector $[z_f \mathbin{\Vert} z_b]$ and autoregressively generates the explicit formal bridging lemma $L_{bridge}$:
$$L_{bridge} = \mathcal{D}_{bridge}([z_f \mathbin{\Vert} z_b])$$
The full proof decomposes into:
$$\text{Proof}(H \implies G) = \text{Proof}(H \implies L_{bridge}) \mathbin{\Vert} \text{Proof}(L_{bridge} \implies G)$$

```
[Hypotheses H] ---> Forward Stream S_f  ---> z_f(1) ---> z_f(2)                                                                  +---> [Latent Intersection: Gap < epsilon]
[Goal G]       ---> Backward Stream S_b ---> z_b(1) ---> z_b(2) /              |
                                                                                v
                                                                   Bridge Decoder D_bridge
                                                                                |
                                                                                v
                                                                     [Bridging Lemma L_bridge]
```

### 4. Training protocol and loss formulation
1. Dataset: 200,000 formal proofs from Lean 4 and Isabelle. Proof dependency DAGs are bisected at their topological median cuts into forward premise trees and backward goal trees, with the connecting node labeled as ground-truth $L_{bridge}$.
2. Loss function:
$$\mathcal{L} = \mathcal{L}_{CE}(S_f) + \mathcal{L}_{CE}(S_b) + \mathcal{L}_{CE}(\mathcal{D}_{bridge}) + \lambda \mathcal{L}_{contrastive}$$
where $\mathcal{L}_{contrastive}$ pulls true meeting pairs $(z_f^*, z_b^*)$ together while pushing non-connecting forward/backward states apart.

#### Architectural details and compute budget tradeoffs
- Total parameter scale: 7B parameters. Lower 20 layers are shared between streams; upper 12 layers are split into dedicated forward and backward heads (6 layers each), plus a 2B parameter bridge decoder.
- Search algorithm: Latent A* beam search. Instead of expanding discrete tokens, the model expands candidate latent states guided by heuristic $h(z_f) = \min_{z_b} \|z_f - z_b\|_2$.
- Compute requirements: Training requires 32x A100-80GB GPUs for 8 days (approx 6,100 GPU hours). Search memory is 2x standard autoregression due to dual KV caches.

### 5. Risk profile and potential failure modes
1. False latent convergence: Forward and backward representations may appear close in cosine distance while representing semantically incompatible propositions.
2. Bridge lemma hallucination: The bridge decoder may emit a lemma that is provable from $H$ but does not actually imply $G$.
3. Asymmetric search collapse: One stream may progress much faster than the other, causing the search to degenerate into standard unidirectional search.

### 6. Concrete falsification test
- Method: Bidirectional Latent Search Transformer (7B) vs unidirectional forward-search 7B baseline and unidirectional backward-search 7B baseline on Lean 4 PutnamBench.
- Setting: 640 formal problems evaluated with equal total token generation budget (maximum 8192 tokens per problem).
- Primary metric: Solved proof rate (pass@1) and average proof search depth.
- Claim: Bidirectional search reduces average proof search depth by >= 40% and improves pass@1 by >= 7.0 percentage points over both unidirectional baselines.
- Falsifier: If search depth reduction is < 15%, or if bridging lemmas fail verification in Lean 4 in >= 50% of candidate solutions, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will assert that meeting in the middle in high-dimensional continuous space is statistically impossible due to the curse of dimensionality. The contrastive loss must prove that latent representations collapse onto a low-dimensional manifold where metric proximity guarantees logical compatibility.

---

---
## Hypothesis card 06: Continuous-to-discrete homotopy continuation training
### 1. Title and core concept
Continuous-to-discrete homotopy continuation training. 
Apply the mathematical principle of homotopy continuation to conjecture resolution. Rather than attacking an unsolved conjecture directly, construct a continuous family of problems deforming a simple, known theorem into the target conjecture. Solve the conjecture by tracking the proof path in latent space.

### 2. Target LLM bottleneck in current models
In reinforcement learning for theorem proving (such as RLVR or DeepSeek-R1-Zero), complex conjectures provide zero learning signal. The probability of an 8B or 70B model randomly generating a valid 100-step proof for an Olympiad-level problem is effectively zero, resulting in a flat zero-reward surface. Discrete curriculum learning helps but requires manual heuristic ordering of problems, failing when intermediate difficulty stepping stones do not exist in the training dataset.

### 3. Concrete mechanism and mathematical formulation
Let $C_{target}$ be an unresolved target conjecture, and let $C_{base}$ be a topologically similar, fully resolved base theorem with known formal proof $P_{base}$.
Let $z_{base} = \mathcal{E}(C_{base})$ and $z_{target} = \mathcal{E}(C_{target})$.
We construct a continuous homotopy path in latent problem space:
$$\mathcal{H}(z, \lambda) = (1 - \lambda) z_{base} + \lambda z_{target}, \quad \lambda \in [0, 1]$$
We define a proof state vector field $w_\lambda$ corresponding to the valid proof at deformation parameter $\lambda$.
The trajectory tracks the zero-set of an energy functional $\mathcal{V}(w, \lambda) = \| \text{VerifierScore}(w, \mathcal{H}(z, \lambda)) - 1 \|^2$.
Using numerical predictor-corrector continuation:
1. Predictor step (Euler step along path tangent):
$$w_{\lambda + \delta}^0 = w_\lambda + \delta \cdot \frac{dw}{d\lambda}$$
where $\frac{dw}{d\lambda} = - \left[ \nabla_w^2 \mathcal{V} \right]^{-1} \frac{\partial \nabla_w \mathcal{V}}{\partial \lambda}$.
2. Corrector step (Neural Newton-Raphson iterations):
$$w_{\lambda + \delta}^{k+1} = w_{\lambda + \delta}^k - \left[ \nabla_w^2 \mathcal{V} \right]^{-1} \nabla_w \mathcal{V}(w_{\lambda + \delta}^k, \lambda + \delta)$$
When continuation reaches $\lambda = 1$, the discrete decoder inverts $w_1$ into formal proof tactics for $C_{target}$.

```
[Solved Theorem C_base] ========================================> [Hard Conjecture C_target]
(lambda = 0, Proof w_0 known)                                      (lambda = 1, Proof w_1 sought)
          \                                                                 ^
           \---> Predictor: w_{lambda + delta}^0 = w_lambda + delta * dw/dlambda   |
            \---> Corrector: Neural Newton-Raphson Optimization ------------/
```

### 4. Training protocol and loss formulation
1. Data generation: Generate continuous families of mathematical problems using parameterized algebraic curves, differential equations, and geometric configurations. 1 million continuous deformation homotopy paths.
2. Hypernetwork training: Train a 2B parameter hypernetwork to output the path tangent $\frac{dw}{d\lambda}$ conditioned on problem embeddings.
3. RL refinement: Apply reinforcement learning along the continuation path, rewarding successful tracking across intermediate $\lambda$ values with dense intermediate rewards:
$$R = \int_0^1 \mathbb{I}(\text{Valid}(w_\lambda)) d\lambda$$

#### Architectural details and compute budget tradeoffs
- Architecture: 13B parameter base model coupled with a 2B parameter continuation hypernetwork.
- Inference compute: Continuation requires 20 predictor-corrector steps along $\lambda \in [0, 1]$. Total inference compute is approximately 4x standard autoregressive generation, but addresses problems where standard sampling has a 0% success rate.
- Training compute: 64x H100 GPUs for 10 days (approx 15,300 GPU hours).

### 5. Risk profile and potential failure modes
1. Bifurcation points: The homotopy path may encounter mathematical singularities where the proof structure undergoes a discontinuous phase transition, causing predictor divergence.
2. Topological mismatch: If $C_{base}$ and $C_{target}$ belong to fundamentally distinct homotopy classes, no continuous deformation path exists without passing through ill-defined mathematical spaces.
3. Accumulation of numerical error: Inaccurate corrector steps early in the path ($\lambda < 0.3$) compound, leaving $w_1$ far outside the convergence basin of $C_{target}$.

### 6. Concrete falsification test
- Method: Homotopy Continuation Training (13B + 2B) vs RLVR baseline (DeepSeek-R1-Zero style 13B model) on Level 5 Olympiad geometry and algebra problems.
- Setting: 200 difficult problems where the base 13B model scores 0% pass@64 under standard sampling.
- Primary metric: Solve rate on the 0%-pass benchmark set.
- Claim: Homotopy Continuation achieves >= 18.0% solve rate on problems where standard RLVR achieves 0.0%, within a compute budget <= 2x standard RLVR training.
- Falsifier: If solve rate on target problems is < 5.0%, or if continuation diverges at bifurcation points on >= 60% of test paths, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will argue that discrete logic cannot be continuously deformed. The test must specifically verify that continuous tracking preserves discrete proof validity across intermediate stages.

---

---
## Hypothesis card 07: Mixed-curvature geometric dependency graph attention over Mathlib
### 1. Title and core concept
Mixed-curvature geometric dependency graph attention over Mathlib. 
Embed the mathematical knowledge base into a mixed-curvature product manifold $\mathbb{M} = \mathbb{H}^{d_1} \times \mathbb{S}^{d_2} \times \mathbb{E}^{d_3}$. Compute attention over theorem dependency graphs using hyperbolic geometry for deep taxonomic hierarchies, spherical geometry for dualities and cyclic relations, and Euclidean geometry for flat local transformations.

### 2. Target LLM bottleneck in current models
Mathematical knowledge forms a massive directed acyclic graph (DAG) of definitions, lemmas, theorems, and proofs. In Lean 4 Mathlib, this graph contains over 150,000 declarations with paths exceeding depth 50. Standard transformers use Euclidean attention, which cannot embed hierarchical graph structures without exponential metric distortion. As trees branch exponentially, Euclidean space only expands polynomially ($r^d$), forcing hierarchical embeddings to crowd and lose distinction near leaves.

### 3. Concrete mechanism and mathematical formulation
We model mathematical knowledge in a product manifold:
$$\mathbb{M} = \mathbb{H}^{d_1}_{\kappa_1} \times \mathbb{S}^{d_2}_{\kappa_2} \times \mathbb{E}^{d_3}$$
where $\mathbb{H}^{d_1}_{\kappa_1}$ is a hyperbolic space with curvature $\kappa_1 < 0$ (capturing tree-like abstraction and depth), $\mathbb{S}^{d_2}_{\kappa_2}$ is a sphere with curvature $\kappa_2 > 0$ (capturing cyclic groups, dualities, and symmetries), and $\mathbb{E}^{d_3}$ is flat Euclidean space (capturing local token features).

For the hyperbolic component, we use the Lorentz (hyperboloid) model:
$$\mathbb{H}^{d} = \{ x \in \mathbb{R}^{d+1} : \langle x, x \rangle_L = -1, x_0 > 0 \}$$
where $\langle x, y \rangle_L = -x_0 y_0 + \sum_{i=1}^d x_i y_i$ is the Lorentzian inner product.
Lorentz distance between nodes $u$ and $v$ is:
$$d_{\mathbb{H}}(u, v) = \text{arcosh}\left( -\langle u, v \rangle_L \right)$$
The attention weight between node $i$ and node $j$ is computed directly on the manifold:
$$A_{ij} = \frac{\exp\left( - \frac{d_{\mathbb{M}}^2(Q_i, K_j)}{\tau} \right)}{\sum_{k} \exp\left( - \frac{d_{\mathbb{M}}^2(Q_i, K_k)}{\tau} \right)}$$
where $d_{\mathbb{M}}^2 = d_{\mathbb{H}}^2 + d_{\mathbb{S}}^2 + d_{\mathbb{E}}^2$. Premise selection is formulated as geodesic cone inclusion in $\mathbb{H}^{d_1}$.

```
Mathematical Knowledge Base DAG (Mathlib)
                  |
                  v
Product Manifold Embedding M = H^{d1} x S^{d2} x E^{d3}
- Hyperbolic H^{d1}: Taxonomic abstraction & lemma depth
- Spherical S^{d2}:  Cyclic symmetries & dualities
- Euclidean E^{d3}:  Local token features
                  |
                  v
Lorentz Multi-Head Attention: A_{ij} ~ exp( - d_M^2(Q_i, K_j) / tau )
```

### 4. Training protocol and loss formulation
1. Graph extraction: Extract the complete declaration DAG from Lean 4 Mathlib (150,000 nodes, 2.5 million directed dependency edges).
2. Riemannian pre-training: Pre-train node embeddings using Riemannian Stochastic Gradient Descent (RSGD) with retraction maps to minimize graph distortion loss:
$$\mathcal{L}_{dist} = \sum_{(u, v) \in E} \left| d_{\mathbb{M}}(x_u, x_v) - d_{graph}(u, v) \right|^2$$
3. Joint transformer training: Inject pre-trained manifold embeddings as structural positional encodings into a 7B parameter transformer fine-tuned on formal premise retrieval and theorem proving.

#### Architectural details and compute budget tradeoffs
- Dimensionality: $d_1 = 256$ (hyperbolic), $d_2 = 64$ (spherical), $d_3 = 512$ (Euclidean). Total hidden dimension per head matches standard 1024.
- Model scale: 7B parameter transformer.
- Kernel requirements: Custom Triton kernel for batched Lorentz inner products and arcosh functions in bfloat16. Eliminates numerical underflow near the light-cone boundary.
- Latency: Lorentz distance calculation adds 6% latency overhead during self-attention compared to standard dot-product attention.

### 5. Risk profile and potential failure modes
1. Numerical instability: Values of $-\langle u, v \rangle_L$ approaching $1.0$ cause gradient explosion in $\frac{d}{dx} \text{arcosh}(x) = \frac{1}{\sqrt{x^2 - 1}}$. Requires clamping $-\langle u, v \rangle_L \ge 1 + 10^{-6}$.
2. Curvature mismatch: Fixed curvature parameters $(\kappa_1, \kappa_2)$ may not match local graph density, requiring learnable per-layer curvature parameters.
3. Disconnected components: Premise subgraphs with isolated components cause unbounded hyperbolic distance drift.

### 6. Concrete falsification test
- Method: Mixed-Curvature Graph Attention Transformer (7B) vs standard Euclidean Graphormer (7B) and dot-product attention with RoPE on Lean 4 premise retrieval.
- Setting: MiniF2F and Isabelle AFP premise selection benchmarks (retrieving relevant lemmas from a corpus of 150k candidates given an open proof goal).
- Primary metric: Premise retrieval Recall@10 and Mean Reciprocal Rank (MRR).
- Claim: Mixed-Curvature Attention improves Recall@10 by >= 15.0 percentage points and MRR by >= 0.12 over Euclidean Graphormer at zero inference latency overhead.
- Falsifier: If Recall@10 gain is < 5.0 percentage points, or if Riemannian optimization fails to converge in > 5% of training runs, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will argue that dense Euclidean embeddings with high dimensionality ($d=4096$) can embed trees without distortion. The baseline must include high-dimensional Euclidean embeddings; the mixed-curvature model must outperform it at equal or lower parameter count.

---

---
## Hypothesis card 08: Dual-clock multi-resolution hierarchical attention
### 1. Title and core concept
Dual-clock multi-resolution hierarchical attention. 
Decompose mathematical reasoning across two distinct temporal clocks: an asynchronous macro-planner running at low temporal frequency (generating strategic proof milestones) and a micro-executor running at full token frequency (executing local algebraic manipulations and tactic syntax).

### 2. Target LLM bottleneck in current models
Current autoregressive models operate on a uniform token-by-token clock. The network burns identical compute and attention bandwidth on high-level strategic decisions ("apply mathematical induction on dimension $n$, then bound the error term via Cauchy-Schwarz") and low-level syntactic execution (balancing parentheses, expanding $(a+b)^2$, writing variable types). Over long proofs (> 4000 tokens), the quadratic attention context becomes saturated with micro-algebraic noise, causing the model to lose strategic focus and hallucinate dead-end steps.

### 3. Concrete mechanism and mathematical formulation
The architecture operates with two synchronized sub-networks:
1. Macro-Planner $\mathcal{M}_{macro}$: Operates on temporal clock $T_{macro} = \{0, 16, 32, \dots\}$. It consumes abstracted milestone tokens and emits high-level semantic intentions $S_k \in \mathbb{R}^{d_{macro}}$.
2. Micro-Executor $\mathcal{M}_{micro}$: Operates on temporal clock $T_{micro} = \{0, 1, 2, \dots\}$. It generates detailed formal tokens conditioned on the active macro-intention $S_k$ via continuous cross-attention.

Communication between clocks is event-driven:
The micro-executor runs autoregressively until it emits a structural boundary token (such as `have`, `obtain`, `show`, `qed`).
At this boundary, the micro-executor pools its intermediate activations into an execution summary vector:
$$e_k = \text{AttentionPool}(h_{t_{k-1}:t_k}^{micro})$$
The summary $e_k$ is passed upward to the macro-planner as a feedback token. The macro-planner performs a single planning step, updates its internal hidden state, and emits the next strategic intention:
$$S_{k+1} = \mathcal{M}_{macro}(S_{1:k}, e_{1:k})$$
Gradients backpropagate across the dual-clock interface using straight-through estimators for discrete boundary decisions.

```
Macro-Planner Clock (1/16):
[Strategy S_0] =============================================> [Strategy S_1]
      |                                                             ^
      v (Cross-Attention Conditioning)                             | (Summary e_1 at boundary)
Micro-Executor Clock (1/1):                                         |
[t_0: expand] -> [t_1: collect] -> [t_2: factor] -> [t_3: 'have'] --+
```

### 4. Training protocol and loss formulation
1. Data segmentation: 500,000 mathematical proofs from ArXiv and formal libraries segmented into hierarchical levels. Macro-steps correspond to lemmas, claims, and main deduction transitions. Micro-steps correspond to internal algebraic verification lines.
2. Two-phase training:
   - Phase 1: Train the macro-planner on proof skeletons and lemma dependency graphs.
   - Phase 2: Train the micro-executor on local step fulfillment conditioned on ground-truth macro-intentions.
   - Phase 3: End-to-end co-training with reinforcement learning (PPO) rewarding successful proof completion with hierarchical credit assignment.

#### Architectural details and compute budget tradeoffs
- Macro-Planner: 1.5B parameters, 16 layers, context length 2048 macro-tokens (equivalent to 32,768 micro-tokens).
- Micro-Executor: 7.0B parameters, 32 layers, local sliding window context length 2048 tokens.
- Total parameter scale: 8.5B parameters.
- Computational efficiency: Reduces KV cache memory by 60% and cuts inference decode FLOPs by 2.5x on long proofs (> 4000 tokens) because the heavy 7B micro-executor only attends to local tokens and the compact macro-state.

### 5. Risk profile and potential failure modes
1. Macro-micro deadlock: The macro-planner may emit an unprovable milestone, trapping the micro-executor in an infinite generation loop until context exhaustion.
2. Boundary detection error: If the micro-executor misses a structural boundary token, the macro-planner desynchronizes from the proof state.
3. Information bottleneck: Compressing 16 micro-steps into a single summary vector $e_k$ may drop fine-grained algebraic constraints required for subsequent macro-decisions.

### 6. Concrete falsification test
- Method: Dual-Clock Hierarchical Transformer (8.5B) vs flat autoregressive 8.5B transformer with identical total pre-training tokens.
- Setting: Long-horizon mathematical theorem proving benchmark (proofs requiring >= 4000 tokens, selected from Putnam competition and IMO long-form solutions).
- Primary metric: Long-proof completion validity rate and decode FLOP consumption.
- Claim: Dual-Clock Transformer maintains >= 85% logical validity on proofs exceeding 4000 tokens where the flat baseline drops below 30% validity, while consuming >= 2.5x fewer decode FLOPs.
- Falsifier: If validity improvement on proofs > 4000 tokens is < 20.0 percentage points, or if micro-executor deadlock occurs on >= 25% of runs, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will claim the hierarchy can be replicated by simple chain-of-thought prompting on standard models. The evaluation must test whether standard prompting fails on 4000+ token proofs due to context window degradation.

---

---
## Hypothesis card 09: Formal verification compiler loss via differentiable intermediate representation co-optimization
### 1. Title and core concept
Formal verification compiler loss via differentiable intermediate representation co-optimization. 
Co-train mathematical language models directly against the compiler intermediate representations (IR) of formal verification systems (such as Lean 4's Expression Tree and Elaboration State), using differentiable surrogate tree-edit and type-consistency losses.

### 2. Target LLM bottleneck in current models
Current autoformalization pipelines treat translation from informal math to formal code as standard machine translation. The loss function is pure token-level cross-entropy on surface text. Under this loss, missing a single comma, variable binder, or type annotation receives the same penalty as asserting a mathematically false lemma. The model receives zero gradient signal regarding Abstract Syntax Tree (AST) validity, type unification, or elaboration errors until text generation finishes and an external compiler rejects the file.

### 3. Concrete mechanism and mathematical formulation
During pre-training and fine-tuning on formal mathematics, we instrument the Lean 4 compiler to emit the intermediate elaboration state graph $\mathcal{T}_{IR}$ for each tactic step. $\mathcal{T}_{IR}$ contains typed expression DAGs with explicit de Bruijn indices, metavariable assignment tables, and universe level constraints.

We attach three auxiliary prediction heads to intermediate layers (layers 12, 24, and 32) of a 7B transformer:
1. AST Structure Head: Predicts the adjacency matrix of the elaborated expression DAG.
2. Type Vector Head: Predicts the continuous embedding of the expected type for each sub-expression.
3. Metavariable Resolution Head: Predicts whether open goals are closed by current term assignments.

The overall loss function optimizes both surface tokens and internal compiler representations:
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \beta_1 \mathcal{L}_{tree}(\hat{\mathcal{T}}_{IR}, \mathcal{T}_{IR}^*) + \beta_2 \mathcal{L}_{type}(\hat{\mathbf{v}}_{type}, \mathbf{v}_{type}^*) + \beta_3 \mathcal{L}_{meta}$$
where $\mathcal{L}_{tree}$ is a differentiable tree-Wasserstein distance between predicted and ground-truth elaboration DAGs:
$$\mathcal{L}_{tree} = \min_{\mathbf{T} \in \Pi} \sum_{i, j} \mathbf{T}_{ij} \| \hat{n}_i - n_j^* \|_2^2$$
and $\mathcal{L}_{type}$ penalizes type vector misalignment in a shared semantic type space. Auxiliary heads are discarded after training, resulting in zero inference latency penalty.

```
Transformer Hidden States (Layer 12, 24, 32)
       |                     |                     |
       v                     v                     v
[AST Structure Head]  [Type Vector Head]  [Metavariable Head]
       |                     |                     |
       v                     v                     v
  Tree-Wasserstein      Type Alignment     Metavariable Loss
  Loss L_tree           Loss L_type        L_meta
       \                     |                     /
        +--------------------+--------------------+
                             |
                             v
           Auxiliary Compiler Loss L_IR (Added to L_CE)
           (Auxiliary heads discarded during inference)
```

### 4. Training protocol and loss formulation
1. Compiler trace extraction: Instrument the Lean 4 server to export elaboration traces, AST graphs, and type tables for all 150,000 declarations in Mathlib.
2. Dataset alignment: Construct 800,000 aligned triples: (Informal Statement, Formal Code, Elaboration IR Graph).
3. Curriculum: Train initially with high auxiliary loss weights $(\beta_1 = 0.5, \beta_2 = 0.3)$ to force hidden representations to align with compiler semantics, decaying to $(\beta_1 = 0.1, \beta_2 = 0.05)$ in later training epochs.

#### Architectural details and compute budget tradeoffs
- Base model: 7B parameter dense transformer.
- Auxiliary heads: 3 heads, adding 180M parameters during training (discarded at inference).
- Memory overhead: Storing elaboration graphs increases training VRAM consumption by 22%, requiring batch size reduction from 4 to 3 per GPU on 80GB A100s.
- Compute budget: 16x A100-80GB GPUs for 12 days (approx 4,600 GPU hours).

### 5. Risk profile and potential failure modes
1. Compiler divergence: Updates to the Lean 4 compiler or syntax macros invalidate pre-computed elaboration target graphs.
2. Overfitting to compiler internals: The model may overfit to specific de Bruijn index numbering schemes rather than invariant mathematical semantics.
3. Graph size explosion: Pathological tactic expansions in Lean produce elaboration graphs with > 100,000 nodes, exceeding memory limits and requiring aggressive subgraph pruning.

### 6. Concrete falsification test
- Method: Compiler IR Co-Trained Transformer (7B) vs standard text-supervised 7B transformer fine-tuned on identical Lean 4 Mathlib code.
- Setting: Zero-shot autoformalization on ProofNet and MiniF2F-informal benchmarks (translating natural language math statements into verified Lean 4 code).
- Primary metric: Formal elaboration error rate (percentage of generated formal statements that fail Lean 4 elaboration or type-checking).
- Claim: Compiler IR Co-Training reduces formal elaboration error rate from 68% (baseline) to <= 22%, and improves pass@1 verified proof rate by >= 10.0 percentage points.
- Falsifier: If elaboration error rate remains > 45%, or if pass@1 gain is < 4.0 percentage points, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will argue that syntax errors can be repaired cheaply via test-time compiler feedback loops. The evaluation must prove that IR co-training produces higher initial semantic accuracy, outperforming 5 rounds of naive compiler feedback repair.

---

---
## Hypothesis card 10: Monoidal category compositionality priors in attention kernels
### 1. Title and core concept
Monoidal category compositionality priors in attention kernels. 
Constrain transformer self-attention weight matrices to satisfy the axioms of monoidal categories. Attention distributions are parameterized as morphisms in a dagger-compact closed category, guaranteeing strict transitivity, associativity, and functorial composition in multi-step deductive chains.

### 2. Target LLM bottleneck in current models
Current self-attention mechanisms compute unconstrained softmax distributions over token inner products. While flexible, this formulation lacks compositional priors. If a proof establishes that step $f: A \to B$ is valid and step $g: B \to C$ is valid, standard attention does not enforce that the composition $g \circ f: A \to C$ holds transitively. Autoregressive models frequently commit associativity violations and syllogistic dropouts, where each individual step appears plausible locally but the global multi-step composition fails catastrophically.

### 3. Concrete mechanism and mathematical formulation
We treat mathematical tokens as objects in a monoidal category $(\mathcal{C}, \otimes, I)$ and attention maps as morphisms.
For attention matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$, we enforce three structural axioms via soft manifold projections:
1. Identity: $\mathbf{A}_{ii} = 1$ (every object has an identity morphism $\text{id}_A: A \to A$).
2. Functorial Compositionality: For any three reasoning tokens $i, j, k$, the composed attention score satisfies:
$$\mathbf{A}_{ik} \ge \mathbf{A}_{ij} \cdot \mathbf{A}_{jk} - \epsilon$$
guaranteeing transitivity along deductive paths.
3. Monoidal Tensor Factorization: Multi-head attention heads represent tensor products $\otimes$. For head projections $H_1, H_2$:
$$\mathbf{A}_{H_1 \otimes H_2} = \mathbf{A}_{H_1} \otimes \mathbf{A}_{H_2}$$

We parameterize the attention score matrix via a continuous decomposition:
$$\mathbf{A} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} \right)$$
and add a Category-Theoretic Consistency Loss during pre-training:
$$\mathcal{L}_{cat} = \frac{1}{N^3} \sum_{i, j, k} \max\left( 0, \mathbf{A}_{ij} \mathbf{A}_{jk} - \mathbf{A}_{ik} - \epsilon \right)^2 + \| \text{diag}(\mathbf{A}) - \mathbf{1} \|_2^2$$
This regularizer penalizes intransitive attention loops and forces representations into a DAG-consistent partial order.

```
Standard Attention:                     Monoidal Category Attention:
   A -------> B                            A -------> B
    \        /                              \        /
     \      /                                \      /
      v    v                                  v    v
        C   (No Transitivity Bound)             C   (A_{ik} >= A_{ij} * A_{jk} - eps)
```

### 4. Training protocol and loss formulation
1. Synthetic categorical reasoning suite: 20 million synthetic syllogisms, poset chains, category diagram chases, and relational transitivity proofs with chain lengths varying from 3 to 20 steps.
2. Abstract algebra corpus: Formal category theory literature, group homomorphism graphs, and order theory libraries from Mathlib.
3. Loss weighting: Pre-train with $\mathcal{L} = \mathcal{L}_{CE} + \gamma \mathcal{L}_{cat}$, where $\gamma = 0.05$.

#### Architectural details and compute budget tradeoffs
- Model scale: 7B parameter dense transformer.
- Projection parameterization: Attention weights computed using low-rank product manifolds $\mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{U}^\dagger$ to inherently constrain eigenvalues.
- Training overhead: Evaluating $\mathcal{L}_{cat}$ across sampled triplets $(i, j, k)$ adds 8% training compute overhead. Zero inference latency overhead because category constraints are baked into the learned weights.
- Compute budget: 16x H100 GPUs for 10 days on 150B tokens (approx 3,800 GPU hours).

### 5. Risk profile and potential failure modes
1. Over-rigid constraints: Strict transitivity priors may impair the model on non-transitive or non-monotonic reasoning tasks (such as game theory or defeasible logic).
2. Optimization stalling: The cubic compositionality constraint $\mathbf{A}_{ij} \mathbf{A}_{jk} \le \mathbf{A}_{ik}$ can cause vanishing gradients if initialized far from a valid category structure.
3. Context length scaling: Triple sampling for $\mathcal{L}_{cat}$ becomes sparse on sequences > 4096 tokens, potentially degrading long-range compositional enforcement.

### 6. Concrete falsification test
- Method: Monoidal Category Attention Transformer (7B) vs standard 7B transformer baseline on Multi-Step Transitive Deduction Suite.
- Setting: Relational deduction chains with lengths $L \in \{5, 10, 15, 20\}$ (e.g., establishing relations in partially ordered sets and group homomorphisms).
- Primary metric: Accuracy on transitive conclusions at chain length $L=10$ and $L=15$.
- Claim: Monoidal Attention achieves >= 98.0% accuracy at $L=10$ and >= 90.0% at $L=15$, whereas standard baseline drops below 45.0% at $L=10$ and below 20.0% at $L=15$.
- Falsifier: If accuracy at $L=10$ is < 75.0%, or if general mathematical benchmark performance (GSM8K) degrades by >= 3.0 percentage points, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will argue that transitivity is learned naturally by large models. The test must demonstrate that even a 70B parameter standard model fails on length-15 synthetic transitive chains, while the 7B Monoidal model succeeds.

---

---
## Hypothesis card 11: Differentiable hyperbolic curvature adaptation for proof tree navigation
### 1. Title and core concept
Differentiable hyperbolic curvature adaptation for proof tree navigation. 
Formulate the hidden state representations of mathematical reasoning within a Riemannian manifold of variable negative curvature $\mathbb{H}^d_\kappa$. Dynamically adjust the local curvature parameter $\kappa(t) < 0$ based on the branching entropy of the current proof state.

### 2. Target LLM bottleneck in current models
Theorem proving is fundamentally a tree search process. The number of candidate deduction paths grows exponentially with proof depth. Flat Euclidean vector spaces possess polynomial volume growth ($V(r) \propto r^d$), forcing Euclidean embeddings to severely compress and distort deep combinatorial trees. Hyperbolic space $\mathbb{H}^d$, by contrast, features exponential volume growth ($V(r) \propto e^{(d-1)r}$), providing an isometric match for tree structures. However, fixed-curvature hyperbolic networks fail because mathematical proofs alternate between exponential branching phases (searching for lemmas) and linear deductive phases (unrolling algebraic equalities).

### 3. Concrete mechanism and mathematical formulation
We represent hidden states in the Poincaré ball model of hyperbolic geometry with variable negative curvature $\kappa < 0$:
$$\mathbb{B}_\kappa^d = \left\{ x \in \mathbb{R}^d : -\kappa \|x\|^2 < 1 \right\}$$
equipped with the Riemannian metric tensor:
$$g_x^\kappa = \lambda_x^2 \mathbf{I}, \quad \lambda_x = \frac{2}{1 + \kappa \|x\|^2}$$
Layer operations are executed using gyrovector algebra:
1. Möbius addition:
$$x \oplus_\kappa y = \frac{(1 - 2\kappa \langle x, y \rangle - \kappa \|y\|^2) x + (1 + \kappa \|x\|^2) y}{1 - 2\kappa \langle x, y \rangle + \kappa^2 \|x\|^2 \|y\|^2}$$
2. Möbius scalar multiplication and matrix-vector multiplication via exponential and logarithmic maps:
$$\mathbf{W} \otimes_\kappa x = \exp_0^\kappa\left( \mathbf{W} \log_0^\kappa(x) \right)$$

We introduce an adaptive curvature module that computes the local curvature $\kappa_l(t)$ for layer $l$ and proof step $t$:
$$\kappa_l(t) = - \text{softplus}\left( \mathbf{w}_\kappa^T h_l(t) + b_\kappa \right)$$
When branching entropy is high (many candidate proof tactics), $\kappa$ becomes strongly negative (e.g., $\kappa = -2.5$), expanding tree capacity. When the proof enters a deterministic linear calculation, $\kappa \to 0^-$, transitioning smoothly to Euclidean geometry.

```
Proof State: High Branching Entropy (Search Phase)
  --> Adaptive Module sets kappa = -2.5 (Strong Hyperbolic Expansion)
  --> Hidden States use exponential volume growth in Poincare Ball

Proof State: Deterministic Linear Calculation (Algebraic Reduction Phase)
  --> Adaptive Module sets kappa = -0.01 (Near-Euclidean Geometry)
  --> Hidden States transition smoothly to flat metric
```

### 4. Training protocol and loss formulation
1. Hierarchical math ontology dataset: MSC2020 (Mathematics Subject Classification) tree, Mathlib hierarchy, and 50,000 proof trees with explicit branching factors.
2. Optimization: Models trained using Riemannian Adam with adaptive gradient clipping to prevent boundary explosion near the Poincaré ball horizon $(\|x\| \to 1/\sqrt{-\kappa})$.
3. Meta-curvature loss:
$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_\kappa \sum_l \left| \kappa_l(t) - \kappa_{target}(\text{BranchEntropy}_t) \right|^2$$

#### Architectural details and compute budget tradeoffs
- Base scale: 7B parameter gyrovector transformer.
- Custom kernel: Requires fused Triton kernels for Möbius addition and Poincaré distance calculations in bfloat16.
- Latency profile: Hyperbolic operations add 22% forward-pass latency compared to standard matrix multiplication.
- Parameter efficiency: Hyperbolic representations reduce required hidden dimension by 40% (from $d=4096$ to $d=2456$) while matching tree embedding fidelity.

### 5. Risk profile and potential failure modes
1. Poincaré boundary trapping: Numerical underflow occurs when vectors approach the ball boundary, where the conformal factor $\lambda_x \to \infty$.
2. Curvature oscillation: Unstable meta-gradients may cause curvature $\kappa$ to oscillate wildly between layers, destroying training stability.
3. Loss of arithmetic efficiency: Simple linear vector arithmetic becomes computationally expensive under Möbius gyrovector transformations.

### 6. Concrete falsification test
- Method: Adaptive Hyperbolic Transformer (7B, $d=2456$) vs standard Euclidean Transformer (7B, $d=4096$) on Mathlib taxonomy classification and proof tree search.
- Setting: Premise retrieval and branch selection across 50,000 proof search trees in Lean 4.
- Primary metric: Tree-distance distortion, branch selection accuracy, and perplexity on hierarchical math concepts.
- Claim: Adaptive Hyperbolic Transformer achieves >= 35% lower tree-distance distortion, >= 8.0 percentage points higher branch selection accuracy, and equal perplexity while using 40% smaller hidden dimension.
- Falsifier: If branch selection accuracy fails to improve by >= 3.0 percentage points, or if training throughput drops by > 40% vs Euclidean baseline, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will assert that hyperbolic arithmetic is too unstable for 7B+ scale models. The test must establish complete training run stability (zero NaN loss events across 100B tokens) to survive review.

---

---
## Hypothesis card 12: Differentiable Knuth-Bendix term rewriting attention modules
### 1. Title and core concept
Differentiable Knuth-Bendix term rewriting attention modules. 
Integrate a differentiable term rewriting engine inside transformer self-attention blocks. The module learns and applies confluent, terminating reduction rules directly on symbolic expression trees, eliminating cyclic loops in equational reasoning.

### 2. Target LLM bottleneck in current models
Equational reasoning (proving that expression $A$ equals expression $B$ via substitution rules, such as simplifying algebraic expressions, matrix identities, or group relations) is a major failure mode for autoregressive LLMs. Standard LLMs treat rewriting as string token prediction. They easily get trapped in infinite rewriting loops (e.g., repeatedly applying commutativity $x + y \to y + x \to x + y$), hallucinate false simplifications on long expressions (> 15 terms), and fail to recognize normal forms.

### 3. Concrete mechanism and mathematical formulation
At layers 6, 12, 18, and 24 of a 28-layer transformer, insert an Equational Rewriting Module (ERM).
Expressions are parsed into prefix syntax trees represented as tree-structured embeddings.
The ERM maintains a dictionary of $K=256$ learned reduction rules $\mathcal{R} = \{ l_k \to r_k \}_{k=1}^K$, where $l_k, r_k$ are term patterns with variable wildcards.

For each subterm $t$ in the expression, the ERM executes three differentiable operations:
1. Differentiable Tree Pattern Matching: Computes a soft matching probability between subterm $t$ and rule left-hand side $l_k$ via tree kernel attention:
$$\alpha_k(t) = \text{softmax}\left( \frac{\text{TreeKernel}(t, l_k)}{\tau} \right)$$
2. Soft Term Replacement:
$$\hat{t}_{new} = (1 - \max_k \alpha_k(t)) t + \sum_k \alpha_k(t) \cdot \text{Instantiate}(r_k, \sigma_{t, l_k})$$
where $\sigma_{t, l_k}$ is the learned variable substitution mapping.
3. Differentiable Knuth-Bendix Completion Loss: To guarantee that the learned rewrite system is confluent and terminating, we enforce a critical-pair resolution loss:
$$\mathcal{L}_{KB} = \sum_{(r_i, r_j) \in \text{CritPairs}} \| \text{Reduce}(s_{ij}; \mathcal{R}) - \text{Reduce}(t_{ij}; \mathcal{R}) \|_2^2 + \lambda_{term} \mathcal{L}_{term}$$
where $(s_{ij}, t_{ij})$ are critical pairs formed by overlapping rule patterns, and $\mathcal{L}_{term}$ penalizes rules where the term size of $r_k$ exceeds $l_k$, enforcing strict termination orderings.

```
Input Expression Subterm t
         |
         v
[Tree Kernel Matching Engine] ---> Matches against Rule Bank {l_k -> r_k}
         |
         v
[Soft Term Replacement]       ---> Applies Substitution sigma: t -> r_k(sigma)
         |
         v
[Knuth-Bendix Completion Loss] ---> Penalizes Non-Confluent Critical Pairs
```

### 4. Training protocol and loss formulation
1. Data pipeline: 100 million equational reduction problems generated from axioms of semigroups, monoids, non-abelian groups, rings, and boolean algebras.
2. Supervision: Supervised initially on canonical Knuth-Bendix completion traces from computer algebra systems.
3. Fine-tuning: Jointly fine-tune on symbolic integration, algebraic simplification, and formal equality proofs in Lean 4 (`ring`, `abel`, `simp` tactic applications).

#### Architectural details and compute budget tradeoffs
- Model scale: 3B parameter transformer.
- ERM parameter footprint: 4 ERM blocks add 65M parameters total.
- Execution kernel: Batched subterm pattern matching implemented via dynamic programming on GPU Tensor Cores.
- Computational overhead: Adds 14% to training wall-clock time and 8 ms per token during generation.

### 5. Risk profile and potential failure modes
1. Critical pair explosion: Highly non-linear equational theories generate thousands of critical pairs, making exact evaluation of $\mathcal{L}_{KB}$ intractable without stochastic pair sampling.
2. Non-terminating cycles: If $\mathcal{L}_{term}$ is weighted too low, the model may learn cyclic rewrite rules that produce infinite loops during generation.
3. Expressive restriction: Restricting reductions to strictly terminating systems prevents the model from exploring temporary term expansions necessary for complex proofs.

### 6. Concrete falsification test
- Method: Differentiable Rewriting Transformer (3B) vs baseline 3B transformer fine-tuned on identical equational simplification traces.
- Setting: Equational word problems in finitely presented non-abelian groups and associative rings with word lengths from 10 to 40 terms.
- Primary metric: Exact equality verification rate and loop avoidance rate (percentage of rollouts with zero cyclic loops).
- Claim: Differentiable Rewriting Transformer achieves 100% accuracy on word problems up to length 30 and >= 92% at length 40, while baseline drops below 25% at length 15 and 5% at length 30. Loop avoidance rate exceeds 99.5%.
- Falsifier: If accuracy on length-30 word problems is < 80.0%, or if cyclic loops occur in > 5.0% of outputs, the hypothesis is falsified and abandoned.
- Pre-mortem: Reviewer 2 will assert that term rewriting cannot be made differentiable without numerical degradation. The test must show that the soft relaxation produces discrete, formally verifiable term reductions upon discretization.

---

---
## Part 2: Dynamic search, cognitive architectures, and evolutionary frontiers (cards 13 to 24)
## Hypothesis card 13: Dynamic entropy annealing via phased scratchpad temperature modulation
### 1. Title and core concept
Dynamic Entropy Annealing via Phased Scratchpad Temperature Modulation. Mathematical exploration requires divergent associative thinking during early problem formulation, followed by strictly deterministic, low-entropy deduction during terminal verification. This method introduces an adaptive, intra-trace entropy schedule that forces high token entropy in early reasoning phases and anneals down to near-zero entropy at proof completion.

### 2. Target LLM bottleneck in current models
Standard autoregressive decoding applies static sampling temperatures (for instance, constant temperature 0.6 or 0.7) across the entire generation budget. When models attempt complex multi-step proofs, static sampling either induces hallucination and arithmetic drift if temperature is maintained high near the conclusion, or locks the model into rigid, sub-optimal reasoning paths if temperature is set low from the start. Current reinforcement learning policies (such as standard PPO or GRPO) suffer from entropy collapse, where reasoning paths rapidly narrow to repetitive heuristics that fail on out-of-distribution problems.

### 3. Concrete mechanism and mathematical formulation
The scratchpad generation is structured into three explicit phases:
1. Divergent Conjecture Phase: Tokens $t \in [1, \tau_1]$. The model explores candidate lemmas, reformulations, and counter-proposals.
2. Deductive Structuring Phase: Tokens $t \in (\tau_1, \tau_2]$. The model selects the most promising path and constructs intermediate lemmas.
3. Deterministic Verification Phase: Tokens $t \in (\tau_2, T]$. The model performs strict formal deduction and final calculation.

The intra-trace temperature schedule $T(t)$ is parameterized by normalized step progress $\hat{t} = t / T_{\max}$ and dynamic confidence metrics derived from the model's hidden states:
$$T(t) = T_{\min} + (T_{\max} - T_{\min}) \cdot \left(1 - \frac{1}{1 + \exp(-\kappa (\hat{t} - \hat{t}_0))}\right) \cdot (1 - c_t)$$
where:
- $T_{\max} \in [1.2, 1.6]$ encourages high-entropy branching.
- $T_{\min} \in [0.05, 0.2]$ enforces deterministic convergence.
- $\kappa$ controls the steepness of the annealing curve.
- $\hat{t}_0$ is the inflection step.
- $c_t = \sigma(W_c h_t)$ is an internal progress estimator head predicting distance to proof completion from hidden state $h_t$.

During decoding, logits $z_{t, i}$ for token $i$ are scaled dynamically by $z'_{t, i} = z_{t, i} / T(t)$. When the progress estimator detects high deductive certainty ($c_t > 0.85$), the system forces early transition into Phase 3 regardless of token position.

### 4. Training protocol and loss formulation
The policy $\pi_\theta$ is trained via Policy Gradient with a Phase-Dependent Entropy Regularization objective:
$$\mathcal{L}(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^T \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{old}}(y_t | x, y_{<t})} A_t + \beta(t) \mathcal{H}(\pi_\theta(\cdot | x, y_{<t})) \right] + \lambda \mathcal{L}_{\text{progress}}(\phi)$$
The entropy regularization coefficient $\beta(t)$ follows an inverse schedule matching the desired target entropy:
$$\beta(t) = \beta_0 \cdot \max\left(0, \frac{\mathcal{H}_{\text{target}}(t) - \mathcal{H}(\pi_\theta(\cdot | x, y_{<t}))}{\mathcal{H}_{\text{target}}(t)}\right)$$
where $\mathcal{H}_{\text{target}}(t)$ declines monotonically from $2.5$ nats at $t=1$ to $0.1$ nats at $t=T_{\max}$.
The auxiliary progress head is trained via binary cross-entropy on ground-truth normalized distance to proof conclusion:
$$\mathcal{L}_{\text{progress}}(\phi) = \text{BCE}(c_t, 1 - \frac{T - t}{T})$$

### 5. Risk profile and potential failure modes
- Mode 1: Premature Cooling. If the progress head over-estimates completion, the temperature drops prematurely, freezing an incorrect proof sketch into a flawed deductive chain.
- Mode 2: Babbling Exploitation. High early entropy can cause policy divergence where the model outputs gibberish or semantic noise that consumes context tokens without proposing viable mathematical abstractions.
- Mode 3: Discontinuous Boundary Artifacts. Sharp temperature transitions can destabilize autoregressive KV-cache representations, resulting in syntactically broken tokens at phase boundaries.

### 6. Concrete falsification test
- Gate 1 Claim: Phased Entropy Annealing improves pass@1 on challenging competition mathematics (MATH-500 and AIME 2024) by at least 4.5 percentage points over constant-temperature sampling (T=0.7) and greedy decoding, across 3 seeds on Qwen2.5-Math-7B, at iso-compute (identical token generation budget of 4096 tokens per problem).
- Baseline: Standard Qwen2.5-Math-7B trained with uniform entropy regularization ($\beta = 0.01$) evaluated at $T=0.7$ with top-p=0.95.
- Minimal Decisive Experiment: Fine-tune Qwen2.5-Math-7B for 1000 PPO steps on NuminaMath-CoT under the phased schedule versus uniform schedule. Measure pass@1, distinct-3 n-gram entropy across early tokens, and final step syntax validity.
- Anti-Goodhart Second Metric: Trace syntactic validity rate in Lean 4 or Python REPL. The early exploration must not degrade the percentage of parsable deductive statements in Phase 3 below 98.0%.
- Kill Condition: Falsify and abandon if pass@1 gain is less than 1.5 percentage points after 1000 steps, or if syntax error rate in Phase 3 exceeds 5.0%.

---

---
## Hypothesis card 14: Lakatosian minimax counterexample co-evolution
### 1. Title and core concept
Lakatosian Minimax Counterexample Co-Evolution. Inspired by Imre Lakatos's epistemological framework in "Proofs and Refutations", mathematical progress proceeds via a continuous dialectic between conjecture, monster counterexample discovery, and lemma incorporation. This approach trains two competing models in a zero-sum minimax game: a Prover network that constructs proofs, and a Refuter network that generates boundary-case counterexamples and searches for tacit assumptions to invalidate the proof.

### 2. Target LLM bottleneck in current models
Current foundation models produce convincing mathematical proofs that contain subtle false lemmas, boundary-case violations (such as division by zero, empty set oversights, or degenerate geometric configurations), and unstated assumptions. Existing RLVR setups reward the model solely on final output correctness against static test sets. They do not penalize tacit, unstated lemmas if the final answer happens to match by coincidence.

### 3. Concrete mechanism and mathematical formulation
The system consists of two agents:
1. Prover $\mathcal{P}_\theta$: Takes problem statement $X$ and generates proof trace $Y$ containing explicit premise declarations, intermediate lemma steps, and conclusion $C$.
2. Refuter $\mathcal{R}_\phi$: Takes $(X, Y)$ and attempts to produce a refutation witness $W$. A witness $W$ is either:
   - A concrete numerical or algebraic counterexample $e \in \text{Domain}(X)$ where the theorem or any intermediate lemma fails.
   - A boundary condition where the premises break down (such as non-invertible matrix, degenerate polygon, zero determinant).

The referee environment $\mathcal{E}$ executes the witness $W$ symbolically using computer algebra systems (SymPy, Z3 SMT solver) or formal Lean 4 kernel:
$$\mathcal{V}(X, Y, W) = \begin{cases} -1 & \text{if } W \text{ is a valid counterexample to } Y \\ +1 & \text{if } Y \text{ is valid and no valid witness exists} \\ 0 & \text{if both fail or syntax errors occur} \end{cases}$$

When a counterexample is discovered, the Prover must engage in monster-barring (revising definitions to explicitly exclude degenerate cases) or lemma incorporation (modifying intermediate steps to accommodate the boundary condition).

### 4. Training protocol and loss formulation
The framework optimizes a minimax zero-sum objective:
$$\min_\theta \max_\phi \mathbb{E}_{X \sim \mathcal{D}} \left[ \mathcal{L}_{\text{game}}(\theta, \phi) \right]$$
where:
$$\mathcal{L}_{\text{game}}(\theta, \phi) = \mathbb{E}_{Y \sim \mathcal{P}_\theta(\cdot | X), W \sim \mathcal{R}_\phi(\cdot | X, Y)} \left[ \mathcal{V}(X, Y, W) \cdot \log \mathcal{P}_\theta(Y | X) - \mathcal{V}(X, Y, W) \cdot \log \mathcal{R}_\phi(W | X, Y) \right]$$
To prevent game-theoretic cycling and collapse, we incorporate an empirical game-theoretic analysis (EGTA) policy pool:
- Both $\mathcal{P}_\theta$ and $\mathcal{R}_\phi$ are trained against mixtures of historical checkpoints using fictitious play.
- An auxiliary reconstruction loss penalizes the Prover if it retreats into vacuous theorems (for example, adding trivial premises that make the theorem empty):
$$\mathcal{L}_{\text{vacuous}} = \max(0, \gamma - \mathbb{D}_{\text{KL}}(\mathcal{P}_\theta(Y | X) \parallel \mathcal{P}_{\text{base}}(Y | X)))$$

### 5. Risk profile and potential failure modes
- Mode 1: Monster-Barring Degeneracy. The Prover avoids counterexamples by adding artificial, overly restrictive hypotheses to the theorem until the result becomes trivial.
- Mode 2: Refuter Stalling. Finding mathematical counterexamples is computationally asymmetric and often harder than generating plausible-sounding proofs. If the Refuter cannot find counterexamples early in training, the Prover receives uninformative positive feedback.
- Mode 3: Adversarial Hallucination. The Refuter generates syntactically invalid counterexamples that overwhelm symbolic verification parsers.

### 6. Concrete falsification test
- Gate 1 Claim: Lakatosian Co-Evolution reduces boundary-case mathematical errors on a dedicated Adversarial Math Benchmark (comprising 300 problems with tricky edge conditions in analysis, number theory, and combinatorics) by at least 25.0% relative to standard RLVR training on DeepSeek-Math-7B, while maintaining equal accuracy on standard non-adversarial benchmarks (GSM8K, MATH).
- Baseline: DeepSeek-Math-7B trained with standard rule-based RLVR (GRPO) on identical problem prompts without the Refuter agent.
- Minimal Decisive Experiment: Train both models on 2,000 intermediate algebra and calculus problems with known edge cases for 500 gradient steps. Evaluate robustness against 50 Z3-falsifiable edge-case problems.
- Anti-Goodhart Second Metric: Theorem utility score. The average generality of proved theorems must not drop (measured by the number of active variables and premise string length; no artificial restriction of problem domains).
- Kill Condition: Falsify and abandon if the Refuter win rate drops below 5.0% after 200 training steps, indicating the Refuter has failed to produce valid counterexamples.

---

---
## Hypothesis card 15: Asynchronous dual-system dialectic (intuitive proposer + skeptical verifier)
### 1. Title and core concept
Asynchronous Dual-System Dialectic Proposer-Verifier Architecture. Human mathematical thought decouples rapid, intuitive associative hypothesis generation (System 1) from rigorous, slow, step-by-step deductive verification (System 2). This hypothesis implements a dual-stream architecture where an unconstrained speculative generator and a skeptical formal verifier communicate asynchronously over a shared blackboard during inference.

### 2. Target LLM bottleneck in current models
Current reasoning models enforce sequential autoregressive decoding where the model must act as both intuitive dreamer and rigorous proof checker at every single token. This binds verification throughput directly to token generation latency. If the model makes an intuitive leap, it either cannot check it rigorously in-flight, or it wastes hundreds of tokens walking through trivial arithmetic verifications, diluting the context window with low-information verification steps.

### 3. Concrete mechanism and mathematical formulation
The architecture decouples inference into two communicating models:
1. System 1 (Intuitive Speculator $M_{\text{fast}}$, 3B parameters): Operates at high temperature ($T=1.0$), generating speculative proof sketches, structural conjectures, and candidate lemmas at 120 tokens/second.
2. System 2 (Skeptical Verifier $M_{\text{slow}}$, 14B parameters): Operates at low temperature ($T=0.1$), consuming the speculative sketch asynchronously. It isolates individual deductive transitions, translates claims into semi-formal verification queries, and issues critique tokens into a shared bidirectional context buffer.

Communication Protocol:
The dialogue operates over an active blackboard buffer $\mathcal{B}$:
- $M_{\text{fast}}$ appends hypotheses: `[PROPOSE: lemma_k, claim, heuristic_justification]`.
- $M_{\text{slow}}$ reads $\mathcal{B}$ concurrently. For each claim, it runs symbolic verification or local deductive proof checking.
- $M_{\text{slow}}$ writes back: `[ACCEPT: lemma_k]` or `[REJECT: lemma_k, error_trace, correction_hint]`.
- Upon reading a `REJECT` token, $M_{\text{fast}}$ immediately halts its current branch, truncates its KV-cache to the pre-lemma state, and branches into an alternative proof direction conditioned on the correction hint.

### 4. Training protocol and loss formulation
The two networks are co-trained using an Asymmetric Actor-Critic Policy Gradient:
$$\mathcal{L}(\theta_{\text{fast}}, \theta_{\text{slow}}) = \mathcal{L}_{\text{propose}}(\theta_{\text{fast}}) + \alpha \mathcal{L}_{\text{verify}}(\theta_{\text{slow}})$$
where:
$$\mathcal{L}_{\text{propose}}(\theta_{\text{fast}}) = -\mathbb{E}\left[ R_{\text{verdict}} \sum_{t \in \text{props}} \log \pi_{\text{fast}}(y_t | y_{<t}, \mathcal{B}_{\text{critique}}) \right]$$
$$\mathcal{L}_{\text{verify}}(\theta_{\text{slow}}) = \mathbb{E}\left[ \text{BCE}(\hat{v}_k, v_k^*) + \sum_{t \in \text{critique}} \log \pi_{\text{slow}}(c_t | y_{\text{claim}}, \mathcal{B}) \right]$$
Here, $v_k^* \in \{0, 1\}$ is ground-truth validity of lemma $k$ confirmed by automated proof checker or terminal reward, and $R_{\text{verdict}}$ rewards accepted lemmas while applying a time penalty to rejected proposals to incentivize informative, high-precision conjectures.

### 5. Risk profile and potential failure modes
- Mode 1: Verifier Latency Bottleneck. If $M_{\text{slow}}$ cannot verify claims faster than $M_{\text{fast}}$ generates them, the proposal buffer overflows, causing $M_{\text{fast}}$ to speculate deeply on top of an unverified, erroneous foundation.
- Mode 2: Communication Collapse. $M_{\text{fast}}$ learns to output trivial, easily verified tautologies to avoid negative reward, abandoning bold, high-risk mathematical jumps.
- Mode 3: Format Drift. The asynchronous communication format degrades over long contexts, causing both models to lose track of lemma naming conventions.

### 6. Concrete falsification test
- Gate 1 Claim: The Asynchronous Dual-System architecture (3B Proposer + 14B Verifier) solves at least 20.0% more problems on MiniF2F-Lean4 and Putnam-level problems than a monolithic 14B model operating under standard chain-of-thought, at equal total inference FLOP budget.
- Baseline: Monolithic Qwen2.5-Math-14B generating standard single-stream reasoning traces at $T=0.7$.
- Minimal Decisive Experiment: Evaluate on 200 MiniF2F formal problems with a wall-clock cap of 120 seconds per problem. Compare solved proof count and total tokens spent on invalid dead ends.
- Anti-Goodhart Second Metric: Branch efficiency ratio. Ratio of accepted proof steps to total proposed steps must exceed 0.40 (preventing brute-force unguided spamming of conjectures).
- Kill Condition: Falsify and abandon if the dual-system architecture fails to outperform the monolithic baseline by at least 3.0 percentage points on MiniF2F after 300 training steps.

---

---
## Hypothesis card 16: Automated lemma distillation and concept abstraction during RL
### 1. Title and core concept
Automated Lemma Distillation and Concept Formation during Reinforcement Learning. Human mathematical capability expands through abstraction: once a complex subproof is established, it is named, modularized into a reusable lemma, and invoked as an atomic premise in subsequent deductions. This method introduces an automated Minimum Description Length (MDL) subroutine during RL training that detects recurring proof sub-graphs across problem solutions, abstracts them into named parameterized lemmas, and expands the model's active reasoning dictionary.

### 2. Target LLM bottleneck in current models
Current language models re-derive basic lemmas from first principles repeatedly across different problems. A 7B parameter model tasked with proving twenty distinct geometry theorems will re-prove the law of cosines or triangle congruence conditions inline in every trace. This wastes context window capacity, increases the cumulative probability of algebraic error, and limits the effective reasoning depth to whatever can fit within a single scratchpad context.

### 3. Concrete mechanism and mathematical formulation
The distillation mechanism operates as a two-level loop:
1. Online Reasoning: Model proves mathematical theorems using its current repertoire of named lemmas $\mathcal{L}_{\text{active}}$.
2. Offline Concept Formation (MDL Consolidation): Every $N$ training iterations, the system aggregates all successful proof traces $\mathcal{T} = \{\tau_1, \ldots, \tau_M\}$.
3. Trace Graph Parsing: Each trace $\tau$ is parsed into a directed acyclic graph (DAG) of logical inferences $G = (V, E)$, where vertices are mathematical propositions and edges represent deduction steps.
4. Subgraph Mining: A grammar compression algorithm (adapted from Sequitur and graph-mining algorithms) identifies frequent isomorphic subgraphs:
   $$g^* = \arg\max_g \left( \text{Freq}(g) \cdot |g| - \text{Complexity}(g) \right)$$
5. Lemma Generalization: The recurring subgraph $g^*$ is abstracted by replacing specific constants with typed free variables. The generalized lemma is verified by Lean 4 or an automated theorem prover.
6. Lexicon Injection: Verified lemmas are assigned unique token identifiers or added to a persistent dynamic prompt module $\mathcal{D}_{\text{lemmas}}$ accessible via specialized attention keys.

### 4. Training protocol and loss formulation
The policy is trained with an explicit compression-augmented reward:
$$R_{\text{total}}(\tau) = R_{\text{correct}}(\tau) + \gamma_{\text{MDL}} \cdot \Delta \text{Length}(\tau)$$
where $\Delta \text{Length}(\tau) = |\tau_{\text{uncompressed}}| - |\tau_{\text{distilled}}|$ measures the token savings achieved by invoking crystallized lemmas rather than inlining subproofs.
The policy loss includes an invocation loss enforcing valid lemma call arguments:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{RL}}(\theta) + \mu \sum_{k \in \text{calls}} \log \pi_\theta(\text{Args}_k | \text{Lemma}_k, \text{Context})$$
If the model invokes a lemma with invalid types or unsatisfied preconditions, an immediate severe penalty (-2.0) is assigned.

### 5. Risk profile and potential failure modes
- Mode 1: Premature Abstraction (Over-fitting). Distilling highly specialized, overly narrow subproofs that are only applicable to one problem variant, cluttering the lemma dictionary with useless definitions.
- Mode 2: Premise Mismatch Hallucination. The model learns to call distilled lemmas by name but hallucinates that the current proof state satisfies the lemma's preconditions.
- Mode 3: Graph Parsing Overhead. Parsing thousands of natural language math traces into formal DAGs can be computationally expensive or inaccurate if informal natural language transitions are ambiguous.

### 6. Concrete falsification test
- Gate 1 Claim: Automated Lemma Distillation reduces average solution token length by at least 35.0% while improving pass@1 by at least 5.0 percentage points on high-depth multi-step math benchmarks (such as ProofNet or Olympiad geometry) compared to standard RLVR on DeepSeek-Math-Base-7B.
- Baseline: DeepSeek-Math-Base-7B trained with standard GRPO on the same training set without lemma distillation.
- Minimal Decisive Experiment: Train for 500 steps on 5,000 synthetic multi-step algebra and Euclidean geometry problems. Measure token length distribution, lemma invocation frequency, and overall solve rate.
- Anti-Goodhart Second Metric: Zero-shot lemma re-use factor. Distilled lemmas must be invoked correctly across at least 3 distinct problem clusters, proving they are genuine abstractions rather than memorized problem-specific chunks.
- Kill Condition: Falsify and abandon if fewer than 5.0% of generated proofs successfully invoke distilled lemmas, or if lemma premise failure rate exceeds 15.0%.

---

---
## Hypothesis card 17: Autotelic curriculum entropy via self-generated conjectures
### 1. Title and core concept
Autotelic Curriculum Entropy via Self-Generated Open-Ended Conjectures. Rather than relying on human-annotated competition problem datasets, the model drives its own learning curriculum by formulating new mathematical conjectures at the boundary of its current proving capability. Conjectures that are neither trivially provable nor demonstrably unprovable maximize learning progress, creating an open-ended self-improving discovery loop.

### 2. Target LLM bottleneck in current models
Existing mathematical training corpora are static and saturated. Models quickly memorize standard Olympiad problem formats (AIME, USAMO, IMO). As a consequence, reinforcement learning suffers from diminishing returns: once a model solves the accessible portion of the static dataset, the remaining unsolved problems provide sparse, uninformative zero-reward signals. The model cannot manufacture intermediate stepping-stone problems that bridge the gap to higher reasoning capabilities.

### 3. Concrete mechanism and mathematical formulation
The learning loop executes three continuous phases:
1. Conjecturing Phase: The model $\mathcal{M}_\theta$ receives an axiomatic domain specification (for instance, Peano arithmetic, group theory, or linear algebra) and generates candidate propositions $P \sim \mathcal{M}_\theta(\cdot | \text{Axioms})$.
2. Triviality and Consistency Filtering:
   - Proposition $P$ is checked by a fast automated solver (Z3 / Vampire) with a 2-second timeout. If solved immediately, it is discarded as trivial.
   - Proposition $P$ is tested against random counterexample instantiation. If refuted immediately, it is discarded as false.
3. Intrinsic Difficulty Targeting: Surviving propositions are tackled by the Prover policy $\mathcal{M}_\theta$ using budget-constrained search ($K=16$ attempts).
   - Empirical solve rate $p(P) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(\text{solved}_k)$ is computed.
   - Intrinsic reward is assigned using an informational entropy metric that peaks at the edge of capability:
     $$R_{\text{curriculum}}(P) = -p(P) \log_2 p(P) - (1 - p(P)) \log_2 (1 - p(P))$$
   - Conjectures with $p(P) \approx 0.5$ (where the model succeeds half the time) receive maximum curriculum weight.
4. Retention and Buffer Update: Hard but solvable conjectures are added to the training replay buffer, accompanied by their verified proof traces.

### 4. Training protocol and loss formulation
The dual generator-prover policy is trained via alternating gradient steps:
$$\mathcal{L}_{\text{conjecture}}(\theta) = -\mathbb{E}_{P \sim \mathcal{M}_\theta} \left[ R_{\text{curriculum}}(P) \cdot \log \mathcal{M}_\theta(P | \text{Domain}) \right]$$
$$\mathcal{L}_{\text{prover}}(\theta) = -\mathbb{E}_{P \sim \mathcal{D}_{\text{active}}, \tau \sim \mathcal{M}_\theta(\cdot | P)} \left[ \mathbb{I}(\text{valid}(\tau)) \cdot \log \mathcal{M}_\theta(\tau | P) \right]$$
To prevent semantic drift where the conjecturer generates syntactically convoluted or trivial variants of the same statement, we apply an empirical diversity penalty based on the pairwise Levenshtein and tree edit distance of statement ASTs:
$$\mathcal{L}_{\text{diversity}} = \max\left(0, \delta - \text{Dist}(AST(P_i), AST(P_j))\right)$$

### 5. Risk profile and potential failure modes
- Mode 1: Tautology Cloaking. The conjecturer finds statements that appear structurally complex but reduce to basic algebraic identities ($x + y = y + x$), fooling the complexity heuristic while remaining mathematically uninformative.
- Mode 2: Unprovable Statement Traps. The model generates undecidable or unprovable conjectures (analogues of the Collatz conjecture or Gödelian sentences) that consume massive proof search budgets without yielding valid learning gradients.
- Mode 3: Domain Narrowing. The model converges on an isolated algebraic sub-niche (such as iterating identities in modular arithmetic) where it can reliably manufacture $p \approx 0.5$ problems, completely neglecting analysis, geometry, and combinatorics.

### 6. Concrete falsification test
- Gate 1 Claim: Training on self-generated autotelic conjectures improves out-of-domain mathematical generalization on novel unseen formal benchmarks (MiniF2F novel subset) by at least 8.0 percentage points over training on a static synthetic dataset of equivalent token volume, using Mistral-7B as base.
- Baseline: Mistral-7B fine-tuned with identical compute on 100,000 static synthetic problems generated via traditional rejection sampling.
- Minimal Decisive Experiment: Run the autotelic loop for 50,000 self-generated problems across group theory and elementary number theory. Compare zero-shot transfer on 100 out-of-distribution formal theorems.
- Anti-Goodhart Second Metric: Axiomatic diversity measure. The generated conjectures must span at least 10 distinct mathematical definitions and operators, verified by AST node distribution entropy $\mathcal{H}_{\text{AST}} \ge 2.8$ bits.
- Kill Condition: Falsify and abandon if more than 60.0% of self-generated conjectures are detected as syntactic variants of existing identities by automated term-rewriting systems, or if proof success rate on held-out human benchmarks fails to improve by at least 2.0 percentage points.

---

---
## Hypothesis card 18: Speculative multi-branch scratchpad with branch-and-bound rollback
### 1. Title and core concept
Speculative Multi-Branch Scratchpad with Branch-and-Bound Rollback. Standard language models decode linear text streams. If an erroneous assumption is generated at step 2, all subsequent steps compound the error. This method equips the reasoning scratchpad with tree-structured speculative execution primitives: the model can spawn multiple exploratory branches, evaluate intermediate premise consistency, explicitly prune failing paths with `<prune>` tokens, and restore the KV-cache to a previous valid checkpoint via an explicit `<rollback_to_k>` token.

### 2. Target LLM bottleneck in current models
Autoregressive language models suffer from the sunk-cost fallacy of linear context. Once a model writes out several lines of flawed algebra, the self-attention mechanism attends heavily to its own incorrect intermediate tokens. Even if the model notices the error, it frequently hallucinates a rationalization or doubles down on the error rather than erasing the mistake. External tree search methods (like MCTS) solve this outside the model, but introduce massive serving latency, complex scheduling, and inability to train the search dynamics end-to-end.

### 3. Concrete mechanism and mathematical formulation
The vocabulary is extended with four dedicated meta-reasoning control tokens:
- `<branch_split: n>`: Instructs the inference engine to fork the current KV-cache into $n$ parallel candidate reasoning threads.
- `<checkpoint: id>`: Tags the current hidden state and KV-cache boundary with an addressable identifier.
- `<evaluate_branch>`: Emits an internal consistency score $s \in [-1, 1]$ computed from a lightweight value head.
- `<rollback: id>`: Discards all KV-cache states generated since checkpoint `id` and forces subsequent generation to proceed from that historical boundary, injecting a negative constraint token that masks the previously chosen invalid branch.

Algorithmic Decoding Protocol:
1. When encountering high predictive uncertainty (entropy $\mathcal{H} > \tau_{\text{branch}}$), the model emits `<checkpoint: A>` followed by `<branch_split: 3>`.
2. Three independent sub-chains generate intermediate reasoning tokens up to fixed length $L_{\text{spec}}$.
3. At the branch conclusion, each thread emits `<evaluate_branch>`. An integrated value head evaluates the validity of the derivation.
4. The winning thread continues. The losing threads are pruned.
5. If all threads yield negative value ($s < 0$), the model emits `<rollback: A>`, resets attention buffers, and explores a completely different lemma direction.

### 4. Training protocol and loss formulation
Training is conducted on synthesized branch-and-rollback trajectories generated from automated theorem proving search trees. The loss combines token language modeling with an explicit rollback policy gradient:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{LM}}(\theta) + \lambda_{\text{tree}} \mathcal{L}_{\text{tree}}(\theta)$$
where:
$$\mathcal{L}_{\text{tree}}(\theta) = -\sum_{k \in \text{decisions}} \left[ Q(s_k, a_k) \log \pi_\theta(a_k | s_k) \right]$$
Here, action $a_k \in \{\text{continue}, \text{split}, \text{prune}, \text{rollback}\}$ and $Q(s_k, a_k)$ is the Monte Carlo return of the final proof.
A heavy latency penalty is imposed on excessive rollbacks to prevent infinite looping:
$$\mathcal{R}_{\text{efficiency}} = -\gamma \cdot N_{\text{rollbacks}}^2$$

### 5. Risk profile and potential failure modes
- Mode 1: Rollback Thrashing. The model gets stuck in an infinite rollback cycle, repeatedly attempting slightly different variations of an impossible step until the generation token limit is exhausted.
- Mode 2: Value Head Over-optimism. The integrated branch evaluation head gives false positive scores to superficially plausible but mathematically broken branches, pruning valid paths.
- Mode 3: Hardware KV-Cache Fragmentation. Non-linear tree branching and dynamic rollback severely complicate PagedAttention and tensor-parallel memory layouts, reducing GPU serving throughput.

### 6. Concrete falsification test
- Gate 1 Claim: In-scratchpad Branch-and-Bound Rollback improves pass@1 on MATH Level 5 problems by at least 6.0 percentage points compared to standard linear chain-of-thought, while matching the accuracy of external MCTS (width=4, depth=3) at 2.5x lower serving latency.
- Baseline: Qwen2.5-Math-7B running standard linear CoT and Qwen2.5-Math-7B wrapped in an external MCTS loop.
- Minimal Decisive Experiment: Evaluate on 300 hardest problems from AMC12 and AIME. Measure solve rate, wall-clock time per solved problem, and average number of explored reasoning states.
- Anti-Goodhart Second Metric: Rollback recovery yield. At least 30.0% of invoked `<rollback>` events must lead to an ultimately correct solution (verifying that rollback is acting as a productive error-recovery mechanism rather than a desperate failure symptom).
- Kill Condition: Falsify and abandon if rollback recovery yield is below 10.0%, or if inference latency exceeds external MCTS by more than 10.0%.

---

---
## Hypothesis card 19: Simplicial complex and hypergraph proof-space navigation
### 1. Title and core concept
Simplicial Complex and Hypergraph Proof-Space Navigation. Formal proof search is fundamentally non-Euclidean: multiple premises jointly combine via multi-ary inference rules to produce new propositions, forming a directed hypergraph or simplicial complex rather than a simple sequence or graph. This approach replaces standard flat embedding retrieval with a topological representation where proof states are modeled as simplices in an abstract simplicial complex, and premise selection is formulated as simplicial boundary navigation.

### 2. Target LLM bottleneck in current models
In autoformalization and formal theorem proving (Lean 4, Isabelle), language models struggle with premise selection. Standard dense retrieval systems compute cosine similarities between the current proof goal and candidate library lemmas in a Euclidean vector space. However, mathematical deduction is combinatorial: lemma A and lemma B might be useless individually (low individual cosine similarity to goal), but their joint intersection produces the exact bridge required. Flat vector retrieval fails to model these high-order multi-way interactions.

### 3. Concrete mechanism and mathematical formulation
We formulate the proof state as an abstract simplicial complex $\mathcal{K} = (V, \Sigma)$:
- 0-simplices (vertices $V$): Elementary mathematical propositions, hypotheses, and goals.
- $k$-simplices ($\sigma \in \Sigma$): A set of $k+1$ propositions that are simultaneously bound by a valid $k$-ary mathematical inference rule or lemma.

The navigation model consists of a Simplicial Neural Network (SNN) operating over the topological representations:
1. Goal Representation: The current open goal $g$ is projected onto the co-chain complex of $\mathcal{K}$.
2. Simplicial Attention: Given active hypotheses $\{h_1, \ldots, h_m\}$, the model computes Hodge-Laplacian operators $\mathcal{L}_k = \mathcal{B}_{k+1} \mathcal{B}_{k+1}^* + \mathcal{B}_k^* \mathcal{B}_k$ to capture both boundary and co-boundary flows across the proof topology.
3. Candidate Premise Scoring: Rather than scoring candidate lemma $L$ independently, the model scores the candidate simplex $\sigma = \{h_{i_1}, \ldots, h_{i_k}, L, g\}$ using the topological persistence of the complex:
   $$\text{Score}(L | \{h\}, g) = \text{softmax}\left( \text{Tr}\left( \Phi(L)^T \mathcal{L}_k \Psi(\{h\}, g) \right) \right)$$
This directly captures whether adding lemma $L$ closes a topological hole (a non-trivial cycle) between the hypotheses and the target theorem.

### 4. Training protocol and loss formulation
The topological premise selector is trained using a contrastive hypergraph ranking loss:
$$\mathcal{L}(\theta) = -\sum_{\sigma^+ \in \mathcal{K}^*} \log \frac{\exp(\text{Score}(\sigma^+))}{\exp(\text{Score}(\sigma^+)) + \sum_{j=1}^M \exp(\text{Score}(\sigma_j^-))}$$
where $\sigma^+$ represents the true combination of premises that completes a proof step in Mathlib, and $\sigma_j^-$ are negative premise combinations sampled from the same topological neighborhood.
An auxiliary topological regularization loss encourages the learned embedding space to preserve persistent homology Betti numbers of the underlying theorem dependency graph:
$$\mathcal{L}_{\text{topo}} = \sum_{p=0}^2 \|\beta_p(\mathcal{K}_{\text{predicted}}) - \beta_p(\mathcal{K}_{\text{ground\_truth}})\|_2^2$$

### 5. Risk profile and potential failure modes
- Mode 1: Combinatorial Explosion. The number of candidate simplices grows exponentially with the number of active premises ($O(m^k)$), requiring aggressive beam filtering that could discard valid multi-premise combinations.
- Mode 2: High Computational Overhead of Hodge-Laplacians. Calculating discrete exterior calculus operations and boundary matrices during inference can introduce severe latency bottlenecks on large formal libraries like Lean 4's Mathlib (over 100,000 theorems).
- Mode 3: Sparse Topological Supervision. Many proof steps in formal libraries are straightforward syntactic rewrites that do not exhibit interesting higher-order simplicial structure, leading to gradient starvation on topological features.

### 6. Concrete falsification test
- Gate 1 Claim: Simplicial Hypergraph Premise Selection achieves at least a 12.0 percentage point improvement in Recall@10 on multi-premise theorems in Lean 4 Mathlib compared to state-of-the-art dense retriever baselines (such as ByT5 or standard text-embedding-3-large), at iso-candidate budget.
- Baseline: Standard bi-encoder dense retrieval model fine-tuned on Mathlib premise selection pairs.
- Minimal Decisive Experiment: Evaluate on 1,000 held-out Lean 4 theorems that strictly require $\ge 3$ simultaneous premises. Measure Recall@5, Recall@10, and mean reciprocal rank.
- Anti-Goodhart Second Metric: Multi-premise joint validity. The top-10 retrieved sets must contain complete premise subsets capable of discharging the goal in at least 25.0% of cases, rather than individual disjoint lemmas.
- Kill Condition: Falsify and abandon if the Simplicial SNN fails to outperform the dense retriever baseline by at least 4.0 percentage points in Recall@10 after training on the full Mathlib corpus, or if inference latency exceeds 200ms per retrieval step.

---

---
## Hypothesis card 20: Negative trace distillation from productive dead ends
### 1. Title and core concept
Negative Trace Distillation from Productive Dead Ends. Mathematicians do not discover proofs along a straight path. They explore blind alleys, encounter contradictions, recognize why the approach failed, and pivot. Standard model training discards all failed reasoning traces, training solely on successful solutions. This hypothesis develops a training paradigm that explicitly distills learning signals from productive dead ends: teaching the model to identify the exact token where an assumption broke down, emit a retrospective error token, and learn negative avoidance policies.

### 2. Target LLM bottleneck in current models
Current RLVR paradigms (such as DeepSeek-R1-Zero, standard PPO/GRPO) assign zero reward to failed traces. If a model generates a brilliant 40-step mathematical exploration that fails only on step 39 due to an arithmetic slip, the entire 40-step trajectory is penalized equally with a trace that hallucinates nonsensical gibberish on step 1. This creates an enormous credit assignment problem, discards the richest cognitive data in mathematical reasoning, and leaves models vulnerable to repeating the exact same dead-end reasoning patterns repeatedly.

### 3. Concrete mechanism and mathematical formulation
The framework captures and structures paired reasoning trajectories:
1. Failed Exploration Generation: During sampling, traces that terminate in a verified dead end or contradiction are retained: $\tau_{\text{fail}} = (x_1, \dots, x_k, x_{k+1}, \dots, x_T)$, where step $x_k$ is the critical divergence point (the false pivot).
2. Automated Pivot Localization: A step-level verifier or retrospective counterfactual search identifies the precise pivot step $x_k$ that doomed the proof.
3. Corrective Recovery Branch: From state $s_{k-1}$, the model is prompted or guided to discover the correct pivot $x_k^*$ that leads to successful completion $\tau_{\text{success}} = (x_1, \dots, x_{k-1}, x_k^*, \dots, x_{T'}^*)$.
4. Paired Contrastive Construction: A synthetic meta-trace is constructed containing:
   `[EXPLORE: x_k, ...]` $\to$ `[CONTRADICTION DETECTED: explanation]` $\to$ `[PIVOT BACK TO s_{k-1}]` $\to$ `[EXECUTE: x_k^*, ...]`.

### 4. Training protocol and loss formulation
The policy is trained with a combined Direct Preference Optimization (DPO) and Unlikelihood Objective:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{DPO}}(\theta) + \alpha \mathcal{L}_{\text{unlike}}(\theta) + \beta \mathcal{L}_{\text{pivot}}(\theta)$$
where:
$$\mathcal{L}_{\text{unlike}}(\theta) = -\sum_{t=k}^{k+L} \log \left( 1 - \pi_\theta(x_t | x_{<t}) \right)$$
penalizes the exact sequence of dead-end tokens conditioned on the historical context $x_{<k}$, and:
$$\mathcal{L}_{\text{pivot}}(\theta) = -\log \pi_\theta(\text{pivot\_token} | x_1, \dots, x_k, \text{error\_witness})$$
trains the model to explicitly emit self-correction signals when an approach is dead.
In reinforcement learning, traces that contain self-corrected dead ends that successfully reach the correct answer receive a bonus relative to straight-line lucky guesses:
$$R(\tau) = R_{\text{correct}} + \eta \cdot \mathbb{I}(\text{self\_corrected})$$

### 5. Risk profile and potential failure modes
- Mode 1: Self-Correction Feigning (Reward Hacking). The model intentionally inserts fake, trivial dead ends into its scratchpad (for example, writing $1+1=3$, then correcting itself to $1+1=2$) solely to collect the self-correction bonus $\eta$.
- Mode 2: Aversion Paralysis. Over-penalizing dead-end tokens can cause the model to become overly cautious, collapsing into repetitive, ultra-short proofs or refusing to explore non-trivial mathematical territory.
- Mode 3: Incorrect Pivot Attribution. If the automated localization assigns the error to step $x_k$ when the actual flaw was a subtle premise oversight at step $x_2$, the model learns to suppress valid mathematical steps.

### 6. Concrete falsification test
- Gate 1 Claim: Training on Productive Dead Ends improves pass@1 on complex Olympiad-level multi-step problems (AIME and OlympiadBench) by at least 5.5 percentage points over standard positive-only RLVR on Qwen2.5-Math-7B, while cutting repetitive looping behavior by $\ge 40.0\%$.
- Baseline: Qwen2.5-Math-7B trained with standard GRPO on verified correct mathematical solutions only.
- Minimal Decisive Experiment: Train both models on 10,000 problems for 800 gradient steps. Test on 200 unseen problems where the initial reasoning step is intentionally perturbed with a subtle false premise.
- Anti-Goodhart Second Metric: Unforced error correction rate. The model must demonstrate an unprompted error recovery rate $\ge 35.0\%$ when injected with intermediate flawed deductions, without degrading straight-line proof speed.
- Kill Condition: Falsify and abandon if the model demonstrates fake self-correction behavior (inserting trivial errors) on more than 5.0% of evaluation traces, or if the pass@1 improvement over the baseline is less than 1.5 percentage points.

---

---
## Hypothesis card 21: Sleep-phase offline proof consolidation and refactoring
### 1. Title and core concept
Sleep-Phase Offline Proof Consolidation and Refactoring. Human mathematicians spend hours exploring messy, rambling computations, followed by periods of offline reflection and sleep during which memories are consolidated, irrelevant steps are forgotten, and elegant, minimal proof paths are crystallized. This method introduces a dual-phase training lifecycle: an online high-entropy exploration phase (Wake), followed by an offline refactoring phase (Sleep) where successful but bloated traces are rewritten into minimal canonical forms, verified, and re-ingested into model weights.

### 2. Target LLM bottleneck in current models
Reinforcement learning models that generate long chains of thought (such as R1-style models) suffer from reasoning bloat. Over training, traces expand from 1,000 tokens to 8,000+ tokens. Much of this length consists of linguistic wandering, repetitive self-doubt, circular recalculations, and redundant checking. While this wandering helps discovery, training directly on these raw, bloated traces via standard SFT or RL clutters the model's memory, consumes massive KV-cache bandwidth, and causes catastrophic semantic drift.

### 3. Concrete mechanism and mathematical formulation
The architecture alternates between two synchronized phases:
1. Wake Phase (Online Exploration):
   - Model generates high-entropy exploratory traces across difficult problems.
   - Traces that reach the correct, verified solution are saved into an episodic buffer $\mathcal{B}_{\text{raw}}$, regardless of length, repetition, or wandering.
2. Sleep Phase (Offline Consolidation and Refactoring):
   - The episodic buffer $\mathcal{B}_{\text{raw}}$ is processed by an offline Neuro-Symbolic Refactoring Pipeline.
   - Semantic Slicing: Dependency analysis removes all dead-end excursions and unreferenced scratch variables.
   - Proof Minimization: A refactoring agent $\mathcal{M}_{\text{refactor}}$ compresses the valid core into an elegant, publication-grade proof that preserves all essential mathematical ideas while minimizing Kolmogorov/token complexity.
   - Equivalence Verification: The refactored proof $\tau^*$ is re-verified by formal kernel or symbolic engine to guarantee that no logical leaps were introduced during compression.
   - Memory Consolidation Fine-Tuning: The original model $\mathcal{M}_\theta$ is trained via distillation on the refactored corpus $\mathcal{B}_{\text{clean}}$ with a replay loss on foundational mathematics to prevent catastrophic forgetting.

### 4. Training protocol and loss formulation
The Sleep Consolidation phase optimizes a joint distillation and length-penalized likelihood objective:
$$\mathcal{L}_{\text{sleep}}(\theta) = -\sum_{(x, \tau^*) \in \mathcal{B}_{\text{clean}}} \log \pi_\theta(\tau^* | x) + \lambda_{\text{replay}} \mathcal{L}_{\text{replay}}(\theta, \mathcal{D}_{\text{core}})$$
where $\tau^*$ satisfies:
$$\tau^* = \arg\min_{\tau \in \text{Equiv}(\tau_{\text{raw}})} |\tau| \quad \text{subject to } \mathcal{V}(x, \tau) = \text{Valid}$$
In the subsequent Wake phase, the policy begins exploration from the consolidated weights, maintaining an entropy anchor to prevent premature collapse:
$$\mathcal{L}_{\text{wake}}(\theta) = \mathcal{L}_{\text{RLVR}}(\theta) + \beta \mathbb{D}_{\text{KL}}(\pi_\theta(\cdot | x) \parallel \pi_{\text{sleep}}(\cdot | x))$$

### 5. Risk profile and potential failure modes
- Mode 1: Over-Compression and Proof Gaps. The refactoring pipeline deletes intermediate steps that it deems redundant, but which were cognitively necessary for the model to span a difficult deductive leap.
- Mode 2: Loss of Exploratory Plasticity. After several sleep cycles, the model becomes too rigid, memorizing compressed templates and losing the chaotic associative flexibility needed to tackle radically new problems in the next wake phase.
- Mode 3: Refactoring Hallucination. The refactoring model introduces subtle mathematical fallacies during compression that pass informal checks but corrupt the training buffer.

### 6. Concrete falsification test
- Gate 1 Claim: Wake-Sleep Consolidation cuts average inference reasoning token length by at least 45.0% while improving pass@1 on MATH-500 by at least 4.0 percentage points compared to uncompressed long-trace RLVR training on Llama-3.1-8B, after 5 wake-sleep cycles.
- Baseline: Llama-3.1-8B trained continuously on uncompressed raw RLVR traces for an equal number of total gradient updates and environment interactions.
- Minimal Decisive Experiment: Run 3 complete Wake-Sleep cycles on 5,000 competition problems. Compare evaluation solve rates, average solution token length, and perplexity on out-of-distribution math textbooks.
- Anti-Goodhart Second Metric: Deductive completeness score. An independent formal proof checker must verify that compressed proofs contain zero unstated inferential gaps ($\ge 95.0\%$ formal step validity).
- Kill Condition: Falsify and abandon if the model's exploratory trace diversity in subsequent Wake phases drops by more than 50.0% (indicating loss of exploratory capability), or if compressed proofs exhibit an unverified gap rate exceeding 5.0%.

---

---
## Hypothesis card 22: Socratic dialectical multi-agent RL for collaborative theorem discovery
### 1. Title and core concept
Socratic Dialectical Multi-Agent Reinforcement Learning for Collaborative Theorem Discovery. Mathematical breakthroughs in human research rarely occur in cognitive isolation; they emerge from dialectical discourse between mathematicians playing distinct cognitive roles. This method formulates mathematical discovery as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) played by three specialized agents: an Exploratory Proposer, a Skeptical Critic, and an Axiomatic Formalizer, trained via multi-agent reinforcement learning with shared theorem discovery rewards.

### 2. Target LLM bottleneck in current models
Single-agent reasoning architectures suffer from self-reinforcing cognitive blindspots. When a single model generates both the ideas and the critiques, it is prone to confirmation bias: it consistently overlooks its own tacit assumptions, misinterprets ambiguous notations, and fails to challenge its chosen line of attack. External multi-agent debate wrappers exist, but they are typically un-tuned prompt-based ensembles that waste tokens on polite conversational filler without learned, game-theoretic role specialization.

### 3. Concrete mechanism and mathematical formulation
The system consists of three agents with learned functional specializations:
1. Proposer $\mathcal{A}_{\text{prop}}$: Trained to generate bold mathematical analogies, candidate invariants, structural conjectures, and heuristic sketches.
2. Critic $\mathcal{A}_{\text{crit}}$: Trained to attack proposals, construct minimal counterexamples, expose implicit assumptions, and demand rigorous lemma justifications.
3. Formalizer $\mathcal{A}_{\text{form}}$: Trained to translate natural language conjectures and critiques into strict formal specifications (Lean 4 / Isabelle) and execute them against the proof kernel.

Interaction Protocol:
- Round $t$: Proposer emits sketch $S_t$.
- Critic analyzes $S_t$, emitting critique $C_t = (\text{flaws}, \text{unproved\_lemmas})$.
- Formalizer attempts to formalize the surviving valid claims into Lean 4 code $F_t$.
- The environment executes $F_t$. If the formal proof succeeds, all three agents receive a large shared positive reward. If Lean returns an error, the error state is fed back into the dialogue history.

### 4. Training protocol and loss formulation
The agents are trained jointly via Multi-Agent Proximal Policy Optimization (MAPPO) with a centralized critic but decentralized actor execution:
$$\mathcal{L}(\theta_i) = -\mathbb{E}_{\vec{\tau}} \left[ \min\left(r_t(\theta_i) A_i(s, \vec{a}), \text{clip}(r_t(\theta_i), 1-\epsilon, 1+\epsilon) A_i(s, \vec{a})\right) \right] + \lambda \mathcal{L}_{\text{comm}}(\theta_i)$$
Credit assignment is determined using Counterfactual Multi-Agent (COMA) policy gradients:
$$A_i(s, \vec{a}) = Q(s, \vec{a}) - \sum_{a'_i} \pi_i(a'_i | \tau_i) Q(s, (\vec{a}_{-i}, a'_i))$$
This directly measures whether agent $i$'s specific contribution improved the theorem outcome compared to a marginal baseline action.
A communication efficiency regularizer penalizes verbose or redundant utterances:
$$\mathcal{L}_{\text{comm}}(\theta_i) = \gamma \cdot \text{Length}(\text{message}_i)$$

### 5. Risk profile and potential failure modes
- Mode 1: Sycophantic Consensus. Critic and Proposer learn to collude by agreeing on simple, trivial mathematical facts that are easily formalizable, collecting small positive rewards while avoiding genuinely difficult open problems.
- Mode 2: Babbling Degeneracy. Under multi-agent RL, the communication tokens between agents can drift away from natural language and standard mathematical notation into an uninterpretable, high-entropy private code that fails to transfer to human-readable proofs.
- Mode 3: Credit Assignment Starvation. If the formalizer agent fails to formalize a correct high-level informal idea generated by the proposer, all agents receive zero reward, unfairly penalizing the proposer.

### 6. Concrete falsification test
- Gate 1 Claim: Socratic Dialectical Multi-Agent RL achieves at least a 15.0 percentage point higher formalization success rate on the Putnam benchmark and MiniF2F than a monolithic base model operating with equivalent total parameter count and token compute budget.
- Baseline: Monolithic 32B model (Qwen2.5-32B) evaluated under standard chain-of-thought and self-consistency (sampling 8 paths).
- Minimal Decisive Experiment: Deploy three 7B agents (Proposer, Critic, Formalizer) against the 32B baseline on 150 complex multi-step theorems. Cap total combined tokens at 16,000 per theorem.
- Anti-Goodhart Second Metric: Formal translation fidelity. At least 70.0% of informal lemmas generated during the dialogue must be syntactically valid and non-vacuous when parsed by the Lean 4 environment.
- Kill Condition: Falsify and abandon if the multi-agent system produces higher conversational token overhead without improving solved proof count by at least 3.0 percentage points over the monolithic baseline.

---

---
## Hypothesis card 23: Evolutionary proof-space code search with semantic mutation operators
### 1. Title and core concept
Evolutionary Proof-Space Code Search with Semantic Mutation Operators. Proof search in formal systems (Lean 4, Coq, Isabelle) is a rugged, discontinuous, non-convex optimization surface where gradient descent on token probabilities frequently gets trapped in barren plateaus. This approach treats formal tactic scripts as executable genetic programs, using domain-aware semantic mutation and crossover operators informed by policy network logits to navigate proof space via Quality-Diversity evolutionary algorithms (MAP-Elites).

### 2. Target LLM bottleneck in current models
Language models generating formal proofs (autoformalization or tactic generation) sample tokens left-to-right using temperature-scaled probabilities. However, mathematical proof development is non-linear: a mathematician might know the beginning (setup) and the end (target reduction), but lack the middle bridge; or they might have an inductive step that works if only the base lemma is generalized slightly. Autoregressive sampling cannot perform global structural operations like swapping induction variables, inserting case splits mid-proof, or abstracting a concrete term into an existential witness.

### 3. Concrete mechanism and mathematical formulation
The algorithm maintains an evolving population $\mathcal{P}$ of candidate formal proof scripts in Lean 4:
1. Quality-Diversity Feature Space (MAP-Elites):
   - The archive is indexed by two behavioral descriptors: Proof Length (structural complexity) and Lemma Novelty (distance of used lemmas from standard library defaults).
2. Domain-Aware Semantic Mutation Operators:
   - Mutation 1: Lemma Generalization. Replaces a concrete variable $x$ in an intermediate tactic with an arbitrary expression or generalizes the inductive hypothesis.
   - Mutation 2: Structural Induction Inversion. Swaps the primary induction variable or switches from structural induction to well-founded strong induction.
   - Mutation 3: Middle-Out Splice (Crossover). Takes the forward premise derivation from Proof A and joins it with the backward goal reduction from Proof B via an automated unification tactic (`nlinarith`, `aesop`, or `ring`).
   - Mutation 4: Tactic Sequence Perturbation. A policy language model $\pi_\theta$ proposes targeted replacement tactics for failing lines, using the Lean compiler's error diagnostics as conditioning context.
3. Fitness Evaluation:
   - Each mutant is compiled in Lean 4.
   - Fitness $f(\tau) = \text{GoalsClosed}(\tau) + \frac{1}{\text{CompileErrors}(\tau) + 1} - \alpha \cdot \text{ComputeTime}(\tau)$.
   - Completely successful proofs receive maximal fitness and are archived.

### 4. Training protocol and loss formulation
The policy model $\pi_\theta$ acting as the evolutionary mutation generator is trained via offline RL on the evolutionary trajectories discovered by the MAP-Elites engine:
$$\mathcal{L}(\theta) = -\sum_{(\text{Parent}, \text{Error}, \text{Mutant}^+) \in \mathcal{D}_{\text{elites}}} \log \pi_\theta(\text{Mutant}^+ | \text{Parent}, \text{Error})$$
This continuously aligns the neural policy to propose mutations that have an empirically high probability of resolving specific Lean compiler errors.
A diversity-maximization loss ensures the mutation operator explores unconventional tactic combinations:
$$\mathcal{L}_{\text{div}}(\theta) = -\sum_{i \neq j} \mathbb{D}_{\text{KL}}(\pi_\theta(\cdot | s_i) \parallel \pi_\theta(\cdot | s_j))$$

### 5. Risk profile and potential failure modes
- Mode 1: Compilation Latency Collapse. Lean 4 compiler startup and elaboration times (typically 50ms to 500ms per attempt) can severely throttle evolutionary throughput, requiring heavy distributed caching and persistent server worker pools.
- Mode 2: Macro Degeneracy. The evolutionary algorithm exploits automated crushing tactics (like spamming `simp; aesop; linarith`) that easily solve shallow goals but fail to build deep conceptual scaffolding for hard theorems.
- Mode 3: Syntactic Invalidation. Blind crossover between two different proof scripts almost always produces un-compilable code unless constrained by strict semantic type-checking at recombination boundaries.

### 6. Concrete falsification test
- Gate 1 Claim: Evolutionary Proof-Space Code Search with Semantic Mutation resolves at least 18.0% of previously unsolved formal theorems in MiniF2F-test and ProofNet that resist standard beam search and temperature sampling (pass@64) with DeepSeek-Prover-V1.5, at iso-compute (matching total Lean environment interactions).
- Baseline: DeepSeek-Prover-V1.5 running standard Monte Carlo Tree Search and pass@64 sampling.
- Minimal Decisive Experiment: Run on 100 benchmark theorems from MiniF2F that failed under standard pass@64 sampling. Grant both methods 1,000 compilation queries per theorem. Measure percentage of closed proofs and average proof AST depth.
- Anti-Goodhart Second Metric: Structural diversity index. The generated proofs must use distinct proof strategies (for example, direct proof vs induction vs contradiction) across different problems rather than overfitting to single-tactic brute force.
- Kill Condition: Falsify and abandon if the evolutionary search fails to solve at least 5.0% of the stubborn baseline-failing theorems within 1,000 compilation steps per problem.

---

---
## Hypothesis card 24: Metamorphic invariant-preserving proof perturbation regularization
### 1. Title and core concept
Metamorphic Invariant-Preserving Proof Perturbation and Equivalence Regularization. In mathematics, truth is invariant under structure-preserving isomorphisms: renaming bound variables, translating coordinates, reflecting geometric configurations, changing algebraic bases, or permuting independent subgoals does not alter the truth value or structural validity of a proof. This hypothesis introduces a rigorous metamorphic training objective that enforces representation and trajectory invariance across mathematically isomorphic transformations of problem statements.

### 2. Target LLM bottleneck in current models
Current foundation models are notoriously sensitive to superficial token perturbations. Renaming variable $x$ to $\xi$, swapping the order of two commutative equations, or stating a problem in the contrapositive frequently causes model performance to drop precipitously (often by 15% to 30% on GSM8K and MATH). Models memorize token co-occurrence patterns rather than internalizing the abstract, coordinate-free mathematical structures.

### 3. Concrete mechanism and mathematical formulation
A Metamorphic Transformation Engine $\mathcal{T}_{\text{iso}}$ applies structure-preserving functors to input problems:
1. Isomorphic Operators:
   - Permutation Operator $\mathcal{P}_{\text{var}}$: Systematically permutes all variable names while preserving typing ($x, y \to u, v$).
   - Commutative Goal Reordering $\mathcal{P}_{\text{goals}}$: Swaps the presentation order of mutually independent subgoals.
   - Dual Transformation $\mathcal{P}_{\text{dual}}$: Transforms a geometric problem into its Cartesian algebraic coordinate dual, or an optimization problem into its Lagrangian dual.
   - Affine / Symmetry Transformations $\mathcal{P}_{\text{sym}}$: Rotates or translates coordinates in analytic geometry problems without changing relational properties.

2. Equivalence Enforcement Architecture:
Given original problem $X$ and transformed isomorphic problem $X' = \mathcal{T}(X)$, the model generates representations $H = f_\theta(X)$ and $H' = f_\theta(X')$.
The internal representation space is constrained such that the topological trajectory of the reasoning states in latent space preserves the morphism:
$$d_{\text{Morphism}}(z_t, z'_t) = \|W_{\text{proj}} h_t - \mathcal{T}_*(W_{\text{proj}} h'_t)\|_2^2$$
where $\mathcal{T}_*$ is the canonical vector space representation of the transformation.

### 4. Training protocol and loss formulation
The objective function augments standard token likelihood with a Metamorphic Consistency Regularization Loss:
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \lambda_{\text{meta}} \mathcal{L}_{\text{consistency}}(\theta) + \mu \mathcal{L}_{\text{outcome}}(\theta)$$
where:
$$\mathcal{L}_{\text{consistency}}(\theta) = \mathbb{E}_{X, X'} \left[ \frac{1}{T} \sum_{t=1}^T \mathbb{D}_{\text{KL}}\left( \pi_\theta(\cdot | X, y_{<t}) \parallel \mathcal{T}^{-1} \pi_\theta(\cdot | X', \mathcal{T}(y)_{<t}) \right) \right]$$
$$\mathcal{L}_{\text{outcome}}(\theta) = |\text{Score}(X) - \text{Score}(X')|$$
If the model solves $X$ but fails on isomorphic variant $X'$, a maximal violation penalty is backpropagated through both the encoder and decoder attention layers.

### 5. Risk profile and potential failure modes
- Mode 1: Representation Smearing. Forcing the model to produce invariant hidden states across distinct token representations can dilute the attention mechanism's ability to track precise variable bindings, causing syntax errors in generation.
- Mode 2: Non-Isomorphic Leakage. If the automated transformation engine $\mathcal{T}_{\text{iso}}$ contains subtle bugs (such as accidentally changing a strict inequality to non-strict during translation), it forces the model to treat mathematically distinct problems as equivalent, corrupting the policy.
- Mode 3: Compute Multiplier. Training on multiple metamorphic variants of every problem multiplies forward-backward FLOP requirements by $2\times$ to $4\times$ per batch.

### 6. Concrete falsification test
- Gate 1 Claim: Metamorphic Invariant Regularization reduces the performance gap between canonical problems and their isomorphic perturbations from 18.5% down to $\le 3.0\%$ on the Perturbed-MATH benchmark, while improving overall unperturbed pass@1 on MATH by $\ge 3.5$ percentage points across 3 seeds on Qwen2.5-Math-7B.
- Baseline: Qwen2.5-Math-7B trained with standard data augmentation (simply adding perturbed text to training mix without the consistency loss).
- Minimal Decisive Experiment: Train for 600 steps on 5,000 algebra and calculus problems with their metamorphic pairs. Test on 500 isomorphic pairs from MATH-500. Measure pass@1 parity and latent distance invariance.
- Anti-Goodhart Second Metric: Cross-isomorphism proof agreement. The model must produce logically equivalent proof conclusions across at least 95.0% of isomorphic pairs.
- Kill Condition: Falsify and abandon if metamorphic consistency loss fails to reduce the perturbation performance gap below 8.0%, or if unperturbed MATH accuracy degrades by more than 1.0 percentage point.

---

---
## 3. Cross-cutting synthesis and unified architectural interaction
The 24 radical hypotheses are not disconnected point solutions. They form a unified system where continuous geometric priors, embedded symbolic execution kernels, adversarial dialectics, and cognitive consolidation interact across the problem-solving lifecycle:

```
                          [Problem Formulation]
                                    |
                                    v
                 +--------------------------------------+
                 |  H17: Autotelic Conjecture Curriculum|
                 |  H24: Metamorphic Invariant Modeling |
                 +------------------+-------------------+
                                    |
                                    v
                     [Divergent Exploration Phase]
                                    |
       +----------------------------+----------------------------+
       v                            v                            v
+--------------+             +--------------+             +--------------+
|  H13: Phased |             |  H15: Dual-  |             |  H18: Multi- |
|    Entropy   |             |    System    |             |     Branch   |
|   Annealing  |             |   Dialectic  |             |    Rollback  |
+------+-+-----+             +------+-+-----+             +------+-+-----+
       | |                          | |                          | |
       | +--------------------------+-+--------------------------+ |
       +----------------------------+----------------------------+
                                    |
                                    v
                     [Latent Geometric Trajectory Planning]
                                    |
       +----------------------------+----------------------------+
       v                                                         v
+------------------------------+          +------------------------------+
|  H01: Geodesic Flow Planning |          |  H05: Bidirectional Latent   |
|       on Proof Manifolds     |          |       Contractive Search     |
+--------------+---------------+          +--------------+---------------+
               |                                         |
               +--------------------+--------------------+
                                    |
                                    v
                     [Intra-Layer Verification and Co-Execution]
                                    |
       +----------------------------+----------------------------+
       v                                                         v
+------------------------------+          +------------------------------+
|  H02: SRAM Differentiable    |          |  H12: Differentiable Knuth-  |
|       Gröbner / SAT Kernels  |          |       Bendix Term Rewriting  |
+--------------+---------------+          +--------------+---------------+
               |                                         |
               +--------------------+--------------------+
                                    |
                                    v
                     [Adversarial and Topological Search]
                                    |
       +----------------------------+----------------------------+
       v                                                         v
+------------------------------+          +------------------------------+
|  H14: Lakatosian Minimax     |          |  H19: Simplicial Complex     |
|      Counterexample Search   |          |      Proof Navigation        |
+--------------+---------------+          +--------------+---------------+
               |                                         |
               +--------------------+--------------------+
                                    |
                                    v
                     [Collaborative and Evolutionary Search]
                                    |
       +----------------------------+----------------------------+
       v                                                         v
+------------------------------+          +------------------------------+
|  H22: Socratic Multi-Agent   |          |  H23: Evolutionary Proof     |
|       Dialectical RL         |          |       Code Search (Elites)   |
+--------------+---------------+          +--------------+---------------+
               |                                         |
               +--------------------+--------------------+
                                    |
                                    v
                     [Consolidation and Abstraction]
                                    |
       +----------------------------+----------------------------+
       v                                                         v
+------------------------------+          +------------------------------+
|  H16: Automated Lemma        |          |  H20: Negative Trace         |
|      Distillation (MDL)      |          |      Distillation (Pivots)   |
+--------------+---------------+          +--------------+---------------+
               |                                         |
               +--------------------+--------------------+
                                    |
                                    v
                 +--------------------------------------+
                 |  H21: Sleep-Phase Offline Refactoring|
                 |       and Memory Consolidation       |
                 +------------------+-------------------+
                                    |
                                    v
                        [Disciplined Final Proof]
```

### Core composability synergies
1. The Discovery-Consolidation Loop (H13 + H20 + H21):
   H13 initiates high-entropy exploration to break through heuristic plateaus. H20 extracts training value from dead-end branches by optimizing error-pivot tokens. H21 refactors sprawling scratchpads offline into minimal canonical proofs, preventing context bloat and updating policy weights cleanly.
2. Continuous-Discrete Feedback Loop (H01 + H02 + H10):
   The geodesic flow planner (H01) charts macro-trajectories across the continuous proposition manifold, avoiding broad regions of unprovability. As waypoints are decoded, intra-layer symbolic co-execution kernels (H02) and monoidal category attention (H10) verify local algebraic consistency, projecting states back onto the valid algebraic manifold.
3. Adversarial and Dialectical Grounding (H14 + H15 + H24):
   Speculative proofs are challenged by an active refuter seeking counterexamples (H14). Fast-proposer and slow-verifier networks communicate asynchronously via shared key-value cache (H15). Metamorphic regularization (H24) prevents agents from being fooled by trivial coordinate shifts or variable re-labelings.
4. Open-Ended Mathematical Exploration (H16 + H17 + H23):
   Autotelic curriculum generation (H17) synthesizes hard mathematical conjectures at the capability boundary. Evolutionary search (H23) overcomes rugged non-convex proof spaces where autoregression stalls. Lemma distillation (H16) extracts recurrent proof motifs, adding verified definitions into Mathlib to expand the model's permanent knowledge base.
## 4. Hardware feasibility and implementation on modern GPU clusters
Deploying these 24 radical architectures on modern GPU clusters (such as NVIDIA HGX H100 with NVLink 4.0 and 900 GB/s inter-GPU bandwidth) introduces concrete hardware requirements:

1. SRAM residency for symbolic and term-rewriting kernels (H02, H12):
   Moving polynomial coefficients or term dictionary tables between HBM3 and Tensor Cores stalls the transformer pipeline. By restricting polynomial degrees and term dictionaries to fit inside the 96 KB SM shared memory (L1/SRAM) of H100 streaming multiprocessors, symbolic kernels execute at register speeds with under 15% latency penalty.
2. Lorentz and hyperbolic numerical precision (H07, H11):
   Standard float16 produces precision loss near the light-cone boundary or Poincaré ball horizon. Custom Triton kernels compute Lorentzian distances in float32 accumulators while activations are stored in bfloat16, clamping metric denominators to $\epsilon = 10^{-6}$.
3. Asynchronous dual-clock and dual-system pipelining (H08, H15):
   Dual-clock architectures map directly to pipeline-parallel clusters. The lightweight macro-planner or intuitive proposer resides on a dedicated GPU, while micro-executors or verifiers run data-parallel across worker nodes, communicating only compact latent vectors over NVLink at structural phase boundaries.
4. Persistent homology and simplicial complex indexing (H03, H19):
   Betti number computation and higher-order simplex indexing are offloaded to host memory threads or dedicated CPU workers, caching simplicial boundary matrices in pinned system RAM to avoid stalling GPU CUDA streams.
## 5. 30-day phased pilot elimination roadmap
To prevent unbounded compute expenditure, testing these 24 radical hypotheses follows a disciplined 30-day phased elimination protocol governed by the ML Research Rigor framework:

```
Phase 0: Baseline reproduction (Days 1 to 3, Gates 1 to 3)
  - Reproduce published benchmarks for DeepSeek-Math-7B, Qwen2.5-Math-7B, and DeepSeek-Prover
  - Verify baseline reproduction bands (+/- 0.25 ppl, +/- 1.5% pass@1)
                 |
                 v
Phase 1: Conceptual gating and minimal decisive pilots (Days 4 to 10, Gate 4)
  - 2-GPU-day small-scale runs (300M to 1B parameter models, synthetic datasets)
  - Evaluate decisive falsification metrics against pre-registered thresholds
  - AUTOMATIC KILL: Any hypothesis failing its minimal threshold is shelved to research/deadends/
                 |
                 v
Phase 2: Intermediate scale validation (Days 11 to 20, Gates 5 to 8)
  - 3B to 7B parameter models on 16x H100 GPUs (10B to 30B tokens)
  - 3 random seeds; report mean +/- standard deviation
  - Anti-Goodhart evaluation against secondary metrics
                 |
                 v
Phase 3: Scaling, formal integration, and synthesis (Days 21 to 30, Gates 9 to 11)
  - Multi-node cluster testing on Lean 4 / Isabelle formal verification suites
  - Integration of surviving hypotheses into unified 14B foundation model
```

### Pilot budget allocation and elimination schedule

| Phase / Batch | Hypotheses included | Model scale | Hardware allocation | Falsification kill condition |
|---|---|---|---|---|
| **Batch A: Structural Priors** | H04 (Equivariant), H10 (Monoidal), H12 (Term Rewriting), H24 (Metamorphic) | 1B params, 10B tokens | 8x H100 for 3 days (576 GPU-hrs) | Zero sample efficiency gain or < 70% transitivity accuracy |
| **Batch B: Neuro-Symbolic** | H02 (SMT Co-Exec), H09 (Compiler IR), H14 (Lakatosian Minimax) | 3B params, 15B tokens | 8x H100 for 4 days (768 GPU-hrs) | Latency overhead > 35% or refuter win rate < 5% |
| **Batch C: Search & Dynamics** | H13 (Entropy Annealing), H18 (Rollback), H20 (Negative Distillation) | 7B params, 20B tokens | 16x H100 for 5 days (1,920 GPU-hrs) | Pass@1 gain < 1.5% or fake self-corrections > 5% |
| **Batch D: Geometric & Latent** | H01 (Geodesic Flow), H05 (Bidirectional), H07 (Curvature), H11 (Hyperbolic) | 7B params, 25B tokens | 16x H100 for 6 days (2,300 GPU-hrs) | Pass@1 gain < 3.0% or continuation divergence > 50% |
| **Batch E: Cognitive & Abstraction** | H15 (Dual-System), H16 (Lemma Distillation), H17 (Autotelic), H21 (Sleep Consolidation) | 7B params, 25B tokens | 16x H100 for 6 days (2,300 GPU-hrs) | Lemma invocation < 5% or formal gap rate > 15% |
| **Batch F: Multi-Agent & Genetic** | H03 (Modular MoE), H19 (Simplicial), H22 (Socratic RL), H23 (Evolutionary Search) | 7B params, 20B tokens | 16x H100 for 5 days (1,920 GPU-hrs) | Multi-premise validity < 25% or conversation bloat |

Total pilot compute expenditure: Approximately 9,784 H100 GPU-hours across 30 days. This structured protocol guarantees that capital is concentrated strictly on hypotheses demonstrating verified, statistically significant deductive gains.
