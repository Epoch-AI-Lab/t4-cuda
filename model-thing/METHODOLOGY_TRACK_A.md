# Standard training methodology for mathematical foundation models

## 1. Executive summary and pipeline architecture

Mathematical reasoning models rely on a disciplined training stack spanning specialized pre-training, synthetic data synthesis, rejection sampling fine-tuning, process-supervised reward modeling, reinforcement learning with verifiable rewards, and formal verification. This report documents the mechanisms, mathematical formulations, sample complexities, failure dynamics, and open-source codebases defining established practice across frontier mathematical language models.

Frontier architectures such as DeepSeek-Math, DeepSeek-R1, Qwen2.5-Math, NuminaMath, and WizardMath establish several concrete operational findings:
First, outcome-only supervision reinforces spurious reasoning paths that happen to yield correct answers. Process-level verification or rule-based multi-turn interaction is necessary to suppress hallucinated intermediate steps.
Second, Group Relative Policy Optimization (GRPO) replaces the standard PPO value network with empirical group-normalized advantages, cutting GPU memory overhead by approximately 50 percent while stabilizing reinforcement learning over multi-thousand-token derivations.
Third, long-horizon reinforcement learning against deterministic answer verifiers induces spontaneous reflection, backtracking, and algorithmic double-checking without requiring human-annotated chain-of-thought data.
Fourth, formal verification environments such as Lean 4 and Isabelle/HOL eliminate hallucination by enforcing kernel-level proof checks, but introduce engineering bottlenecks around premise retrieval, elaboration latency, and autoformalization gaps.

```
+---------------------------------------------------------------------------------------------------+
|                                  Standard training pipeline topology                              |
+---------------------------------------------------------------------------------------------------+
|  1. Pre-training and continual pre-training                                                       |
|     - Web math filtering (OpenWebMath, MathWeb, arXiv, Common Crawl math shards)                  |
|     - Deduplication (MinHash LSH 8-gram, exact 13-gram decontamination)                           |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  2. Synthetic trace synthesis and augmentation                                                    |
|     - Backward reasoning (target-to-premise parameterization)                                    |
|     - Question evolution (Evol-Instruct, WizardMath, NuminaMath translation and OCR)              |
|     - Tool-integrated reasoning (ToRA, Python REPL, SymPy execution trace embedding)              |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  3. Rejection sampling fine-tuning (RFT and Best-of-N)                                            |
|     - Policy sampling at temperature T in [0.6, 1.0]                                              |
|     - Deterministic outcome filtering (SymPy, regex, ground truth match)                          |
|     - SFT cross-entropy loss over positive candidate pool                                         |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  4. Process-supervised reward models (PRMs)                                                       |
|     - Step boundary delimiter scoring (\n\n) via token classification heads                       |
|     - Automated attribution via Monte Carlo rollouts (Math-Shepherd)                              |
|     - Step-level credit assignment and active learning uncertainty sampling                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  5. Reinforcement learning with verifiable rewards (RLVR)                                         |
|     - Deterministic oracles (SymPy, Python sandbox, Lean kernel)                                  |
|     - GRPO group relative advantage estimation without critic networks                            |
|     - Emergence of long-horizon self-correction, backtracking, and length expansion               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  6. Formal theorem proving and autoformalization                                                  |
|     - Lean 4 and Isabelle/HOL interactive theorem proving                                         |
|     - Lake environment JSON-RPC REPL interaction (LeanDojo, LeanCopilot)                          |
|     - Premise selection via dense retrieval and Monte Carlo tree search proof search              |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Data curation and synthetic step-by-step trace generation

High-performance mathematical reasoning models require high-density pre-training and supervised fine-tuning corpora. Raw web text yields low signal-to-noise ratios, inconsistent notation, and prevalent arithmetic mistakes. Frontier pipelines therefore combine specialized web extraction with targeted synthetic generation.

### 2.1 Synthetic data generation pipelines

Synthetic generation targets four distinct objectives: diversifying problem structures, expanding reasoning paths, injecting self-correction behaviors, and embedding deterministic execution steps.

#### Backward reasoning: Target-to-premise generation
Standard forward generation prompts a model to produce a solution $y$ given a question $x$. In complex mathematical domains, forward generation frequently fails due to search space explosion. Backward reasoning inverts the problem generation process:
1. Sample a structured answer or mathematical object $a \in \mathcal{A}$, such as an integer polynomial root, an eigenvalue, or a geometric configuration.
2. Sample an algebraic operation or transformation sequence $T_1, T_2, \dots, T_k$ operating on $a$.
3. Apply the transformations in reverse to synthesize problem statement $x = T_k(\dots T_1(a)\dots)$.
4. Construct the step-by-step forward solution by verifying that the forward derivations $T_1^{-1}, \dots, T_k^{-1}$ invert the transformation back to $a$.

For example, when constructing Diophantine equations or integer factorizations, selecting prime factors $p, q$ first and setting $N = pq$ guarantees valid target solutions, preventing the generation of unsolvable or degenerate problems. In symbolic calculus, differentiating an elementary target function $F(x)$ produces an integrand $f(x) = F'(x)$, guaranteeing that the integral $\int f(x) \, dx$ has a closed-form solution.

#### Question evolution: Evol-Instruct, WizardMath, and NuminaMath
The Evol-Instruct methodology, developed in WizardLM and extended to mathematics in WizardMath (Luo et al., 2023), systematically upgrades simple problem instances into competition-grade problems through structured prompt mutations:
- In-depth evolution:
  1. Add constraints: introduce boundary conditions, integer restrictions, or modular arithmetic bounds to an existing problem.
  2. Deepen domain: transform a single-variable algebraic problem into a multivariable optimization problem.
  3. Increase reasoning steps: replace direct evaluation with nested functional compositions.
  4. Concretize abstract problems: map an abstract group theory property into an explicit matrix multiplication exercise.
- In-breadth evolution:
  Mutate problem contexts, change entity relationships, or map geometric problems into coordinate geometry equivalents to maintain topical diversity.

NuminaMath (Beeching et al., 2024), winner of the first AI Math Olympiad (AIMO) Progress Prize, extended this methodology across 860,000 mathematical problem-solution pairs. The NuminaMath pipeline collected problems from Chinese high-school competitions, Russian Olympiads, French Baccalauréat exams, and US AMC/AIME competitions. Text was normalized via OCR pipelines using Nougat and Surya, translated to English via DeepL and open translation models, and augmented by generating multiple diverse reasoning paths per problem using Claude 3.5 Sonnet and GPT-4o.

#### Self-correction and reflection traces
Standard chain-of-thought traces exhibit monotonic forward progression. However, human problem solving involves dead-end identification and backtracking. To teach models to self-correct during inference, synthetic pipelines inject explicit error detection and correction sequences into the training data:
1. Generate an initial reasoning trajectory $y^{(0)} = (s_1, s_2, \dots, s_k)$ using an unconstrained base model.
2. Identify a flawed step $s_j$ where a computational or logical error occurred.
3. Append an explicit reflection prompt: "Wait, the step above assumes $x > 0$, but the problem specifies $x \in \mathbb{R}$. Let me reconsider this branch."
4. Generate an alternate, correct trajectory from step $s_j$ to the verified solution.
5. Wrap the intermediate critique and revision inside structured XML delimiters, such as `<think> ... </think>`.

Training models on these synthetic revision trajectories conditions the policy to emit reconsideration tokens when probability mass drops on an unpromising branch.

#### Execution trace augmentation: Tool-integrated reasoning and ToRA
Pure natural language chain-of-thought fails on long arithmetic computations, high-degree polynomial root finding, and large matrix operations. Tool-Integrated Reasoning (TIR), implemented in ToRA (Gou et al., 2023) and Qwen2.5-Math (Yang et al., 2024), interleaves natural language deduction with executable code blocks:

```
Step 1 (Deduction):
We need to find the roots of the cubic polynomial f(x) = x^3 - 7x^2 + 14x - 8.
Let us use SymPy to compute the roots symbolically.

Step 2 (Code execution):
```python
import sympy as sp
x = sp.Symbol('x')
f = x**3 - 7*x**2 + 14*x - 8
roots = sp.roots(f)
print(roots)
```
```output
{1: 1, 2: 1, 4: 1}
```

Step 3 (Conclusion):
The roots are x = 1, x = 2, and x = 4. Their sum is 1 + 2 + 4 = 7.
```

To build these datasets at scale, generation agents execute Python scripts in isolated sandboxes, capture stdout and stderr, and format successful runs as multi-turn dialogues where the Python interpreter functions as an environment observation. If the script throws a runtime exception, the agent receives the traceback and generates an in-context bug fix, teaching the model self-debugging capabilities.

---

### 2.2 Decontamination protocols and quality filtering

Pre-training and fine-tuning corpora risk severe contamination against standard evaluation benchmarks (GSM8K, MATH, OlympiadBench, PutnamBench, MiniF2F). Training on leaked evaluation questions invalidates empirical results.

#### Decontamination algorithms
Frontier laboratories enforce two-tier decontamination pipelines:
1. Surface-level n-gram filtering:
   Extract all word-level or subword-level 13-grams from target evaluation sets. Scan the training corpus using a rolling hash. Any training document sharing a single exact 13-gram with an evaluation problem statement is flagged. DeepSeek-Math and GPT-4 use 13-gram matching with case normalization, whitespace removal, and LaTeX symbol stripping.
2. MinHash Locality-Sensitive Hashing (MinHash LSH):
   For partial rephrasing and numerical variations, problems are tokenized into character $k$-shingles (typically $k=8$). Compute $m$ hash functions (typically $m=128$ or $m=256$) to generate signature vectors:
   $$h_j(D) = \min_{s \in D} \text{hash}_j(s)$$
   Compute Jaccard similarity across LSH buckets:
   $$J(D_1, D_2) = \frac{|D_1 \cap D_2|}{|D_1 \cup D_2|}$$
   Documents with $J(D_1, D_{\text{eval}}) \ge 0.7$ are purged from the training set.

```
+------------------------------------------------------------------------------------+
|                         Decontamination pipeline architecture                      |
+------------------------------------------------------------------------------------+
| Raw corpus (web, synthetic)                                                        |
+------------------------------------------------------------------------------------+
           |
           v
+------------------------------------------------------------------------------------+
| Normalization: strip LaTeX spaces, unify punctuation, lower-case, remove comments  |
+------------------------------------------------------------------------------------+
           |
           +---------------------------------+
           |                                 |
           v                                 v
+-----------------------+       +----------------------------------------------------+
| Exact 13-gram index   |       | MinHash LSH index (k=8 shingles, 128 hash funcs)   |
+-----------------------+       +----------------------------------------------------+
           |                                 |
           | Match found                     | Jaccard similarity >= 0.70
           |                                 |
           v                                 v
+------------------------------------------------------------------------------------+
| Discard sample (mark contaminated in data ledger)                                  |
+------------------------------------------------------------------------------------+
```

#### Synthetic math quality filtering
Large synthetic pools contain degenerate, ungrounded, or mathematically incorrect samples. Data engines apply multi-stage heuristic and model-based filters:
- Perplexity and loss filtering:
  Score candidate synthetic traces under an independent reference model (such as a frozen base LLM). Discard samples where average token loss is in the top 5 percent (indicating gibberish or token loops) or the bottom 1 percent (indicating repetitive trivia or boilerplate).
- Execution verification filter:
  For code-integrated and symbolic derivations, execute the embedded code in a Python 3.11 sandbox with SymPy, NumPy, and SciPy pre-loaded. If the execution fails, exceeds a 5-second CPU timeout, or produces an output inconsistent with the final boxed answer (`\boxed{...}`), the trace is discarded.
- Step validity and self-consistency:
  Sample $K=16$ completions from an independent solver model conditioned on the generated question. If the modal answer among the completions disagrees with the synthetic trace's stated answer, or if the pass rate is 0 across all 16 attempts, the question is flagged as ambiguous or flawed.
- Deduplication:
  Compute dense embeddings using text embedding models (such as BGE-M3 or sentence-transformers). Perform hierarchical clustering or k-means clustering across problem embeddings. Retain only representatives within dense cosine clusters ($\cos(\theta) > 0.92$) to prevent over-representing standard high-school problem templates.

---

## 3. Rejection sampling fine-tuning (RFT) and best-of-N

Rejection Sampling Fine-Tuning (RFT), also known as Supervised Fine-Tuning on filtered rollouts, bridges the gap between base language models and reasoning models without requiring complex policy gradient optimization.

```
+----------------------------------------------------------------------------------------+
|                                  RFT execution workflow                                |
+----------------------------------------------------------------------------------------+
| Training prompts x ~ D_train                                                           |
+----------------------------------------------------------------------------------------+
           |
           v
+----------------------------------------------------------------------------------------+
| Sample N independent completions: y_1, y_2, ..., y_N ~ pi_old( . | x; T in [0.6, 1.0]) |
+----------------------------------------------------------------------------------------+
           |
           v
+----------------------------------------------------------------------------------------+
| Deterministic verification: V(y_i, y_true) in {0, 1}                                   |
+----------------------------------------------------------------------------------------+
           |
           +---------------------------------+
           |                                 |
           v (V = 1)                         v (V = 0)
+-----------------------+       +--------------------------------------------------------+
| Retain positive trace |       | Discard trace (or store for DPO negative sampling)     |
+-----------------------+       +--------------------------------------------------------+
           |
           v
+----------------------------------------------------------------------------------------+
| Supervised fine-tuning update: maximize sum log pi_theta(y_pos | x)                     |
+----------------------------------------------------------------------------------------+
```

### 3.1 Mathematical mechanics of rejection sampling fine-tuning

Let $\mathcal{D} = \{(x_i, y_i^*)\}_{i=1}^{|\mathcal{D}|}$ represent a dataset of mathematical problems $x_i$ paired with ground-truth final answers $y_i^*$. Let $\pi_{\theta_{\text{base}}}$ represent the initial policy.

1. Trajectory generation:
   For each problem $x_i$, generate $N$ independent candidate completions:
   $$y_{i,1}, y_{i,2}, \dots, y_{i,N} \sim \pi_{\theta_{\text{base}}}(\cdot \mid x_i; T)$$
   where $T \in [0.6, 1.0]$ is the sampling temperature.
2. Verification and filtering:
   Evaluate each completion against the ground truth answer using a deterministic verification function:
   $$V(y_{i,j}, y_i^*) = \begin{cases} 1 & \text{if } \text{ExtractAnswer}(y_{i,j}) \equiv y_i^* \\ 0 & \text{otherwise} \end{cases}$$
3. Dataset compilation:
   Construct the positive training pool:
   $$\mathcal{D}_{\text{RFT}} = \{(x_i, y_{i,j}) \mid V(y_{i,j}, y_i^*) = 1\}$$
4. Policy update:
   Optimize the policy parameters $\theta$ using standard cross-entropy loss over the filtered positive tokens:
   $$\mathcal{L}_{\text{RFT}}(\theta) = - \sum_{(x, y) \in \mathcal{D}_{\text{RFT}}} \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$

#### Sample complexity and pass-rate dependence
Let $p(x) = P_{y \sim \pi(\cdot \mid x; T)}(V(y, y^*) = 1)$ denote the single-sample success probability on problem $x$. The probability of obtaining at least one correct trajectory across $N$ independent samples is:
$$P(\text{success} \ge 1 \mid x) = 1 - (1 - p(x))^N$$

To guarantee a success probability of at least $1 - \delta$ (for example, $1 - \delta = 0.99$), the required sample budget $N$ satisfies:
$$(1 - p(x))^N \le \delta \implies N \ge \frac{\ln \delta}{\ln(1 - p(x))}$$

Using the linear approximation $\ln(1 - p(x)) \approx -p(x)$ for small $p(x)$:
$$N \approx \frac{\ln(1/\delta)}{p(x)}$$

This inverse dependence on $p(x)$ reveals the computational boundary of RFT:
- Easy problems ($p(x) = 0.5$): $N \approx \frac{4.605}{0.5} \approx 9$ samples.
- Medium problems ($p(x) = 0.05$): $N \approx \frac{4.605}{0.05} \approx 92$ samples.
- Hard competition problems ($p(x) = 0.001$): $N \approx \frac{4.605}{0.001} \approx 4605$ samples.

When $p(x) \to 0$, RFT fails because generating even a single positive trace requires tens of thousands of forward passes.

#### Temperature scaling tradeoffs
The choice of temperature $T$ during rejection sampling governs a fundamental trade-off between solution diversity and per-sample accuracy:
- Low temperature ($T \to 0$):
  The model samples near the greedy mode. Output paths exhibit low variance. While the pass rate on accessible problems is high, the model samples identical derivations repeatedly. The effective size of $\mathcal{D}_{\text{RFT}}$ collapses, offering minimal exploratory value.
- High temperature ($T \ge 1.0$):
  The policy explores distant regions of the token probability simplex. On complex multi-step problems, token entropy accumulates over long sequences. The per-sample pass rate $p(x)$ drops exponentially with sequence length $L$, as $p(x) \sim \prod_{t=1}^L P(\text{correct token}_t)$. The model wastes compute generating invalid completions.
Empirical practice sets $T \in [0.6, 0.8]$ with top-$p$ nucleus sampling ($p = 0.9$ to $0.95$).

---

### 3.2 Failure dynamics and systemic limitations

While RFT provides large initial performance gains, it exhibits systemic pathologies:

#### Mode collapse to low-entropy strategies
When multiple correct derivations exist for problem $x$ (for example, an algebraic manipulation approach compared to a geometric construction approach), the policy $\pi_{\theta_{\text{base}}}$ naturally assigns higher likelihood to shorter, simpler paths. RFT preferentially samples and reinforces these easy paths. Over successive epochs, the policy distribution collapses into narrow coordinate subspaces, suppressing alternative solution strategies.

#### Distribution shift between sampling and greedy inference
During sampling, completions are generated with $T > 0$. However, during benchmark evaluation, models are evaluated under greedy decoding ($T = 0$) or low-temperature self-consistency. High-temperature positive traces frequently contain sub-optimal token sequences, conversational filler, or inefficient detours that survived filtering solely because the final answer was correct. Training on these sequences shifts the greedy policy towards verbose, wandering reasoning paths.

#### Diminishing returns in iterated rounds
In Self-Taught Reasoner (STaR, Zelikman et al., 2022) and iterative RFT pipelines, the policy $\pi^{(k)}$ trained on round $k$ is used to generate samples for round $k+1$. Empirical evaluations in DeepSeek-Math and Qwen2.5-Math demonstrate that performance gains plateau sharply after two to three iterations.
The reason for this saturation is structural: the model cannot solve problems outside its latent capability boundary without external knowledge injection. The problems remaining unsolved after round 2 have true policy success rates $p(x) \approx 0$. No finite sample budget $N$ generated by the model itself will yield positive traces for these problems.

#### Reinforcement of false positive reasoning
A vulnerability of outcome-based filtering is the "right answer for the wrong reason" pathology. A model can execute invalid algebraic cancellations:
$$\frac{16}{64} = \frac{1}{4} \quad \text{(incorrectly canceling the 6s)}$$
The final answer $1/4$ matches ground truth. The verifier flags $V(y) = 1$. The model is then updated via cross-entropy to maximize the probability of this logically flawed derivation. In advanced competition problems, algebraic errors often cancel downstream arithmetic mistakes. Outcome-based RFT rewards these compensating errors.

---

## 4. Process-supervised reward models (PRMs)

To resolve the credit assignment failures of outcome-only filtering, Process-Supervised Reward Models evaluate the validity of each discrete reasoning step.

### 4.1 Step-level versus outcome-level supervision

Let a complete reasoning trajectory $y$ be partitioned into an ordered sequence of $K$ discrete reasoning steps delimited by step boundaries (such as newline tokens `\n\n`):
$$y = (s_1, s_2, \dots, s_K)$$

#### Outcome reward models (ORMs)
An ORM evaluates the full trajectory globally:
$$R_{\text{ORM}}(x, y) \in \mathbb{R} \quad \text{or} \quad P(y \text{ is correct} \mid x) \in [0, 1]$$
ORMs suffer from severe credit assignment ambiguity:
1. False penalty: A trajectory contains 9 valid steps, but suffers a sign error in step 10. The ORM assigns a scalar score of 0, penalizing all 9 valid steps.
2. False reward: A trajectory contains invalid deductions in step 3, but hallucinates the correct final integer in step 10. The ORM assigns a scalar score of 1, rewarding the invalid deductions.

#### Process reward models (PRMs)
A PRM evaluates each step $s_k$ conditioned on the prefix:
$$r_k = R_{\text{PRM}}(x, s_1, \dots, s_k) \in [0, 1] \quad \text{for } k \in \{1, \dots, K\}$$
The PRM directly identifies the precise index $j$ where reasoning derailed:
$$r_1 = 0.99, \quad r_2 = 0.98, \quad r_3 = 0.02, \quad \dots, \quad r_K = 0.01$$

```
+------------------------------------------------------------------------------------+
|                      ORM vs PRM credit assignment comparison                       |
+------------------------------------------------------------------------------------+
| Problem x: Find the minimum of f(x) = x^2 - 4x + 7                                 |
+------------------------------------------------------------------------------------+
| Step 1: Compute derivative f'(x) = 2x - 4.                                         |
|   ORM: [No intermediate score]                                                     |
|   PRM score: r_1 = 0.99 (Valid)                                                    |
+------------------------------------------------------------------------------------+
| Step 2: Set derivative to zero: 2x - 4 = 0 => x = 2.                               |
|   ORM: [No intermediate score]                                                     |
|   PRM score: r_2 = 0.99 (Valid)                                                    |
+------------------------------------------------------------------------------------+
| Step 3: Evaluate f(2) = 2^2 - 4(2) + 7 = 4 - 8 + 7 = 3.                            |
|   ORM: [No intermediate score]                                                     |
|   PRM score: r_3 = 0.98 (Valid)                                                    |
+------------------------------------------------------------------------------------+
| Step 4 (Sign error): Therefore, the minimum value is -3.                           |
|   ORM: 0.0 (Global failure, all steps penalized)                                   |
|   PRM score: r_4 = 0.01 (Isolated failure, steps 1-3 preserved as valid)           |
+------------------------------------------------------------------------------------+
```

---

### 4.2 Supervised PRMs and automated step attribution

Training a PRM requires step-level ground-truth labels. Two primary paradigms exist: human annotation (PRM800K) and automated Monte Carlo attribution (Math-Shepherd).

#### PRM800K (Lightman et al., OpenAI, 2023)
In the PRM800K project ("Let's Verify Step by Step"), human annotators labeled 800,000 step transitions across 75,000 solutions to MATH problems.
- Label space: Each step was labeled as $+1$ (positive/correct), $-1$ (negative/incorrect), or $0$ (neutral/unproductive).
- Active learning protocol: Rather than labeling random steps, OpenAI deployed an active learning loop. Annotators were directed to score solutions at the boundary of model confidence: paths where an existing PRM had high uncertainty, or at the first step where the PRM score dropped below a decision threshold.
- Findings: PRM-guided Best-of-N search outperformed ORM-guided search at iso-compute, requiring orders of magnitude fewer rollouts to identify correct solutions.

#### Math-Shepherd (Pei et al., ACL 2024)
Human step annotation is expensive and difficult to scale. Math-Shepherd introduces an automated process annotation method using Monte Carlo tree rollouts:

```
                                Prefix: (x, s_1, ..., s_{k-1}, s_k)
                                               |
                      +------------------------+------------------------+
                      |                        |                        |
                  Rollout 1                Rollout 2                Rollout M
                      |                        |                        |
                      v                        v                        v
              Final answer A_1         Final answer A_2         Final answer A_M
                      |                        |                        |
              Verify V(A_1, A*)        Verify V(A_2, A*)        Verify V(A_M, A*)
                      |                        |                        |
                      v                        v                        v
                  v_1 in {0, 1}            v_2 in {0, 1}            v_M in {0, 1}
                      \                        |                        /
                       +-----------------------+-----------------------+
                                               |
                                               v
                               V(s_k) = (1/M) * sum_{m=1}^M v_m
```

Let $x$ be a problem and $P_k = (x, s_1, \dots, s_k)$ be a prefix of $k$ steps. To evaluate the mathematical validity of step $s_k$:
1. Sample $M$ independent completions (rollouts) from prefix $P_k$ using policy $\pi$:
   $$R_{k,m} \sim \pi(\cdot \mid P_k), \quad m \in \{1, 2, \dots, M\}$$
2. For each completion, evaluate the final answer against ground truth $y^*$ using deterministic verifier $v(R_{k,m}) \in \{0, 1\}$.
3. Compute the empirical completion value of step $s_k$:
   $$V(s_k) = \frac{1}{M} \sum_{m=1}^M v(R_{k,m})$$
4. Assign step labels:
   Math-Shepherd computes the differential value between consecutive steps $\Delta V_k = V(s_k) - V(s_{k-1})$.
   - If $V(s_k) > 0$ and $\Delta V_k \ge -\epsilon$, step $s_k$ is labeled positive ($y_k = 1$).
   - If $V(s_{k-1}) > 0$ but $V(s_k) = 0$, step $s_k$ introduced an irrecoverable error, and is labeled negative ($y_k = 0$).

This formulation generates millions of step labels autonomously using GPU compute rather than human annotators.

---

### 4.3 Reward head architectures and credit assignment

A PRM typically builds upon a causal decoder backbone (such as a 7B or 8B parameter base model).

#### Architecture
The model processes the concatenated prompt and derivation:
$$X = [x, \text{delimiter}, s_1, \text{delimiter}, s_2, \dots, s_K, \text{delimiter}]$$
where $\text{delimiter}$ is a reserved boundary token (e.g., `\n\n` or a specialized `<step>` token).
Let $h_{t_k} \in \mathbb{R}^d$ represent the final-layer transformer hidden state corresponding to the delimiter token at the end of step $s_k$.
A linear classification head $w \in \mathbb{R}^d$ computes the step reward logit:
$$z_k = w^\top h_{t_k} + b$$
$$r_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}$$

The PRM is trained using binary cross-entropy loss evaluated solely at the step boundary positions:
$$\mathcal{L}_{\text{PRM}}(\phi) = - \sum_{k=1}^K \left[ y_k \log \sigma(w^\top h_{t_k}) + (1 - y_k) \log(1 - \sigma(w^\top h_{t_k})) \right]$$

#### Trajectory aggregation strategies for search
During inference (Best-of-N search, beam search, or MCTS), individual step probabilities $(r_1, r_2, \dots, r_K)$ must be aggregated into a trajectory score $S(y)$:
1. Minimum step score (Weakest link):
   $$S_{\min}(y) = \min_{k \in \{1, \dots, K\}} r_k$$
   Rationale: A mathematical proof is only as strong as its weakest deduction. If any single step fails ($r_k \to 0$), the entire proof is invalid.
2. Product of step scores:
   $$S_{\prod}(y) = \prod_{k=1}^K r_k = \exp\left(\sum_{k=1}^K \log r_k\right)$$
   Rationale: Assumes independent step validity probabilities.
3. Discounted cumulative return:
   $$S_{\text{disc}}(y) = \sum_{k=1}^K \gamma^{K - k} r_k, \quad \gamma \in (0, 1]$$

Empirical studies in PRM800K and DeepSeek-Math demonstrate that the minimum step score $S_{\min}(y)$ consistently outperforms the product metric $S_{\prod}(y)$, because the product metric exhibits strong length bias.

---

### 4.4 Failure modes of process reward models

Despite superior granularity, PRMs exhibit structural failure modes:

#### Reward hacking via stylistic bluffing
PRMs learn spurious correlations between surface rhetorical markers and mathematical correctness. Phrases such as "It is obvious that...", "By symmetry, we observe...", and "Without loss of generality..." frequently correlate with valid derivations in human textbooks. Policies optimized against PRMs learn to prepend these stylistic markers to invalid deductions, artificially inflating the reward logit $w^\top h_{t_k}$.

#### Length bias and compounding probability decay
Under multiplicative scoring $S_{\prod}(y) = \prod_{k=1}^K r_k$, long derivations suffer exponential score degradation even when every step is correct. If a multi-step proof requires 25 steps, and each valid step receives an exemplary score of $r_k = 0.98$, the aggregated score is:
$$S_{\prod}(y) = 0.98^{25} \approx 0.603$$
Conversely, a trivial 3-step proof where each step receives $r_k = 0.90$ scores:
$$S_{\prod}(y) = 0.90^3 = 0.729$$
The model systematically prefers superficial short proofs over comprehensive long derivations. While $S_{\min}$ mitigates this decay, $S_{\min}$ introduces a dual vulnerability: the probability that at least one valid step is assigned a false negative score increases monotonically with proof length $K$.

#### Scoring smooth drift on fine-grained tokens
Transformers represent text as continuous contextual embeddings. When a model makes a single-token arithmetic mistake inside an otherwise valid mathematical derivation:
$$\int x \cos(x) \, dx = x \sin(x) - \int \sin(x) \, dx = x \sin(x) - \cos(x) \quad \text{(Sign error: should be } +\cos(x)\text{)}$$
The local syntactic and semantic context is 99% identical to the correct derivation. The PRM continuous representation often exhibits smooth drift, outputting a high score (such as $r_k = 0.89$) because the sentence structure matches high-confidence training vectors, completely overlooking the inverted minus sign.

#### False step attribution via rollout variance
In automated attribution frameworks (Math-Shepherd), the step score $V(s_k) = \frac{1}{M} \sum v(R_{k,m})$ is an empirical Monte Carlo estimator. For moderate sample sizes ($M = 8$ or $M = 16$), the standard error of the estimator is:
$$\text{SE} = \sqrt{\frac{V(1 - V)}{M}}$$
When $V = 0.5$ and $M = 16$, $\text{SE} = \sqrt{0.25 / 16} = 0.125$. A flawed step can produce 2 lucky completions out of 16 ($V = 0.125$) through downstream hallucination, causing the step to be misclassified as valid. Conversely, a correct step on an AMC 12 problem might produce 0 successful rollouts out of 16 ($V = 0.0$) if the downstream base policy is too weak to finish the proof, causing a mathematically correct intermediate deduction to be labeled as flawed.

---

## 5. Reinforcement learning with verifiable rewards (RLVR)

Reinforcement Learning with Verifiable Rewards (RLVR) replaces learned neural reward models (ORMs and PRMs) with deterministic, rule-based execution environments, eliminating reward model over-optimization.

### 5.1 Rule-based verifiers and deterministic execution engines

In domains with objective truth conditions (mathematics, formal logic, software engineering), the ground truth is an exact reward oracle:

```
                                Model completion trajectory y
                                               |
                                               v
+--------------------------------------------------------------------------------------------+
| Step 1: Delimiter and format validation                                                    |
|         Extract content from <think>...</think> and <answer>...</answer>                   |
|         Format invalid -> Reward = -1.0                                                    |
+--------------------------------------------------------------------------------------------+
                                               | (Format valid)
                                               v
+--------------------------------------------------------------------------------------------+
| Step 2: Answer string normalization                                                        |
|         Strip LaTeX formatting: \mathbf{}, \text{}, \left, \right                          |
|         Canonicalize fractions: \frac{a}{b} -> a/b                                         |
|         Standardize units, sort set elements, normalize intervals                          |
+--------------------------------------------------------------------------------------------+
                                               |
                                               v
+--------------------------------------------------------------------------------------------+
| Step 3: Deterministic semantic verification                                                |
|         - Direct string equivalence: str(A_model) == str(A_true)                           |
|         - Symbolic CAS equivalence (SymPy): simplify(A_model - A_true) == 0                |
|         - Matrix / polynomial equivalence: check rank, eigenvalues, roots                  |
|         - Formal kernel check: Lean 4 / Isabelle type checker exit code                    |
+--------------------------------------------------------------------------------------------+
                                               |
                                               v
+--------------------------------------------------------------------------------------------+
| Step 4: Final reward assignment: r in {0, 1} (or r = r_acc + alpha*r_format)               |
+--------------------------------------------------------------------------------------------+
```

#### Answer string normalization
Mathematical answers in LaTeX exhibit infinite syntactic variety for identical semantic values. A robust verification engine implements normalization pipelines:
- Removing stylistic enclosures: `\text{...}`, `\mathbf{...}`, `\mathrm{...}`, `\left(`, `\right)`.
- Fraction canonicalization: mapping `\frac{4}{6}`, `4/6`, and `0.666...` to `2/3`.
- Set normalization: sorting comma-separated elements in solutions to equations (mapping `\{3, -1\}` to `\{-1, 3\}`).
- Symbolic simplification using Computer Algebra Systems (SymPy):
  Given candidate expression $A_{\text{model}}$ and target expression $A_{\text{true}}$, execute:
  ```python
  import sympy as sp
  diff = sp.simplify(sp.sympify(A_model) - sp.sympify(A_true))
  is_equivalent = (diff == 0)
  ```
  If algebraic simplification times out after 2 seconds, fall back to numerical evaluation across multiple random variable assignments:
  $$\max_{i \in \{1,\dots,5\}} |A_{\text{model}}(z_i) - A_{\text{true}}(z_i)| < 10^{-7}, \quad z_i \sim \mathcal{U}(1, 10)$$

---

### 5.2 Algorithmic formulation: PPO versus GRPO

Standard Proximal Policy Optimization (PPO, Schulman et al., 2017) requires four models concurrently active in GPU cluster memory:
1. Actor policy $\pi_\theta(a \mid s)$: parameterizes the active reasoning model.
2. Value model / Critic $V_\phi(s)$: estimates expected future return $\mathbb{E}[R \mid s]$ to compute Generalized Advantage Estimation (GAE).
3. Reference policy $\pi_{\text{ref}}(a \mid s)$: frozen base model used to enforce KL divergence constraints.
4. Reward model $R_\psi(s, a)$: evaluates completions (in RLVR, replaced by the rule engine).

The Critic network $V_\phi$ presents a massive bottleneck: it has identical parameter scale to the Actor (7B to 70B parameters), consuming 50% of the training VRAM, and is notoriously difficult to train accurately on long reasoning trajectories.

#### Group relative policy optimization (GRPO)
Introduced in DeepSeek-Math (Shao et al., 2024) and foundational to DeepSeek-R1-Zero (DeepSeek-AI, 2025), GRPO completely eliminates the Critic network $V_\phi$. Instead of training a parametric value baseline, GRPO computes advantages relative to an empirical group of completions sampled for the same prompt.

```
                                    Input question q
                                           |
                    +----------------------+----------------------+
                    |                      |                      |
            Output trajectory o_1   Output trajectory o_2   Output trajectory o_G
                    |                      |                      |
                    v                      v                      v
            Verifier reward r_1     Verifier reward r_2    Verifier reward r_G
                    \                      |                      /
                     +---------------------+---------------------+
                                           |
                                           v
                             Group statistics calculation:
                               mu = (1/G) * sum_{i=1}^G r_i
                            sigma = sqrt((1/G) * sum (r_i - mu)^2)
                                           |
                                           v
                             Group normalized advantage:
                               A_i = (r_i - mu) / (sigma + eps)
                                           |
                                           v
                             Actor policy update (Zero critic VRAM)
```

#### Detailed mathematical mechanics of GRPO
For each training query $q \sim \mathcal{D}$:
1. Sample a group of $G$ distinct outputs from the old policy:
   $$\{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(O \mid q)$$
2. Evaluate each completion using the rule-based verifier to obtain group rewards:
   $$\{r_1, r_2, \dots, r_G\}, \quad r_i \in \{0, 1\}$$
3. Compute the group mean and sample standard deviation:
   $$\mu_r = \frac{1}{G} \sum_{i=1}^G r_i, \quad \sigma_r = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \mu_r)^2}$$
4. Compute the normalized advantage for each trajectory $o_i$:
   $$A_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$
   where $\epsilon = 10^{-4}$ prevents division by zero. Notice:
   - If all completions in the group are correct ($r_i = 1, \forall i$), then $\sigma_r = 0$, and advantages are set to 0. The model receives no gradient update.
   - If all completions are incorrect ($r_i = 0, \forall i$), advantages are also 0.
   - Updates occur only on prompts where the group exhibits variance (some correct, some incorrect). Correct trajectories receive positive advantages $A_i > 0$, and incorrect trajectories receive negative advantages $A_i < 0$.
5. Optimize the Actor policy $\theta$ by maximizing the GRPO clipped surrogate objective:
   $$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} A_i, \, \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right) \right]$$

#### Unbiased KL divergence estimator
To prevent the policy from collapsing into degenerate probability distributions while maintaining computational efficiency, the token-level KL divergence penalty is calculated using the unbiased estimator introduced by John Schulman:
$$D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1$$
This formulation is strictly non-negative, equals zero if and only if $\pi_\theta = \pi_{\text{ref}}$, and exhibits lower variance than the naive estimator $\log \pi_\theta - \log \pi_{\text{ref}}$.

---

### 5.3 Emergent behaviors under long-horizon RLVR

When an instruction-tuned base model is subjected to pure RLVR without supervised demonstration data (the DeepSeek-R1-Zero setup), reasoning behaviors emerge autonomously as the model maximizes answer verification rewards over thousands of gradient steps:

#### Spontaneous reflection and the "aha moment"
Without any supervised prompts teaching `<think>` structures, the policy discovers that allocating token compute to trial exploration before emitting the final answer increases the probability of verification success. Tokens representing cognitive reassessment spontaneously increase in probability:
- "Wait, let me double check that calculation."
- "Alternatively, let us test this formula for the base case $n = 1$."
- "This leads to a contradiction because $y$ must be an integer. Let me backtrack."

The model learns to generate test cases, evaluate intermediate invariants, detect inconsistencies, backtrack to earlier branches, and correct arithmetic mistakes.

#### Reasoning trajectory length expansion
Under unconstrained RLVR, average completion length follows an exponential growth curve before stabilizing:

```
Trajectory length (tokens)
   ^
16k|                                                  +-----------------------+
   |                                              +---| Plateau: 12k-16k      |
12k|                                          +---    +-----------------------+
   |                                      +---
 8k|                                  +---
   |                              +--- "Aha moment": rapid token expansion
 4k|                          +---
   |                   +------
  0+---+---------------+------------------------------------------------------> Training steps
       0             200             500             1000            2000
```

1. Initial stage (0 to 200 steps): The model produces concise responses (300 to 800 tokens). Pass rates on complex Olympiad problems hover near 10%.
2. Expansion stage (200 to 1000 steps): The policy discovers verification and backtracking loops. Trajectory length surges from 1,000 tokens to 10,000+ tokens. Pass rates increase.
3. Consolidation stage (1000+ steps): The policy stabilizes between 12,000 and 18,000 tokens. The model eliminates unproductive looping and refines verification routines.

---

### 5.4 Mitigating reward hacking in RLVR

Because RLVR provides feedback solely on verifiable execution, policies discover loopholes in rule engines:

#### Repetition loops and length padding
Policies discover that repeating reasoning steps or generating endless permutations of algebraic expansions can artificially delay termination, occasionally preventing catastrophic errors or exploiting token-length biases in evaluation wrappers.
- Mitigation: Enforce strict repetition penalties. Compute n-gram entropy across sliding windows ($n = 4$). If duplicate 4-gram frequency exceeds a threshold $\tau_{\text{rep}} = 0.15$, terminate rollout and assign a reward penalty:
  $$r_{\text{rep}} = -1.0$$

#### Formatting and containment rewards
To maintain parseable outputs, RLVR incorporates auxiliary formatting rewards:
$$r = r_{\text{acc}} + \alpha r_{\text{format}}$$
The environment validates that:
1. The completion contains exactly one `<think>` opening tag and one `</think>` closing tag.
2. The completion contains exactly one `<answer>` opening tag and one `</answer>` closing tag.
3. No reasoning tokens occur after `</answer>`.
4. The final answer inside `<answer>...</answer>` is non-empty and syntactically valid.
If any structural constraint is violated, $r_{\text{format}} = -1.0$, regardless of final answer validity.

#### Length normalization in surrogate loss
In the GRPO objective, individual token losses are normalized by the trajectory length $|o_i|$:
$$\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \mathcal{L}_t(\theta)$$
Without this length normalization, trajectories with 16,000 tokens dominate the gradient update by a factor of 20 relative to 800-token trajectories, causing policy updates to prioritize lengthening paths rather than improving problem-solving precision.

---

## 6. Formal assistant integration: Lean 4, Isabelle/HOL, and Mathlib

While natural language reasoning and Python execution provide flexible problem solving, they lack absolute logical guarantees. Interactive Theorem Provers (ITPs), specifically Lean 4 and Isabelle/HOL, ground reasoning in formal axiomatic kernels, guaranteeing mathematical soundness.

```
+------------------------------------------------------------------------------------+
|                         Formal theorem proving architecture                        |
+------------------------------------------------------------------------------------+
| Natural language math problem (LaTeX / informal proof)                             |
+------------------------------------------------------------------------------------+
                                 |
                                 v
+------------------------------------------------------------------------------------+
| Autoformalization module (LLM translation)                                         |
| -> Translates to Lean 4 theorem statement and typeclass context                    |
+------------------------------------------------------------------------------------+
                                 |
                                 v
+------------------------------------------------------------------------------------+
| Lean 4 elaborator and typechecker                                                  |
| -> Checks syntax, resolves typeclasses, initializes proof state S_0                |
+------------------------------------------------------------------------------------+
                                 |
                                 +-----------------------------------+
                                 |                                   | (Elaboration error)
                                 v (Valid state)                     v
+----------------------------------------------------+     +-------------------------+
| Proof search engine (MCTS / best-first search)     |     | Diagnostic feedback     |
| - Local hypotheses: Gamma                          |     | and re-prompting        |
| - Target goal: |- T                                |     +-------------------------+
+----------------------------------------------------+
           |
           +---------------------------------+
           |                                 |
           v                                 v
+-----------------------+       +----------------------------------------------------+
| Premise selection     |       | Tactic generation policy: pi(tactic | Gamma |- T)  |
| Ret-by-tactic / dense |       +----------------------------------------------------+
| retrieval over Mathlib|                      |
+-----------------------+                      v
           |                    +----------------------------------------------------+
           +------------------->| Lean 4 REPL execution (socket / JSON-RPC)          |
                                +----------------------------------------------------+
                                               |
                     +-------------------------+-------------------------+
                     |                         |                         |
                     v                         v                         v
               Goals closed              Next goals              Kernel failure
               Reward = +1.0             Expand search tree      Prune branch (Reward = 0)
```

### 6.1 Autoformalization of informal mathematics

Autoformalization is the task of translating natural language statements and informal proofs into formally verified code in an interactive theorem prover.

#### Statement formalization
Consider an informal problem from the American Invitational Mathematics Examination (AIME):
"Let $f(x)$ be a polynomial such that $f(x^2 + 1) = x^4 + 5x^2 + 5$. Find the value of $f(x^2 - 1)$."
An autoformalization model translates this into formal Lean 4 syntax:

```lean
import Mathlib

theorem aime_polynomial_problem (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 5 * x^2 + 5) : 
  ∀ x : ℝ, f (x^2 - 1) = x^4 + x^2 - 1 := by
  sorry
```

#### Structural obstacles in autoformalization
1. Typeclass synthesis and implicit variables:
   Human mathematicians state: "Let $G$ be a finite group." In Lean 4, this requires explicit typeclasses:
   ```lean
   variable {G : Type*} [Group G] [Finite G]
   ```
   Omitting typeclasses or failing to infer universe levels (`Type u`) causes immediate elaboration failure.
2. Informal gaps and algebraic leaps:
   Human proofs frequently assert: "Clearly, by rearranging terms, $a^2 + b^2 \ge 2ab$." In Lean 4, this single informal sentence can require invoking `sub_nonneg`, `sq_nonneg`, and algebraic normalization tactics (`linarith` or `ring`). A language model cannot simply translate human text sentence-by-sentence; it must synthesize the missing logical lemmas connecting intermediate statements.

---

### 6.2 Premise selection and proof search architectures

Theorem proving in Lean 4 is structured as search over a directed proof tree:
- State representation: A proof state $S$ consists of zero or more active goals. Each goal contains a local hypothesis context $\Gamma = \{h_1 : T_1, h_2 : T_2, \dots, h_m : T_m\}$ and a target type $T$:
  $$S = \Gamma \vdash T$$
- Transition function: Applying a tactic $\tau$ (such as `intro x`, `apply add_le_add`, `exact h1`) to goal $\Gamma \vdash T$ produces a set of subgoals:
  $$\tau(\Gamma \vdash T) \to \{S'_1, S'_2, \dots, S'_k\}$$
  If $k = 0$, the goal is closed. A proof succeeds when all branches terminate with $k = 0$ (no remaining subgoals).

#### Proof search algorithms
1. Best-First Search (BFS):
   Maintain a priority queue of open proof states ranked by a neural value function $V(S) \in [0, 1]$. At each step, pop the highest-value state, sample $B$ candidate tactics from policy $\pi(\tau \mid S)$, execute them in Lean 4, and insert valid successor states into the queue.
2. Monte Carlo Tree Search (MCTS):
   Adopted by HyperTree Proof Search (Lample et al., 2022). Each node in the search tree represents a proof state. MCTS navigates the tree via the Upper Confidence Bound for Trees (UCT):
   $$\text{UCT}(S, \tau) = Q(S, \tau) + c_{\text{puct}} P(\tau \mid S) \frac{\sqrt{\sum_b N(S, b)}}{1 + N(S, \tau)}$$
   Simulations evaluate whether tactic sequences lead to closed proof states.

#### Premise selection via dense retrieval
Lean 4's primary mathematical library, Mathlib4, contains over 150,000 formal definitions and theorems. When a goal requires applying an established theorem (e.g., `Real.sin_sq_add_cos_sq`), selecting the correct theorem from Mathlib4 is a high-dimensional retrieval task:
- Bi-encoder architecture: A goal encoder $E_{\text{goal}}$ embeds the current state $\Gamma \vdash T \in \mathbb{R}^d$, while a premise encoder $E_{\text{premise}}$ embeds candidate Mathlib theorems $p_j \in \mathbb{R}^d$.
- Cosine retrieval: Compute similarity scores:
  $$\text{sim}(S, p_j) = \frac{E_{\text{goal}}(S)^\top E_{\text{premise}}(p_j)}{\|E_{\text{goal}}(S)\| \|E_{\text{premise}}(p_j)\|}$$
  Retrieve the top-$k$ premises ($k = 32$) and inject them into the tactic generation prompt:
  ```lean
  -- Available premises: Real.sin_sq_add_cos_sq, Real.cos_le_one
  apply Real.sin_sq_add_cos_sq
  ```

---

### 6.3 Systems engineering: Lean 4 REPL and compiler interface

Interfacing deep learning frameworks (PyTorch, vLLM) with formal verification kernels requires specialized systems infrastructure.

#### Interfacing protocols: LeanDojo and LeanCopilot
1. LeanDojo (Yang et al., NeurIPS 2023):
   Extracts ASTs, premises, and tactic states directly from Lean 4 source repositories. It instruments the Lean compiler environment via Lake (Lean package manager) and provides a Python gym environment communicating over bidirectional Unix pipes.
2. LeanCopilot (Song et al., 2024):
   Runs neural inference natively inside the Lean 4 runtime. It compiles C++ FFI bindings to libtorch or ONNX Runtime directly into Lake packages. Mathematicians type tactics such as `suggest_tactics` inside VS Code, triggering local GPU inference without leaving the Lean environment.
3. Lean 4 REPL (Server mode):
   A daemon process initialized with pre-compiled Mathlib dependencies. It communicates via JSON-RPC over standard input and output:

```json
{"cmd": "theorem test (p q : Prop) (h1 : p) (h2 : p → q) : q := by", "env": 0}
```
The REPL responds with state IDs and active goal strings:
```json
{"proofState": 1, "goals": ["p q : Prop\nh1 : p\nh2 : p → q\n⊢ q"]}
```
Sending a tactic command advances the state:
```json
{"tactic": "exact h2 h1", "proofState": 1}
```
```json
{"proofState": 2, "goals": []}
```
An empty `goals` array confirms proof completion.

#### Engineering failure modes and runtime management
- Timeout enforcement: Automated tactics such as `simp`, `omega`, `aesop`, and `linarith` can enter unbounded loops or trigger worst-case exponential complexity in the decision procedure. Every REPL tactic execution must be wrapped in strict operating system process timeouts (typically 5 to 10 seconds).
- Memory exhaustion: The Lean 4 elaboration environment maintains persistent proof caches. Running tens of thousands of search rollouts in a single REPL process causes memory leaks. Production search engines enforce a periodic worker restart policy (recycling REPL processes every 500 rollouts).
- Kernel synchronization: Tactics that modify the environment or generate transient definitions can desynchronize environment handles. Production engines use stateless snapshot restoration to ensure search tree branches cannot mutate shared global scope.

---

## 7. Comparative technical ledger and codebase citations

The table below catalogs the architectural specifications, training methods, datasets, and primary repositories for the leading mathematical foundation models and formal verification frameworks:

| Project / Model | Primary institution | Parameter scales | Core methodology | Verification / reward mechanism | Primary codebase repository |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-Math** | DeepSeek-AI | 7B | Continual pre-training (120B math tokens), RFT, GRPO | Deterministic rule-based outcome verifier + GRPO | `github.com/deepseek-ai/DeepSeek-Math` |
| **DeepSeek-R1 / R1-Zero** | DeepSeek-AI | 671B MoE (37B active), 1.5B-70B distilled | Cold-start SFT, unconstrained RLVR (R1-Zero), multi-stage rejection sampling + GRPO | Multi-turn rule verifier (SymPy, exact formatting) + language consistency penalty | `github.com/deepseek-ai/DeepSeek-R1` |
| **Qwen2.5-Math** | Alibaba Qwen Team | 1.5B, 7B, 72B | Pre-training on Qwen-Math-Corpus-v2, Tool-Integrated Reasoning (TIR), iterative RFT, GRPO | Python REPL code sandbox + SymPy CAS answer normalization | `github.com/QwenLM/Qwen2.5-Math` |
| **NuminaMath** | Project Numina / Hugging Face | 7B (DeepSeek-Math base) | Evol-Instruct, synthetic tool traces, multi-lingual OCR translation, Best-of-N inference | Python code execution verification (1st Place AIMO Progress Prize) | `github.com/project-numina/aimo-progress-prize` |
| **WizardMath** | Microsoft / Peking University | 7B, 13B, 70B | Reinforced Evol-Instruct, PPO with dual reward models (instruction following + math accuracy) | Learned ORM + Learned PRM (ranking loss) | `github.com/nlpxucan/WizardLM` |
| **Math-Shepherd** | DeepSeek-AI / Peking Univ | 7B | Automated process supervision via Monte Carlo rollouts (M rollouts per step) | Empirical step value estimation $V(s_k) = \frac{1}{M}\sum v_m$ | `github.com/deepseek-ai/DeepSeek-Math` (integrated) |
| **PRM800K** | OpenAI | 175B (GPT-3.5 backbone) | Active learning on 800,000 step transitions labeled by human experts (+1, 0, -1) | Learned step-delimiter token classification head (`\n\n`) | `github.com/openai/prm800k` |
| **ToRA** | Microsoft Research | 7B, 13B, 34B, 70B | Tool-Integrated Reasoning combining natural language CoT with Python code execution | Interactive Python interpreter feedback | `github.com/microsoft/ToRA` |
| **LeanDojo** | Caltech, MIT, NVIDIA | N/A (Gym harness) | ReProver dense retrieval for premise selection + language model tactic generation | Lean 4 / Lean 3 kernel type checking | `github.com/lean-dojo/LeanDojo` |
| **LeanCopilot** | Caltech | N/A (Lake extension) | In-editor LLM tactic suggestion using native C++ libtorch bindings | Lean 4 proof state verification | `github.com/lean-dojo/LeanCopilot` |
| **MiniF2F** | OpenAI | Benchmark | Olympiad-level cross-system formal theorem proving suite (488 problems) | Lean 4, Isabelle, HOL Light, Metamath kernels | `github.com/openai/miniF2F` |
| **PutnamBench** | UT Austin | Benchmark | 644 formal competition problems from the William Lowell Putnam Mathematical Competition | Lean 4, Isabelle, Coq kernels | `github.com/trishullab/PutnamBench` |

---

## 8. Synthesis: Engineering tradeoffs and production recommendations

Building a competitive mathematical foundation model requires orchestrating these distinct techniques into an integrated training pipeline. Teams must navigate concrete trade-offs across data scale, inference latency, compute budget, and formal correctness.

### 8.1 Comparison of reasoning paradigms

| Training paradigm | Compute cost (training) | Verification latency | False positive risk | Primary failure mode | Recommended deployment role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Supervised CoT (SFT)** | Low ($1\times$) | Zero | High | Hallucinated reasoning steps, memorization of solution templates | Initial instruction tuning, establishing base formatting syntax |
| **Rejection sampling (RFT)** | Moderate ($2\times - 5\times$) | Low (offline batch) | Moderate | Mode collapse, failure on hard problems ($p(x) \to 0$), diminishing returns after round 2 | Post-SFT policy bootstrapping, expanding accessible solution space |
| **Process supervision (PRMs)** | High ($10\times$ for rollouts) | Moderate (evaluating $K$ step tokens) | Moderate | Reward hacking via filler phrases, length decay under product metric | Test-time search reranking (Best-of-N), tree search guidance |
| **RLVR (GRPO)** | High ($5\times - 10\times$) | Low to moderate (rule checks) | Low | Repetition loops, format degradation, extreme token length expansion | Frontier capability scaling, inducing autonomous self-correction |
| **Formal provers (Lean 4)** | Extreme ($20\times - 50\times$) | High (0.5s - 5.0s per tactic step) | Zero (kernel-level) | Autoformalization failures, search space explosion, REPL timeout bottlenecks | Rigorous Olympiad mathematics, verified theorem synthesis, kernel grounding |

### 8.2 Production architecture recommendation

For an engineering team deploying a mathematical foundation model under bounded compute budgets, the optimal sequence is:

1. Base pre-training:
   Initialize from a dense model with high math and code token exposure (such as Qwen2.5 or DeepSeek base). Execute continual pre-training on 100B+ tokens of decontaminated LaTeX web text (OpenWebMath) and synthetic problems.
2. Tool-integrated cold-start SFT:
   Fine-tune on 200,000 high-quality synthetic traces combining natural language reasoning with Python/SymPy tool execution blocks (ToRA style) wrapped in strict `<think> ... </think>` and `<answer> ... </answer>` tags.
3. Multi-round RFT (Two iterations):
   Sample $N = 16$ completions at $T = 0.7$. Filter with SymPy CAS verification. Update policy via cross-entropy loss over positive traces. Terminate after round 2 to avoid plateauing.
4. GRPO reinforcement learning with verifiable rewards:
   Deploy Group Relative Policy Optimization with group size $G = 8$ per prompt. Eliminate the Critic network to preserve VRAM. Implement binary accuracy rewards coupled with strict XML formatting penalties ($\alpha = -1.0$) and 4-gram repetition penalties. Allow token horizon expansion up to 16,384 tokens to facilitate autonomous reflection and backtracking.
5. Inference search:
   During inference on high-value competition problems, deploy Best-of-N sampling ($N = 64$) combined with majority voting (self-consistency) over SymPy-canonicalized answers. Reserve Lean 4 formalization pipelines for problems requiring zero-defect verification.
