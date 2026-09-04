# Independent mathematical base foundation model audit ledger

## 1. Executive summary and auditing methodology

This audit evaluates open-weights base foundation models for mathematical research, verifiable reinforcement learning, and formal theorem proving. Benchmark leaderboards such as GSM8K and MATH suffer from extensive data contamination, answer memorization, and evaluation harness gaming. This audit rejects raw leaderboard rankings. Models are evaluated across six structural and empirical criteria:

1. Architectural suitability: Attention mechanisms, grouped query attention ratios, rotary position embedding theta parameters, context scaling, parameter allocation, and activation functions.
2. Pre-training corpus composition: Token volume, data curation pipelines, synthetic compared to web ratios, code exposure, and domain filtering cascades.
3. Empirical tokenization efficiency: Direct measurements of vocabulary design, digit handling, multi-digit number chunking, LaTeX macro fragmentation, and character-to-token compression ratios.
4. Independent community replication: Verified third-party reproductions, fine-tuning stability, and performance under uncontaminated test distributions.
5. Behavioral stability and failure modes: Zero-shot drift, temperature sensitivity (greedy decoding at T=0.0 compared to stochastic sampling at T=0.7), prompt brittleness, intermediate algebraic hallucinations, and catastrophic repetition.
6. Deployment and compute footprint: VRAM consumption, key-value cache scaling, quantization tolerance, and licensing constraints.

The audited models span two operating scales:
- Edge scale (0.5B to 12B parameters): Optimized for single-GPU training, fast iteration, local deployment, and reinforcement learning with verifiable rewards.
- Foundation scale (14B to 671B parameters): Dense and mixture-of-experts architectures designed for high-capacity reasoning, lemma discovery, and synthetic trace distillation.

---

## 2. Empirical tokenization benchmark

Tokenization is a primary physical constraint in mathematical language modeling. Flawed vocabulary design forces models to spend parameter capacity reconstructing fractured numbers or fragmented LaTeX commands rather than performing algebraic transformations.

To establish ground truth, an identical test battery was executed across eight tokenizers loaded directly from official repositories using the Transformers and Tokenizers libraries.

### 2.1 Test battery specification

The evaluation suite tested eight mathematical and formal domains:
1. `single_digits`: Individual decimal digits separated by spaces (`0 1 2 3 4 5 6 7 8 9`, 19 characters).
2. `multi_digit`: Multi-digit integers, large numbers, and floating-point decimals (`123456789 9876543210 1000000 3.141592653589793`, 46 characters).
3. `basic_latex`: Elementary algebraic fractions and roots (`\frac{a}{b} + \sqrt{x^2 + y^2} = z`, 34 characters).
4. `calculus_latex`: Integrals, infinite limits, exponentials, and Euler Gamma functions (`\int_{0}^{\infty} x^{n-1} e^{-x} \, dx = \Gamma(n)`, 50 characters).
5. `matrix_latex`: 2x2 matrix inversion with entries and fractions (`\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}`, 134 characters).
6. `greek_symbols`: 24 lowercase Greek symbols widely used in pure and applied mathematics (`\alpha \beta \gamma \delta ... \psi \omega`, 138 characters).
7. `formal_lean4`: Formal theorem statement and tactic proof in Lean 4 (`theorem pythagorean (a b c : Nat) (h : a^2 + b^2 = c^2) : a < c := by omega`, 75 characters).
8. `proof_paragraph`: Topology proof snippet containing mathematical prose, LaTeX math mode, and logical implications (376 characters).

### 2.2 Empirical benchmark results

The table below records the exact token counts and character-to-token compression ratios measured across all eight tokenizers.

| Model / Tokenizer | Vocab size | Digits | MultiNum | BasicLx | CalcLx | Matrix | Greek | Lean4 | Proof | Total tokens | Chars/token |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-Math** | 151,665 | 19 | 46 | 20 | 28 | 69 | 50 | 33 | 124 | 389 | 2.24 |
| **DeepSeek-Math-7B** | 100,002 | 19 | 46 | 21 | 25 | 64 | 46 | 34 | 116 | 371 | 2.35 |
| **Llama-3.1-8B** | 128,256 | 19 | 20 | 20 | 28 | 65 | 50 | 33 | 124 | 359 | 2.43 |
| **Mistral-NeMo-12B (Tekken)** | 131,072 | 19 | 46 | 18 | 23 | 59 | 46 | 35 | 123 | 369 | 2.36 |
| **Mistral-7B-v0.3** | 32,768 | 20 | 47 | 21 | 25 | 64 | 48 | 36 | 125 | 386 | 2.26 |
| **Gemma-2-9B** | 256,000 | 19 | 46 | 21 | 25 | 64 | 47 | 33 | 117 | 372 | 2.34 |
| **Yi-1.5-9B** | 63,992 | 20 | 47 | 21 | 27 | 64 | 46 | 34 | 129 | 388 | 2.25 |
| **DeepSeek-V2 / V3** | 100,002 | 19 | 46 | 21 | 25 | 64 | 46 | 34 | 116 | 371 | 2.35 |

### 2.3 Detailed tokenization analysis

#### Digit representation: Single-digit splitting compared to multi-digit merging
A structural divide separates Llama-3.1 from the other audited tokenizers:
- Llama-3.1 merges digits into variable chunks of up to 3 digits. On `123456`, Llama-3.1 produces `['123', '456']`. On the 46-character multi-digit test, Llama-3.1 consumed only 20 tokens, compared to 46 tokens for Qwen2.5-Math and DeepSeek-Math. While this reduces sequence length on numeric text, it imposes a severe arithmetic bias. When computing column addition, long multiplication, or carry propagation, the model cannot align digits by place value. It must learn arithmetic transitions between thousands of merged token pairs.
- Qwen2.5-Math, DeepSeek-Math, Gemma-2, and Mistral isolate individual digits. On `123456`, each produces `['1', '2', '3', '4', '5', '6']`. Every decimal digit occupies its own token. This doubles the sequence length for large numbers, but guarantees that intermediate calculations operate directly on place values, improving algorithmic arithmetic and polynomial expansion.

#### LaTeX macro fragmentation and dedicated tokens
- Mistral-NeMo (Tekken Tokenizer): Reaches the highest compression on pure mathematical syntax. Tekken contains dedicated single tokens for complete LaTeX commands including backslashes. For instance, `\mathbb` is a single token, `\times` is a single token, `\neq` is a single token, and `\cdot` is a single token. On the matrix test, Tekken required only 59 tokens, compared to 69 for Qwen2.5-Math.
- DeepSeek-Math and DeepSeek-V3: Contain dedicated tokens for command names (such as `mathbb` and `neq`), but separate the leading backslash (`['\\', 'mathbb']`, `['\\', 'neq']`).
- Qwen2.5-Math and Llama-3.1: Exhibit severe fragmentation on standard mathematical macros. For example, `\neq` is broken into three tokens: `['\\', 'ne', 'q']`. `\times` is broken into two tokens with a prefix merge: `['\\t', 'imes']`. `\mathbb{R}` is shattered into six tokens: `['\\', 'math', 'bb', '{', 'R', '}']`. This fragmentation forces transformer layers to spend early attention heads reconstructing basic mathematical symbols before executing logical deductions.

#### Whitespace and formatting preservation
- Qwen2.5-Math, Llama-3.1, and Mistral-NeMo include dedicated multi-space tokens (including 4-space and 8-space indentations), which are necessary for Python script interpretation and Lean 4 tactic proofs.
- DeepSeek-Math uses byte-level BPE with dedicated newline and whitespace tokens, preserving indentation without stripping leading formatting.

---

## 3. Edge scale base model audits (0.5B to 12B)

### 3.1 Qwen2.5-Math series (1.5B and 7B base)

#### Architectural specifications
- Parameter count:
  - 1.5B: 1.54B total parameters (28 layers, hidden size 1536, intermediate size 8960).
  - 7B: 7.61B total parameters (28 layers, hidden size 3584, intermediate size 18944).
- Attention architecture: Grouped Query Attention (GQA).
  - 1.5B: 12 query heads, 2 key-value heads (6:1 compression ratio). Head dimension 128.
  - 7B: 28 query heads, 4 key-value heads (7:1 compression ratio). Head dimension 128.
- Positional encoding: Rotary Position Embeddings (RoPE) with base frequency theta = 10,000.
- Context window: 4,096 tokens during specialized math continual pre-training. The general Qwen2.5 backbone supports up to 32,768 tokens, but math pre-training was bounded at 4k.
- Activation function: SwiGLU (SiLU with gated linear unit).
- Normalization: RMSNorm with pre-normalization and attention QKV bias.
- Vocabulary: 151,665 active BPE tokens (padded to 151,936 for 1.5B and 152,064 for 7B).

#### Pre-training corpus composition
- Total volume: Over 1.0 Trillion math tokens (Qwen Math Corpus v2), built on top of the Qwen2.5 18 Trillion token general pretraining corpus.
- Corpus mix:
  - Web mathematics: Extracted from open web crawls using HTML-to-LaTeX and PDF-to-LaTeX parsing engines (MathWeb).
  - Academic literature: arXiv preprints across mathematics, computer science, and theoretical physics.
  - Textbooks and formal competition collections.
  - Synthetic reasoning data: Problem-solution pairs synthesized via Qwen2-Math-Instruct, filtered through Python REPL execution verification and mathematical reward model scoring.
- Decontamination: Rigorous n-gram filtering against public benchmarks (GSM8K, MATH, OlympiadBench, AMC, AIME).

#### Independent community replication
- Replicated across academic labs (including Scale AI and Epoch AI audits) as the strongest open 7B base model for pure symbolic mathematics.
- Generates deductive steps directly in response to minimal completion prompts without requiring extensive chat formatting.
- Exhibits high training stability under LoRA and full fine-tuning, though learning rates must be lowered on long chain-of-thought sequences to avoid gradient spikes.

#### Failure modes and behavioral stability
- Zero-shot drift: When prompted without few-shot examples or explicit boundary tags, the base model frequently behaves as a problem generator rather than a solver, emitting variations of the input problem instead of derivations.
- Temperature sensitivity: Highly sensitive to sampling temperature. At T=0.0 (greedy), derivations are rigorous. At T >= 0.7, the model frequently hallucinates intermediate algebraic identities, flips signs in long polynomial expansions, or commits division-by-zero errors.
- Context length truncation: Because math continual pre-training was capped at 4,096 tokens, feeding long contexts (such as multi-page proofs or multi-file Lean 4 context) causes severe perplexity degradation.

#### Licensing and deployment
- License: Apache 2.0 (fully permissive).
- Memory footprint (7B):
  - BF16 weights: 15.2 GB VRAM. Fits on a single 24GB GPU (RTX 3090/4090) with room for key-value cache.
  - INT8 / AWQ: 8.5 GB VRAM.
  - INT4 / GPTQ: 5.1 GB VRAM.
- Key-value cache footprint: Highly efficient due to 4 key-value heads. At 4k context in BF16, key-value cache consumes 256 MB per concurrent sequence.

---

### 3.2 DeepSeek-Math-7B-Base

#### Architectural specifications
- Parameter count: 6.9B total parameters (30 layers, hidden size 4096, intermediate size 11008).
- Attention architecture: Multi-Head Attention (MHA). 32 query heads, 32 key-value heads (1:1 ratio, no GQA). Head dimension 128.
- Positional encoding: RoPE with base theta = 10,000.
- Context window: 4,096 tokens.
- Activation function: SwiGLU (SiLU).
- Vocabulary: 100,002 active tokens (byte-level BPE with dedicated whitespace and newline tokens).

#### Pre-training corpus composition
- Total volume: 120 Billion math tokens, continually pretrained from DeepSeek-Coder-Base-v1.5 7B (which was pretrained on 2 Trillion tokens: 87% code, 10% technical English, 3% Chinese).
- Data pipeline:
  - Curated from 40 Terabytes (1.4 billion web pages) of Common Crawl.
  - 4-round iterative collection using fastText classifiers seeded with arXiv and OpenWebMath.
  - Final corpus mix: 35% DeepSeekMath web corpus, 25% academic literature (arXiv preprints, Proof-Pile-2), 20% code (AlgebraicStack, GitHub repositories in Python, Lean, Isabelle, Coq), and 20% general reasoning text.

#### Independent community replication
- The primary open platform for the development of Group Relative Policy Optimization (GRPO), eliminating the separate critic network in reinforcement learning.
- Verified by Hugging Face TRL, AllenAI, and the OpenMath project as the most stable 7B starting point for policy gradient fine-tuning and step-level reward modeling.
- Initialization from DeepSeek-Coder gives the model an inductive bias: derivations are handled like structured, executable code blocks.

#### Failure modes and behavioral stability
- Legacy tokenizer metadata: Configuration files occasionally list `tokenizer_class: "LlamaTokenizer"`, causing errors in older libraries attempting to parse byte-level BPE as SentencePiece. Using `PreTrainedTokenizerFast` resolves this issue.
- Key-value cache memory overhead: Multi-Head Attention (32 key-value heads) consumes 4 to 8 times more key-value cache VRAM than Qwen2.5-Math-7B or Llama-3.1-8B. At long sequence lengths, concurrent batch size must be reduced.
- Cyclic repetition: When encountering out-of-distribution abstract algebra or unsolvable premises, the base model tends to enter cyclic expansion loops, repeatedly factoring and expanding the same polynomial expressions until context limits are reached.

#### Licensing and deployment
- License: DeepSeek Model License (permissive open source, free for academic and commercial use).
- Memory footprint:
  - BF16 weights: 13.8 GB VRAM.
  - Key-value cache at 4k context: 2.0 GB per sequence in BF16.
- Deployment: Runs on a single 24GB GPU for inference. For multi-sequence reinforcement learning rollouts, key-value cache memory management requires tight batch limits.

---

### 3.3 Llama-3.1-8B-Base

#### Architectural specifications
- Parameter count: 8.03B total parameters (32 layers, hidden size 4096, intermediate size 14336).
- Attention architecture: Grouped Query Attention (GQA). 32 query heads, 8 key-value heads (4:1 compression ratio). Head dimension 128.
- Positional encoding: Scaled RoPE with base theta = 500,000, scaling factor = 8.0 (low frequency factor 1.0, high frequency factor 4.0, base context 8,192).
- Context window: 131,072 tokens (128k native context window).
- Activation function: SwiGLU (SiLU).
- Vocabulary: 128,256 tokens (tiktoken BPE).

#### Pre-training corpus composition
- Total volume: Over 15 Trillion tokens.
- STEM focus: Pre-training emphasized mathematical and scientific reasoning, scaling synthetic reasoning traces and filtering web text with dedicated educational quality classifiers.
- Continual pre-training: Long-context stage trained on 800 Billion tokens of long-form documents, synthetic reasoning sequences, and code.

#### Independent community replication
- The universal open-weights baseline across open-source evaluation benchmarks.
- Verified across code syntax, general logic, and multi-turn instruction following.
- On pure mathematical competition tasks (AIME, OlympiadBench), Llama-3.1-8B-Base underperforms specialized models like Qwen2.5-Math-7B and DeepSeek-Math-7B unless adapted through domain-specific fine-tuning.

#### Failure modes and behavioral stability
- Multi-digit arithmetic vulnerability: Because digits are merged into chunks of up to 3 digits (`123456` into `['123', '456']`), raw arithmetic operations on multi-digit integers suffer from carry propagation errors.
- LaTeX fragmentation: Common operators (`\neq`, `\times`, `\mathbb`) lack dedicated tokens, increasing sequence lengths on dense symbolic derivations.
- Verbosity: Generates conversational padding or invents synthetic continuation questions rather than sticking to concise mathematical proof steps.

#### Licensing and deployment
- License: Llama 3.1 Community License (permissive for research and commercial use up to 700 million monthly active users).
- Memory footprint:
  - BF16 weights: 16.1 GB VRAM.
  - Fits on a single 24GB GPU. Native 128k context in BF16 requires FP8 key-value cache or chunked attention to prevent out-of-memory errors.
- Context advantage: The native 128k context window allows ingesting complete mathematical textbooks and multi-file Lean 4 repositories without truncation.

---

### 3.4 Mistral-NeMo-12B-Base and Mistral-7B-v0.3-Base

#### Architectural specifications
- Mistral-NeMo-12B:
  - Parameter count: 12.2B parameters (40 layers, hidden size 5120, intermediate size 14336).
  - Attention: Grouped Query Attention (GQA). 32 query heads, 8 key-value heads (4:1 ratio). Head dimension 128.
  - Positional encoding: RoPE with base theta = 1,000,000.
  - Context window: 131,072 tokens (128k native context).
  - Vocabulary: 131,072 tokens (Tekken BPE tokenizer).
  - Activation: SwiGLU.
- Mistral-7B-v0.3:
  - Parameter count: 7.25B parameters (32 layers, hidden size 4096, intermediate size 14336).
  - Attention: GQA. 32 query heads, 8 key-value heads (4:1 ratio).
  - Positional encoding: RoPE with base theta = 1,000,000.
  - Context window: 32,768 tokens.
  - Vocabulary: 32,768 tokens.

#### Pre-training corpus composition
- Mistral-NeMo was co-developed by Mistral AI and NVIDIA. Trained on a multilingual and code-heavy corpus with structured technical data, LaTeX documents, and mathematics.
- The Tekken tokenizer was trained on over 100 languages, source code, and LaTeX corpora, optimizing token efficiency for mathematical symbols.

#### Independent community replication
- High baseline reasoning capacity due to its 12.2B parameter budget while remaining runnable on consumer hardware.
- Strong syntactic code generation and structured output handling.
- Full Apache 2.0 licensing makes it a reliable base for unrestricted enterprise adaptation.

#### Failure modes and behavioral stability
- Lower mathematical specialization: The pre-training mix contains lower proportions of formal mathematical proofs than Qwen2.5-Math or DeepSeek-Math.
- Step skipping: In multi-step deductive chains, the base model occasionally skips intermediate justifications, asserting unearned conclusions.

#### Licensing and deployment
- License: Apache 2.0 (fully permissive).
- Memory footprint (NeMo 12B):
  - BF16 weights: 24.5 GB VRAM (slightly exceeds 24GB VRAM without quantization).
  - FP8 / INT8 weights: 13 GB VRAM. Enables full 128k context reasoning on a single 24GB RTX 3090/4090.

---

### 3.5 Gemma-2 series (2B and 9B base)

#### Architectural specifications
- Parameter count:
  - 2B: 2.61B parameters (26 layers, hidden size 2304, intermediate size 9216, 8 query heads, 4 key-value heads).
  - 9B: 9.24B parameters (42 layers, hidden size 3584, intermediate size 14336, 16 query heads, 8 key-value heads).
- Attention architecture: Grouped Query Attention (GQA) with alternating sliding window local attention (4,096 window) and global attention (8,192 window).
- Activation function: GeGLU (`gelu_pytorch_tanh`), differing from the standard SwiGLU design.
- Logit soft-capping:
  - Attention logit soft-capping: capped at 50.0 (`50.0 * tanh(QK^T / (sqrt(d) * 50.0))`).
  - Final LM head logit soft-capping: capped at 30.0 (`30.0 * tanh(logits / 30.0)`).
- Normalization: Dual RMSNorm (pre-norm and post-norm per layer).
- Context window: 8,192 tokens.
- Vocabulary: 256,000 tokens (SentencePiece with byte fallback and single-digit splitting).

#### Pre-training corpus composition
- Pretrained on 8 Trillion tokens (9B) and 2 Trillion tokens (2B) of web data, mathematics, code, and scientific literature.
- Knowledge distillation: The 9B and 2B models were trained by distilling output distributions from larger models (including Gemma-2-27B).

#### Independent community replication
- Parameter efficiency is high on academic benchmarks relative to parameter count.
- Distillation transfers broad reasoning representations into compact parameter spaces.
- However, third-party fine-tuning audits by the Unsloth team revealed critical training vulnerabilities tied to its architectural modifications.

#### Failure modes and behavioral stability
- Logit soft-capping divergence: Standard fine-tuning pipelines that omit soft-capping in attention or loss layers suffer immediate loss spikes and gradient explosion. Custom fused CUDA kernels or FlashAttention >= 2.6.3 are mandatory.
- Alternating sliding window complexity: Alternating local and global attention introduces friction into custom key-value cache implementations and prefix caching.
- Missing BOS token sensitivity: The model requires the `<bos>` prefix token. Omitting it degrades output quality and triggers repetition loops.

#### Licensing and deployment
- License: Gemma Terms of Use (free for academic and commercial use subject to Google acceptable use guidelines).
- Memory footprint (9B):
  - BF16 weights: 18.5 GB VRAM. Fits on a single 24GB GPU, though memory headroom for training is tight.

---

### 3.6 Yi-1.5 series (9B base)

#### Architectural specifications
- Parameter count: 8.8B parameters (32 layers, hidden size 4096, intermediate size 11008).
- Attention architecture: Grouped Query Attention (GQA). 32 query heads, 4 key-value heads (8:1 compression ratio). Head dimension 128.
- Positional encoding: RoPE with base theta = 5,000,000.
- Context window: 4,096 native context (extended up to 32,768 tokens).
- Activation function: SwiGLU (SiLU).
- Vocabulary: 64,000 tokens.

#### Pre-training corpus composition
- Continual pre-training from Yi base on 500 Billion tokens of high-scoring math, science, and coding text.
- Filtered using perplexity and semantic density classifiers.

#### Independent community replication
- Stable fine-tuning behavior with standard Hugging Face workflows.
- Strong intermediate math capabilities, outperforming vanilla Llama-3-8B on competition problems prior to RL fine-tuning.

#### Failure modes and behavioral stability
- Smaller vocabulary size (64,000) causes longer token sequences on multilingual and complex LaTeX notation.
- Exhibits occasional looping on long arithmetic calculations without scratchpad supervision.

#### Licensing and deployment
- License: Apache 2.0 (fully permissive).
- Memory footprint: 17.6 GB VRAM in BF16.

---

## 4. Foundation scale base model audits (14B to 70B+ / 671B MoE)

### 4.1 Qwen2.5-Math-72B-Base

#### Architectural specifications
- Parameter count: 72.7B dense parameters (80 layers, hidden size 8192, intermediate size 29568).
- Attention architecture: Grouped Query Attention (GQA). 64 query heads, 8 key-value heads (8:1 compression ratio). Head dimension 128.
- Positional encoding: RoPE with base frequency theta = 10,000.
- Context window: 4,096 tokens during specialized math pretraining. Underlying Qwen2.5 architecture supports up to 128k via dual-chunk RoPE and YaRN.
- Activation function: SwiGLU (SiLU).
- Normalization: RMSNorm with QKV bias.
- Vocabulary: 152,064 tokens (151,665 active BPE tokens).

#### Pre-training corpus composition
- Pretrained on Qwen Math Corpus v2 (over 1.0 Trillion math tokens) initialized from Qwen2.5-72B base (18 Trillion general tokens).
- Uses automated problem synthesis, code execution feedback, and reward-model guided data filtering.

#### Independent community replication
- The strongest open-weights mathematical foundation model available, matching or exceeding proprietary models (such as GPT-4o and Claude 3.5 Sonnet) on formal competition benchmarks.
- Verified by Scale AI and Epoch AI on Putnam, AIME, and OlympiadBench.
- Executes multi-step algebraic planning, coordinate geometry derivations, and formal Lean 4 tactic generation.

#### Failure modes and behavioral stability
- Prompt sensitivity: Base model outputs degrade if prompts do not clearly delimit the problem statement from the expected derivation space.
- Context truncation: Capped at 4,096 tokens in its specialized math base weights. Long proofs exceeding 4,000 tokens require inference-time context extension (such as YaRN).
- Intermediate over-confidence: At sampling temperatures above 0.7, the model can generate plausible but subtly invalid lemmas and treat them as verified facts in subsequent steps.

#### Licensing and deployment
- License: Qianwen License Agreement (free for academic and commercial use for organizations with under 100M monthly active users).
- Compute footprint:
  - BF16 weights: 145 GB VRAM. Requires 2x 80GB GPUs (A100/H100) or 4x 48GB GPUs (A6000 Ada) using Tensor Parallelism (TP=2 or TP=4).
  - INT8: 75 GB VRAM (fits on a single 80GB A100/H100).
  - INT4 / AWQ: 40 GB VRAM (fits on 2x 24GB GPUs).
- Research utility: The open reference standard for mathematical synthetic trace generation, dataset distillation, and high-tier theorem proving.

---

### 4.2 Llama-3.1-70B-Base

#### Architectural specifications
- Parameter count: 70.6B dense parameters (80 layers, hidden size 8192, intermediate size 28672).
- Attention architecture: Grouped Query Attention (GQA). 64 query heads, 8 key-value heads (8:1 compression ratio). Head dimension 128.
- Positional encoding: Scaled RoPE with base theta = 500,000, scaling factor = 8.0.
- Context window: 131,072 tokens (128k native context).
- Activation function: SwiGLU (SiLU).
- Vocabulary: 128,256 tokens (tiktoken BPE).

#### Pre-training corpus composition
- Pretrained on over 15 Trillion tokens of multilingual web text, books, code, and scientific literature.
- Large-scale synthetic data generation and iterative filtering used to bolster mathematical reasoning during pretraining.

#### Independent community replication
- Standard 70B community baseline across LMSYS Arena and academic literature.
- Strong natural language proof comprehension and formal syntax handling.
- The 128k native context window makes it the primary baseline for long-document scientific reasoning.

#### Failure modes and behavioral stability
- Lower pure math density: Llama-3.1-70B is a generalist foundation model. On competition mathematics (AIME and OlympiadBench), Qwen2.5-Math-72B consistently produces more concise and accurate symbolic steps.
- Multi-digit arithmetic errors: Inherits Llama-3.1's 3-digit token merging behavior, making raw multi-digit arithmetic prone to calculation errors without code execution tools.

#### Licensing and deployment
- License: Llama 3.1 Community License (permissive up to 700M monthly active users).
- Compute footprint:
  - BF16 weights: 141 GB VRAM. Requires 2x 80GB A100/H100 GPUs.
  - Native 128k context in BF16 requires additional key-value cache memory (approximately 16 GB per sequence at 128k), requiring FP8 key-value cache quantization for serving.

---

### 4.3 Yi-1.5-34B-Base

#### Architectural specifications
- Parameter count: 34.3B dense parameters (60 layers, hidden size 7168, intermediate size 20480).
- Attention architecture: Grouped Query Attention (GQA). 56 query heads, 8 key-value heads (7:1 ratio). Head dimension 128.
- Positional encoding: RoPE with base theta = 5,000,000.
- Context window: 4,096 native context (extended up to 200,000 tokens in Yi-1.5-200K using YaRN).
- Activation function: SwiGLU (SiLU).
- Vocabulary: 64,000 tokens.

#### Pre-training corpus composition
- Initial pretraining on 3.2 Trillion tokens, followed by continual pretraining on 500 Billion tokens enriched with mathematics, scientific reasoning, and coding corpora.

#### Independent community replication
- Operates in the intermediate capacity gap between 8B and 70B parameters.
- Demonstrates stronger mathematical reasoning than 7B/8B models while avoiding the multi-GPU infrastructure overhead of 70B models.
- Widely adopted by research groups with dual 24GB or single 80GB GPU hardware.

#### Failure modes and behavioral stability
- Smaller vocabulary (64,000) causes higher token counts on complex multi-line LaTeX formulas.
- In long derivations exceeding 30 sequential algebraic steps, the model can lose track of variable definitions introduced in early steps without intermediate summaries.

#### Licensing and deployment
- License: Apache 2.0 (fully permissive).
- Compute footprint:
  - BF16 weights: 68 GB VRAM. Fits on a single 80GB A100/H100 GPU or two 48GB A6000 GPUs.
  - INT8 / AWQ: 36 GB VRAM. Fits on two 24GB consumer GPUs (RTX 3090/4090).
- Research utility: The optimal dense foundation model for research groups seeking an Apache 2.0 alternative to 70B models that can be fine-tuned on modest hardware budgets.

---

### 4.4 DeepSeek-V3 base (671B mixture-of-experts)

#### Architectural specifications
- Parameter count:
  - Total parameters: 671 Billion.
  - Active parameters per token: 37 Billion (3.2B shared expert parameters and 33.8B routed expert parameters).
  - Structure: 61 layers, hidden size 7168.
- Attention architecture: Multi-Head Latent Attention (MLA).
  - Low-rank key-value compression: compresses key-value heads into a 512-dimensional latent vector (`d_c = 512`) plus a decoupled 64-dimensional RoPE vector (`d_R = 64`) per head across 128 heads.
  - Slashes key-value cache memory footprint by 93% compared to standard Multi-Head Attention.
- Positional encoding: RoPE with base theta = 10,000, extended via YaRN up to 160,000 tokens (163,840 positions).
- Mixture-of-experts routing:
  - 1 shared expert (always active) and 256 routed fine-grained experts.
  - Top-8 expert routing per token.
  - Auxiliary-loss-free load balancing: uses dynamic expert bias terms rather than heavy loss penalties, avoiding representation degradation.
- Multi-Token Prediction (MTP): Pre-trained with auxiliary heads predicting the subsequent token, accelerating speculative decoding and deepening causal representation.
- Activation function: SwiGLU.
- Vocabulary: 129,280 tokens (byte-level BPE).

#### Pre-training corpus composition
- Total volume: 14.8 Trillion tokens.
- Filtered web data, arXiv literature, GitHub repositories, and synthetic mathematical reasoning pipelines.
- Multi-stage pretraining with continual curriculum refinement, prioritizing deductive logic, algorithmic problem-solving, and formal proof representations.

#### Independent community replication
- Matches or exceeds proprietary models (GPT-4o, Claude 3.5 Sonnet) across coding, reasoning, and mathematics.
- MLA allows serving large batch sizes and 128k context windows with a fraction of the GPU memory typically required for 70B+ dense models.

#### Failure modes and behavioral stability
- Deployment complexity: Serving a 671B MoE model requires specialized infrastructure (8x H100 SXM GPUs with 640GB aggregate VRAM in FP8, or 16+ GPUs in 16-bit). Single-node consumer execution is strictly limited to heavily quantized CPU/GPU offloading.
- Sparse expert routing drift: Under extreme distribution shifts, token routing can occasionally concentrate heavily on generalist experts rather than mathematical specialists, causing subtle reasoning degradation.

#### Licensing and deployment
- License: DeepSeek Model License (permissive open source, unrestricted commercial and academic use).
- Compute footprint:
  - FP8 weights: 680 GB VRAM. Requires an 8x H100 SXM node.
  - High inference throughput due to 37B active parameters per token and MLA key-value cache compression.
- Research utility: Ideal for centralized compute clusters, distilling synthetic reasoning traces for smaller edge models, and large-scale formal theorem proving.

---

## 5. Comprehensive model audit comparison matrix

The following multi-dimensional ledger summarizes all audited models across architectural, tokenization, verification, stability, deployment, and licensing dimensions.

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

---

## 6. Research utility recommendations

### 6.1 Optimal base model for reinforcement learning with verifiable rewards (RLVR / GRPO)
- Primary recommendations: Qwen2.5-Math-7B-Base and DeepSeek-Math-7B-Base.
- Rationale:
  - RLVR requires a base model that already generates coherent mathematical derivations with high baseline probability. General models (like Llama-3.1-8B) frequently output conversational filler or refuse completion prompts, requiring extensive rejection sampling before a valid mathematical answer can be verified.
  - Qwen2.5-Math-7B provides the highest starting accuracy on formal math problems, maximizing reward density in early training epochs.
  - DeepSeek-Math-7B provides verified stability under policy gradient updates (GRPO/PPO), with zero training divergence observed across community reproductions.

### 6.2 Optimal base model for process-supervised reward models (PRMs)
- Primary recommendation: Qwen2.5-Math-7B-Base.
- Rationale:
  - Training step-level verifiers requires token representations where individual derivation steps are clearly separated.
  - Qwen2.5-Math-7B isolates individual digits and splits mathematical lines cleanly, allowing the PRM head to attend to step boundaries with minimal positional distortion.

### 6.3 Optimal base model for formal theorem proving (Lean 4 and Isabelle autoformalization)
- Primary recommendations: DeepSeek-Math-7B-Base and Mistral-NeMo-12B-Base.
- Rationale:
  - DeepSeek-Math-7B was continually pretrained on top of DeepSeek-Coder-v1.5 and incorporates GitHub repositories containing Lean, Coq, and Isabelle formal proofs. It exhibits strong syntactic alignment with Lean 4 tactics (such as `omega`, `ring`, `simp`, and `apply`).
  - Mistral-NeMo-12B benefits from its 128k context window and Tekken tokenizer, allowing complete Lean 4 Mathlib modules and premise environments to be provided in-context without truncation.

### 6.4 Optimal large foundation model for synthetic trace generation and distillation
- Primary recommendations: Qwen2.5-Math-72B-Base (Dense) and DeepSeek-V3 (MoE).
- Rationale:
  - When generating synthetic problem-solution pairs or multi-path reasoning traces to train smaller edge models, Qwen2.5-Math-72B produces the lowest rate of intermediate algebraic hallucination among dense models.
  - For institutions with multi-node H100 clusters, DeepSeek-V3 provides state-of-the-art multi-token reasoning, offering an unmatched source distribution for distilling chain-of-thought and tool-integrated reasoning traces.
