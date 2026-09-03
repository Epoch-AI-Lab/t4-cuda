# Implementation Research: Calibrated Honesty ("I Don't Know") RL Training under CP-Hybrid Stack

## 1. The Goal
Train a compact 0.5B language model (`Qwen/Qwen2.5-0.5B-Instruct`) on a single Tesla T4 using GRPO to achieve calibrated honesty:
- When a question is within its knowledge boundary and answerable: produce the correct answer.
- When a question is factually impossible (false premise, nonexistent entity), unanswerable, or hopelessly out-of-league (e.g. unsolvable open math problems): output a concise "I don't know" or `<abstain>`. No preachiness, no essays explaining why the user's premise was flawed.
- The entire training loop runs under our **CP-Hybrid kernel stack** (Attention in FP16, MLPs in group=128 INT4 with dynamic $M \le 2$ decode dispatch), guaranteeing fast rollouts, low VRAM, and zero gradient degradation during backward passes.

## 2. Dataset Design & Balancing
To prevent the model from collapsing into trivial degenerate solutions (e.g. always saying "I don't know" to cheat rewards, or never saying "I don't know" and hallucinating), the training dataset must be strictly balanced (50/50):

1. **Answerable Set (50%)**:
   - Math & Logic: GSM8K and simple arithmetic.
   - Factual Knowledge: Basic geography, history, science questions with verifiable ground truth.
2. **Unanswerable / Impossible / Out-of-League Set (50%)**:
   - Impossible Premise Traps: "What year did Napoleon conquer Jupiter?", "Who was the president of the Pacific Ocean in 1840?", "What is the capital of Antarctica?"
   - SQuAD 2.0 Unanswerable: Questions specifically crafted to have no answer in the provided context.
   - Out-of-League Open Problems: "What is the exact decimal expansion of Chaitin's constant?", "What is the exact non-trivial zero of the Riemann zeta function closest to 1000?"

## 3. Reward Function Mechanics

Let $y$ be the model completion, $q$ be the question, and $label \in \{\text{"answerable"}, \text{"unanswerable"}\}$.

$$R(y, q) = R_{\text{correct}}(y, q) + R_{\text{format}}(y)$$

### $R_{\text{correct}}(y, q)$:
- **Case 1: $label == \text{"answerable"}$**
  - If model outputs ground-truth answer: $R = +1.0$
  - If model says "I don't know" (false abstention / cowardice): $R = 0.0$
  - If model gives wrong answer: $R = 0.0$
- **Case 2: $label == \text{"unanswerable"}$**
  - If model outputs "I don't know" / "idk" / `<abstain>`: $R = +0.6$ (calibrated honesty reward)
  - If model hallucinates a confident fake answer: $R = 0.0$ (penalized)

### $R_{\text{format}}(y)$:
- Encourages concise `<think>...</think>` internal chain-of-thought, followed by the answer: $+0.2$.

## 4. Verification Protocol
1. **Pre-Training Evaluation**:
   Evaluate untrained `Qwen2.5-0.5B-Instruct` on a 50-example held-out honesty benchmark. Measure:
   - Factual accuracy on answerable questions.
   - Hallucination rate on impossible questions (how often does it hallucinate vs say "I don't know").
2. **GRPO Training under CP-Hybrid Stack**:
   Run 30-50 steps on a physical Tesla T4. Monitor:
   - Convergence of honesty reward.
   - Rollout latency and peak VRAM under CP-Hybrid.
3. **Post-Training Evaluation**:
   Run the exact same 50-example held-out honesty benchmark.
   Verify that:
   - Hallucination rate on impossible questions drops drastically (replaces fake facts with "I don't know").
   - Factual accuracy on answerable questions remains intact.
