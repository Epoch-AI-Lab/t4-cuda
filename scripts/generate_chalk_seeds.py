#!/usr/bin/env python3
"""
Chalk Parallel Synthetic Seed Dataset Generator
Generates 2,000 golden math reasoning traces using parallel workers querying
dynamic free models across OpenRouter with rate-limit evasion and variable-assignment normalization:
- inclusionai/ling-3.0-flash-fin:free
- minimax/minimax-m3:free
- nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
- nvidia/nemotron-3.5-lightning:free
- google/gemma-4-26b-a4b-it:free
- google/gemma-4-31b-it:free
- z-ai/glm-5.2:free

Enforces:
1. Deep mathematical problem from 6 disciplines + contest transmutations.
2. AST Anti-templating validator ensuring no structural template appears more than twice (C(T) <= 2).
3. 5-tag mathematician scratchpad: <explore>, <conjecture>, <test_edge_cases>, <lemma_isolate>, <formal_proof>.
4. Exact symbolic ground-truth verification using SymPy and Z3.
"""

import os
import sys
import json
import time
import re
import random
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
CHALK_SRC = Path("/home/kriday/teamwork_projects/chalk_synthetic_data/src")
if str(CHALK_SRC) not in sys.path:
    sys.path.insert(0, str(CHALK_SRC))

ENV_FILE = REPO_ROOT / ".env"
OUTPUT_FILE = REPO_ROOT / "data" / "chalk_seeds_2k.jsonl"

def load_env():
    """Load variables from .env file if present."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

# Imports from chalk library
from chalk.core.models import Discipline, ProblemInstance
from chalk.generators.number_theory import NumberTheoryGenerator
from chalk.generators.abstract_algebra import AbstractAlgebraGenerator
from chalk.generators.analysis import AnalysisGenerator
from chalk.generators.combinatorics import CombinatoricsGenerator
from chalk.generators.geometry import GeometryGenerator
from chalk.generators.spectral_optimization import SpectralOptimizationGenerator
from chalk.generators.transmutations import TransmutationEngine
from chalk.anti_templating.validator import AntiTemplateValidator
from chalk.verification.verifier import SymbolicVerifier
from chalk.reasoning.validators import extract_boxed_answers

# Dynamic model pool
MODEL_POOL = [
    "inclusionai/ling-3.0-flash-fin:free",
    "minimax/minimax-m3:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "z-ai/glm-5.2:free",
]

SYSTEM_PROMPT = """You are an expert mathematician solving challenging mathematical problems.
Do not output a simple direct answer. You must structure your entire mathematical thought process inside five explicit XML tags:

1. <explore>
Explore the problem informally. Examine the constraints, symmetries, degrees of freedom, and dimensional units. Brainstorm potential avenues of attack.

2. <conjecture>
State a precise, falsifiable intermediate lemma or claim that would make the problem tractable.

3. <test_edge_cases>
Actively attempt to break or verify your conjecture by testing small, concrete, or extreme cases (e.g. n = 0, 1, 2, -1, degenerate values, or extreme boundaries). If the test reveals a flaw, note the refutation and patch the conjecture.

4. <lemma_isolate>
State the surviving, verified lemma clearly in clean mathematical notation.

5. <formal_proof>
Write the clean, rigorous, deductive proof or calculation leading to the final result. Highlight the final boxed answer in LaTeX: \\boxed{answer}.

Every response MUST contain all five tags in order."""

def call_openrouter(
    model: str,
    prompt: str,
    api_key: str,
    timeout: int = 70
) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenRouter HTTP API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Epoch-AI-Lab/t4-cuda",
        "X-Title": "Chalk Math Generator",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 3000,
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return content, None
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return None, f"HTTP {e.code}: {err_body[:120]}"
    except Exception as e:
        return None, str(e)

def validate_response_tags(text: str) -> bool:
    """Check that text contains the 5 required tags and a boxed answer."""
    if not text:
        return False
    required = ["explore", "conjecture", "test_edge_cases", "lemma_isolate", "formal_proof"]
    for tag in required:
        if not re.search(rf"<\s*{tag}\s*>", text, re.IGNORECASE):
            return False
    return bool(extract_boxed_answers(text))

def clean_boxed_answer(ans: str) -> str:
    """Normalizes candidate answer by stripping variable assignments and trailing punctuation."""
    ans = ans.strip()
    ans = re.sub(r'[\.,;]+$', '', ans).strip()
    if '=' in ans:
        parts = ans.split('=')
        lhs = parts[0].strip()
        # If LHS is a variable (e.g. 'd', 'x_0', '\text{ans}', '\lambda')
        if re.match(r'^(?:[a-zA-Z_]\w*(?:_[a-zA-Z0-9{}]+)?|\\[a-zA-Z]+(?:\{[^}]*\})?)$', lhs):
            ans = '='.join(parts[1:]).strip()
    return ans

class ProblemFactory:
    """Thread-safe problem generator across 6 disciplines and contest transmutations."""

    def __init__(self):
        self._lock = threading.Lock()
        self.generators = {
            Discipline.NUMBER_THEORY: NumberTheoryGenerator(),
            Discipline.ABSTRACT_ALGEBRA: AbstractAlgebraGenerator(),
            Discipline.ANALYSIS: AnalysisGenerator(),
            Discipline.COMBINATORICS: CombinatoricsGenerator(),
            Discipline.GEOMETRY: GeometryGenerator(),
            Discipline.SPECTRAL_OPTIMIZATION: SpectralOptimizationGenerator(),
        }
        self.disciplines = list(self.generators.keys())
        self.transmutation_sources = TransmutationEngine.get_supported_contests()

    def generate(self, seed: int) -> ProblemInstance:
        with self._lock:
            rng = random.Random(seed)
            if rng.random() < 0.15 and self.transmutation_sources:
                contest = rng.choice(self.transmutation_sources)
                try:
                    return TransmutationEngine.transmute(contest, seed=seed)
                except Exception:
                    pass
            disc = self.disciplines[seed % len(self.disciplines)]
            return self.generators[disc].generate(seed=seed)

class ParallelDatasetGenerator:
    """Coordinates parallel worker threads with dynamic model pool rotation."""

    def __init__(self, api_key: str, num_workers: int = 6, target_count: int = 2000):
        self.api_key = api_key
        self.num_workers = num_workers
        self.target_count = target_count
        self.factory = ProblemFactory()
        self.verifier = SymbolicVerifier()
        self.validator = AntiTemplateValidator(max_occurrences=2, similarity_threshold=0.65, raise_on_violation=False)
        self.file_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.seed_counter = 1000
        self.saved_count = 0
        self.model_cooldowns: Dict[str, float] = {m: 0.0 for m in MODEL_POOL}

    def load_existing(self):
        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            self.saved_count += 1
                            self.validator.check_and_register(rec["problem"], rec.get("discipline", "general"))
                        except Exception:
                            pass
        self.seed_counter = 2000 + self.saved_count * 29
        print(f"Loaded {self.saved_count} existing certified seeds in {OUTPUT_FILE.name}")

    def get_model(self) -> Optional[str]:
        """Returns a non-cooling model from the pool, prioritized by readiness."""
        with self.state_lock:
            now = time.time()
            ready = [m for m in MODEL_POOL if self.model_cooldowns.get(m, 0) <= now]
            if not ready:
                return None
            return random.choice(ready)

    def mark_cooldown(self, model: str, duration: float = 40.0):
        with self.state_lock:
            self.model_cooldowns[model] = time.time() + duration

    def worker_loop(self, worker_id: int):
        print(f"[Worker {worker_id}] Thread active.")

        while True:
            with self.state_lock:
                if self.saved_count >= self.target_count:
                    break
                self.seed_counter += 1
                curr_seed = self.seed_counter

            # Dynamically get available model
            model = self.get_model()
            if not model:
                # All models cooling down, brief pause
                time.sleep(3.0)
                continue

            # 1. Generate problem
            prob = self.factory.generate(curr_seed)

            # 2. Anti-templating check (C(T) <= 2)
            with self.state_lock:
                report = self.validator.check_and_register(
                    prob.problem,
                    prob.discipline.value if isinstance(prob.discipline, Discipline) else str(prob.discipline)
                )
            if not report.passed:
                continue

            # 3. Call OpenRouter
            resp, err = call_openrouter(model, prob.problem, self.api_key, timeout=75)

            if err:
                if "429" in err:
                    self.mark_cooldown(model, duration=45.0)
                    print(f"[Worker {worker_id}] {model} rate-limited (429). Cooldown 45s. Rotating...")
                time.sleep(1.0)
                continue

            # 4. Tag validation
            if not validate_response_tags(resp):
                time.sleep(0.5)
                continue

            # 5. Extract, clean, and symbolically verify boxed answer
            boxed_answers = extract_boxed_answers(resp)
            raw_answer = boxed_answers[-1].strip()
            candidate_answer = clean_boxed_answer(raw_answer)

            verif = self.verifier.verify(candidate_answer, str(prob.ground_truth_answer).strip())

            if not verif.verified:
                # Also try raw answer in case normalization differed
                verif = self.verifier.verify(raw_answer, str(prob.ground_truth_answer).strip())

            if not verif.verified:
                print(f"[Worker {worker_id}] ✗ SymPy mismatch: got '{candidate_answer}' (raw '{raw_answer}') vs GT '{prob.ground_truth_answer}' ({prob.id})")
                time.sleep(0.5)
                continue

            # 6. Save certified record
            with self.file_lock:
                with self.state_lock:
                    self.saved_count += 1
                    current_id = self.saved_count

                record = {
                    "id": current_id,
                    "problem_id": prob.id,
                    "discipline": prob.discipline.value if isinstance(prob.discipline, Discipline) else str(prob.discipline),
                    "subfield": prob.subfield,
                    "theorem_source": prob.theorem_source,
                    "problem": prob.problem,
                    "ground_truth": prob.ground_truth_answer,
                    "model_answer": candidate_answer,
                    "model": model,
                    "provider": "openrouter",
                    "trace": resp,
                    "ast_hash": report.ast_hash,
                    "occurrence_count": report.occurrence_count,
                    "verification_engine": verif.verifier,
                    "timestamp": time.time(),
                }
                with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                model_short = model.split('/')[1].split(':')[0]
                print(f"[Worker {worker_id}] ✓ Certified seed #{current_id}/{self.target_count} saved! "
                      f"[{prob.discipline.value}] Model: {model_short} | GT: {prob.ground_truth_answer}")

            time.sleep(1.0)

    def run(self):
        print(f"Commencing parallel generation with {self.num_workers} workers targeting {self.target_count} seeds...")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.worker_loop, i) for i in range(self.num_workers)]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"Worker error: {e}", file=sys.stderr)

def main():
    print("=" * 65)
    print("  CHALK PARALLEL SYNTHETIC SEED GENERATOR")
    print("=" * 65)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key or openrouter_key == "your_openrouter_api_key_here":
        print("ERROR: OPENROUTER_API_KEY not found in .env. Exiting.", file=sys.stderr)
        sys.exit(1)

    workers = 6
    for arg in sys.argv:
        if arg.startswith("--workers="):
            workers = int(arg.split("=")[1])

    generator = ParallelDatasetGenerator(api_key=openrouter_key, num_workers=workers, target_count=2000)
    generator.load_existing()

    if "--run" not in sys.argv:
        print("\nReady. Run with: python3 scripts/generate_chalk_seeds.py --run [--workers=6]")
        return

    generator.run()

if __name__ == "__main__":
    main()
