#!/usr/bin/env python3
"""
Chalk Phase-1 Bootstrap Seed Generator
Generates a golden bootstrap dataset of 50-70 pristine, certified math reasoning traces
across all 7 disciplines (Graph Theory, Number Theory, Abstract Algebra, Analysis,
Combinatorics, Geometry, Discrete Optimization & Spectral Theory).

Enforces:
1. Human chalkboard mathematician voice with natural swearing and unslop scratchpad.
2. Complete 5-tag schema: <explore>, <conjecture>, <test_edge_cases>, <lemma_isolate>, <formal_proof>.
3. Multi-tier symbolic verification (SymPy + Z3) with zero floating point.
4. AST Anti-templating validator ensuring C(T) <= 2.
5. Certified output saved to data/chalk_seeds_bootstrap.jsonl.
"""

import os
import sys
import json
import time
import re
import random
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTH_SRC = Path("/home/kriday/teamwork_projects/chalk_synthetic_data/src")
MANUAL_SCRIPTS = Path("/home/kriday/teamwork_projects/chalk_manual_synthesis/scripts")

for p in (SYNTH_SRC, MANUAL_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ENV_FILE = REPO_ROOT / ".env"
OUTPUT_FILE = REPO_ROOT / "data" / "chalk_seeds_bootstrap.jsonl"

def load_env():
    """Load variables from .env file."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

# Imports from verify_record
from verify_record import validate_record, AntiTemplatingLedger, extract_boxed_answers

# Imports from chalk generators
from chalk.generators.number_theory import NumberTheoryGenerator
from chalk.generators.abstract_algebra import AbstractAlgebraGenerator
from chalk.generators.analysis import AnalysisGenerator
from chalk.generators.combinatorics import CombinatoricsGenerator
from chalk.generators.geometry import GeometryGenerator
from chalk.generators.spectral_optimization import SpectralOptimizationGenerator

MODELS = [
    "minimax/minimax-m3:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

SYSTEM_PROMPT = """You are a real human mathematician working through challenging mathematical problems late at night on a chalkboard.
Do NOT talk like a polite, sterile AI assistant. Avoid all AI buzzwords and textbook recitation.
Do NOT use em dashes (—) or en dashes (–); use standard punctuation (commas or periods).
Use simple, direct words, natural human muttering, and authentic swearing when grappling with the problem.

Structure your thought process inside five explicit XML tags in this exact order:

1. <explore>
Raw, intuitive chalkboard exploration. Talk like a real mathematician wrestling with the problem at 2am.
MANDATORY: You MUST include natural human swearing and reactions here (e.g. 'What the fuck is this asking?', 'Daym, that is soo spaghetti maths', 'SHIIIII that constraint is tight', 'I need more coffee for this shit', 'whoever wrote this contest problem is a sadist').
Mutter out loud, sanity check the constraints, degrees of freedom, invariants, and symmetries.
Close with </explore>.

2. <conjecture>
State a precise, falsifiable intermediate lemma candidate with explicit parameter domains.
Close with </conjecture>.

3. <test_edge_cases>
Actively and aggressively try to break your conjecture on small or extreme cases (n=0, 1, 2, -1, boundary values, empty cases).
MANDATORY: You MUST include authentic human reactions, sanity checks, and swearing here as well (e.g. 'sanity check: wait what the fuck, n=1 blows up', 'damn it, why did I assume that was positive', 'holy shit, boundary term does not vanish').
Patch the flaw honestly.
Close with </test_edge_cases>.

4. <lemma_isolate>
State the surviving, verified lemma clearly in clean mathematical notation.
Close with </lemma_isolate>.

5. <formal_proof>
Write the clean, disciplined, rigorous deductive proof or calculation leading to the final result.
End strictly with exactly one final boxed answer in LaTeX: \\boxed{answer}.
Close with </formal_proof>.

Every response MUST contain all five tags in order, with explicit opening and closing tags."""

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
        "X-Title": "Chalk Bootstrap Generator",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
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

def generate_graph_theory_problem(seed: int) -> Dict[str, Any]:
    """Generates rich Graph Theory problems with known symbolic ground truth."""
    rng = random.Random(seed)
    archetype = rng.choice([
        "planar_triangulation",
        "cayley_trees",
        "complete_bipartite_edges",
        "chromatic_k_n",
        "hypercube_edges",
        "petersen_chromatic_index",
        "bipartite_max_edges"
    ])

    if archetype == "planar_triangulation":
        v = rng.choice([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20])
        e = 3 * v - 6
        return {
            "subfield": "Planar Graphs & Euler Formula",
            "theorem": "Euler Polyhedral Formula for Maximal Planar Graphs",
            "problem": f"Let $G$ be a connected maximal planar graph (a triangulation of the sphere) on {v} vertices. Determine the exact number of edges in $G$.",
            "ground_truth": str(e)
        }
    elif archetype == "cayley_trees":
        n = rng.choice([4, 5, 6, 7, 8])
        ans = n ** (n - 2)
        return {
            "subfield": "Enumerative Graph Theory",
            "theorem": "Cayley Formula for Number of Labeled Trees",
            "problem": f"Find the total number of distinct labeled trees on {n} vertices with vertex set $V = \\{{1, 2, \\dots, {n}\\}}$.",
            "ground_truth": str(ans)
        }
    elif archetype == "complete_bipartite_edges":
        m = rng.choice([5, 6, 7, 8, 9])
        n = rng.choice([8, 9, 10, 11, 12])
        ans = m * n
        return {
            "subfield": "Bipartite Graphs & Extremal Theory",
            "theorem": "Complete Bipartite Edge Capacity",
            "problem": f"Let $K_{{{m},{n}}}$ denote the complete bipartite graph with partition sizes {m} and {n}. What is the exact total number of edges in $K_{{{m},{n}}}$?",
            "ground_truth": str(ans)
        }
    elif archetype == "chromatic_k_n":
        n = rng.choice([5, 6, 7, 8, 9, 10])
        return {
            "subfield": "Chromatic Graph Theory",
            "theorem": "Chromatic Number of Complete Graphs",
            "problem": f"Let $K_{{{n}}}$ be the complete graph on {n} vertices. What is the chromatic number $\\chi(K_{{{n}}})$?",
            "ground_truth": str(n)
        }
    elif archetype == "hypercube_edges":
        d = rng.choice([4, 5, 6, 7, 8])
        edges = d * (2 ** (d - 1))
        return {
            "subfield": "Structural Graph Theory",
            "theorem": "Hypercube Graph Edge Count and Degree Regularity",
            "problem": f"Let $Q_{{{d}}}$ denote the {d}-dimensional hypercube graph. Find the exact total number of edges in $Q_{{{d}}}$.",
            "ground_truth": str(edges)
        }
    elif archetype == "petersen_chromatic_index":
        return {
            "subfield": "Edge Colorings & Snarks",
            "theorem": "Vizing Class 2 Theorem and Petersen Chromatic Index",
            "problem": "Determine the chromatic index $\\chi'(P)$ (the edge chromatic number) of the standard Petersen graph.",
            "ground_truth": "4"
        }
    else:
        n = rng.choice([6, 8, 10, 12, 14])
        # Turan max edges for triangle-free graph on n vertices: floor(n^2 / 4)
        ans = (n * n) // 4
        return {
            "subfield": "Extremal Graph Theory",
            "theorem": "Mantel Theorem on Maximum Edges in Triangle-Free Graphs",
            "problem": f"By Mantel's Theorem, what is the maximum number of edges in a simple triangle-free graph on {n} vertices?",
            "ground_truth": str(ans)
        }

class UnifiedProblemGenerator:
    """Combines all 7 disciplines."""

    def __init__(self):
        self.generators = {
            "Number Theory": NumberTheoryGenerator(),
            "Abstract Algebra": AbstractAlgebraGenerator(),
            "Real & Complex Analysis": AnalysisGenerator(),
            "Combinatorics": CombinatoricsGenerator(),
            "Geometry & Topology": GeometryGenerator(),
            "Discrete Optimization & Spectral Theory": SpectralOptimizationGenerator(),
        }
        self.disciplines = ["Graph Theory"] + list(self.generators.keys())

    def generate(self, seed: int) -> Dict[str, Any]:
        disc = self.disciplines[seed % len(self.disciplines)]
        if disc == "Graph Theory":
            p = generate_graph_theory_problem(seed)
            return {
                "discipline": "Graph Theory",
                "subfield": p["subfield"],
                "theorem": p["theorem"],
                "problem": p["problem"],
                "ground_truth": p["ground_truth"]
            }
        else:
            gen = self.generators[disc]
            prob = gen.generate(seed=seed)
            return {
                "discipline": disc,
                "subfield": prob.subfield,
                "theorem": prob.theorem_source,
                "problem": prob.problem,
                "ground_truth": str(prob.ground_truth_answer)
            }

def clean_trace_formatting(trace: str) -> str:
    """Strips forbidden em/en dashes and auto-repairs unclosed tags."""
    trace = trace.replace("—", ", ").replace("–", "-")
    
    # Auto-repair missing closing tags
    tags = ["explore", "conjecture", "test_edge_cases", "lemma_isolate", "formal_proof"]
    for i, tag in enumerate(tags):
        open_pat = rf"<\s*{tag}\s*>"
        close_pat = rf"</\s*{tag}\s*>"
        if re.search(open_pat, trace, re.IGNORECASE) and not re.search(close_pat, trace, re.IGNORECASE):
            # If next tag starts, insert closing tag before it
            if i + 1 < len(tags):
                next_open = rf"<\s*{tags[i+1]}\s*>"
                m = re.search(next_open, trace, re.IGNORECASE)
                if m:
                    idx = m.start()
                    trace = trace[:idx].rstrip() + f"\n</{tag}>\n" + trace[idx:]
                else:
                    trace = trace.strip() + f"\n</{tag}>\n"
            else:
                # formal_proof is the last tag
                trace = trace.strip() + f"\n</{tag}>\n"
    return trace

def main():
    print("=" * 65)
    print("  CHALK PHASE-1 BOOTSTRAP GENERATOR (TARGET: 50 CERTIFIED SEEDS)")
    print("=" * 65)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "your_openrouter_api_key_here":
        print("ERROR: OPENROUTER_API_KEY not found in .env.", file=sys.stderr)
        sys.exit(1)

    ledger = AntiTemplatingLedger()
    generator = UnifiedProblemGenerator()

    # Load existing certified records
    existing_records = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        valid, _, _ = validate_record(rec, ledger)
                        if valid:
                            existing_records.append(rec)
                    except Exception:
                        pass

    count = len(existing_records)
    target = 50
    print(f"Loaded {count} existing certified seeds in {OUTPUT_FILE.name}")
    print(f"Target count: {target} seeds")

    model_idx = 0
    seed_idx = 5000 + count * 37
    cooldowns: Dict[str, float] = {}

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        while count < target:
            seed_idx += 1
            prob_info = generator.generate(seed_idx)

            # Pick ready model
            now = time.time()
            ready_models = [m for m in MODELS if cooldowns.get(m, 0) <= now]
            if not ready_models:
                print("All models cooling down, sleeping 5s...")
                time.sleep(5.0)
                continue

            model = ready_models[model_idx % len(ready_models)]
            model_idx += 1

            model_short = model.split('/')[1].split(':')[0]
            print(f"[{count + 1}/{target}] Querying {model_short} for [{prob_info['discipline']}]...")

            resp, err = call_openrouter(model, prob_info["problem"], api_key, timeout=70)

            if err:
                print(f"  ✗ {model_short} failed: {err}")
                if "429" in err:
                    cooldowns[model] = time.time() + 40.0
                time.sleep(2.0)
                continue

            # Clean formatting and repair unclosed tags
            resp = clean_trace_formatting(resp)

            # Extract boxed answer
            boxed = extract_boxed_answers(resp)
            if not boxed:
                print(f"  ✗ No boxed answer found in response")
                time.sleep(1.0)
                continue

            candidate_box = boxed[-1].strip()
            # Clean variable assignment if present (e.g. d = 30 -> 30)
            if "=" in candidate_box:
                parts = candidate_box.split("=")
                if re.match(r'^(?:[a-zA-Z_]\w*|\\[a-zA-Z]+(?:\{[^}]*\})?)$', parts[0].strip()):
                    candidate_box = parts[1].strip()

            record = {
                "id": f"chalk_bootstrap_{count + 1:04d}",
                "discipline": prob_info["discipline"],
                "subfield": prob_info["subfield"],
                "theorem_source": prob_info["theorem"],
                "problem": prob_info["problem"],
                "ground_truth": prob_info["ground_truth"],
                "boxed_answer": candidate_box,
                "trace": resp,
            }

            # Run full certification through verify_record
            valid, errors, meta = validate_record(record, ledger)

            if not valid:
                print(f"  ✗ Certification failed: {errors[:2]}")
                time.sleep(1.0)
                continue

            # Attach verification metadata
            record["symbolic_verification"] = meta.get("symbolic_verification", {})
            record["anti_template_metadata"] = meta.get("anti_template_metadata", {})
            record["model"] = model
            record["timestamp"] = time.time()

            # Save certified seed
            count += 1
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(f"  ✓ Certified seed #{count}/{target} saved! [{prob_info['discipline']}] GT: {prob_info['ground_truth']} | Box: {candidate_box}")
            time.sleep(2.0)

    print("\n" + "=" * 65)
    print(f"  PHASE-1 COMPLETE: {count} CERTIFIED SEEDS GENERATED")
    print(f"  Saved to: {OUTPUT_FILE}")
    print("=" * 65)

if __name__ == "__main__":
    main()
