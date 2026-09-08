#!/usr/bin/env python3
"""Build RL training pool for Chalk competition math post-training.

Pulls from:
  1. AI-MO/NuminaMath-CoT  — AMC/AIME/IMO shortlist / olympiad problems
  2. qq8933/AIME_1983_2024 — full AIME archive 1983-2024

Then:
  - Deduplicates against the existing SFT data (chalk_seeds_500.jsonl)
  - Writes data/rl_pool.jsonl ready to drop into train_baby_chalk_rl.py

Usage:
  python3 data/build_rl_pool.py
  python3 data/build_rl_pool.py --max-per-source 2000 --out data/rl_pool.jsonl

Run on Colab / Kaggle where `datasets` is available.
Locally: pip install datasets first.
Set HF_TOKEN env var if any dataset requires HuggingFace login.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Deps check
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] `datasets` not installed. Run: pip install datasets")
    sys.exit(1)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SFT_FILE = os.path.join(REPO_DIR, "data", "chalk_seeds_500.jsonl")
OUT_FILE = os.path.join(REPO_DIR, "data", "rl_pool.jsonl")

# NuminaMath sources to keep. "olympiads" is a broad catch-all so we also
# apply a post-hoc answer-quality filter (see _is_valid_answer).
NUMINA_COMPETITION_SOURCES = {"amc_aime", "olympiads"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Normalise problem text for deduplication across different formatting styles."""
    s = text.strip().lower()
    # Normalize common math operator variants
    s = s.replace(r"\times", "*").replace("×", "*").replace(r"\cdot", "*")
    s = s.replace(r"\leq", "<=").replace(r"\le", "<=").replace("≤", "<=")
    s = s.replace(r"\geq", ">=").replace(r"\ge", ">=").replace("≥", ">=")
    s = s.replace(r"\neq", "!=").replace(r"\ne", "!=").replace("≠", "!=")
    # Strip common LaTeX math block delimiters ($$, $, \(, \), \[, \])
    s = s.replace(r"\[", " ").replace(r"\]", " ").replace(r"\(", " ").replace(r"\)", " ")
    s = s.replace("$$", " ").replace("$", " ")
    # Remove non-breaking spaces and redundant whitespace
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _fingerprint(text: str) -> str:
    return hashlib.md5(_norm(text).encode("utf-8")).hexdigest()


def _extract_boxed(text: str) -> Optional[str]:
    """Brace-balanced extraction of last \\boxed{...} in text.

    Matches '\\boxed{' and '\\boxed {' with arbitrary whitespace before brace.
    Handles escaped braces correctly (e.g. '\\boxed{\\{1, 2\\}}').
    Handles nested braces correctly, e.g. \\boxed{\\frac{3}{4}}.
    Returns None if no valid \\boxed is found.
    """
    matches = list(re.finditer(r"\\\\boxed\s*\{", text))
    if not matches:
        return None

    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    chars = []

    for i in range(start, len(text)):
        c = text[i]
        # Check if character is preceded by an odd number of backslashes (escaped)
        is_escaped = False
        bs_count = 0
        j = i - 1
        while j >= start and text[j] == "\\":
            bs_count += 1
            j -= 1
        if bs_count % 2 == 1:
            is_escaped = True

        if c == "{" and not is_escaped:
            depth += 1
        elif c == "}" and not is_escaped:
            depth -= 1
            if depth == 0:
                break
        chars.append(c)

    if depth == 0:
        result = "".join(chars).strip()
        if not result:
            return None
        # Normalize signed or unsigned pure numbers with commas like "1,000" or "-1,000" -> "1000" / "-1000"
        if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", result):
            result = result.replace(",", "")
        return result
    return None


def _is_valid_answer(answer: str) -> bool:
    """Reject answers that are clearly not numerical/algebraic.

    Filters out open-ended answers from the broad "olympiads" bucket
    (e.g. proof-only problems, inequality statements, pure text answers).
    """
    if not answer or len(answer) > 200:
        return False

    # Reject prose using whole-word boundaries to avoid "improved" matching "prove"
    prose_pattern = r"\b(prove|show that|there exist|for all|if and only|we have|it follows|therefore|hence|since)\b"
    if re.search(prose_pattern, answer.lower()):
        return False

    # Allow standalone LaTeX math constants: \pi, \infty, \phi etc.
    if re.fullmatch(r"\\[a-zA-Z]+", answer.strip()):
        return True

    # Allow short single/multi-letter algebraic variables (e.g. 'x', 'n', 'ab', 'k')
    if re.fullmatch(r"[a-zA-Z]{1,3}", answer.strip()):
        return True

    # After stripping LaTeX commands and delimiters, must contain a digit, operator, or coordinate separator
    stripped = re.sub(r"\\[a-zA-Z]+", "", answer)          # remove \frac, \sqrt, \pi ...
    stripped = re.sub(r"[{}\[\]()\\ ]", "", stripped)       # remove delimiters
    if re.search(r"[\d+\-*/=^<>,_]", stripped):
        return True

    return False


def _make_record(idx: int, source: str, problem: str, ground_truth: str,
                 boxed_answer: str, competition: str = "") -> dict:
    return {
        "id": f"rl_{source}_{idx:05d}",
        "source": source,
        "competition": competition,
        "problem": problem.strip(),
        "ground_truth": ground_truth,
        "boxed_answer": boxed_answer,
    }


# ---------------------------------------------------------------------------
# Load existing SFT fingerprints (for deduplication)
# ---------------------------------------------------------------------------

def load_sft_fingerprints() -> set:
    fps: set = set()
    if not os.path.exists(SFT_FILE):
        print(f"[WARN] SFT file not found at {SFT_FILE}, skipping dedup.")
        return fps
    with open(SFT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                fps.add(_fingerprint(rec.get("problem", "")))
    print(f"[INFO] Loaded {len(fps)} SFT fingerprints for deduplication.")
    return fps


# ---------------------------------------------------------------------------
# Source 1: NuminaMath-CoT
# ---------------------------------------------------------------------------

def pull_numina(max_per_source: int, seen: set) -> list:
    print("[INFO] Pulling AI-MO/NuminaMath-CoT ...")
    hf_token = os.environ.get("HF_TOKEN")
    try:
        ds = load_dataset(
            "AI-MO/NuminaMath-CoT",
            split="train",
            trust_remote_code=True,
            token=hf_token,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load NuminaMath-CoT: {e}")
        return []

    # Shuffle before iterating — NuminaMath is clustered by source, so sequential
    # reads would exhaust max_per_source on early AMC rows and never reach olympiads.
    ds = ds.shuffle(seed=42)

    records = []
    counter = 0  # sequential ID counter (not raw row index)
    skipped_source = skipped_dup = skipped_no_answer = skipped_bad_answer = 0

    for row in ds:
        if len(records) >= max_per_source:
            break

        source = (row.get("source") or "").lower().strip()
        if source not in NUMINA_COMPETITION_SOURCES:
            skipped_source += 1
            continue

        problem = row.get("problem", "").strip()
        solution = row.get("solution", "") or ""

        if not problem:
            continue

        fp = _fingerprint(problem)
        if fp in seen:
            skipped_dup += 1
            continue

        boxed = _extract_boxed(solution)
        if not boxed:
            skipped_no_answer += 1
            continue

        if not _is_valid_answer(boxed):
            skipped_bad_answer += 1
            continue

        seen.add(fp)
        records.append(_make_record(
            idx=counter,
            source="numina",
            problem=problem,
            ground_truth=boxed,
            boxed_answer=boxed,
            competition=source,
        ))
        counter += 1

    print(f"[INFO] NuminaMath: kept {len(records)} | "
          f"skipped source={skipped_source} dup={skipped_dup} "
          f"no_answer={skipped_no_answer} bad_answer={skipped_bad_answer}")
    return records


# ---------------------------------------------------------------------------
# Source 2: AIME 1983-2024
# ---------------------------------------------------------------------------

def pull_aime(max_per_source: int, seen: set) -> list:
    print("[INFO] Pulling qq8933/AIME_1983_2024 ...")
    hf_token = os.environ.get("HF_TOKEN")
    try:
        ds = load_dataset(
            "qq8933/AIME_1983_2024",
            split="train",
            trust_remote_code=True,
            token=hf_token,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load AIME archive: {e}")
        print("[INFO] Check the dataset ID at: https://huggingface.co/datasets?search=AIME")
        return []

    records = []
    counter = 0
    skipped_dup = skipped_no_answer = 0

    for row in ds:
        if len(records) >= max_per_source:
            break

        # Field names vary by dataset version — handle both casings
        problem = (row.get("Problem") or row.get("problem") or "").strip()

        # AIME answers are integers 000-999 stored as int or str.
        # Do NOT use `row.get("Answer") or row.get("answer")` — 0 is falsy and would be dropped.
        # Do NOT fall back to "solution" — that field is a full text explanation.
        if "Answer" in row:
            raw_answer = row["Answer"]
        elif "answer" in row:
            raw_answer = row["answer"]
        else:
            skipped_no_answer += 1
            continue

        if raw_answer is None:
            skipped_no_answer += 1
            continue

        # AIME answers are integers 0-999, but may be stored as int, "42", or "42.0"
        try:
            answer = str(int(float(str(raw_answer).strip())))
        except (ValueError, TypeError, OverflowError):
            # Unexpected format — skip rather than corrupt ground truth
            skipped_no_answer += 1
            continue

        year = row.get("Year") or row.get("year") or ""
        contest = row.get("Contest") or row.get("contest") or "AIME"

        if not problem:
            skipped_no_answer += 1
            continue

        fp = _fingerprint(problem)
        if fp in seen:
            skipped_dup += 1
            continue

        seen.add(fp)
        records.append(_make_record(
            idx=counter,
            source="aime_archive",
            problem=problem,
            ground_truth=answer,   # plain integer string, e.g. "042"
            boxed_answer=answer,   # Rottweiler compares against \boxed{} in model output
            competition=f"AIME {year} {contest}".strip(),
        ))
        counter += 1

    print(f"[INFO] AIME archive: kept {len(records)} | "
          f"skipped dup={skipped_dup} no_answer={skipped_no_answer}")
    return records


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(records: list) -> None:
    from collections import Counter
    total = len(records)
    sources = Counter(r["source"] for r in records)
    competitions = Counter(r["competition"] for r in records)

    print("\n" + "=" * 60)
    print("  RL POOL SUMMARY")
    print("=" * 60)
    print(f"  Total records  : {total}")
    print("  By source:")
    for k, v in sources.most_common():
        print(f"    {k}: {v}")
    print("  Top competitions:")
    for k, v in competitions.most_common(8):
        print(f"    {k}: {v}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chalk RL training pool")
    parser.add_argument("--max-per-source", type=int, default=5000,
                        help="Max records to pull per source (default: 5000)")
    parser.add_argument("--out", type=str, default=OUT_FILE,
                        help="Output JSONL path")
    args = parser.parse_args()

    seen = load_sft_fingerprints()

    all_records: list = []
    all_records += pull_numina(args.max_per_source, seen)
    all_records += pull_aime(args.max_per_source, seen)

    if not all_records:
        print("[ERROR] No records collected. Check network access and dataset names.")
        sys.exit(1)

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print_stats(all_records)
    print(f"[OK] Wrote {len(all_records)} records to {out_path}")
    print(f"[OK] Pass --data_path {out_path} to train_baby_chalk_rl.py")


if __name__ == "__main__":
    main()
