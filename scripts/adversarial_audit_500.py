#!/usr/bin/env python3
"""
adversarial_audit_500.py

Exhaustive adversarial audit of data quality and template diversity in:
/home/kriday/epoch_website/t4-cuda/data/chalk_seeds_500.jsonl
"""

import json
import re
import math
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer

DATASET_PATH = Path("/home/kriday/epoch_website/t4-cuda/data/chalk_seeds_500.jsonl")
TOKENIZER_PATH = "/home/kriday/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e"

TAGS = ["explore", "conjecture", "test_edge_cases", "lemma_isolate", "formal_proof"]

def extract_tag_content(trace: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, trace, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def get_first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    first_line = text.split("\n")[0].strip()
    m = re.match(r"^(.*?[.?!:])(?:\s|$)", first_line)
    if m:
        return m.group(1).strip()
    return first_line

def get_first_n_words(text: str, n: int = 8) -> str:
    words = text.split()
    return " ".join(words[:n])

def compute_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in counter.values():
        p = cnt / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def main():
    print(f"Loading dataset: {DATASET_PATH}")
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n_records = len(records)
    print(f"Loaded {n_records} records.")

    print(f"Loading tokenizer: {TOKENIZER_PATH}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    tag_contents = {tag: [] for tag in TAGS}
    tag_first_sentences = {tag: Counter() for tag in TAGS}
    tag_first_8words = {tag: Counter() for tag in TAGS}
    tag_full_exact_body = {tag: Counter() for tag in TAGS}
    tag_template_abstracted = {tag: Counter() for tag in TAGS}

    track_counts = Counter()
    disciplines = Counter()
    subfields = Counter()
    sources = Counter()

    proof_token_lengths = []
    trace_token_lengths = []
    problem_token_lengths = []
    tag_token_lengths = {tag: [] for tag in TAGS}

    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()

    tag_sequences = Counter()

    for idx, r in enumerate(records):
        source = r.get("theorem_source", "")
        if "MATH" in source:
            track = "Track A (MATH Olympiad)"
        else:
            track = "Track B (Advanced Pure Math)"
        track_counts[track] += 1

        disciplines[r.get("discipline", "Unknown")] += 1
        subfields[r.get("subfield", "Unknown")] += 1
        sources[source] += 1

        prob = r["problem"]
        trace = r["trace"]
        gt = r.get("boxed_answer", r.get("ground_truth", ""))

        prob_toks = tok.encode(prob)
        trace_toks = tok.encode(trace)
        problem_token_lengths.append(len(prob_toks))
        trace_token_lengths.append(len(trace_toks))

        found_tags = re.findall(r"<([a-z_]+)>", trace)
        tag_sequences[tuple(found_tags)] += 1

        words = re.findall(r"\b\w+\b", trace.lower())
        for w in words:
            unigrams[w] += 1
        for i in range(len(words) - 1):
            bigrams[(words[i], words[i+1])] += 1
        for i in range(len(words) - 2):
            trigrams[(words[i], words[i+1], words[i+2])] += 1

        for tag in TAGS:
            c = extract_tag_content(trace, tag)
            tag_contents[tag].append(c)
            c_toks = len(tok.encode(c)) if c else 0
            tag_token_lengths[tag].append(c_toks)

            if tag == "formal_proof":
                proof_token_lengths.append(c_toks)

            fs = get_first_sentence(c)
            tag_first_sentences[tag][fs] += 1

            f8 = get_first_n_words(c, 8)
            tag_first_8words[tag][f8] += 1

            tag_full_exact_body[tag][c] += 1

            c_abs = c
            if gt:
                c_abs = c_abs.replace(gt, "<ANS>")
            c_abs = re.sub(r"\b\d+\b", "<NUM>", c_abs)
            tag_template_abstracted[tag][c_abs] += 1

    print("\n" + "=" * 80)
    print("AUDIT SECTION 1: DATASET COMPOSITION & SPLIT")
    print("=" * 80)
    print(f"Total Certified Records: {n_records}")
    for trk, cnt in track_counts.items():
        print(f"  {trk}: {cnt} ({cnt/n_records*100:.2f}%)")

    print("\nDisciplines Breakdown:")
    for d, cnt in disciplines.most_common():
        print(f"  {d:38s}: {cnt:4d} ({cnt/n_records*100:5.2f}%)")

    print("\nSubfields Breakdown:")
    for sf, cnt in subfields.most_common():
        print(f"  {sf:38s}: {cnt:4d} ({cnt/n_records*100:5.2f}%)")

    print("\nTag Sequence Compliance:")
    for seq, cnt in tag_sequences.items():
        print(f"  Sequence {list(seq)}: {cnt} ({cnt/n_records*100:.2f}%)")

    print("\n" + "=" * 80)
    print("AUDIT SECTION 2: TEMPLATE DIVERSITY VS BOILERPLATE ACROSS TAGS")
    print("=" * 80)

    for tag in TAGS:
        fs_counter = tag_first_sentences[tag]
        f8_counter = tag_first_8words[tag]
        abs_counter = tag_template_abstracted[tag]
        exact_counter = tag_full_exact_body[tag]
        unique_openings = len(fs_counter)
        unique_f8 = len(f8_counter)
        unique_skeletons = len(abs_counter)
        unique_exact = len(exact_counter)
        
        top1_fs, top1_fs_cnt = fs_counter.most_common(1)[0]
        top5_fs = fs_counter.most_common(5)
        top5_fs_pct = sum(c for _, c in top5_fs) / n_records * 100

        top1_abs, top1_abs_cnt = abs_counter.most_common(1)[0]

        print(f"\n--- TAG: <{tag}> ---")
        print(f"  Unique Opening Sentences: {unique_openings}")
        print(f"  Unique First-8-Words:     {unique_f8}")
        print(f"  Unique Exact Full Bodies: {unique_exact}")
        print(f"  Unique Abstract Skeletons (Numbers/<ANS> stripped): {unique_skeletons}")
        print(f"  Top-1 Opening Sentence:   '{top1_fs}' -> {top1_fs_cnt}/{n_records} ({top1_fs_cnt/n_records*100:.2f}%)")
        print(f"  Top-5 Opening Cumulative: {top5_fs_pct:.2f}%")
        print("  Top 5 Opening Sentences:")
        for rank, (sent, c) in enumerate(top5_fs, 1):
            print(f"    [{rank}] ({c:3d} | {c/n_records*100:5.2f}%) {sent[:90]}")
        print(f"  Top-1 Abstracted Skeleton Frequency: {top1_abs_cnt}/{n_records} ({top1_abs_cnt/n_records*100:.2f}%)")
        if top1_abs_cnt > 5:
            print("  Top-1 Skeleton Preview:")
            print("    " + repr(top1_abs)[:160])

    print("\n" + "=" * 80)
    print("AUDIT SECTION 3: FORMAL PROOF SUBSTANCE & TOKEN LENGTHS")
    print("=" * 80)

    def stats(arr):
        s = sorted(arr)
        n = len(s)
        mean_v = sum(s) / n
        std_v = math.sqrt(sum((x - mean_v)**2 for x in s) / n)
        return {
            "min": s[0],
            "p10": s[int(0.10 * n)],
            "p25": s[int(0.25 * n)],
            "median": s[int(0.50 * n)],
            "p75": s[int(0.75 * n)],
            "p90": s[int(0.90 * n)],
            "max": s[-1],
            "mean": mean_v,
            "std": std_v
        }

    st_proof = stats(proof_token_lengths)
    st_trace = stats(trace_token_lengths)
    st_prob = stats(problem_token_lengths)

    print(f"Problem Token Length: mean={st_prob['mean']:.1f}, median={st_prob['median']}, min={st_prob['min']}, max={st_prob['max']}")
    print(f"Trace Token Length:   mean={st_trace['mean']:.1f}, median={st_trace['median']}, min={st_trace['min']}, max={st_trace['max']}")
    print(f"Proof Token Length:   mean={st_proof['mean']:.1f}, median={st_proof['median']}, min={st_proof['min']}, max={st_proof['max']}, p10={st_proof['p10']}, p90={st_proof['p90']}, std={st_proof['std']:.1f}")

    print("\nPer-Tag Token Length Statistics:")
    for tag in TAGS:
        st = stats(tag_token_lengths[tag])
        print(f"  <{tag:16s}>: mean={st['mean']:5.1f} | std={st['std']:5.1f} | min={st['min']:3d} | p25={st['p25']:3d} | median={st['median']:3d} | p75={st['p75']:3d} | p90={st['p90']:3d} | max={st['max']:4d}")

    equation_count = sum(1 for c in tag_contents["formal_proof"] if "$" in c or "\\(" in c)
    steps_count = sum(1 for c in tag_contents["formal_proof"] if any(step_kw in c.lower() for step_kw in ["since", "therefore", "hence", "we have", "because", "implies", "thus"]))
    short_proofs = sum(1 for l in proof_token_lengths if l < 40)
    very_long_proofs = sum(1 for l in proof_token_lengths if l > 400)

    print(f"\nProof Substance Diagnostics:")
    print(f"  Proofs with LaTeX math expressions: {equation_count}/{n_records} ({equation_count/n_records*100:.2f}%)")
    print(f"  Proofs with explicit reasoning step connectors: {steps_count}/{n_records} ({steps_count/n_records*100:.2f}%)")
    print(f"  Proofs with < 40 tokens (potentially too brief): {short_proofs}/{n_records} ({short_proofs/n_records*100:.2f}%)")
    print(f"  Proofs with > 400 tokens: {very_long_proofs}/{n_records} ({very_long_proofs/n_records*100:.2f}%)")

    print("\n" + "=" * 80)
    print("AUDIT SECTION 4: LEXICAL DIVERSITY & INFORMATION THEORY")
    print("=" * 80)
    total_words = sum(unigrams.values())
    unique_words = len(unigrams)
    ttr = unique_words / total_words if total_words else 0
    total_bigrams = sum(bigrams.values())
    unique_bigrams = len(bigrams)
    total_trigrams = sum(trigrams.values())
    unique_trigrams = len(trigrams)

    print(f"Total Trace Words:         {total_words:,}")
    print(f"Distinct 1-grams (Words):  {unique_words:,}")
    print(f"Type-Token Ratio (TTR):    {ttr:.4f}")
    print(f"Distinct 2-grams:          {unique_bigrams:,} (total {total_bigrams:,}, ratio={unique_bigrams/total_bigrams:.4f})")
    print(f"Distinct 3-grams:          {unique_trigrams:,} (total {total_trigrams:,}, ratio={unique_trigrams/total_trigrams:.4f})")

    print("\nTag Shannon Entropy (Word-level):")
    for tag in TAGS:
        words_tag = Counter()
        for c in tag_contents[tag]:
            for w in re.findall(r"\b\w+\b", c.lower()):
                words_tag[w] += 1
        ent = compute_entropy(words_tag)
        print(f"  <{tag:16s}> Word Entropy: {ent:.2f} bits | Unique Words: {len(words_tag):4d} | Total Words: {sum(words_tag.values()):6d}")

if __name__ == "__main__":
    main()
