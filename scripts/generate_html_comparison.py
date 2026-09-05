#!/usr/bin/env python3
import json
import os
import html

def build_comparison_html(
    bench_path="data/external_math_eval.json",
    sft_path="results/external_eval_results.json",
    base_path="results/base_eval_results.json",
    output_html="reports/sft_vs_base_benchmark.html"
):
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    with open(bench_path, "r", encoding="utf-8") as f:
        bench = json.load(f)
    with open(sft_path, "r", encoding="utf-8") as f:
        sft = json.load(f)
    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    problems = []
    
    for p, s, b in zip(bench["contest_problems"], sft["contest_metrics"]["results"], base["contest_metrics"]["results"]):
        problems.append({
            "type": "contest",
            "id": p["id"],
            "competition": p.get("competition", "Contest"),
            "year": p.get("year", ""),
            "discipline": p.get("discipline", "General Math"),
            "problem": p["problem"],
            "ground_truth": p["ground_truth"],
            "sft_pass": s["pass_at_1"],
            "sft_pred": s["rollouts"][0]["pred_boxed"],
            "sft_toks": s["rollouts"][0]["num_tokens"],
            "sft_tok_s": s["rollouts"][0]["tok_s"],
            "sft_tags": s["rollouts"][0]["adherence"]["tags"],
            "sft_snippet": s["rollouts"][0]["completion_snippet"],
            "base_pass": b["pass_at_1"],
            "base_pred": b["rollouts"][0]["pred_boxed"],
            "base_toks": b["rollouts"][0]["num_tokens"],
            "base_tok_s": b["rollouts"][0]["tok_s"],
            "base_tags": b["rollouts"][0]["adherence"]["tags"],
            "base_snippet": b["rollouts"][0]["completion_snippet"]
        })

    for p, s, b in zip(bench["degradation_checks"], sft["degradation_metrics"]["results"], base["degradation_metrics"]["results"]):
        problems.append({
            "type": "degradation",
            "id": p["id"],
            "competition": "Sanity",
            "year": "",
            "discipline": "Basic Math",
            "problem": p["problem"],
            "ground_truth": p["ground_truth"],
            "sft_pass": s["pass_at_1"],
            "sft_pred": s["rollouts"][0]["pred_boxed"],
            "sft_toks": s["rollouts"][0]["num_tokens"],
            "sft_tok_s": s["rollouts"][0]["tok_s"],
            "sft_tags": s["rollouts"][0]["adherence"]["tags"],
            "sft_snippet": s["rollouts"][0]["completion_snippet"],
            "base_pass": b["pass_at_1"],
            "base_pred": b["rollouts"][0]["pred_boxed"],
            "base_toks": b["rollouts"][0]["num_tokens"],
            "base_tok_s": b["rollouts"][0]["tok_s"],
            "base_tags": b["rollouts"][0]["adherence"]["tags"],
            "base_snippet": b["rollouts"][0]["completion_snippet"]
        })

    cards_html = []
    for idx, p in enumerate(problems, 1):
        s_badge_class = "badge-pass" if p["sft_pass"] else "badge-fail"
        s_badge_text = "PASS" if p["sft_pass"] else "FAIL"
        b_badge_class = "badge-pass" if p["base_pass"] else "badge-fail"
        b_badge_text = "PASS" if p["base_pass"] else "FAIL"

        s_tags_html = "".join(
            f\x27<span class="tag-pill {"tag-active" if val else "tag-inactive"}">{tag}</span>\x27
            for tag, val in p["sft_tags"].items()
        )
        b_tags_html = \x27<span class="tag-pill tag-inactive">0 tags generated</span>\x27

        diff_class = ""
        if p["sft_pass"] and not p["base_pass"]:
            diff_class = "sft-win"
        elif not p["sft_pass"] and p["base_pass"]:
            diff_class = "base-win"
        elif p["sft_pass"] and p["base_pass"]:
            diff_class = "both-pass"
        else:
            diff_class = "both-fail"

        prob_type_badge = "Contest Problem" if p["type"] == "contest" else "Sanity Check"
        prob_type_class = "type-contest" if p["type"] == "contest" else "type-sanity"

        card = f"""
        <div class="problem-card {p[\x27type\x27]} {diff_class}" data-id="{p[\x27id\x27]}">
            <div class="card-header">
                <div class="meta-left">
                    <span class="index-num">#{idx:02d}</span>
                    <span class="prob-id">{html.escape(p[\x27id\x27])}</span>
                    <span class="type-pill {prob_type_class}">{prob_type_badge}</span>
                    <span class="disc-pill">{html.escape(p[\x27discipline\x27])}</span>
                </div>
                <div class="meta-right">
                    <span class="gt-label">Gold Answer:</span>
                    <span class="gt-value">\\boxed{{{html.escape(str(p[\x27ground_truth\x27]))}}}</span>
                </div>
            </div>

            <div class="problem-statement">
                <div class="section-title">Problem Statement</div>
                <p>{html.escape(p[\x27problem\x27])}</p>
            </div>

            <div class="comparison-grid">
                <div class="model-col sft-col">
                    <div class="col-head">
                        <div class="col-title">
                            <strong>Cold-Start SFT LoRA</strong>
                            <span class="col-sub">605-seed checkpoint</span>
                        </div>
                        <span class="status-badge {s_badge_class}">{s_badge_text}</span>
                    </div>

                    <div class="answer-row">
                        <span class="ans-label">Boxed Answer:</span>
                        <span class="ans-value {s_badge_class}">
                            {html.escape(str(p[\x27sft_pred\x27])) if p[\x27sft_pred\x27] is not None else \x27<span class="none-val">None (truncated/missing)</span>\x27}
                        </span>
                    </div>

                    <div class="tags-container">
                        <span class="tags-label">5-Tag Reasoning:</span>
                        <div class="tags-list">{s_tags_html}</div>
                    </div>

                    <div class="metric-row">
                        <span>Tokens: <strong>{p[\x27sft_toks\x27]}</strong></span>
                        <span>Speed: <strong>{p[\x27sft_tok_s\x27]:.1f} tok/s</strong></span>
                    </div>

                    <div class="snippet-box">
                        <div class="snippet-label">Completion Preview:</div>
                        <pre><code>{html.escape(p[\x27sft_snippet\x27])}</code></pre>
                    </div>
                </div>

                <div class="model-col base-col">
                    <div class="col-head">
                        <div class="col-title">
                            <strong>Base Qwen2.5-Math-1.5B</strong>
                            <span class="col-sub">Un-tuned baseline</span>
                        </div>
                        <span class="status-badge {b_badge_class}">{b_badge_text}</span>
                    </div>

                    <div class="answer-row">
                        <span class="ans-label">Boxed Answer:</span>
                        <span class="ans-value {b_badge_class}">
                            {html.escape(str(p[\x27base_pred\x27])) if p[\x27base_pred\x27] is not None else \x27<span class="none-val">None (missing)</span>\x27}
                        </span>
                    </div>

                    <div class="tags-container">
                        <span class="tags-label">5-Tag Reasoning:</span>
                        <div class="tags-list">{b_tags_html}</div>
                    </div>

                    <div class="metric-row">
                        <span>Tokens: <strong>{p[\x27base_toks\x27]}</strong></span>
                        <span>Speed: <strong>{p[\x27base_tok_s\x27]:.1f} tok/s</strong></span>
                    </div>

                    <div class="snippet-box">
                        <div class="snippet-label">Completion Preview:</div>
                        <pre><code>{html.escape(p[\x27base_snippet\x27])}</code></pre>
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    cards_joined = "\n".join(cards_html)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chalk SFT vs Base Qwen2.5-Math-1.5B Head-to-Head</title>
    <style>
        :root {{
            --bg: #000000;
            --card-bg: #0a0a0a;
            --card-border: #1f1f1f;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --green: #22c55e;
            --green-bg: rgba(34, 197, 94, 0.1);
            --green-border: rgba(34, 197, 94, 0.3);
            --red: #ef4444;
            --red-bg: rgba(239, 68, 68, 0.1);
            --red-border: rgba(239, 68, 68, 0.3);
            --blue: #38bdf8;
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg);
            color: var(--text-primary);
            font-family: var(--font-sans);
            line-height: 1.5;
            padding: 2.5rem 1.5rem 6rem;
            -webkit-font-smoothing: antialiased;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
        }}

        .header-pre {{
            font-family: var(--font-mono);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--blue);
            margin-bottom: 0.5rem;
        }}

        h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
            color: #ffffff;
        }}

        .header-sub {{
            color: var(--text-secondary);
            font-size: 1.05rem;
        }}

        .scoreboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin-bottom: 2.5rem;
        }}

        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 1.25rem;
        }}

        .stat-title {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .stat-main {{
            display: flex;
            align-items: baseline;
            gap: 1.25rem;
        }}

        .stat-val {{
            font-size: 1.8rem;
            font-weight: 700;
            font-family: var(--font-mono);
        }}

        .stat-val.sft {{ color: var(--green); }}
        .stat-val.base {{ color: var(--blue); }}

        .controls-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 2rem;
            align-items: center;
            background: #080808;
            padding: 0.75rem;
            border-radius: 8px;
            border: 1px solid var(--card-border);
        }}

        .filter-label {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-right: 0.5rem;
            text-transform: uppercase;
        }}

        .filter-btn {{
            background: #141414;
            color: var(--text-secondary);
            border: 1px solid #282828;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-family: var(--font-mono);
            cursor: pointer;
            transition: background 0.15s, color 0.15s, border-color 0.15s;
        }}

        .filter-btn:hover {{
            background: #202020;
            color: #ffffff;
            border-color: #404040;
        }}

        .filter-btn.active {{
            background: #ffffff;
            color: #000000;
            font-weight: 600;
            border-color: #ffffff;
        }}

        .problem-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            margin-bottom: 1.75rem;
            padding: 1.5rem;
            transition: border-color 0.2s;
        }}

        .problem-card:hover {{
            border-color: #333333;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #161616;
        }}

        .meta-left {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            flex-wrap: wrap;
        }}

        .index-num {{
            font-family: var(--font-mono);
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .prob-id {{
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 0.95rem;
            color: #ffffff;
        }}

        .type-pill, .disc-pill {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: #151515;
            color: var(--text-secondary);
            border: 1px solid #252525;
        }}

        .type-pill.type-contest {{
            border-color: rgba(56, 189, 248, 0.4);
            color: var(--blue);
        }}

        .type-pill.type-sanity {{
            border-color: rgba(168, 85, 247, 0.4);
            color: #c084fc;
        }}

        .meta-right {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: var(--font-mono);
            font-size: 0.85rem;
        }}

        .gt-label {{
            color: var(--text-muted);
        }}

        .gt-value {{
            background: #111827;
            color: #60a5fa;
            border: 1px solid #1e3a8a;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }}

        .problem-statement {{
            background: #050505;
            border: 1px solid #141414;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1.25rem;
            font-size: 0.95rem;
            color: #e5e7eb;
        }}

        .section-title {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }}

        @media (max-width: 820px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .model-col {{
            background: #080808;
            border: 1px solid #1a1a1a;
            border-radius: 8px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}

        .model-col.sft-col {{
            border-color: #222222;
        }}

        .col-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #161616;
            padding-bottom: 0.6rem;
        }}

        .col-title strong {{
            display: block;
            font-size: 0.95rem;
            color: #ffffff;
        }}

        .col-sub {{
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }}

        .status-badge {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .badge-pass {{
            background: var(--green-bg);
            color: var(--green);
            border: 1px solid var(--green-border);
        }}

        .badge-fail {{
            background: var(--red-bg);
            color: var(--red);
            border: 1px solid var(--red-border);
        }}

        .answer-row {{
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            font-size: 0.9rem;
        }}

        .ans-label {{
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 0.8rem;
        }}

        .ans-value {{
            font-family: var(--font-mono);
            font-weight: 600;
        }}

        .ans-value.badge-pass {{
            color: var(--green);
        }}

        .ans-value.badge-fail {{
            color: var(--red);
        }}

        .none-val {{
            color: var(--text-muted);
            font-weight: 400;
        }}

        .tags-container {{
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }}

        .tags-label {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .tags-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }}

        .tag-pill {{
            font-family: var(--font-mono);
            font-size: 0.7rem;
            padding: 0.15rem 0.45rem;
            border-radius: 3px;
        }}

        .tag-pill.tag-active {{
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.4);
            font-weight: 600;
        }}

        .tag-pill.tag-inactive {{
            background: #121212;
            color: #555555;
            border: 1px solid #1c1c1c;
        }}

        .metric-row {{
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--text-secondary);
            border-top: 1px solid #141414;
            padding-top: 0.5rem;
        }}

        .metric-row strong {{
            color: #ffffff;
        }}

        .snippet-box {{
            background: #020202;
            border: 1px solid #161616;
            border-radius: 4px;
            padding: 0.6rem;
            margin-top: auto;
        }}

        .snippet-label {{
            font-family: var(--font-mono);
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }}

        pre {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: #d1d5db;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.4;
        }}

        .is-hidden {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-pre">Physical Tesla T4 Silicon Benchmark · External Evaluation</div>
            <h1>Chalk SFT (1.5B) vs Base Model</h1>
            <p class="header-sub">Held-out competition math (AIME 2023-2024 / AMC 12 2023) and arithmetic grounding sanity checks.</p>
        </header>

        <section class="scoreboard">
            <div class="stat-card">
                <div class="stat-title">Contest Pass@1 (16 Problems)</div>
                <div class="stat-main">
                    <div>
                        <div class="stat-val sft">25.0%</div>
                        <span style="font-size:0.75rem;color:var(--text-muted);">SFT (4/16)</span>
                    </div>
                    <div>
                        <div class="stat-val base">43.8%</div>
                        <span style="font-size:0.75rem;color:var(--text-muted);">Base (7/16)</span>
                    </div>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-title">Sanity / Grounding Pass (10 Problems)</div>
                <div class="stat-main">
                    <div>
                        <div class="stat-val sft">60.0%</div>
                        <span style="font-size:0.75rem;color:var(--text-muted);">SFT (6/10)</span>
                    </div>
                    <div>
                        <div class="stat-val" style="color:var(--red);">10.0%</div>
                        <span style="font-size:0.75rem;color:var(--text-muted);">Base (1/10)</span>
                    </div>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-title">5-Tag Reasoning Behavior</div>
                <div class="stat-main">
                    <div>
                        <div class="stat-val sft">Active</div>
                        <span style="font-size:0.75rem;color:var(--text-muted);">Explore/Conjecture</span>
                    </div>
                    <div>
                        <div class="stat-val base">0.0%</div>
                        <span style="font-size:0.75rem;color:var(--text-muted);">No tags generated</span>
                    </div>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-title">Tesla T4 Peak VRAM</div>
                <div class="stat-main">
                    <div>
                        <div class="stat-val" style="color:#ffffff;">3.2 GB</div>
                        <span style="font-size:0.75rem;color:var(--text-muted);">LoRA Eval VRAM</span>
                    </div>
                    <div>
                        <div class="stat-val" style="color:#ffffff;">3.0 GB</div>
                        <span style="font-size:0.75rem;color:var(--text-muted);">Base Eval VRAM</span>
                    </div>
                </div>
            </div>
        </section>

        <div class="controls-bar">
            <span class="filter-label">Filter:</span>
            <button class="filter-btn active" data-filter="all">All (26)</button>
            <button class="filter-btn" data-filter="contest">Contest (16)</button>
            <button class="filter-btn" data-filter="degradation">Sanity Checks (10)</button>
            <button class="filter-btn" data-filter="sft-win">SFT Won (5)</button>
            <button class="filter-btn" data-filter="base-win">Base Won (6)</button>
            <button class="filter-btn" data-filter="both-pass">Both Passed (4)</button>
            <button class="filter-btn" data-filter="both-fail">Both Failed (11)</button>
        </div>

        <main id="problem-list">
            {cards_joined}
        </main>
    </div>

    <script>
        const filterBtns = document.querySelectorAll(\x27.filter-btn\x27);
        const cards = document.querySelectorAll(\x27.problem-card\x27);

        filterBtns.forEach(btn => {{
            btn.addEventListener(\x27click\x27, () => {{
                filterBtns.forEach(b => b.classList.remove(\x27active\x27));
                btn.classList.add(\x27active\x27);

                const filter = btn.getAttribute(\x27data-filter\x27);

                cards.forEach(card => {{
                    if (filter === \x27all\x27) {{
                        card.classList.remove(\x27is-hidden\x27);
                    }} else if (filter === \x27contest\x27) {{
                        card.classList.toggle(\x27is-hidden\x27, !card.classList.contains(\x27contest\x27));
                    }} else if (filter === \x27degradation\x27) {{
                        card.classList.toggle(\x27is-hidden\x27, !card.classList.contains(\x27degradation\x27));
                    }} else if (filter === \x27sft-win\x27) {{
                        card.classList.toggle(\x27is-hidden\x27, !card.classList.contains(\x27sft-win\x27));
                    }} else if (filter === \x27base-win\x27) {{
                        card.classList.toggle(\x27is-hidden\x27, !card.classList.contains(\x27base-win\x27));
                    }} else if (filter === \x27both-pass\x27) {{
                        card.classList.toggle(\x27is-hidden\x27, !card.classList.contains(\x27both-pass\x27));
                    }} else if (filter === \x27both-fail\x27) {{
                        card.classList.toggle(\x27is-hidden\x27, !card.classList.contains(\x27both-fail\x27));
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Generated comparison HTML: {output_html}")
    return output_html

if __name__ == "__main__":
    build_comparison_html()
