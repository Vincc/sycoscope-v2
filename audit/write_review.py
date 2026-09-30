"""Write reports/audit_existing_methods/REVIEW.md and review.html from the audit's result files.

Every number comes from coverage.csv, parity.csv, vennemeyer_fig8b.csv, controls.csv, length_confound.csv,
design_choices.jsonl, environment.json and regression.json in --dir.

Run from the repo root: python -m audit.write_review --dir reports/audit_existing_methods
"""
import argparse
import html
import json
from pathlib import Path

from audit.analyze import CELLS, read_csv, short_label
from utils.io import check_counts, read_json, read_jsonl, write_meta

NOT_IN_TASK = [
    ("CLiF", "code not released."),
    ("Baez et al.", "repository unreachable."),
    ("Cheng et al.", "labels need GPT-4o (OSF data not checked for labels in this pass)."),
    ("Goodfire SAE features", "no sycophancy features published."),
    ("Wang et al.", "not a detector."),
    ("Papadatos & Freedman", "they probe a reward model."),
    ("Beacon", "no code."),
    ("Steering and causal tests", "out of scope for this pass."),
    ("Other models", "Llama-3.1-8B-Instruct only."),
    ("New detection methods", "out of scope."),
    ("Any LLM API call", "none made; Persona Vectors held at the judge step."),
]
REF_NAMES = {"universal": "Contrastive universal (P00)", "matched_cell": "Contrastive matched-cell pair",
             "max_taxonomy_posthoc": "Contrastive max over taxonomy pairs (post hoc, optimistic)"}


def fnum(x, d=3):
    if x in (None, "", "None"):
        return ""
    try:
        return f"{float(x):.{d}f}"
    except ValueError:
        return str(x)


def detector_label(r: dict) -> str:
    return short_label(r).removeprefix("REF ") if r["role"] == "detector" else REF_NAMES[r["role"]]


def row_key(r: dict) -> str:
    return r["detector"] if r["role"] == "detector" else r["role"]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    d = args.dir
    coverage, parity = read_csv(d / "coverage.csv"), read_csv(d / "parity.csv")
    fig8b, controls, length = read_csv(d / "vennemeyer_fig8b.csv"), read_csv(d / "controls.csv"), read_csv(d / "length_confound.csv")
    choices, env, reg = read_jsonl(d / "design_choices.jsonl"), read_json(d / "environment.json"), read_json(d / "regression.json")

    benches = sorted({(r["benchmark"], r["display"], r["benchmark_cell"], r["cell_source"], r["target_label"], r["n"]) for r in coverage},
                     key=lambda b: (CELLS.index(b[2]), b[1]))
    keys = list(dict.fromkeys(row_key(r) for r in coverage))
    cov = {(row_key(r), r["benchmark"]): r for r in coverage}
    first = {k: cov[(k, benches[0][0])] for k in keys}

    # ---- Markdown ----
    md = [f"# Sycophancy detector audit\n", "Branch `audit/existing-methods`. Model `meta-llama/Llama-3.1-8B-Instruct`. "
          "All numbers are generated from the CSV files in this directory by `audit/write_review.py`.\n"]
    md.append("## 1. Summary\n")
    md.append(md_table(["Detector", "Status", "Reason"], [
        ["CAA sycophancy vector", "reproduced", "difference-in-means on the 1,000 A/B items, Llama-3.1 chat template"],
        ["Vennemeyer SyA, GA, SyPr", "reproduced, parity mismatch", "literal 'last' pooling selects BOS on Llama-3.1; two reconstructed poolings scored (user decision)"],
        ["Genadi head probes", "reproduced", "extension/ recipe on TruthfulQA pushback dialogues; best head + LR on top-16 [our addition]"],
        ["Pandey shared-circuit direction", "reproduced (code-faithful position)", "residual DIM at L27 and top-15 heads; LR on heads [our addition]"],
        ["Persona Vectors (sycophantic)", "held at judge step", f"needs {env['persona_judge_calls']} GPT-4.1-mini calls; rollouts and activations cached"],
    ]))
    md.append("\nNot in this task:\n")
    md += [f"- {n}: {t}" for n, t in NOT_IN_TASK]
    md.append("\nEnvironment: " + "; ".join(f"{k} {v}" for k, v in env["environment"].items()) + "\n")

    md.append("## 2. Parity (held-out AUROC on each method's own source data)\n")
    md.append(md_table(["Method", "Unit", "Variant", "Layer", "Our AUROC [95% CI]", "n", "Paper value", "Paper reference", "Verdict"], [
        [p["method"], p["unit"], p["variant"], p["layer"], f"{fnum(p['auroc'])} [{fnum(p['ci_lo'])}, {fnum(p['ci_hi'])}]" if p["auroc"] else "held",
         p["n_heldout"], fnum(p["paper_value"], 2), p["paper_reference"], p["verdict"]] for p in parity]))
    md.append("\nVennemeyer, SIMPLE MATH, plain text: our test AUROC by layer against Fig. 8b (read by eye, about ±0.02).\n")
    md.append(md_table(["Pooling", "Unit", "Layer", "Paper", "Ours", "Diff"],
                       [[r["pool"], r["unit"], r["layer"], r["paper_fig8b"], fnum(r["ours_test_auroc"], 2), f"{float(r['diff']):+.2f}"] for r in fig8b]))

    md.append("\n## 3. Coverage (AUROC [95% CI] on the balanced judged benchmarks)\n")
    md.append("Columns grouped by the benchmark's taxonomy cell (* = cell inferred, not in the SPEC table). "
              "Native = the scored position equals where the method reads activations; ≈ approximate; ✗ not native. "
              "Heatmap: `coverage_heatmap.png`.\n")
    header = ["Detector", "Built on", "Native"] + [f"{b[1]}{'*' if b[3] == 'inferred' else ''} ({b[2]}; {b[4]}, n={b[5]})" for b in benches]
    nat = {"yes": "✓", "approx": "≈", "no": "✗"}
    md.append(md_table(header, [[detector_label(first[k]), first[k]["built_on"], nat[first[k]["matches_native"]]] +
                                [f"{fnum(cov[(k, b[0])]['auroc'], 2)} [{fnum(cov[(k, b[0])]['ci_lo'], 2)}, {fnum(cov[(k, b[0])]['ci_hi'], 2)}]"
                                 for b in benches] for k in keys]))
    md.append("\nMatched-cell and post-hoc reference pairs per benchmark: " + "; ".join(
        f"{b[1]}: {cov[('matched_cell', b[0])]['unit']} / {cov[('max_taxonomy_posthoc', b[0])]['unit']}" for b in benches) + "\n")

    md.append("## 4. Controls and length confound\n")
    md.append("Controls: AUROC separating the two sides of each held-out control pair (0.5 = the detector ignores that correlate).\n")
    cpairs = sorted({(int(r["control_pair"]), r["control_cell"]) for r in controls})
    cc = {(r["detector"], int(r["control_pair"])): r for r in controls}
    cdets = list(dict.fromkeys(r["detector"] for r in controls))
    cfirst = {r["detector"]: r for r in controls}
    clabel = {k: detector_label(first[k]) if k in first else f"REF {cfirst[k]['unit']}" for k in cdets}
    md.append(md_table(["Detector", "Layer", "Position"] + [f"P{p:02d} {c}" for p, c in cpairs],
                       [[clabel[k], cfirst[k]["layer"], cfirst[k]["position"]] +
                        [f"{fnum(cc[(k, p)]['auroc'], 2)} [{fnum(cc[(k, p)]['ci_lo'], 2)}, {fnum(cc[(k, p)]['ci_hi'], 2)}]" for p, _ in cpairs]
                        for k in cdets]))
    md.append("\nLength confound, summarised per detector over the 15 benchmarks (full rows in `length_confound.csv`): "
              "Spearman ρ between detector score and n_response_tokens, and the spread of AUROC across length terciles.\n")
    summ, label_rho = length_summary(length)
    md.append(md_table(["Detector / role", "median ρ(score, length)", "max |ρ| (benchmark)", "median tercile AUROC range",
                        "max tercile range (benchmark)"], [[s["name"], fnum(s["med_rho"], 2), f"{fnum(s['max_rho'], 2)} ({s['max_rho_b']})",
                        fnum(s["med_range"], 2), f"{fnum(s['max_range'], 2)} ({s['max_range_b']})"] for s in summ]))
    md.append("\nρ(label, length) per benchmark (same for every detector):\n")
    md.append(md_table(["Benchmark", "n", "ρ(label, length)"], [[b, n, fnum(v, 2)] for b, n, v in label_rho]))
    md.append("\n## 5. Design choices, assumptions and inferences\n")
    md += [f"- *{c['tag']}* ({c['topic']}): {c['text']}" for c in choices]

    md.append("\n## 6. Regression evidence\n")
    md += [f"- {line}" for line in reg["lines"]]
    (d / "REVIEW.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- HTML ----
    data = {"benches": [{"id": b[0], "display": b[1], "cell": b[2], "inferred": b[3] == "inferred", "label": b[4], "n": int(b[5])} for b in benches],
            "rows": [{"key": k, "label": detector_label(first[k]), "role": first[k]["role"], "method": first[k]["audit_method"].removesuffix("_diag"),
                      "position": first[k]["position"], "native": first[k]["matches_native"], "built_on": first[k]["built_on"],
                      "cells": [[round(float(cov[(k, b[0])]["auroc"]), 4), round(float(cov[(k, b[0])]["ci_lo"]), 4),
                                 round(float(cov[(k, b[0])]["ci_hi"]), 4), cov[(k, b[0])]["unit"]] for b in benches]} for k in keys]}
    template = (Path(__file__).parent / "review_template.html").read_text(encoding="utf-8")
    page = template.replace("/*DATA*/null", json.dumps(data))
    page = page.replace("<!--SUMMARY-->", html_summary(env)).replace("<!--PARITY-->", html_parity(parity, fig8b))
    page = page.replace("<!--CONTROLS-->", html_controls(controls, cpairs, cdets, cfirst, cc, clabel))
    page = page.replace("<!--LENGTH-->", html_length(length)).replace("<!--CHOICES-->", html_choices(choices))
    page = page.replace("<!--REGRESSION-->", "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in reg["lines"]) + "</ul>")
    (d / "review.html").write_text(page, encoding="utf-8")
    inputs = [d / n for n in ("coverage.csv", "parity.csv", "vennemeyer_fig8b.csv", "controls.csv", "length_confound.csv",
                              "design_choices.jsonl", "environment.json", "regression.json")]
    for name in ("REVIEW.md", "review.html"):
        write_meta(d / name, inputs, args, check_counts(len(coverage), {}, len(coverage), name), {"model": "meta-llama/Llama-3.1-8B-Instruct"})
    print(f"wrote {d / 'REVIEW.md'} and {d / 'review.html'}")


def esc(x) -> str:
    return html.escape(str(x))


def html_table(header, rows, cls="") -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def html_summary(env) -> str:
    rows = [["CAA sycophancy vector", '<span class="pill ok">reproduced</span>', "Difference-in-means on the 1,000 A/B items, Llama-3.1 chat template."],
            ["Vennemeyer SyA, GA, SyPr", '<span class="pill warn">parity mismatch</span>', "The literal 'last' pooling selects BOS on Llama-3.1. Two reconstructed poolings are scored (your decision)."],
            ["Genadi head probes", '<span class="pill ok">reproduced</span>', "extension/ recipe on TruthfulQA pushback dialogues. Best head, plus LR on the top 16 heads (our addition)."],
            ["Pandey shared circuit", '<span class="pill ok">reproduced</span>', "Code-faithful position. Residual DIM at L27 and the top 15 heads; LR on the heads (our addition)."],
            ["Persona Vectors", '<span class="pill held">held</span>', f"Stopped at the judge step: {env['persona_judge_calls']} GPT-4.1-mini calls needed."]]
    t = html_table(["Detector", "Status", "Reason"], [[esc(a), b, esc(c)] for a, b, c in rows])
    nt = "<ul class='compact'>" + "".join(f"<li><b>{esc(n)}</b>: {esc(x)}</li>" for n, x in NOT_IN_TASK) + "</ul>"
    ev = "<dl class='env'>" + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in env["environment"].items()) + "</dl>"
    return t + "<h3>Not in this task</h3>" + nt + "<h3>Environment</h3>" + ev


def html_parity(parity, fig8b) -> str:
    rows = []
    for p in parity:
        v = p["verdict"]
        cls = "warn" if v.startswith("mismatch") else "held" if v.startswith("held") else "ok" if v.startswith("match") else "muted"
        auc = f"{fnum(p['auroc'])} <span class='ci'>[{fnum(p['ci_lo'])}, {fnum(p['ci_hi'])}]</span>" if p["auroc"] else "held"
        rows.append([esc(p["method"]), esc(p["unit"]), esc(p["variant"]), esc(p["layer"]), auc, esc(p["n_heldout"]),
                     esc(fnum(p["paper_value"], 2)), esc(p["paper_reference"]), f"<span class='pill {cls}'>{esc(v)}</span>"])
    t = html_table(["Method", "Unit", "Variant", "Layer", "Our AUROC [95% CI]", "n", "Paper", "Reference", "Verdict"], rows, "num")
    f = html_table(["Pooling", "Unit", "Layer", "Paper (Fig. 8b)", "Ours", "Diff"],
                   [[esc(r["pool"]), esc(r["unit"]), esc(r["layer"]), esc(r["paper_fig8b"]), fnum(r["ours_test_auroc"], 2),
                     f"<span class='{'bad' if abs(float(r['diff'])) > 0.05 else ''}'>{float(r['diff']):+.2f}</span>"] for r in fig8b], "num")
    return t + "<h3>Vennemeyer, SIMPLE MATH, plain text, by layer</h3><p class='note'>Fig. 8b values read by eye (about ±0.02). Differences above 0.05 are marked.</p>" + f


def html_controls(controls, cpairs, cdets, cfirst, cc, clabel) -> str:
    rows = []
    for k in cdets:
        r = cfirst[k]
        rows.append([esc(clabel[k]), esc(r["layer"]), esc(r["position"])] +
                    [f"{fnum(cc[(k, p)]['auroc'], 2)} <span class='ci'>[{fnum(cc[(k, p)]['ci_lo'], 2)}, {fnum(cc[(k, p)]['ci_hi'], 2)}]</span>" for p, _ in cpairs])
    return html_table(["Detector", "Layer", "Position"] + [f"P{p:02d} {c}" for p, c in cpairs], rows, "num")


def length_summary(length):
    import numpy as np

    by = {}
    for r in length:
        name = short_label(r) if r["role"] == "detector" else f"REF {r['unit']} ({r['role'].replace('_', ' ')})"
        by.setdefault(name, []).append(r)
    out = []
    for name, rs in by.items():
        rho = [(float(r["spearman_score_vs_n_response_tokens"]), r["display"]) for r in rs]
        rng = []
        for r in rs:
            t = [float(r[f"t{k}_auroc"]) for k in (1, 2, 3) if r[f"t{k}_auroc"] not in ("", "None")]
            rng.append((max(t) - min(t), r["display"]))
        mr, mrb = max(rho, key=lambda x: abs(x[0]))
        mg, mgb = max(rng)
        out.append({"name": name, "med_rho": float(np.median([x for x, _ in rho])), "max_rho": mr, "max_rho_b": mrb,
                    "med_range": float(np.median([x for x, _ in rng])), "max_range": mg, "max_range_b": mgb})
    seen, label_rho = set(), []
    for r in length:
        if r["display"] not in seen:
            seen.add(r["display"])
            label_rho.append((r["display"], r["n"], r["spearman_label_vs_n_response_tokens"]))
    return out, label_rho


def html_length(length) -> str:
    summ, label_rho = length_summary(length)
    t = html_table(["Detector / role", "median ρ(score, length)", "max |ρ| (benchmark)", "median tercile AUROC range", "max range (benchmark)"],
                   [[esc(s["name"]), fnum(s["med_rho"], 2), f"{fnum(s['max_rho'], 2)} <span class='ci'>{esc(s['max_rho_b'])}</span>",
                     fnum(s["med_range"], 2), f"{fnum(s['max_range'], 2)} <span class='ci'>{esc(s['max_range_b'])}</span>"] for s in summ], "num")
    lr = html_table(["Benchmark", "n", "ρ(label, length)"], [[esc(b), esc(n), fnum(v, 2)] for b, n, v in label_rho], "num")
    full = [[esc(short_label(r) if r["role"] == "detector" else f"REF {r['unit']}"), esc(r["role"]),
             esc(r["display"]), esc(r["n"]), fnum(r["spearman_score_vs_n_response_tokens"], 2),
             fnum(r["t1_auroc"], 2), fnum(r["t2_auroc"], 2), fnum(r["t3_auroc"], 2)] for r in length]
    ft = html_table(["Detector", "Role", "Benchmark", "n", "ρ(score, length)", "AUROC T1 (short)", "T2", "T3 (long)"], full, "num")
    return ("<p class='note'>Spearman ρ between detector score and response length (n_response_tokens) on each benchmark, and AUROC within "
            "length terciles; summarised per detector over the 15 benchmarks.</p>" + t +
            "<h3>Label vs length, per benchmark</h3>" + lr + f"<details><summary>All {len(length)} rows</summary>{ft}</details>")


def html_choices(choices) -> str:
    return "<ul class='choices'>" + "".join(
        f"<li><span class='tag tag-{c['tag'].replace(' ', '-')}'>{esc(c['tag'])}</span> <b>{esc(c['topic'])}</b>: {esc(c['text'])}</li>"
        for c in choices) + "</ul>"


if __name__ == "__main__":
    main()
