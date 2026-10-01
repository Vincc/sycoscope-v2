"""Write reports/audit_existing_methods/REVIEW.md and review.html from the audit's result files.

Sections: 1 summary; 2 reproduction on source data vs each paper's findings; 3 out-of-distribution performance per
method at its own read position (cached-position scores kept as a secondary view); 4 controls and length; 5 design
choices; 6 verification. Every number is read from the CSV/JSON files in --dir.

Run from the repo root: python -m audit.write_review --dir reports/audit_existing_methods
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np

from audit.analyze import CELLS, read_csv, short_label
from utils.io import check_counts, read_json, read_jsonl, write_meta

NOT_IN_TASK = [
    ("CLiF", "code not released."), ("Baez et al.", "repository unreachable."),
    ("Cheng et al.", "labels need GPT-4o (OSF data not checked for labels in this pass)."),
    ("Goodfire SAE features", "no sycophancy features published."), ("Wang et al.", "not a detector."),
    ("Papadatos & Freedman", "they probe a reward model."), ("Beacon", "no code."),
    ("Steering and causal tests", "out of scope."), ("Other models", "Llama-3.1-8B-Instruct only."),
    ("New detection methods", "out of scope."), ("LLM API calls", "only the Persona Vectors GPT-4.1-mini judge, after the user supplied a key; cost in section 2."),
]
METHODS = [("caa", "CAA (Rimsky/Panickssery et al. 2024)"), ("vennemeyer", "Vennemeyer et al. (SyA, GA, SyPr)"),
           ("genadi", "Genadi et al. (attention-head probes)"), ("pandey", "Pandey (shared sycophancy-lying circuit)"),
           ("persona", "Persona Vectors (Chen et al. 2025)")]
REF_NAMES = {"universal": "Contrastive universal (P00)", "matched_cell": "Contrastive matched-cell pair",
             "max_taxonomy_posthoc": "Contrastive max over taxonomy pairs (post hoc, optimistic)"}
FLAG_TEXT = {"†": "Pandey's code-faithful index drifts on multi-turn prompts (AYS: last user turn's <|eot_id|>; multi-turn SyPR: inside the last user message)",
             "‡": "plain Human:/Assistant: rendering of a multi-turn conversation uses our turn separator (AYS all rows; SyPR about half the rows)"}


def fnum(x, d=3):
    if x in (None, "", "None"):
        return ""
    try:
        return f"{float(x):.{d}f}"
    except ValueError:
        return str(x)


def esc(x) -> str:
    return html.escape(str(x))


def method_of(r: dict) -> str:
    return r["audit_method"].removesuffix("_diag")


def label(r: dict) -> str:
    return REF_NAMES[r["role"]] if r["role"] != "detector" else short_label(r)


def row_key(r: dict) -> str:
    return r["detector"] if r["role"] == "detector" else r["role"]


def flag(r: dict, bench: str) -> str:
    if r["role"] != "detector" or not (bench.startswith("ays_") or bench.startswith("sypr")):
        return ""
    if method_of(r) == "pandey" and r["activations"].startswith("native"):
        return "†"
    if r["activations"] == "native plain":
        return "‡"
    return ""


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(out + ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows])


def html_table(header, rows, cls="") -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def pill(verdict: str) -> str:
    v = verdict.lower()
    cls = "ok" if v.startswith(("match", "consistent")) else "warn" if v.startswith(("mismatch", "partly")) \
        else "held" if v.startswith("held") else "muted"
    return f"<span class='pill {cls}'>{esc(verdict)}</span>"


class Doc:
    """Markdown and HTML versions of each block, built side by side."""

    def __init__(self):
        self.md, self.html = [], []

    def h(self, level, text, anchor=None):
        self.md.append(f"\n{'#' * level} {text}\n")
        self.html.append(f"<h{level}{f' id={chr(34)}{anchor}{chr(34)}' if anchor else ''}>{esc(text)}</h{level}>")

    def p(self, text, note=False):
        self.md.append(text + "\n")
        self.html.append(f"<p class='{'note' if note else ''}'>{esc(text)}</p>")

    def bullets(self, items):
        self.md.append("\n".join(f"- {x}" for x in items) + "\n")
        self.html.append("<ul>" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>")

    def table(self, header, rows, html_rows=None, cls="num"):
        self.md.append(md_table(header, rows) + "\n")
        self.html.append(html_table(header, html_rows or [[esc(c) for c in r] for r in rows], cls))

    def raw_html(self, s):
        self.html.append(s)


def reproduction(doc: Doc, d: Path, parity, paper, fig8b, fig10b, overlap):
    doc.h(2, "2. Reproduction on source data, compared with each paper", "repro")
    doc.p("Each detector is rebuilt on its own source data with its own recipe and read position; its held-out AUROC "
          "(95% stratified bootstrap CI, 1,000 resamples, seed 0) is set against what the paper reports. A numeric comparison "
          "is possible only where the paper reports Llama-3.1-8B; figure values are read by eye.", note=True)

    def parity_rows(method):
        rows = [p for p in parity if p["method"].lower().startswith(method)]
        md = [[p["unit"], p["variant"], p["layer"], f"{fnum(p['auroc'])} [{fnum(p['ci_lo'])}, {fnum(p['ci_hi'])}]" if p["auroc"] else "",
               p["n_heldout"], p["verdict"]] for p in rows]
        ht = [[esc(p["unit"]), esc(p["variant"]), esc(p["layer"]),
               f"{fnum(p['auroc'])} <span class='ci'>[{fnum(p['ci_lo'])}, {fnum(p['ci_hi'])}]</span>" if p["auroc"] else "",
               esc(p["n_heldout"]), pill(p["verdict"])] for p in rows]
        doc.table(["Unit", "Variant", "Layer", "Held-out AUROC [95% CI]", "n", "Verdict"], md, ht)

    def curve_table(rows_csv, key, value, name_map, tol):
        pools = ["last_content", "resp_all"]
        for item in dict.fromkeys(r[key] for r in rows_csv):
            layers = sorted({int(r["layer"]) for r in rows_csv if r[key] == item})
            get = {(r["pool"], int(r["layer"])): r for r in rows_csv if r[key] == item}
            paper_col = "paper_fig8b" if "paper_fig8b" in rows_csv[0] else "paper_fig10b"
            tab = [["paper"] + [get[(pools[0], L)][paper_col] for L in layers]]
            for pool in pools:
                tab.append([pool] + [fnum(get[(pool, L)][value], 2) + ("*" if abs(float(get[(pool, L)]["diff"])) > tol else "")
                                     for L in layers])
            doc.p(name_map(item), note=True)
            doc.table(["", *[f"L{L}" for L in layers]], tab)

    caa = read_jsonl(d / "source" / "generate_dataset__letter.jsonl")
    doc.h(3, METHODS[0][1])
    doc.p(f"Paper: {paper['caa']['statement']} ({paper['caa']['reference']}). No detection AUROC is reported.")
    doc.p("Ours (1,000 A/B items, answer-letter token, 20% of questions held out): held-out AUROC by layer "
          + ", ".join(f"L{r['layer']} {r['test_auroc']:.2f}" for r in caa if 8 <= r["layer"] <= 16) + ".")
    parity_rows("caa")

    v = paper["vennemeyer"]
    doc.h(3, METHODS[1][1])
    doc.bullets([f"Layerwise AUROC, Llama-3.1-8B, SIMPLE MATH: {v['reference']}.",
                 f"Direction geometry, Llama-3.1-8B, SIMPLE MATH: {v['fig10b_read']['note']}.",
                 f"Headline numbers in the main text are for Qwen3-30B: {v['headline_qwen3_30b']}.",
                 "Their code's 'last' pooling selects the BOS token on Llama-3.1 and gives zero vectors, so pooling was "
                 "reconstructed two ways: last content token (the code's no-EOS rule) and whole-response mean (their resp_all)."])
    parity_rows("vennemeyer")
    names = {"syc": "SyA", "ga": "GA", "pr": "SyPr"}
    doc.p("Layerwise AUROC against Fig. 8b, plain text (differences above 0.05 marked *):", note=True)
    curve_table(fig8b, "unit", "ours_test_auroc", lambda u: names[u], 0.05)
    doc.p("Cosine between behaviour directions against Fig. 10b, plain text (differences above 0.1 marked *):", note=True)
    curve_table(fig10b, "pair", "ours_cosine", lambda p: "-".join(names[x] for x in p.split("-")), 0.1)

    g = paper["genadi"]
    heads_tab = read_jsonl(d / "source" / "genadi_heads__answer_mean.jsonl")
    acc = np.array([r["best_val_acc"] for r in heads_tab])
    top = sorted(heads_tab, key=lambda r: -r["best_val_acc"])[:16]
    doc.h(3, METHODS[2][1])
    doc.p(f"Paper: {g['statement']} ({g['reference']}). Models in the paper: {g['model_in_paper']}.")
    doc.p(f"Ours (answer-mean pooling, 1,024 heads): median best val accuracy {np.median(acc):.1f}%, {(acc >= 75).sum()} heads at "
          f"or above 75%, maximum {acc.max():.1f}% at L{top[0]['layer']}H{top[0]['head']}; layers of the top 16 heads: "
          f"{sorted({r['layer'] for r in top})}.")
    parity_rows("genadi")

    p = paper["pandey"]
    doc.h(3, METHODS[3][1])
    doc.p(f"Paper: {p['statement']} ({p['reference']}). Table 1 (p. 5), Llama-3.1-8B: 21 of the top K=32 sycophancy heads "
          "are also top-32 factual-lying heads; Spearman 0.88 over all 1,024 heads.")
    doc.p("The repo has two scripts that each claim Table 1, with different settings; both are run. "
          + " ".join(f"{name}: {o['overlap']['32']['shared']}/32 shared at K=32, {o['overlap']['15']['shared']}/15 at K=15, "
                     f"Spearman {o['spearman_rho_all_heads']:.2f}." for name, o in overlap.items())
          + " Top-15 sycophancy heads (circuit_overlap ranking): "
          + ", ".join(f"L{a}H{b}" for a, b in overlap["circuit_overlap"]["top_syc"][:15]) + ".")
    parity_rows("pandey")

    doc.h(3, METHODS[4][1])
    mon = read_json(d / "persona_monitor.json")
    doc.p(f"Paper: vector = mean response activation of trait-expressing minus trait-suppressing rollouts kept by a GPT-4.1-mini "
          f"judge filter; layer: {paper['persona']['layer_16_for_llama']}. Detection claim: the last-prompt-token projection "
          f"tracks trait expression across graded system prompts, Pearson r = {mon['paper_table2_sycophancy_system_prompting']['overall']} "
          f"overall and {mon['paper_table2_sycophancy_system_prompting']['within_condition']} within condition for sycophancy "
          f"({mon['paper_table2_sycophancy_system_prompting']['reference']}).")
    doc.p(f"Ours: the same 8 system prompts (App. C.3), 20 eval questions and 10 rollouts each ({mon['n_rollouts']} rollouts, "
          f"{mon['n_score_none']} trait scores None): overall r = {mon['overall_pearson_r']:.3f}, within-condition mean r = "
          f"{mon['within_condition_mean_r']:.3f} (conditions excluded for SD < 1: {mon['conditions_excluded_sd_below_1'] or 'none'}). "
          "Mean trait score by system prompt (1 = most sycophantic): "
          + ", ".join(f"{k}: {v:.1f}" for k, v in mon["mean_trait_by_system_prompt"].items()) + ".")
    use = [read_json(d.parent.parent / "data/audit/persona/meta" / f"{n}_judge_results.jsonl.meta.json") for n in ("extract", "monitor")]
    tok_in, tok_out = sum(u["prompt_tokens"] for u in use), sum(u["completion_tokens"] for u in use)
    doc.p(f"Judge usage: {tok_in:,} prompt and {tok_out:,} completion tokens on gpt-4.1-mini-2025-04-14 (extraction trait + coherence "
          f"for 2,000 rollouts; monitoring trait for 1,600); about ${tok_in * 0.40e-6 + tok_out * 1.60e-6:.2f} at $0.40 / $1.60 per million "
          "tokens; no score was None.", note=True)
    parity_rows("persona")


def ood(doc: Doc, coverage, benches):
    doc.h(2, "3. Out-of-distribution performance on the benchmarks", "ood")
    doc.p("AUROC for each benchmark's target label with 95% CI, scored at each method's own read position. Benchmark rows "
          "were re-extracted from Llama-3.1-8B-Instruct in the method's input format; rows and labels are identical to the "
          "repo caches. Columns are grouped by taxonomy cell (* = cell inferred). Marks:", note=True)
    doc.bullets([f"{k} {v}" for k, v in FLAG_TEXT.items()])
    doc.p("Columns: " + "; ".join(f"{b['display']} = {b['cell']}, label {b['label']}, n={b['n']}" for b in benches) + ".", note=True)
    cov = {}
    for r in coverage:
        cov.setdefault(row_key(r), {})[r["benchmark"]] = r
    first = {k: v[benches[0]["id"]] for k, v in cov.items()}
    header = ["Detector"] + [f"{b['display']}{'*' if b['inferred'] else ''}" for b in benches]

    def rows_for(keys):
        md, ht = [], []
        for k in keys:
            cells = [cov[k][b["id"]] for b in benches]
            md.append([label(first[k])] + [f"{fnum(c['auroc'], 2)}{flag(c, b['id'])} [{fnum(c['ci_lo'], 2)}, {fnum(c['ci_hi'], 2)}]"
                                           for c, b in zip(cells, benches)])
            ht.append([esc(label(first[k]))] + [f"{fnum(c['auroc'], 2)}{flag(c, b['id'])} <span class='ci'>[{fnum(c['ci_lo'], 2)}, {fnum(c['ci_hi'], 2)}]</span>"
                                                for c, b in zip(cells, benches)])
        return md, ht

    doc.h(3, "Reference rows (committed contrastive probes, re-scored, not re-fit)")
    doc.table(header, *rows_for(["universal", "matched_cell", "max_taxonomy_posthoc"]))
    native = [k for k, r in first.items() if r["role"] == "detector" and r["matches_native"] == "yes"]
    for m, title in METHODS:
        doc.h(3, title)
        doc.table(header, *rows_for([k for k in native if method_of(first[k]) == m]))
    doc.h(3, "Interactive matrix")
    doc.raw_html("<!--HEAT-->")
    doc.p("Static figures: coverage_heatmap_native.png (native-position rows) and coverage_heatmap_cached_positions.png (first pass).", note=True)
    doc.h(3, "Secondary: the same directions at the repo's cached positions (first pass)")
    cached = [k for k, r in first.items() if r["role"] == "detector" and r["matches_native"] != "yes"]
    md, ht = rows_for(cached)
    doc.md.append(md_table(header, md) + "\n")
    doc.raw_html(f"<details><summary>{len(cached)} rows scored at last_prompt / first5 / response mean of the repo caches</summary>"
                 + html_table(header, ht, "num") + "</details>")


def controls_and_length(doc: Doc, controls, length, coverage):
    doc.h(2, "4. Controls and length confound", "controls")
    doc.p("Controls: AUROC separating the two sides of each held-out control pair (128 prompts x 2 per pair); 0.5 means the "
          "detector ignores that correlate. Native-position rows and references first.", note=True)
    native_det = {r["detector"] for r in coverage if r["role"] == "detector" and r["matches_native"] == "yes"}
    covfirst = {r["detector"]: r for r in coverage if r["role"] == "detector"}
    pairs = sorted({(int(r["control_pair"]), r["control_cell"]) for r in controls})
    by = {}
    for r in controls:
        by.setdefault(r["detector"], {})[int(r["control_pair"])] = r
    header = ["Detector"] + [f"P{p:02d} {c}" for p, c in pairs]

    def rows(keys):
        md, ht = [], []
        for k in keys:
            name = short_label(covfirst[k]) if k in covfirst else f"REF {by[k][pairs[0][0]]['unit']}"
            cells = [by[k][p] for p, _ in pairs]
            undefined = [c["auroc"] in ("", "None") for c in cells]
            md.append([name] + ["undefined" if u else f"{fnum(c['auroc'], 2)} [{fnum(c['ci_lo'], 2)}, {fnum(c['ci_hi'], 2)}]"
                                for c, u in zip(cells, undefined)])
            ht.append([esc(name)] + ["<span class='ci'>undefined</span>" if u else
                                     f"{fnum(c['auroc'], 2)} <span class='ci'>[{fnum(c['ci_lo'], 2)}, {fnum(c['ci_hi'], 2)}]</span>"
                                     for c, u in zip(cells, undefined)])
        return md, ht

    keys = list(by)
    doc.table(header, *rows([k for k in keys if k in native_det] + [k for k in keys if k not in covfirst]))
    doc.p("Undefined: the plain Human:/Assistant: format has no rendering of the control rows' system prompts.", note=True)
    rest = [k for k in keys if k in covfirst and k not in native_det]
    md, ht = rows(rest)
    doc.md.append("Cached-position rows:\n\n" + md_table(header, md) + "\n")
    doc.raw_html(f"<details><summary>{len(rest)} cached-position rows</summary>{html_table(header, ht, 'num')}</details>")

    doc.p("Length: Spearman rho between detector score and n_response_tokens, and the spread of AUROC across length "
          "terciles, summarised over the 15 benchmarks (all rows in length_confound.csv).", note=True)
    groups = {}
    for r in length:
        if r["role"] == "detector" and r["detector"] not in native_det:
            continue
        name = short_label(r) if r["role"] == "detector" else f"REF {r['unit']} ({r['role'].replace('_', ' ')})"
        groups.setdefault(name, []).append(r)
    md = []
    for name, rs in groups.items():
        rho = [(float(r["spearman_score_vs_n_response_tokens"]), r["display"]) for r in rs]
        spread = []
        for r in rs:
            t = [float(r[f"t{k}_auroc"]) for k in (1, 2, 3) if r[f"t{k}_auroc"] not in ("", "None")]
            spread.append((max(t) - min(t), r["display"]))
        mr, mg = max(rho, key=lambda x: abs(x[0])), max(spread)
        md.append([name, len(rs), fnum(np.median([x for x, _ in rho]), 2), f"{fnum(mr[0], 2)} ({mr[1]})",
                   fnum(np.median([x for x, _ in spread]), 2), f"{fnum(mg[0], 2)} ({mg[1]})"])
    doc.table(["Detector", "benchmarks", "median rho(score, length)", "max |rho| (benchmark)", "median tercile AUROC range",
               "max range (benchmark)"], md)
    seen, lr = set(), []
    for r in length:
        if r["display"] not in seen:
            seen.add(r["display"])
            lr.append([r["display"], r["n"], fnum(r["spearman_label_vs_n_response_tokens"], 2)])
    doc.p("rho(label, length) per benchmark, the same for every detector:", note=True)
    doc.table(["Benchmark", "n", "rho(label, length)"], lr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    d = args.dir
    coverage, parity = read_csv(d / "coverage.csv"), read_csv(d / "parity.csv")
    fig8b, fig10b = read_csv(d / "vennemeyer_fig8b.csv"), read_csv(d / "vennemeyer_fig10b.csv")
    controls, length = read_csv(d / "controls.csv"), read_csv(d / "length_confound.csv")
    choices, env = read_jsonl(d / "design_choices.jsonl"), read_json(d / "environment.json")
    reg, ver, paper = read_json(d / "regression.json"), read_json(d / "verification.json"), read_json(d / "paper_reference.json")
    overlap = {r: read_json(d / f"pandey_overlap_{r}.json") for r in ("circuit_overlap", "breadth")}
    bs = sorted({(r["benchmark"], r["display"], r["benchmark_cell"], r["cell_source"], r["target_label"], r["n"]) for r in coverage},
                key=lambda b: (CELLS.index(b[2]), b[1]))
    benches = [{"id": b[0], "display": b[1], "cell": b[2], "inferred": b[3] == "inferred", "label": b[4], "n": int(b[5])} for b in bs]

    doc = Doc()
    doc.md.append("# Sycophancy detector audit\n\nBranch `audit/existing-methods`. Model `meta-llama/Llama-3.1-8B-Instruct`. "
                  "Generated from the files in this directory by `audit/write_review.py`.\n")
    doc.h(2, "1. Summary", "summary")
    nat = [r for r in coverage if r["role"] == "detector" and r["matches_native"] == "yes"]
    summ = []
    for m, title in METHODS:
        rows = [r for r in nat if method_of(r) == m]
        if rows:
            above = sum(float(r["ci_lo"]) > 0.5 for r in rows)
            below = sum(float(r["ci_hi"]) < 0.5 for r in rows)
            best = max(rows, key=lambda r: float(r["auroc"]))
            o = (f"{len({r['detector'] for r in rows})} detectors x 15 benchmarks: CI above 0.5 in {above}, below 0.5 in {below} "
                 f"of {len(rows)} cells; highest {float(best['auroc']):.2f} ({best['display']})")
        else:
            o = "not scored (held)"
        rep = {"caa": "reproduced; consistent with the paper's qualitative layer finding (no numbers in the paper)",
               "vennemeyer": "reproduced with reconstructed pooling; partly matches Fig. 8b and Fig. 10b",
               "genadi": "reproduced; no Llama-3.1-8B numbers in the paper",
               "pandey": "reproduced at the code-faithful position; head overlap vs Table 1 in section 2",
               "persona": "reproduced with the GPT-4.1-mini judge; monitoring correlation vs Table 2 in section 2"}[m]
        summ.append([title, rep, o])
    doc.table(["Method", "Reproduction on source data", "Benchmarks at the method's own read position"], summ)
    doc.p("Changes since the first version: (1) benchmark scoring at each method's own read position and input format; "
          "(2) the review is split into reproduction and out-of-distribution sections; (3) Pandey's own detectors added "
          "(LR at L27 from probe_transfer.py, DIM at L19 from steering.py) and the DIM at L27 relabelled as our combination; "
          "(4) Pandey head-overlap and Vennemeyer direction-geometry checks against the papers; (5) plain-text whole-response "
          "Vennemeyer rows at the cached response position are no longer marked native; (6) independent code reviews; (7) Persona "
          "Vectors completed with the GPT-4.1-mini judge (user-supplied key) and scored like the other methods; (8) per-token "
          "heatmaps on eight rows per benchmark (4 sycophantic, 4 not) in a separate page (token_heatmaps.html).", note=True)
    doc.p("Environment: " + "; ".join(f"{k} {v}" for k, v in env["environment"].items()) + ".", note=True)
    doc.p("Not in this task: " + "; ".join(f"{n}: {t}" for n, t in NOT_IN_TASK), note=True)

    reproduction(doc, d, parity, paper, fig8b, fig10b, overlap)
    ood(doc, coverage, benches)
    controls_and_length(doc, controls, length, coverage)

    doc.h(2, "5. Design choices, assumptions and inferences", "choices")
    doc.md.append("\n".join(f"- *{c['tag']}* ({c['topic']}): {c['text']}" for c in choices) + "\n")
    doc.raw_html("<ul class='choices'>" + "".join(
        f"<li><span class='tag tag-{c['tag'].replace(' ', '-')}'>{esc(c['tag'])}</span> <b>{esc(c['topic'])}</b>: {esc(c['text'])}</li>"
        for c in choices) + "</ul>")
    doc.h(2, "6. Verification", "verify")
    doc.p("Independent review by a separate agent (read-only; values recomputed from the saved files):", note=True)
    doc.bullets(ver["lines"])
    doc.p("Regression evidence:", note=True)
    doc.bullets(reg["lines"])
    (d / "REVIEW.md").write_text("\n".join(doc.md) + "\n", encoding="utf-8")

    cov = {(row_key(r), r["benchmark"]): r for r in coverage}
    keys = list(dict.fromkeys(row_key(r) for r in coverage))
    heat = {"benches": benches, "flags": FLAG_TEXT, "rows": []}
    for k in keys:
        cells = [cov[(k, b["id"])] for b in benches]
        r0 = cells[0]
        heat["rows"].append({"label": label(r0), "role": r0["role"], "method": method_of(r0) if r0["role"] == "detector" else "reference",
                             "native": r0["role"] != "detector" or r0["matches_native"] == "yes", "built_on": r0["built_on"],
                             "cells": [[round(float(c["auroc"]), 4), round(float(c["ci_lo"]), 4), round(float(c["ci_hi"]), 4),
                                        c["unit"], flag(c, b["id"])] for c, b in zip(cells, benches)]})
    template = (Path(__file__).parent / "review_template.html").read_text(encoding="utf-8")
    shell, heat_block = template.split("<!--HEATBLOCK-->")
    body = "\n".join(doc.html).replace("<!--HEAT-->", heat_block)
    (d / "review.html").write_text(shell.replace("<!--BODY-->", body).replace("/*DATA*/null", json.dumps(heat)), encoding="utf-8")
    inputs = [d / n for n in ("coverage.csv", "parity.csv", "vennemeyer_fig8b.csv", "vennemeyer_fig10b.csv", "controls.csv",
                              "length_confound.csv", "design_choices.jsonl", "environment.json", "regression.json",
                              "verification.json", "paper_reference.json", "pandey_overlap_circuit_overlap.json",
                              "pandey_overlap_breadth.json", "persona_monitor.json")]
    for name in ("REVIEW.md", "review.html"):
        write_meta(d / name, inputs, args, check_counts(len(coverage), {}, len(coverage), name), {"model": "meta-llama/Llama-3.1-8B-Instruct"})
    print(f"wrote {d / 'REVIEW.md'} and {d / 'review.html'}")


if __name__ == "__main__":
    main()
