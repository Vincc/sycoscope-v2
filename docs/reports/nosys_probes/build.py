"""Build the no-system-prompt probes page from reports/llama31_nosys_probes/summary.json, the sweeps' manifests and the
cache's per-row response stats. Usage: python docs/reports/nosys_probes/build.py <repo root>"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
HERE = Path(__file__).parent
P = REPO / "probes/Llama-3.1-8B-Instruct"
CACHE = REPO / ("activations/cache/Llama-3.1-8B-Instruct/training/synthetic_dataset/synthetic_trait_emotion_fact_opinion/"
                "synthetic_trait_emotion_fact_opinion_contrastive_responses")
MATCH = {"syco": 1, "both_nta": 3, "framing": 4, "validation": 5, "indirectness": 6}  # docs/SPEC.md benchmark -> cell


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def selected(manifest):
    """{(pair, method, position, layer): row}: logistic at best val AUROC (ties: smaller C), DiM as is."""
    best = {}
    for m in read(manifest):
        k = (m["pair_index"], m["method"], m["position"], m["layer"])
        o = best.get(k)
        if o is None or m["val_auroc"] > o["val_auroc"] or (m["val_auroc"] == o["val_auroc"] and m["C"] < o["C"]):
            best[k] = m
    return best


S = json.loads((REPO / "reports/llama31_nosys_probes/summary.json").read_text(encoding="utf-8"))
layers = S["layers"]
pairs = {int(k): v for k, v in S["pairs"].items()}

# Per-row response stats under each context, from the cache meta files
stats = {}
for cond, root in (("sys", CACHE), ("nosys", CACHE.with_name(CACHE.name + "_nosys"))):
    a, n = [], []
    for f in sorted((root / "response_mean/meta").glob("part-*.meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        a += m["next_token_agreement"]
        n += m["response_nll"]
    if len(a) != 17920:
        raise AssertionError(f"{root}: {len(a)} rows of response stats")
    stats[cond] = {"agreement_mean": float(np.mean(a)), "nll_mean": float(np.mean(n)), "nll_median": float(np.median(n))}

# Null sweep on no-system prompt_last
null_sel = selected(P / "tefo_nosys_cache_prompt_last_null/manifest.jsonl")
null = {L: {meth: sorted(m["test_auroc"] for k, m in null_sel.items() if k[3] == L and k[1] == meth)
            for meth in ("logistic", "dim")} for L in layers}
null_top = max(null_sel.values(), key=lambda m: m["test_auroc"])

sys_with_prompt_last = selected(P / "tefo_sys_cache/manifest.jsonl")
prompt_last_sys = [m["test_auroc"] for k, m in sys_with_prompt_last.items() if k[2] == "prompt_last"]

likelihood = {(r["pair"], r["condition"], r["stat"]): r["auroc"] for r in S["likelihood"] if "auroc" in r}
spear = {(r["pair"], r["condition"], r["method"], r["layer"]): r["value"] for r in S["likelihood"]
         if r["stat"] == "spearman_probe_vs_nll"}
data = {
    "layers": layers,
    "pairs": [{"i": p, "cell": c} for p, c in sorted(pairs.items())],
    "in_dist": S["in_dist"],
    "transfer": S["transfer"],
    "cosine": S["cosine"],
    "likelihood": [{"pair": p, "nll_sys": likelihood[(p, "sys", "response_nll")],
                    "nll_nosys": likelihood[(p, "nosys", "response_nll")],
                    "agree_sys": likelihood[(p, "sys", "next_token_agreement")],
                    "agree_nosys": likelihood[(p, "nosys", "next_token_agreement")],
                    "spearman": {m: {str(L): spear[(p, "nosys", m, L)] for L in layers} for m in ("logistic", "dim")},
                    "spearman_sys": {m: {str(L): spear[(p, "sys", m, L)] for L in layers} for m in ("logistic", "dim")}}
                   for p in sorted(pairs)],
    "ood": {},  # unit -> method -> pair -> {"s": [auroc per layer], "n": [...], "ci": {"s": [lo, hi], "n": [lo, hi]}}
    "ood_units": S["ood_units"],
    "match": MATCH,
    "parity": S["parity"],
    "null_spread": S["null_check"],
    "null_sweep": null,
    "null_top": {"pair": null_top["pair_index"], "method": null_top["method"], "layer": null_top["layer"],
                 "test_auroc": null_top["test_auroc"], "val_auroc": null_top["val_auroc"], "n_test": null_top["n_test"]},
    "prompt_last_sys": {"min": min(prompt_last_sys), "n": len(prompt_last_sys)},
    "stats": stats,
    "counts": {"rows": S["sys_cache"]["counts"]["n_out"], "n_probes_sys": len(read(P / "tefo_sys_cache/manifest.jsonl")),
               "n_probes_nosys": len(read(P / "tefo_nosys_cache/manifest.jsonl"))},
}
ci_layer = sorted({r["layer"] for r in S["ood"] if "ci_low" in r})
if len(ci_layer) != 1:
    raise AssertionError(f"expected intervals at one block, got {ci_layer}")
data["ci_layer"] = ci_layer[0]
for r in S["ood"]:
    cell = data["ood"].setdefault(r["unit"], {}).setdefault(r["method"], {}).setdefault(
        str(r["pair"]), {"s": [None] * len(layers), "n": [None] * len(layers), "ci": {}})
    c = r["condition"][0]
    cell[c][layers.index(r["layer"])] = round(r["auroc"], 4)
    if "ci_low" in r:
        cell["ci"][c] = [round(r["ci_low"], 4), round(r["ci_high"], 4)]

# Headline aggregates for the text
diffs = np.array([r_n - r_s for u in data["ood"].values() for meth in u.values() for c in meth.values()
                  for r_s, r_n in zip(c["s"], c["n"])])
ci_l = data["ci_layer"]
ci_cells = [c for u in data["ood"].values() for meth in u.values() for c in meth.values()]
if any(set(c["ci"]) != {"s", "n"} for c in ci_cells):
    raise AssertionError("a probe x label cell lacks an interval at the interval block")
overlap = [not (c["ci"]["n"][0] > c["ci"]["s"][1] or c["ci"]["s"][0] > c["ci"]["n"][1]) for c in ci_cells]
indist = {cond: [r["test_auroc"] for r in S["in_dist"] if r["condition"] == cond] for cond in ("sys", "nosys")}
sep = [max(v, 1 - v) for (p, cond, stat), v in likelihood.items() if cond == "nosys" and stat == "response_nll"]
data["headline"] = {
    "n_ood_cells": int(diffs.size), "ood_diff_mean": float(diffs.mean()), "ood_diff_median": float(np.median(diffs)),
    "ood_diff_abs_p95": float(np.quantile(np.abs(diffs), 0.95)), "ood_diff_max_abs": float(np.abs(diffs).max()),
    "ood_frac_nosys_higher": float((diffs > 0).mean()), "ood_ci_overlap_frac": float(np.mean(overlap)),
    "n_ci_cells": len(overlap),
    "indist_sys_min": min(indist["sys"]), "indist_nosys_min": min(indist["nosys"]),
    "indist_nosys_median": float(np.median(indist["nosys"])),
    "nll_nosys_sep_max": max(sep), "nll_nosys_sep_median": float(np.median(sep)),
}
html = (HERE / "template.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
(HERE / "nosys_probes.html").write_text(html, encoding="utf-8", newline="\n")
print(f"wrote {HERE / 'nosys_probes.html'} ({len(html) / 1e3:.0f} KB)")
