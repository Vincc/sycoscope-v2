"""Parity table: each detector's held-out AUROC on its own source data (95% stratified bootstrap CI) next to the paper.

Residual detectors are re-scored from the cached held-out rows of the extract_source npz; head detectors from its
H__<pool> test rows. A paper comparison is made only where the paper gives a Llama-3.1-8B value (Vennemeyer Fig. 8b,
read by eye): match if |ours - paper| <= --tolerance at every listed layer. Writes parity.csv and
vennemeyer_fig8b.csv under --out-dir.

Run from the repo root: python -m audit.parity --out-dir reports/audit_existing_methods --n-boot 1000 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np

from audit.analyze import bootstrap_ci, read_csv, write_csv
from audit.fit_directions import directions_from_sums
from audit.fit_heads import head_features
from utils import probes
from utils.io import REPO_ROOT, check_counts, read_json, read_jsonl, write_meta

ACT = REPO_ROOT / "activations" / "audit"
PROBES = REPO_ROOT / "probes" / "Llama-3.1-8B-Instruct"
SOURCE = REPO_ROOT / "reports" / "audit_existing_methods" / "source"


def residual_heldout(npz: Path, pool: str, unit: str, layer: int):
    z = np.load(npz)
    d, _, _, _ = directions_from_sums(z, pool, unit)
    cached = z["cached"]
    test = z["split"][cached] == "test"
    y = z[f"labels__{unit}"][cached][test].astype(int)
    return y, z[f"X__{pool}"][test, layer] @ d[layer]


def head_heldout(npz: Path, pool: str, det_dir: Path, probe_id: str):
    z = np.load(npz)
    m = next(r for r in read_jsonl(det_dir / "manifest.jsonl") if r["probe_id"] == probe_id)
    test = z["split"] == "test"
    det = probes.load_probe(np.load(det_dir / "detectors.npz"), probe_id, "logistic")
    return z["labels__syc"][test].astype(int), probes.score_logistic(det, head_features(z[f"H__{pool}"][test], m["heads"])), m


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    paper = read_json(args.out_dir / "paper_reference.json")
    rows = []

    def add(method, unit, variant, layer, y, s, paper_value, paper_ref, verdict, **extra):
        lo, hi = bootstrap_ci(y, s, args.n_boot, args.seed)
        rows.append({"method": method, "unit": unit, "variant": variant, "layer": layer, "n_heldout": len(y),
                     "n_heldout_pos": int((y == 1).sum()), "auroc": probes.auroc(y, s), "ci_lo": lo, "ci_hi": hi,
                     "paper_value": paper_value, "paper_reference": paper_ref, "verdict": verdict, **extra})

    # CAA: selected (val) and recipe layer 13
    for m in read_jsonl(PROBES / "audit_caa" / "manifest.jsonl"):
        if m["position"] != "last_prompt":
            continue
        y, s = residual_heldout(ACT / "caa" / "generate_dataset.npz", "letter", "caa", m["layer"])
        table = read_jsonl(SOURCE / "generate_dataset__letter.jsonl")
        emerge = next(r["layer"] for r in table if r["test_auroc"] >= 0.9)
        add("CAA", "caa", {"extra": "recipe layer (paper's 7B steering layer)", "val": "val-selected layer"}[m["layer_rule"]], m["layer"], y, s, None, paper["caa"]["reference"],
            f"qualitative: held-out AUROC first >= 0.9 at L{emerge}; paper: behavioural clustering emerges ~1/3 depth (L10, Llama-2-7B)")

    # Vennemeyer math: both poolings, plain and chat; paper comparison for plain only
    fig = paper["vennemeyer"]["fig8b_read"]
    fig_rows = []
    for variant in ("plain", "chat"):
        for pool, act_dir, sweep in (("last_content", "vennemeyer", "last"), ("resp_all", "vennemeyer_diag", "respall")):
            npz = ACT / act_dir / f"math_factorial__{variant}.npz"
            table = read_jsonl(SOURCE / f"math_factorial__{variant}__{pool}.jsonl")
            for unit in ("syc", "ga", "pr"):
                sel = next(r["layer"] for r in table if r["unit"] == unit and r["selected"])
                y, s = residual_heldout(npz, pool, unit, sel)
                by_layer = {r["layer"]: r["test_auroc"] for r in table if r["unit"] == unit}
                verdict, pv = "no paper value (chat variant)", None
                if variant == "plain":
                    diffs = {int(k): by_layer[int(k)] - v for k, v in fig[unit].items()}
                    worst = max(diffs, key=lambda k: abs(diffs[k]))
                    verdict = ("match" if abs(diffs[worst]) <= args.tolerance else "mismatch") + \
                        f" (max |diff| {abs(diffs[worst]):.2f} at L{worst})"
                    pv = "see by-layer table"
                    for k, v in fig[unit].items():
                        fig_rows.append({"pool": pool, "unit": unit, "layer": int(k), "paper_fig8b": v,
                                         "ours_test_auroc": by_layer[int(k)], "diff": by_layer[int(k)] - v})
                add("Vennemeyer", {"syc": "SyA", "ga": "GA", "pr": "SyPr"}[unit], f"math {variant} {pool}", sel, y, s, pv,
                    paper["vennemeyer"]["reference"], verdict)

    # Pandey residual (recipe layer 27) and heads
    y, s = residual_heldout(ACT / "pandey" / "triviaqa_syc.npz", "faithful", "syc", 27)
    add("Pandey", "residual DIM-27 (our combination)", "code-faithful position", 27, y, s, None, paper["pandey"]["reference"],
        "no published Llama-3.1-8B value")
    z = np.load(ACT / "pandey" / "triviaqa_syc.npz")
    test = z["split"] == "test"
    for name, sweep in (("LR-27 (their probe_transfer)", "audit_pandey_lr27_native"), ("DIM-19 (their steering direction)", "audit_pandey_dim19_native")):
        (m,) = read_jsonl(PROBES / sweep / "manifest.jsonl")
        probe = probes.load_probe(np.load(PROBES / sweep / m["file"]), m["probe_id"], m["method"])
        s = probes.score(m["method"], probe, z["X__faithful"][test, m["layer"]].astype(np.float64))
        if not np.isclose(probes.auroc(z["labels__syc"][test].astype(int), s), m["source_heldout_auroc"], atol=1e-9):
            raise AssertionError(f"{sweep}: held-out AUROC differs from its manifest")
        add("Pandey", name, "code-faithful position", m["layer"], z["labels__syc"][test].astype(int), s, None,
            paper["pandey"]["reference"], "no published Llama-3.1-8B value")
    y, s, m = head_heldout(ACT / "pandey" / "triviaqa_syc.npz", "faithful", PROBES / "audit_pandey_heads", "pandey__lr_top15")
    add("Pandey", "LR top-15 heads [ours]", "code-faithful position", m["layers"], y, s, None, paper["pandey"]["reference"],
        "no published Llama-3.1-8B value", heads=m["heads"])

    # Genadi heads, both poolings
    for pool, d in (("answer_mean", "audit_genadi_heads"), ("last", "audit_genadi_heads_last")):
        for pid, name in (("genadi__best_head", "best head"), ("genadi__lr_top16", "LR top-16 heads [ours]")):
            y, s, m = head_heldout(ACT / "genadi" / "truthfulqa_dialogues.npz", pool, PROBES / d, pid)
            add("Genadi", name, f"{pool} pooling", m["layers"], y, s, None, paper["genadi"]["reference"],
                "no published Llama-3.1-8B value", heads=m["heads"], source_val_acc=m.get("source_val_acc"))

    ov = read_json(REPO_ROOT / "reports" / "audit_existing_methods" / "pandey_overlap.json")
    shared, rho = ov["overlap"]["32"]["shared"], ov["spearman_rho_all_heads"]
    ok = abs(shared - 21) <= 3 and abs(rho - 0.88) <= 0.05
    rows.append({"method": "Pandey", "unit": "head overlap, syc vs lie, K=32", "variant": f"first {ov['n_rank_pairs']} pairs per task",
                 "layer": "all heads", "n_heldout": None, "n_heldout_pos": None, "auroc": None, "ci_lo": None, "ci_hi": None,
                 "paper_value": "21/32 shared; Spearman 0.88", "paper_reference": "Table 1 (p. 5), Llama-3.1-8B row",
                 "verdict": f"{'match' if ok else 'mismatch'}: ours {shared}/32 shared (chance 1.0), Spearman {rho:.2f} "
                            f"(criterion: within 3 heads and 0.05)"})
    geo = read_csv(args.out_dir / "vennemeyer_fig10b.csv")
    for pool in ("last_content", "resp_all"):
        for pair in ("syc-ga", "syc-pr", "ga-pr"):
            g = [r for r in geo if r["pool"] == pool and r["pair"] == pair]
            worst = max(g, key=lambda r: abs(float(r["diff"])))
            n_ok = sum(r["within_tolerance"] == "True" for r in g)
            rows.append({"method": "Vennemeyer", "unit": f"cosine {pair.replace('syc', 'SyA').replace('ga', 'GA').replace('pr', 'SyPr')}",
                         "variant": f"math plain {pool}", "layer": "by layer", "n_heldout": None, "n_heldout_pos": None,
                         "auroc": None, "ci_lo": None, "ci_hi": None, "paper_value": "see by-layer table",
                         "paper_reference": paper["vennemeyer"]["fig10b_read"]["note"],
                         "verdict": f"{'match' if n_ok == len(g) else 'mismatch'}: {n_ok}/{len(g)} layers within 0.1; "
                                    f"max |diff| {abs(float(worst['diff'])):.2f} at L{worst['layer']}"})
    rows.append({"method": "Persona Vectors", "unit": "sycophantic", "variant": "held at judge step", "layer": 15,
                 "n_heldout": None, "n_heldout_pos": None, "auroc": None, "ci_lo": None, "ci_hi": None, "paper_value": None,
                 "paper_reference": paper["persona"]["layer_16_for_llama"], "verdict": "held: 4000 GPT-4.1-mini judge calls needed"})
    for r in rows:
        r.setdefault("heads", None)
        r.setdefault("source_val_acc", None)
    rows = [{k: r[k] for k in rows[0]} | {k: r[k] for k in r if k not in rows[0]} for r in rows]
    write_csv(args.out_dir / "parity.csv", rows)
    write_csv(args.out_dir / "vennemeyer_fig8b.csv", fig_rows)
    for name, n in (("parity.csv", len(rows)), ("vennemeyer_fig8b.csv", len(fig_rows))):
        write_meta(args.out_dir / name, [args.out_dir / "paper_reference.json"], args, check_counts(n, {}, n, name),
                   {"model": "meta-llama/Llama-3.1-8B-Instruct"})
    for r in rows:
        print(r["method"], r["unit"], r["variant"], r["layer"], r["auroc"], r["verdict"])


if __name__ == "__main__":
    main()
