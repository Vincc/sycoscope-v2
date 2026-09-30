"""Coverage, controls and length-confound tables for the audited detectors, with the committed contrastive references.

Every detector and reference probe is re-scored on the benchmark rows (never re-fit) and its AUROC is asserted equal
to the eval JSONL written by evaluate_probes / evaluate_heads. CIs: stratified bootstrap (positives and negatives
resampled separately), --n-boot resamples, --seed. Writes coverage.csv, controls.csv, length_confound.csv and
coverage_heatmap.png under --out-dir.

Run from the repo root: python -m audit.analyze --audit-sweeps ... --head-dirs ... --out-dir reports/audit_existing_methods
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from audit.evaluate_heads import features
from utils import probes
from utils.activations import NO_LABEL, act_key, read_labels
from utils.io import REPO_ROOT, check_counts, read_json, read_jsonl, write_meta

JUDGING = REPO_ROOT / "generations" / "Llama-3.1-8B-Instruct" / "judging"
HEAD_CACHE = REPO_ROOT / "activations" / "audit" / "heads"
CELLS = ("Position-Verifiable / Explicit", "Position-Subjective / Explicit", "Position-Subjective / Implicit",
         "Person-Traits / Explicit", "Person-Traits / Implicit")
# Benchmark taxonomy cell, from the docs/SPEC.md benchmark table; "inferred" = not listed there.
BENCHMARK_CELL = {
    "ays_mc": ("Position-Verifiable / Explicit", "spec"), "ays_freeform": ("Position-Verifiable / Explicit", "spec"),
    "tqa_false_answer": ("Position-Verifiable / Explicit", "inferred"),
    "aita_nta_flip": ("Position-Subjective / Explicit", "spec"), "aita_nta_og": ("Position-Subjective / Explicit", "inferred"),
    "framing": ("Position-Subjective / Implicit", "spec"), "validation": ("Person-Traits / Explicit", "spec"),
    "sypr": ("Person-Traits / Explicit", "spec"), "indirectness": ("Person-Traits / Implicit", "spec"),
}
CELL_PAIR = {"Position-Verifiable / Explicit": 1, "Position-Subjective / Explicit": 3, "Position-Subjective / Implicit": 4,
             "Person-Traits / Explicit": 5, "Person-Traits / Implicit": 6}


def benchmark_cell(stem: str) -> tuple[str, str]:
    for key, value in BENCHMARK_CELL.items():
        if stem.startswith(key) or f"_{key}_" in stem:
            return value
    raise KeyError(f"no taxonomy cell for {stem}")


def read_csv(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def fast_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney AUROC: P(pos > neg) + 0.5 P(pos == neg); equals sklearn's roc_auc_score."""
    neg = np.sort(neg_scores)
    below = np.searchsorted(neg, pos_scores, side="left")
    ties = np.searchsorted(neg, pos_scores, side="right") - below
    return float((below + 0.5 * ties).sum() / (len(pos_scores) * len(neg)))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("bootstrap needs both classes")
    stats = [fast_auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def tercile_aurocs(y, s, length) -> list[dict]:
    cuts = np.quantile(length, [1 / 3, 2 / 3])
    bins = np.digitize(length, cuts, right=True)  # 0: <= q1, 1: (q1, q2], 2: > q2
    out = []
    for t in range(3):
        m = bins == t
        out.append({"n": int(m.sum()), "n_pos": int((y[m] == 1).sum()), "auroc": probes.auroc(y[m], s[m]) if m.any() else None,
                    "max_tokens": int(length[m].max()) if m.any() else None})
    return out


def residual_detectors(sweep_dirs: list[Path]) -> list[dict]:
    dets = []
    for d in sweep_dirs:
        for m in read_jsonl(d / "manifest.jsonl"):
            dets.append({"detector": f"{d.name}:{m['probe_id']}", "kind": "residual", "sweep": d, "manifest": m,
                         "audit_method": m["audit_method"], "unit": m["audit_unit"], "variant": d.name,
                         "layer": m["layer"], "position": m["position"], "layer_rule": m["layer_rule"],
                         "native_position": m["native_position"], "matches_native": "yes" if m["position_matches_native"] else "no",
                         "built_on": m["cell"], "our_addition": False})
    return dets


def head_detectors(head_dirs: list[tuple[Path, str]]) -> list[dict]:
    dets = []
    for d, pooling in head_dirs:
        for m in read_jsonl(d / "manifest.jsonl"):
            dets.append({"detector": f"{d.name}:{m['probe_id']}", "kind": "heads", "sweep": d, "manifest": m,
                         "audit_method": m["audit_method"], "unit": m["audit_unit"], "variant": d.name,
                         "layer": m["layers"], "position": pooling, "layer_rule": "source val" if m["audit_method"] == "genadi" else "recipe",
                         "native_position": m["native_position"], "matches_native": m["matches_native"],
                         "built_on": m["cell"], "our_addition": m["our_addition"]})
    return dets


def reference_detectors(ref_sweep: Path, selected: list[dict]) -> list[dict]:
    """Universal pair 00 and each taxonomy pair's val-selected probe (same probe for every benchmark)."""
    manifest = {m["probe_id"]: m for m in read_jsonl(ref_sweep / "manifest.jsonl")}
    per_pair = {}
    for r in selected:
        per_pair.setdefault(int(r["pair_index"]), set()).add(r["probe_id"])
    dets = []
    for pair, ids in sorted(per_pair.items()):
        if len(ids) != 1:
            raise ValueError(f"pair {pair}: {len(ids)} validation-selected probes")
        m = manifest[ids.pop()]
        if m["pair_type"] == "control":
            continue
        dets.append({"detector": f"{ref_sweep.name}:{m['probe_id']}", "kind": "reference", "sweep": ref_sweep, "manifest": m,
                     "audit_method": "contrastive", "unit": f"P{pair:02d} {m['cell']}", "variant": "validation_selected",
                     "layer": m["layer"], "position": m["position"], "layer_rule": "synthetic val",
                     "native_position": m["position"], "matches_native": "yes", "built_on": m["cell"],
                     "our_addition": False, "pair_index": pair})
    return dets


def score(det: dict, z, zh, cache: dict) -> np.ndarray:
    m = det["manifest"]
    if det["kind"] == "heads":
        return probes.score_logistic(probes.load_probe(np.load(det["sweep"] / "detectors.npz"), m["probe_id"], "logistic"),
                                     features(zh, det["position"], m["heads"]))
    key = act_key(m["position"], m["layer"])
    if key not in cache:
        cache[key] = z[key]
    probe = probes.load_probe(np.load(det["sweep"] / m["file"]), m["probe_id"], m["method"])
    return probes.score(m["method"], probe, cache[key])


def eval_auroc(det: dict, stem: str, label: str) -> float:
    if det["kind"] == "heads":
        rows = read_jsonl(det["sweep"] / "eval" / f"{stem}__{det['position']}.jsonl")
    else:
        rows = read_jsonl(det["sweep"] / "eval" / f"{stem}.jsonl")
    (row,) = [r for r in rows if r["probe_id"] == det["manifest"]["probe_id"] and r["eval_label"] == label]
    return row["auroc"], row["n"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-sweeps", type=Path, nargs="+", required=True)
    parser.add_argument("--head-dirs", nargs="+", required=True, help="<detector dir>=<pooling>")
    parser.add_argument("--reference-sweep", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True, help="validation_selected.csv")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--controls-cache", type=Path, required=True, help="TEFO contrastive activations npz")
    parser.add_argument("--controls-heads", type=Path, required=True, help="get_head_activations npz of control test rows")
    parser.add_argument("--n-boot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replot", action="store_true", help="Only redraw the heatmap from an existing coverage.csv.")
    args = parser.parse_args()
    if args.replot:
        heatmap(args, read_csv(args.out_dir / "coverage.csv"), read_csv(args.targets))
        return

    targets = read_csv(args.targets)
    selected = read_csv(args.selected)
    head_dirs = [(Path(x.split("=")[0]), x.split("=")[1]) for x in args.head_dirs]
    dets = residual_detectors(args.audit_sweeps) + head_detectors(head_dirs)
    refs = reference_detectors(args.reference_sweep, selected)
    coverage, length_rows = [], []
    for t in targets:
        stem, label = t["eval_dataset"], t["target_label"]
        cell, cell_source = benchmark_cell(stem)
        z = np.load(JUDGING / f"{stem}_activations.npz")
        zh = np.load(HEAD_CACHE / f"{stem}_heads.npz")
        if not np.array_equal(z["id"], zh["id"]):
            raise AssertionError(f"{stem}: head cache rows differ from the benchmark cache")
        y_all = read_labels(z)[label].astype(int)
        keep = y_all != NO_LABEL
        y, length = y_all[keep], z["n_response_tokens"][keep]
        if keep.sum() != int(t["n"]):
            raise AssertionError(f"{stem}: {keep.sum()} labelled rows, targets.csv says {t['n']}")
        cache = {}
        matched_pair = CELL_PAIR[cell]
        tax_rows = {int(r["pair_index"]): r for r in selected if r["eval_dataset"] == stem}
        best_tax = max((p for p in tax_rows if 1 <= p <= 8), key=lambda p: float(tax_rows[p]["auroc"]))
        for det in dets + refs:
            if det["kind"] == "reference":
                p = det["pair_index"]
                roles = (["universal"] if p == 0 else []) + (["matched_cell"] if p == matched_pair else []) + \
                        (["max_taxonomy_posthoc"] if p == best_tax else [])
                if not roles:
                    continue
            else:
                roles = ["detector"]
            s = score(det, z, zh, cache)[keep]
            auc = probes.auroc(y, s)
            ref_auc, ref_n = eval_auroc(det, stem, label)
            if ref_n != len(y) or not np.isclose(auc, ref_auc, rtol=0, atol=1e-9):
                raise AssertionError(f"{det['detector']} on {stem}: AUROC {auc} (n {len(y)}) != eval file {ref_auc} (n {ref_n})")
            lo, hi = bootstrap_ci(y, s, args.n_boot, args.seed)
            base = {k: det[k] for k in ("detector", "audit_method", "unit", "variant", "layer", "position", "layer_rule",
                                        "matches_native", "built_on", "our_addition")}
            for role in roles:
                coverage.append({**base, "role": role, "benchmark": stem, "display": t["display"], "family": t["family"],
                                 "benchmark_cell": cell, "cell_source": cell_source, "target_label": label, "n": len(y),
                                 "n_pos": int((y == 1).sum()), "auroc": auc, "ci_lo": lo, "ci_hi": hi,
                                 "n_boot": args.n_boot, "boot_seed": args.seed})
            if roles == ["detector"] or "universal" in roles or "matched_cell" in roles:
                terc = tercile_aurocs(y, s, length)
                length_rows.append({**base, "role": "/".join(roles), "benchmark": stem, "display": t["display"],
                                    "n": len(y), "spearman_score_vs_n_response_tokens": spearman(s, length),
                                    "spearman_label_vs_n_response_tokens": spearman(y, length),
                                    **{f"t{k + 1}_{f}": terc[k][f] for k in range(3) for f in ("n", "n_pos", "max_tokens", "auroc")}})
        print(f"{stem}: {len(dets) + 3} rows", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save(args, "coverage.csv", coverage)
    save(args, "length_confound.csv", length_rows)
    heatmap(args, coverage, targets)
    save(args, "controls.csv", controls(args, dets, refs))


def save(args, name: str, rows: list[dict]) -> None:
    write_csv(args.out_dir / name, rows)
    write_meta(args.out_dir / name, [args.targets, args.selected], args, check_counts(len(rows), {}, len(rows), name),
               {"model": "meta-llama/Llama-3.1-8B-Instruct", "n_boot": args.n_boot, "seed": args.seed})


def controls(args, dets, refs) -> list[dict]:
    """AUROC of each detector (and the universal / taxonomy references) on held-out rows of the 5 control pairs."""
    z = np.load(args.controls_cache)
    split = read_json(args.reference_sweep / "split.json")
    test = np.isin(z[split["group_field"]], split["test"])
    ctrl = (z["pair_type"] == "control") & test
    zh = np.load(args.controls_heads)
    ids = z["id"][ctrl]
    order = {i: k for k, i in enumerate(zh["id"])}
    if sorted(order) != sorted(ids.tolist()):
        raise AssertionError("control head cache rows differ from the held-out control rows of the TEFO cache")
    head_idx = np.array([order[i] for i in ids])
    y_all, pair = z["label"][ctrl].astype(int), z["pair_index"][ctrl]
    cells = z["cell"][ctrl]
    rows, cache = [], {}
    for det in dets + refs:
        if det["kind"] == "heads":
            s_all = score(det, None, zh, cache)[head_idx]
        else:
            key = act_key(det["manifest"]["position"], det["manifest"]["layer"])
            if key not in cache:
                cache[key] = z[key][ctrl]
            probe = probes.load_probe(np.load(det["sweep"] / det["manifest"]["file"]), det["manifest"]["probe_id"],
                                      det["manifest"]["method"])
            s_all = probes.score(det["manifest"]["method"], probe, cache[key])
        for p in sorted(set(pair.tolist())):
            m = pair == p
            lo, hi = bootstrap_ci(y_all[m], s_all[m], args.n_boot, args.seed)
            rows.append({"detector": det["detector"], "audit_method": det["audit_method"], "unit": det["unit"],
                         "variant": det["variant"], "layer": det["layer"], "position": det["position"],
                         "our_addition": det["our_addition"], "control_pair": int(p), "control_cell": str(cells[m][0]),
                         "n": int(m.sum()), "n_pos": int((y_all[m] == 1).sum()), "auroc": probes.auroc(y_all[m], s_all[m]),
                         "ci_lo": lo, "ci_hi": hi})
    return rows


def short_label(r: dict) -> str:
    """Row label: method, unit, variant, layer, scored position, native mark."""
    if r["role"] != "detector":
        return {"universal": "REF universal P00", "matched_cell": "REF matched-cell pair",
                "max_taxonomy_posthoc": "REF max over taxonomy pairs (post hoc)"}[r["role"]]
    method = r["audit_method"].removesuffix("_diag")
    variant = r["variant"].removeprefix("audit_").removeprefix(method).removeprefix("_").replace("_", " ")
    variant = "" if variant in ("", "heads") else f" · {variant}"
    layer = str(r["layer"])
    if layer.startswith("["):
        layers = layer.strip("[]").split(",")
        layer = f"heads L{layers[0].strip()}" if len(layers) == 1 else f"heads in {len(layers)} layers"
    else:
        layer = f"L{layer}" + (" recipe" if r["layer_rule"] == "extra" else "")
    nat = {"yes": "native", "approx": "≈native", "no": "non-native"}[r["matches_native"]]
    tag = " [ours]" if str(r["our_addition"]) == "True" else ""
    unit = "" if r["unit"] == method else " " + {"syc": "SyA", "ga": "GA", "pr": "SyPr"}.get(r["unit"], r["unit"])
    if method == "pandey" and r["unit"] == "syc":
        unit = " residual"
    return f"{method}{unit}{variant} · {layer} · {r['position']} ({nat}){tag}"


def heatmap(args, coverage: list[dict], targets: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    norm = TwoSlopeNorm(vmin=0.30, vcenter=0.50, vmax=0.95)  # scripts/plot_probe_transfer.py NORM
    cols = sorted(targets, key=lambda t: (CELLS.index(benchmark_cell(t["eval_dataset"])[0]), t["family"], t["display"]))
    row_keys = list(dict.fromkeys((r["role"], r["detector"] if r["role"] == "detector" else r["role"]) for r in coverage))
    lookup = {((r["role"], r["detector"] if r["role"] == "detector" else r["role"]), r["benchmark"]): r for r in coverage}
    M = np.array([[float(lookup[(k, c["eval_dataset"])]["auroc"]) for c in cols] for k in row_keys])
    labels = [short_label(lookup[(k, cols[0]["eval_dataset"])]) for k in row_keys]
    fig, ax = plt.subplots(figsize=(19, 0.32 * len(row_keys) + 4))
    im = ax.imshow(M, aspect="auto", cmap="RdBu", norm=norm)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.8 or M[i, j] < 0.36 else "#172033")
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.set_xticks(range(len(cols)), [f"{c['display']}\n{benchmark_cell(c['eval_dataset'])[0]}" for c in cols],
                  rotation=60, ha="right", fontsize=7.5)
    for j in range(1, len(cols)):
        if benchmark_cell(cols[j]["eval_dataset"])[0] != benchmark_cell(cols[j - 1]["eval_dataset"])[0]:
            ax.axvline(j - 0.5, color="#172033", linewidth=1.7)
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01, label="AUROC")
    ax.set_title("Existing sycophancy detectors on the balanced judged benchmarks (Llama-3.1-8B-Instruct)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = args.out_dir / "coverage_heatmap.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    write_meta(path, [args.out_dir / "coverage.csv"], args, check_counts(len(coverage), {}, len(coverage), path.name),
               {"model": "meta-llama/Llama-3.1-8B-Instruct"})


if __name__ == "__main__":
    main()
