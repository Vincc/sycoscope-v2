"""Difference-in-means directions per unit and block from an extract_source npz; val/test projection AUROC per block.

direction = mean(label 1) - mean(label 0) over 'fit' rows (unit-normed); threshold = midpoint of the projected class
means, the rule of utils.probes.fit_dim. The layer is chosen on val AUROC (--select val) or fixed by the recipe.
Writes activations/audit/<method>/directions/<stem>__<pool>.npz and
reports/audit_existing_methods/source/<stem>__<pool>.jsonl (one row per unit x block).

Run from the repo root: python -m audit.fit_directions --activations activations/audit/caa/generate_dataset.npz --pool letter --select val
"""
import argparse
from pathlib import Path

import numpy as np

from utils import probes
from utils.io import REPO_ROOT, check_counts, meta_path, read_json, write_jsonl, write_meta

REPORT_DIR = REPO_ROOT / "reports" / "audit_existing_methods" / "source"


def source_provenance(act_meta: dict) -> dict:
    """Model, method and external repo/commit, read from the source JSONL's meta behind an extract_source npz."""
    (src,) = [p for p in act_meta["inputs"] if p.endswith(".jsonl")]
    src_meta = read_json(meta_path(REPO_ROOT / src))
    return {"model": act_meta["model"], "method": act_meta["method"], "source_file": src,
            "source_repo": src_meta["source_repo"], "source_commit": src_meta["source_commit"]}


def directions_from_sums(z, pool: str, unit: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(unit directions (L, d), thresholds (L,), raw differences (L, d), class counts) from class sums."""
    n1, n0 = int(z[f"count__{pool}__{unit}__1"]), int(z[f"count__{pool}__{unit}__0"])
    if n1 == 0 or n0 == 0:
        raise ValueError(f"{unit}: fit rows have one class only ({n1} pos, {n0} neg)")
    mu1, mu0 = z[f"sum__{pool}__{unit}__1"] / n1, z[f"sum__{pool}__{unit}__0"] / n0
    diff = mu1 - mu0
    norm = np.linalg.norm(diff, axis=1, keepdims=True)
    if not (np.isfinite(norm).all() and (norm > 0).all()):
        raise ValueError(f"{unit}: zero or non-finite mean difference at some block")
    d = diff / norm
    thr = ((mu1 * d).sum(1) + (mu0 * d).sum(1)) / 2.0
    if not ((mu1 * d).sum(1) > (mu0 * d).sum(1)).all():
        raise AssertionError(f"{unit}: direction does not point toward label 1")
    return d, thr, diff, {"n_fit_pos": n1, "n_fit_neg": n0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--select", required=True, help="'val' (max val AUROC, ties to the lower block) or a block index.")
    parser.add_argument("--units", nargs="+", help="Default: every unit in the npz.")
    args = parser.parse_args()

    z = np.load(args.activations)
    meta = read_json(meta_path(args.activations))
    units = args.units or meta["units"]
    split, cached = z["split"], z["cached"]
    X = z[f"X__{args.pool}"]  # (m, L, d), rows where cached
    csplit = split[cached]
    n_layers = X.shape[1]
    if X.shape[0] != int(cached.sum()):
        raise AssertionError("cached rows and X disagree")

    out, rows, chosen = {}, [], {}
    for unit in units:
        d, thr, diff, fit_counts = directions_from_sums(z, args.pool, unit)
        y = z[f"labels__{unit}"][cached].astype(int)
        scores = np.einsum("mld,ld->ml", X, d)  # projection per cached row and block
        per_layer = []
        for L in range(n_layers):
            rec = {"source": args.activations.stem, "pool": args.pool, "unit": unit, "layer": L, **fit_counts}
            for name in ("val", "test"):
                m = csplit == name
                rec[f"n_{name}"] = int(m.sum())
                rec[f"n_{name}_pos"] = int((y[m] == 1).sum())
                rec[f"{name}_auroc"] = probes.auroc(y[m], scores[m, L]) if m.any() else None
            per_layer.append(rec)
        if args.select == "val":
            vals = [r["val_auroc"] for r in per_layer]
            if any(v is None for v in vals):
                raise ValueError(f"{unit}: val AUROC undefined; cannot select a layer")
            best = int(np.argmax(vals))  # first maximum = lowest block
        else:
            best = int(args.select)
        chosen[unit] = best
        for r in per_layer:
            r["selected"] = r["layer"] == best
            r["select_rule"] = args.select
        rows += per_layer
        out[f"{unit}__direction"] = d
        out[f"{unit}__threshold"] = thr
        out[f"{unit}__diff"] = diff
        print(f"{unit}: layer {best} val {per_layer[best]['val_auroc']} test {per_layer[best]['test_auroc']}")

    out_path = args.activations.parent / "directions" / f"{args.activations.stem}__{args.pool}.npz"
    out_path.parent.mkdir(exist_ok=True)
    np.savez(out_path, **out)
    table = REPORT_DIR / f"{args.activations.stem}__{args.pool}.jsonl"
    write_jsonl(table, rows)
    counts = check_counts(len(rows), {}, len(rows), table.name)
    extra = source_provenance(meta)
    for path in (out_path, table):
        write_meta(path, [args.activations], args, counts, {**extra, "selected_layers": chosen, "units": units})


if __name__ == "__main__":
    main()
