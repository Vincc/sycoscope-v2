"""Pandey parity: overlap of the top-K sycophancy and factual-lying heads, and Spearman rho over all heads.

Heads are ranked with fit_heads.head_deltas on the first --n-rank pairs of each task at the code-faithful position.
Writes reports/audit_existing_methods/pandey_overlap.json.

Run from the repo root: python -m audit.pandey_overlap --syc ... --lie ... --n-rank 50 --k 15 32
"""
import argparse
from pathlib import Path

import numpy as np

from audit.fit_heads import head_deltas, o_proj_weights
from utils.io import REPO_ROOT, check_counts, write_json, write_meta

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--syc", type=Path, required=True)
    parser.add_argument("--lie", type=Path, required=True)
    parser.add_argument("--n-rank", type=int, required=True)
    parser.add_argument("--k", type=int, nargs="+", required=True)
    args = parser.parse_args()
    from scipy.stats import spearmanr

    w_o = o_proj_weights(MODEL, list(range(32)))
    ranked = {}
    for task, path, label in (("syc", args.syc, "labels__syc"), ("lie", args.lie, "labels__lie")):
        z = np.load(path)
        rows = z["group"] < f"pair{args.n_rank:03d}"
        ranked[task] = head_deltas(z["H__faithful"], z[label].astype(int), rows, w_o)
    syc = np.array([r["delta_norm"] for r in ranked["syc"]])
    lie = np.array([r["delta_norm"] for r in ranked["lie"]])
    order = lambda v: [(int(i) // 32, int(i) % 32) for i in np.argsort(-v, kind="stable")]
    out = {"n_rank_pairs": args.n_rank, "spearman_rho_all_heads": float(spearmanr(syc, lie).statistic), "overlap": {}}
    for k in args.k:
        shared = set(order(syc)[:k]) & set(order(lie)[:k])
        out["overlap"][str(k)] = {"shared": len(shared), "chance": k * k / 1024, "heads_shared": sorted(shared)}
    out["top_syc"], out["top_lie"] = order(syc)[:max(args.k)], order(lie)[:max(args.k)]
    path = REPO_ROOT / "reports" / "audit_existing_methods" / "pandey_overlap.json"
    write_json(path, out)
    write_meta(path, [args.syc, args.lie], args, check_counts(1024, {}, 1024, "heads"), {"model": MODEL})
    print({k: v["shared"] for k, v in out["overlap"].items()}, "rho", out["spearman_rho_all_heads"])


if __name__ == "__main__":
    main()
