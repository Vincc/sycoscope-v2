"""Vennemeyer parity, geometry: cosine between the SyA, GA and SyPr difference-in-means directions per layer, next to
their Fig. 10b (read by eye). Writes reports/audit_existing_methods/vennemeyer_fig10b.csv.

Run from the repo root: python -m audit.vennemeyer_geometry --tolerance 0.1
"""
import argparse

import numpy as np

from audit.analyze import write_csv
from utils.io import REPO_ROOT, check_counts, read_json, write_meta

DIRS = {"last_content": REPO_ROOT / "activations/audit/vennemeyer/directions/math_factorial__plain__last_content.npz",
        "resp_all": REPO_ROOT / "activations/audit/vennemeyer_diag/directions/math_factorial__plain__resp_all.npz"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, required=True)
    args = parser.parse_args()
    ref_path = REPO_ROOT / "reports/audit_existing_methods/paper_reference.json"
    fig = read_json(ref_path)["vennemeyer"]["fig10b_read"]
    rows = []
    for pool, path in DIRS.items():
        z = np.load(path)
        for pair in ("syc-ga", "syc-pr", "ga-pr"):
            a, b = (z[f"{u}__diff"] for u in pair.split("-"))
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            for k, v in fig[pair].items():
                rows.append({"pool": pool, "pair": pair, "layer": int(k), "paper_fig10b": v, "ours_cosine": float(cos[int(k)]),
                             "diff": float(cos[int(k)]) - v, "within_tolerance": abs(float(cos[int(k)]) - v) <= args.tolerance})
    out = REPO_ROOT / "reports/audit_existing_methods/vennemeyer_fig10b.csv"
    write_csv(out, rows)
    write_meta(out, [ref_path, *DIRS.values()], args, check_counts(len(rows), {}, len(rows), out.name),
               {"model": "meta-llama/Llama-3.1-8B-Instruct"})
    for r in rows:
        print(r["pool"], r["pair"], r["layer"], r["paper_fig10b"], round(r["ours_cosine"], 2))


if __name__ == "__main__":
    main()
