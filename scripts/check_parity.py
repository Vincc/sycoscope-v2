"""Compare new outputs with the old repo's: a split, an activation file, or contrastive probe metrics. Exits non-zero on mismatch."""
import argparse
import sys
from pathlib import Path

import numpy as np

from utils.io import read_json, read_jsonl


def check_split(args) -> bool:
    new, old = read_json(args.new), read_json(args.old)
    ok = new["train"] == old["train"] and new["test"] == old["test"]
    ok &= new["seed"] == old["seed"] and new["test_frac"] == old["test_frac"]
    print(f"train {len(new['train'])} vs {len(old['train'])}, test {len(new['test'])} vs {len(old['test'])}: {'MATCH' if ok else 'MISMATCH'}")
    return ok


def check_activations(args) -> bool:
    """Old index rows are keyed by example_id = new id + '__s0'."""
    new_index = {r["id"]: r for r in read_jsonl(args.new_index)}
    old_index = {r["example_id"].removesuffix("__s0"): r for r in read_jsonl(args.old_index)}
    common = sorted(set(new_index) & set(old_index))
    if not common:
        raise ValueError("no ids in common")
    ok = True
    for rid in common:
        n, o = new_index[rid], old_index[rid]
        for field in ("prompt_len", "n_response_tokens", "n_first5_tokens", "first5_text"):
            if n[field] != o[field]:
                print(f"{rid}: {field} {n[field]!r} != {o[field]!r}")
                ok = False
    with np.load(args.new) as zn, np.load(args.old) as zo:
        keys = sorted(set(zn.files) & set(zo.files))
        if not keys:
            raise ValueError("no array keys in common")
        rn = [new_index[i]["row"] for i in common]
        ro = [old_index[i]["row"] for i in common]
        for key in keys:
            a, b = zn[key][rn], zo[key][ro]
            max_abs = float(np.abs(a - b).max())
            rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            match = max_abs <= args.atol
            ok &= match
            print(f"{key}: {len(common)} rows, max |diff| {max_abs:.3e}, rel norm diff {rel:.3e} {'ok' if match else 'MISMATCH'}")
    print(f"{len(common)} ids compared ({len(new_index)} new, {len(old_index)} old)")
    return ok


def check_probes(args) -> bool:
    """New logistic metrics vs rows of the old summary.json, matched on (slug, position, layer)."""
    old = {(r["slug"], r["position"], r["layer"]): r for r in read_json(args.old_summary)["rows"]}
    new = [m for p in sorted(args.new_dir.glob("*.metrics.jsonl")) for m in read_jsonl(p) if m["method"] == args.method]
    if not new:
        raise ValueError(f"no {args.method} metrics in {args.new_dir}")
    ok, worst = True, 0.0
    pairs = [("holdout_auroc", "holdout_auc"), ("holdout_accuracy", "holdout_accuracy"), ("holdout_paired_win_rate", "holdout_paired_win_rate")]
    for m in new:
        o = old[(m["target"], m["position"], m["layer"])]
        for nf, of in pairs:
            if (m[nf] is None) != (o[of] is None):
                print(f"{m['target']} {m['position']} L{m['layer']} {nf}: {m[nf]} vs {o[of]}")
                ok = False
                continue
            if m[nf] is None:
                continue
            d = abs(m[nf] - o[of])
            worst = max(worst, d)
            if d > args.atol:
                print(f"{m['target']} {m['position']} L{m['layer']} {nf}: {m[nf]:.6f} vs {o[of]:.6f}")
                ok = False
        if m["n_train"] + m["n_test"] != o["n_pos"] + o["n_neg"]:
            print(f"{m['target']} {m['position']} L{m['layer']}: {m['n_train'] + m['n_test']} rows vs old {o['n_pos'] + o['n_neg']}")
            ok = False
    print(f"{len(new)} probes compared, max |diff| {worst:.2e}: {'MATCH' if ok else 'MISMATCH'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="what", required=True)
    p = sub.add_parser("split")
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--old", type=Path, required=True)
    p = sub.add_parser("activations")
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--new-index", type=Path, required=True)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--old-index", type=Path, required=True)
    p.add_argument("--atol", type=float, required=True)
    p = sub.add_parser("probes")
    p.add_argument("--new-dir", type=Path, required=True)
    p.add_argument("--old-summary", type=Path, required=True)
    p.add_argument("--method", choices=["logistic", "dim"], required=True)
    p.add_argument("--atol", type=float, required=True)
    args = parser.parse_args()
    ok = {"split": check_split, "activations": check_activations, "probes": check_probes}[args.what](args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
