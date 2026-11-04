"""Counterfactual prompting, analysis (paper Fig. A11): for each assumption probe, score the positive and the
negative version of each prompt at the probe's layer and report the mean paired difference (pos - neg) with a
paired t-test. Scores are the probe's predicted assumption score X @ w + b, as in training."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True, help="npz from extract_reps.py over the counterfactual prompts file")
    ap.add_argument("--prompts", required=True, help="counterfactual prompts jsonl")
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reps = np.load(args.reps)
    idx = {i: k for k, i in enumerate(reps["ids"].tolist())}
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8")]
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["dim"], r["base_id"])][r["polarity"]] = idx[r["id"]]
    assert all(set(v) == {"pos", "neg"} for v in pairs.values())
    out = {"reps": args.reps, "prompts": args.prompts, "n_prompts": len({b for _, b in pairs}), "probes": {}}
    print(f"{'probe':42s} {'layer':5s} {'mean diff':>9s} {'sd':>7s} {'t':>7s} {'p':>8s}")
    for path in sorted(glob.glob(os.path.join(args.probe_dir, "*.npz"))):
        name = os.path.basename(path)[:-4]
        lab, dim = name.split("_", 1)
        p = np.load(path)
        L, w, b = int(p["layer"]), p["w"], float(p["b"])
        X = reps[f"layer_{L}"]
        keys = sorted(k for k in pairs if k[0] == dim)
        pos = np.array([X[pairs[k]["pos"]] @ w + b for k in keys])
        neg = np.array([X[pairs[k]["neg"]] @ w + b for k in keys])
        d = pos - neg
        t, pv = ttest_rel(pos, neg)
        out["probes"][name] = {"layer": L, "n": len(d), "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                               "t": float(t), "p": float(pv), "frac_positive": float((d > 0).mean()),
                               "mean_pos": float(pos.mean()), "mean_neg": float(neg.mean())}
        print(f"{name:42s} {L:5d} {d.mean():+9.4f} {d.std(ddof=1):7.4f} {t:7.2f} {pv:8.1e}")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
