"""Compare our steering Spearman rho (elephant5_score.py summary) with the paper's Table A9 (8B probe model)."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="*_summary.json from elephant5_score.py")
    ap.add_argument("--paper", required=True, help="data/paper_table_a9_8b.json")
    args = ap.parse_args()
    ours = json.load(open(args.summary))
    paper = json.load(open(args.paper))["rho"]
    pairs = []
    print(f"{'probe/dim':50s} {'ours':>6s} {'p':>6s} {'paper':>6s}")
    for k in sorted(paper):
        if k not in ours:
            print(f"{k:50s} missing"); continue
        r, p = ours[k]["spearman_rho"], ours[k]["p"]
        print(f"{k:50s} {'undef' if r is None else f'{r:+.2f}':>6s} {'' if p is None else f'{p:.3f}':>6s} {paper[k]:+.2f}")
        if r is not None:
            pairs.append((r, paper[k], p))
    o, pp, p = (np.array(x) for x in zip(*pairs))
    print(f"\n{len(pairs)} defined of {len(paper)}; undefined (constant means): {len(paper) - len(pairs)}")
    print(f"sign agreement (both |rho| >= 0.5): {np.mean(np.sign(o[(abs(o) >= .5) & (abs(pp) >= .5)]) == np.sign(pp[(abs(o) >= .5) & (abs(pp) >= .5)])):.2f} "
          f"over {int(((abs(o) >= .5) & (abs(pp) >= .5)).sum())} cells")
    print(f"correlation of rho vectors: {np.corrcoef(o, pp)[0, 1]:.2f}")
    for d in ("validation", "indirectness", "framing"):
        m = np.array([k.endswith("/" + d) for k in sorted(paper) if k in ours and ours[k]["spearman_rho"] is not None])
        print(f"  {d:12s} ours significant (p<.05): {int((p[m] < .05).sum())}/{int(m.sum())}; paper |rho|>=0.79: {int((abs(pp[m]) >= .79).sum())}")


if __name__ == "__main__":
    main()
