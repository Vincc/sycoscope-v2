"""Per-cell steering results: mean trait and coherence with 95% t-intervals over per-question means, and the best cell.

Best cell = highest mean trait among steered cells with mean coherence >= 50. The peak is unresolved when another
layer's cell at the best cell's coefficient has a trait CI that overlaps the best cell's.

Run from the repo root:
  python -m reproduction.persona_vector.steer_summary --rollouts-dir ... --scores-dir ... --out ...
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from utils.io import read_jsonl, write_json, write_meta

TRAIT = "sycophantic"


def question_ci(values_by_question: dict) -> dict:
    """Mean of rollouts, and a 95% t-interval over the per-question means. Questions with any None score are kept with their non-None rollouts."""
    q_means = np.array([np.mean(v) for v in values_by_question.values() if v])
    n_rollouts = sum(len(v) for v in values_by_question.values())
    mean = float(np.mean([x for v in values_by_question.values() for x in v]))
    half = float(stats.t.ppf(0.975, len(q_means) - 1) * q_means.std(ddof=1) / np.sqrt(len(q_means)))
    return {"mean": mean, "question_mean": float(q_means.mean()), "ci_low": float(q_means.mean() - half),
            "ci_high": float(q_means.mean() + half), "n_questions": len(q_means), "n_rollouts": n_rollouts}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cells, inputs = [], []
    for path in sorted(args.rollouts_dir.glob("L*_c*.jsonl")):
        score_path = args.scores_dir / f"{path.name.removesuffix('.jsonl')}.scores.jsonl"
        rows, scores = read_jsonl(path), {s["id"]: s for s in read_jsonl(score_path)}
        inputs += [path, score_path]
        if set(scores) != {r["id"] for r in rows}:
            raise AssertionError(f"{score_path} ids differ from {path}")
        cell = {"layer": rows[0]["layer"], "block": rows[0]["layer"] - 1, "coef": rows[0]["coef"], "n": len(rows),
                "n_unfinished": sum(not r["finished"] for r in rows)}
        for metric in (TRAIT, "coherence"):
            by_q = defaultdict(list)
            for r in rows:
                if scores[r["id"]][metric] is not None:
                    by_q[r["question_index"]].append(scores[r["id"]][metric])
            cell[metric] = question_ci(by_q)
            cell[metric]["n_none"] = sum(scores[r["id"]][metric] is None for r in rows)
        cells.append(cell)

    steered = [c for c in cells if c["coef"] != 0]
    eligible = [c for c in steered if c["coherence"]["mean"] >= 50]
    if not eligible:
        raise ValueError("no steered cell has mean coherence >= 50")
    best = max(eligible, key=lambda c: c[TRAIT]["mean"])
    overlapping = [c["layer"] for c in steered if c["coef"] == best["coef"] and c["layer"] != best["layer"]
                   and c[TRAIT]["ci_high"] >= best[TRAIT]["ci_low"] and c[TRAIT]["ci_low"] <= best[TRAIT]["ci_high"]]
    summary = {"cells": cells, "best": {"layer": best["layer"], "block": best["block"], "coef": best["coef"],
                                        "trait_mean": best[TRAIT]["mean"], "coherence_mean": best["coherence"]["mean"]},
               "layers_overlapping_best_ci": sorted(overlapping), "peak_resolved": not overlapping,
               "n_cells": len(cells), "n_eligible": len(eligible)}
    write_json(args.out, summary)
    write_meta(args.out, inputs, args, {"n_cells": len(cells), "n_eligible": len(eligible)})
    print(f"best cell layer {best['layer']} coef {best['coef']}: trait {best[TRAIT]['mean']:.1f}, "
          f"coherence {best['coherence']['mean']:.1f}; overlapping layers {sorted(overlapping)}")


if __name__ == "__main__":
    main()
