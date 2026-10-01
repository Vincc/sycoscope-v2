"""Persona Vectors parity: system-prompt monitoring (paper Sec. 3.3, App. C.2 Table 2).

For each graded system prompt x eval question: mean trait score over its rollouts (None scores skipped, as pandas
mean) and the projection of the last prompt token onto the persona vector (cal_projection.py prompt_last_proj:
a . v / |v|, hidden_states[16] = block 15). Overall Pearson r over all (prompt, question) points; within-condition r
computed per system prompt and averaged, excluding conditions whose score SD is below 1.
Writes reports/audit_existing_methods/persona_monitor.json.

Run from the repo root: python -m audit.persona_monitor --layer 15
"""
import argparse

import numpy as np

from utils.io import REPO_ROOT, check_counts, read_jsonl, write_json, write_meta

ROLLOUTS = REPO_ROOT / "data/audit/persona/monitor_rollouts.jsonl"
JUDGE = REPO_ROOT / "data/audit/persona/monitor_judge_results.jsonl"
ACTS = REPO_ROOT / "activations/audit/persona/monitor_rollouts.npz"
VECTOR = REPO_ROOT / "activations/audit/persona/directions/extract_rollouts__response_mean.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    from scipy.stats import pearsonr

    rows = read_jsonl(ROLLOUTS)
    score = {r["rollout_id"]: r["score"] for r in read_jsonl(JUDGE) if r["metric"] == "sycophantic"}
    z = np.load(ACTS)
    if z["id"].tolist() != [r["id"] for r in rows] or not z["cached"].all():
        raise AssertionError("activation rows differ from the rollouts")
    v = np.load(VECTOR)["persona__diff"][args.layer]
    proj_all = z["X__last_prompt"][:, args.layer].astype(np.float64) @ v / np.linalg.norm(v)
    cells = {}
    for r, p in zip(rows, proj_all):
        cells.setdefault((r["system_index"], r["question_index"]), {"proj": [], "score": []})
        cells[(r["system_index"], r["question_index"])]["proj"].append(p)
        cells[(r["system_index"], r["question_index"])]["score"].append(score[r["id"]])
    points, spread, n_none = [], 0.0, 0
    for (si, qi), c in sorted(cells.items()):
        s = [x for x in c["score"] if x is not None]
        n_none += len(c["score"]) - len(s)
        if not s:
            raise ValueError(f"system {si} question {qi}: every trait score is None")
        spread = max(spread, float(np.ptp(c["proj"])))  # same prompt in every rollout; only batch numerics differ
        points.append({"system_index": si, "question_index": qi, "projection": float(np.mean(c["proj"])),
                       "mean_trait": float(np.mean(s)), "n_scored": len(s)})
    P = np.array([p["projection"] for p in points])
    S = np.array([p["mean_trait"] for p in points])
    overall = float(pearsonr(P, S).statistic)
    within, excluded = [], []
    for si in sorted({p["system_index"] for p in points}):
        m = np.array([p["system_index"] == si for p in points])
        if S[m].std() < 1:
            excluded.append(si)
            continue
        within.append(float(pearsonr(P[m], S[m]).statistic))
    out = {"layer_block": args.layer, "n_points": len(points), "n_rollouts": len(rows), "n_score_none": n_none,
           "overall_pearson_r": overall, "within_condition_mean_r": float(np.mean(within)), "within_condition_r": within,
           "conditions_excluded_sd_below_1": excluded, "max_projection_spread_within_prompt": spread,
           "mean_trait_by_system_prompt": {str(si): float(np.mean([p["mean_trait"] for p in points if p["system_index"] == si]))
                                           for si in sorted({p["system_index"] for p in points})},
           "paper_table2_sycophancy_system_prompting": {"overall": 0.798, "within_condition": 0.669,
                                                        "reference": "App. C.2, Table 2 (p. 31); model not stated (Fig. 4 neither; only Fig. 14 is captioned Qwen)"},
           "points": points}
    path = REPO_ROOT / "reports/audit_existing_methods/persona_monitor.json"
    write_json(path, out)
    write_meta(path, [ROLLOUTS, JUDGE, ACTS, VECTOR], args, check_counts(len(rows), {}, len(rows), "rollouts"),
               {"model": "meta-llama/Llama-3.1-8B-Instruct"})
    print({k: out[k] for k in ("overall_pearson_r", "within_condition_mean_r", "conditions_excluded_sd_below_1", "n_score_none",
                               "max_projection_spread_within_prompt", "mean_trait_by_system_prompt")})


if __name__ == "__main__":
    main()
