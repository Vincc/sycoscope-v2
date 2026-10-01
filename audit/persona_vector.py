"""Persona vector for the sycophantic trait from judged extraction rollouts.

Filter (generate_vec.get_persona_effective, threshold 50): keep a pos/neg rollout pair (same question, instruction and
rollout index) when pos trait >= 50, neg trait < 50 and both coherence >= 50; a None score fails. Vector per block =
mean response activation of kept pos rows - mean of kept neg rows (their response_avg_diff). Held-out check (ours):
refit on the kept rows of 80% of the questions, AUROC of the projection on the kept rows of the other 20%.
Writes activations/audit/persona/directions/extract_rollouts__response_mean.npz and
reports/audit_existing_methods/source/extract_rollouts__response_mean.jsonl (fit_directions format; --layer selected).

Run from the repo root: python -m audit.persona_vector --layer 15 --test-frac 0.2 --seed 0
"""
import argparse
from collections import Counter

import numpy as np

from audit.fit_directions import REPORT_DIR
from utils import probes
from utils.io import REPO_ROOT, check_counts, meta_path, read_json, read_jsonl, write_jsonl, write_meta
from utils.splits import group_split

ROLLOUTS = REPO_ROOT / "data/audit/persona/extract_rollouts.jsonl"
JUDGE = REPO_ROOT / "data/audit/persona/extract_judge_results.jsonl"
ACTS = REPO_ROOT / "activations/audit/persona/extract_rollouts.npz"


def effective_pairs(rows: list[dict], scores: dict, threshold: float) -> tuple[list[tuple[int, int]], Counter]:
    """[(pos row index, neg row index)] kept by the filter, and why the others were dropped."""
    by_key = {}
    for i, r in enumerate(rows):
        by_key.setdefault((r["question_index"], r["instruction_index"], r["rollout"]), {})[r["polarity"]] = i
    kept, why = [], Counter()
    for key, d in sorted(by_key.items()):
        if set(d) != {"pos", "neg"}:
            raise ValueError(f"{key}: pos/neg pair incomplete")
        p, n = rows[d["pos"]]["id"], rows[d["neg"]]["id"]
        sp, sn = scores[(p, "sycophantic")], scores[(n, "sycophantic")]
        cp, cn = scores[(p, "coherence")], scores[(n, "coherence")]
        if sp is None or sn is None or cp is None or cn is None:
            why["score_none"] += 1
        elif not sp >= threshold:
            why["pos_trait_below_threshold"] += 1
        elif not sn < 100 - threshold:
            why["neg_trait_not_below_threshold"] += 1
        elif not (cp >= 50 and cn >= 50):
            why["coherence_below_50"] += 1
        else:
            kept.append((d["pos"], d["neg"]))
    return kept, why


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, required=True, help="Block index (paper layer 16 = hidden_states[16] = block 15).")
    parser.add_argument("--threshold", type=float, default=50)
    parser.add_argument("--test-frac", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    rows = read_jsonl(ROLLOUTS)
    scores = {(r["rollout_id"], r["metric"]): r["score"] for r in read_jsonl(JUDGE)}
    z = np.load(ACTS)
    if z["id"].tolist() != [r["id"] for r in rows] or not z["cached"].all():
        raise AssertionError("activation rows differ from the rollouts")
    X = z["X__response_mean"]  # (n, n_layers, hidden), block outputs
    kept, why = effective_pairs(rows, scores, args.threshold)
    n_pairs = len(rows) // 2
    pair_counts = check_counts(n_pairs, why, len(kept), "rollout pairs")
    print(pair_counts)
    pos, neg = np.array([p for p, _ in kept]), np.array([n for _, n in kept])

    def fit(pi, ni):
        mu1, mu0 = X[pi].astype(np.float64).mean(0), X[ni].astype(np.float64).mean(0)
        diff = mu1 - mu0
        d = diff / np.linalg.norm(diff, axis=1, keepdims=True)
        thr = ((mu1 * d).sum(1) + (mu0 * d).sum(1)) / 2
        return d, thr, diff

    d, thr, diff = fit(pos, neg)
    groups = [f"q{rows[p]['question_index']:02d}" for p in pos]
    split = group_split(groups, args.test_frac, 0.0, args.seed)
    test_q = set(split["test"])
    tr = np.array([g not in test_q for g in groups])
    d_tr, _, _ = fit(pos[tr], neg[tr])
    te_idx = np.concatenate([pos[~tr], neg[~tr]])
    y_te = np.concatenate([np.ones((~tr).sum(), int), np.zeros((~tr).sum(), int)])
    table = []
    for L in range(X.shape[1]):
        s = X[te_idx, L].astype(np.float64) @ d_tr[L]
        table.append({"source": "extract_rollouts", "pool": "response_mean", "unit": "persona", "layer": L,
                      "n_fit_pos": len(pos), "n_fit_neg": len(neg), "n_val": 0, "n_val_pos": 0, "val_auroc": None,
                      "n_test": len(te_idx), "n_test_pos": int(y_te.sum()), "test_auroc": probes.auroc(y_te, s),
                      "test_note": "refit on kept pairs of the train questions, scored on kept pairs of held-out questions",
                      "selected": L == args.layer, "select_rule": "recipe"})
    out = ACTS.parent / "directions" / "extract_rollouts__response_mean.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, persona__direction=d, persona__threshold=thr, persona__diff=diff)
    tab = REPORT_DIR / "extract_rollouts__response_mean.jsonl"
    write_jsonl(tab, table)
    src = read_json(meta_path(ROLLOUTS))
    extra = {"model": "meta-llama/Llama-3.1-8B-Instruct", "method": "persona", "source_repo": src["source_repo"],
             "source_commit": read_json(meta_path(REPO_ROOT / "data/audit/persona/extract_prompts.jsonl"))["source_commit"],
             "pair_counts": pair_counts, "test_questions": sorted(test_q), "selected_layers": {"persona": args.layer},
             "units": ["persona"]}
    for path in (out, tab):
        write_meta(path, [ROLLOUTS, JUDGE, ACTS], args, check_counts(len(table), {}, len(table), path.name), extra)
    print(f"kept {len(kept)} of {n_pairs} pairs; layer {args.layer} held-out AUROC {table[args.layer]['test_auroc']}")


if __name__ == "__main__":
    main()
