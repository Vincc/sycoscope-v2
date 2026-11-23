"""The sycophancy persona vector as a classifier of the persona judge's score and of the benchmark labels.

Each --scores file (reproduction/persona_vector/judge.py output: one 0-100 "sycophantic" score per response) is joined by id
to its judge input (persona_trait/rollouts/<stem>.jsonl), its judged file (judged/<stem>_judged.jsonl) and the cached
activations of its responses; the judge's question and answer must equal judge_question(messages) and the response.
Rows are split into sets by decoding and source. Targets per set: the judge's score >= --threshold, and each benchmark label
of interp.generalization_matrix.unit_rows under --judge-name (on the rows with that label).
At each position the classifier score is the projection on a unit vector built from the persona vector (response_avg_diff;
block L uses its hidden_states index L + 1), chosen by --vector: the persona vector itself, the persona vector minus its
component along the Assistant Axis (--assistant-axis, an assistant-axis pipeline axis.pt whose row L is block L), or that
component alone. Scores are raw and length-controlled. The length-controlled score is the projection's percentile within its
response-length bin (--length-bins equal-count bins of response tokens, per set), so it ranks rows only against rows of
similar length. The judge's score is also scored against each benchmark label. For each score and target: AUROC with a 95%
group-bootstrap interval (groups are prompts, scores held fixed), the share of positives among the top R rows, R = the
number of positives, and Spearman correlations with the judge's score and with response length; AUROC, interval and hit
rate are None when the target has one class.
Writes <output> (metrics per set x target x score), optionally <rows-output> (one record per response), and their meta files.

Run from the repo root: python -m interp.persona_judge_classifier --scores ... --judge-name ... --persona-vector ... ...
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from interp.cache_activations import POSITIONS, load, read_parts
from interp.generalization_matrix import LAYER_CONVENTION, PERSONA_KEY, bootstrap_auroc_ci, position_dir, unit_rows
from probe_training.get_activations import skyline_group
from reproduction.persona_vector.judge_rollouts import judge_question
from utils.io import check_counts, read_json, read_jsonl, write_json, write_jsonl, write_meta
from utils.probes import auroc


def length_bins(n_tokens: np.ndarray, k: int) -> np.ndarray:
    """Bin 0..k-1 of each row: equal-count bins by response length, ties broken by row order."""
    rank = np.empty(len(n_tokens), dtype=np.int64)
    rank[np.argsort(n_tokens, kind="stable")] = np.arange(len(n_tokens))
    return rank * k // len(n_tokens)


def within_bin_percentile(s: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """(rank of s within its bin - 0.5) / bin size, with tied scores given their average rank."""
    out = np.empty(len(s))
    for b in np.unique(bins):
        members = bins == b
        out[members] = (rankdata(s[members]) - 0.5) / members.sum()
    return out


def top_hit_rate(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """Share of label-1 rows among the k highest scores; rows tied with the k-th score fill the remaining places in
    proportion to their label-1 share."""
    if not 0 < k <= len(s):
        raise ValueError(f"k={k} with {len(s)} rows")
    cut = np.sort(s)[::-1][k - 1]
    above, tied = s > cut, s == cut
    return float((y[above].sum() + (k - above.sum()) * y[tied].mean()) / k)


def score_metrics(y: np.ndarray, s: np.ndarray, groups: np.ndarray, judge: np.ndarray, n_tokens: np.ndarray,
                  n_boot: int, seed: int) -> dict:
    """Metrics of score s against 0/1 target y; AUROC, interval and hit rate are None when y has one class."""
    n_pos = int(y.sum())
    out = {"n": len(y), "n_pos": n_pos, "auroc": None, "ci_low": None, "ci_high": None, "ci_resamples": 0,
           "top_r_hit_rate": None, "base_rate": float(y.mean()),
           "spearman_judge": float(spearmanr(s, judge).statistic), "spearman_tokens": float(spearmanr(s, n_tokens).statistic)}
    if 0 < n_pos < len(y):
        ci_low, ci_high, n_ok = bootstrap_auroc_ci(y, s, groups, n_boot, seed)
        out.update(auroc=auroc(y, s), ci_low=ci_low, ci_high=ci_high, ci_resamples=n_ok, top_r_hit_rate=top_hit_rate(y, s, n_pos))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, nargs="+", required=True, help="<decoding>/judging/persona_trait/*.scores.jsonl")
    parser.add_argument("--judge-name", required=True, help="Label suffix of the benchmark labels, e.g. sonnet55.")
    parser.add_argument("--persona-vector", type=Path, required=True)
    parser.add_argument("--vector", choices=["persona", "persona_minus_axis", "axis_part"], required=True)
    parser.add_argument("--assistant-axis", type=Path, help="axis.pt; required unless --vector persona.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--positions", nargs="+", choices=list(POSITIONS), required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Judge score at or above which a row is positive.")
    parser.add_argument("--length-bins", type=int, required=True)
    parser.add_argument("--bootstrap", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path)
    args = parser.parse_args()
    if (args.vector == "persona") != (args.assistant_axis is None):
        raise ValueError("--assistant-axis goes with --vector persona_minus_axis or axis_part, and only with them")

    persona_meta = read_json(args.persona_vector.parent / "meta" / f"{args.persona_vector.name}.meta.json")
    persona_model = (persona_meta["model"], persona_meta["revision"])
    persona = np.load(args.persona_vector)[PERSONA_KEY][args.block + 1].astype(np.float64)
    vector_info = {"vector": args.vector}
    if args.vector == "persona":
        v = persona
    else:
        axis = torch.load(args.assistant_axis, map_location="cpu", weights_only=True)[args.block].double().numpy()
        axis /= np.linalg.norm(axis)
        along = (persona @ axis) * axis
        v = persona - along if args.vector == "persona_minus_axis" else along
        vector_info["persona_cos_axis"] = float(persona @ axis / np.linalg.norm(persona))
    v /= np.linalg.norm(v)
    judge_target = f"judge>={args.threshold:g}"

    metrics, records, counts, inputs = [], [], {}, []
    for scores_path in args.scores:
        # join scores, judge inputs and judged rows by id
        judge_dir = scores_path.parent
        stem = scores_path.name.removesuffix(".scores.jsonl")
        rollouts_path = judge_dir / "rollouts" / f"{stem}.jsonl"
        judged_path = judge_dir.parent / "judged" / f"{stem}_judged.jsonl"
        decoding = judge_dir.resolve().parent.parent.name  # generations/<model>/<decoding>/judging/persona_trait/
        inputs += [scores_path, rollouts_path, judged_path]
        scores = read_jsonl(scores_path)
        rollout = {r["id"]: r for r in read_jsonl(rollouts_path)}
        judged_rows = read_jsonl(judged_path)
        judged = {r["id"]: r for r in judged_rows}
        if len({r["id"] for r in scores}) != len(scores) or {r["id"] for r in scores} != set(rollout):
            raise ValueError(f"{scores_path}: ids differ from {rollouts_path} or repeat")
        units, _ = unit_rows(judged_rows, args.judge_name)
        by_source, excluded = defaultdict(list), defaultdict(Counter)
        for r in scores:
            j, q = judged[r["id"]], rollout[r["id"]]
            if (q["question"], q["answer"]) != (judge_question(j["messages"]), j["response"]):
                raise ValueError(f"{r['id']}: judge input differs from the judged row")
            by_source[j["source"]].append(r)
            if r["sycophantic"] is None:
                excluded[j["source"]]["score_none"] += 1

        # cached activations of this file at each position
        cached = {}
        for position in args.positions:
            pdir = position_dir(args.cache_dir, judged_path, position)
            for meta in read_parts(pdir, {"definition": POSITIONS[position]}):
                if (meta["model"], meta["model_revision"]) != persona_model or meta["layer_convention"] != LAYER_CONVENTION:
                    raise ValueError(f"{pdir}: {meta['model']}@{meta['model_revision']}; persona vector is {persona_model}")
            cache_ids, acts = load(pdir, [args.block])
            cached[position] = ({i: k for k, i in enumerate(cache_ids)}, acts[args.block])
        span = {}
        for meta in read_parts(position_dir(args.cache_dir, judged_path, "response_mean"),
                               {"definition": POSITIONS["response_mean"]}):
            span.update(zip(meta["ids"], meta["spans"]))

        for source, rows in sorted(by_source.items()):
            name = f"{decoding}:{source}"
            kept = [r for r in rows if r["sycophantic"] is not None]
            counts[name] = check_counts(len(rows), excluded[source], len(kept), name)
            ids = sorted(r["id"] for r in kept)
            score = {r["id"]: r["sycophantic"] for r in kept}
            judge = np.array([score[i] for i in ids])
            y = (judge >= args.threshold).astype(int)
            groups = np.array([f"{source}::{skyline_group(judged[i])}" for i in ids])
            n_tokens = np.array([span[i][1] - span[i][0] for i in ids])
            bins = length_bins(n_tokens, args.length_bins)
            labels = {u.split("/", 1)[1]: dict(m) for u, m in units.items() if u.split("/", 1)[0] == source}
            counts[name].update(n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), n_groups=len(set(groups)),
                                labels={k: [sum(lab.get(i) == 1 for i in ids), sum(lab.get(i) == 0 for i in ids)]
                                        for k, lab in sorted(labels.items())})
            rec = {i: {"id": i, "set": name, "judge": score[i], "positive": int(yi), "response_tokens": int(t),
                       "length_bin": int(b), "labels": {k: lab.get(i) for k, lab in sorted(labels.items())}}
                   for i, yi, t, b in zip(ids, y, n_tokens, bins)}

            # classifier scores: projections, raw and length-controlled, and the judge's score
            scored = {}
            for position in args.positions:
                row, A = cached[position]
                proj = A[[row[i] for i in ids]].astype(np.float64) @ v
                controlled = within_bin_percentile(proj, bins)
                for i, p, c in zip(ids, proj, controlled):
                    rec[i][position] = {"projection": float(p), "length_controlled": float(c)}
                scored[position, "raw"], scored[position, "length_controlled"] = proj, controlled

            # metrics: every projection score against the judge's cut-off and against each label; the judge against each label
            def add(target, position, kind, keep, yt, s):
                metrics.append({"set": name, "target": target, "position": position, "score": kind,
                                **score_metrics(yt, s[keep], groups[keep], judge[keep], n_tokens[keep],
                                                args.bootstrap, args.seed)})

            everyone = np.arange(len(ids))
            for (position, kind), s in scored.items():
                add(judge_target, position, kind, everyone, y, s)
            for label, lab in sorted(labels.items()):
                keep = np.array([k for k, i in enumerate(ids) if i in lab], dtype=np.int64)
                yt = np.array([lab[ids[k]] for k in keep])
                add(label, None, "persona_judge", keep, yt, judge)
                for (position, kind), s in scored.items():
                    add(label, position, kind, keep, yt, s)
            print(name, {f"{m['target']}|{m['position']}|{m['score']}": m["auroc"] and round(m["auroc"], 3)
                         for m in metrics if m["set"] == name}, flush=True)
            records += [rec[i] for i in ids]

    total = {"n_in": sum(c["n_in"] for c in counts.values()), "n_out": len(records),
             "excluded": dict(sum((Counter(c["excluded"]) for c in counts.values()), Counter()))}
    check_counts(total["n_in"], total["excluded"], total["n_out"], "all files")
    write_json(args.output, {"block": args.block, **vector_info, "threshold": args.threshold, "judge_target": judge_target,
                             "length_bins": args.length_bins, "counts": counts, "metrics": metrics})
    outputs = [args.output]
    if args.rows_output is not None:
        write_jsonl(args.rows_output, records)
        outputs.append(args.rows_output)
    vector_inputs = [args.persona_vector, *([args.assistant_axis] if args.assistant_axis is not None else [])]
    for output in outputs:
        write_meta(output, [*inputs, *vector_inputs], args, total, {"sets": counts})
    print(f"{total}: wrote {', '.join(map(str, outputs))}")


if __name__ == "__main__":
    main()
