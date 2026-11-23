"""Sycophancy fine-tuning runs against the base model: training loss, eval-question trait and coherence per steering
cell, and the shift of the last-prompt-token projection onto the persona vector.

Eval questions: per model and cell, mean trait / coherence with a 95% t-interval over the 20 per-question means
(steer_summary.question_ci), and the paired difference to the base model's same cell over questions. Base cells are
data/rollouts/steer/<cell>, fine-tuned ones data/rollouts/ft/<run>/<cell>; scores sit in the matching judge dirs.
Projection shift: monitor/ft/<run>.jsonl minus monitor/ft/base.jsonl, both on the same prompts, averaged per question
then t-interval over questions, at every hidden_states index.

Run from the repo root:
  python -m reproduction.persona_vector.finetune_summary --runs syc-normal syc-mis2 ... --out ...
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from reproduction.persona_vector.steer_summary import question_ci
from utils.io import read_json, read_jsonl, write_json, write_meta

P = Path("reproduction/persona_vector")
CELLS = ["L00_c0.0", "L16_c-2.0", "L16_c-1.0", "L16_c1.0", "L16_c2.0", "L16_c2.5"]
METRICS = ("sycophantic", "coherence")
VECTOR = "sycophantic_float32"
LAYER = 16


def t_ci(x) -> dict:
    """Mean and 95% t-interval of the values."""
    x = np.asarray(x, dtype=np.float64)
    half = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return {"mean": float(x.mean()), "ci_low": float(x.mean() - half), "ci_high": float(x.mean() + half), "n": len(x)}


def per_question(rollouts: Path, scores: Path, metric: str) -> dict[int, list[float]]:
    """{question index: scores of its rollouts}, None scores left out."""
    rows, by_id = read_jsonl(rollouts), {s["id"]: s for s in read_jsonl(scores)}
    if set(by_id) != {r["id"] for r in rows}:
        raise AssertionError(f"{scores} ids differ from {rollouts}")
    out = defaultdict(list)
    for r in rows:
        if by_id[r["id"]][metric] is not None:
            out[r["question_index"]].append(by_id[r["id"]][metric])
    return out


def question_proj(path: Path) -> dict[int, np.ndarray]:
    """{question index: mean projection over its rollouts, one value per hidden_states index}."""
    by_q = defaultdict(list)
    for r in read_jsonl(path):
        by_q[r["question_index"]].append(r["proj"][VECTOR])
    return {q: np.mean(v, axis=0) for q, v in by_q.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dirs = {"base": (P / "data/rollouts/steer", P / "data/judge/steer")}
    dirs |= {run: (P / "data/rollouts/ft" / run, P / "data/judge/ft" / run) for run in args.runs}
    inputs, out = [], {"cells": CELLS, "models": {}}
    base_proj = question_proj(P / "data/monitor/ft/base.jsonl")
    inputs.append(P / "data/monitor/ft/base.jsonl")
    for model, (rdir, sdir) in dirs.items():
        m = {"eval": {}}
        for cell in CELLS:
            rpath, spath = rdir / f"{cell}.jsonl", sdir / f"{cell}.scores.jsonl"
            inputs += [rpath, spath]
            m["eval"][cell] = {}
            for metric in METRICS:
                qs = per_question(rpath, spath, metric)
                res = {**question_ci(qs), "n_none": len(read_jsonl(rpath)) - sum(map(len, qs.values()))}
                if model != "base":
                    b = per_question(dirs["base"][0] / f"{cell}.jsonl", dirs["base"][1] / f"{cell}.scores.jsonl", metric)
                    res["diff_vs_base"] = t_ci([np.mean(qs[q]) - np.mean(b[q]) for q in sorted(qs) if qs[q] and b[q]])
                m["eval"][cell][metric] = res
        if model != "base":
            ppath = P / "data/monitor/ft" / f"{model}.jsonl"
            inputs.append(ppath)
            proj = question_proj(ppath)
            if set(proj) != set(base_proj):
                raise AssertionError(f"{ppath}: questions differ from base")
            diffs = np.array([proj[q] - base_proj[q] for q in sorted(proj)])  # (questions, 33)
            m["proj_shift"] = {"layer": LAYER, **t_ci(diffs[:, LAYER]),
                               "by_layer": diffs.mean(axis=0).tolist()}
            log = [r for r in read_json(P / "adapters" / model / "train_log.json") if "loss" in r]
            inputs.append(P / "adapters" / model / "train_log.json")
            m["train"] = {"n_steps": len(log), "loss_first10": float(np.mean([r["loss"] for r in log[:10]])),
                          "loss_last50": float(np.mean([r["loss"] for r in log[-50:]]))}
        if model == "base":
            m["proj"] = t_ci([v[LAYER] for v in base_proj.values()])
        out["models"][model] = m
    write_json(args.out, out)
    write_meta(args.out, inputs, args, {"n_models": len(dirs)})
    for model, m in out["models"].items():
        line = " ".join(f"{c}:{m['eval'][c]['sycophantic']['mean']:.0f}/{m['eval'][c]['coherence']['mean']:.0f}"
                        for c in CELLS)
        extra = "" if model == "base" else f" proj {m['proj_shift']['mean']:+.2f} loss {m['train']['loss_last50']:.3f}"
        print(f"{model:18s} {line}{extra}")


if __name__ == "__main__":
    main()
