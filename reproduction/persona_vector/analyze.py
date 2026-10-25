"""All section-2 checks from saved outputs: steering, monitoring, cross-trait, precision, noise floor, audit, cost.

Layers are hidden_states indices (paper layer L = index L = block L-1). Writes reports/results.json.

Run from the repo root: python -m reproduction.persona_vector.analyze --primary-layer 16 --noise-seed 0
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from utils.io import read_json, read_jsonl, write_json, write_meta

ROOT = Path("reproduction/persona_vector")
VEC = ROOT / "vectors"
TRAIT = "sycophantic"
PAPER = {"monitor_overall_r": 0.798, "monitor_within_r": 0.669, "cos_evil": 0.412, "cos_hallucinating": 0.252,
         "best_layer": 16, "audit_monitor_overall_r": 0.954, "audit_monitor_within_r": 0.621}
PRICE_PER_M = {"prompt_tokens": 0.40, "completion_tokens": 1.60}
GPU_PER_HOUR = 1.10
AUDIT_VECTOR = ROOT / "activations/audit_persona.probes.npz"
AUDIT_KEY = "persona__dim__response_L15__direction"
TOLERANCE = 0.1


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def monitoring(rows, scores, projections, name, layer):
    """Points = (condition, question): mean last-prompt projection vs mean trait over its rollouts."""
    by_point = defaultdict(lambda: {"proj": [], "trait": [], "coherence": []})
    for r in rows:
        p = by_point[(r["condition"], r["question_index"])]
        p["proj"].append(projections[r["id"]][name][layer])
        if scores[r["id"]][TRAIT] is not None:
            p["trait"].append(scores[r["id"]][TRAIT])
        if scores[r["id"]]["coherence"] is not None:
            p["coherence"].append(scores[r["id"]]["coherence"])
    points = [{"condition": c, "question_index": q, "proj": float(np.mean(v["proj"])),
               "proj_spread": float(np.ptp(v["proj"])), "trait": float(np.mean(v["trait"])),
               "coherence": float(np.mean(v["coherence"])), "n_trait": len(v["trait"])}
              for (c, q), v in sorted(by_point.items())]
    x, y = np.array([p["proj"] for p in points]), np.array([p["trait"] for p in points])
    per_condition, excluded = {}, []
    for c in sorted({p["condition"] for p in points}):
        xs = np.array([p["proj"] for p in points if p["condition"] == c])
        ys = np.array([p["trait"] for p in points if p["condition"] == c])
        entry = {"n": len(ys), "trait_mean": float(ys.mean()), "trait_sd": float(ys.std(ddof=1)),
                 "proj_mean": float(xs.mean()),
                 "coherence_mean": float(np.mean([p["coherence"] for p in points if p["condition"] == c]))}
        if entry["trait_sd"] < 1:
            excluded.append(c)
            entry["r"] = None
        else:
            entry["r"] = float(stats.pearsonr(xs, ys)[0])
        per_condition[c] = entry
    within = [e["r"] for e in per_condition.values() if e["r"] is not None]
    return {"vector": name, "layer": layer, "n_points": len(points), "overall_r": float(stats.pearsonr(x, y)[0]),
            "within_r": float(np.mean(within)) if within else None, "conditions_excluded_sd_below_1": excluded,
            "per_condition": per_condition, "points": points}


def noise_floor(acts_path, seed):
    """Per-layer cosine between response_avg vectors built from two random halves of the extraction questions."""
    z = np.load(acts_path)
    questions = sorted(set(z["question_index"].tolist()))
    perm = np.random.default_rng(seed).permutation(len(questions))
    half_a = {questions[i] for i in perm[: len(questions) // 2]}
    in_a = np.isin(z["question_index"], list(half_a))
    acts, is_pos = z["response_avg"], z["is_pos"].astype(bool)
    vec = {}
    for name, m in (("a", in_a), ("b", ~in_a)):
        if not (m & is_pos).any() or not (m & ~is_pos).any():
            raise ValueError(f"half {name} lacks a class")
        vec[name] = acts[m & is_pos].mean(0) - acts[m & ~is_pos].mean(0)
    return {"seed": seed, "questions_a": sorted(half_a), "questions_b": sorted(set(questions) - half_a),
            "n_a": int(in_a.sum()), "n_b": int((~in_a).sum()),
            "cos_by_layer": [cos(vec["a"][l], vec["b"][l]) for l in range(1, acts.shape[1])]}


def judge_cost():
    usage, n = {"prompt_tokens": 0, "completion_tokens": 0}, 0
    for path in sorted((ROOT / "data/judge").rglob("*.responses.jsonl")):
        for row in read_jsonl(path):
            n += 1
            for k in usage:
                usage[k] += row["usage"][k]
    return {"n_calls": n, **usage, "usd": sum(usage[k] * PRICE_PER_M[k] / 1e6 for k in usage)}


def gpu_seconds():
    """wall_seconds recorded by the GPU scripts' meta files."""
    out = {}
    for path in sorted(ROOT.rglob("meta/*.meta.json")):
        meta = read_json(path)
        if "wall_seconds" in meta:
            out[str(path)] = meta["wall_seconds"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary-layer", type=int, required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    args = parser.parse_args()
    L = args.primary_layer
    inputs = []

    vec_paths = {k: VEC / f"{k}.npz" for k in ("sycophantic_float32", "sycophantic_bfloat16", "evil_float32",
                                               "hallucinating_float32")}
    vec = {k: np.load(p)["response_avg_diff"] for k, p in vec_paths.items()}
    inputs += list(vec_paths.values())
    n_layers = vec["sycophantic_float32"].shape[0]

    steering = read_json(ROOT / "reports/steering_cells.json")
    inputs.append(ROOT / "reports/steering_cells.json")
    best_layer = steering["best"]["layer"]
    layers = sorted({L, best_layer})

    extraction = {}
    for k, p in vec_paths.items():
        meta = read_json(p.parent / "meta" / f"{p.name}.meta.json")
        extraction[k] = {"counts": meta["counts"], "boundary_merges": meta["boundary_merges"],
                         "norm_by_layer": [float(np.linalg.norm(v)) for v in vec[k]]}

    mon_rows = read_jsonl(ROOT / "data/rollouts/monitor.jsonl")
    mon_scores = {s["id"]: s for s in read_jsonl(ROOT / "data/judge/monitor.scores.jsonl")}
    mon_proj = {r["id"]: r["proj"] for r in read_jsonl(ROOT / "data/monitor/projections.jsonl")}
    inputs += [ROOT / "data/rollouts/monitor.jsonl", ROOT / "data/judge/monitor.scores.jsonl",
               ROOT / "data/monitor/projections.jsonl"]
    if set(mon_scores) != {r["id"] for r in mon_rows} or set(mon_proj) != set(mon_scores):
        raise AssertionError("monitoring rollouts, scores and projections have different ids")
    monitor = {name: {l: monitoring(mon_rows, mon_scores, mon_proj, name, l) for l in layers}
               for name in ("sycophantic_float32", "sycophantic_bfloat16")}
    monitor_by_layer = {name: [{k: monitoring(mon_rows, mon_scores, mon_proj, name, l)[k]
                                for k in ("overall_r", "within_r")} for l in range(1, n_layers)]
                        for name in monitor}

    cross = {t: {"cos_primary": cos(vec["sycophantic_float32"][L], vec[f"{t}_float32"][L]),
                 "cos_by_layer": [cos(vec["sycophantic_float32"][l], vec[f"{t}_float32"][l]) for l in range(1, n_layers)]}
             for t in ("evil", "hallucinating")}
    precision = {"cos_float32_bfloat16_by_layer": [cos(vec["sycophantic_float32"][l], vec["sycophantic_bfloat16"][l])
                                                   for l in range(1, n_layers)]}
    acts = ROOT / "activations/sycophantic_float32_acts.npz"
    noise = noise_floor(acts, args.noise_seed)
    inputs += [acts, AUDIT_VECTOR]
    audit_dir = np.load(AUDIT_VECTOR)[AUDIT_KEY]
    audit = {"audit_block": 15, "our_index": 16, "cos_float32": cos(vec["sycophantic_float32"][16], audit_dir),
             "cos_bfloat16": cos(vec["sycophantic_bfloat16"][16], audit_dir), "noise_floor_cos": noise["cos_by_layer"][15],
             "audit_source": "origin/audit/existing-methods d81f1d6 probes/Llama-3.1-8B-Instruct/audit_persona/persona.probes.npz"}

    m = monitor["sycophantic_float32"][L]
    verdicts = [
        {"check": "best steering layer", "paper": PAPER["best_layer"], "ours": best_layer,
         "reproduced": best_layer == PAPER["best_layer"] or PAPER["best_layer"] in steering["layers_overlapping_best_ci"],
         "note": "peak resolved" if steering["peak_resolved"] else f"peak unresolved: CI overlaps layers {steering['layers_overlapping_best_ci']}"},
        {"check": "monitoring r, overall", "paper": PAPER["monitor_overall_r"], "ours": m["overall_r"],
         "reproduced": abs(m["overall_r"] - PAPER["monitor_overall_r"]) <= TOLERANCE},
        {"check": "monitoring r, within condition", "paper": PAPER["monitor_within_r"], "ours": m["within_r"],
         "reproduced": m["within_r"] is not None and abs(m["within_r"] - PAPER["monitor_within_r"]) <= TOLERANCE},
        {"check": "cos(sycophantic, evil), layer 16", "paper": PAPER["cos_evil"], "ours": cross["evil"]["cos_primary"],
         "reproduced": abs(cross["evil"]["cos_primary"] - PAPER["cos_evil"]) <= TOLERANCE},
        {"check": "cos(sycophantic, hallucinating), layer 16", "paper": PAPER["cos_hallucinating"],
         "ours": cross["hallucinating"]["cos_primary"],
         "reproduced": abs(cross["hallucinating"]["cos_primary"] - PAPER["cos_hallucinating"]) <= TOLERANCE},
        {"check": "audit vector cosine vs noise floor (layer 16)", "paper": None, "ours": audit["cos_float32"],
         "reference": audit["noise_floor_cos"], "reproduced": audit["cos_float32"] >= audit["noise_floor_cos"],
         "note": "reproduced = cosine to the audit vector at least the split-half noise floor"},
        {"check": "audit monitoring r, overall", "paper": PAPER["audit_monitor_overall_r"], "ours": m["overall_r"],
         "reproduced": abs(m["overall_r"] - PAPER["audit_monitor_overall_r"]) <= TOLERANCE, "note": "reference is the audit, not the paper"},
        {"check": "audit monitoring r, within condition", "paper": PAPER["audit_monitor_within_r"], "ours": m["within_r"],
         "reproduced": m["within_r"] is not None and abs(m["within_r"] - PAPER["audit_monitor_within_r"]) <= TOLERANCE,
         "note": "reference is the audit, not the paper"},
        {"check": "precision: cos(float32, bf16 vector), layer 16", "paper": None,
         "ours": precision["cos_float32_bfloat16_by_layer"][L - 1], "reproduced": None,
         "note": f"bf16-vector monitoring r overall {monitor['sycophantic_bfloat16'][L]['overall_r']:.3f}, "
                 f"within {monitor['sycophantic_bfloat16'][L]['within_r']:.3f}"},
        {"check": "noise floor: split-half cosine, layer 16", "paper": None, "ours": noise["cos_by_layer"][L - 1],
         "reproduced": None},
    ]
    gpu = gpu_seconds()
    cost = judge_cost()
    results = {"primary_layer": L, "best_layer": best_layer, "tolerance": TOLERANCE, "paper": PAPER,
               "verdicts": verdicts, "extraction": extraction, "steering": steering,
               "monitoring": {n: {str(l): v for l, v in d.items()} for n, d in monitor.items()},
               "monitoring_by_layer": monitor_by_layer, "cross_trait": cross, "precision": precision,
               "noise_floor": noise, "audit": audit, "judge_cost": cost,
               "gpu": {"seconds_by_output": gpu, "hours": sum(gpu.values()) / 3600,
                       "usd": sum(gpu.values()) / 3600 * GPU_PER_HOUR}}
    out = ROOT / "reports/results.json"
    write_json(out, results)
    write_meta(out, inputs, args, {"n_verdicts": len(verdicts)})
    for v in verdicts:
        print(f"{v['check']}: paper {v['paper']} ours {v['ours']} reproduced {v['reproduced']}")
    print(f"judge ${cost['usd']:.2f} over {cost['n_calls']} calls; GPU {results['gpu']['hours']:.2f} h")


if __name__ == "__main__":
    main()
