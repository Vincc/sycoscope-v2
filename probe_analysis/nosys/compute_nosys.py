"""Compare contrastive probes trained with and without the system prompt in context, on the same responses and split.

Probes: each sweep's logistic probes at the C with the best validation AUROC per pair x position x block (ties go to
the smaller C), and its DiM probes. Writes one JSON with
  parity        the with-system cache against the earlier get_activations npz, and the retrained with-system sweep
                against the earlier sweep (same rows, split and grid)
  null_check    largest within-prompt spread of no-system prompt_last vectors per block (identical prompts)
  in_dist       test AUROC and paired win rate per sweep, pair, method and block (response_mean)
  transfer      each sweep's probes scored on the other condition's test rows
  cosine        cosine between the two sweeps' directions per pair, method and block
  likelihood    test-row AUROC of response NLL and next-token agreement for the label, in each condition, and the
                Spearman correlation of each probe's test scores with NLL
  ood           AUROC of every selected probe on every benchmark unit (generalization_matrix.unit_rows), with a
                group-bootstrap 95% interval at --ci-block

Run from the repo root: python -m probe_analysis.nosys.compute_nosys --sys-sweep ... --nosys-sweep ... --output ...
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from interp.cache_activations import load_for_probes
from interp.generalization_matrix import bootstrap_auroc_ci, position_dir, unit_rows
from probe_training.get_activations import skyline_group
from utils import probes
from utils.activations import act_key
from utils.io import read_json, read_jsonl, write_json, write_meta


def selected_probes(sweep: Path) -> dict[tuple, dict]:
    """{(pair, method, position, layer): manifest row}: DiM, and logistic at its best validation AUROC."""
    best = {}
    for m in read_jsonl(sweep / "manifest.jsonl"):
        if m["val_auroc"] is None:
            raise ValueError(f"{m['probe_id']}: no validation AUROC")
        key = (m["pair_index"], m["method"], m["position"], m["layer"])
        old = best.get(key)
        if old is None or m["val_auroc"] > old["val_auroc"] or (m["val_auroc"] == old["val_auroc"] and m["C"] < old["C"]):
            best[key] = m
    files = {}
    for m in best.values():
        if m["file"] not in files:
            files[m["file"]] = np.load(sweep / m["file"])
        m["probe"] = probes.load_probe(files[m["file"]], m["probe_id"], m["method"])
    return best


def within_prompt_spread(x: np.ndarray, prompt_ids: np.ndarray) -> float:
    """Largest relative L2 distance of a row's vector from the mean vector of its prompt's rows."""
    _, group = np.unique(prompt_ids, return_inverse=True)
    sums = np.zeros((group.max() + 1, x.shape[1]))
    np.add.at(sums, group, x)
    means = sums / np.bincount(group)[:, None]
    return float((np.linalg.norm(x - means[group], axis=1) / np.linalg.norm(means[group], axis=1)).max())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sys-sweep", type=Path, required=True)
    parser.add_argument("--nosys-sweep", type=Path, required=True)
    parser.add_argument("--reference-sweep", type=Path, required=True, help="Earlier with-system sweep, from the npz.")
    parser.add_argument("--reference-npz", type=Path, required=True, help="The npz the reference sweep was trained on.")
    parser.add_argument("--rows", type=Path, required=True, help="Contrastive JSONL both caches were made from.")
    parser.add_argument("--sys-cache", type=Path, required=True)
    parser.add_argument("--nosys-cache", type=Path, required=True)
    parser.add_argument("--judged", type=Path, nargs="+", required=True)
    parser.add_argument("--judge-name", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Cache root holding the judged rows' responses.")
    parser.add_argument("--ci-block", type=int, required=True)
    parser.add_argument("--bootstrap", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sel = {"sys": selected_probes(args.sys_sweep), "nosys": selected_probes(args.nosys_sweep)}
    split = read_json(args.sys_sweep / "split.json")
    for other in (args.nosys_sweep, args.reference_sweep):
        o = read_json(other / "split.json")
        if any(o[k] != split[k] for k in ("train", "val", "test", "group_field")):
            raise AssertionError(f"{other} has a different split from {args.sys_sweep}")
    layers = sorted({k[3] for k in sel["nosys"]})
    pairs = sorted({k[0] for k in sel["nosys"]})
    rows = read_jsonl(args.rows)
    out = {"layers": layers, "pairs": {}, "selection": "logistic C by best val AUROC (ties: smaller C); DiM as is"}

    # Training activations, both conditions, same row order
    z = {}
    for cond, cache in (("sys", args.sys_cache), ("nosys", args.nosys_cache)):
        z[cond], info = load_for_probes(cache, rows, ["response_mean"], layers)
        out[f"{cond}_cache"] = {k: v for k, v in info.items() if k != "part_metas"}
    if z["sys"]["id"].tolist() != z["nosys"]["id"].tolist():
        raise AssertionError("the two caches hold different rows")
    ids, y_all, pair_of, pid = z["sys"]["id"], z["sys"]["label"].astype(int), z["sys"]["pair_index"], z["sys"]["prompt_id"]
    out["pairs"] = {int(p): str(z["sys"]["cell"][pair_of == p][0]) for p in pairs}
    test = np.isin(pid, split["test"])

    # Parity with the earlier extraction and sweep
    ref = np.load(args.reference_npz)
    ref_index = {r: i for i, r in enumerate(ref["id"].tolist())}
    order = np.array([ref_index[r] for r in ids.tolist()])
    parity = {"activations": {}, "sweep": {}}
    for L in layers:
        a, b = z["sys"][act_key("response_mean", L)], ref[act_key("response", L)][order]
        rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
        cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        parity["activations"][L] = {"max_rel_l2": float(rel.max()), "median_rel_l2": float(np.median(rel)),
                                    "min_cos": float(cos.min())}
    reference = {(m["pair_index"], m["method"], m["layer"], m["C"]): m for m in read_jsonl(args.reference_sweep / "manifest.jsonl")
                 if m["position"] == "response"}
    diffs = [abs(m["test_auroc"] - reference[(m["pair_index"], m["method"], m["layer"], m["C"])]["test_auroc"])
             for m in read_jsonl(args.sys_sweep / "manifest.jsonl") if m["position"] == "response_mean"]
    parity["sweep"] = {"n_probes": len(diffs), "max_abs_test_auroc_diff": float(max(diffs)),
                       "median_abs_test_auroc_diff": float(np.median(diffs))}
    out["parity"] = parity
    del ref
    print("parity", parity, flush=True)

    # Null check: identical prompts give identical prompt_last vectors, up to batching numerics
    zp, _ = load_for_probes(args.nosys_cache, rows, ["prompt_last"], layers)
    out["null_check"] = {L: within_prompt_spread(zp[act_key("prompt_last", L)], zp["prompt_id"]) for L in layers}
    del zp
    print("null check", out["null_check"], flush=True)

    # In-distribution, transfer, cosine, likelihood
    out["in_dist"], out["transfer"], out["cosine"], out["likelihood"] = [], [], [], []
    for p in pairs:
        rows_p = (pair_of == p) & test
        y = y_all[rows_p]
        g = pid[rows_p].tolist()
        for cond in ("sys", "nosys"):
            for stat in ("response_nll", "next_token_agreement"):
                out["likelihood"].append({"pair": int(p), "condition": cond, "stat": stat,
                                          "auroc": probes.auroc(y, z[cond][stat][rows_p])})
        for method in probes.METHODS:
            for L in layers:
                key = (p, method, "response_mean", L)
                ms, mn = sel["sys"][key], sel["nosys"][key]
                for cond, m in (("sys", ms), ("nosys", mn)):
                    out["in_dist"].append({"pair": int(p), "method": method, "layer": L, "condition": cond,
                                           "C": m["C"], "val_auroc": m["val_auroc"], "test_auroc": m["test_auroc"],
                                           "test_paired_win_rate": m["test_paired_win_rate"]})
                    s_own = probes.score(method, m["probe"], z[cond][act_key("response_mean", L)][rows_p])
                    out["likelihood"].append({"pair": int(p), "condition": cond, "method": method, "layer": L,
                                              "stat": "spearman_probe_vs_nll",
                                              "value": spearman(s_own, z[cond]["response_nll"][rows_p])})
                    if abs(probes.auroc(y, s_own) - m["test_auroc"]) > 1e-9:
                        raise AssertionError(f"{m['probe_id']}: rescored test AUROC differs from the manifest")
                    target = "nosys" if cond == "sys" else "sys"
                    s = probes.score(method, m["probe"], z[target][act_key("response_mean", L)][rows_p])
                    win, n_pairs = probes.paired_win_rate(s, y, g)
                    out["transfer"].append({"pair": int(p), "method": method, "layer": L, "trained_on": cond,
                                            "scored_on": target, "auroc": probes.auroc(y, s),
                                            "paired_win_rate": win, "n_pairs": n_pairs})
                out["cosine"].append({"pair": int(p), "method": method, "layer": L,
                                      "cosine": float(ms["probe"]["direction"] @ mn["probe"]["direction"])})

    # Benchmarks
    leaked = set(split["train_ids"]) | set(split["val_ids"])
    out["ood"], out["ood_units"] = [], {}
    for path in args.judged:
        judged = read_jsonl(path)
        units, excluded = unit_rows(judged, args.judge_name)
        zb, info = load_for_probes(position_dir(args.cache_dir, path, "response_mean").parent, judged, ["response_mean"],
                                   layers)
        if leaked & set(zb["id"].tolist()):
            raise AssertionError(f"{path}: probe training or validation rows among the benchmark rows")
        index = {r: i for i, r in enumerate(zb["id"].tolist())}
        group_of = {r["id"]: skyline_group(r) for r in judged}
        for unit, members in sorted(units.items()):
            idx = np.array([index[i] for i, _ in members])
            yb = np.array([v for _, v in members])
            groups = np.array([group_of[i] for i, _ in members])
            out["ood_units"][unit] = {"n_pos": int(yb.sum()), "n_neg": int(len(yb) - yb.sum()),
                                      "excluded": dict(excluded[unit]), "file": path.as_posix()}
            for L in layers:
                X = zb[act_key("response_mean", L)][idx]
                for cond in ("sys", "nosys"):
                    for p in pairs:
                        for method in probes.METHODS:
                            s = probes.score(method, sel[cond][(p, method, "response_mean", L)]["probe"], X)
                            row = {"unit": unit, "condition": cond, "pair": int(p), "method": method, "layer": L,
                                   "auroc": probes.auroc(yb, s)}
                            if L == args.ci_block and row["auroc"] is not None:
                                row["ci_low"], row["ci_high"], _ = bootstrap_auroc_ci(yb, s, groups, args.bootstrap,
                                                                                      args.seed)
                            out["ood"].append(row)
        print(f"{path.name}: {len(units)} units", flush=True)

    write_json(args.output, out)
    inputs = [args.rows, args.reference_npz, *args.judged,
              *(d / f for d in (args.sys_sweep, args.nosys_sweep, args.reference_sweep) for f in ("manifest.jsonl", "split.json"))]
    counts = {"n_training_rows": len(ids), "n_test_rows": int(test.sum()), "ood_units": len(out["ood_units"])}
    write_meta(args.output, inputs, args, counts)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
