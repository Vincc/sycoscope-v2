"""Cosine similarity of benchmark label directions with a sweep's contrastive (system-prompt) probe directions.

Units are those of interp.generalization_matrix. A unit with at least --min-class rows per class is fitted on all its
rows: DiM at every block and logistic (--C) at --logistic-blocks. The sweep's contrastive DiM and logistic (--probe-C)
probes are read at their own position and block. Every direction is a unit vector toward label 1 (for a contrastive
probe, the pair's "sycophantic" prompt). At each position and block the output has the cosine of every pair of
directions, benchmark directions first. For a benchmark DiM paired with a contrastive direction, the unit's DiM is
refitted on --permutations shuffles of its labels (class sizes kept), giving the cosine of a DiM with no label
information: the 95th percentile of its absolute value and a two-sided permutation p-value.
Writes <output> and its meta file.

Run from the repo root: python -m interp.direction_cosine --judged ... --cache-dir ... --probe-sweep ... --output ...
"""
import argparse
from pathlib import Path

import numpy as np

from interp.cache_activations import load
from interp.generalization_matrix import LAYER_CONVENTION, contrastive_probes, position_dir, unit_rows
from utils import probes
from utils.io import check_counts, read_json, read_jsonl, write_jsonl, write_meta

POSITIONS = ("prompt_last", "response_first5", "response_mean")  # cache positions that have contrastive probes


def dim_directions(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Unit DiM direction (label-1 mean minus label-0 mean) for each label vector in the rows of Y."""
    n_pos = Y.sum(axis=1, keepdims=True)
    W = np.where(Y == 1, 1 / n_pos, -1 / (Y.shape[1] - n_pos))  # weights that turn W @ X into mean differences
    D = W @ X.astype(np.float64)
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", type=Path, nargs="+", required=True)
    parser.add_argument("--judge-name", required=True, help="Label suffix, e.g. sonnet55.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--positions", nargs="+", choices=POSITIONS, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", required=True)
    parser.add_argument("--logistic-blocks", type=int, nargs="+", required=True)
    parser.add_argument("--C", type=float, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--min-class", type=int, required=True, help="Rows per class needed to fit a unit.")
    parser.add_argument("--probe-sweep", type=Path, required=True)
    parser.add_argument("--probe-C", type=float, required=True)
    parser.add_argument("--permutations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not set(args.logistic_blocks) <= set(args.blocks):
        raise ValueError("--logistic-blocks must be among --blocks")

    # units
    units, excluded, files_of, seen = {}, {}, {}, set()
    for path in args.judged:
        rows = read_jsonl(path)
        ids = {r["id"] for r in rows}
        if len(ids) != len(rows) or ids & seen:
            raise ValueError(f"{path}: row ids repeat within or across judged files")
        seen |= ids
        file_units, file_excluded = unit_rows(rows, args.judge_name)
        decoding = path.resolve().parent.parent.parent.name  # generations/<model>/<decoding>/judging/judged/
        for name, members in file_units.items():
            unit = f"{decoding}:{name}"
            if unit in units:
                raise ValueError(f"unit {unit} appears in two judged files")
            units[unit], excluded[unit], files_of[unit] = members, file_excluded[name], path
    unit_counts, sources = {}, []
    for unit in sorted(units):
        y = np.array([v for _, v in units[unit]])
        unit_counts[unit] = {**check_counts(len(y) + sum(excluded[unit].values()), excluded[unit], len(y), unit),
                             "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
        if min(unit_counts[unit]["n_pos"], unit_counts[unit]["n_neg"]) >= args.min_class:
            sources.append(unit)
    print({u: (c["n_pos"], c["n_neg"]) for u, c in unit_counts.items()}, f"fitted: {len(sources)}", flush=True)

    # activations: one model, the layer convention of the probes, hashes checked once
    files = sorted(set(files_of.values()))
    pdirs = {(path, p): position_dir(args.cache_dir, path, p) for path in files for p in args.positions}
    models = set()
    for pdir in pdirs.values():
        for meta in map(read_json, sorted((pdir / "meta").glob("part-*.meta.json"))):
            if meta["layer_convention"] != LAYER_CONVENTION:
                raise ValueError(f"{pdir}: layer convention {meta['layer_convention']!r}")
            models.add((meta["model"], meta["model_revision"]))
        load(pdir, [])
    if len(models) != 1:
        raise ValueError(f"activations come from {len(models)} models: {sorted(models)}")
    probe_sets, probe_counts = contrastive_probes(args.probe_sweep, args.probe_C, next(iter(models))[0])
    print(f"{args.probe_sweep}: {probe_counts}", flush=True)

    rng = np.random.default_rng(args.seed)
    out = []
    for position in args.positions:
        for block in args.blocks:
            contrastive = probe_sets.get((position, block))
            if not contrastive:
                raise ValueError(f"{args.probe_sweep} has no contrastive probes at {position} block {block}")
            loaded = [load(pdirs[path, position], [block], verify=False) for path in files]
            A = np.concatenate([acts[block] for _, acts in loaded])
            row_of = {i: k for k, i in enumerate(i for ids, _ in loaded for i in ids)}
            if len(row_of) != len(A):
                raise ValueError(f"{position}: cached row ids repeat")
            del loaded

            directions, nulls = [], {}  # directions: (kind, source, method, unit vector)
            for unit in sources:
                X = A[[row_of[i] for i, _ in units[unit]]]
                y = np.array([v for _, v in units[unit]])
                directions.append(("benchmark", unit, "dim", probes.fit_dim(X, y)["direction"]))
                if block in args.logistic_blocks:
                    probe = probes.fit_logistic(X, y, args.C, args.max_iter)
                    directions.append(("benchmark", unit, "logistic", probe["direction"]))
                nulls[unit] = dim_directions(X, np.stack([rng.permutation(y) for _ in range(args.permutations)]))
            for source, method, probe in contrastive:
                directions.append(("contrastive", source, method, probe["direction"].astype(np.float64)))
            M = np.stack([d for *_, d in directions])
            if not np.allclose(np.linalg.norm(M, axis=1), 1.0):
                raise AssertionError("a direction is not a unit vector")
            cos = M @ M.T

            for a in range(len(directions)):
                for b in range(a + 1, len(directions)):
                    (kind_a, src_a, met_a, _), (kind_b, src_b, met_b, v_b) = directions[a], directions[b]
                    row = {"position": position, "block": block, "a_kind": kind_a, "a_source": src_a,
                           "a_method": met_a, "b_kind": kind_b, "b_source": src_b, "b_method": met_b,
                           "cosine": float(cos[a, b])}
                    if kind_a == "benchmark" and met_a == "dim" and kind_b == "contrastive":
                        null = np.abs(nulls[src_a] @ v_b)
                        row["null_abs_p95"] = float(np.percentile(null, 95))
                        row["perm_p"] = float((1 + (null >= abs(row["cosine"])).sum()) / (1 + len(null)))
                    out.append(row)
            print(f"{position} block {block}: {len(directions)} directions, {len(out)} pairs so far", flush=True)

    write_jsonl(args.output, out)
    probe_inputs = [args.probe_sweep / "manifest.jsonl", *sorted(args.probe_sweep.glob("*.probes.npz"))]
    write_meta(args.output, [*args.judged, *probe_inputs], args, {"n_in": len(out), "excluded": {}, "n_out": len(out)},
               {"units": unit_counts, "fitted": sources, "probes": probe_counts})
    print(f"Wrote {len(out)} pairs to {args.output}")


if __name__ == "__main__":
    main()
