"""AUROC matrix: the sycophancy persona vector and per-dataset DiM and logistic directions, scored on every benchmark label.

Units are (dataset, label) pairs from judged files: ELEPHANT AITA-YTA, OEQ and SS x validation, indirectness and
framing; AITA-NTA-OG verdict_nta; AITA-NTA-FLIP both_nta on its flipped_story rows; Are You Sure syco per source
dataset. A unit keeps the rows whose label is 0 or 1. Every unit is a target; a unit with at least --min-class rows per
class is also a fitted source.
Cells whose source and target come from the same dataset share rows, so they are cross-fitted over grouped folds: the
direction is fit on the source label outside fold k and scores the target label inside fold k. Other cells fit on all
source rows. The persona vector is applied as is (block L uses its hidden_states index L + 1). AUROC is computed on
the pooled scores; at --ci-blocks a group bootstrap of the target rows (scores held fixed) gives a 95% interval.
Writes <output> (one row per source x target x position x block) and its meta file.

Run from the repo root: python -m interp.generalization_matrix --judged ... --persona-vector ... --cache-dir ... --output ...
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from interp.cache_activations import load
from judging.judge_responses import syco_none_reason
from probe_training.get_activations import skyline_group
from utils import probes
from utils.io import REPO_ROOT, check_counts, read_json, read_jsonl, write_jsonl, write_meta

PERSONA_KEY = {"prompt_last": "prompt_last_diff", "prompt_mean": "prompt_avg_diff", "response_mean": "response_avg_diff"}
LAYER_CONVENTION = "block L = hidden_states[L + 1]; the last block is after the final norm"
ELEPHANT_LABELS = {"AITA-NTA-FLIP": ["both_nta"], "AITA-NTA-OG": ["verdict_nta"]}  # other sources: ELEPHANT_METRICS


def unit_rows(rows: list[dict], judge: str) -> tuple[dict[str, list[tuple[str, int]]], dict[str, Counter]]:
    """({unit: [(row id, label)]}, {unit: excluded rows by reason}) for one judged file."""
    units, excluded = defaultdict(list), defaultdict(Counter)
    for r in rows:
        labels = {k.removesuffix(f"__{judge}"): v for k, v in r["labels"].items() if k.endswith(f"__{judge}")}
        if r["benchmark"] == "are_you_sure":
            names = ["syco"]
        elif r["benchmark"] == "elephant":
            if r["source"] == "AITA-NTA-FLIP" and r["prompt_col"] != "flipped_story":  # pair label: scored on the flipped telling
                excluded["AITA-NTA-FLIP/both_nta"]["original_telling"] += 1
                continue
            names = ELEPHANT_LABELS.get(r["source"], ["validation", "indirectness", "framing"])
        else:
            raise ValueError(f"{r['id']}: unsupported benchmark {r['benchmark']!r}")
        for name in names:
            unit = f"{r['source']}/{name}"
            if labels[name] is not None:
                units[unit].append((r["id"], int(labels[name])))
            elif r["benchmark"] == "are_you_sure":
                excluded[unit][syco_none_reason({**r, "labels": labels})] += 1
            else:
                excluded[unit]["truncated" if r["truncated"] else "label_none"] += 1
    return units, excluded


def assign_folds(groups: list[str], k: int, seed: int) -> dict[str, int]:
    """Group -> fold in 0..k-1, from a seeded permutation of the sorted distinct groups."""
    distinct = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(distinct))
    return {distinct[i]: rank % k for rank, i in enumerate(order)}


def bootstrap_auroc_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, int]:
    """95% percentile interval of AUROC over group resamples, and the number of resamples with both classes (the
    others are undefined and left out); ties count half, as in roc_auc_score."""
    distinct, g = np.unique(groups, return_inverse=True)
    counts = np.random.default_rng(seed).multinomial(len(distinct), np.full(len(distinct), 1 / len(distinct)), n_boot)
    weights = counts[:, g].astype(np.float64)  # (n_boot, rows): each row weighted by its group's draw count
    values = weighted_auroc(y, s, weights)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), len(values)


def weighted_auroc(y: np.ndarray, s: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AUROC per weight vector (rows of `weights`); NaN where a class has zero weight."""
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted, w = s[order], y[order], weights[:, order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])  # first row of each run of equal scores
    pos = np.add.reduceat(w * (y_sorted == 1), starts, axis=1)
    neg = np.add.reduceat(w * (y_sorted == 0), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg  # negative weight strictly below each score
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (below + 0.5 * neg)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))


def position_dir(cache_dir: Path, judged: Path, position: str) -> Path:
    """Cache folder of the responses file a judged file was made from (judging/judged/X_judged -> responses/X_responses)."""
    decoding_dir = judged.parent.parent.parent
    responses = decoding_dir / "responses" / judged.name.replace("_judged.jsonl", "_responses.jsonl")
    rel = responses.resolve().relative_to(REPO_ROOT / "generations")
    return cache_dir / Path(*rel.parts).with_suffix("") / position


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", type=Path, nargs="+", required=True)
    parser.add_argument("--judge-name", required=True, help="Label suffix, e.g. sonnet55.")
    parser.add_argument("--persona-vector", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--positions", nargs="+", choices=sorted(PERSONA_KEY), required=True)
    parser.add_argument("--blocks", type=int, nargs="+", required=True, help="Blocks for persona and DiM.")
    parser.add_argument("--logistic-blocks", type=int, nargs="+", required=True)
    parser.add_argument("--ci-blocks", type=int, nargs="+", required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--C", type=float, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--min-class", type=int, required=True, help="Rows per class needed to fit a source.")
    parser.add_argument("--bootstrap", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not set(args.logistic_blocks) | set(args.ci_blocks) <= set(args.blocks):
        raise ValueError("--logistic-blocks and --ci-blocks must be among --blocks")

    # units, groups and folds
    units, excluded, group_of, dataset_of, files_of = {}, {}, {}, {}, {}
    for path in args.judged:
        rows = read_jsonl(path)
        file_units, file_excluded = unit_rows(rows, args.judge_name)
        for r in rows:
            if r["id"] in group_of:
                raise ValueError(f"{r['id']} appears twice in the judged files")
            group_of[r["id"]] = f"{r['source']}::{skyline_group(r)}"
        for unit, members in file_units.items():
            if unit in units:
                raise ValueError(f"unit {unit} appears in two judged files")
            units[unit], excluded[unit], files_of[unit] = members, file_excluded[unit], path
            dataset_of[unit] = unit.split("/")[0]
    fold_of = {}
    for dataset in sorted(set(dataset_of.values())):
        ids = sorted({i for u, m in units.items() if dataset_of[u] == dataset for i, _ in m})
        folds = assign_folds([group_of[i] for i in ids], args.folds, args.seed)
        fold_of.update({i: folds[group_of[i]] for i in ids})
    unit_counts, sources = {}, []
    for unit in sorted(units):
        y = np.array([v for _, v in units[unit]])
        folds = np.array([fold_of[i] for i, _ in units[unit]])
        n_out = len(y)
        unit_counts[unit] = {**check_counts(n_out + sum(excluded[unit].values()), excluded[unit], n_out, unit),
                             "n_pos": int(y.sum()), "n_neg": int(n_out - y.sum()),
                             "n_groups": len({group_of[i] for i, _ in units[unit]})}
        # a source needs --min-class rows of each class in the full fit and in every fold's training rows
        train_minority = [min(int(y[folds != k].sum()), int((folds != k).sum() - y[folds != k].sum()))
                          for k in range(args.folds)]
        if min(unit_counts[unit]["n_pos"], unit_counts[unit]["n_neg"], *train_minority) >= args.min_class:
            sources.append(unit)
    print({u: (c["n_pos"], c["n_neg"]) for u, c in unit_counts.items()}, f"sources: {len(sources)}", flush=True)

    persona = np.load(args.persona_vector)
    persona_meta = read_json(args.persona_vector.parent / "meta" / f"{args.persona_vector.name}.meta.json")
    persona_model = (persona_meta["model"], persona_meta["revision"])
    out, rng_seed = [], args.seed
    chunks = [args.blocks[i:i + 2] for i in range(0, len(args.blocks), 2)]  # blocks loaded together
    pdirs = {(path, p): position_dir(args.cache_dir, path, p) for path in set(files_of.values()) for p in args.positions}
    for pdir in pdirs.values():  # check model and files once; later loads skip the hash
        for meta in map(read_json, sorted((pdir / "meta").glob("part-*.meta.json"))):
            if meta["layer_convention"] != LAYER_CONVENTION or (meta["model"], meta["model_revision"]) != persona_model:
                raise ValueError(f"{pdir}: {meta['model']}@{meta['model_revision']}, {meta['layer_convention']!r}; "
                                 f"persona vector is {persona_model}")
        load(pdir, [])
    for position, chunk in [(p, c) for p in args.positions for c in chunks]:
        cache = {path: load(pdirs[path, position], chunk, verify=False) for path in sorted(set(files_of.values()))}
        for block in chunk:
            index = {}  # row id -> activation
            for path, (ids, acts) in cache.items():
                index.update(zip(ids, acts[block]))
            X = {u: np.stack([index[i] for i, _ in m]) for u, m in units.items()}
            Y = {u: np.array([v for _, v in m]) for u, m in units.items()}
            F = {u: np.array([fold_of[i] for i, _ in m]) for u, m in units.items()}
            G = {u: np.array([group_of[i] for i, _ in m]) for u, m in units.items()}

            def cell(kind, source, target, mode, scores):
                y = Y[target]
                row = {"position": position, "block": block, "kind": kind, "source": source, "target": target,
                       "mode": mode, "auroc": probes.auroc(y, scores), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
                if block in args.ci_blocks and row["auroc"] is not None:
                    row["ci_low"], row["ci_high"], row["ci_resamples"] = bootstrap_auroc_ci(y, scores, G[target],
                                                                                           args.bootstrap, rng_seed)
                out.append(row)

            v = persona[PERSONA_KEY[position]][block + 1].astype(np.float64)
            for target in sorted(units):
                cell("persona", "persona_vector", target, "zero_shot", X[target] @ (v / np.linalg.norm(v)))
            methods = ["dim"] + (["logistic"] if block in args.logistic_blocks else [])
            for method in methods:
                for source in sources:
                    full = probes.fit(method, X[source], Y[source], args.C, args.max_iter)
                    same = [t for t in sorted(units) if dataset_of[t] == dataset_of[source]]
                    pooled = {t: np.empty(len(Y[t])) for t in same}
                    for k in range(args.folds):
                        train = F[source] != k
                        probe = probes.fit(method, X[source][train], Y[source][train], args.C, args.max_iter)
                        for t in same:
                            held = F[t] == k
                            pooled[t][held] = probes.score(method, probe, X[t][held])
                    for target in sorted(units):
                        if target in pooled:
                            cell(method, source, target, "cross_fit", pooled[target])
                        else:
                            cell(method, source, target, "transfer", probes.score(method, full, X[target]))
            print(f"{position} block {block}: {len(out)} cells", flush=True)

    write_jsonl(args.output, out)
    n_cells = len(out)
    write_meta(args.output, [*args.judged, args.persona_vector], args, {"n_in": n_cells, "excluded": {}, "n_out": n_cells},
               {"units": unit_counts, "sources": sources})
    print(f"Wrote {n_cells} cells to {args.output}")


if __name__ == "__main__":
    main()
