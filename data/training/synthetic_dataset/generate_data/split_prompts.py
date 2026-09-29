"""Split prompts.jsonl into probe-training prompt files: one per combination (5 files), named
synthetic_<components>.jsonl, where combinations with a fact hold both versions; plus one per fact version of those
combinations (6 files), named synthetic_<components>_<fact>.jsonl.

Output rows follow perez_user_prompts.jsonl (prompt_id, source, user_prompt), the probe_training.generate_response
input, plus the scenario fields and fact ("true", "false", or None without a fact).

Usage (from the repo root):
    uv run python data/training/synthetic_dataset/generate_data/split_prompts.py \
        --input data/training/synthetic_dataset/prompts.jsonl --out-dir data/training/synthetic_dataset/splits
"""
import argparse
from collections import defaultdict
from pathlib import Path

from utils.io import check_counts, read_jsonl, write_jsonl, write_meta


def dataset_names(row):
    """synthetic_<components>, and for prompts with a fact also synthetic_<components>_<fact>."""
    has_fact = "fact" in row["components"]
    if has_fact != (row["fact"] is not None) or row["fact"] not in (None, "true", "false"):
        raise ValueError(f"{row['id']}: fact={row['fact']!r} does not match components {row['components']}")
    name = "synthetic_" + "_".join(row["components"])
    return [name, f"{name}_{row['fact']}"] if has_fact else [name]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{args.input}: duplicate ids")

    datasets = defaultdict(list)
    for r in rows:
        for name in dataset_names(r):
            datasets[name].append({
                "prompt_id": r["id"],
                "source": name,
                "user_prompt": r["prompt"],
                "scenario_id": r["scenario_id"],
                "domain": r["domain"],
                "trait_type": r["trait_type"],
                "emotion_type": r["emotion_type"],
                "components": r["components"],
                "fact": r["fact"],
            })
    n_fact = sum(r["fact"] is not None for r in rows)
    if sum(len(d) for d in datasets.values()) != len(rows) + n_fact:  # fact rows are in two files
        raise AssertionError("rows lost in split")

    counts = {}
    for name, out_rows in sorted(datasets.items()):
        merged = out_rows[0]["fact"] is not None and not name.endswith(("_true", "_false"))
        per_scenario = 2 if merged else 1  # merged files hold the true and false versions
        if len({r["scenario_id"] for r in out_rows}) * per_scenario != len(out_rows):
            raise AssertionError(f"{name}: expected {per_scenario} prompt(s) per scenario")
        out_path = args.out_dir / f"{name}.jsonl"
        counts[out_path.name] = check_counts(len(rows), {"other dataset": len(rows) - len(out_rows)}, len(out_rows), name)
        write_jsonl(out_path, out_rows)
        print(f"{len(out_rows):5d}  {out_path}")
    write_meta(args.out_dir, [args.input], args, counts)  # one meta for the directory: <out-dir>.meta.json


if __name__ == "__main__":
    main()
