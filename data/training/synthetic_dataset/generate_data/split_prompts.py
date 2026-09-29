"""Split prompts.jsonl into probe-training prompt files, one per combination (5 files) named synthetic_<components>.jsonl.
Combinations with a fact hold both versions, labelled by fact ("true"/"false"); otherwise fact is None.

Output rows follow perez_user_prompts.jsonl (prompt_id, source, user_prompt), the probe_training.generate_response
input, plus the scenario fields and fact.

Usage (from the repo root):
    uv run python data/training/synthetic_dataset/generate_data/split_prompts.py \
        --input data/training/synthetic_dataset/prompts.jsonl --out-dir data/training/synthetic_dataset/splits
"""
import argparse
from collections import defaultdict
from pathlib import Path

from utils.io import check_counts, read_jsonl, write_jsonl, write_meta


def dataset_name(row):
    """synthetic_<components>."""
    has_fact = "fact" in row["components"]
    if has_fact != (row["fact"] is not None) or row["fact"] not in (None, "true", "false"):
        raise ValueError(f"{row['id']}: fact={row['fact']!r} does not match components {row['components']}")
    return "synthetic_" + "_".join(row["components"])


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
        name = dataset_name(r)
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
    if sum(len(d) for d in datasets.values()) != len(rows):
        raise AssertionError("rows lost in split")

    counts = {}
    for name, out_rows in sorted(datasets.items()):
        per_scenario = 2 if out_rows[0]["fact"] is not None else 1  # true and false versions
        if len({r["scenario_id"] for r in out_rows}) * per_scenario != len(out_rows):
            raise AssertionError(f"{name}: expected {per_scenario} prompt(s) per scenario")
        out_path = args.out_dir / f"{name}.jsonl"
        counts[out_path.name] = check_counts(len(rows), {"other dataset": len(rows) - len(out_rows)}, len(out_rows), name)
        write_jsonl(out_path, out_rows)
        print(f"{len(out_rows):5d}  {out_path}")
    write_meta(args.out_dir, [args.input], args, counts)  # one meta for the directory: <out-dir>.meta.json


if __name__ == "__main__":
    main()
