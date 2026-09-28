"""Group-level train/test split over the union of a field's values across row files."""
import argparse
from pathlib import Path

from utils.io import read_jsonl, write_json, write_meta
from utils.splits import group_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--group-field", required=True, help="prompt_id for contrastive rows, group_id for benchmark rows.")
    parser.add_argument("--test-frac", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    groups = set()
    n_rows = 0
    for path in args.rows:
        rows = read_jsonl(path)
        n_rows += len(rows)
        groups.update(r[args.group_field] for r in rows)
    split = group_split(sorted(groups), args.test_frac, args.seed)
    split["group_field"] = args.group_field
    write_json(args.out, split)
    counts = {"n_rows": n_rows, "n_groups": len(groups), "n_train_groups": len(split["train"]), "n_test_groups": len(split["test"])}
    write_meta(args.out, args.rows, args, counts)
    print(counts)


if __name__ == "__main__":
    main()
