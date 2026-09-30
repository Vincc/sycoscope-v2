"""Keep the control-pair rows of a contrastive JSONL whose prompt group is in a sweep's test split.

Run from the repo root: python -m audit.filter_control_rows --input <contrastive jsonl> --split <sweep>/split.json --output ...
"""
import argparse
from collections import Counter
from pathlib import Path

from utils.io import check_counts, read_json, read_jsonl, write_jsonl, write_meta


def select_control_test(rows: list[dict], split: dict) -> tuple[list[dict], Counter]:
    test, other = set(split["test"]), set(split["train"]) | set(split["val"])
    field = split["group_field"]
    kept, excluded = [], Counter()
    for r in rows:
        if r["pair_type"] != "control":
            excluded[f"pair_type_{r['pair_type']}"] += 1
        elif r[field] in test:
            kept.append(r)
        elif r[field] in other:
            excluded["group_in_train_or_val"] += 1
        else:
            raise ValueError(f"{r['id']}: group {r[field]} is in no side of the split")
    return kept, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_jsonl(args.input)
    kept, excluded = select_control_test(rows, read_json(args.split))
    counts = check_counts(len(rows), excluded, len(kept), args.input.name)
    write_jsonl(args.output, kept)
    write_meta(args.output, [args.input, args.split], args, counts,
               {"pairs": sorted({(r["pair_index"], r["cell"]) for r in kept})})
    print(counts)


if __name__ == "__main__":
    main()
