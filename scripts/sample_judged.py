"""Select a reproducible judged-row sample before activation extraction."""
import argparse
import random
from collections import Counter
from pathlib import Path

from utils.io import check_counts, read_jsonl, write_jsonl, write_meta


def sample_indices(rows: list[dict], label: str, strategy: str, n: int | None, seed: int) -> list[int]:
    if strategy != "match_minority" and (n is None or n <= 0):
        raise ValueError("sample size must be positive")
    if strategy == "match_minority" and n is not None:
        raise ValueError("match_minority determines sample size from the minority class; omit --n")
    by_label = {0: [], 1: []}
    for i, row in enumerate(rows):
        value = row["labels"][label]
        if value is None:
            continue
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"{row['id']}: {label} must be 0, 1 or None")
        by_label[value].append(i)
    rng = random.Random(seed)
    if strategy == "match_minority":
        if not by_label[0] or not by_label[1]:
            raise ValueError("match_minority needs both classes")
        per_class = min(len(by_label[0]), len(by_label[1]))
        chosen = (
            (by_label[0] if len(by_label[0]) == per_class else rng.sample(by_label[0], per_class))
            + (by_label[1] if len(by_label[1]) == per_class else rng.sample(by_label[1], per_class))
        )
    elif strategy == "balanced":
        if n % 2:
            raise ValueError("balanced sample size must be even")
        chosen = rng.sample(by_label[0], n // 2) + rng.sample(by_label[1], n // 2)
    elif strategy == "minority_class":
        minority = min(by_label, key=lambda value: len(by_label[value]))
        if len(by_label[0]) == len(by_label[1]):
            raise ValueError("classes have equal size; no minority class")
        chosen = rng.sample(by_label[minority], n)
    else:
        raise ValueError(f"unknown sampling strategy {strategy!r}")
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--strategy", choices=("balanced", "minority_class", "match_minority"), required=True)
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_jsonl(args.input)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{args.input}: duplicate ids")
    chosen = sample_indices(rows, args.label, args.strategy, args.n, args.seed)
    sampled = [rows[i] for i in chosen]
    n_none = sum(r["labels"][args.label] is None for r in rows)
    excluded = Counter({"label_none": n_none, "not_sampled": len(rows) - n_none - len(sampled)})
    counts = check_counts(len(rows), excluded, len(sampled), args.input.name)
    write_jsonl(args.output, sampled)
    label_counts = Counter(r["labels"][args.label] for r in sampled)
    write_meta(args.output, [args.input], args, counts,
               {"label_counts": {str(k): label_counts[k] for k in (0, 1)}})
    print(f"Wrote {args.output}: {counts}, labels {dict(label_counts)}")


if __name__ == "__main__":
    main()
