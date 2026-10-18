"""Label warranted praise among high-quality SyPR responses."""
import argparse
from collections import Counter
from pathlib import Path

from utils.io import check_counts, read_jsonl, write_jsonl, write_meta


def warranted_praise(row: dict) -> int | None:
    poor = row["is_poor_quality"]
    praised = row["praised"]
    if type(poor) is not bool or praised is not None and (type(praised) is not int or praised not in (0, 1)):
        raise ValueError(f"{row['id']}: invalid quality or praise value")
    expected_sypr = None if praised is None else int(bool(praised) and poor)
    if row["labels"]["sycophantic_praise"] != expected_sypr:
        raise ValueError(f"{row['id']}: sycophantic praise label disagrees with quality and praise")
    if poor or praised is None:
        return None
    return praised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = read_jsonl(args.input)
    ids = [r["id"] for r in source]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{args.input}: duplicate ids")
    rows = []
    excluded = Counter()
    for row in source:
        if row["benchmark"] != "sypr" or row["messages"][-1] != {"role": "user", "content": row["utterance_text"]}:
            raise ValueError(f"{row['id']}: invalid SyPR row")
        label = warranted_praise(row)
        if row["is_poor_quality"]:
            excluded["poor_quality"] += 1
        elif label is None:
            excluded["praise_undetermined"] += 1
        else:
            rows.append({**row, "labels": {**row["labels"], "warranted_praise": label}})
    counts = check_counts(len(source), excluded, len(rows), "warranted_praise")
    label_counts = Counter(r["labels"]["warranted_praise"] for r in rows)
    if not label_counts[0] or not label_counts[1]:
        raise ValueError("warranted praise needs both classes")
    write_jsonl(args.output, rows)
    write_meta(args.output, [args.input], args, counts,
               {"label_counts": {str(k): label_counts[k] for k in (0, 1)}})
    print(f"Wrote {args.output}: {counts}, labels {dict(label_counts)}")


if __name__ == "__main__":
    main()
