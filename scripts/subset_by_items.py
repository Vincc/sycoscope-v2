"""Keep the rows of --input whose id is in --items, in the order of --items; raises if an item has no row.

Run from the repo root:
  python scripts/subset_by_items.py --input ... --items ... --output ...
"""
import argparse
from pathlib import Path

from utils.io import check_counts, read_jsonl, write_jsonl, write_meta


def subset(rows: list[dict], ids: list[str]) -> list[dict]:
    """Rows for `ids`, in that order."""
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} items have no row, e.g. {missing[0]}")
    return [by_id[i] for i in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_jsonl(args.input)
    out = subset(rows, [r["id"] for r in read_jsonl(args.items)])
    counts = check_counts(len(rows), {"not_in_items": len(rows) - len(out)}, len(out), args.input.name)
    write_jsonl(args.output, out)
    write_meta(args.output, [args.input, args.items], args, counts)
    print(f"Wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
