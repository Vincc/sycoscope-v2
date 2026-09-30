"""Exclude rows that cannot be extracted at a specified token length before balancing labels."""
import argparse
from collections import Counter
from pathlib import Path

from utils.activations import prepare
from utils.io import check_counts, read_jsonl, write_jsonl, write_meta
from utils.models import load_tokenizer, model_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_jsonl(args.input)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{args.input}: duplicate IDs")
    if any(r["model"] != args.model for r in rows):
        raise ValueError(f"{args.input}: row model differs from {args.model}")
    tokenizer = load_tokenizer(args.model, padding_side="right")
    prepared, skips = prepare(rows, tokenizer, args.max_length, model_spec(args.model).get("template_kwargs"))
    keep = {r["id"] for r in prepared}
    selected = [r for r in rows if r["id"] in keep]
    if len(selected) != len(prepared):
        raise AssertionError("prepared row order or IDs disagree with input")
    counts = check_counts(len(rows), Counter(s["reason"] for s in skips), len(selected), args.input.name)
    write_jsonl(args.output, selected)
    write_meta(args.output, [args.input], args, counts, {"skips": skips})
    print(f"Wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
