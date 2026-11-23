"""Select benchmark items for a steering run from a judged file: one --source, --n rows drawn with --seed.

Are You Sure rows must have a turn-1 answer judged correct by --judge-name, not be a later copy of a repeated prompt,
and not have a cut-off turn 1. Their `messages` (question, turn-1 answer, pushback) are kept and the turn-1 generation
settings move to `turn1_generation`. Response and judge fields are dropped. Drawn rows are kept in file order.

Run from the repo root:
  python scripts/select_steer_items.py --judged ... --judge-name sonnet55 --source OEQ --n 300 --seed 0 --output ...
"""
import argparse
import random
from collections import Counter
from pathlib import Path

from utils.io import check_counts, read_jsonl, write_jsonl, write_meta

DROP = ("response", "truncated", "model", "generation", "request_seeds", "duplicate_of", "labels", "judge_model",
        "judge_raw", "verdict")


def exclusion(row: dict, source: str, judge: str) -> str | None:
    """Why a judged row is not a candidate item, or None if it is."""
    if row["source"] != source:
        return "other_source"
    if row["benchmark"] == "are_you_sure":
        if row["duplicate_of"] is not None:
            return "duplicate_question"
        if row["turn1_truncated"]:
            return "turn1_truncated"
        t1 = row["labels"][f"turn1_correct__{judge}"]
        if t1 is None:
            return "turn1_unparsed"
        if t1 != 1:
            return "turn1_incorrect"
    return None


def to_item(row: dict) -> dict:
    """Row without its response and judgments; an Are You Sure row keeps the settings that generated its turn 1."""
    item = {k: v for k, v in row.items() if k not in DROP}
    if row["benchmark"] == "are_you_sure":
        if row["messages"][-1]["role"] != "user" or row["messages"][-2]["content"] != row["turn1_response"]:
            raise ValueError(f"{row['id']}: messages do not end with the turn-1 answer and the pushback")
        item["turn1_generation"] = {**row["generation"], "request_seed": row["request_seeds"][0]}
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--judge-name", required=True, help="Label suffix whose turn-1 labels select Are You Sure rows.")
    parser.add_argument("--source", required=True, help="Row `source`, e.g. OEQ or trivia_qa.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = read_jsonl(args.judged)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{args.judged}: duplicate ids")
    excluded, eligible = Counter(), []
    for r in rows:
        reason = exclusion(r, args.source, args.judge_name)
        if reason is None:
            eligible.append(r)
        else:
            excluded[reason] += 1
    if args.n > len(eligible):
        raise ValueError(f"--n {args.n} exceeds the {len(eligible)} eligible rows")
    chosen = sorted(random.Random(args.seed).sample(range(len(eligible)), args.n))
    items = [to_item(eligible[i]) for i in chosen]
    excluded["not_sampled"] = len(eligible) - len(items)
    counts = check_counts(len(rows), dict(excluded), len(items), args.judged.name)
    write_jsonl(args.output, items)
    write_meta(args.output, [args.judged], args, {**counts, "n_eligible": len(eligible)})
    print(f"Wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
