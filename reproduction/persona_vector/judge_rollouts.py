"""Judged benchmark rows as judge.py rollouts: one row per labelled response with id, question and answer.

The answer is the response. The question is the conversation before it (judge_question): the user message itself when
there is only one (ELEPHANT), otherwise every turn as "User: ..." or "Assistant: ..." separated by blank lines (Are You
Sure: question, turn-1 reply, pushback). A row is kept when it has a 0/1 label in at least one unit of
interp.generalization_matrix.unit_rows under --judge-name; the others are counted by reason, as unit_rows counts them.
For each --judged file writes <out-dir>/<stem>.jsonl, stem = the file name without _judged.jsonl, and its meta file.

Run from the repo root:
  python -m reproduction.persona_vector.judge_rollouts --judged ... --judge-name sonnet55 --out-dir ...
"""
import argparse
from collections import Counter
from pathlib import Path

from interp.generalization_matrix import unit_rows
from judging.judge_responses import syco_none_reason
from utils.io import check_counts, read_jsonl, write_jsonl, write_meta

SPEAKER = {"user": "User", "assistant": "Assistant"}


def judge_question(messages: list[dict]) -> str:
    """The judge's {question}: a lone user message as is, otherwise a User/Assistant transcript ending on a user turn."""
    if any(m["role"] not in SPEAKER for m in messages) or messages[-1]["role"] != "user":
        raise ValueError(f"expected user and assistant turns ending on a user turn, got {[m['role'] for m in messages]}")
    if len(messages) == 1:
        return messages[0]["content"]
    return "\n\n".join(f"{SPEAKER[m['role']]}: {m['content']}" for m in messages)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", type=Path, nargs="+", required=True)
    parser.add_argument("--judge-name", required=True, help="Label suffix, e.g. sonnet55.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in args.judged:
        rows = read_jsonl(path)
        if len({r["id"] for r in rows}) != len(rows):
            raise AssertionError(f"{path}: duplicate ids")
        units, _ = unit_rows(rows, args.judge_name)
        labelled = {i for members in units.values() for i, _ in members}
        out, excluded = [], Counter()
        for r in rows:
            if r["id"] in labelled:
                out.append({"id": r["id"], "question": judge_question(r["messages"]), "answer": r["response"]})
            elif r["benchmark"] == "are_you_sure":
                labels = {k.removesuffix(f"__{args.judge_name}"): v for k, v in r["labels"].items()
                          if k.endswith(f"__{args.judge_name}")}
                excluded[syco_none_reason({**r, "labels": labels})] += 1
            else:
                excluded["truncated" if r["truncated"] else "label_none"] += 1
        target = args.out_dir / f"{path.name.removesuffix('_judged.jsonl')}.jsonl"
        if target.exists():
            raise FileExistsError(target)
        write_jsonl(target, out)
        write_meta(target, [path], args, check_counts(len(rows), dict(excluded), len(out), str(path)))
        print(f"{path}: {len(rows)} in, excluded {dict(excluded)}, {len(out)} out -> {target}")


if __name__ == "__main__":
    main()
