"""Convert the Meta Llama 3.1 TruthfulQA checkpoint to judged activation rows."""
import argparse
from collections import Counter
from pathlib import Path

from utils.io import check_counts, read_jsonl, write_jsonl, write_meta

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SOURCE_MODEL = "meta-llama/llama-3.1-8b-instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = read_jsonl(args.input)
    rows, excluded = [], Counter()
    for i, row in enumerate(source):
        rid = f"tqa__{row['domain']}__{row['template']}__{i:05d}"
        if row["model"] != SOURCE_MODEL or row["system_prompt"] is not None or row["reasoning"]:
            raise ValueError(f"{rid}: model, system prompt, or reasoning differs from the expected run")
        if row["label"] not in (0, 1) or type(row["label"]) is not int:
            raise ValueError(f"{rid}: invalid binary label")
        if row["label"] != int(row["verdict"] == "FALSE") or row["verdict"] not in ("TRUE", "FALSE"):
            raise ValueError(f"{rid}: label disagrees with verdict")
        if row["finish_reason"] != "stop":
            excluded[f"finish_{row['finish_reason']}"] += 1
            continue
        if not row["response"].strip():
            excluded["empty_response"] += 1
            continue
        messages = row["messages"]
        if [m["role"] for m in messages] != ["user", "assistant"] or messages[1]["content"] != row["response"]:
            raise ValueError(f"{rid}: messages disagree with response")
        if row["template"] == "plain" and messages[0]["content"] != row["question"]:
            raise ValueError(f"{rid}: plain prompt disagrees with question")
        rows.append({
            "id": rid, "benchmark": "truthfulqa", "source": row["domain"],
            "template": row["template"], "question": row["question"],
            "messages": messages[:1], "response": row["response"],
            "verdict": row["verdict"], "truncated": False,
            "model": MODEL, "generation_backend": "openrouter",
            "labels": {"false_answer": row["label"]},
        })
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate TruthfulQA IDs")
    counts = check_counts(len(source), excluded, len(rows), args.input.name)
    write_jsonl(args.output, rows)
    write_meta(args.output, [args.input], args, counts,
               {"model": MODEL, "label_counts": dict(Counter(r["labels"]["false_answer"] for r in rows))})
    print(f"Wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
