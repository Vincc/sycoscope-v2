"""Sample the prompts used for steering: social-sycophancy test prompts (OEQ, AITA, IR) drawn from the union
of every labeller's test files, so no probe's own training prompts are steered on. Writes {id, dataset, user_text}."""
import argparse
import collections
import json
import os
import random

LABELLERS = ("gemini", "gpt4o", "llama70b", "llama8b")
GROUPS = ("4dims", "supporttypes")
SOCIAL = ("oeq", "aita", "ir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train, test = set(), {}
    for lab in LABELLERS:
        for g in GROUPS:
            for r in map(json.loads, open(os.path.join(args.prepared, f"{lab}_{g}_train.jsonl"), encoding="utf-8")):
                train.add(r["id"])
            for r in map(json.loads, open(os.path.join(args.prepared, f"{lab}_{g}_test.jsonl"), encoding="utf-8")):
                test[r["id"]] = r
    pool = sorted((r for r in test.values() if r["dataset"] in SOCIAL), key=lambda r: r["id"])
    sample = random.Random(args.seed).sample(pool, args.n)
    rows = [{"id": r["id"], "dataset": r["dataset"], "user_text": r["user_text"], "in_some_train_file": r["id"] in train} for r in sample]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"pool {len(pool)} social test prompts; sampled {len(rows)}: {dict(collections.Counter(r['dataset'] for r in rows))}; "
          f"{sum(r['in_some_train_file'] for r in rows)} appear in some labeller's train file")


if __name__ == "__main__":
    main()
