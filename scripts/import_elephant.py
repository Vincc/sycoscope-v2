"""Convert a judged ELEPHANT social-sycophancy file from the old repo into judging rows.

Output rows match judging/judge_responses.py: benchmark "elephant", messages, response, truncated, model and
labels {validation, indirectness, framing} (1 = sycophantic, None = unparseable judgment, never coerced).
The judged file does not record the generator, so each row is matched against the raw generation file
(prompt and response must be identical) and the generator model is taken from there.
"""
import argparse
from collections import Counter
from pathlib import Path

from utils.io import REPO_ROOT, check_counts, parse_jsonl_bytes, read_git_blob, write_jsonl, write_meta
from utils.models import load_tokenizer

LABELS = ("validation", "indirectness", "framing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-repo", type=Path, required=True)
    parser.add_argument("--judged-ref", required=True, help="Old-repo ref holding the judged file, e.g. origin/building-agent.")
    parser.add_argument("--judged-path", required=True)
    parser.add_argument("--raw-ref", required=True, help="Old-repo ref holding the raw generations, e.g. origin/SAE.")
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--dataset", required=True, help="Expected value of the rows' dataset field, e.g. OEQ.")
    parser.add_argument("--max-new-tokens", type=int, required=True, help="Generation cap; responses within 2 tokens of it are flagged truncated.")
    args = parser.parse_args()

    judged_bytes, judged_src = read_git_blob(args.old_repo, args.judged_ref, args.judged_path)
    raw_bytes, raw_src = read_git_blob(args.old_repo, args.raw_ref, args.raw_path)
    old = parse_jsonl_bytes(judged_bytes)
    raw = {(r["row_id"], r["prompt_col"]): r for r in parse_jsonl_bytes(raw_bytes) if r["sample_idx"] == 0}
    models = {r["model"] for r in raw.values()}
    if len(models) != 1:
        raise ValueError(f"raw generations come from several models: {sorted(models)}")
    model = models.pop()
    tokenizer = load_tokenizer(model, padding_side="right")

    rows = []
    for r in old:
        if r["dataset"] != args.dataset:
            raise ValueError(f"row {r['row_id']}: dataset {r['dataset']!r}")
        src = raw[(r["row_id"], r["prompt_col"])]
        if src["prompt"] != r["prompt"] or src["response"] != r["response"]:
            raise ValueError(f"row {r['row_id']}: prompt or response differs from the raw generation")
        labels = {k: r[k] for k in LABELS}
        bad = {k: v for k, v in labels.items() if v not in (0, 1, None) or isinstance(v, bool)}
        if bad:
            raise ValueError(f"row {r['row_id']}: labels must be 0, 1 or None, got {bad}")
        if not r["response"].strip():
            raise ValueError(f"row {r['row_id']}: empty response")
        n_tok = len(tokenizer(r["response"], add_special_tokens=False)["input_ids"])
        rows.append(
            {
                "id": f"elephant__{args.dataset}__{r['row_id']:05d}",
                "benchmark": "elephant",
                "source": args.dataset,
                "old_row_id": r["row_id"],
                "messages": [{"role": "user", "content": r["prompt"]}],
                "response": r["response"],
                "truncated": n_tok >= args.max_new_tokens - 2,
                "model": model,
                "labels": labels,
            }
        )

    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row_id")
    out = REPO_ROOT / "generations" / model.split("/")[-1] / "judging" / f"{args.dataset}_judged.jsonl"
    write_jsonl(out, rows)
    counts = check_counts(len(old), {}, len(rows), args.dataset)
    counts["truncated"] = sum(r["truncated"] for r in rows)
    counts["labels"] = {k: dict(Counter(str(r["labels"][k]) for r in rows)) for k in LABELS}
    # Judge model is not in the data; the old results/generations/README.md names it.
    extra = {"old_repo_sources": {"judged": judged_src, "raw": raw_src}, "generator_model": model,
             "judge_model": "claude-sonnet-5 per old results/generations/README.md; unverified, that README also misnames the generator"}
    write_meta(out, [], args, counts, extra)
    print(out, counts)


if __name__ == "__main__":
    main()
