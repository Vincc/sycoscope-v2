"""Generate responses to each user prompt under both system prompts of each pair.

Writes one JSONL per pair to --out-dir. Rows are appended per batch; re-running the same command resumes.
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

from utils.contrastive import POLARITIES, classify_degenerate, load_pairs, make_row, row_id
from utils.io import check_counts, read_jsonl, write_meta
from utils.models import (
    contrastive_messages,
    generate,
    load_model_and_tokenizer,
    model_spec,
    render_prompt,
    strip_think,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--slugs", nargs="+", required=True, help="Pair slugs to generate, or 'all'.")
    parser.add_argument("--prompts", type=Path, required=True, help="User-prompt JSONL with id, text, source.")
    parser.add_argument("--n-prompts", type=int, required=True, help="Use the first N prompts of the file.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True, help="0 means greedy.")
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-response-chars", type=int, required=True, help="Shorter responses are flagged too_short.")
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    import torch

    pairs = load_pairs(args.pairs)
    slugs = list(pairs) if args.slugs == ["all"] else args.slugs
    unknown = sorted(set(slugs) - set(pairs))
    if unknown:
        raise ValueError(f"unknown slugs: {unknown}")
    all_prompts = read_jsonl(args.prompts)
    if args.n_prompts > len(all_prompts):
        raise ValueError(f"--n-prompts {args.n_prompts} > {len(all_prompts)} prompts in file")
    prompts = all_prompts[: args.n_prompts]

    thinking = model_spec(args.model)["thinking"]
    model, tokenizer = load_model_and_tokenizer(args.model, padding_side="left")
    torch.manual_seed(args.seed)  # seeded once; a resumed run draws different samples than an uninterrupted one
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for slug in slugs:
        pair = pairs[slug]
        out_path = args.out_dir / f"{slug}.jsonl"
        done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
        for polarity in POLARITIES:
            todo = [p for p in prompts if row_id(slug, polarity, p["id"]) not in done]
            print(f"[{slug} / {polarity}] {len(todo)} to generate ({len(prompts) - len(todo)} on disk)")
            for b in range(0, len(todo), args.batch_size):
                chunk = todo[b : b + args.batch_size]
                rendered = [render_prompt(tokenizer, contrastive_messages(pair[polarity], p["text"])) for p in chunk]
                raw, truncated = generate(model, tokenizer, rendered, args.max_new_tokens, args.temperature, args.top_p)
                with open(out_path, "a", encoding="utf-8", newline="\n") as f:
                    for prompt, text, trunc in zip(chunk, raw, truncated):
                        response = strip_think(text) if thinking else text
                        if response is None:
                            response, flag = "", "unclosed_think"
                        else:
                            flag = classify_degenerate(response, trunc, args.min_response_chars)
                        n_tok = len(tokenizer(response, add_special_tokens=False)["input_ids"])
                        row = make_row(pair, polarity, prompt, response, n_tok, flag, args.model)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                print(f"  {min(b + args.batch_size, len(todo))}/{len(todo)}", flush=True)

        rows = read_jsonl(out_path)
        expected = {row_id(slug, pol, p["id"]) for pol in POLARITIES for p in prompts}
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids) or set(ids) != expected:
            raise AssertionError(f"{out_path}: rows do not match the expected {len(expected)} ids")
        counts = check_counts(len(expected), {}, len(rows), slug)
        counts["degenerate"] = dict(Counter(str(r["degenerate"]) for r in rows))
        write_meta(out_path, [args.pairs, args.prompts], args, counts)
        print(f"[{slug}] {counts}")


if __name__ == "__main__":
    main()
