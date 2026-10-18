import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from replication.sypr.core import (
    messages_for_artifact,
    read_jsonl,
    response_id,
    trim_generated_token_ids,
    write_stage_meta,
)
from utils.io import check_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-dirty", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    all_rows = list(read_jsonl(args.benchmark))
    if args.max_rows is None and len(all_rows) != 13200:
        raise ValueError(f"expected 13,200 benchmark rows, found {len(all_rows)}")
    selected = [
        row for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index
    ]
    if args.max_rows is not None:
        selected = selected[: args.max_rows]
    if len({row["artifact_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("benchmark artifact IDs are not unique")

    completed = {}
    if args.output.exists():
        completed = {row["artifact_id"]: row for row in read_jsonl(args.output)}
    unknown = set(completed) - {row["artifact_id"] for row in selected}
    if unknown:
        raise ValueError(f"output contains {len(unknown)} artifact IDs outside this shard")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = [row for row in selected if row["artifact_id"] not in completed]
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            messages = [messages_for_artifact(row) for row in batch]
            prompts = [
                tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
                for item in messages
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_width = inputs["input_ids"].shape[1]
            for row, output in zip(batch, outputs, strict=True):
                generated = output[prompt_width:]
                token_ids, truncated = trim_generated_token_ids(
                    generated.tolist(), tokenizer.eos_token_id, args.max_new_tokens
                )
                token_count = len(token_ids)
                response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                result = {
                    "artifact_id": row["artifact_id"],
                    "response_id": response_id(row["artifact_id"], args.model),
                    "model": args.model,
                    "response": response,
                    "response_tokens": token_count,
                    "max_new_tokens": args.max_new_tokens,
                    "truncated": truncated,
                    "generation": {
                        "temperature": 0.0,
                        "do_sample": False,
                        "chat_template": tokenizer.chat_template,
                    },
                }
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()

    output_rows = list(read_jsonl(args.output))
    if len(output_rows) != len(selected):
        raise AssertionError(f"generated {len(output_rows)} of {len(selected)} selected rows")
    counts = check_counts(len(selected), {}, len(output_rows), "sypr_generation")
    write_stage_meta(
        args.output,
        [args.benchmark],
        args,
        counts,
        {
            "model": args.model,
            "method_deviation": None,
            "truncated": sum(row["truncated"] for row in output_rows),
        },
    )


if __name__ == "__main__":
    main()
