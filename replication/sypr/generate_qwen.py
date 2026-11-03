import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from replication.sypr.core import (
    check_resume_row,
    generate_kwargs,
    messages_for_artifact,
    raw_prompt_for_artifact,
    read_jsonl,
    response_id,
    response_model_tag,
    resume_offset,
    sampling_settings,
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
    # chat: model's chat template. raw: authors' plain System:/User:/Assistant: lines.
    parser.add_argument("--prompt-format", choices=["chat", "raw"], default="chat")
    # Greedy when temperature is 0. Otherwise nucleus sampling; top-k 0 disables the top-k filter.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-dirty", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    sampling = sampling_settings(args.temperature, args.top_p, args.top_k, args.seed)

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

    model_tag = response_model_tag(args.model, args.prompt_format, sampling)
    completed = {}
    if args.output.exists():
        completed = {row["artifact_id"]: row for row in read_jsonl(args.output)}
    resume_start = resume_offset(selected, list(completed), args.batch_size, sampling["do_sample"])
    # Resuming must not mix rows from a different model, prompt format or token cap.
    for row in completed.values():
        check_resume_row(
            row, args.model, args.max_new_tokens, args.prompt_format, sampling, args.batch_size
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_token_ids = model.generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    if tokenizer.eos_token_id not in eos_token_ids:
        raise ValueError("tokenizer EOS is not among the model's generation EOS ids")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for start in range(resume_start, len(selected), args.batch_size):
            batch = selected[start : start + args.batch_size]
            if args.prompt_format == "chat":
                prompts = [
                    tokenizer.apply_chat_template(
                        messages_for_artifact(row), tokenize=False, add_generation_prompt=True
                    )
                    for row in batch
                ]
            else:
                prompts = [raw_prompt_for_artifact(row) for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            if sampling["do_sample"]:
                # Seeded by shard position so a resumed run draws what an uninterrupted run would.
                torch.manual_seed(sampling["seed"] * 1_000_003 + start)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    **generate_kwargs(sampling),
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                )
            prompt_width = inputs["input_ids"].shape[1]
            for row, prompt, output in zip(batch, prompts, outputs, strict=True):
                generated = output[prompt_width:]
                token_ids, truncated = trim_generated_token_ids(
                    generated.tolist(), eos_token_ids, args.max_new_tokens
                )
                token_count = len(token_ids)
                response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                result = {
                    "artifact_id": row["artifact_id"],
                    "response_id": response_id(row["artifact_id"], model_tag),
                    "model": args.model,
                    "response": response,
                    "response_tokens": token_count,
                    "max_new_tokens": args.max_new_tokens,
                    "truncated": truncated,
                    "generation": {
                        **sampling,
                        "prompt_format": args.prompt_format,
                        "batch_size": args.batch_size,
                        "chat_template": (
                            tokenizer.chat_template if args.prompt_format == "chat" else None
                        ),
                        "prompt": prompt if args.prompt_format == "raw" else None,
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
            "prompt_format": args.prompt_format,
            "sampling": sampling,
            "method_deviation": (
                "chat template instead of authors' plain System:/User: serialization"
                if args.prompt_format == "chat"
                else None
            ),
            "truncated": sum(row["truncated"] for row in output_rows),
        },
    )


if __name__ == "__main__":
    main()
