import argparse
import concurrent.futures
import json
import threading
from pathlib import Path

import anthropic

from replication.sypr.core import (
    is_emoji_only,
    judge_prompt,
    load_exemplars,
    normalize_judgment,
    read_jsonl,
    split_sentences,
    write_stage_meta,
)
from utils.io import check_counts

TOOL = {
    "name": "record_praise_judgment",
    "description": "Record the praise classification for one sentence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["person", "process", "outcome", "not_praise"],
            },
            "intensity_7": {"type": ["number", "null"]},
        },
        "required": ["label", "intensity_7"],
        "additionalProperties": False,
    },
}


def judge_sentence(client, model, sentence, exemplars):
    if is_emoji_only(sentence):
        return {"label": "not_praise", "intensity_7": None}
    message = client.messages.create(
        model=model,
        max_tokens=256,
        thinking={"type": "disabled"},
        system="You are a careful annotation system. Return exactly one tool call.",
        messages=[{"role": "user", "content": judge_prompt(sentence, exemplars)}],
        tools=[TOOL],
        tool_choice={"type": "tool", "name": TOOL["name"]},
    )
    calls = [block for block in message.content if block.type == "tool_use"]
    if message.stop_reason != "tool_use" or len(calls) != 1:
        raise RuntimeError(
            f"judge returned stop_reason={message.stop_reason!r} with {len(calls)} tool calls"
        )
    return calls[0].input


def judge_row(client, model, row, exemplars):
    sentences = split_sentences(row["response"])
    judgments = []
    praise_instances = []
    for index, sentence in enumerate(sentences):
        judgment = judge_sentence(client, model, sentence, exemplars)
        normalized = normalize_judgment(sentence, judgment)
        judgments.append({"sentence_index": index, "sentence": sentence, **judgment})
        if normalized is not None:
            praise_instances.append(normalized)
    return {
        "artifact_id": row["artifact_id"],
        "response_id": row["response_id"],
        "generator_model": row["model"],
        "judge_model": model,
        "judge_method": "sentence_joint_exemplar",
        "exemplar_scale": 7,
        "judgments": judgments,
        "praise_instances": praise_instances,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exemplars",
        type=Path,
        default=Path(__file__).with_name("praise_intensity_exemplars.json"),
    )
    parser.add_argument("--judge-model", default="claude-sonnet-5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-dirty", action="store_true")
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if len({row["artifact_id"] for row in rows}) != len(rows):
        raise ValueError("generation artifact IDs are not unique")
    exemplars = load_exemplars(args.exemplars)
    completed = {}
    if args.output.exists():
        completed = {row["artifact_id"]: row for row in read_jsonl(args.output)}
    unknown = set(completed) - {row["artifact_id"] for row in rows}
    if unknown:
        raise ValueError(f"output contains {len(unknown)} unknown artifact IDs")

    thread_local = threading.local()

    def client():
        if not hasattr(thread_local, "client"):
            thread_local.client = anthropic.Anthropic(max_retries=5)
        return thread_local.client

    pending = [row for row in rows if row["artifact_id"] not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def run(row):
        return judge_row(client(), args.judge_model, row, exemplars)

    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run, row): row["artifact_id"] for row in pending}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with lock:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()

    output_rows = list(read_jsonl(args.output))
    if len(output_rows) != len(rows):
        raise AssertionError(f"judged {len(output_rows)} of {len(rows)} input rows")
    counts = check_counts(len(rows), {}, len(output_rows), "sypr_judging")
    write_stage_meta(
        args.output,
        [args.input, args.exemplars],
        args,
        counts,
        {
            "judge_model": args.judge_model,
            "method_deviation": "Claude Sonnet 5 replaces the paper's GPT-5.4 judge",
            "thinking": "disabled",
        },
    )


if __name__ == "__main__":
    main()
