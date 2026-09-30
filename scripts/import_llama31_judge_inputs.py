"""Convert Meta Llama 3.1 moral, social, and SyPR checkpoints to judge inputs."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from utils.io import check_counts, read_jsonl, write_jsonl, write_meta

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SOURCE_MODEL = "meta-llama/llama-3.1-8b-instruct"
SOURCES = {
    "aita_nta_flip": "AITA-NTA-FLIP", "aita_nta_og": "AITA-NTA-OG",
    "aita_yta": "AITA-YTA", "oeq": "OEQ", "ss": "SS", "sypr": "sypr",
}


def sypr_poor(row: dict) -> bool:
    if row["domain_family"] == "reasoning":
        if type(row["ground_truth_correctness"]) is not bool:
            raise ValueError(f"{row['id']}: missing reasoning correctness")
        return not row["ground_truth_correctness"]
    if row["domain_family"] == "social" and row["utterance_quality"] in ("low", "high"):
        return row["utterance_quality"] == "low"
    raise ValueError(f"{row['id']}: invalid SyPR quality")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = SOURCES[args.dataset]
    raw = read_jsonl(args.input)
    excluded = Counter()
    candidates = []
    for row in raw:
        rid = str(row["row_id"])
        if row["model"] != SOURCE_MODEL or row["system_prompt"] is not None or row["reasoning"]:
            raise ValueError(f"{rid}: model, system prompt, or reasoning differs from the expected run")
        if row["finish_reason"] != "stop":
            excluded[f"finish_{row['finish_reason']}"] += 1
            continue
        if not row["response"].strip():
            excluded["empty_response"] += 1
            continue
        messages = row["messages"]
        if messages[-1] != {"role": "assistant", "content": row["response"]}:
            raise ValueError(f"{rid}: final message disagrees with response")
        if args.dataset == "sypr":
            if row["id"] != row["row_id"] or messages[-2] != {"role": "user", "content": row["utterance_text"]}:
                raise ValueError(f"{rid}: SyPR IDs or utterance disagree")
            candidates.append({
                "id": f"sypr__{rid}", "benchmark": "sypr", "source": row["domain"],
                "domain_family": row["domain_family"], "row_id": row["row_id"],
                "utterance_text": row["utterance_text"], "is_poor_quality": sypr_poor(row),
                "messages": messages[:-1], "response": row["response"],
                "truncated": False, "model": MODEL, "generation_backend": "openrouter",
            })
        else:
            if row["dataset"] != source or [m["role"] for m in messages] != ["user", "assistant"] or \
                    messages[0]["content"] != row["prompt"]:
                raise ValueError(f"{rid}: dataset or messages disagree with checkpoint")
            candidates.append({
                "id": f"elephant__{source}__{row['prompt_col']}__{rid}",
                "benchmark": "elephant", "source": source, "row_id": row["row_id"],
                "prompt_col": row["prompt_col"], "messages": messages[:1],
                "response": row["response"], "truncated": False,
                "model": MODEL, "generation_backend": "openrouter",
            })

    if args.dataset == "aita_nta_flip":
        by_pair = defaultdict(dict)
        for row in candidates:
            sides = by_pair[str(row["row_id"])]
            side = row["prompt_col"]
            if side in sides or side not in ("original_post", "flipped_story"):
                raise ValueError(f"{row['row_id']}: duplicate or invalid flip side {side}")
            sides[side] = row
        rows = [r for r in candidates if len(by_pair[str(r["row_id"])]) == 2]
        excluded["pair_partner_missing"] = len(candidates) - len(rows)
    else:
        rows = candidates
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{args.input}: duplicate IDs")
    counts = check_counts(len(raw), excluded, len(rows), args.input.name)
    write_jsonl(args.output, rows)
    write_meta(args.output, [args.input], args, counts, {"model": MODEL, "source": source})
    print(f"Wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
