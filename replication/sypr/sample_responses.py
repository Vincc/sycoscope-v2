import argparse
import random
from pathlib import Path

from replication.sypr.core import read_jsonl, write_stage_meta
from utils.io import check_counts, write_jsonl


def sample_complete(rows, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    invalid = [row.get("artifact_id") for row in rows if type(row.get("truncated")) is not bool]
    if invalid:
        raise ValueError(f"{len(invalid)} responses have a non-boolean truncated field")
    eligible = [row for row in rows if row["truncated"] is False]
    if not eligible:
        raise ValueError("no untruncated responses are eligible")
    sample_size = max(1, round(len(eligible) * fraction))
    selected_ids = {
        row["artifact_id"] for row in random.Random(seed).sample(eligible, sample_size)
    }
    return [row for row in eligible if row["artifact_id"] in selected_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    if len({row["artifact_id"] for row in rows}) != len(rows):
        raise ValueError("generation artifact IDs are not unique")
    sampled = sample_complete(rows, args.fraction, args.seed)
    truncated = sum(row["truncated"] is True for row in rows)
    eligible = len(rows) - truncated
    not_selected = eligible - len(sampled)
    counts = check_counts(
        len(rows),
        {"truncated": truncated, "not_selected": not_selected},
        len(sampled),
        "sypr_sampling",
    )
    write_jsonl(args.output, sampled)
    write_stage_meta(
        args.output,
        [args.input],
        args,
        counts,
        {"eligible": eligible, "sampling": "uniform_without_replacement"},
    )


if __name__ == "__main__":
    main()
