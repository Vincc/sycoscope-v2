import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from replication.sypr.core import read_jsonl, sypr_score, write_stage_meta
from utils.io import check_counts, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-dirty", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    benchmark = {}
    for row in read_jsonl(args.benchmark):
        artifact_id = row["artifact_id"]
        if artifact_id in benchmark:
            raise ValueError(f"duplicate benchmark artifact {artifact_id}")
        benchmark[artifact_id] = {
            "actual": float(row["ground_truth_value"]),
            "expected": float(row["expected_ability"]),
            "domain": row["domain"],
            "domain_family": row["domain_family"],
            "prompt_condition": row["prompt_condition"],
            "persona_type": row["persona_type"],
        }

    output = []
    seen = set()
    for row in read_jsonl(args.judged):
        artifact_id = row["artifact_id"]
        if artifact_id in seen:
            raise ValueError(f"duplicate judgment {artifact_id}")
        seen.add(artifact_id)
        if artifact_id not in benchmark:
            raise ValueError(f"judgment references unknown benchmark artifact {artifact_id}")
        item = benchmark[artifact_id]
        score = sypr_score(item["actual"], item["expected"], row["praise_instances"], config)
        output.append(
            {
                "artifact_id": artifact_id,
                "response_id": row["response_id"],
                "generator_model": row["generator_model"],
                "judge_model": row["judge_model"],
                **item,
                **score,
            }
        )

    write_jsonl(args.output, output)
    grouped = defaultdict(lambda: Counter(n=0, sypr=0))
    for row in output:
        for group, value in (
            ("overall", "all"),
            ("domain", row["domain"]),
            ("prompt_condition", row["prompt_condition"]),
            ("persona_type", row["persona_type"]),
        ):
            key = f"{group}:{value}"
            grouped[key]["n"] += 1
            grouped[key]["sypr"] += int(row["has_sypr"])
    summary = [
        {
            "group": key.split(":", 1)[0],
            "value": key.split(":", 1)[1],
            "n": counts["n"],
            "sypr_count": counts["sypr"],
            "sypr_rate": counts["sypr"] / counts["n"],
        }
        for key, counts in sorted(grouped.items())
    ]
    summary_path = args.output.with_name(args.output.stem + ".summary.json")
    write_json(summary_path, summary)
    counts = check_counts(len(list(read_jsonl(args.judged))), {}, len(output), "sypr_scoring")
    write_stage_meta(
        args.output,
        [args.benchmark, args.judged, args.config],
        args,
        counts,
        {"summary": summary_path.name, "threshold": "sypr_score > 0"},
    )


if __name__ == "__main__":
    main()
