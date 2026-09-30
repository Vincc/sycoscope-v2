"""Validate OpenRouter generation checkpoints and write import metadata beside each data file."""
import argparse
import json
from collections import Counter
from pathlib import Path

from utils.io import check_counts, meta_path, read_json, sha256, write_meta


def write_checkpoint_meta(path: Path, inputs: list[Path], args, counts: dict, extra: dict) -> None:
    write_meta(path, inputs, args, counts, extra)
    written = meta_path(path)
    wanted = path.parent / "meta" / f"{path.name}.meta.json"
    if written != wanted:
        wanted.parent.mkdir(exist_ok=True)
        written.replace(wanted)


def scan(path: Path, model_slug: str, dataset: str) -> tuple[int, Counter]:
    statuses, ids, n = Counter(), set(), 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict) or row["model"] != model_slug:
                raise ValueError(f"{path}:{line_no}: invalid row or model")
            messages = row["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError(f"{path}:{line_no}: final message is not assistant")
            if dataset.startswith("are_you_sure/"):
                if [m["role"] for m in messages] != ["user", "assistant", "user", "assistant"] or \
                        messages[1]["content"] != row["turn1_response"] or \
                        messages[3]["content"] != row["turn2_response"]:
                    raise ValueError(f"{path}:{line_no}: Are You Sure messages differ from responses")
                if len(row["turn_generation_statuses"]) != 2:
                    raise ValueError(f"{path}:{line_no}: expected two generation statuses")
                statuses.update(row["turn_generation_statuses"])
            else:
                if messages[-1]["content"] != row["response"]:
                    raise ValueError(f"{path}:{line_no}: assistant message differs from response")
                statuses[row["generation_status"]] += 1
            if "id" in row:
                if row["id"] in ids:
                    raise ValueError(f"{path}:{line_no}: duplicate id {row['id']}")
                ids.add(row["id"])
            n += 1
    if not n:
        raise ValueError(f"{path}: empty checkpoint")
    return n, statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--exclude", nargs="*", default=[], help="Dataset paths relative to --root to leave unvalidated.")
    args = parser.parse_args()

    excluded = set(args.exclude)
    found = {p.parent.relative_to(args.root).as_posix(): p.parent for p in args.root.rglob("checkpoint.metadata.json")}
    unknown = excluded - set(found)
    if unknown:
        raise ValueError(f"excluded datasets not found: {sorted(unknown)}")
    for dataset, folder in sorted(found.items()):
        if dataset in excluded:
            print(f"{dataset}: excluded")
            continue
        meta_path = folder / "checkpoint.metadata.json"
        source_meta = read_json(meta_path)
        if source_meta["model"] != args.model_slug:
            raise ValueError(f"{meta_path}: model slug mismatch")
        files = sorted(folder.glob("*.jsonl"))
        if not files or folder / "checkpoint.jsonl" not in files:
            raise ValueError(f"{folder}: missing checkpoint.jsonl")
        results = {p.name: scan(p, args.model_slug, dataset) for p in files}
        main_statuses = results["checkpoint.jsonl"][1]
        if main_statuses != Counter(source_meta["generation_status_counts"]):
            raise ValueError(f"{folder}: checkpoint statuses {main_statuses} disagree with metadata")
        if dataset == "truthfulqa":
            if results["checkpoint.all.jsonl"][1] != Counter(source_meta["source_generation_status_counts"]):
                raise ValueError(f"{folder}: source generation statuses disagree with metadata")
            summary_path = folder / "summary.json"
            summary = read_json(summary_path)
            if summary["n_judged"] != results["checkpoint.jsonl"][0]:
                raise ValueError(f"{summary_path}: n_judged disagrees with checkpoint")
        elif "checkpoint.truncated.jsonl" in results and \
                results["checkpoint.truncated.jsonl"][1] != Counter(source_meta["excluded_generation_status_counts"]):
            raise ValueError(f"{folder}: excluded statuses disagree with metadata")

        for path in files:
            n, statuses = results[path.name]
            counts = check_counts(n, {}, n, path.as_posix())
            write_checkpoint_meta(path, [meta_path], args, counts,
                                  {"model": args.model_slug, "artifact_sha256": sha256(path),
                                   "source": source_meta["source"], "generation_status_counts": dict(statuses),
                                   "format": "raw_openrouter_checkpoint"})
            print(f"{dataset}/{path.name}: {n} rows verified")
        if dataset == "truthfulqa":
            write_checkpoint_meta(summary_path, [folder / "checkpoint.jsonl", meta_path], args,
                                  check_counts(1, {}, 1, summary_path.as_posix()),
                                  {"model": args.model_slug, "artifact_sha256": sha256(summary_path)})


if __name__ == "__main__":
    main()
