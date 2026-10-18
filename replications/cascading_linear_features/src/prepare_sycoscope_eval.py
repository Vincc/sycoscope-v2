"""Build the aligned SycoScope benchmark input used for CLiF evaluation."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from utils.io import check_counts, read_json, write_jsonl, write_meta


def read_worktree_or_parent_of_deletion(path: Path) -> tuple[bytes, str]:
    if path.exists():
        return path.read_bytes(), "worktree"

    deletion = subprocess.run(
        ["git", "log", "--diff-filter=D", "-1", "--format=%H", "--", path.as_posix()],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    if not deletion:
        raise FileNotFoundError(path)
    pointer = subprocess.run(
        ["git", "show", f"{deletion}^:{path.as_posix()}"], capture_output=True, check=True
    ).stdout
    data = subprocess.run(["git", "lfs", "smudge"], input=pointer, capture_output=True, check=True).stdout
    return data, f"{deletion}^"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--activation-meta-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    targets = list(csv.DictReader(args.targets.open(encoding="utf-8")))
    assert targets
    output = []
    source_records = []
    per_dataset = {}
    for target in targets:
        dataset = target["eval_dataset"]
        label_name = target["target_label"]
        meta_path = args.activation_meta_dir / f"{dataset}_activations.npz.meta.json"
        meta = read_json(meta_path)
        assert len(meta["inputs"]) == 1, (dataset, meta["inputs"])
        source_path = Path(next(iter(meta["inputs"])))
        data, source_ref = read_worktree_or_parent_of_deletion(source_path)
        assert hashlib.sha256(data).hexdigest() == meta["inputs"][source_path.as_posix()]
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        assert len(rows) == int(target["n"]) == meta["counts"]["n_out"]
        assert {row["model"] for row in rows} == {"meta-llama/Llama-3.1-8B-Instruct"}
        labels = [row["labels"][label_name] for row in rows]
        assert set(labels) == {0, 1}, (dataset, set(labels))
        assert len({row["id"] for row in rows}) == len(rows), dataset
        for row, label in zip(rows, labels, strict=True):
            output.append(
                {
                    "dataset": dataset,
                    "family": target["family"],
                    "display": target["display"],
                    "target_label": label_name,
                    "id": row["id"],
                    "messages": row["messages"],
                    "response": row["response"],
                    "label": label,
                }
            )
        per_dataset[dataset] = {"n": len(rows), "negative": labels.count(0), "positive": labels.count(1)}
        source_records.append(
            {
                "path": source_path.as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "source_ref": source_ref,
            }
        )

    write_jsonl(args.out, output)
    counts = check_counts(len(output), {}, len(output), "prepare SycoScope CLiF evaluation")
    write_meta(
        args.out,
        [args.targets, *[args.activation_meta_dir / f"{x['eval_dataset']}_activations.npz.meta.json" for x in targets]],
        args,
        counts,
        {"datasets": per_dataset, "sources": source_records},
    )
    print(f"wrote {len(output)} rows across {len(targets)} datasets to {args.out}")


if __name__ == "__main__":
    main()
