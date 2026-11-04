"""Stage A: OSF structured-assumption CSVs -> clean JSONL for probe training.

One output file per input CSV (the 4dims and supporttypes files of a labeller use different train/test
splits, so they are not merged). Writes
  data/prepared/<labeller>_<group>_<split>.jsonl   {id, occurrence, user_text, dataset, scores: {dim: float}}
  data/prepared/prompts.jsonl              {id, user_text} union over every file, for extraction
  data/prepared/meta/*.meta.json           input hashes, counts, exclusions
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from utils.io import git_state, sha256, write_json, write_jsonl  # noqa: E402

csv.field_size_limit(1 << 30)

LABELLERS = ("gemini", "gpt4o", "llama70b", "llama8b")
SPLITS = ("train", "test")
DIMS = {
    "4dims": ("validation_seeking", "user_rightness", "user_information_advantage", "objectivity_seeking"),
    "supporttypes": ("emotional_support", "social_companionship", "belonging_support", "information_guidance",
                     "tangible_support"),
}
# substring of source_file -> dataset name; the paper's six datasets plus the expectation-gap set
DATASETS = {"factual": "factual", "OEQ": "oeq", "AITA": "aita", "resps": "ir", "cancer": "cancer",
            "wildchat": "wildchat", "valobj": "valobj"}


def dataset_of(source_file):
    hits = [name for key, name in DATASETS.items() if key in source_file]
    if len(hits) != 1:
        raise ValueError(f"source_file {source_file!r} matches {hits}")
    return hits[0]


def prompt_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_scores(path, dims, excluded):
    """{(user_text, occurrence): (dataset, {dim: score})} for one CSV. Rows with a bad score are excluded."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out, seen = {}, Counter()
    for r in rows:
        text = r["user_text"]
        if not text.strip():
            excluded["blank user_text"] += 1
            continue
        scores = {}
        for d in dims:
            raw = r[f"{d}_score"].strip()
            try:
                v = float(raw)
            except ValueError:
                excluded[f"{d}: missing score"] += 1
                scores = None
                break
            if not 0.0 <= v <= 1.0:
                excluded[f"{d}: score out of [0,1]"] += 1
                scores = None
                break
            scores[d] = v
        if scores is None:
            continue
        key = (text, seen[text])
        seen[text] += 1
        out[key] = (dataset_of(r["source_file"]), scores)
    return len(rows), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osf_dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "osf"))
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "prepared"))
    args = ap.parse_args()

    prompts = {}
    for lab in LABELLERS:
        for group, dims in DIMS.items():
            split_ids = {}
            for split in SPLITS:
                excluded = Counter()
                path = os.path.join(args.osf_dir, f"{split}_{lab}_{group}.csv")
                n_in, scored = read_scores(path, dims, excluded)
                out = []
                for (text, occ), (ds, scores) in sorted(scored.items()):
                    if split == "test" and prompt_id(text) in split_ids["train"]:
                        excluded["test prompt also in train"] += 1  # duplicated short prompts such as "Are you sure?"
                        continue
                    out.append({"id": prompt_id(text), "occurrence": occ, "user_text": text, "dataset": ds, "scores": scores})
                    prompts[prompt_id(text)] = text
                split_ids[split] = {r["id"] for r in out}
                counts = {"rows_in": n_in, "rows_out": len(out), "excluded": dict(excluded),
                          "by_dataset": dict(Counter(r["dataset"] for r in out))}
                assert n_in == len(out) + sum(excluded.values())
                name = f"{lab}_{group}_{split}.jsonl"
                write_jsonl(os.path.join(args.out_dir, name), out)
                write_json(os.path.join(args.out_dir, "meta", name + ".meta.json"),
                           {"inputs": {os.path.basename(path): sha256(path)}, "git": git_state(), "argv": sys.argv, "counts": counts})
                print(f"{lab:9s} {group:12s} {split:5s} in {n_in:5d} out {len(out):5d} excluded {sum(excluded.values()):3d}  {counts['by_dataset']}")
            assert not split_ids["train"] & split_ids["test"]

    # one extraction set: every distinct prompt text across all files
    rows = [{"id": i, "user_text": t} for i, t in sorted(prompts.items())]
    assert len({r["id"] for r in rows}) == len(rows)
    write_jsonl(os.path.join(args.out_dir, "prompts.jsonl"), rows)
    write_json(os.path.join(args.out_dir, "meta", "prompts.jsonl.meta.json"),
               {"git": git_state(), "argv": sys.argv, "counts": {"unique_prompts": len(rows)}})
    print(f"prompts.jsonl: {len(rows)} unique prompts")


if __name__ == "__main__":
    main()
