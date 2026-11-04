"""Sycophancy-probe baseline, data: OSF {train,test}_probe_data_syco.csv -> JSONL (paper Section 3.1, Table A8).

Labels are binary ELEPHANT judgements of one response model's reply to the prompt, columns <dim>_<Model>
(validation, indirectness, framing x GPT-4o, Gemini, Llama-70B, Llama-8B). Text column is `prompt`
(upstream train_probe_bin.py picks the text column with fewest NaNs; `user_text` is mostly empty here).
Writes data/prepared_syco/<model>_<split>.jsonl {id, user_text, dataset, labels}, one row per prompt with all
three labels present, and data/prepared_syco/prompts_missing.jsonl = prompts not already in data/prepared/prompts.jsonl.
"""
import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import prompt_id  # noqa: E402
from utils.io import git_state, sha256, write_json, write_jsonl  # noqa: E402

csv.field_size_limit(1 << 30)
MODELS = {"gpt4o": "GPT-4o", "gemini": "Gemini", "llama70b": "Llama-70B", "llama8b": "Llama-8B"}
SYCO_DIMS = ("validation", "indirectness", "framing")
SOURCES = {"combined_factual_wide.csv": "factual", "../mental_models/OEQ_full_paper.csv": "oeq",
           "../mental_models/AITA-YTA_Qwen_1116_gpt4o.csv": "aita", "combined_elephant_wide.csv": "elephant",
           "combined_cancer_myths_wide.csv": "cancer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osf_dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "osf"))
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "..", "data", "prepared", "prompts.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "prepared_syco"))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "meta"), exist_ok=True)
    import json
    have = {json.loads(l)["id"] for l in open(args.prompts, encoding="utf-8")}
    all_prompts = {}
    for split in ("train", "test"):
        path = os.path.join(args.osf_dir, f"{split}_probe_data_syco.csv")
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for short, model in MODELS.items():
            out, excluded = [], Counter()
            for r in rows:
                text = r["prompt"]
                if not text.strip():
                    excluded["blank prompt"] += 1
                    continue
                labels = {}
                for d in SYCO_DIMS:
                    raw = r[f"{d}_{model}"].strip()
                    if raw not in ("0.0", "1.0", "0", "1"):
                        excluded[f"{d}: missing or non-binary label ({raw[:10]!r})" if raw else f"{d}: missing label"] += 1
                        labels = None
                        break
                    labels[d] = int(float(raw))
                if labels is None:
                    continue
                out.append({"id": prompt_id(text), "user_text": text, "dataset": SOURCES[r["source_file"]], "labels": labels})
                all_prompts[prompt_id(text)] = text
            assert len(rows) == len(out) + sum(excluded.values())
            outp = os.path.join(args.out_dir, f"{short}_{split}.jsonl")
            write_jsonl(outp, out)
            write_json(os.path.join(args.out_dir, "meta", f"{short}_{split}.jsonl.meta.json"),
                       {"inputs": {path: sha256(path)}, "git_state": git_state(), "argv": sys.argv,
                        "counts": {"rows_in": len(rows), "excluded": dict(excluded), "rows_out": len(out),
                                   "unique_prompts": len({r["id"] for r in out}),
                                   "by_dataset": dict(Counter(r["dataset"] for r in out))}})
            print(f"{short:9s} {split:5s} in {len(rows)} out {len(out)} excluded {sum(excluded.values())}")
    for short in MODELS:
        tr = {r["id"] for r in map(json.loads, open(os.path.join(args.out_dir, f"{short}_train.jsonl")))}
        te = {r["id"] for r in map(json.loads, open(os.path.join(args.out_dir, f"{short}_test.jsonl")))}
        print(f"{short}: train/test prompt overlap {len(tr & te)} (upstream keeps them; we drop these test rows at training time)")
    missing = [{"id": i, "user_text": t} for i, t in all_prompts.items() if i not in have]
    write_jsonl(os.path.join(args.out_dir, "prompts_missing.jsonl"), missing)
    print(f"{len(all_prompts)} unique prompts, {len(missing)} not in {args.prompts}")


if __name__ == "__main__":
    main()
