"""Build the source rows of one audited method: data/audit/<method>/<name>.jsonl (+ meta).

Needs the pinned clone under external/<name>/ and, for genadi and pandey, the `datasets` package:
  uv run --with datasets==<version> python -m audit.build_source --method genadi

Run from the repo root: python -m audit.build_source --method caa|vennemeyer|genadi|pandey|persona
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from audit import sources
from utils.io import REPO_ROOT, check_counts, git_state, write_json, write_jsonl, write_meta
from utils.models import load_tokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EXTERNAL = {  # method -> (clone dir, repo URL, pinned commit)
    "caa": ("external/caa", "https://github.com/nrimsky/CAA", "5dabbbd"),
    "vennemeyer": ("external/vennemeyer", "https://github.com/cincynlp/disentangle-sycophancy", "217b898"),
    "genadi": ("external/genadi", "https://github.com/rifoagenadi/sycophancy", "09d51c4"),
    "pandey": ("external/pandey", "https://github.com/MVPandey/shared-sycophancy-lying-circuit", "c55356c"),
    "persona": ("external/persona", "https://github.com/safety-research/persona_vectors", "b8e0f04"),
}
VENNEMEYER_FILES = ("math_factorial", "math_factorial_small", "cities_pos_factorial", "cities_neg_factorial",
                    "claims_factorial", "companies_factorial", "counterfactual_factorial", "larger_than_factorial",
                    "smaller_than_factorial", "sp_en_trans_factorial", "sp_en_trans_pro_factorial")


def external_state(method: str) -> dict:
    path, url, pin = EXTERNAL[method]
    state = git_state(REPO_ROOT / path)
    if not state["commit"].startswith(pin):
        raise ValueError(f"{path} is at {state['commit']}, expected pin {pin}")
    return {"source_repo": url, "source_commit": state["commit"], "source_dirty": state["dirty"]}


def split_counts(rows) -> dict:
    return dict(Counter(r["split"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(EXTERNAL))
    parser.add_argument("--val-frac", type=float, default=0.1, help="caa, vennemeyer: val fraction for layer choice.")
    parser.add_argument("--seed", type=int, default=0, help="caa, vennemeyer: seed of our split / val carve.")
    args = parser.parse_args()

    ext = external_state(args.method)
    ext_dir = REPO_ROOT / EXTERNAL[args.method][0]
    out_dir = REPO_ROOT / "data" / "audit" / args.method
    tokenizer = load_tokenizer(MODEL, padding_side="right")
    outputs = []  # (path, rows, inputs, extra)

    if args.method == "caa":
        src = ext_dir / "datasets/generate/sycophancy/generate_dataset.json"
        items = json.loads(src.read_text(encoding="utf-8"))
        rows = sources.caa_rows(items, tokenizer)
        split = sources.assign_group_split(rows, test_frac=0.2, val_frac=args.val_frac, seed=args.seed)
        outputs.append((out_dir / "generate_dataset.jsonl", rows, [src], {"n_items": len(items),
                        "split": {k: split[k] for k in ("test_frac", "val_frac", "seed")}}))

    elif args.method == "vennemeyer":
        for name in VENNEMEYER_FILES:
            src = ext_dir / f"data/factorial/{name}.json"
            items = json.loads(src.read_text(encoding="utf-8"))
            for variant in ("plain", "chat"):
                rows = sources.vennemeyer_rows(items, name, variant, tokenizer, args.val_frac, args.seed)
                outputs.append((out_dir / f"{name}__{variant}.jsonl", rows, [src], {"n_items": len(items), "variant": variant}))

    elif args.method == "genadi":
        import datasets

        ds = datasets.load_dataset("truthfulqa/truthful_qa", "generation")
        split_ds = ds["validation"].train_test_split(test_size=0.25, seed=3407)
        parts = {}
        for name in ("train", "test"):
            d = split_ds[name]
            parts[name] = sources.genadi_rows(name, d["question"], [a[0] for a in d["correct_answers"]],
                                              d["incorrect_answers"], tokenizer, seed=3407)
        val = sources.genadi_val_split(len(parts["train"]), seed=3407)
        for r, v in zip(parts["train"], val):
            r["split"] = "val" if v else "fit"
        for r in parts["test"]:
            r["split"] = "test"
        rows = parts["train"] + parts["test"]
        outputs.append((out_dir / "truthfulqa_dialogues.jsonl", rows, [], {
            "hf_dataset": "truthfulqa/truthful_qa:generation/validation", "hf_fingerprint": ds["validation"]._fingerprint,
            "datasets_version": datasets.__version__, "n_questions": {k: len(split_ds[k]) for k in split_ds}}))

    elif args.method == "pandey":
        import datasets

        stream = datasets.load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="validation", streaming=True)
        pairs = sources.triviaqa_pairs(stream, n=400)
        if len(pairs) != 400:
            raise ValueError(f"expected 400 TriviaQA pairs, got {len(pairs)}")
        rows = sources.pandey_rows(pairs, tokenizer, n_fit=100)
        write_json(out_dir / "triviaqa_pairs.json", [list(p) for p in pairs])
        outputs.append((out_dir / "triviaqa_syc.jsonl", rows, [out_dir / "triviaqa_pairs.json"], {
            "hf_dataset": "mandarjoshi/trivia_qa:unfiltered.nocontext/validation (streamed)",
            "datasets_version": datasets.__version__}))

    elif args.method == "persona":
        extract = ext_dir / "data_generation/trait_data_extract/sycophantic.json"
        trait_data = json.loads(extract.read_text(encoding="utf-8"))
        rows = sources.persona_prompts(trait_data, "sycophantic", n_per_question=10)
        outputs.append((out_dir / "extract_prompts.jsonl", rows, [extract], {"n_per_question": 10}))

    for path, rows, inputs, extra in outputs:
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate ids")
        write_jsonl(path, rows)
        counts = check_counts(len(rows), {}, len(rows), path.name)
        splits = split_counts(rows) if "split" in rows[0] else None
        write_meta(path, inputs, args, counts, {"model": MODEL, **ext, **extra, "split_counts": splits})
        print(f"{path.relative_to(REPO_ROOT)}: {len(rows)} rows {splits or ''}")


if __name__ == "__main__":
    main()
