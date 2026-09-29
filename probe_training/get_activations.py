"""Activations from every block at the last template token, first-five mean and full-answer mean.

Input rows have id, messages, response, and live in generations/<model name>/<training|evaluation|judging>/.
Writes <input stem>_activations.npz next to the input: arrays <position>_L<layer> of shape (n, hidden),
plus per-row id, n_tokens, prompt_len, n_reasoning_tokens (a thinking block between prompt and answer, pooled
into no position), n_response_tokens (visible answer), first5_text, and
  contrastive inputs: label, polarity, pair_index, cell, pair_type, prompt_id
  judging inputs: benchmark, group (skyline split group), truncated, and if judged labels__<name> (int8, -1 = None)
Skipped rows are listed in meta/<output>.meta.json. Optional unpooled token vectors are stored in
compressed NPZ shards under judging/activations/<output stem>_raw/, with row IDs and token spans.
For AITA-NTA-FLIP, --aita-flipped-only keeps the flipped_story response from each pair.

Run from the repo root: python -m probe_training.get_activations --model ... --input ...
"""
import argparse
import hashlib
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from utils.activations import LABEL_PREFIX, NO_LABEL, POSITIONS, extract, prepare, raw_token_span, resolve_layers
from utils.io import check_counts, read_jsonl, write_meta
from utils.models import load_model_and_tokenizer, model_spec, text_config

SPLITS = ("training", "evaluation", "judging")
CONTRASTIVE_FIELDS = ("label", "polarity", "pair_index", "cell", "pair_type", "prompt_id")


def skyline_group(row: dict) -> str:
    """Rows sharing a group must share a side of a skyline split.

    AYS and truthfulqa by question, SyPR by utterance, ELEPHANT by prompt text, and both tellings of an
    AITA-NTA-FLIP pair by the pair's row_id.
    """
    if row["benchmark"] in ("are_you_sure", "truthfulqa"):
        text = row["question"]
    elif row["benchmark"] == "sypr":
        if row["messages"][-1] != {"role": "user", "content": row["utterance_text"]}:
            raise ValueError(f"{row['id']}: final SyPR message differs from utterance_text")
        text = row["utterance_text"]
    elif row["benchmark"] == "elephant" and row.get("source") == "AITA-NTA-FLIP":
        text = f"AITA-NTA-FLIP:{row['row_id']}"
    elif row["benchmark"] == "elephant":
        if row["messages"][-1]["role"] != "user":
            raise ValueError(f"{row['id']}: last message is not the user prompt")
        text = row["messages"][-1]["content"]
    else:
        raise ValueError(f"{row['id']}: no group rule for benchmark {row['benchmark']!r}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def judging_fields(rows: list[dict]) -> dict[str, np.ndarray]:
    """Per-row benchmark, group, truncated and, if the rows are judged, one labels__<name> array per label."""
    fields = {
        "benchmark": np.array([r["benchmark"] for r in rows]),
        "group": np.array([skyline_group(r) for r in rows]),
        "truncated": np.array([r["truncated"] for r in rows], dtype=bool),
    }
    if not any("labels" in r for r in rows):
        return fields
    names = sorted(rows[0]["labels"])
    for r in rows:
        if sorted(r["labels"]) != names:
            raise ValueError(f"{r['id']}: label names {sorted(r['labels'])} differ from {names}")
    for name in names:
        values = []
        for r in rows:
            v = r["labels"][name]
            if v is None:
                values.append(NO_LABEL)
            elif v in (0, 1) and not isinstance(v, bool):
                values.append(v)
            else:
                raise ValueError(f"{r['id']}: label {name}={v!r} is not 0, 1 or None")
        fields[LABEL_PREFIX + name] = np.array(values, dtype=np.int8)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", help="0-based block indices. Omit for all blocks.")
    layer_group.add_argument("--layer-fracs", type=float, nargs="+", help="Depth fractions, round(frac * n_layers). Omit layer flags for all blocks.")
    parser.add_argument("--max-length", type=int, required=True, help="Rows with more tokens are skipped, not truncated.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--raw-token-scope", choices=("first5", "answer", "full"),
                        help="Also save unpooled token vectors in compressed NPZ shards; omit to save only pooled vectors.")
    parser.add_argument("--aita-flipped-only", action="store_true",
                        help="For AITA-NTA-FLIP, extract only flipped_story responses, keeping their pair labels.")
    args = parser.parse_args()

    split_dir = args.input.resolve().parent
    if split_dir.name not in SPLITS or split_dir.parent.parent.name != "generations":
        raise ValueError(f"{args.input} is not under generations/<model>/<{'|'.join(SPLITS)}>/")
    if split_dir.parent.name != args.model.split("/")[-1]:
        raise ValueError(f"{args.input} was generated by {split_dir.parent.name}, not {args.model}")
    suffix = "_flipped" if args.aita_flipped_only else ""
    out_path = split_dir / f"{args.input.stem}{suffix}_activations.npz"
    if out_path.exists():
        raise FileExistsError(out_path)
    raw_final = split_dir / "activations" / f"{out_path.stem}_raw"
    if args.raw_token_scope and raw_final.exists():
        raise FileExistsError(raw_final)

    all_rows = read_jsonl(args.input)
    ids = [r["id"] for r in all_rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{args.input}: duplicate ids")
    selected_out = Counter()
    if args.aita_flipped_only:
        if any(r["benchmark"] != "elephant" or r["source"] != "AITA-NTA-FLIP" or
               r["prompt_col"] not in ("original_post", "flipped_story") for r in all_rows):
            raise ValueError("--aita-flipped-only requires AITA-NTA-FLIP judging rows")
        rows = [r for r in all_rows if r["prompt_col"] == "flipped_story"]
        selected_out["original_post_not_selected"] = len(all_rows) - len(rows)
    else:
        rows = all_rows

    spec = model_spec(args.model)
    if spec["thinking"] and any("reasoning" not in r for r in rows):
        raise ValueError(f"{args.model} is a thinking model but {args.input} has rows without a reasoning field")
    model, tokenizer = load_model_and_tokenizer(args.model, padding_side="right")
    n_layers = text_config(model.config).num_hidden_layers
    layers = resolve_layers(n_layers, args.layers, args.layer_fracs)
    print(f"{args.model}: extracting {len(layers)}/{n_layers} blocks from hidden_states[L + 1]")

    prepared, skips = prepare(rows, tokenizer, args.max_length, spec.get("template_kwargs"))
    if not prepared:
        raise ValueError(f"{args.input}: every row was skipped ({Counter(s['reason'] for s in skips)})")
    print(f"{len(prepared)} rows, {len(skips)} skipped")
    pooled_parent = split_dir / "activations"
    pooled_parent.mkdir(exist_ok=True)
    pooled_dir = Path(tempfile.mkdtemp(prefix=f"{out_path.stem}_pooled_tmp_", dir=pooled_parent))
    raw_dir = None
    if args.raw_token_scope:
        raw_dir = Path(tempfile.mkdtemp(prefix=f"{raw_final.name}_tmp_", dir=pooled_parent))
        raw_tokens = sum(raw_token_span(p, args.raw_token_scope)[1] - raw_token_span(p, args.raw_token_scope)[0]
                         for p in prepared)
        raw_gb = raw_tokens * text_config(model.config).hidden_size * len(layers) * 4 / 1e9
        print(f"Uncompressed raw token vectors: {raw_tokens} tokens, about {raw_gb:.1f} GB before ZIP", flush=True)
    raw_shards = [] if raw_dir is not None else None
    try:
        arrays = extract(model, tokenizer, prepared, layers, args.batch_size, raw_dir, args.raw_token_scope,
                         raw_shards, pooled_dir)

        # Pair fields and judged labels are kept so probes need not re-join the JSONL.
        by_id = {r["id"]: r for r in rows}
        kept = [by_id[p["id"]] for p in prepared]
        row_fields = {}
        if "label" in rows[0]:
            row_fields = {k: np.array([r[k] for r in kept]) for k in CONTRASTIVE_FIELDS}
        if split_dir.name == "judging":
            row_fields = judging_fields(kept)
        np.savez_compressed(
            out_path,
            **arrays,
            **row_fields,
            id=np.array([p["id"] for p in prepared]),
            n_tokens=np.array([p["n_tokens"] for p in prepared]),
            prompt_len=np.array([p["prompt_len"] for p in prepared]),
            n_reasoning_tokens=np.array([p["answer_start"] - p["prompt_len"] for p in prepared]),
            n_response_tokens=np.array([p["resp_end"] - p["answer_start"] for p in prepared]),
            first5_text=np.array([p["first5_text"] for p in prepared]),
        )
        counts = check_counts(len(all_rows), selected_out + Counter(s["reason"] for s in skips),
                              len(prepared), args.input.name)
        extra = {
            "model": args.model,
            "n_layers": n_layers,
            "layers": layers,
            "layer_convention": "hidden_states[layer + 1]",
            "final_layer_post_norm": n_layers - 1 in layers,
            "positions": list(POSITIONS),
            "template_kwargs": spec.get("template_kwargs"),
            "dtype": "float32",
            "compression": "ZIP_DEFLATED",
            "skips": skips,
        }
        if split_dir.name == "judging":
            labels = {k[len(LABEL_PREFIX):]: v for k, v in row_fields.items() if k.startswith(LABEL_PREFIX)}
            extra["label_counts"] = {n: {"pos": int((y == 1).sum()), "neg": int((y == 0).sum()), "none": int((y == NO_LABEL).sum())}
                                     for n, y in labels.items()}
        write_meta(out_path, [args.input], args, counts, extra)
        if raw_dir is not None:
            for shard in raw_shards:
                shard_counts = check_counts(shard["n_rows"], {}, shard["n_rows"], shard["path"].name)
                write_meta(shard["path"], [args.input], args, shard_counts,
                           {"model": args.model, "layers": layers, "scope": args.raw_token_scope,
                            "n_token_vectors": shard["n_tokens"], "dtype": "float32", "compression": "ZIP_DEFLATED",
                            "source_activations": out_path.name})
            raw_dir.rename(raw_final)
        print(f"Wrote {out_path}: {counts}")
    finally:
        shutil.rmtree(pooled_dir)
        if raw_dir is not None and raw_dir.exists():
            shutil.rmtree(raw_dir)


if __name__ == "__main__":
    main()
