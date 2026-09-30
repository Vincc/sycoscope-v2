"""Pooled o_proj inputs (attention heads before the output projection) at chosen blocks, for judged or contrastive rows.

Spans come from utils.activations.prepare: last_prompt = final template token, answer_mean = mean over the visible
answer, last = final token of the sequence. Writes activations/audit/heads/<input stem>[_flipped]_heads.npz with
<pooling>_L<layer> (n, n_heads * head_dim) float32 and the row fields get_activations writes.

Run from the repo root: python -m audit.get_head_activations --input ... --layers 12 14 --poolings answer_mean ...
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from audit import heads
from probe_training.get_activations import CONTRASTIVE_FIELDS, judging_fields
from utils.activations import check_right_padded, prepare
from utils.io import REPO_ROOT, check_counts, read_jsonl, write_meta
from utils.models import load_model_and_tokenizer, model_spec


def pooling_span(p: dict, pooling: str) -> tuple[int, int]:
    if pooling == "last_prompt":
        return tuple(p["spans"]["last_prompt"])
    if pooling == "answer_mean":
        return tuple(p["spans"]["response"])
    if pooling == "last":
        return p["n_tokens"] - 1, p["n_tokens"]
    raise ValueError(f"unknown pooling {pooling!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--poolings", nargs="+", required=True, choices=heads.POOLINGS)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--aita-flipped-only", action="store_true", help="As get_activations: keep flipped_story rows.")
    args = parser.parse_args()

    split = args.input.resolve().relative_to(REPO_ROOT / "generations").parts[1]
    suffix = "_flipped" if args.aita_flipped_only else ""
    out_path = REPO_ROOT / "activations" / "audit" / "heads" / f"{args.input.stem}{suffix}_heads.npz"
    if out_path.exists():
        raise FileExistsError(out_path)
    all_rows = read_jsonl(args.input)
    if any(r["model"] != args.model for r in all_rows):
        raise ValueError(f"{args.input}: rows from another model")
    excluded = Counter()
    rows = all_rows
    if args.aita_flipped_only:
        rows = [r for r in all_rows if r["prompt_col"] == "flipped_story"]
        excluded["original_post_not_selected"] = len(all_rows) - len(rows)

    import torch

    model, tokenizer = load_model_and_tokenizer(args.model, padding_side="right")
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model is not on the GPU")
    prepared, skips = prepare(rows, tokenizer, args.max_length, model_spec(args.model).get("template_kwargs"))
    excluded += Counter(s["reason"] for s in skips)
    n = len(prepared)
    out = {f"{p}_L{L:02d}": np.zeros((n, model.config.hidden_size), dtype=np.float32) for p in args.poolings for L in args.layers}
    store = {}
    handles = heads.register_oproj_inputs(model, args.layers, store)
    order = sorted(range(n), key=lambda i: prepared[i]["n_tokens"], reverse=True)
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for b in range(0, n, args.batch_size):
                idx = order[b:b + args.batch_size]
                enc = tokenizer([prepared[i]["full_text"] for i in idx], return_tensors="pt", padding=True,
                                add_special_tokens=False)  # chat template already includes BOS
                check_right_padded(enc["attention_mask"], [prepared[i]["n_tokens"] for i in idx], [prepared[i]["id"] for i in idx])
                T = enc["input_ids"].shape[1]
                model(**{k: v.to(device) for k, v in enc.items()}, use_cache=False)
                for p in args.poolings:
                    w = torch.from_numpy(heads.pooling_weights([pooling_span(prepared[i], p) for i in idx], T)).to(device)
                    for L in args.layers:
                        pooled = torch.einsum("bt,btd->bd", w, store[L].float()).cpu().numpy()
                        out[f"{p}_L{L:02d}"][idx] = pooled
                if (b // args.batch_size) % 25 == 0:
                    print(f"  {min(b + args.batch_size, n)}/{n}", flush=True)
    finally:
        for h in handles:
            h.remove()
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise ValueError(f"{k}: non-finite values")

    by_id = {r["id"]: r for r in rows}
    kept = [by_id[p["id"]] for p in prepared]
    row_fields = {k: np.array([r[k] for r in kept]) for k in CONTRASTIVE_FIELDS} if "label" in kept[0] else {}
    if split == "judging":
        row_fields = judging_fields(kept)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out, **row_fields, id=np.array([p["id"] for p in prepared]),
             n_tokens=np.array([p["n_tokens"] for p in prepared]), prompt_len=np.array([p["prompt_len"] for p in prepared]),
             n_response_tokens=np.array([p["resp_end"] - p["answer_start"] for p in prepared]))
    counts = check_counts(len(all_rows), excluded, n, args.input.name)
    write_meta(out_path, [args.input], args, counts, {
        "model": args.model, "layers": args.layers, "poolings": args.poolings,
        "site": "self_attn.o_proj input (n_heads * head_dim)", "dtype": "float32", "skips": skips})
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}: {counts}")


if __name__ == "__main__":
    main()
