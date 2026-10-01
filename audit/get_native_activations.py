"""Block outputs and o_proj inputs of benchmark (or control) rows at each audited method's own read position.

Rows are rendered with audit.sources.native_row in one --format and kept in the order of --cached-npz (the repo's
activation cache of the same file), so scores line up with the cached rows and their labels. Writes
activations/audit/native/<cached stem>__<format>_activations.npz with <pool>_L<layer> (block output) and
oproj_<pool>_L<layer> (o_proj input), plus id, n_tokens and the label fields of the cached file.

Run from the repo root: python -m audit.get_native_activations --input <jsonl> --cached-npz <npz> --format plain ...
"""
import argparse
from pathlib import Path

import numpy as np

from audit import heads, sources
from utils.activations import LABEL_PREFIX
from utils.io import REPO_ROOT, check_counts, meta_path, read_json, read_jsonl, write_meta
from utils.models import load_model_and_tokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
COPY_FIELDS = ("benchmark", "group", "truncated", "label", "polarity", "pair_index", "cell", "pair_type", "prompt_id",
               "n_response_tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cached-npz", type=Path, required=True, help="Repo activation cache of the same rows (order, labels).")
    parser.add_argument("--format", required=True, choices=("chat", "plain", "chat_double_bos"))
    parser.add_argument("--name-suffix", default="", help="Appended to the format in the output name, e.g. _L19.")
    parser.add_argument("--pools", nargs="*", default=[])
    parser.add_argument("--layers", type=int, nargs="*", default=[])
    parser.add_argument("--head-pools", nargs="*", default=[])
    parser.add_argument("--head-layers", type=int, nargs="*", default=[])
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    cached = np.load(args.cached_npz)
    stem = args.cached_npz.name.removesuffix("_activations.npz")
    out_path = REPO_ROOT / "activations" / "audit" / "native" / f"{stem}__{args.format}{args.name_suffix}_activations.npz"
    if out_path.exists():
        raise FileExistsError(out_path)
    by_id = {r["id"]: r for r in read_jsonl(args.input)}
    ids = cached["id"].tolist()
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} cached ids are not in {args.input}, e.g. {missing[:3]}")
    rows = [by_id[i] for i in ids]
    if any(r["model"] != MODEL for r in rows):
        raise ValueError("rows from another model")

    model, tokenizer = load_model_and_tokenizer(MODEL, padding_side="right")
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model is not on the GPU")
    prepared = [sources.native_row(r, args.format, tokenizer) for r in rows]
    too_long = [r["id"] for r, p in zip(rows, prepared) if len(p["ids"]) > args.max_length]
    if too_long:
        raise ValueError(f"{len(too_long)} rows exceed {args.max_length} tokens, e.g. {too_long[:3]}")
    for p in prepared:
        missing = [k for k in args.pools + args.head_pools if k not in p["pool"]]
        if missing:
            raise ValueError(f"format {args.format} has no pooling {missing}")

    import torch

    n, hidden = len(rows), model.config.hidden_size
    out = {f"{p}_L{L:02d}": np.zeros((n, hidden), dtype=np.float32) for p in args.pools for L in args.layers}
    out |= {f"oproj_{p}_L{L:02d}": np.zeros((n, hidden), dtype=np.float32) for p in args.head_pools for L in args.head_layers}
    block_store, oproj_store = {}, {}
    handles = heads.register_block_outputs(model, args.layers, block_store)
    handles += heads.register_oproj_inputs(model, args.head_layers, oproj_store)
    order = sorted(range(n), key=lambda i: len(prepared[i]["ids"]), reverse=True)
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for b in range(0, n, args.batch_size):
                idx = order[b:b + args.batch_size]
                T = len(prepared[idx[0]]["ids"])
                input_ids = torch.full((len(idx), T), tokenizer.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(idx), T), dtype=torch.long)
                for j, i in enumerate(idx):
                    t = prepared[i]["ids"]
                    input_ids[j, :len(t)] = torch.tensor(t)
                    mask[j, :len(t)] = 1  # right padding: real positions keep their single-row values
                model(input_ids=input_ids.to(device), attention_mask=mask.to(device), use_cache=False)
                for pools, store, prefix, layers in ((args.pools, block_store, "", args.layers),
                                                     (args.head_pools, oproj_store, "oproj_", args.head_layers)):
                    for p in pools:
                        w = torch.from_numpy(heads.pooling_weights([prepared[i]["pool"][p] for i in idx], T)).to(device)
                        for L in layers:
                            out[f"{prefix}{p}_L{L:02d}"][idx] = torch.einsum("bt,btd->bd", w, store[L].float()).cpu().numpy()
                if (b // args.batch_size) % 50 == 0:
                    print(f"  {stem}: {min(b + args.batch_size, n)}/{n}", flush=True)
    finally:
        for h in handles:
            h.remove()
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise ValueError(f"{k}: non-finite values")

    copied = {k: cached[k] for k in cached.files if k in COPY_FIELDS or k.startswith(LABEL_PREFIX)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out, **copied, id=cached["id"], n_tokens=np.array([len(p["ids"]) for p in prepared]),
             **{f"span_{k}": np.array([p["pool"][k] for p in prepared]) for k in prepared[0]["pool"]})
    counts = check_counts(n, {}, n, out_path.name)
    cmeta = read_json(meta_path(args.cached_npz))
    write_meta(out_path, [args.input, args.cached_npz], args, counts, {
        "model": MODEL, "format": args.format, "layers": sorted(set(args.layers) | set(args.head_layers)),
        "pools": args.pools, "head_pools": args.head_pools, "layer_convention": "decoder block output (hook)",
        "positions": [f"{p}" for p in args.pools], "rows_from": cmeta["output"]})
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}: {counts}", flush=True)


if __name__ == "__main__":
    main()
