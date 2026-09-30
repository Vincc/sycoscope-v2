"""Pooled block outputs (and o_proj inputs) for audit source rows, at every block.

Per source file writes activations/audit/<method>/<stem>.npz:
  sum__<pool>__<unit>__<0|1> (n_layers, hidden) float64 and count__<pool>__<unit>__<0|1>: class sums over 'fit' rows
  X__<pool> (m, n_layers, hidden) float32 for cached rows (--cache heldout: val and test rows; all: every row)
  H__<pool> (n, n_layers, heads * head_dim) float16 o_proj inputs for every row (--head-pools)
  row fields: id, split, group, labels__<unit>, n_tokens, cached (bool)
Block L is the decoder block's output (hook), so the last block is before the final norm.

Run from the repo root: python -m audit.extract_source --source data/audit/caa/generate_dataset.jsonl --pools letter ...
"""
import argparse
from pathlib import Path

import numpy as np

from audit import heads, sources
from utils.io import REPO_ROOT, check_counts, read_jsonl, write_meta
from utils.models import load_model_and_tokenizer, text_config

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FP16_MAX = 65504.0


def extract_file(model, tokenizer, path: Path, args, max_cached: int):
    rows = read_jsonl(path)
    method = rows[0]["method"]
    units = sorted(rows[0]["labels"])
    n, n_layers, hidden = len(rows), text_config(model.config).num_hidden_layers, text_config(model.config).hidden_size
    for r in rows:
        missing = [p for p in args.pools + args.head_pools if p not in r["pool"]]
        if missing:
            raise ValueError(f"{r['id']}: no pooling {missing}")
        if sorted(r["labels"]) != units:
            raise ValueError(f"{r['id']}: units {sorted(r['labels'])} differ from {units}")
    cached = np.array([bool(args.pools) and (args.cache == "all" or r["split"] in ("val", "test")) for r in rows])
    if cached.sum() > max_cached:
        raise ValueError(f"{path.name}: {cached.sum()} rows to cache > cap {max_cached}")
    cache_row = np.cumsum(cached) - 1

    import torch

    encs = []
    for r in rows:
        ids = sources.tokenize(tokenizer, r)["input_ids"]
        if len(ids) != r["n_tokens"]:
            raise AssertionError(f"{r['id']}: {len(ids)} tokens, source row says {r['n_tokens']}")
        encs.append(ids)

    sums = {(p, u, c): np.zeros((n_layers, hidden)) for p in args.pools for u in units for c in (0, 1)}
    counts = {k: 0 for k in sums}
    X = {p: np.zeros((int(cached.sum()), n_layers, hidden), dtype=np.float32) for p in args.pools}
    H = {p: None for p in args.head_pools}
    layers = list(range(n_layers))
    block_store, oproj_store = {}, {}
    handles = heads.register_block_outputs(model, layers, block_store)
    if args.head_pools:
        handles += heads.register_oproj_inputs(model, layers, oproj_store)
    order = sorted(range(n), key=lambda i: len(encs[i]), reverse=True)
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    try:
        with torch.no_grad():
            for b in range(0, n, args.batch_size):
                idx = order[b:b + args.batch_size]
                T = len(encs[idx[0]])
                input_ids = torch.full((len(idx), T), pad, dtype=torch.long)
                mask = torch.zeros((len(idx), T), dtype=torch.long)
                for j, i in enumerate(idx):
                    input_ids[j, :len(encs[i])] = torch.tensor(encs[i])
                    mask[j, :len(encs[i])] = 1  # right padding: real positions keep their single-row values
                model(input_ids=input_ids.to(device), attention_mask=mask.to(device), use_cache=False)
                for p in args.pools:
                    w = torch.from_numpy(heads.pooling_weights([rows[i]["pool"][p] for i in idx], T)).to(device)
                    pooled = torch.stack([torch.einsum("bt,btd->bd", w, block_store[L].float()) for L in layers], 1)
                    pooled = pooled.cpu().numpy().astype(np.float64)  # (batch, n_layers, hidden)
                    if not np.isfinite(pooled).all():
                        raise ValueError(f"non-finite activations in batch at {b}")
                    for j, i in enumerate(idx):
                        if cached[i]:
                            X[p][cache_row[i]] = pooled[j]
                        if rows[i]["split"] == "fit":
                            for u in units:
                                c = rows[i]["labels"][u]
                                sums[(p, u, c)] += pooled[j]
                                counts[(p, u, c)] += 1
                for p in args.head_pools:
                    w = torch.from_numpy(heads.pooling_weights([rows[i]["pool"][p] for i in idx], T)).to(device)
                    pooled = torch.stack([torch.einsum("bt,btd->bd", w, oproj_store[L].float()) for L in layers], 1).cpu().numpy()
                    if H[p] is None:
                        H[p] = np.zeros((n,) + pooled.shape[1:], dtype=np.float16)
                    if not np.isfinite(pooled).all() or np.abs(pooled).max() >= FP16_MAX:
                        raise ValueError(f"o_proj inputs not representable in float16 at batch {b}")
                    H[p][idx] = pooled.astype(np.float16)
                if (b // args.batch_size) % 20 == 0:
                    print(f"  {path.stem}: {min(b + args.batch_size, n)}/{n}", flush=True)
    finally:
        for h in handles:
            h.remove()

    out_path = REPO_ROOT / "activations" / "audit" / method / f"{path.stem}.npz"
    if out_path.exists():
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"sum__{p}__{u}__{c}": s for (p, u, c), s in sums.items()}
    arrays |= {f"count__{p}__{u}__{c}": np.int64(k) for (p, u, c), k in counts.items()}
    arrays |= {f"X__{p}": x for p, x in X.items()} | {f"H__{p}": h for p, h in H.items()}
    np.savez(out_path, **arrays, id=np.array([r["id"] for r in rows]), split=np.array([r["split"] for r in rows]),
             group=np.array([r["group"] for r in rows]), n_tokens=np.array([r["n_tokens"] for r in rows]),
             cached=cached, **{f"labels__{u}": np.array([r["labels"][u] for r in rows], dtype=np.int8) for u in units})
    n_fit = sum(r["split"] == "fit" for r in rows)
    counts_rec = check_counts(n, {}, n, path.name)
    write_meta(out_path, [path], args, counts_rec, {
        "model": MODEL, "method": method, "units": units, "n_layers": n_layers, "layers": layers,
        "layer_convention": "decoder block output (hook); last block before final norm",
        "pools": args.pools, "head_pools": args.head_pools, "n_fit": n_fit, "n_cached": int(cached.sum()),
        "class_counts": {f"{p}/{u}/{c}": k for (p, u, c), k in counts.items()}, "dtype": {"X": "float32", "H": "float16"}})
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}: {n} rows, {n_fit} fit, {int(cached.sum())} cached", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--pools", nargs="*", default=[], help="Residual poolings to sum and cache.")
    parser.add_argument("--head-pools", nargs="*", default=[], help="o_proj-input poolings stored for every row.")
    parser.add_argument("--cache", choices=("heldout", "all"), required=True)
    parser.add_argument("--max-cached", type=int, required=True, help="Crash if more rows would be cached.")
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    import torch

    model, tokenizer = load_model_and_tokenizer(MODEL, padding_side="right")
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model is not on the GPU")
    torch.manual_seed(0)
    for path in args.source:
        extract_file(model, tokenizer, path, args, args.max_cached)


if __name__ == "__main__":
    main()
