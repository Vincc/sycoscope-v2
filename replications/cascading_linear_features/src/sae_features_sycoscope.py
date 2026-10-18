"""Extract response-mean CLiF SAE features for grouped SycoScope benchmark rows."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sae_features import load_sae
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_dataset(out_dir, dataset, feats, resid, ids, labels, n_tokens, common_meta):
    out = out_dir / f"{dataset}.npz"
    np.savez_compressed(
        out,
        feats=np.stack(feats),
        resid=np.stack(resid),
        ids=np.array(ids),
        labels=np.array(labels, dtype=np.int8),
        n_tokens=np.array(n_tokens, dtype=np.int64),
        feature_indices=common_meta["feature_indices"],
    )
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    meta = {
        **common_meta,
        "output": out.name,
        "dataset": dataset,
        "counts": {"n_in": len(ids), "excluded": {}, "n_out": len(ids)},
        "label_counts": {"negative": int((np.array(labels) == 0).sum()), "positive": int((np.array(labels) == 1).sum())},
        "response_tokens": {"min": int(min(n_tokens)), "max": int(max(n_tokens)), "mean": float(np.mean(n_tokens))},
        "output_sha256": sha256(out),
    }
    (meta_dir / f"{out.name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(ids)} rows to {out}", flush=True)


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--feature-mask", type=Path, required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--git-dirty", choices=["true", "false"], required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    assert rows
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded {len(rows)} input rows", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("loaded tokenizer", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print("loaded model", flush=True)
    sae = load_sae(model.device)
    print("loaded SAE", flush=True)
    assert model.config.hidden_size == 4096
    feature_mask = np.load(args.feature_mask)
    assert feature_mask.shape == (2, 65536) and feature_mask.dtype == np.bool_
    feature_indices = np.flatnonzero(feature_mask.any(0))
    encoder_weight = sae.encoder_linear.weight[feature_indices].contiguous()
    encoder_bias = sae.encoder_linear.bias[feature_indices].contiguous()
    del sae
    torch.cuda.empty_cache()
    print(f"selected {len(feature_indices)} CLiF features", flush=True)

    common_meta = {
        "input": args.input.name,
        "input_sha256": sha256(args.input),
        "feature_mask_sha256": sha256(args.feature_mask),
        "git": {"commit": args.git_commit, "dirty": args.git_dirty == "true"},
        "argv": sys.argv,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "layer": args.layer,
        "layer_convention": "hidden_states[layer + 1]",
        "pooling": "mean SAE activation over response tokens",
        "dtype": {"feats": "float16", "resid": "float32"},
        "feature_indices": feature_indices.tolist(),
    }
    datasets = []
    for row in rows:
        if not datasets or row["dataset"] != datasets[-1]:
            datasets.append(row["dataset"])
    assert len(datasets) == len(set(datasets)), "rows for each dataset must be contiguous"

    captured = []

    class StopAfterTargetLayer(Exception):
        pass

    def capture_layer(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
        raise StopAfterTargetLayer

    hook = model.model.layers[args.layer].register_forward_hook(capture_layer)
    done = 0
    try:
        for dataset in datasets:
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            existing = args.out_dir / f"{dataset}.npz"
            existing_meta = args.out_dir / "meta" / f"{dataset}.npz.meta.json"
            if args.resume and existing.exists() and existing_meta.exists():
                z = np.load(existing)
                assert len(z["ids"]) == len(dataset_rows)
                assert np.array_equal(z["feature_indices"], feature_indices)
                assert z["ids"].tolist() == [row["id"] for row in dataset_rows]
                assert z["labels"].tolist() == [row["label"] for row in dataset_rows]
                assert sha256(existing) == json.loads(existing_meta.read_text())["output_sha256"]
                done += len(dataset_rows)
                print(f"resumed {len(dataset_rows)} verified rows from {existing}", flush=True)
                continue
            feats, resid, ids, labels, n_tokens = [], [], [], [], []
            for batch in batches(dataset_rows, args.batch_size):
                full_tokens, spans = [], []
                for row in batch:
                    assert row["messages"] and row["messages"][-1]["role"] == "user", row["id"]
                    prefix = tokenizer.apply_chat_template(
                        row["messages"], add_generation_prompt=True, tokenize=True
                    )
                    full = tokenizer.apply_chat_template(
                        [*row["messages"], {"role": "assistant", "content": row["response"]}],
                        tokenize=True,
                    )
                    assert full[: len(prefix)] == prefix, row["id"]
                    assert len(full) <= args.max_length, (row["id"], len(full), args.max_length)
                    end = len(full) - int(full[-1] in tokenizer.all_special_ids)
                    assert end > len(prefix), row["id"]
                    full_tokens.append(full)
                    spans.append((len(prefix), end))
                width = max(map(len, full_tokens))
                input_ids = torch.full(
                    (len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=model.device
                )
                attention_mask = torch.zeros_like(input_ids)
                for j, tokens in enumerate(full_tokens):
                    input_ids[j, : len(tokens)] = torch.tensor(tokens, device=model.device)
                    attention_mask[j, : len(tokens)] = 1
                captured.clear()
                try:
                    model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                except StopAfterTargetLayer:
                    pass
                assert len(captured) == 1
                layer_hidden = captured[0]
                response_hidden = [layer_hidden[j, start:end] for j, (start, end) in enumerate(spans)]
                encoded = torch.relu(F.linear(torch.cat(response_hidden), encoder_weight, encoder_bias))
                offset = 0
                for row, hidden in zip(batch, response_hidden, strict=True):
                    count = len(hidden)
                    row_encoded = encoded[offset : offset + count]
                    offset += count
                    feats.append(row_encoded.float().mean(0).half().cpu().numpy())
                    resid.append(hidden.float().mean(0).cpu().numpy())
                    ids.append(row["id"])
                    labels.append(row["label"])
                    n_tokens.append(count)
                assert offset == len(encoded)
                done += len(batch)
                if done % 100 < len(batch):
                    l0 = (encoded > 0).float().sum(1).mean().item()
                    print(f"{done}/{len(rows)} {dataset} mean_L0={l0:.1f}", flush=True)
            write_dataset(args.out_dir, dataset, feats, resid, ids, labels, n_tokens, common_meta)
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
