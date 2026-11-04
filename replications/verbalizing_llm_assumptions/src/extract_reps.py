"""Stage B (GPU): mean-pooled hidden states of each user prompt at several layers.

Follows upstream train_assumption_probe.extract_reps: the raw user text is tokenized without a chat template
(BOS added by the tokenizer, truncated to --max_length), run through the model, and hidden_states[L] is
averaged over the attention-masked tokens. hidden_states[0] is the embedding output, so index L is the output
of block L-1. Writes <out>.npz with ids (str), n_tokens (int), truncated (bool) and one float32 [N, d] array
per layer under key layer_<L>, plus <out>.meta.json.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="prompts.jsonl from prepare_data.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", required=True, help="comma-separated hidden_states indices")
    ap.add_argument("--max_length", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate prompt ids"
    texts = [r["user_text"] for r in rows]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    n_layers = model.config.num_hidden_layers
    assert all(0 <= L <= n_layers for L in layers), f"layers must be in [0, {n_layers}]"
    device = next(model.parameters()).device

    # longest first so padding waste and the first batch's memory are the worst case
    full_len = [len(tok(t, truncation=False)["input_ids"]) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: -full_len[i])
    reps = {L: np.zeros((len(texts), model.config.hidden_size), dtype=np.float32) for L in layers}
    n_tokens = np.zeros(len(texts), dtype=np.int64)
    with torch.no_grad():
        for b in range(0, len(order), args.batch_size):
            idx = order[b : b + args.batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
            mask = enc["attention_mask"]
            n_tokens[idx] = mask.sum(1).numpy()
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            m = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
            for L in layers:
                pooled = (hs[L].float() * m).sum(1) / m.sum(1)
                reps[L][idx] = pooled.cpu().numpy()
            if (b // args.batch_size) % 50 == 0:
                print(f"{b + len(idx)}/{len(order)}", flush=True)

    truncated = np.array([full_len[i] > args.max_length for i in range(len(texts))])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, ids=np.array(ids), n_tokens=n_tokens, truncated=truncated,
             **{f"layer_{L}": reps[L] for L in layers})
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    meta = {
        "inputs": {args.prompts: sha256(args.prompts)}, "model": args.model, "layers": layers,
        "max_length": args.max_length, "batch_size": args.batch_size, "pooling": "mean", "chat_template": False,
        "dtype": "bfloat16", "hidden_size": model.config.hidden_size, "num_hidden_layers": n_layers,
        "counts": {"prompts": len(texts), "truncated": int(truncated.sum())},
        "git": git.stdout.strip() if git.returncode == 0 else "no git on this machine",
        "argv": sys.argv, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=2)
    print(f"wrote {args.out}: {len(texts)} prompts, {int(truncated.sum())} truncated at {args.max_length} tokens")


if __name__ == "__main__":
    main()
