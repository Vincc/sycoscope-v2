"""Persona vector from judged extraction rollouts: generate_vec.py's filter, activations and mean differences.

Kept pair (generate_vec.get_persona_effective): pos trait >= 50, neg trait < 50, both coherence >= 50.
For every kept row: mean prompt-token, mean response-token and last-prompt-token activation at all 33 hidden_states,
in --dtype, then pos mean minus neg mean in that dtype, cast to float32 (generate_vec.py:70-72).
Writes the vectors npz and a per-row activations npz (response_avg and prompt_last, float32) for later checks.

Run from the repo root:
  python -m reproduction.persona_vector.extract_vectors --trait sycophantic --rollouts ... --scores ... \
      --dtype float32 --batch-size 16 --out-vector ... --out-acts ...
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from utils.io import check_counts, read_jsonl, write_meta
from utils.models import load_tokenizer, model_spec, text_config

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
THRESHOLD = 50
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def keep_pairs(rows, scores, trait):
    """Pos/neg rows paired by (question, instruction, rollout), in pos-file order. Returns (pos, neg, excluded)."""
    by_id = {r["id"]: r for r in rows}
    pos_rows = [r for r in rows if r["polarity"] == "pos"]
    if 2 * len(pos_rows) != len(rows):
        raise AssertionError(f"{len(pos_rows)} pos rows of {len(rows)}")
    pos, neg, excluded = [], [], Counter()
    for p in pos_rows:
        n = by_id[f"{trait}_{p['question_index']}_neg_{p['instruction_index']}_{p['rollout']}"]
        sp, sn = scores[p["id"]], scores[n["id"]]
        values = [sp[trait], sn[trait], sp["coherence"], sn["coherence"]]
        if any(v is None for v in values):
            excluded["score_none"] += 1  # NaN comparisons are False in generate_vec.py
        elif not sp[trait] >= THRESHOLD:
            excluded["pos_trait_below_50"] += 1
        elif not sn[trait] < 100 - THRESHOLD:
            excluded["neg_trait_at_least_50"] += 1
        elif not (sp["coherence"] >= 50 and sn["coherence"] >= 50):
            excluded["coherence_below_50"] += 1
        else:
            pos.append(p)
            neg.append(n)
    return pos, neg, excluded


def hidden_p_and_r(model, tokenizer, rows, batch_size, stats):
    """generate_vec.get_hidden_p_and_r, batched with right padding. Returns three (n, 33, hidden) tensors in model dtype."""
    texts = [r["prompt"] + r["answer"] for r in rows]
    encs = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]  # templated prompt already has BOS
    prompt_lens = [len(tokenizer.encode(r["prompt"], add_special_tokens=False)) for r in rows]
    for r, ids, pl in zip(rows, encs, prompt_lens):
        if ids[:pl] != tokenizer.encode(r["prompt"], add_special_tokens=False):
            stats["boundary_merge"] += 1  # upstream uses prompt_len regardless; counted only
        if len(ids) <= pl:
            raise AssertionError(f"{r['id']}: empty response span")
    n_states = text_config(model.config).num_hidden_layers + 1
    hidden = text_config(model.config).hidden_size
    dtype = next(model.parameters()).dtype
    out = {k: torch.empty((len(rows), n_states, hidden), dtype=dtype) for k in ("prompt_avg", "response_avg", "prompt_last")}
    order = sorted(range(len(rows)), key=lambda i: len(encs[i]), reverse=True)
    pad = tokenizer.pad_token_id
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idxs = order[b : b + batch_size]
            width = max(len(encs[i]) for i in idxs)
            ids = torch.full((len(idxs), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idxs), width), dtype=torch.long)
            for r, i in enumerate(idxs):
                ids[r, : len(encs[i])] = torch.tensor(encs[i])
                mask[r, : len(encs[i])] = 1
            hs = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                       output_hidden_states=True, use_cache=False).hidden_states
            for r, i in enumerate(idxs):
                n, pl = len(encs[i]), prompt_lens[i]
                for layer in range(n_states):
                    h = hs[layer][r]
                    out["prompt_avg"][i, layer] = h[:pl].mean(dim=0).cpu()
                    out["response_avg"][i, layer] = h[pl:n].mean(dim=0).cpu()
                    out["prompt_last"][i, layer] = h[pl - 1].cpu()
            print(f"  {min(b + batch_size, len(order))}/{len(order)}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--out-vector", type=Path, required=True)
    parser.add_argument("--out-acts", type=Path, required=True)
    args = parser.parse_args()
    for p in (args.out_vector, args.out_acts):
        if p.exists():
            raise FileExistsError(p)

    rows = read_jsonl(args.rollouts)
    scores = {s["id"]: s for s in read_jsonl(args.scores)}
    if set(scores) != {r["id"] for r in rows}:
        raise AssertionError("scores and rollouts have different ids")
    pos, neg, excluded = keep_pairs(rows, scores, args.trait)
    counts = check_counts(len(rows) // 2, excluded, len(pos), "pairs")
    print(f"pairs {counts}")

    from transformers import AutoModelForCausalLM

    t0 = time.time()
    tokenizer = load_tokenizer(MODEL, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPES[args.dtype], device_map="auto")
    model.eval()
    if text_config(model.config).num_hidden_layers != model_spec(MODEL)["n_layers"]:
        raise AssertionError("layer count differs from utils.models registry")
    stats = Counter()
    acts = {"pos": hidden_p_and_r(model, tokenizer, pos, args.batch_size, stats),
            "neg": hidden_p_and_r(model, tokenizer, neg, args.batch_size, stats)}

    vectors = {f"{k}_diff": torch.stack([acts["pos"][k][:, l].mean(0).float() - acts["neg"][k][:, l].mean(0).float()
                                         for l in range(acts["pos"][k].shape[1])]).numpy()
               for k in ("prompt_avg", "response_avg", "prompt_last")}
    for k, v in vectors.items():
        if not np.isfinite(v).all():
            raise AssertionError(f"{k} has non-finite values")
    args.out_vector.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_vector, **vectors, pos_ids=np.array([r["id"] for r in pos]), neg_ids=np.array([r["id"] for r in neg]))
    args.out_acts.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_acts, ids=np.array([r["id"] for r in pos + neg]),
             question_index=np.array([r["question_index"] for r in pos + neg]),
             is_pos=np.array([1] * len(pos) + [0] * len(neg), dtype=np.int8),
             **{k: torch.cat([acts["pos"][k], acts["neg"][k]]).float().numpy() for k in ("response_avg", "prompt_last")})
    extra = {"model": MODEL, "revision": model.config._commit_hash, "dtype": args.dtype, "n_kept_pairs": len(pos),
             "boundary_merges": stats["boundary_merge"], "wall_seconds": time.time() - t0,
             "layer_index": "hidden_states index 0..32; paper layer L = index L = block L-1"}
    write_meta(args.out_vector, [args.rollouts, args.scores], args, counts, extra)
    write_meta(args.out_acts, [args.rollouts, args.scores], args, counts, extra)
    print(f"kept {len(pos)} pairs, boundary merges {stats['boundary_merge']} -> {args.out_vector}")


if __name__ == "__main__":
    main()
