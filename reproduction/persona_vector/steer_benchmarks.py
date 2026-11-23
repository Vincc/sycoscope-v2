"""Generate benchmark responses with a steering vector added at one block; one output file per coefficient.

Each input row's `messages` end on a user turn (ELEPHANT: the prompt; Are You Sure: question, turn-1 answer, pushback).
The prompt is the chat template with its single BOS, as in the benchmark generations. Steering adds coef x the
unnormalised `response_avg_diff[--layer]` (hidden_states index, hook on block layer - 1) at the last position of every
forward pass (generate.steer_last_position); coefficient 0 runs without a hook. --random-direction-seed replaces the
vector by a seeded random direction of the same norm. Every cell restarts the torch seed with the same batches.

Writes <out-dir>/<items stem>_L<layer>[_rand<seed>]_c<coef>_responses.jsonl; an existing cell file is skipped.

Run from the repo root:
  python -m reproduction.persona_vector.steer_benchmarks --items ... --vector ... --layer 16 --coefs -1 0 1 ...
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from reproduction.persona_vector.generate import steer_last_position
from utils.io import check_counts, read_jsonl, write_jsonl, write_meta
from utils.models import load_model_and_tokenizer, merge_adapter, render_prompt, resolve_terminators

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VECTOR_KEY = "response_avg_diff"


def encode_prompts(tokenizer, rows: list[dict]) -> dict:
    """Left-padded chat-template prompts; raises unless each starts with exactly one BOS and ends unpadded."""
    prompts = [render_prompt(tokenizer, r["messages"]) for r in rows]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)  # template already has BOS
    ids, mask = enc["input_ids"], enc["attention_mask"]
    if not bool(mask[:, -1].all()):
        raise AssertionError("last position is padding; generation needs left padding")
    first = mask.int().argmax(dim=1)
    rows_idx = torch.arange(len(rows))
    if not bool((ids[rows_idx, first] == tokenizer.bos_token_id).all()) or \
            bool((ids[rows_idx, first + 1] == tokenizer.bos_token_id).any()):
        raise AssertionError("a prompt does not start with exactly one BOS")
    return enc


def split_new_tokens(new: list[int], terminators: list[int], max_new_tokens: int) -> tuple[list[int], bool]:
    """(ids before the first terminator, truncated). Truncated means no terminator within max_new_tokens."""
    stop = next((i for i, t in enumerate(new) if t in terminators), None)
    if stop is None:
        if len(new) != max_new_tokens:
            raise AssertionError(f"no terminator but only {len(new)} of {max_new_tokens} new tokens")
        return new, True
    return new[:stop], False


def steering_direction(vectors: np.ndarray, layer: int, random_seed: int | None) -> np.ndarray:
    """vectors[layer], or a random direction from random_seed scaled to its norm."""
    v = vectors[layer]
    if random_seed is None:
        return v
    d = np.random.default_rng(random_seed).standard_normal(v.shape[0])
    return (d / np.linalg.norm(d) * np.linalg.norm(v)).astype(np.float32)


def generate_cell(model, tokenizer, rows, batch_size, sampling, max_new_tokens, terminators, seed) -> list[dict]:
    """One sampled response per row, in input order."""
    torch.manual_seed(seed)
    out = []
    for b in range(0, len(rows), batch_size):
        batch = rows[b : b + batch_size]
        enc = {k: v.to(model.device) for k, v in encode_prompts(tokenizer, batch).items()}
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, eos_token_id=terminators,
                                 pad_token_id=tokenizer.pad_token_id, use_cache=True, do_sample=True, **sampling)
        prompt_len = enc["input_ids"].shape[1]
        for r, row_ids in zip(batch, ids):
            kept, truncated = split_new_tokens(row_ids[prompt_len:].tolist(), terminators, max_new_tokens)
            out.append({**r, "response": tokenizer.decode(kept, skip_special_tokens=True).strip(),
                        "truncated": truncated, "n_response_tokens": len(kept)})
        print(f"  {len(out)}/{len(rows)}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, required=True, help="JSONL rows with `messages` ending on a user turn.")
    parser.add_argument("--vector", type=Path, required=True, help=f"npz with {VECTOR_KEY} (n_layers + 1, hidden).")
    parser.add_argument("--layer", type=int, required=True, help="hidden_states index; the hook is on block layer - 1.")
    parser.add_argument("--coefs", type=float, nargs="+", required=True)
    parser.add_argument("--random-direction-seed", type=int, help="Steer along a random direction of the same norm.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True, help="0 means off.")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--adapter", type=Path, help="LoRA adapter directory merged into the model; adds _ft-<dir name>.")
    args = parser.parse_args()
    if args.random_direction_seed is not None and 0 in args.coefs:
        raise ValueError("coefficient 0 has no direction; run it without --random-direction-seed")

    rows = read_jsonl(args.items)
    if len({r["id"] for r in rows}) != len(rows) or any(r["messages"][-1]["role"] != "user" for r in rows):
        raise ValueError(f"{args.items}: duplicate ids, or messages not ending on a user turn")
    model, tokenizer = load_model_and_tokenizer(MODEL, padding_side="left")
    if args.adapter is not None:
        model = merge_adapter(model, args.adapter)
    adapter = [args.adapter / "adapter_model.safetensors"] if args.adapter is not None else []
    vectors = np.load(args.vector)[VECTOR_KEY]
    if vectors.shape != (model.config.num_hidden_layers + 1, model.config.hidden_size):
        raise ValueError(f"{args.vector}: {VECTOR_KEY} has shape {vectors.shape}")
    if not 1 <= args.layer <= model.config.num_hidden_layers:
        raise ValueError(f"layer {args.layer} outside 1..{model.config.num_hidden_layers}")
    direction = steering_direction(vectors, args.layer, args.random_direction_seed)
    terminators = resolve_terminators(model, tokenizer)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    generation = {"model": MODEL, "revision": model.config._commit_hash, "backend": "transformers", "dtype": "bfloat16",
                  "max_new_tokens": args.max_new_tokens, **sampling, "seed": args.seed,
                  "adapter": None if args.adapter is None else args.adapter.name}
    persona = vectors[args.layer]
    cos = float(direction @ persona / (np.linalg.norm(direction) * np.linalg.norm(persona)))

    rand = "" if args.random_direction_seed is None else f"_rand{args.random_direction_seed}"
    ft = "" if args.adapter is None else f"_ft-{args.adapter.name}"
    for coef in args.coefs:
        path = args.out_dir / f"{args.items.stem}_L{args.layer}{rand}_c{coef:+.1f}{ft}_responses.jsonl"
        if path.exists():
            print(f"{path} exists, skipping")
            continue
        steer = {"vector": args.vector.as_posix(), "key": VECTOR_KEY,
                 "direction": "persona" if args.random_direction_seed is None else "random",
                 "random_seed": args.random_direction_seed, "layer": args.layer, "block": args.layer - 1, "coef": coef,
                 "norm": float(np.linalg.norm(direction)), "cos_with_persona": cos}
        print(f"{path.name}: block {args.layer - 1}, coef {coef}, {steer['direction']}", flush=True)
        t0 = time.time()
        if coef == 0:
            out = generate_cell(model, tokenizer, rows, args.batch_size, sampling, args.max_new_tokens, terminators, args.seed)
        else:
            with steer_last_position(model, direction, coef, args.layer - 1):
                out = generate_cell(model, tokenizer, rows, args.batch_size, sampling, args.max_new_tokens, terminators,
                                    args.seed)
        out = [{**r, "model": MODEL, "generation": generation, "steer": steer} for r in out]
        write_jsonl(path, out)
        counts = check_counts(len(rows), {}, len(out), path.name)
        write_meta(path, [args.items, args.vector, *adapter], args, {**counts, "n_truncated": sum(r["truncated"] for r in out)},
                   {"generation": generation, "steer": steer, "terminators": terminators,
                    "wall_seconds": time.time() - t0})


if __name__ == "__main__":
    main()
