"""Sample rollouts for a prompt set with upstream's two generation paths, optionally steering one block.

--mode sample: upstream's vLLM path (eval_persona.sample) made explicit for HF generate.
--mode steer:  upstream's HF path (eval_persona.sample_steering); each --cells entry LAYER:COEF writes one file,
               LAYER is the paper layer (hidden_states index), the hook is on block LAYER - 1; COEF 0 means no hook.

Run from the repo root:
  python -m reproduction.persona_vector.generate --mode sample --prompts ... --out ...
  python -m reproduction.persona_vector.generate --mode steer --prompts ... --out-dir ... --vector ... --cells 16:0.8
"""
import argparse
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from utils.io import read_jsonl, write_jsonl, write_meta
from utils.models import load_model_and_tokenizer, resolve_terminators

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 1000
SAMPLING = {
    # vLLM SamplingParams(temperature=1, top_p=1, min_tokens=1); vLLM has no top-k by default, HF needs top_k=0.
    "sample": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 1},
    # model.generate(do_sample=True, temperature=1, top_p=1, min_new_tokens=1); HF's default top_k=50 applies.
    "steer": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50, "min_new_tokens": 1},
}


@contextmanager
def steer_last_position(model, vector, coef, block):
    """Add coef * vector to block `block`'s output at the last position of every forward pass (activation_steer.py positions="response")."""
    steer = coef * torch.as_tensor(vector, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)

    def hook(module, inputs, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        t2 = t.clone()
        t2[:, -1, :] += steer.to(t.device)
        return (t2, *out[1:]) if isinstance(out, (tuple, list)) else t2

    handle = model.model.layers[block].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def sample_rows(model, tokenizer, rows, batch_size, sampling, terminators, seed):
    """Generate one rollout per row in order. Prompts get two BOS tokens: tokenizer() on templated text, as upstream."""
    torch.manual_seed(seed)
    out = []
    for b in range(0, len(rows), batch_size):
        batch = rows[b : b + batch_size]
        prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True)  # adds a second BOS, as vLLM and upstream's HF path do
        if not bool((enc["input_ids"][:, -1] != tokenizer.pad_token_id).all()):
            raise AssertionError("last position is padding; generation needs left padding")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=terminators,
                                 pad_token_id=tokenizer.pad_token_id, use_cache=True, **sampling)
        prompt_len = enc["input_ids"].shape[1]
        for r, p, row_ids, mask in zip(batch, prompts, ids, enc["attention_mask"]):
            new = row_ids[prompt_len:].tolist()
            stop = next((i for i, t in enumerate(new) if t in terminators), None)  # padding after a stop is eos too
            first = int(mask.argmax())
            out.append({**r, "prompt": p, "answer": tokenizer.decode(row_ids[prompt_len:], skip_special_tokens=True),
                        "n_prompt_tokens": int(mask.sum()),
                        "n_bos": int((row_ids[first:first + 2] == tokenizer.bos_token_id).sum()),
                        "n_new_tokens": len(new) if stop is None else stop + 1, "finished": stop is not None})
        print(f"  {min(b + batch_size, len(rows))}/{len(rows)}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(SAMPLING), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="--mode sample: output rollouts JSONL.")
    parser.add_argument("--out-dir", type=Path, help="--mode steer: one L<layer>_c<coef>.jsonl per cell.")
    parser.add_argument("--vector", type=Path, help="--mode steer: vector npz with response_avg_diff (33, hidden).")
    parser.add_argument("--cells", nargs="+", help="--mode steer: LAYER:COEF, paper layer numbering.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "sample" and (args.out is None or args.cells or args.vector):
        raise ValueError("--mode sample takes --out only")
    if args.mode == "steer" and (args.out_dir is None or not args.cells or args.vector is None):
        raise ValueError("--mode steer needs --out-dir, --vector and --cells")

    rows = read_jsonl(args.prompts)
    model, tokenizer = load_model_and_tokenizer(MODEL, padding_side="left")  # pad = eos, as upstream
    terminators = resolve_terminators(model, tokenizer)
    sampling = SAMPLING[args.mode]
    revision = model.config._commit_hash

    if args.mode == "sample":
        if args.out.exists():
            raise FileExistsError(args.out)
        t0 = time.time()
        out = sample_rows(model, tokenizer, rows, args.batch_size, sampling, terminators, args.seed)
        write_jsonl(args.out, out)
        write_meta(args.out, [args.prompts], args, {"n_in": len(rows), "n_out": len(out)},
                   {"model": MODEL, "revision": revision, "sampling": sampling, "max_new_tokens": MAX_NEW_TOKENS,
                    "terminators": terminators, "wall_seconds": time.time() - t0})
        return

    vectors = np.load(args.vector)["response_avg_diff"]
    for cell in args.cells:
        layer, coef = int(cell.split(":")[0]), float(cell.split(":")[1])
        path = args.out_dir / f"L{layer:02d}_c{coef:.1f}.jsonl"
        if path.exists():
            print(f"{path} exists, skipping")
            continue
        if coef == 0 and layer != 0:
            raise ValueError("the unsteered cell is written as 0:0")
        if coef != 0 and not 1 <= layer < vectors.shape[0]:
            raise ValueError(f"layer {layer} outside 1..{vectors.shape[0] - 1}")
        print(f"cell layer {layer} (block {layer - 1}) coef {coef}", flush=True)
        t0 = time.time()
        if coef == 0:
            out = sample_rows(model, tokenizer, rows, args.batch_size, sampling, terminators, args.seed)
        else:
            with steer_last_position(model, vectors[layer], coef, layer - 1):
                out = sample_rows(model, tokenizer, rows, args.batch_size, sampling, terminators, args.seed)
        out = [{**r, "layer": layer, "coef": coef} for r in out]
        write_jsonl(path, out)
        write_meta(path, [args.prompts, args.vector], args, {"n_in": len(rows), "n_out": len(out)},
                   {"model": MODEL, "revision": revision, "sampling": sampling, "max_new_tokens": MAX_NEW_TOKENS,
                    "terminators": terminators, "layer": layer, "block": layer - 1, "coef": coef,
                    "wall_seconds": time.time() - t0})


if __name__ == "__main__":
    main()
