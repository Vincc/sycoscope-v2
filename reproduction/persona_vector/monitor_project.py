"""Last-prompt-token projection of each rollout onto persona vectors (cal_projection.py, projection_type prompt_last_proj).

bf16 model, one rollout per forward pass on prompt + answer (add_special_tokens=False). For every vector file and every
hidden_states index 0..32: (a . v) / ||v||, a = hidden_states[layer][prompt_len - 1] (bf16), v = response_avg_diff[layer]
(float32). Writes one JSONL row per rollout with {vector name: [33 projections]}.

Run from the repo root:
  python -m reproduction.persona_vector.monitor_project --rollouts ... --vectors a.npz b.npz --out ...
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from utils.io import read_jsonl, write_jsonl, write_meta
from utils.models import load_model_and_tokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def a_proj_b(a, b):
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    vectors = {p.name.removesuffix(".npz"): torch.tensor(np.load(p)["response_avg_diff"]) for p in args.vectors}
    rows = read_jsonl(args.rollouts)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL, padding_side="right")
    out = []
    with torch.no_grad():
        for n, r in enumerate(rows):
            inputs = tokenizer(r["prompt"] + r["answer"], return_tensors="pt", add_special_tokens=False).to(model.device)
            prompt_len = len(tokenizer.encode(r["prompt"], add_special_tokens=False))
            hs = model(**inputs, output_hidden_states=True).hidden_states
            last = torch.stack([h[0, prompt_len - 1, :] for h in hs]).detach().cpu()  # (33, hidden) bf16
            proj = {name: a_proj_b(last, v).tolist() for name, v in vectors.items()}  # bf16 * float32 -> float32
            out.append({"id": r["id"], "condition": r["condition"], "question_index": r["question_index"],
                        "rollout": r["rollout"], "prompt_len": prompt_len, "proj": proj})
            if (n + 1) % 200 == 0:
                print(f"  {n + 1}/{len(rows)}", flush=True)
    write_jsonl(args.out, out)
    write_meta(args.out, [args.rollouts, *args.vectors], args, {"n_in": len(rows), "n_out": len(out)},
               {"model": MODEL, "revision": model.config._commit_hash, "dtype": "bfloat16",
                "layer_index": "hidden_states index 0..32", "wall_seconds": time.time() - t0})


if __name__ == "__main__":
    main()
