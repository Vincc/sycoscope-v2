"""Parity of interp.cache_activations with the two existing extractions, on the first --n-rows rows of an input.

Caches the rows into a temporary directory, then recomputes with batch size 1
  utils.activations.extract (get_activations): last_prompt, first5, response, float32 means, and
  reproduction.persona_vector.extract_vectors.hidden_p_and_r: prompt_last, prompt_avg, response_avg at hidden_states
  index L + 1, means in model dtype,
and prints the worst relative L2 error and lowest cosine over rows and blocks, and the largest cached magnitude per
position. Raises if a comparison is outside its tolerance.

Run from the repo root: python -m interp.check_cache_parity --model ... --input ... --n-rows 64 --max-length 4096
"""
import argparse
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from interp.cache_activations import POSITIONS, cache_input, load, settings
from reproduction.persona_vector.extract_vectors import hidden_p_and_r
from utils.activations import act_key, extract, prepare
from utils.io import read_jsonl, write_jsonl
from utils.models import load_model_and_tokenizer, render_prompt

# (our position, reference name, tolerance on relative L2 error). Ours is stored in float16 (~5e-4 relative);
# the persona-vector means are rounded to bfloat16 (~4e-3 relative).
COMPARISONS = [
    ("prompt_last", "get_activations last_prompt", 1e-3),
    ("response_first5", "get_activations first5", 1e-3),
    ("response_mean", "get_activations response", 1e-3),
    ("prompt_last", "persona prompt_last", 1e-2),
    ("prompt_mean", "persona prompt_avg", 1e-2),
    ("response_mean", "persona response_avg", 1e-2),
]
MIN_COS = 0.9999


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)[: args.n_rows]
    model, tokenizer = load_model_and_tokenizer(args.model, padding_side="right")
    expect = settings(args.model, model)
    n_layers = expect["n_layers"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
        write_jsonl(path, rows)
        cache_input(model, tokenizer, path, Path(tmp) / "cache", list(POSITIONS), expect, args.max_length, 1,
                    len(rows), args)
        ours = {pos: load(Path(tmp) / "cache" / pos) for pos in POSITIONS}

    prepared, skips = prepare(rows, tokenizer, args.max_length, expect["template_kwargs"])
    if skips or expect["thinking"]:
        raise ValueError(f"parity needs rows without skips or reasoning; skips {Counter(s['reason'] for s in skips)}")
    old = extract(model, tokenizer, prepared, list(range(n_layers)), batch_size=1)
    persona_rows = [{"id": r["id"], "prompt": render_prompt(tokenizer, r["messages"], expect["template_kwargs"]),
                     "answer": r["response"]} for r in rows]
    stats = Counter()
    persona = hidden_p_and_r(model, tokenizer, persona_rows, 1, stats)
    if stats:
        raise AssertionError(f"persona extraction saw {dict(stats)}")
    reference = {
        "get_activations last_prompt": lambda L: old[act_key("last_prompt", L)],
        "get_activations first5": lambda L: old[act_key("first5", L)],
        "get_activations response": lambda L: old[act_key("response", L)],
        "persona prompt_last": lambda L: persona["prompt_last"][:, L + 1].float().numpy(),
        "persona prompt_avg": lambda L: persona["prompt_avg"][:, L + 1].float().numpy(),
        "persona response_avg": lambda L: persona["response_avg"][:, L + 1].float().numpy(),
    }
    ref_ids = [p["id"] for p in prepared]
    if ref_ids != [r["id"] for r in persona_rows]:
        raise AssertionError("prepared and persona rows are in different orders")

    failures = []
    for pos, ref_name, tol in COMPARISONS:
        ids, acts = ours[pos]
        order = [ids.index(rid) for rid in ref_ids]
        worst_rel, worst_cos = 0.0, 1.0
        for L in range(n_layers):
            a, b = acts[L][order], reference[ref_name](L)
            rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
            cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
            worst_rel, worst_cos = max(worst_rel, float(rel.max())), min(worst_cos, float(cos.min()))
        ok = worst_rel <= tol and worst_cos >= MIN_COS
        print(f"{pos:16} vs {ref_name:28} max rel err {worst_rel:.2e} (tol {tol:.0e})  min cos {worst_cos:.8f}  "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append((pos, ref_name))
    for pos, (ids, acts) in ours.items():
        biggest = {L: float(np.abs(a).max()) for L, a in acts.items()}
        L = max(biggest, key=biggest.get)
        print(f"{pos:16} largest |value| {biggest[L]:.1f} at block {L} (float16 max 65504)")
    if failures:
        raise AssertionError(f"parity failed: {failures}")


if __name__ == "__main__":
    main()
