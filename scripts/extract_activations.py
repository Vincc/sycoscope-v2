"""Residual-stream activations at last_prompt, first5 and response for rows with id, messages, response.

For each input `<stem>.jsonl` writes `<out-dir>/<stem>.npz` (keys `<position>_L<layer>`),
`<stem>.index.jsonl` (row i of every array) and `<stem>.skips.jsonl`.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from utils.activations import POSITIONS, extract, prepare, resolve_layers
from utils.io import check_counts, read_jsonl, write_jsonl, write_meta
from utils.models import load_model_and_tokenizer, text_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layers", type=int, nargs="+", help="0-based block indices.")
    layer_group.add_argument("--layer-fracs", type=float, nargs="+", help="Depth fractions, round(frac * n_layers).")
    parser.add_argument("--max-length", type=int, required=True, help="Rows with more tokens are skipped, not truncated.")
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    stems = [p.stem for p in args.rows]
    if len(set(stems)) != len(stems):
        raise ValueError("input files must have distinct names")
    model, tokenizer = load_model_and_tokenizer(args.model, padding_side="right")
    n_layers = text_config(model.config).num_hidden_layers
    layers = resolve_layers(n_layers, args.layers, args.layer_fracs)
    print(f"{args.model}: {n_layers} blocks; extracting blocks {layers} from hidden_states[L + 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.rows:
        rows = read_jsonl(path)
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate ids")
        prepared, skips = prepare(rows, tokenizer, args.max_length)
        if not prepared:
            raise ValueError(f"{path}: every row was skipped ({Counter(s['reason'] for s in skips)})")
        print(f"[{path.name}] {len(prepared)} rows, {len(skips)} skipped")
        arrays = extract(model, tokenizer, prepared, layers, args.batch_size)

        npz_path = args.out_dir / f"{path.stem}.npz"
        index_path = args.out_dir / f"{path.stem}.index.jsonl"
        skips_path = args.out_dir / f"{path.stem}.skips.jsonl"
        np.savez(npz_path, **arrays)
        index = [
            {
                "row": i,
                "id": p["id"],
                "n_tokens": p["n_tokens"],
                "prompt_len": p["prompt_len"],
                "n_response_tokens": p["resp_end"] - p["prompt_len"],
                "n_first5_tokens": p["spans"]["first5"][1] - p["spans"]["first5"][0],
                "first5_text": p["first5_text"],
            }
            for i, p in enumerate(prepared)
        ]
        write_jsonl(index_path, index)
        write_jsonl(skips_path, skips)

        counts = check_counts(len(rows), Counter(s["reason"] for s in skips), len(prepared), path.name)
        extra = {
            "model": args.model,
            "n_layers": n_layers,
            "layers": layers,
            "layer_convention": "hidden_states[layer + 1]",
            "positions": list(POSITIONS),
            "dtype": "float32",
        }
        for out in (npz_path, index_path, skips_path):
            write_meta(out, [path], args, counts, extra)


if __name__ == "__main__":
    main()
