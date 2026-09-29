"""Port an old prompt_probes run's Perez-prompt generations and activations into the probe_training layout.

Writes to generations/<model>/training/model_written_evals/perez_user_prompts/:
  <name>_contrastive_responses.jsonl + _activations.npz   every pair x polarity x prompt
  <name>_neutral_responses.jsonl + _activations.npz       no system prompt, one row per prompt
in the same row and array format as generate_response.py and get_activations.py.

Every row is re-tokenized with utils.activations.prepare, and its prompt_len, response length and
first5_text must equal the old index, so the old arrays sit on the spans the current code would use.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from utils.activations import POSITIONS, act_key, prepare
from utils.contrastive import LABEL, TAG
from utils.io import REPO_ROOT, check_counts, read_json, read_jsonl, write_jsonl, write_meta
from utils.models import load_tokenizer

CONTRASTIVE_FIELDS = ("label", "polarity", "pair_index", "cell", "pair_type", "prompt_id")
NEUTRAL_SLUG = "neutral"


def new_row(old: dict, pair_index: int | None) -> dict:
    """Old generation row in generate_response.py's output format."""
    user = {"role": "user", "content": old["user_prompt"]}
    row = {
        "prompt_id": old["prompt_id"],
        "source": old["prompt_source"],
        "user_prompt": old["user_prompt"],
    }
    if pair_index is None:
        row = {"id": old["prompt_id"], **row, "messages": [user]}
    else:
        row = {
            "id": f"pair{pair_index:02d}__{TAG[old['polarity']]}__{old['prompt_id']}",
            **row,
            "pair_index": pair_index,
            "cell": old["cell"],
            "pair_type": old["pair_type"],
            "polarity": old["polarity"],
            "label": old["label"],
            "messages": [{"role": "system", "content": old["system_prompt"]}, user],
        }
    return {
        **row,
        "response": old["response"],
        "truncated": old["n_response_tokens"] >= old["max_new_tokens"] - 2,  # old classify_degenerate rule
        "degenerate": old["degenerate"],
        "model": old["model"],
        "old_example_id": old["example_id"],
    }


def load_slug(old_run: Path, slug: str, layers: list[int]):
    """(generation rows by example_id, index rows, {act_key: array}) for one old cell."""
    gens = {r["example_id"]: r for r in read_jsonl(old_run / "generations" / f"{slug}.jsonl")}
    index = read_jsonl(old_run / "activations" / f"{slug}_index.jsonl")
    if [i["row"] for i in index] != list(range(len(index))):
        raise AssertionError(f"{slug}: index rows are not 0..n-1")
    with np.load(old_run / "activations" / f"{slug}.npz") as z:
        arrays = {act_key(p, L): z[act_key(p, L)] for p in POSITIONS for L in layers}
    for k, a in arrays.items():
        if a.shape[0] != len(index) or a.dtype != np.float32:
            raise AssertionError(f"{slug} {k}: shape {a.shape} dtype {a.dtype}, index has {len(index)} rows")
    return gens, index, arrays


def check_spans(rows: list[dict], index: list[dict], tokenizer, max_length: int) -> None:
    """Current tokenization must reproduce the old index for every row, with nothing skipped."""
    prepared, skips = prepare(rows, tokenizer, max_length)
    if skips:
        raise AssertionError(f"{len(skips)} rows skipped on re-tokenization: {skips[:3]}")
    for p, old in zip(prepared, index):
        got = (p["prompt_len"], p["resp_end"] - p["prompt_len"], p["first5_text"])
        want = (old["prompt_len"], old["n_response_tokens"], old["first5_text"])
        if got != want:
            raise AssertionError(f"{old['example_id']}: spans {got} != old index {want}")


def write_set(out_stem: Path, rows: list[dict], arrays: dict, extra_fields: dict, index: list[dict], args, meta_extra):
    responses = out_stem.with_name(out_stem.name + "_responses.jsonl")
    acts = out_stem.with_name(out_stem.name + "_responses_activations.npz")
    write_jsonl(responses, rows)
    n_resp = np.array([i["n_response_tokens"] for i in index])
    prompt_len = np.array([i["prompt_len"] for i in index])
    np.savez(
        acts,
        **arrays,
        **extra_fields,
        id=np.array([r["id"] for r in rows]),
        n_tokens=prompt_len + n_resp,
        prompt_len=prompt_len,
        n_response_tokens=n_resp,
        first5_text=np.array([i["first5_text"] for i in index]),
    )
    counts = check_counts(len(rows), {}, len(rows), out_stem.name)
    counts["truncated"] = sum(r["truncated"] for r in rows)
    write_meta(responses, [], args, counts, meta_extra)
    write_meta(acts, [responses], args, counts, {**meta_extra, "skips": []})
    print(f"{responses.name}: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True, help="Old prompt_probes/results/<run> dir with generations/ and activations/.")
    parser.add_argument("--pairs", type=Path, required=True, help="Pairs JSON; old system prompts must match it exactly.")
    parser.add_argument("--name", required=True, help="Output name prefix, e.g. old_main.")
    args = parser.parse_args()

    old_meta = json.loads((args.old_run / "activations" / "meta.json").read_text(encoding="utf-8"))
    if old_meta["layer_convention"] != "hidden_states[layer + 1]" or list(old_meta["positions"]) != list(POSITIONS):
        raise ValueError(f"old run uses {old_meta['layer_convention']} / {old_meta['positions']}")
    if old_meta["dtype"] != "float32":
        raise ValueError(f"old activations are {old_meta['dtype']}")
    model, layers = old_meta["model"], old_meta["layers"]
    tokenizer = load_tokenizer(model, padding_side="right")
    pairs = read_json(args.pairs)
    slug_of = {}
    for slug in old_meta["cells"]:
        if slug != NEUTRAL_SLUG:
            first = read_jsonl(args.old_run / "generations" / f"{slug}.jsonl")[0]
            slug_of[first["cell"]] = slug
    if set(slug_of) != {p["cell"] for p in pairs}:
        raise ValueError(f"old cells {sorted(slug_of)} != pairs file cells")

    out_dir = REPO_ROOT / "generations" / model.split("/")[-1] / "training" / "model_written_evals" / "perez_user_prompts"
    meta_extra = {"model": model, "n_layers": old_meta["n_layers"], "layers": layers,
                  "layer_convention": old_meta["layer_convention"], "positions": list(POSITIONS), "dtype": "float32",
                  "old_run": args.old_run.as_posix(), "old_activations_meta": old_meta}

    rows, index, arrays = [], [], {k: [] for k in (act_key(p, L) for p in POSITIONS for L in layers)}
    for k, pair in enumerate(pairs):
        slug = slug_of[pair["cell"]]
        gens, idx, arrs = load_slug(args.old_run, slug, layers)
        if len(idx) != len(gens):
            raise AssertionError(f"{slug}: {len(gens)} generations but {len(idx)} activation rows")
        cell_rows = []
        for i in idx:
            g = gens[i["example_id"]]
            if g["model"] != model or g["pair_index"] != k or g["pair_type"] != pair["type"]:
                raise ValueError(f"{g['example_id']}: model/pair_index/pair_type differ from the pairs file position {k}")
            if g["system_prompt"] != pair[g["polarity"]] or g["label"] != LABEL[g["polarity"]]:
                raise ValueError(f"{g['example_id']}: system prompt or label differs from {args.pairs}")
            cell_rows.append(new_row(g, k))
        check_spans(cell_rows, idx, tokenizer, old_meta["max_length"])
        rows += cell_rows
        index += idx
        for key in arrays:
            arrays[key].append(arrs[key])
        print(f"[{slug} -> pair{k:02d}] {len(cell_rows)} rows, spans match")
    if Counter(Counter((r["pair_index"], r["prompt_id"]) for r in rows).values()) != Counter({2: len(rows) // 2}):
        raise AssertionError("some (pair, prompt) lack exactly one row per polarity")
    fields = {f: np.array([r[f] for r in rows]) for f in CONTRASTIVE_FIELDS}
    write_set(out_dir / f"{args.name}_contrastive", rows, {k: np.concatenate(v) for k, v in arrays.items()},
              fields, index, args, meta_extra)

    gens, idx, arrs = load_slug(args.old_run, NEUTRAL_SLUG, layers)
    if len(idx) != len(gens):
        raise AssertionError(f"neutral: {len(gens)} generations but {len(idx)} activation rows")
    neutral = [new_row(gens[i["example_id"]], None) for i in idx]
    for r in neutral:
        if r["model"] != model:
            raise ValueError(f"{r['id']}: model {r['model']}")
    check_spans(neutral, idx, tokenizer, old_meta["max_length"])
    print(f"[neutral] {len(neutral)} rows, spans match")
    write_set(out_dir / f"{args.name}_neutral", neutral, arrs, {}, idx, args, meta_extra)


if __name__ == "__main__":
    main()
