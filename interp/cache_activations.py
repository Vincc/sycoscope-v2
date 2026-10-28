"""Cache span-mean activations of every block at named positions, computing only (row, position) pairs not yet cached.

Input rows have id, messages, response and model, under generations/<model name>/. Each position of an input file gets
parts <cache-dir>/<model name>/<input path under the model dir, no suffix>/<position>/part-NNNNN.safetensors holding
float16 arrays L00..L<n-1> of shape (rows, hidden), block L = hidden_states[L + 1], with meta/<part>.meta.json listing
the part's row ids, row text SHA-256, the position definition and per-row response stats (next-token agreement and
mean NLL under the model, from the same forward pass). A rerun reads only the meta files: it stops if a
cached row's text, the model revision or a position definition changed, and forward-passes only rows missing a position.

Run from the repo root: python -m interp.cache_activations --model ... --inputs ... --positions ... --cache-dir ...
"""
import argparse
import hashlib
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from utils.activations import check_right_padded, prepare
from utils.io import REPO_ROOT, check_counts, meta_path, read_json, read_jsonl, sha256, write_meta
from utils.models import load_model_and_tokenizer, model_spec, text_config

# n = number of response tokens. A thinking block between prompt and response is in no position.
POSITIONS = {
    "prompt_last": "last prompt token: the final chat-template token before the response",
    "prompt_mean": "mean over all prompt tokens, including BOS and chat-template tokens",
    "response_mean": "mean over all response tokens",
    "response_first5": "mean over the first min(5, n) response tokens",
    "response_first10": "mean over the first min(10, n) response tokens",
    "response_frac10": "mean over the first ceil(10 * n / 100) response tokens",
    "response_frac25": "mean over the first ceil(25 * n / 100) response tokens",
    "response_frac50": "mean over the first ceil(50 * n / 100) response tokens",
    "response_frac75": "mean over the first ceil(75 * n / 100) response tokens",
}


def cache_spans(prompt_len: int, answer_start: int, resp_end: int) -> dict[str, tuple[int, int]]:
    """[start, end) token span of each position in POSITIONS."""
    n = resp_end - answer_start
    if prompt_len < 1 or n < 1:
        raise ValueError(f"empty prompt or response: prompt_len={prompt_len}, n={n}")
    a = answer_start
    spans = {
        "prompt_last": (prompt_len - 1, prompt_len),
        "prompt_mean": (0, prompt_len),
        "response_mean": (a, resp_end),
        "response_first5": (a, a + min(5, n)),
        "response_first10": (a, a + min(10, n)),
    }
    for pct in (10, 25, 50, 75):
        spans[f"response_frac{pct}"] = (a, a + -(-pct * n // 100))  # integer ceil; float ceil(0.1 * 30) is 4
    if spans.keys() != POSITIONS.keys():
        raise AssertionError("cache_spans and POSITIONS name different positions")
    return spans


def to_float16(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        raise ValueError("non-finite activation")
    biggest = float(np.abs(x).max())
    if biggest > float(np.finfo(np.float16).max):  # float(): comparing to a float16 scalar casts biggest to float16
        raise ValueError(f"activation magnitude {biggest} is outside the float16 range")
    return x.astype(np.float16)


def read_parts(position_dir: Path, expect: dict) -> list[dict]:
    """Meta records of a position's parts in part order, each checked against expect.

    Raises on a part file without a meta file (an interrupted write) and on a row id in two parts.
    """
    metas = sorted((position_dir / "meta").glob("part-*.safetensors.meta.json"))
    recorded = {m.name.removesuffix(".meta.json") for m in metas}
    orphans = sorted(p.name for p in position_dir.glob("part-*.safetensors") if p.name not in recorded)
    if orphans:
        raise ValueError(f"{position_dir}: part files without a meta file, delete them: {orphans}")
    records, seen = [], set()
    for m in metas:
        meta = read_json(m)
        for key, value in expect.items():
            if meta[key] != value:
                raise ValueError(f"{m}: {key} is {meta[key]!r}, expected {value!r}")
        repeated = seen & set(meta["ids"])
        if repeated or len(set(meta["ids"])) != len(meta["ids"]):
            raise ValueError(f"{m}: row ids cached twice, e.g. {sorted(repeated)[:3]}")
        seen |= set(meta["ids"])
        records.append(meta)
    return records


def load(position_dir: Path, layers: list[int] | None = None, verify: bool = True) -> tuple[list[str], dict[int, np.ndarray]]:
    """Row ids and {block: (rows, hidden) float32} of one cached position, over all parts in part order.

    verify=False skips the per-file SHA-256 check, for callers that already verified the files in this process.
    """
    records = read_parts(position_dir, {"definition": POSITIONS[position_dir.name]})
    if not records:
        raise FileNotFoundError(f"{position_dir}: no cached parts")
    ids, chunks = [], []
    for meta in records:
        path = position_dir / meta["output"]
        if verify and sha256(path) != meta["file_sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from its meta file")
        with safe_open(path, framework="numpy") as f:
            blocks = range(meta["n_layers"]) if layers is None else layers
            chunks.append({L: f.get_tensor(f"L{L:02d}").astype(np.float32) for L in blocks})
        ids += meta["ids"]
    return ids, {L: np.concatenate([c[L] for c in chunks]) for L in chunks[0]}


def pool(model, tokenizer, rows: list[dict], todo: dict[str, set], n_layers: int, batch_size: int):
    """({position: (rows needing it, (n_layers, rows, hidden) float32 span means)}, {row id: response stats})."""
    import torch

    members = {pos: [r for r in rows if r["id"] in ids] for pos, ids in todo.items()}
    members = {pos: rs for pos, rs in members.items() if rs}
    slot = {pos: {r["id"]: i for i, r in enumerate(rs)} for pos, rs in members.items()}
    hidden = text_config(model.config).hidden_size
    out = {pos: np.empty((n_layers, len(rs), hidden), dtype=np.float32) for pos, rs in members.items()}
    stats = {}
    device = next(model.parameters()).device
    with torch.no_grad():
        for b in range(0, len(rows), batch_size):
            batch = rows[b : b + batch_size]
            enc = tokenizer([r["full_text"] for r in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False)  # chat template already has BOS
            check_right_padded(enc["attention_mask"], [r["n_tokens"] for r in batch], [r["id"] for r in batch])
            hs = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True, use_cache=False,
                       logits_to_keep=1).hidden_states  # logits for every token would need tokens x vocab memory
            for r, row in enumerate(batch):
                a, e = row["answer_start"], row["resp_end"]
                # hidden_states[-1] is after the final norm, so lm_head gives the logits that predict tokens a..e-1
                logits = model.get_output_embeddings()(hs[-1][r, a - 1 : e - 1]).float()
                targets = enc["input_ids"][r, a:e].to(device)
                stats[row["id"]] = {
                    "next_token_agreement": float((logits.argmax(dim=-1) == targets).float().mean()),
                    "response_nll": float(torch.nn.functional.cross_entropy(logits, targets)),
                }
                for pos in members:
                    if row["id"] not in slot[pos]:
                        continue
                    s, e = row["spans"][pos]
                    # block L = hidden_states[L + 1]; the last one is after the final norm
                    means = torch.stack([hs[L + 1][r, s:e].float().mean(dim=0) for L in range(n_layers)])
                    out[pos][:, slot[pos][row["id"]]] = means.cpu().numpy()
    return {pos: (members[pos], out[pos]) for pos in members}, stats


def write_part(position_dir: Path, part: int, position: str, rows: list[dict], acts: np.ndarray, stats: dict,
               input_path: Path, args, expect: dict) -> None:
    path = position_dir / f"part-{part:05d}.safetensors"
    if path.exists() or meta_path(path).exists():
        raise FileExistsError(path)
    half = to_float16(acts)
    position_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        save_file({f"L{L:02d}": np.ascontiguousarray(half[L]) for L in range(len(half))}, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    extra = {**expect, "position": position, "definition": POSITIONS[position], "file_sha256": sha256(path),
             "max_abs": float(np.abs(acts).max()), "ids": [r["id"] for r in rows],
             "text_sha256": [r["text_sha256"] for r in rows], "spans": [list(r["spans"][position]) for r in rows],
             **{k: [stats[r["id"]][k] for r in rows] for k in ("next_token_agreement", "response_nll")}}
    write_meta(path, [input_path], args, check_counts(len(rows), {}, len(rows), path.name), extra)


def cache_input(model, tokenizer, input_path: Path, out_root: Path, positions: list[str], expect: dict,
                max_length: int, batch_size: int, rows_per_part: int, args) -> dict[str, int]:
    """Cache the missing (row, position) pairs of one input file. Returns rows written per position."""
    rows = read_jsonl(input_path)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{input_path}: duplicate ids")
    wrong_model = [r["id"] for r in rows if r["model"] != expect["model"]]
    if wrong_model:
        raise ValueError(f"{input_path}: {len(wrong_model)} rows are not from {expect['model']}, e.g. {wrong_model[:3]}")
    if expect["thinking"] and any("reasoning" not in r for r in rows):
        raise ValueError(f"{expect['model']} is a thinking model but {input_path} has rows without a reasoning field")

    prepared, skips = prepare(rows, tokenizer, max_length, expect["template_kwargs"])
    for p in prepared:
        p["spans"] = cache_spans(p["prompt_len"], p["answer_start"], p["resp_end"])
        p["text_sha256"] = hashlib.sha256(p["full_text"].encode("utf-8")).hexdigest()
    counts = check_counts(len(rows), Counter(s["reason"] for s in skips), len(prepared), input_path.name)
    print(f"{input_path}: {counts}", flush=True)
    text_hash = {p["id"]: p["text_sha256"] for p in prepared}

    todo, next_part = {}, {}
    for pos in positions:
        records = read_parts(out_root / pos, {**expect, "position": pos, "definition": POSITIONS[pos]})
        cached = {rid: digest for m in records for rid, digest in zip(m["ids"], m["text_sha256"], strict=True)}
        stale = [rid for rid, digest in cached.items() if text_hash.get(rid) != digest]
        if stale:
            raise ValueError(f"{out_root / pos}: {len(stale)} cached rows are not among this run's rows or their text "
                             f"changed, e.g. {stale[:3]}")
        todo[pos] = set(text_hash) - set(cached)
        next_part[pos] = 1 + max((int(m["output"][5:10]) for m in records), default=-1)
        print(f"  {pos}: {len(cached)} cached, {len(todo[pos])} to compute", flush=True)

    needed = sorted((p for p in prepared if any(p["id"] in todo[pos] for pos in positions)),
                    key=lambda p: p["n_tokens"], reverse=True)
    n_layers = expect["n_layers"]
    written = Counter()
    t0, n_done_tokens, all_stats = time.time(), 0, {}
    for start in range(0, len(needed), rows_per_part):
        chunk = needed[start : start + rows_per_part]
        pooled, stats = pool(model, tokenizer, chunk, todo, n_layers, batch_size)
        for pos, (members, acts) in pooled.items():
            write_part(out_root / pos, next_part[pos], pos, members, acts, stats, input_path, args, expect)
            next_part[pos] += 1
            written[pos] += len(members)
        all_stats |= stats
        n_done_tokens += sum(p["n_tokens"] for p in chunk)
        print(f"  forwarded {start + len(chunk)}/{len(needed)} rows, {n_done_tokens} tokens in {time.time() - t0:.0f} s",
              flush=True)
    if all_stats:
        agree = np.array([s["next_token_agreement"] for s in all_stats.values()])
        nll = np.array([s["response_nll"] for s in all_stats.values()])
        print(f"  next-token agreement mean {agree.mean():.4f}, min {agree.min():.4f}; "
              f"response NLL mean {nll.mean():.4f}, max {nll.max():.4f}", flush=True)

    for pos in positions:
        records = read_parts(out_root / pos, {**expect, "position": pos, "definition": POSITIONS[pos]})
        if {rid for m in records for rid in m["ids"]} != set(text_hash):
            raise AssertionError(f"{out_root / pos}: cached rows differ from this run's {len(text_hash)} rows")
    return {pos: written[pos] for pos in positions}


def settings(name: str, model) -> dict:
    """Run settings every part records; a cached part with different settings stops a rerun."""
    revision = model.config._commit_hash
    if not revision:
        raise ValueError(f"{name}: no Hub commit hash in the loaded config")
    if getattr(text_config(model.config), "final_logit_softcapping", None):
        raise ValueError(f"{name}: response NLL from lm_head would skip the model's logit softcapping")
    spec = model_spec(name)
    return {
        "model": name,
        "model_revision": revision,
        "thinking": spec["thinking"],
        "template_kwargs": spec.get("template_kwargs"),
        "n_layers": text_config(model.config).num_hidden_layers,
        "layer_convention": "block L = hidden_states[L + 1]; the last block is after the final norm",
        "forward_dtype": str(next(model.parameters()).dtype),
        "pooling": "float32 mean over the span",
        "dtype": "float16",
        "row_stats": "next_token_agreement: share of response tokens that are the model's argmax given the tokens "
                     "before them; response_nll: mean -log p of the response tokens, in nats",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--positions", nargs="+", choices=list(POSITIONS), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True, help="Rows with more tokens are skipped, not truncated.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--rows-per-part", type=int, default=1024, help="Rows per part file.")
    args = parser.parse_args()

    model_dir = args.model.split("/")[-1]
    out_roots = []
    for path in args.inputs:
        parts = path.resolve().relative_to(REPO_ROOT / "generations").parts
        if parts[0] != model_dir or path.suffix != ".jsonl":
            raise ValueError(f"{path} is not a .jsonl under generations/{model_dir}/")
        out_roots.append(args.cache_dir / model_dir / Path(*parts[1:]).with_suffix(""))

    model, tokenizer = load_model_and_tokenizer(args.model, padding_side="right")
    expect = settings(args.model, model)
    for path, out_root in zip(args.inputs, out_roots):
        written = cache_input(model, tokenizer, path, out_root, args.positions, expect, args.max_length,
                              args.batch_size, args.rows_per_part, args)
        print(f"{path}: rows written per position {written}", flush=True)


if __name__ == "__main__":
    main()
