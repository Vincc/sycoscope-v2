"""Per-token detector scores on a few benchmark rows (for heatmaps), on the GPU.

For each benchmark: --per-label rows with target label 1 and as many with label 0, drawn with --seed among rows whose
response has at most --max-response-tokens tokens. Each detector scores every token's activation with its own rule
(DIM: h . d - threshold; logistic: standardised decision function; Genadi head: logistic on the head's o_proj-input
slice). Residual detectors and references read the chat template (one BOS); Genadi and Pandey read the dialogue with
a second BOS, aligned to the chat tokens by a shift of one. Shown span: the last user message and the response.
Writes reports/audit_existing_methods/token_scores.json.

Run from the repo root: python -m audit.token_scores --per-label 1 --seed 0 --max-response-tokens 300
"""
import argparse
import csv

import numpy as np

from audit import heads, sources
from audit.analyze import CELL_PAIR, benchmark_cell
from utils import probes
from utils.activations import NO_LABEL, prepare, read_labels
from utils.io import REPO_ROOT, check_counts, read_jsonl, write_json, write_meta
from utils.models import load_model_and_tokenizer, render_prompt

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
P = REPO_ROOT / "probes" / "Llama-3.1-8B-Instruct"
J = REPO_ROOT / "generations" / "Llama-3.1-8B-Instruct" / "judging"
TARGETS = REPO_ROOT / "reports" / "llama31_tefo_probe_transfer" / "targets.csv"
SELECTED = REPO_ROOT / "reports" / "llama31_tefo_probe_transfer" / "validation_selected.csv"
# name, file, probe id, method, layer, format, native read position, cell built on
DETECTORS = [
    ("CAA (L13, recipe layer)", P / "audit_caa_respmean/caa.probes.npz", "caa__dim__response_L13", "dim", 13, "chat",
     "mean over response tokens (our construction)", "Position-Subjective / Explicit"),
    ("Vennemeyer SyA (chat, L30)", P / "audit_vennemeyer_math_chat_last_native/syc.probes.npz", "syc__dim__last_content_L30", "dim", 30,
     "chat", "last content token", "Position-Verifiable / Explicit"),
    ("Vennemeyer GA (chat, L30)", P / "audit_vennemeyer_math_chat_last_native/ga.probes.npz", "ga__dim__last_content_L30", "dim", 30,
     "chat", "last content token", "Position-Verifiable / Explicit"),
    ("Vennemeyer SyPr (chat, L16)", P / "audit_vennemeyer_math_chat_last_native/pr.probes.npz", "pr__dim__last_content_L16", "dim", 16,
     "chat", "last content token", "Person-Traits / Explicit"),
    ("Genadi best head (L12H12)", P / "audit_genadi_heads/detectors.npz", "genadi__best_head", "head", 12, "chat_double_bos",
     "mean over the final answer", "Position-Verifiable / Explicit"),
    ("Pandey LR-27 (their probe)", P / "audit_pandey_lr27_native/syc.probes.npz", "syc__logistic__pandey_faithful_L27", "logistic", 27,
     "chat_double_bos", "code-faithful prompt index", "Position-Verifiable / Explicit"),
    ("Persona Vectors (L15)", P / "audit_persona/persona.probes.npz", "persona__dim__response_L15", "dim", 15, "chat",
     "mean over response tokens", "Mixed (Position + Person-Traits)"),
]


def reference_probes():
    """{pair index: (manifest row)} for the universal pair and the matched-cell pairs (validation-selected)."""
    manifest = {m["probe_id"]: m for m in read_jsonl(P / "synthetic_tefo_C_sweep" / "manifest.jsonl")}
    with open(SELECTED, encoding="utf-8", newline="") as f:
        sel = {int(r["pair_index"]): r["probe_id"] for r in csv.DictReader(f)}
    return {p: manifest[sel[p]] for p in {0, *CELL_PAIR.values()}}


def row_token_scores(row: dict, tokenizer, model, store: dict, ostore: dict, res_layers: list[int], refs: dict, cell: str,
                     max_user_tokens: int) -> dict:
    """Tokens of the last user message and the response, with every detector's per-token score and the read positions."""
    import torch

    (prep,), _ = prepare([row], tokenizer, 4096)
    chat_ids = tokenizer(prep["full_text"], add_special_tokens=False)["input_ids"]
    dbl = sources.native_row(row, "chat_double_bos", tokenizer)
    if dbl["ids"][1:1 + len(chat_ids)] != chat_ids:
        raise AssertionError(f"{row['id']}: double-BOS tokens do not align with the chat tokens")
    outs = {}
    with torch.no_grad():
        for fmt, ids in (("chat", chat_ids), ("chat_double_bos", dbl["ids"])):
            model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
            outs[fmt] = ({L: store[L][0].float().cpu().numpy() for L in res_layers}, ostore[12][0].float().cpu().numpy())
    p0, a0, end = prep["prompt_len"], prep["answer_start"], prep["resp_end"]
    user_start = max(prep["full_text"].rfind(row["messages"][-1]["content"]), 0)
    enc = tokenizer(prep["full_text"], add_special_tokens=False, return_offsets_mapping=True)
    u_first = next(k for k, (s, _) in enumerate(enc["offset_mapping"]) if s >= user_start)
    u0 = max(u_first, p0 - 5 - max_user_tokens)
    span = list(range(u0, end))
    det_scores = {}
    for name, path, pid, method, L, fmt, _, _ in DETECTORS:
        res, oproj = outs[fmt]
        idx = [k + (1 if fmt == "chat_double_bos" else 0) for k in span]
        if method == "head":
            s = probes.score_logistic(probes.load_probe(np.load(path), pid, "logistic"), heads.split_heads(oproj[idx], 32)[:, 12])
        else:
            s = probes.score(method, probes.load_probe(np.load(path), pid, method), res[L][idx].astype(np.float64))
        det_scores[name] = [round(float(v), 4) for v in s]
    for tag, pair in (("Contrastive universal (P00)", 0), ("Contrastive matched-cell pair", CELL_PAIR[cell])):
        m = refs[pair]
        pr = probes.load_probe(np.load(P / "synthetic_tefo_C_sweep" / m["file"]), m["probe_id"], m["method"])
        det_scores[tag] = [round(float(v), 4) for v in probes.score(m["method"], pr, outs["chat"][0][m["layer"]][span].astype(np.float64))]
    return {"tokens": [tokenizer.decode([chat_ids[k]]) for k in span],
            "segment": ["user" if k < p0 - 5 else "template" if k < a0 else "response" for k in span],
            "n_turns": len(row["messages"]), "user_truncated": u0 > u_first,
            "native_index": {"last_prompt": p0 - 1 - u0, "last_content": sources.last_content_index(tokenizer, chat_ids) - u0,
                             "pandey_faithful": dbl["pool"]["pandey_faithful"][0] - 1 - u0},
            "scores": det_scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-label", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-response-tokens", type=int, required=True)
    parser.add_argument("--extra-per-label", type=int, default=0,
                        help="Further rows per label, drawn from the remaining rows with a second generator (seed + 1000), so the first draw is unchanged.")
    parser.add_argument("--max-user-tokens", type=int, default=120, help="Show at most this many tokens of the last user message.")
    args = parser.parse_args()
    import torch

    with open(TARGETS, encoding="utf-8", newline="") as f:
        targets = list(csv.DictReader(f))
    refs = reference_probes()
    model, tokenizer = load_model_and_tokenizer(MODEL, padding_side="right")
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model is not on the GPU")
    res_layers = sorted({d[4] for d in DETECTORS if d[3] != "head"} | {m["layer"] for m in refs.values()})
    store, ostore = {}, {}
    handles = heads.register_block_outputs(model, res_layers, store) + heads.register_oproj_inputs(model, [12], ostore)
    rng, rng_extra = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 1000)
    samples, n_seen = [], 0
    try:
        for t in targets:
            stem, label = t["eval_dataset"], t["target_label"]
            z = np.load(J / f"{stem}_activations.npz")
            y = read_labels(z)[label]
            n_resp = z["n_response_tokens"]
            by_id = {r["id"]: r for r in read_jsonl(J / f"{stem.removesuffix('_flipped')}.jsonl")}
            cell = benchmark_cell(stem)[0]
            for want in (1, 0):
                pool = np.flatnonzero((y == want) & (n_resp <= args.max_response_tokens))
                if len(pool) < args.per_label + args.extra_per_label:
                    raise ValueError(f"{stem}: {len(pool)} rows with label {want} and short responses")
                first = rng.choice(pool, args.per_label, replace=False)
                rest = np.setdiff1d(pool, first)
                extra = rng_extra.choice(rest, args.extra_per_label, replace=False) if args.extra_per_label else []
                for draw, i in [("first", k) for k in first] + [("extra", k) for k in extra]:
                    row = by_id[z["id"][i]]
                    n_seen += 1
                    rec = row_token_scores(row, tokenizer, model, store, ostore, res_layers, refs, cell, args.max_user_tokens)
                    samples.append({"benchmark": stem, "display": t["display"], "cell": cell, "target_label": label, "draw": draw,
                                    "label": int(y[i]), "id": row["id"], **rec})
                    span = rec["tokens"]
                    print(f"  {stem} label {want}: {len(span)} tokens", flush=True)
    finally:
        for h in handles:
            h.remove()
    meta_dets = [{"name": d[0], "file": str(d[1].relative_to(REPO_ROOT)), "probe_id": d[2], "method": d[3], "layer": d[4],
                  "format": d[5], "native_position": d[6], "built_on": d[7]} for d in DETECTORS]
    meta_dets += [{"name": "Contrastive universal (P00)", "probe_id": refs[0]["probe_id"], "layer": refs[0]["layer"], "format": "chat",
                   "native_position": refs[0]["position"], "built_on": "General (baseline)"},
                  {"name": "Contrastive matched-cell pair", "probe_id": "per benchmark cell (validation-selected)", "layer": "per pair",
                   "format": "chat", "native_position": "per pair", "built_on": "the benchmark's own cell"}]
    out = REPO_ROOT / "reports" / "audit_existing_methods" / "token_scores.json"
    write_json(out, {"detectors": meta_dets, "seed": args.seed, "per_label": args.per_label, "extra_per_label": args.extra_per_label,
                     "max_response_tokens": args.max_response_tokens, "samples": samples})
    write_meta(out, [TARGETS, SELECTED], args, check_counts(n_seen, {}, len(samples), "samples"), {"model": MODEL})
    print(f"{len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()
