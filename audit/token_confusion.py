"""Per-detector 2x2 (judge label x detector prediction) on each benchmark, with one per-token example per cell.

Pooled score per row at the detector's own read position (the files analyze.py scored); prediction = the detector's
own decision rule (DIM: score >= 0; logistic: score > 0, as utils.probes.balanced_accuracy), thresholds from source
data. AUROC from these scores is asserted equal to coverage.csv. Example per cell: drawn with a generator seeded by
(seed, detector, benchmark, cell) among rows with at most --max-response-tokens response tokens; if the cell has
none, the row with the shortest response (flagged); an empty cell is reported empty.
Writes reports/audit_existing_methods/token_confusion.json.

Run from the repo root: python -m audit.token_confusion --seed 0 --max-response-tokens 300
"""
import argparse
import csv

import numpy as np

from audit import heads
from audit.analyze import CELL_PAIR, benchmark_cell
from audit.token_scores import DETECTORS, J, MODEL, P, TARGETS, reference_probes, row_token_scores
from utils import probes
from utils.activations import NO_LABEL, act_key, read_labels
from utils.io import REPO_ROOT, check_counts, read_jsonl, write_json, write_meta
from utils.models import load_model_and_tokenizer

NATIVE = REPO_ROOT / "activations" / "audit" / "native"
COVERAGE_KEY = {  # detector name -> (coverage.csv detector id or role)
    "CAA (L13, recipe layer)": "audit_caa_respmean:caa__dim__response_L13",
    "Vennemeyer SyA (chat, L30)": "audit_vennemeyer_math_chat_last_native:syc__dim__last_content_L30@chat",
    "Vennemeyer GA (chat, L30)": "audit_vennemeyer_math_chat_last_native:ga__dim__last_content_L30@chat",
    "Vennemeyer SyPr (chat, L16)": "audit_vennemeyer_math_chat_last_native:pr__dim__last_content_L16@chat",
    "Genadi best head (L12H12)": "audit_genadi_heads:genadi__best_head@chat_double_bos",
    "Pandey LR-27 (their probe)": "audit_pandey_lr27_native:syc__logistic__pandey_faithful_L27@chat_double_bos",
    "Persona Vectors (L15)": "audit_persona:persona__dim__response_L15",
    "Contrastive universal (P00)": "universal",
    "Contrastive matched-cell pair": "matched_cell",
}
NATIVE_KEY = {"Vennemeyer": ("chat", "last_content"), "Pandey": ("chat_double_bos", "pandey_faithful")}


def pooled(name: str, stem: str, z, refs: dict) -> tuple[np.ndarray, str]:
    """(score per row of the repo cache, method) at the detector's own read position."""
    if name.startswith("Contrastive"):
        m = refs[0 if "universal" in name else CELL_PAIR[benchmark_cell(stem)[0]]]
        pr = probes.load_probe(np.load(P / "synthetic_tefo_C_sweep" / m["file"]), m["probe_id"], m["method"])
        return probes.score(m["method"], pr, z[act_key(m["position"], m["layer"])]), m["method"]
    (_, path, pid, method, L, fmt, _, _) = next(d for d in DETECTORS if d[0] == name)
    if method == "head":
        zn = np.load(NATIVE / f"{stem}__chat_double_bos_activations.npz")
        x = heads.split_heads(zn[f"oproj_genadi_answer_mean_L{L:02d}"], 32)[:, 12]
        return probes.score_logistic(probes.load_probe(np.load(path), pid, "logistic"), x), "logistic"
    key = next((v for k, v in NATIVE_KEY.items() if name.startswith(k)), None)
    if key:
        zn = np.load(NATIVE / f"{stem}__{key[0]}_activations.npz")
        if not np.array_equal(zn["id"], z["id"]):
            raise AssertionError(f"{stem}: native rows differ from the repo cache")
        x = zn[f"{key[1]}_L{L:02d}"]
    else:
        x = z[f"response_L{L:02d}"]
    return probes.score(method, probes.load_probe(np.load(path), pid, method), x.astype(np.float64)), method


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-response-tokens", type=int, required=True)
    parser.add_argument("--max-user-tokens", type=int, default=120)
    args = parser.parse_args()

    with open(TARGETS, encoding="utf-8", newline="") as f:
        targets = list(csv.DictReader(f))
    with open(REPO_ROOT / "reports/audit_existing_methods/coverage.csv", encoding="utf-8", newline="") as f:
        cov = {((r["detector"] if r["role"] == "detector" else r["role"]), r["benchmark"]): float(r["auroc"]) for r in csv.DictReader(f)}
    refs = reference_probes()
    names = list(COVERAGE_KEY)
    table, wanted = {}, {}
    for bi, t in enumerate(targets):
        stem, label = t["eval_dataset"], t["target_label"]
        z = np.load(J / f"{stem}_activations.npz")
        y_all = read_labels(z)[label]
        keep = y_all != NO_LABEL
        n_resp = z["n_response_tokens"]
        for di, name in enumerate(names):
            s, method = pooled(name, stem, z, refs)
            auc = probes.auroc(y_all[keep].astype(int), s[keep])
            if not np.isclose(auc, cov[(COVERAGE_KEY[name], stem)], atol=1e-9):
                raise AssertionError(f"{name} on {stem}: AUROC {auc} != coverage.csv {cov[(COVERAGE_KEY[name], stem)]}")
            pred = (s > 0) if method == "logistic" else (s >= 0)
            cells = {}
            for ci, (tl, pl) in enumerate(((1, 1), (1, 0), (0, 1), (0, 0))):
                members = np.flatnonzero(keep & (y_all == tl) & (pred == bool(pl)))
                rec = {"target": tl, "pred": pl, "n": int(len(members))}
                if len(members):
                    short = members[n_resp[members] <= args.max_response_tokens]
                    rng = np.random.default_rng([args.seed, di, bi, ci])
                    i = int(rng.choice(short)) if len(short) else int(members[np.argmin(n_resp[members])])
                    rec |= {"row": f"{stem}|{z['id'][i]}", "pooled_score": float(s[i]), "long_response": not len(short),
                            "n_response_tokens": int(n_resp[i])}
                    wanted.setdefault(rec["row"], {"benchmark": stem, "index": i, "label": int(y_all[i]), "detectors": set()})
                    wanted[rec["row"]]["detectors"].add(name)
                cells[f"t{tl}p{pl}"] = rec
            table.setdefault(name, {})[stem] = {"auroc": auc, "rule": "score > 0" if method == "logistic" else "score >= 0", "cells": cells}
        print(f"{stem}: done", flush=True)

    model, tokenizer = load_model_and_tokenizer(MODEL, padding_side="right")
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model is not on the GPU")
    res_layers = sorted({d[4] for d in DETECTORS if d[3] != "head"} | {m["layer"] for m in refs.values()})
    store, ostore = {}, {}
    handles = heads.register_block_outputs(model, res_layers, store) + heads.register_oproj_inputs(model, [12], ostore)
    rows_out, jsonl = {}, {}
    try:
        for ref, w in wanted.items():
            stem = w["benchmark"]
            if stem not in jsonl:
                jsonl[stem] = {r["id"]: r for r in read_jsonl(J / f"{stem.removesuffix('_flipped')}.jsonl")}
            row = jsonl[stem][ref.split("|", 1)[1]]
            rec = row_token_scores(row, tokenizer, model, store, ostore, res_layers, refs, benchmark_cell(stem)[0], args.max_user_tokens)
            rec["scores"] = {k: v for k, v in rec["scores"].items() if k in w["detectors"]}
            rows_out[ref] = {"benchmark": stem, "id": row["id"], "label": w["label"], **rec}
    finally:
        for h in handles:
            h.remove()
    displays = {t["eval_dataset"]: t["display"] for t in targets}
    out = REPO_ROOT / "reports" / "audit_existing_methods" / "token_confusion.json"
    write_json(out, {"seed": args.seed, "max_response_tokens": args.max_response_tokens, "detectors": names,
                     "benchmarks": [{"id": t["eval_dataset"], "display": t["display"], "cell": benchmark_cell(t["eval_dataset"])[0],
                                     "target_label": t["target_label"]} for t in targets],
                     "table": table, "rows": rows_out, "displays": displays})
    n_cells = sum(len(v) * 4 for v in table.values())
    filled = sum(1 for v in table.values() for b in v.values() for c in b["cells"].values() if c["n"])
    write_meta(out, [TARGETS, REPO_ROOT / "reports/audit_existing_methods/coverage.csv"], args,
               check_counts(n_cells, {"empty_cell": n_cells - filled}, filled, "cells"), {"model": MODEL, "n_rows": len(rows_out)})
    print(f"{filled}/{n_cells} cells with an example; {len(rows_out)} distinct rows -> {out}")


if __name__ == "__main__":
    main()
