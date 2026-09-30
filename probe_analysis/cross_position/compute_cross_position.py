"""Score every contrastive probe at every token position of the same layer (cross-position transfer).

Each probe (trained at one of last_prompt, first5, response) is applied unchanged to the activations of all three
positions at its own layer, on:
  - each judged OOD target in --targets (target label only; rows excluded as in probe_training.evaluate_probes), and
  - its own pair's contrastive test split (prompts in split.json "test"; never used for fitting or selection).
Only AUROC is reported: the decision threshold is calibrated to the training position. Rows where the eval position
equals the training position must reproduce the stored AUROCs (eval/*.jsonl and manifest test_auroc).

Writes <out-dir>/cross_position.csv, one row per probe x eval position x dataset.

Run from the repo root: python -m probe_analysis.cross_position.compute_cross_position --sweep-dir ... --train-activations ... --targets ... --eval-dir ... --out-dir ...
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from utils import probes
from utils.activations import NO_LABEL, act_key, read_labels
from utils.io import check_counts, meta_path, read_json, read_jsonl, sha256, write_meta

POSITIONS = ("last_prompt", "first5", "response")
IN_DIST = "contrastive_test"
FIELDS = ("probe_id", "source_pair", "cell", "pair_type", "method", "C", "train_position", "layer", "eval_position",
          "dataset", "family", "display", "label", "n", "n_pos", "auroc")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def linear_form(method: str, probe: dict) -> tuple[np.ndarray, float]:
    """(w, b) with probes.score(method, probe, X) == X @ w + b."""
    if method == "logistic":
        w = probe["coef"] / probe["scale"]
        return w, float(probe["intercept"] - (probe["mean"] / probe["scale"]) @ probe["coef"])
    if method == "dim":
        return probe["direction"], float(-probe["threshold"])
    raise ValueError(f"unknown method {method!r}")


def score_all(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Scores of every probe (columns of W) on rows of X, in float64."""
    return X.astype(np.float64) @ W + b


def probe_forms(manifest: list[dict], sweep_dir: Path):
    """Layers, manifest rows per layer, and per layer the stacked (W, b) of those probes in the same order."""
    layers = sorted({int(m["layer"]) for m in manifest})
    by_layer = {layer: [m for m in manifest if int(m["layer"]) == layer] for layer in layers}
    probe_files = {f: np.load(sweep_dir / f) for f in sorted({m["file"] for m in manifest})}
    forms = {}
    for layer, ms in by_layer.items():
        wb = [linear_form(m["method"], probes.load_probe(probe_files[m["file"]], m["probe_id"], m["method"])) for m in ms]
        forms[layer] = (np.stack([w for w, _ in wb], axis=1), np.array([b for _, b in wb]))
    return layers, by_layer, forms, probe_files


def ood_rows(path: Path, t: dict, split: dict, stored_rows: list[dict], model: str):
    """Open a judged OOD npz and keep the rows probe_training.evaluate_probes scored. Returns (z, keep mask, y, counts)."""
    if {r["eval_sha256"] for r in stored_rows} != {sha256(path)}:
        raise ValueError(f"{path} is not the file the stored evaluation was scored on")
    if read_json(meta_path(path))["model"] != model:
        raise ValueError(f"{path}: activation model does not match probe model")
    z = np.load(path)
    ids = z["id"]
    in_train = np.isin(ids, split["train_ids"] + split["val_ids"])
    if split["group_field"] in z.files:
        in_train |= np.isin(z[split["group_field"]], split["train"] + split["val"])
    y_all = read_labels(z)[t["target_label"]].astype(int)
    none = (y_all == NO_LABEL) & ~in_train
    keep = ~in_train & ~none
    counts = check_counts(len(ids), Counter({"in_probe_train": int(in_train.sum()), "label_none": int(none.sum())}),
                          int(keep.sum()), t["eval_dataset"])
    y = y_all[keep]
    if len(y) != int(t["n"]) or {(r["n"], r["n_pos"]) for r in stored_rows} != {(len(y), int(y.sum()))}:
        raise AssertionError(f"{t['eval_dataset']}: {len(y)} kept rows differ from targets.csv n={t['n']} or stored counts")
    probes.check_binary(y, t["eval_dataset"])
    return z, keep, y, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, required=True, help="probes/<model>/<sweep> with manifest.jsonl, split.json, eval/.")
    parser.add_argument("--train-activations", type=Path, required=True, help="Contrastive *_activations.npz the sweep was trained on.")
    parser.add_argument("--targets", type=Path, required=True, help="CSV with eval_dataset, family, display, target_label, n.")
    parser.add_argument("--eval-dir", type=Path, required=True, help="Directory holding <eval_dataset>_activations.npz.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path, split_path = args.sweep_dir / "manifest.jsonl", args.sweep_dir / "split.json"
    manifest, split, targets = read_jsonl(manifest_path), read_json(split_path), read_csv(args.targets)
    sweep_meta = read_json(args.sweep_dir / "manifest.jsonl.meta.json")  # this sweep keeps meta beside outputs
    if len({m["probe_id"] for m in manifest}) != len(manifest):
        raise ValueError("manifest has duplicate probe IDs")
    if {m["probe_type"] for m in manifest} != {"contrastive"} or {m["position"] for m in manifest} != set(POSITIONS):
        raise ValueError("expected contrastive probes at exactly the three positions")
    models = {m["model"] for m in manifest}
    if len(models) != 1:
        raise ValueError(f"manifest mixes models: {models}")
    model = next(iter(models))
    if split["activations"] != args.train_activations.as_posix():
        raise ValueError(f"split was built from {split['activations']}, not {args.train_activations}")
    if sha256(args.train_activations) != sweep_meta["inputs"][args.train_activations.as_posix()]:
        raise ValueError(f"{args.train_activations} differs from the file the sweep was trained on")
    if read_json(args.train_activations.with_name(args.train_activations.name + ".meta.json"))["model"] != model:  # meta beside this file
        raise ValueError("training activation model does not match probe model")

    layers, by_layer, forms, probe_files = probe_forms(manifest, args.sweep_dir)

    # One output row per probe x eval position x dataset; same-position rows are checked against stored AUROCs.
    out_rows, counts, inputs, max_diag_err = [], {}, [args.train_activations, manifest_path, split_path, args.targets], 0.0

    def emit(ms, scores, y, eval_position, dataset, family, display, label, stored):
        nonlocal max_diag_err
        for k, m in enumerate(ms):
            auc = probes.auroc(y, scores[:, k])
            if auc is None:
                raise ValueError(f"{dataset}: undefined AUROC (one class)")
            if eval_position == m["position"]:
                err = abs(auc - stored[m["probe_id"]])
                max_diag_err = max(max_diag_err, err)
                if err > 1e-9:
                    raise AssertionError(f"{m['probe_id']} on {dataset}: AUROC {auc} != stored {stored[m['probe_id']]}")
            out_rows.append({
                "probe_id": m["probe_id"], "source_pair": m["pair_index"], "cell": m["cell"], "pair_type": m["pair_type"],
                "method": m["method"], "C": "" if m["C"] is None else m["C"], "train_position": m["position"],
                "layer": m["layer"], "eval_position": eval_position, "dataset": dataset, "family": family,
                "display": display, "label": label, "n": len(y), "n_pos": int(y.sum()), "auroc": auc,
            })

    # OOD judged targets.
    for t in targets:
        path = args.eval_dir / f"{t['eval_dataset']}_activations.npz"
        stored_path = args.sweep_dir / "eval" / f"{t['eval_dataset']}.jsonl"
        inputs += [path, stored_path]
        stored_rows = [r for r in read_jsonl(stored_path) if r["eval_label"] == t["target_label"]]
        if {r["probe_id"] for r in stored_rows} != {m["probe_id"] for m in manifest} or len(stored_rows) != len(manifest):
            raise ValueError(f"{stored_path}: target-label rows do not match the manifest one-to-one")
        z, keep, y, counts[t["eval_dataset"]] = ood_rows(path, t, split, stored_rows, model)
        stored = {r["probe_id"]: r["auroc"] for r in stored_rows}
        for layer in layers:
            W, b = forms[layer]
            for pos in POSITIONS:
                X = z[act_key(pos, layer)]
                if X.shape[0] != len(keep):
                    raise AssertionError(f"{path} {act_key(pos, layer)}: {X.shape[0]} rows, id has {len(keep)}")
                emit(by_layer[layer], score_all(X[keep], W, b), y, pos, t["eval_dataset"], t["family"], t["display"],
                     t["target_label"], stored)
        print(f"{t['eval_dataset']}: n={len(y)}", flush=True)

    # In-distribution: each probe on its own pair's test rows.
    z = np.load(args.train_activations)
    ids, prompt_ids, labels, pair_index = z["id"], z["prompt_id"], z["label"].astype(int), z["pair_index"].astype(int)
    test = np.isin(prompt_ids, split["test"])
    if set(ids[test].tolist()) & set(split["train_ids"] + split["val_ids"]):
        raise AssertionError("a training or validation ID is among the test rows")
    counts[IN_DIST] = check_counts(len(ids), Counter({"not_test_prompt": int((~test).sum())}), int(test.sum()), IN_DIST)
    test_rows = np.flatnonzero(test)
    stored = {m["probe_id"]: m["test_auroc"] for m in manifest}
    for layer in layers:
        W, b = forms[layer]
        ms = by_layer[layer]
        for pos in POSITIONS:
            X = z[act_key(pos, layer)][test_rows]
            s = score_all(X, W, b)
            for j in sorted({int(m["pair_index"]) for m in ms}):
                cols = [k for k, m in enumerate(ms) if int(m["pair_index"]) == j]
                local = pair_index[test_rows] == j
                y = labels[test_rows][local]
                if {(ms[k]["n_test"], ms[k]["n_test_pos"]) for k in cols} != {(len(y), int(y.sum()))}:
                    raise AssertionError(f"pair{j:02d}: test rows differ from manifest n_test")
                emit([ms[k] for k in cols], s[np.ix_(local, cols)], y, pos, IN_DIST, "In-distribution",
                     "Contrastive test (own pair)", "system_prompt", stored)
    print(f"{IN_DIST}: {len(test_rows)} rows", flush=True)

    expected = len(manifest) * len(POSITIONS) * (len(targets) + 1)
    if len(out_rows) != expected:
        raise AssertionError(f"{len(out_rows)} rows, expected {expected}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "cross_position.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    write_meta(out, [*inputs, *(args.sweep_dir / f for f in probe_files)], args,
               {"n_in": len(manifest), "excluded": {}, "n_out": len(out_rows), "datasets": counts},
               {"model": model, "n_probes": len(manifest), "n_datasets": len(targets) + 1, "eval_positions": list(POSITIONS),
                "probe_application": "stored probe unchanged (logistic scaler from the training position)",
                "diagonal_max_abs_diff_vs_stored_auroc": max_diag_err})
    print(f"{len(out_rows)} rows -> {out}; diagonal max |diff| vs stored {max_diag_err:.2e}")


if __name__ == "__main__":
    main()
