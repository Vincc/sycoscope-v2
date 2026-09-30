"""Score every contrastive probe in a sweep on every pair's validation rows (in-distribution transfer).

For each probe (source pair x method x position x layer x C) and each target pair, AUROC and paired win rate are
computed on the target pair's validation rows, using activations at the probe's own position and layer. All pairs
share one prompt split, so no target row comes from a prompt the source probe was trained on. The diagonal
(source == target) must reproduce the manifest's val_auroc.

Writes <out-dir>/transfer_val.csv, one row per probe x target pair.

Run from the repo root: python -m probe_analysis.in_distribution_transfer.compute_transfer --activations ... --sweep-dir ... --out-dir ...
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from utils import probes
from utils.activations import act_key
from utils.io import check_counts, read_json, sha256, write_meta

FIELDS = ("probe_id", "source_pair", "source_cell", "source_type", "method", "position", "layer", "C",
          "target_pair", "target_cell", "target_type", "n", "n_pos", "auroc", "paired_win_rate", "n_pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", type=Path, required=True, help="Contrastive *_activations.npz the sweep was trained on.")
    parser.add_argument("--sweep-dir", type=Path, required=True, help="probes/<model>/<sweep> with manifest.jsonl and split.json.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path, split_path = args.sweep_dir / "manifest.jsonl", args.sweep_dir / "split.json"
    manifest = [json.loads(line) for line in manifest_path.open(encoding="utf-8")]
    split = read_json(split_path)
    sweep_meta = read_json(args.sweep_dir / "manifest.jsonl.meta.json")  # this sweep keeps meta beside outputs
    if len({m["probe_id"] for m in manifest}) != len(manifest):
        raise ValueError("manifest has duplicate probe IDs")
    if {m["probe_type"] for m in manifest} != {"contrastive"}:
        raise ValueError("in-distribution transfer is defined for contrastive probes only")
    if split["activations"] != args.activations.as_posix():
        raise ValueError(f"split was built from {split['activations']}, not {args.activations}")
    if sha256(args.activations) != sweep_meta["inputs"][args.activations.as_posix()]:
        raise ValueError(f"{args.activations} differs from the file the sweep was trained on")
    act_meta = read_json(args.activations.with_name(args.activations.name + ".meta.json"))
    if {m["model"] for m in manifest} != {act_meta["model"]}:
        raise ValueError("activation model does not match probe model")

    z = np.load(args.activations)
    ids, prompt_ids = z["id"], z["prompt_id"]
    labels, pair_index, cells, pair_types = z["label"].astype(int), z["pair_index"].astype(int), z["cell"], z["pair_type"]
    val = np.isin(prompt_ids, split["val"])
    if set(ids[val].tolist()) != set(split["val_ids"]):
        raise AssertionError("validation rows differ from split.json val_ids")
    if set(ids[val].tolist()) & set(split["train_ids"]):
        raise AssertionError("a training ID is among the validation rows")
    val_rows = np.flatnonzero(val)
    counts = check_counts(len(ids), Counter({"not_val_prompt": int((~val).sum())}), len(val_rows), "validation rows")

    pairs = sorted({int(m["pair_index"]) for m in manifest})
    targets = {}
    for j in pairs:
        local = np.flatnonzero(pair_index[val_rows] == j)  # indices into the val-row activation slice
        y = labels[val_rows][local]
        probes.check_binary(y, f"pair{j:02d} val")
        targets[j] = {"local": local, "y": y, "groups": prompt_ids[val_rows][local].tolist(),
                      "cell": str(cells[val_rows][local][0]), "type": str(pair_types[val_rows][local][0])}
    if sum(len(t["local"]) for t in targets.values()) != len(val_rows):
        raise AssertionError("validation rows belong to pairs outside the manifest")

    by_site = {}
    for m in manifest:
        by_site.setdefault((m["position"], int(m["layer"])), []).append(m)
    probe_files = {f: np.load(args.sweep_dir / f) for f in sorted({m["file"] for m in manifest})}

    out_rows, max_diag_err = [], 0.0
    for (position, layer), site in sorted(by_site.items()):
        X = z[act_key(position, layer)][val_rows]
        if X.shape[0] != len(val_rows):
            raise AssertionError("activation array and row fields differ in length")
        for m in site:
            probe = probes.load_probe(probe_files[m["file"]], m["probe_id"], m["method"])
            s = probes.score(m["method"], probe, X)
            for j, t in targets.items():
                st = s[t["local"]]
                auc = probes.auroc(t["y"], st)
                win, n_pairs = probes.paired_win_rate(st, t["y"], t["groups"])
                if j == m["pair_index"]:
                    max_diag_err = max(max_diag_err, abs(auc - m["val_auroc"]))
                    if abs(auc - m["val_auroc"]) > 1e-9:
                        raise AssertionError(f"{m['probe_id']}: diagonal AUROC {auc} != manifest val_auroc {m['val_auroc']}")
                out_rows.append({
                    "probe_id": m["probe_id"], "source_pair": m["pair_index"], "source_cell": m["cell"],
                    "source_type": m["pair_type"], "method": m["method"], "position": position, "layer": layer,
                    "C": "" if m["C"] is None else m["C"], "target_pair": j, "target_cell": t["cell"],
                    "target_type": t["type"], "n": len(t["y"]), "n_pos": int(t["y"].sum()),
                    "auroc": auc, "paired_win_rate": win, "n_pairs": n_pairs,
                })
        print(f"{position} L{layer:02d}: {len(site)} probes x {len(targets)} targets", flush=True)

    if len(out_rows) != len(manifest) * len(targets):
        raise AssertionError("output is not a complete probe x target grid")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "transfer_val.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    write_meta(out, [args.activations, manifest_path, split_path, *(args.sweep_dir / f for f in probe_files)], args, counts,
               {"model": act_meta["model"], "n_probes": len(manifest), "n_targets": len(targets), "n_rows": len(out_rows),
                "eval_split": "val", "diagonal_max_abs_diff_vs_manifest_val_auroc": max_diag_err})
    print(f"{len(out_rows)} rows -> {out}; diagonal max |diff| vs manifest {max_diag_err:.2e}")


if __name__ == "__main__":
    main()
