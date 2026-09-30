"""Score head-level audit detectors on a get_head_activations npz; same row schema as evaluate_probes.

AUROC per detector and label; balanced_accuracy is null (the detectors' thresholds come from another distribution
and the committed comparison uses AUROC). Rows with label None are excluded. Writes <detector dir>/eval/<stem>.jsonl.

Run from the repo root: python -m audit.evaluate_heads --detector-dir probes/.../audit_genadi_heads --activations ...
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from audit.fit_heads import N_HEADS
from utils import probes
from utils.activations import NO_LABEL, read_labels
from utils.io import check_counts, meta_path, read_json, read_jsonl, sha256, write_jsonl, write_meta


def features(z, pooling: str, head_list) -> np.ndarray:
    parts = []
    for L, h in head_list:
        x = z[f"{pooling}_L{L:02d}"]
        parts.append(x.reshape(len(x), N_HEADS, -1)[:, h].astype(np.float32))
    return np.concatenate(parts, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--pooling", required=True, help="Pooling of the head cache to score on.")
    args = parser.parse_args()

    manifest = read_jsonl(args.detector_dir / "manifest.jsonl")
    act_meta = read_json(meta_path(args.activations))
    if {m["model"] for m in manifest} != {act_meta["model"]}:
        raise ValueError("detectors and activations come from different models")
    stem = args.activations.name.removesuffix("_heads.npz")
    out_path = args.detector_dir / "eval" / f"{stem}__{args.pooling}.jsonl"
    if out_path.exists():
        raise FileExistsError(out_path)
    z = np.load(args.activations)
    dets = np.load(args.detector_dir / "detectors.npz")
    labels = read_labels(z)
    if not labels and "label" in z:
        labels = {"label": z["label"].astype(int)}
    eval_sha = sha256(args.activations)
    benchmarks = sorted(set(z["benchmark"].tolist())) if "benchmark" in z else None
    rows, counts = [], {}
    for name, y_all in labels.items():
        none = y_all == NO_LABEL
        counts[name] = check_counts(len(y_all), Counter({"label_none": int(none.sum())}), int((~none).sum()), name)
    for m in manifest:
        det = probes.load_probe(dets, m["probe_id"], "logistic")
        s_all = probes.score_logistic(det, features(z, args.pooling, m["heads"]))
        for name, y_all in labels.items():
            keep = y_all != NO_LABEL
            y = y_all[keep].astype(int)
            rows.append({"probe_id": m["probe_id"], "file": "detectors.npz", "probe_type": m["probe_type"],
                         "method": m["method"], "position": args.pooling, "layer": m["layers"], "heads": m["heads"],
                         "cell": m["cell"], "eval_activations": args.activations.as_posix(), "eval_sha256": eval_sha,
                         "eval_benchmarks": benchmarks, "eval_label": name, "n": int(keep.sum()),
                         "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
                         "n_excluded_none": counts[name]["excluded"]["label_none"], "n_excluded_train": 0,
                         "auroc": probes.auroc(y, s_all[keep]), "balanced_accuracy": None})
    write_jsonl(out_path, rows)
    write_meta(out_path, [args.activations, args.detector_dir / "manifest.jsonl"], args, counts,
               {"model": act_meta["model"], "eval_sha256": eval_sha})
    print(f"{len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
