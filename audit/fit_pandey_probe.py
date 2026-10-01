"""Pandey's own residual detectors, as sweeps evaluate_probes can score.

lr27: probe_transfer.py / stats/probes.py: unstandardised LogisticRegression(C=1, max_iter=2000) at layer int(0.85 L) = 27
  on the first 100 sycophancy pairs (wrong = 1, correct = 0).
dim19: steering.py: unit mean(wrong) - mean(correct) at layer int(0.6 L) = 19 on the first 100 pairs (fit_dim threshold).
Both read the code-faithful position. Held-out AUROC on pairs 100-199 goes into the manifest.
Writes probes/<model name>/audit_pandey_<lr27|dim19>_native/ (probes npz, manifest.jsonl, split.json).

Run from the repo root: python -m audit.fit_pandey_probe --activations activations/audit/pandey/triviaqa_syc.npz
"""
import argparse
from pathlib import Path

import numpy as np

from audit.fit_directions import directions_from_sums, source_provenance
from audit.package_directions import empty_split
from utils import probes
from utils.io import REPO_ROOT, check_counts, meta_path, read_json, write_json, write_jsonl, write_meta

POSITION = "pandey_faithful"  # name of the native position in get_native_activations files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()
    from sklearn.linear_model import LogisticRegression

    z = np.load(args.activations)
    prov = source_provenance(read_json(meta_path(args.activations)))
    X, y, split = z["X__faithful"], z["labels__syc"].astype(int), z["split"]
    fit, test = split == "fit", split == "test"
    if not np.all(z["cached"]) or (z["group"][fit] >= "pair100").any() or (z["group"][test] < "pair100").any():
        raise AssertionError("expected all rows cached, fit = pairs 0-99, test = pairs 100-199")

    clf = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(X[fit, 27].astype(np.float64), y[fit])
    lr = {"mean": np.zeros(X.shape[2]), "scale": np.ones(X.shape[2]), "coef": clf.coef_[0].astype(np.float64),
          "intercept": np.float64(clf.intercept_[0])}
    lr["direction"] = lr["coef"] / np.linalg.norm(lr["coef"])
    d, thr, _, _ = directions_from_sums(z, "faithful", "syc")
    dim = {"direction": d[19], "threshold": np.float64(thr[19])}
    sweeps = {
        "lr27": ("logistic", 27, lr, probes.score_logistic(lr, X[test, 27].astype(np.float64)),
                 {"recipe": "probe_transfer.py: LogisticRegression(C=1, max_iter=2000), no scaling", "n_iter": int(clf.n_iter_[0]),
                  "converged": bool(clf.n_iter_[0] < args.max_iter)}),
        "dim19": ("dim", 19, dim, X[test, 19] @ d[19], {"recipe": "steering.py: mean(wrong) - mean(correct), layer int(0.6 * 32)"}),
    }
    for name, (method, layer, probe, s_test, extra) in sweeps.items():
        out_dir = REPO_ROOT / "probes" / prov["model"].split("/")[-1] / f"audit_pandey_{name}_native"
        if out_dir.exists():
            raise FileExistsError(out_dir)
        out_dir.mkdir(parents=True)
        pid = f"syc__{probes.probe_key(method, POSITION, layer)}"
        probes.save_probes(out_dir / "syc.probes.npz", {pid: probe})
        auc = probes.auroc(y[test], s_test)
        manifest = [{"probe_id": pid, "file": "syc.probes.npz", "probe_type": "audit", "method": method, "position": POSITION,
                     "layer": layer, "model": prov["model"], "C": args.C if method == "logistic" else None,
                     "cell": "Position-Verifiable / Explicit", "pair_type": "audit", "audit_method": "pandey",
                     "audit_unit": f"syc_{name}", "layer_rule": "recipe",
                     "native_position": "code-faithful index (prompt tokens != <|eot_id|>).sum() - 1 with double BOS",
                     "position_matches_native": True, "source_repo": prov["source_repo"], "source_commit": prov["source_commit"],
                     "source_val_auroc": None, "source_heldout_auroc": auc, "source_n_heldout": int(test.sum()), **extra}]
        write_jsonl(out_dir / "manifest.jsonl", manifest)
        write_json(out_dir / "split.json", empty_split())
        counts = check_counts(len(y), {}, len(y), "rows")
        for path in (out_dir / "syc.probes.npz", out_dir / "manifest.jsonl", out_dir / "split.json"):
            write_meta(path, [args.activations], args, counts, {**prov, **extra})
        print(f"{name}: held-out AUROC {auc} {extra}")


if __name__ == "__main__":
    main()
