"""Package fit_directions output as a sweep that the unchanged probe_training.evaluate_probes can score.

Writes probes/<model name>/audit_<name>/: <unit>.probes.npz (method dim: unit direction + fit_dim threshold),
manifest.jsonl (one row per unit x layer x position) and split.json (no training rows; group field absent from
evaluation files). Layers: the one fit_directions selected per unit, plus --extra-layers (e.g. a recipe layer).

Run from the repo root: python -m audit.package_directions --directions ... --table ... --name caa --cell ... --native-position ...
"""
import argparse
from pathlib import Path

import numpy as np

from utils import probes
from utils.activations import POSITIONS
from utils.io import REPO_ROOT, check_counts, meta_path, read_json, read_jsonl, write_json, write_jsonl, write_meta

NO_GROUP_FIELD = "audit_no_group_field"


def build_sweep(directions: dict, table: list[dict], units: list[str], extra_layers: list[int], positions: list[str],
                info: dict) -> tuple[dict[str, dict], list[dict]]:
    """({unit: {probe_id: probe}}, manifest rows). `info` holds model, audit_method, native_position, source_repo, ..."""
    by_key = {(r["unit"], r["layer"]): r for r in table}
    probe_sets, manifest = {}, []
    for unit in units:
        selected = [r["layer"] for r in table if r["unit"] == unit and r["selected"]]
        if len(selected) != 1:
            raise ValueError(f"{unit}: {len(selected)} selected layers in the table")
        layers = sorted({selected[0], *extra_layers})
        d, thr = directions[f"{unit}__direction"], directions[f"{unit}__threshold"]
        probe_sets[unit] = {}
        for L in layers:
            rec = by_key[(unit, L)]
            if not np.isclose(np.linalg.norm(d[L]), 1.0):
                raise AssertionError(f"{unit} L{L}: direction is not unit norm")
            for pos in positions:
                pid = f"{unit}__{probes.probe_key('dim', pos, L)}"
                probe_sets[unit][pid] = {"direction": d[L].astype(np.float64), "threshold": np.float64(thr[L])}
                manifest.append({
                    "probe_id": pid, "file": f"{unit}.probes.npz", "probe_type": "audit", "method": "dim",
                    "position": pos, "layer": L, "model": info["model"], "C": None, "cell": info["cell"][unit],
                    "pair_type": "audit", "audit_method": info["audit_method"], "audit_unit": unit,
                    "layer_rule": rec["select_rule"] if L == selected[0] else "extra",
                    "native_position": info["native_position"],
                    "position_matches_native": pos == info.get("native_matches_cached_position"),
                    "source_repo": info["source_repo"], "source_commit": info["source_commit"],
                    "source_val_auroc": rec["val_auroc"], "source_heldout_auroc": rec["test_auroc"],
                    "source_n_heldout": rec["n_test"],
                })
    return probe_sets, manifest


def empty_split() -> dict:
    return {"train": [], "val": [], "test": [], "train_ids": [], "val_ids": [], "group_field": NO_GROUP_FIELD,
            "note": "audit detectors are fit on external source data; no benchmark row is used for fitting"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directions", type=Path, required=True, help="fit_directions .npz")
    parser.add_argument("--table", type=Path, required=True, help="fit_directions layerwise .jsonl")
    parser.add_argument("--name", required=True, help="Sweep directory is audit_<name>.")
    parser.add_argument("--units", nargs="+", required=True)
    parser.add_argument("--extra-layers", type=int, nargs="*", default=[])
    parser.add_argument("--positions", nargs="+", default=list(POSITIONS))
    parser.add_argument("--cell", nargs="+", required=True, help="<unit>=<Ye et al. cell the unit was built on>, one per unit.")
    parser.add_argument("--native-position", required=True, help="Where the method reads activations (text).")
    parser.add_argument("--native-matches", choices=list(POSITIONS), help="Cached position equal to the native one, if any.")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "probes")
    args = parser.parse_args()

    dmeta = read_json(meta_path(args.directions))
    cells = dict(c.split("=", 1) for c in args.cell)
    if sorted(cells) != sorted(args.units):
        raise ValueError(f"--cell units {sorted(cells)} differ from --units {sorted(args.units)}")
    info = {"model": dmeta["model"], "audit_method": dmeta["method"], "cell": cells,
            "native_position": args.native_position, "native_matches_cached_position": args.native_matches,
            "source_repo": dmeta["source_repo"], "source_commit": dmeta["source_commit"]}
    table = read_jsonl(args.table)
    directions = dict(np.load(args.directions))
    probe_sets, manifest = build_sweep(directions, table, args.units, args.extra_layers, args.positions, info)

    out_dir = args.out_root / dmeta["model"].split("/")[-1] / f"audit_{args.name}"
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    counts = check_counts(len(manifest), {}, len(manifest), "manifest")
    for unit, ps in probe_sets.items():
        path = out_dir / f"{unit}.probes.npz"
        probes.save_probes(path, ps)
        write_meta(path, [args.directions, args.table], args, {"n_probes": len(ps)}, info)
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    write_meta(out_dir / "manifest.jsonl", [args.directions, args.table], args, counts, info)
    write_json(out_dir / "split.json", empty_split())
    write_meta(out_dir / "split.json", [args.directions], args, counts, info)
    print(f"{len(manifest)} probes -> {out_dir}")


if __name__ == "__main__":
    main()
