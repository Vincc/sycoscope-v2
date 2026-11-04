"""Project ELEPHANT positive and negative class means into Assistant Axis persona space at one block.

Persona space is a PCA of the role vectors in --role-vectors (assistant-axis pipeline files <role>.pt holding
{"vector": (blocks, hidden), "type", "role"}): the "pos_3" role vectors are centred on their mean and fitted, and
default.pt ("mean") is the default Assistant vector, projected but not fitted. PC1 is signed toward the Assistant Axis,
default minus the role mean. Units and labels are those of interp.generalization_matrix; each unit gives the mean
activation of its label-1 rows and of its label-0 rows. A pooled point averages one metric's per-source class means
with equal weight. Differences between sources' mean activations give a baseline for the label differences.
--reference-vectors, an independent run of the same pipeline, is fitted the same way to check the PCs.
Writes <output> (JSON) and its meta file.

Run from the repo root: python -m interp.persona_space --role-vectors ... --judged ... --cache-dir ... --output ...
"""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from interp.cache_activations import load
from interp.generalization_matrix import LAYER_CONVENTION, position_dir, unit_rows
from utils.io import check_counts, read_json, read_jsonl, write_json, write_meta

POOLED_SOURCES = ("AITA-YTA", "OEQ", "SS")
METRICS = ("validation", "indirectness", "framing")
N_TOP_ROLES = 5


def read_vectors(directory: Path, index: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Role names, (roles, hidden) role vectors and the default vector, from row `index` of each file, as float64."""
    roles, rows, default = [], [], None
    for path in sorted(directory.glob("*.pt")):
        d = torch.load(path, map_location="cpu", weights_only=True)
        if d["role"] != path.stem:
            raise ValueError(f"{path}: role {d['role']!r}")
        v = d["vector"][index].double().numpy()
        if d["type"] == "mean" and d["role"] == "default":
            default = v
        elif d["type"] == "pos_3":
            roles.append(d["role"])
            rows.append(v)
        else:
            raise ValueError(f"{path}: type {d['type']!r}")
    if default is None or not roles:
        raise ValueError(f"{directory}: needs default.pt and role vectors")
    return roles, np.stack(rows), default


def fit_space(R: np.ndarray, default: np.ndarray, n_pcs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Role mean, top n_pcs unit PCs (PC1 signed toward default - role mean), variance ratio and all PCs."""
    mean = R.mean(0)
    pca = PCA().fit(R - mean)
    pcs = pca.components_[:n_pcs].copy()
    if pcs[0] @ (default - mean) < 0:
        pcs[0] *= -1
    return mean, pcs, pca.explained_variance_ratio_, pca.components_


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role-vectors", type=Path, required=True)
    parser.add_argument("--axis", type=Path, required=True, help="The run's axis.pt, rebuilt here as a check.")
    parser.add_argument("--reference-vectors", type=Path, required=True)
    parser.add_argument("--reference-index", type=int, required=True, help="Row of the reference vectors at --block.")
    parser.add_argument("--model", required=True, help="Model the role vectors were taken from.")
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--n-pcs", type=int, required=True)
    parser.add_argument("--judged", type=Path, nargs="+", required=True)
    parser.add_argument("--judge-name", required=True, help="Label suffix, e.g. sonnet55.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--position", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # persona space, the axis check, and the same fit on the reference run
    roles, R, default = read_vectors(args.role_vectors, args.block)
    ref_roles, R_ref, default_ref = read_vectors(args.reference_vectors, args.reference_index)
    if roles != ref_roles:
        raise ValueError("main and reference runs have different roles")
    mean, pcs, var_ratio, all_pcs = fit_space(R, default, args.n_pcs)
    ref_mean, ref_pcs, ref_var_ratio, _ = fit_space(R_ref, default_ref, args.n_pcs)
    ref_pcs *= np.sign(np.sum(ref_pcs * pcs, axis=1))[:, None]  # sign each reference PC like the main run's
    axis = default - mean
    saved_axis = torch.load(args.axis, map_location="cpu", weights_only=True)[args.block].double().numpy()
    axis_check_cos = float(axis @ saved_axis / np.linalg.norm(axis) / np.linalg.norm(saved_axis))
    if axis_check_cos < 0.9999:  # the file is bf16
        raise AssertionError(f"default - role mean differs from {args.axis}: cos {axis_check_cos}")
    if var_ratio[len(roles) - 1] > 1e-10:
        raise AssertionError("centred role vectors should span len(roles) - 1 dimensions")
    span = all_pcs[: len(roles) - 1]

    def project(v):
        return (v - mean) @ pcs.T

    # units and their cached activations
    units, excluded = {}, {}
    for path in args.judged:
        file_units, file_excluded = unit_rows(read_jsonl(path), args.judge_name)
        for name, members in file_units.items():
            if name in units:
                raise ValueError(f"unit {name} appears in two judged files")
            units[name], excluded[name] = members, file_excluded[name]
    chunks, row_of, cache_parts = [], {}, {}
    for path in args.judged:
        pdir = position_dir(args.cache_dir, path, args.position)
        for meta in map(read_json, sorted((pdir / "meta").glob("part-*.meta.json"))):
            if meta["model"] != args.model or meta["layer_convention"] != LAYER_CONVENTION:
                raise ValueError(f"{pdir}: {meta['model']}, {meta['layer_convention']!r}")
            cache_parts[(pdir / meta["output"]).as_posix()] = meta["file_sha256"]
        ids, acts = load(pdir, [args.block])
        for i in ids:
            if i in row_of:
                raise ValueError(f"{i} is cached twice")
            row_of[i] = len(row_of)
        chunks.append(acts[args.block])
    A = np.concatenate(chunks)

    # class means: per unit, then pooled per metric over sources with equal weight
    class_mean, counts = {}, {}
    for name, members in sorted(units.items()):
        y = np.array([v for _, v in members])
        idx = np.array([row_of[i] for i, _ in members])
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        if min(n_pos, n_neg) == 0:
            raise ValueError(f"{name}: needs both classes, has {n_pos} positive and {n_neg} negative")
        counts[name] = {**check_counts(len(members) + sum(excluded[name].values()), excluded[name], len(members), name),
                        "n_pos": n_pos, "n_neg": n_neg}
        for label in (1, 0):
            class_mean[name, label] = A[idx[y == label]].astype(np.float64).mean(0)
    for metric in METRICS:
        for label in (1, 0):
            class_mean[f"pooled/{metric}", label] = np.mean(
                [class_mean[f"{s}/{metric}", label] for s in POOLED_SOURCES], axis=0)

    points = [{"kind": "role", "name": r, "pc": project(v), "pc_reference": (v_ref - ref_mean) @ ref_pcs.T}
              for r, v, v_ref in zip(roles, R, R_ref)]
    points.append({"kind": "default", "name": "default", "pc": project(default),
                   "pc_reference": (default_ref - ref_mean) @ ref_pcs.T})
    for (name, label), v in class_mean.items():
        points.append({"kind": "pooled" if name.startswith("pooled/") else "unit", "name": name, "label": label,
                       "pc": project(v), "pc_reference": (v - ref_mean) @ ref_pcs.T})

    # differences: positive - negative mean of each unit, and source - source mean as a baseline (rows of all its units)
    diffs = {name: class_mean[name, 1] - class_mean[name, 0] for name in sorted({n for n, _ in class_mean})}
    source_ids = defaultdict(set)
    for name, members in units.items():
        source_ids[name.split("/")[0]].update(i for i, _ in members)
    source_mean = {s: A[sorted(row_of[i] for i in source_ids[s])].astype(np.float64).mean(0) for s in POOLED_SOURCES}
    for a, b in combinations(POOLED_SOURCES, 2):
        diffs[f"source/{a} - {b}"] = source_mean[a] - source_mean[b]

    # where each difference points among the PCs and the roles
    differences = []
    role_dirs = (R - mean) / np.linalg.norm(R - mean, axis=1, keepdims=True)
    for name, d in diffs.items():
        cos_roles = role_dirs @ d / np.linalg.norm(d)
        order = np.argsort(cos_roles)
        differences.append({
            "name": name, "norm": float(np.linalg.norm(d)), "pc": d @ pcs.T,
            "share_in_top_pcs": [float(np.sum((pcs[:k] @ d) ** 2) / (d @ d)) for k in range(1, args.n_pcs + 1)],
            "share_in_role_span": float(np.sum((span @ d) ** 2) / (d @ d)),
            "cos_axis": float(d @ axis / np.linalg.norm(d) / np.linalg.norm(axis)),
            "toward_roles": [[roles[i], float(cos_roles[i])] for i in order[::-1][:N_TOP_ROLES]],
            "away_roles": [[roles[i], float(cos_roles[i])] for i in order[:N_TOP_ROLES]],
        })

    # every labelled row, for the row cloud
    row_labels = defaultdict(dict)
    for name, members in units.items():
        for i, y in members:
            row_labels[i][name] = y
    row_ids = sorted(row_labels)
    row_pcs = project(A[[row_of[i] for i in row_ids]].astype(np.float64))

    rnd = lambda a: [round(float(x), 5) for x in a]
    out = {
        "model": args.model, "block": args.block, "position": args.position, "n_roles": len(roles),
        "hidden": int(R.shape[1]),
        "variance_ratio": rnd(var_ratio[:10]), "variance_ratio_reference": rnd(ref_var_ratio[:10]),
        "n_pcs_for_70pct": int(np.argmax(np.cumsum(var_ratio) >= 0.70) + 1),
        "pc_cos_reference": rnd(np.sum(pcs * ref_pcs, axis=1)),
        "pc1_cos_axis": float(pcs[0] @ axis / np.linalg.norm(axis)),
        "axis_check_cos": axis_check_cos, "axis_norm": float(np.linalg.norm(axis)),
        "points": [{**p, "pc": rnd(p["pc"]), "pc_reference": rnd(p["pc_reference"])} for p in points],
        "differences": [{**d, "pc": rnd(d["pc"])} for d in differences],
        "rows": [{"id": i, "labels": row_labels[i], "pc": rnd(p)} for i, p in zip(row_ids, row_pcs)],
        "counts": counts,
    }
    write_json(args.output, out)
    inputs = [*args.judged, *sorted(args.role_vectors.glob("*.pt")), *sorted(args.reference_vectors.glob("*.pt")), args.axis]
    write_meta(args.output, inputs, args, counts, {"cache_parts": cache_parts})
    print({n: (c["n_pos"], c["n_neg"]) for n, c in counts.items()})
    print("variance ratio PC1-3:", out["variance_ratio"][:3], "reference:", out["variance_ratio_reference"][:3])
    print("PC cos with reference:", out["pc_cos_reference"], "PC1 cos axis:", round(out["pc1_cos_axis"], 4))


if __name__ == "__main__":
    main()
