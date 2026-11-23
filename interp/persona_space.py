"""Place benchmark responses and the sycophancy persona vector in Assistant Axis persona space at one block.

Persona space is a PCA of the role vectors in --role-vectors (assistant-axis pipeline files <role>.pt holding
{"vector": (blocks, hidden), "type", "role"}): the "pos_3" role vectors are centred on their mean and fitted, and
default.pt ("mean") is the default Assistant vector, projected but not fitted. PC1 is signed toward the Assistant Axis,
default minus the role mean. Units and labels are those of interp.generalization_matrix (ELEPHANT and Are You Sure);
each unit gives the mean activation of its label-1 rows and of its label-0 rows. A pooled point averages one label's
per-source class means with equal weight, over the sources in POOLS.
Locations: each source's mean activation (its labelled rows), the ELEPHANT and Are You Sure centres and the default vector,
with their nearest roles by distance and by cosine after subtracting the role mean, and the parts of their offset from the
role mean inside and outside the role span. Variance: the share of each benchmark's response-to-response variance along
the top role PCs and inside the role span, against the benchmark's own top PCs and a random subspace. Differences: label-1 minus label-0 means, source minus source means, and the
persona vector (--persona-vector, its response-mean difference; block L uses its hidden_states index L + 1), each with the
roles it points toward and away from. For label differences, --shuffles permutations of the labels within each source give a
chance line: the 95th percentile of the best role cosine, the 5th of the worst, and the 95th of |cosine with the persona
vector|. Cosines among the pooled shifts, the persona vector, the Are You Sure minus ELEPHANT offset and the Assistant
Axis are reported as a matrix. --reference-vectors, an independent run of the same pipeline, is fitted the same way to
check the PCs.
--contrastive-activations (get_activations output for the system-prompt pair responses) adds each pair's sycophantic and
non-sycophantic side means as points, their difference as a shift (chance line: --shuffles random sign flips of the
per-prompt differences), the pooled taxonomy and control shifts, the top PC of the prompt-centred activations (signed
and scaled by the sycophantic-minus-non-sycophantic gap over the baseline and taxonomy pairs), the sides as locations
and the pairs as a variance group.
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
from interp.generalization_matrix import LAYER_CONVENTION, PERSONA_KEY, PROBE_POSITION, position_dir, unit_rows
from utils.activations import act_key
from utils.io import check_counts, read_json, read_jsonl, write_json, write_meta

ELEPHANT_SOURCES = ("AITA-YTA", "OEQ", "SS")
AYS_SOURCES = ("mmlu_mc_cot", "math_mc_cot", "aqua_mc", "truthful_qa_mc", "trivia_qa", "truthful_qa")
POOLS = {"validation": ELEPHANT_SOURCES, "indirectness": ELEPHANT_SOURCES, "framing": ELEPHANT_SOURCES, "syco": AYS_SOURCES}
BENCHMARKS = {"ELEPHANT": ELEPHANT_SOURCES, "Are You Sure": AYS_SOURCES}
CENTRES = {f"{b} centre": sources for b, sources in BENCHMARKS.items()}
N_TOP_ROLES = 5
CONTRASTIVE_LAYER_CONVENTION = "hidden_states[layer + 1]"  # get_activations' layer L is the cache's block L
SYCOPHANCY_PAIR_TYPES = ("baseline", "taxonomy")  # the other pair type is "control"
CONTRASTIVE_POOLS = {"contrastive/pooled taxonomy": "taxonomy", "contrastive/pooled controls": "control"}


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


def class_differences(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(labelings, hidden): mean of the label-1 rows minus mean of the label-0 rows of X, for each row of the 0/1 matrix Y."""
    pos = Y / Y.sum(1, keepdims=True)
    neg = (1 - Y) / (1 - Y).sum(1, keepdims=True)
    return (pos - neg) @ X


def pair_rows(polarity: np.ndarray, pair_index: np.ndarray, cell: np.ndarray, pair_type: np.ndarray,
              prompt_id: np.ndarray) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """{"<pair index> <cell>": (pair type, sycophantic row indices, non-sycophantic row indices)}, the two index arrays
    aligned by prompt. Each prompt must have exactly one row of each polarity in each pair."""
    out = {}
    for p in sorted(set(pair_index.tolist())):
        idx = np.flatnonzero(pair_index == p)
        cells, types = set(cell[idx].tolist()), set(pair_type[idx].tolist())
        if len(cells) != 1 or len(types) != 1:
            raise ValueError(f"pair {p}: cells {cells}, pair types {types}")
        sides = []
        for side in ("sycophantic", "non_sycophantic"):
            rows = idx[polarity[idx] == side]
            sides.append(rows[np.argsort(prompt_id[rows], kind="stable")])
        s, n = sides
        if len(s) + len(n) != len(idx):
            raise ValueError(f"pair {p}: polarities {sorted(set(polarity[idx].tolist()))}")
        if not np.array_equal(prompt_id[s], prompt_id[n]) or len(set(prompt_id[s].tolist())) != len(s):
            raise ValueError(f"pair {p}: each prompt needs one sycophantic and one non-sycophantic row")
        out[f"{p:02d} {cells.pop()}"] = (types.pop(), s, n)
    return out


def prompt_centred_pc1(X: np.ndarray, prompt_id: np.ndarray) -> tuple[np.ndarray, float]:
    """Top unit eigenvector of X with each prompt's mean row subtracted, and its share of that centred variance."""
    _, inverse, counts = np.unique(prompt_id, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), X.shape[1]))
    np.add.at(sums, inverse, X)
    Xp = X - (sums / counts[:, None])[inverse]
    evals, evecs = np.linalg.eigh(Xp.T @ Xp)
    return evecs[:, -1], float(evals[-1] / evals.sum())


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
    parser.add_argument("--persona-vector", type=Path, help="Persona vector .npz; its response-mean difference is placed too.")
    parser.add_argument("--contrastive-activations", type=Path,
                        help="get_activations .npz of the system-prompt pair responses; their pairs are placed too.")
    parser.add_argument("--shuffles", type=int, required=True, help="Label permutations per source for the chance line.")
    parser.add_argument("--seed", type=int, required=True)
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
    role_dirs = (R - mean) / np.linalg.norm(R - mean, axis=1, keepdims=True)

    def project(v):
        return (v - mean) @ pcs.T

    persona = None
    if args.persona_vector is not None:
        persona_meta = read_json(args.persona_vector.parent / "meta" / f"{args.persona_vector.name}.meta.json")
        if persona_meta["model"] != args.model:
            raise ValueError(f"persona vector is for {persona_meta['model']}, role vectors for {args.model}")
        persona = np.load(args.persona_vector)[PERSONA_KEY][args.block + 1].astype(np.float64)

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

    # class means: per unit, then pooled per label over its sources with equal weight
    class_mean, counts, X, Y = {}, {}, {}, {}
    for name, members in sorted(units.items()):
        Y[name] = np.array([v for _, v in members])
        X[name] = A[[row_of[i] for i, _ in members]].astype(np.float64)
        n_pos, n_neg = int(Y[name].sum()), int(len(Y[name]) - Y[name].sum())
        if min(n_pos, n_neg) == 0:
            raise ValueError(f"{name}: needs both classes, has {n_pos} positive and {n_neg} negative")
        counts[name] = {**check_counts(len(members) + sum(excluded[name].values()), excluded[name], len(members), name),
                        "n_pos": n_pos, "n_neg": n_neg}
        for label in (1, 0):
            class_mean[name, label] = X[name][Y[name] == label].mean(0)
    pools = {}
    for label, sources in POOLS.items():
        present = [f"{s}/{label}" in units for s in sources]
        if any(present) and not all(present):
            raise ValueError(f"pooled {label} needs all of {sources}")
        if all(present):
            pools[f"pooled/{label}"] = [f"{s}/{label}" for s in sources]
            for y in (1, 0):
                class_mean[f"pooled/{label}", y] = np.mean([class_mean[u, y] for u in pools[f"pooled/{label}"]], axis=0)

    # source means (all labelled rows of the source) and the centres of each benchmark
    source_ids = defaultdict(set)
    for name, members in units.items():
        source_ids[name.split("/")[0]].update(i for i, _ in members)
    source_mean = {s: A[sorted(row_of[i] for i in ids)].astype(np.float64).mean(0) for s, ids in sorted(source_ids.items())}
    centres = {c: np.mean([source_mean[s] for s in ss], axis=0) for c, ss in CENTRES.items() if set(ss) <= set(source_mean)}

    # contrastive pairs: each side's mean over the pair's prompts (label 1 = sycophantic side), pooled per pair type
    pairs, contrastive_locations = {}, []
    if args.contrastive_activations is not None:
        c_meta = read_json(args.contrastive_activations.parent / "meta" / f"{args.contrastive_activations.name}.meta.json")
        if (c_meta["model"], c_meta["layer_convention"]) != (args.model, CONTRASTIVE_LAYER_CONVENTION) \
                or args.block not in c_meta["layers"]:
            raise ValueError(f"{args.contrastive_activations}: {c_meta['model']}, {c_meta['layer_convention']!r}, "
                             f"layers {c_meta['layers']}")
        if args.position not in PROBE_POSITION:
            raise ValueError(f"no contrastive activations at {args.position}")
        z = np.load(args.contrastive_activations)
        if not np.array_equal(z["label"], (z["polarity"] == "sycophantic").astype(z["label"].dtype)):
            raise ValueError("contrastive label is not 1 exactly on the sycophantic side")
        Xc = z[act_key(PROBE_POSITION[args.position], args.block)].astype(np.float64)
        c_prompts = z["prompt_id"]
        pairs = {f"contrastive/{k}": v for k, v in
                 pair_rows(z["polarity"], z["pair_index"], z["cell"], z["pair_type"], c_prompts).items()}
        n_side = sum(len(s) for _, s, _ in pairs.values())
        counts["contrastive"] = {**check_counts(len(Xc), {}, 2 * n_side, "contrastive rows"), "n_pos": n_side, "n_neg": n_side}
        for name, (_, s, n) in pairs.items():
            class_mean[name, 1], class_mean[name, 0] = Xc[s].mean(0), Xc[n].mean(0)
        for pool, pair_type in CONTRASTIVE_POOLS.items():
            pools[pool] = [name for name, (t, _, _) in pairs.items() if t == pair_type]
            if not pools[pool]:
                raise ValueError(f"no {pair_type} pairs for {pool}")
            for y in (1, 0):
                class_mean[pool, y] = np.mean([class_mean[u, y] for u in pools[pool]], axis=0)
        syc = np.concatenate([s for t, s, _ in pairs.values() if t in SYCOPHANCY_PAIR_TYPES])
        non = np.concatenate([n for t, _, n in pairs.values() if t in SYCOPHANCY_PAIR_TYPES])
        contrastive_locations = [("contrastive centre", Xc.mean(0)),
                                 ("contrastive sycophantic side (baseline, taxonomy)", Xc[syc].mean(0)),
                                 ("contrastive non-sycophantic side (baseline, taxonomy)", Xc[non].mean(0))]

    points = [{"kind": "role", "name": r, "pc": project(v), "pc_reference": (v_ref - ref_mean) @ ref_pcs.T}
              for r, v, v_ref in zip(roles, R, R_ref)]
    points.append({"kind": "default", "name": "default", "pc": project(default),
                   "pc_reference": (default_ref - ref_mean) @ ref_pcs.T})
    for (name, label), v in class_mean.items():
        kind = name.split("/")[0] if name.startswith(("pooled/", "contrastive/")) else "unit"
        points.append({"kind": kind, "name": name, "label": label,
                       "pc": project(v), "pc_reference": (v - ref_mean) @ ref_pcs.T})
    for name, v in [*source_mean.items(), *centres.items()]:
        points.append({"kind": "source" if name in source_mean else "centre", "name": name, "pc": project(v),
                       "pc_reference": (v - ref_mean) @ ref_pcs.T})

    # locations: nearest roles by distance and by cosine from the role mean, offset inside and outside the role span
    locations = []
    for name, v in [*source_mean.items(), *centres.items(), ("default Assistant", default), *contrastive_locations]:
        off = v - mean
        inside = span @ off
        outside = float(np.sqrt(max(off @ off - inside @ inside, 0.0)))
        dist = np.linalg.norm(R - v, axis=1)
        cos = role_dirs @ off / np.linalg.norm(off)
        by_dist, by_cos = np.argsort(dist), np.argsort(-cos)
        locations.append({
            "name": name, "offset_norm": float(np.linalg.norm(off)), "offset_in_span": float(np.linalg.norm(inside)),
            "offset_outside_span": outside, "pc": project(v),
            "nearest_by_distance": [[roles[i], float(dist[i])] for i in by_dist[:N_TOP_ROLES]],
            "nearest_by_cosine": [[roles[i], float(cos[i])] for i in by_cos[:N_TOP_ROLES]],
        })
    # variance: share of each benchmark's response-to-response variance along PC1, PC1-n_pcs, the PCs holding 70% of the
    # role variance and the whole role span; centred on the benchmark (pooled) or on each source (within source)
    n70 = int(np.argmax(np.cumsum(var_ratio) >= 0.70) + 1)
    ks = sorted({1, args.n_pcs, n70, len(roles) - 1})

    def persona_shares(X):
        Xc = X - X.mean(0)
        total = float((Xc ** 2).sum())
        return total, [float(((Xc @ all_pcs[:k].T) ** 2).sum() / total) for k in ks]

    def variance_group(by_source):
        Xb = np.concatenate(list(by_source.values()))
        total, pooled = persona_shares(Xb)
        own = np.linalg.svd(Xb - Xb.mean(0), compute_uv=False) ** 2
        per = {s: {"n": len(x), "persona": persona_shares(x)[1]} for s, x in by_source.items()}
        return {
            "n": len(Xb), "variance_per_response": total / len(Xb), "persona_pooled": pooled,
            "persona_within_source_mean": [float(np.mean([p["persona"][j] for p in per.values()])) for j in range(len(ks))],
            "own_top_pcs": [float(own[:k].sum() / total) for k in ks], "per_source": per,
        }

    variance = {"ks": ks, "random_subspace": [k / R.shape[1] for k in ks], "benchmarks": {}}
    for bench, sources in BENCHMARKS.items():
        if set(sources) <= set(source_ids):
            variance["benchmarks"][bench] = variance_group(
                {s: A[sorted(row_of[i] for i in source_ids[s])].astype(np.float64) for s in sources})
    if pairs:  # sources are the pairs, each with both sides
        variance["benchmarks"]["Contrastive"] = variance_group({name: Xc[np.concatenate([s, n])] for name, (_, s, n) in pairs.items()})

    pairwise = np.linalg.norm(R[:, None] - R[None], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    yardsticks = {"role_to_role_mean_median": float(np.median(np.linalg.norm(R - mean, axis=1))),
                  "nearest_role_neighbour_median": float(np.median(pairwise.min(1)))}

    # differences: labels (with a shuffled-label chance line), sources, the persona vector
    diffs = {name: class_mean[name, 1] - class_mean[name, 0] for name in sorted({n for n, _ in class_mean})}
    rng = np.random.default_rng(args.seed)
    shuffled = {u: class_differences(X[u], np.stack([rng.permutation(Y[u]) for _ in range(args.shuffles)]))
                for u in sorted(units)}
    for name, (_, s, n) in pairs.items():  # paired design: flip the sign of random prompts' differences
        signs = rng.choice([-1.0, 1.0], size=(args.shuffles, len(s)))
        shuffled[name] = signs @ (Xc[s] - Xc[n]) / len(s)
    shuffled.update({p: np.mean([shuffled[u] for u in us], axis=0) for p, us in pools.items()})
    for a, b in combinations([s for s in ELEPHANT_SOURCES if s in source_mean], 2):
        diffs[f"source/{a} - {b}"] = source_mean[a] - source_mean[b]
    if len(centres) == 2:
        diffs["source/Are You Sure centre - ELEPHANT centre"] = centres["Are You Sure centre"] - centres["ELEPHANT centre"]
    if persona is not None:
        diffs["persona vector"] = persona
    contrastive = None
    if pairs:
        pc1, pc1_share = prompt_centred_pc1(Xc, c_prompts)
        gap = float((Xc[syc] @ pc1).mean() - (Xc[non] @ pc1).mean())
        diffs["contrastive/prompt-centred PC1"] = gap * pc1  # points toward the sycophantic side, length = the gap
        contrastive = {"pairs": {name: {"pair_type": t, "n_prompts": len(s)} for name, (t, s, _) in pairs.items()},
                       "pc1_variance_share": pc1_share, "pc1_gap": abs(gap)}

    def unit_cos(a, b):
        return a @ b / np.linalg.norm(a, axis=-1) / np.linalg.norm(b)

    differences = []
    for name, d in diffs.items():
        cos_roles = role_dirs @ d / np.linalg.norm(d)
        order = np.argsort(cos_roles)
        chance = None
        if name in shuffled:
            S = shuffled[name]
            C = (S / np.linalg.norm(S, axis=1, keepdims=True)) @ role_dirs.T
            chance = {"toward_95": float(np.percentile(C.max(1), 95)), "away_5": float(np.percentile(C.min(1), 5)),
                      "abs_cos_persona_95": None if persona is None else float(np.percentile(np.abs(unit_cos(S, persona)), 95))}
        differences.append({
            "name": name, "norm": float(np.linalg.norm(d)), "pc": d @ pcs.T,
            "share_in_top_pcs": [float(np.sum((pcs[:k] @ d) ** 2) / (d @ d)) for k in range(1, args.n_pcs + 1)],
            "share_in_role_span": float(np.sum((span @ d) ** 2) / (d @ d)),
            "cos_axis": float(d @ axis / np.linalg.norm(d) / np.linalg.norm(axis)),
            "cos_persona": None if persona is None else float(unit_cos(d, persona)),
            "toward_roles": [[roles[i], float(cos_roles[i])] for i in order[::-1][:N_TOP_ROLES]],
            "away_roles": [[roles[i], float(cos_roles[i])] for i in order[:N_TOP_ROLES]],
            "chance": chance,
        })

    # cosines among the pooled shifts, the persona vector, the benchmark offset, the contrastive baseline, pools and PC1,
    # and the Assistant Axis
    keep = {"persona vector", "source/Are You Sure centre - ELEPHANT centre", *CONTRASTIVE_POOLS,
            "contrastive/prompt-centred PC1", *(name for name, (t, _, _) in pairs.items() if t == "baseline")}
    main = {n: d for n, d in diffs.items() if n.startswith("pooled/") or n in keep}
    main["Assistant Axis"] = axis
    names = list(main)
    M = np.stack([main[n] / np.linalg.norm(main[n]) for n in names])
    direction_cosines = {"names": names, "cos": [[round(float(x), 5) for x in row] for row in M @ M.T]}

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
        "locations": [{**loc, "pc": rnd(loc["pc"])} for loc in locations], "yardsticks": yardsticks, "variance": variance,
        "differences": [{**d, "pc": rnd(d["pc"])} for d in differences], "direction_cosines": direction_cosines,
        "rows": [{"id": i, "labels": row_labels[i], "pc": rnd(p)} for i, p in zip(row_ids, row_pcs)],
        "contrastive": contrastive, "counts": counts,
    }
    write_json(args.output, out)
    inputs = [*args.judged, *sorted(args.role_vectors.glob("*.pt")), *sorted(args.reference_vectors.glob("*.pt")), args.axis,
              *([args.persona_vector] if args.persona_vector is not None else []),
              *([args.contrastive_activations] if args.contrastive_activations is not None else [])]
    write_meta(args.output, inputs, args, counts, {"cache_parts": cache_parts})
    print({n: (c["n_pos"], c["n_neg"]) for n, c in counts.items()})
    print("variance ratio PC1-3:", out["variance_ratio"][:3], "reference:", out["variance_ratio_reference"][:3])
    print("PC cos with reference:", out["pc_cos_reference"], "PC1 cos axis:", round(out["pc1_cos_axis"], 4))


if __name__ == "__main__":
    main()
