"""Compare token position, layer and probe setting (C or diff-in-means) across the contrastive probe sweep."""
import argparse
import csv
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import rankdata, spearmanr

from utils.io import check_counts, read_jsonl, write_meta

POSITIONS = ("last_prompt", "first5", "response")
POSITION_NAMES = {"last_prompt": "last prompt token", "first5": "first 5 response tokens", "response": "response mean"}
POSITION_COLORS = {"last_prompt": "#2a78d6", "first5": "#eb6834", "response": "#1baf7a"}
FACTORS = ("cell", "eval", "setting", "position", "layer")
DIVERGING = LinearSegmentedColormap.from_list("auroc", ["#b3261e", "#e34948", "#f0efec", "#2a78d6", "#104281"])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdad5"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def auroc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of an AUROC."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    return float(np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)))


def variance_components(a: np.ndarray, names: tuple[str, ...]) -> list[dict]:
    """Sums of squares for main effects and two-way interactions of a balanced full-factorial array; the rest is higher-order."""
    if a.ndim != len(names) or not np.isfinite(a).all():
        raise ValueError("array must be finite with one axis per factor name")
    grand, n = a.mean(), a.size
    total = float(((a - grand) ** 2).sum())
    axes = range(a.ndim)

    def marginal(keep):
        return a.mean(axis=tuple(i for i in axes if i not in keep), keepdims=True)

    main = {i: marginal((i,)) - grand for i in axes}
    rows = []
    for i in axes:
        rows.append({"term": names[i], "df": a.shape[i] - 1, "ss": float((main[i] ** 2).sum() * n / a.shape[i])})
    for i, j in combinations(axes, 2):
        effect = marginal((i, j)) - main[i] - main[j] - grand
        rows.append({"term": f"{names[i]} x {names[j]}", "df": (a.shape[i] - 1) * (a.shape[j] - 1),
                     "ss": float((effect ** 2).sum() * n / (a.shape[i] * a.shape[j]))})
    df_total = n - 1
    rows.append({"term": "higher-order interactions", "df": df_total - sum(r["df"] for r in rows),
                 "ss": total - sum(r["ss"] for r in rows)})
    for r in rows:
        r["share"] = r["ss"] / total
    if rows[-1]["ss"] < -1e-9 * total:
        raise AssertionError("negative residual sum of squares: array is not a balanced factorial")
    return rows


def rank_and_wins(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank (1 = highest, ties averaged) and win credit (ties split) along the last axis."""
    ranks = rankdata(-scores, axis=-1, method="average")
    is_max = scores == scores.max(axis=-1, keepdims=True)
    return ranks, is_max / is_max.sum(axis=-1, keepdims=True)


def load(args):
    targets = read_csv(args.targets)
    manifest_path = args.sweep_dir / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    if len({r["probe_id"] for r in manifest}) != len(manifest):
        raise ValueError("manifest has duplicate probe IDs")
    models = {r["model"] for r in manifest}
    if len(models) != 1:
        raise ValueError(f"manifest mixes models: {models}")
    for r in manifest:
        if (r["method"] == "logistic") != (r["C"] is not None) or r["method"] not in ("logistic", "dim"):
            raise ValueError(f"{r['probe_id']}: method {r['method']} with C={r['C']}")
    pairs = sorted({r["pair_index"] for r in manifest})
    cell_of = {r["pair_index"]: (r["cell"], r["pair_type"]) for r in manifest}
    if len({(r["pair_index"], r["cell"], r["pair_type"]) for r in manifest}) != len(pairs):
        raise ValueError("a pair index maps to more than one cell")
    cs = sorted({r["C"] for r in manifest if r["C"] is not None})
    settings = [f"C={c:g}" for c in cs] + ["DIM"]
    layers = sorted({r["layer"] for r in manifest})
    if {r["position"] for r in manifest} != set(POSITIONS):
        raise ValueError("unexpected token positions in manifest")

    index = {}
    for r in manifest:
        s = settings.index("DIM" if r["C"] is None else f"C={r['C']:g}")
        index[r["probe_id"]] = (pairs.index(r["pair_index"]), s, POSITIONS.index(r["position"]), layers.index(r["layer"]))
    grid = (len(pairs), len(settings), len(POSITIONS), len(layers))
    if len(set(index.values())) != len(manifest) or len(manifest) != np.prod(grid):
        raise ValueError(f"manifest is not a complete {grid} grid")
    val, test = np.full(grid, np.nan), np.full(grid, np.nan)
    for r in manifest:
        val[index[r["probe_id"]]], test[index[r["probe_id"]]] = r["val_auroc"], r["test_auroc"]
    test_n = {(r["n_test_pos"], r["n_test"] - r["n_test_pos"]) for r in manifest}
    if len(test_n) != 1:
        raise ValueError(f"test split sizes differ across probes: {test_n}")

    auroc = np.full((len(pairs), len(targets), *grid[1:]), np.nan)
    eval_n, counts, inputs = [], {}, [args.targets, manifest_path]
    for e, t in enumerate(targets):
        path = args.sweep_dir / "eval" / f"{t['eval_dataset']}.jsonl"
        inputs.append(path)
        rows = read_jsonl(path)
        primary = [r for r in rows if r["eval_label"] == t["target_label"]]
        counts[t["eval_dataset"]] = check_counts(len(rows), {"off_target_label": len(rows) - len(primary)}, len(primary), path.name)
        if len(primary) != len(manifest) or {r["probe_id"] for r in primary} != set(index):
            raise ValueError(f"{path}: target label rows do not match the manifest one-to-one")
        sizes = {(r["n"], r["n_pos"], r["n_neg"]) for r in primary}
        if len(sizes) != 1 or next(iter(sizes))[0] != int(t["n"]):
            raise ValueError(f"{path}: row counts {sizes} differ across probes or from targets.csv n={t['n']}")
        if any(r["auroc"] is None for r in primary):
            raise ValueError(f"{path}: undefined AUROC for target label {t['target_label']}")
        eval_n.append(next(iter(sizes)))
        for r in primary:
            p, s, pos, lay = index[r["probe_id"]]
            auroc[p, e, s, pos, lay] = r["auroc"]
    if np.isnan(auroc).any() or np.isnan(val).any() or np.isnan(test).any():
        raise AssertionError("grid has unfilled entries")
    return dict(targets=targets, pairs=pairs, cell_of=cell_of, settings=settings, layers=layers, auroc=auroc,
                val=val, test=test, test_n=next(iter(test_n)), eval_n=eval_n, counts=counts, inputs=inputs,
                model=next(iter(models)))


def meta(path, d, args, n_in, n_out, extra=None):
    write_meta(path, d["inputs"], args, {"n_in": n_in, "excluded": {}, "n_out": n_out, "eval_files": d["counts"]},
               {"model": d["model"], "source_sweep": args.sweep_dir.name, **(extra or {})})


def save_fig(fig, path, d, args, n_in, extra=None):
    fig.savefig(path, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    meta(path, d, args, n_in, 1, extra)


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6)


def plot_cell(d, p, ylim, path, args):
    targets, settings, layers = d["targets"], d["settings"], d["layers"]
    cell, pair_type = d["cell_of"][d["pairs"][p]]
    rows = ["In-distribution test"] + [t["display"] for t in targets]
    fig, axes = plt.subplots(len(rows), len(settings), figsize=(1.85 * len(settings) + 1.2, 1.2 * len(rows) + 1.2),
                             sharex=True, squeeze=False)
    id_lo = np.floor(d["test"][p].min() * 20) / 20
    for i, label in enumerate(rows):
        for s, setting in enumerate(settings):
            ax = axes[i, s]
            style(ax)
            if i == 0:
                values, n_pos, n_neg = d["test"][p, s], *d["test_n"]
                ax.set_ylim(min(id_lo, 0.95), 1.005)
            else:
                values, (_, n_pos, n_neg) = d["auroc"][p, i - 1, s], d["eval_n"][i - 1]
                half = 1.96 * auroc_se(0.5, n_pos, n_neg)
                ax.axhspan(0.5 - half, 0.5 + half, color=GRID, alpha=0.6, lw=0)
                ax.axhline(0.5, color=MUTED, lw=0.6)
                ax.set_ylim(*ylim)
            for pos, name in enumerate(POSITIONS):
                ax.plot(layers, values[pos], color=POSITION_COLORS[name], lw=1.2, marker="o", ms=2.2)
            if i == 0:
                ax.set_title(setting, fontsize=8, color=INK)
            if s == 0:
                ax.set_ylabel(label, fontsize=6.5, color=INK, rotation=0, ha="right", va="center")
            else:
                ax.tick_params(labelleft=False)
            if i == len(rows) - 1:
                ax.set_xticks(layers)
                ax.set_xlabel("layer", fontsize=7, color=MUTED)
    handles = [Line2D([], [], color=POSITION_COLORS[n], lw=1.5, marker="o", ms=3, label=POSITION_NAMES[n]) for n in POSITIONS]
    handles.append(Line2D([], [], color=GRID, lw=6, label="chance ± 1.96 SE (Hanley–McNeil)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.55 / fig.get_figheight()),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"pair{d['pairs'][p]:02d} · {cell} ({pair_type}) — AUROC by layer", fontsize=11, color=INK,
                 y=1 - 0.15 / fig.get_figheight())
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.8 / fig.get_figheight(), top=1 - 1.0 / fig.get_figheight(),
                        hspace=0.25, wspace=0.08)
    save_fig(fig, path, d, args, int(d["auroc"][p].size + d["test"][p].size),
             {"rows": rows, "columns": settings, "ylim_ood": list(ylim)})


def heatmap(ax, values, xlabels, ylabels, norm, fmt, cmap=DIVERGING, title="", pad=6):
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=7)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.tick_params(length=0, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i, j), v in np.ndenumerate(values):
        rgba = im.cmap(im.norm(v))
        light = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.55
        ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6, color=INK if light else "white")
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=pad)
    return im


def summarize(d, args):
    out = args.out_dir
    a, val, test = d["auroc"], d["val"], d["test"]
    P, E, S, NP, L = a.shape
    layers, settings, targets = d["layers"], d["settings"], d["targets"]
    pos_names = [POSITION_NAMES[n] for n in POSITIONS]

    # Position x layer, pooled over cells, evals and settings; ranks within each (cell, eval, setting) block of 27.
    flat27 = a.reshape(P, E, S, NP * L)
    rank27, win27 = rank_and_wins(flat27)
    pl = {
        "mean_auroc": a.mean(axis=(0, 1, 2)), "median_auroc": np.median(a, axis=(0, 1, 2)),
        "mean_rank": rank27.mean(axis=(0, 1, 2)).reshape(NP, L), "win_share": win27.mean(axis=(0, 1, 2)).reshape(NP, L),
        "frac_below_half": (a < 0.5).mean(axis=(0, 1, 2)),
        "mean_val_auroc": val.mean(axis=(0, 1)), "mean_test_auroc": test.mean(axis=(0, 1)),
    }
    rows = [{"position": POSITIONS[i], "layer": layers[j], **{k: float(v[i, j]) for k, v in pl.items()}, "n_scores": P * E * S}
            for i in range(NP) for j in range(L)]
    save_csv(out / "position_layer_summary.csv", rows)
    meta(out / "position_layer_summary.csv", d, args, a.size, len(rows), {"rank": "1 = best of 27 position x layer configs within each (cell, eval, setting)"})

    # Position x layer x setting; ranks within each (cell, eval) block of 189.
    flat189 = a.reshape(P, E, S * NP * L)
    rank189, win189 = rank_and_wins(flat189)
    cfg = {
        "mean_auroc": a.mean(axis=(0, 1)), "median_auroc": np.median(a, axis=(0, 1)),
        "sd_auroc": a.std(axis=(0, 1)), "mean_rank": rank189.mean(axis=(0, 1)).reshape(S, NP, L),
        "win_share": win189.mean(axis=(0, 1)).reshape(S, NP, L), "frac_below_half": (a < 0.5).mean(axis=(0, 1)),
        "mean_val_auroc": val.mean(axis=0), "mean_test_auroc": test.mean(axis=0),
    }
    rows = [{"setting": settings[s], "position": POSITIONS[i], "layer": layers[j],
             **{k: float(v[s, i, j]) for k, v in cfg.items()}, "n_scores": P * E}
            for s in range(S) for i in range(NP) for j in range(L)]
    rows.sort(key=lambda r: r["mean_rank"])
    save_csv(out / "config_summary.csv", rows)
    meta(out / "config_summary.csv", d, args, a.size, len(rows), {"rank": "1 = best of 189 configs within each (cell, eval)"})

    # C robustness: mean AUROC per setting for each position x layer, and agreement of the 27-config profile across settings.
    by_setting = a.mean(axis=(0, 1))  # S, NP, L
    logistic = [s for s, name in enumerate(settings) if name != "DIM"]
    rows = []
    for i in range(NP):
        for j in range(L):
            v = by_setting[:, i, j]
            lv = v[logistic]
            rows.append({"position": POSITIONS[i], "layer": layers[j], **{settings[s]: float(v[s]) for s in range(S)},
                         "logistic_range": float(lv.max() - lv.min()), "logistic_sd": float(lv.std()),
                         "best_setting": settings[int(np.argmax(v))]})
    save_csv(out / "c_robustness.csv", rows)
    meta(out / "c_robustness.csv", d, args, a.size, len(rows))
    profiles = by_setting.reshape(S, NP * L)
    rho = np.array([[spearmanr(profiles[x], profiles[y]).statistic for y in range(S)] for x in range(S)])
    rows = [{"setting": settings[x], **{settings[y]: float(rho[x, y]) for y in range(S)}} for x in range(S)]
    save_csv(out / "c_rank_agreement.csv", rows)
    meta(out / "c_rank_agreement.csv", d, args, a.size, len(rows), {"statistic": "Spearman rho between settings over the 27 position x layer mean AUROCs"})

    # Variance decomposition over the full OOD grid.
    comps = variance_components(a, FACTORS)  # axes of a are FACTORS in order
    save_csv(out / "variance_decomposition.csv", comps)
    meta(out / "variance_decomposition.csv", d, args, a.size, len(comps), {"factors": dict(zip(FACTORS, a.shape))})

    # Noise floor per eval.
    rows = [{"eval_dataset": t["eval_dataset"], "display": t["display"], "n": n, "n_pos": n_pos, "n_neg": n_neg,
             "se_at_0.5": auroc_se(0.5, n_pos, n_neg), "ci95_half_width_at_0.5": 1.96 * auroc_se(0.5, n_pos, n_neg),
             "se_at_0.7": auroc_se(0.7, n_pos, n_neg)}
            for t, (n, n_pos, n_neg) in zip(targets, d["eval_n"])]
    se_mean = float(np.sqrt(sum(r["se_at_0.5"] ** 2 for r in rows)) / len(rows))
    save_csv(out / "noise_floor.csv", rows)
    meta(out / "noise_floor.csv", d, args, len(rows), len(rows),
         {"se_of_mean_over_evals_at_0.5_if_independent": se_mean,
          "note": "Two probes scored on the same eval share its rows, so the SE of their AUROC difference is smaller than these values imply."})

    # Per cell: position x layer mean over evals and settings; regret of the pooled best config.
    per_cell = a.mean(axis=(1, 2))  # P, NP, L
    g_i, g_j = np.unravel_index(np.argmax(pl["mean_auroc"]), (NP, L))
    rows = []
    for p in range(P):
        i, j = np.unravel_index(np.argmax(per_cell[p]), (NP, L))
        cell, pair_type = d["cell_of"][d["pairs"][p]]
        rows.append({"pair_index": d["pairs"][p], "cell": cell, "pair_type": pair_type,
                     "best_position": POSITIONS[i], "best_layer": layers[j], "best_mean_auroc": float(per_cell[p, i, j]),
                     "pooled_best_position": POSITIONS[g_i], "pooled_best_layer": layers[g_j],
                     "pooled_best_mean_auroc": float(per_cell[p, g_i, g_j]),
                     "regret": float(per_cell[p, i, j] - per_cell[p, g_i, g_j])})
    save_csv(out / "per_cell_best.csv", rows)
    meta(out / "per_cell_best.csv", d, args, a.size, len(rows), {"selection": "post hoc on OOD mean AUROC over evals and settings; descriptive"})

    # Criterion agreement: in-distribution validation vs OOD mean, per config.
    cfg_val, cfg_ood = val.mean(axis=0).ravel(), a.mean(axis=(0, 1)).ravel()
    pl_val, pl_ood = pl["mean_val_auroc"].ravel(), pl["mean_auroc"].ravel()
    labels189 = [(settings[s], POSITIONS[i], layers[j]) for s in range(S) for i in range(NP) for j in range(L)]
    labels27 = [(POSITIONS[i], layers[j]) for i in range(NP) for j in range(L)]
    agreement = []
    for level, v, o, labels in (("setting x position x layer (189)", cfg_val, cfg_ood, labels189),
                                ("position x layer (27)", pl_val, pl_ood, labels27)):
        top_val, top_ood = int(np.argmax(v)), int(np.argmax(o))
        agreement.append({"level": level, "spearman_rho": float(spearmanr(v, o).statistic),
                          "best_by_val": " ".join(map(str, labels[top_val])), "best_by_val_ood_auroc": float(o[top_val]),
                          "best_by_val_val_auroc": float(v[top_val]), "n_tied_at_max_val": int((v == v.max()).sum()),
                          "best_by_ood": " ".join(map(str, labels[top_ood])), "best_by_ood_ood_auroc": float(o[top_ood]),
                          "best_by_ood_val_auroc": float(v[top_ood])})
    save_csv(out / "criterion_agreement.csv", agreement)
    meta(out / "criterion_agreement.csv", d, args, a.size + val.size, len(agreement))

    # Figures.
    norm = TwoSlopeNorm(vmin=0.35, vcenter=0.5, vmax=0.75)
    fig, axes = plt.subplots(5, 1, figsize=(9, 11))
    xl = [str(l) for l in layers]
    heatmap(axes[0], pl["mean_auroc"], xl, pos_names, norm, lambda v: f"{v:.3f}", title="Mean OOD AUROC (14 cells × 15 evals × 7 settings)")
    heatmap(axes[1], pl["mean_rank"], xl, pos_names, None, lambda v: f"{v:.1f}", cmap="Blues_r",
            title="Mean rank among 27 position × layer configs (1 = best), within each cell × eval × setting")
    heatmap(axes[2], pl["win_share"], xl, pos_names, None, lambda v: f"{v:.1%}", cmap="Blues",
            title="Share of cell × eval × setting blocks where the config is best")
    heatmap(axes[3], pl["frac_below_half"], xl, pos_names, None, lambda v: f"{v:.0%}", cmap="Reds",
            title="Fraction of OOD scores with AUROC < 0.5 (inverted direction)")
    heatmap(axes[4], pl["mean_val_auroc"], xl, pos_names, None, lambda v: f"{v:.3f}", cmap="Blues",
            title="Mean in-distribution validation AUROC (14 cells × 7 settings)")
    axes[4].set_xlabel("layer", fontsize=8, color=MUTED)
    fig.tight_layout()
    save_fig(fig, out / "summary_position_layer.png", d, args, a.size + val.size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [3.2, 1]})
    xl27 = [f"{l}" for _ in POSITIONS for l in layers]
    heatmap(ax1, profiles, xl27, settings, norm, lambda v: f"{v:.2f}", title="Mean OOD AUROC by setting (rows) and layer, grouped by position", pad=22)
    for k in range(1, NP):
        ax1.axvline(k * L - 0.5, color="white", lw=3)
    for k, name in enumerate(pos_names):
        ax1.text(k * L + (L - 1) / 2, -0.7, name, ha="center", fontsize=8, color=INK)
    heatmap(ax2, rho, settings, settings, TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), lambda v: f"{v:.2f}",
            title="Spearman ρ of 27-config profiles")
    ax2.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    save_fig(fig, out / "c_robustness.png", d, args, a.size)

    fig, ax = plt.subplots(figsize=(7, 5))
    order = sorted(comps, key=lambda r: r["share"])
    ax.barh([r["term"] for r in order], [r["share"] for r in order], color="#2a78d6", height=0.7)
    for k, r in enumerate(order):
        ax.text(r["share"] + 0.004, k, f"{r['share']:.1%}", va="center", fontsize=7, color=INK)
    style(ax)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("share of total sum of squares in OOD AUROC", fontsize=8, color=MUTED)
    ax.set_title("Variance decomposition: " + " × ".join(f"{f}({n})" for f, n in zip(FACTORS, a.shape)), fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save_fig(fig, out / "variance_decomposition.png", d, args, a.size)

    fig, ax = plt.subplots(figsize=(7, 5))
    style(ax)
    for i, name in enumerate(POSITIONS):
        v, o = val.mean(axis=0)[:, i, :].ravel(), a.mean(axis=(0, 1))[:, i, :].ravel()
        ax.scatter(v, o, s=18, color=POSITION_COLORS[name], edgecolor="white", lw=0.5, label=POSITION_NAMES[name])
    ax.set_xlabel("mean in-distribution validation AUROC (over cells)", fontsize=8, color=MUTED)
    ax.set_ylabel("mean OOD AUROC (over cells × evals)", fontsize=8, color=MUTED)
    ax.set_title(f"Selection criteria, 189 configs: Spearman ρ = {agreement[0]['spearman_rho']:.2f}", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_fig(fig, out / "criterion_agreement.png", d, args, a.size + val.size)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ylabels = [f"{d['cell_of'][q][0]} ({d['cell_of'][q][1]})" for q in d["pairs"]]
    heatmap(ax, per_cell.reshape(P, NP * L), xl27, ylabels, norm, lambda v: f"{v:.2f}",
            title="Mean OOD AUROC per cell (over 15 evals × 7 settings), by layer grouped by position", pad=22)
    for k in range(1, NP):
        ax.axvline(k * L - 0.5, color="white", lw=3)
    for k, name in enumerate(pos_names):
        ax.text(k * L + (L - 1) / 2, -0.7, name, ha="center", fontsize=8, color=INK)
    fig.tight_layout()
    save_fig(fig, out / "per_cell_position_layer.png", d, args, a.size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True, help="CSV with eval_dataset, display, target_label, n.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, required=True)
    args = parser.parse_args()
    d = load(args)
    (args.out_dir / "cells").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    ood = d["auroc"]
    ylim = (np.floor(ood.min() * 20) / 20, np.ceil(ood.max() * 20) / 20)
    for p, pair in enumerate(d["pairs"]):
        slug = re.sub(r"[^a-z0-9]+", "_", d["cell_of"][pair][0].lower()).strip("_")
        plot_cell(d, p, ylim, args.out_dir / "cells" / f"pair{pair:02d}_{slug}.png", args)
    summarize(d, args)
    print(f"{len(d['pairs'])} cell figures and summary tables written to {args.out_dir}")


if __name__ == "__main__":
    main()
