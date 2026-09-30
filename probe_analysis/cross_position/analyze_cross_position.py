"""Summarize cross-position transfer: how last-prompt probes score at the first-5 and response-mean positions.

Reads cross_position.csv (compute_cross_position.py) into a complete grid
pair x dataset x setting x train position x eval position x layer and writes summary tables and figures.
Also reports the cosine between each last-prompt probe direction and the probe of the same pair, setting and
layer trained at first5 or response. --layer picks the layer for the per-dataset and per-cell breakdowns.

Run from the repo root: python -m probe_analysis.cross_position.analyze_cross_position --in-dir ... --sweep-dir ... --layer 13 --dpi 130
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from utils import probes
from utils.io import read_jsonl, write_meta

POSITIONS = ("last_prompt", "first5", "response")
NAMES = {"last_prompt": "last prompt token", "first5": "first 5 response tokens", "response": "response mean"}
COLORS = {"last_prompt": "#2a78d6", "first5": "#eb6834", "response": "#1baf7a"}
IN_DIST = "contrastive_test"
DIVERGING = LinearSegmentedColormap.from_list("auroc", ["#b3261e", "#e34948", "#f0efec", "#2a78d6", "#104281"])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdad5"
LP = 0  # index of last_prompt in POSITIONS


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def setting_of(c: str) -> str:
    return "DIM" if c == "" else f"C={float(c):g}"


def load(path: Path) -> dict:
    rows = read_csv(path)
    pairs = sorted({int(r["source_pair"]) for r in rows})
    datasets = list(dict.fromkeys(r["dataset"] for r in rows if r["dataset"] != IN_DIST)) + [IN_DIST]
    display = {r["dataset"]: r["display"] for r in rows}
    cs = sorted({float(r["C"]) for r in rows if r["C"] != ""})
    settings = [f"C={c:g}" for c in cs] + ["DIM"]
    layers = sorted({int(r["layer"]) for r in rows})
    cell_of = {int(r["source_pair"]): (r["cell"], r["pair_type"]) for r in rows}
    shape = (len(pairs), len(datasets), len(settings), len(POSITIONS), len(POSITIONS), len(layers))
    a = np.full(shape, np.nan)
    for r in rows:
        idx = (pairs.index(int(r["source_pair"])), datasets.index(r["dataset"]), settings.index(setting_of(r["C"])),
               POSITIONS.index(r["train_position"]), POSITIONS.index(r["eval_position"]), layers.index(int(r["layer"])))
        if not np.isnan(a[idx]):
            raise ValueError(f"duplicate grid entry {idx}")
        a[idx] = float(r["auroc"])
    if len(rows) != a.size or np.isnan(a).any():
        raise ValueError(f"{len(rows)} rows do not fill the {shape} grid")
    return dict(a=a, pairs=pairs, datasets=datasets, display=display, settings=settings, layers=layers,
                cell_of=cell_of, n_rows=len(rows))


def cosines(sweep_dir: Path, d: dict, position_pairs: list[tuple[str, str]]) -> np.ndarray:
    """cos(direction at a, direction at b) per pair x setting x position pair (a, b) x layer."""
    manifest = read_jsonl(sweep_dir / "manifest.jsonl")
    files = {f: np.load(sweep_dir / f) for f in sorted({m["file"] for m in manifest})}
    direction = {}
    for m in manifest:
        key = (int(m["pair_index"]), "DIM" if m["C"] is None else f"C={m['C']:g}", m["position"], int(m["layer"]))
        direction[key] = probes.load_probe(files[m["file"]], m["probe_id"], m["method"])["direction"]
    out = np.full((len(d["pairs"]), len(d["settings"]), len(position_pairs), len(d["layers"])), np.nan)
    for p, pair in enumerate(d["pairs"]):
        for s, setting in enumerate(d["settings"]):
            for k, (pa, pb) in enumerate(position_pairs):
                for j, layer in enumerate(d["layers"]):
                    u, v = direction[(pair, setting, pa, layer)], direction[(pair, setting, pb, layer)]
                    out[p, s, k, j] = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    if np.isnan(out).any() or len(direction) != len(manifest):
        raise AssertionError("cosine grid incomplete or manifest has duplicate keys")
    return out


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def heatmap(ax, values, xlabels, ylabels, norm, title):
    im = ax.imshow(values, cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=7)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.tick_params(length=0, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i, j), v in np.ndenumerate(values):
        rgba = im.cmap(im.norm(v))
        light = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.55
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=INK if light else "white")
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=Path, required=True, help="Directory with cross_position.csv; outputs go here too.")
    parser.add_argument("--sweep-dir", type=Path, required=True, help="Sweep the CSV was computed from (for probe directions).")
    parser.add_argument("--layer", type=int, required=True, help="Layer for per-dataset and per-cell breakdowns.")
    parser.add_argument("--dpi", type=int, required=True)
    args = parser.parse_args()
    src = args.in_dir / "cross_position.csv"
    d = load(src)
    a, layers, settings, datasets = d["a"], d["layers"], d["settings"], d["datasets"]
    if args.layer not in layers:
        raise ValueError(f"layer {args.layer} not in {layers}")
    L = layers.index(args.layer)
    ood, idt = a[:, :-1], a[:, -1]  # P,E,S,T,V,L and P,S,T,V,L
    n_ood = ood.shape[1]
    out = args.in_dir
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    inputs = [src, args.sweep_dir / "manifest.jsonl"]

    def meta(path, n_in, n_out, extra=None):
        write_meta(path, inputs, args, {"n_in": n_in, "excluded": {}, "n_out": n_out}, extra)

    # 1. Train position x eval position x layer.
    rows = []
    for t, tp in enumerate(POSITIONS):
        for v, ep in enumerate(POSITIONS):
            for j, layer in enumerate(layers):
                x = ood[:, :, :, t, v, j]
                rows.append({"train_position": tp, "eval_position": ep, "layer": layer,
                             "ood_mean_auroc": float(x.mean()), "ood_median_auroc": float(np.median(x)),
                             "ood_frac_below_half": float((x < 0.5).mean()), "ood_n_scores": x.size,
                             "in_dist_test_mean_auroc": float(idt[:, :, t, v, j].mean()),
                             "in_dist_test_min_auroc": float(idt[:, :, t, v, j].min()), "in_dist_n_scores": idt[:, :, t, v, j].size})
    save_csv(out / "transfer_matrix.csv", rows)
    meta(out / "transfer_matrix.csv", a.size, len(rows), {"pooled_over": "14 pairs x 7 settings (x 15 OOD targets for ood_*)"})

    # 2. Last-prompt probes: paired comparisons per layer, within (pair, dataset, setting).
    rows = []
    for v in (1, 2):
        ep = POSITIONS[v]
        for j, layer in enumerate(layers):
            lp_here, lp_home, native = ood[:, :, :, LP, v, j], ood[:, :, :, LP, LP, j], ood[:, :, :, v, v, j]
            rows.append({
                "eval_position": ep, "layer": layer,
                "ood_lp_probe_here": float(lp_here.mean()), "ood_lp_probe_at_last_prompt": float(lp_home.mean()),
                "ood_native_probe_here": float(native.mean()),
                "ood_diff_vs_lp_at_last_prompt": float((lp_here - lp_home).mean()),
                "ood_frac_better_than_lp_at_last_prompt": float((lp_here > lp_home).mean()),
                "ood_diff_vs_native": float((lp_here - native).mean()),
                "ood_frac_better_than_native": float((lp_here > native).mean()),
                "ood_frac_side_of_half_flips_vs_lp_at_last_prompt": float(((lp_here - 0.5) * (lp_home - 0.5) < 0).mean()),
                "ood_spearman_vs_lp_at_last_prompt": float(spearmanr(lp_here.ravel(), lp_home.ravel()).statistic),
                "ood_spearman_vs_native": float(spearmanr(lp_here.ravel(), native.ravel()).statistic),
                "in_dist_lp_probe_here": float(idt[:, :, LP, v, j].mean()),
                "in_dist_lp_probe_here_min": float(idt[:, :, LP, v, j].min()),
                "in_dist_native_probe_here": float(idt[:, :, v, v, j].mean()),
                "n_blocks": lp_here.size,
            })
    save_csv(out / "last_prompt_paired.csv", rows)
    meta(out / "last_prompt_paired.csv", a.size, len(rows),
         {"block": "(pair, OOD target, setting); diffs and fractions are within block, then averaged",
          "native": "probe of the same pair, setting and layer trained at the eval position"})

    # 3. Per dataset at --layer (mean over pairs and settings).
    combos = [(t, v) for t in range(3) for v in range(3)]
    rows = [{"dataset": ds, "display": d["display"][ds], "layer": args.layer,
             **{f"{POSITIONS[t]}->{POSITIONS[v]}": float(a[:, e, :, t, v, L].mean()) for t, v in combos}}
            for e, ds in enumerate(datasets)]
    save_csv(out / f"per_dataset_L{args.layer}.csv", rows)
    meta(out / f"per_dataset_L{args.layer}.csv", a.size, len(rows), {"columns": "train_position->eval_position, mean over pairs x settings"})

    # 4. Per cell at --layer (mean over OOD targets and settings; in-distribution separately).
    rows = []
    for p, pair in enumerate(d["pairs"]):
        cell, pair_type = d["cell_of"][pair]
        r = {"pair_index": pair, "cell": cell, "pair_type": pair_type, "layer": args.layer}
        for t, v in combos:
            r[f"ood {POSITIONS[t]}->{POSITIONS[v]}"] = float(ood[p, :, :, t, v, L].mean())
        for t, v in combos:
            r[f"in_dist {POSITIONS[t]}->{POSITIONS[v]}"] = float(idt[p, :, t, v, L].mean())
        rows.append(r)
    save_csv(out / f"per_cell_L{args.layer}.csv", rows)
    meta(out / f"per_cell_L{args.layer}.csv", a.size, len(rows))

    # 5. Direction cosines.
    cos = cosines(args.sweep_dir, d, [("last_prompt", pos) for pos in POSITIONS[1:]])  # P,S,2,L
    rows = []
    for k, pos in enumerate(POSITIONS[1:]):
        for j, layer in enumerate(layers):
            for s, setting in enumerate(settings):
                x = cos[:, s, k, j]
                rows.append({"other_position": pos, "layer": layer, "setting": setting, "mean_cosine": float(x.mean()),
                             "min_cosine": float(x.min()), "max_cosine": float(x.max()), "n_pairs": x.size})
    save_csv(out / "direction_cosine.csv", rows)
    meta(out / "direction_cosine.csv", cos.size, len(rows), {"direction": "stored unit direction (logistic: coef/scale normalized; DIM: class-mean difference)"})

    # Figures.
    def save(fig, name, n_in):
        for ext in ("png", "svg"):
            path = out / f"{name}.{ext}"
            fig.savefig(path, dpi=args.dpi, facecolor="white")
            meta(path, n_in, 1)
        plt.close(fig)

    # 01: last-prompt probes by eval position across layers, OOD and in-distribution.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title in ((axes[0], ood.mean(axis=(0, 1, 2)), f"OOD: mean AUROC over 14 pairs × {n_ood} targets × 7 settings"),
                            (axes[1], idt.mean(axis=(0, 1)), "In-distribution test (own pair): mean over 14 pairs × 7 settings")):
        style(ax)
        for v, pos in enumerate(POSITIONS):
            ax.plot(layers, data[LP, v], color=COLORS[pos], lw=2, marker="o", ms=4)
            if v:
                ax.plot(layers, data[v, v], color=COLORS[pos], lw=1.2, ls="--")
        ax.set_xticks(layers)
        ax.set_xlabel("layer", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].axhline(0.5, color=MUTED, lw=0.6)
    axes[0].set_ylabel("AUROC", fontsize=8, color=MUTED)
    handles = [Line2D([], [], color=COLORS[p], lw=2, marker="o", ms=4, label=f"last-prompt probe on {NAMES[p]}") for p in POSITIONS]
    handles += [Line2D([], [], color=COLORS[p], lw=1.2, ls="--", label=f"native {NAMES[p]} probe (reference)") for p in POSITIONS[1:]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "01_last_prompt_by_layer", a.size)

    # 02: 3x3 train x eval matrices at --layer.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    labels = [NAMES[p] for p in POSITIONS]
    heatmap(axes[0], ood[:, :, :, :, :, L].mean(axis=(0, 1, 2)), labels, labels, TwoSlopeNorm(vmin=0.4, vcenter=0.5, vmax=0.65),
            f"OOD mean AUROC, layer {args.layer} (rows: trained at, columns: scored at)")
    heatmap(axes[1], idt[:, :, :, :, L].mean(axis=(0, 1)), labels, labels, TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
            f"In-distribution test mean AUROC, layer {args.layer}")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.5)
    fig.tight_layout()
    save(fig, f"02_matrix_L{args.layer}", a.size)

    # 03: per OOD target at --layer.
    fig, ax = plt.subplots(figsize=(8, 6))
    style(ax)
    ys = np.arange(n_ood)[::-1]
    series = [(LP, LP, "last-prompt probe on last prompt token", COLORS["last_prompt"], "o"),
              (LP, 1, "last-prompt probe on first 5 tokens", COLORS["first5"], "o"),
              (LP, 2, "last-prompt probe on response mean", COLORS["response"], "o"),
              (2, 2, "native response-mean probe", COLORS["response"], "D")]
    for t, v, label, color, marker in series:
        vals = ood[:, :, :, t, v, L].mean(axis=(0, 2))
        ax.scatter(vals, ys, s=36, color=color if marker == "o" else "white", edgecolor=color, lw=1.5 if marker == "D" else 0.8,
                   marker=marker, label=label, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=0.6)
    ax.set_yticks(ys, [d["display"][ds] for ds in datasets[:-1]], fontsize=7)
    ax.set_xlabel("mean AUROC over 14 pairs × 7 settings", fontsize=8, color=MUTED)
    ax.set_title(f"Per OOD target, layer {args.layer}", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, f"03_per_target_L{args.layer}", a.size)

    # 04: same probe, same row: AUROC at last prompt vs at response / first5, at --layer.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    home = ood[:, :, :, LP, LP, L].ravel()
    for ax, v in zip(axes, (1, 2)):
        style(ax)
        here = ood[:, :, :, LP, v, L].ravel()
        ax.scatter(home, here, s=6, color=COLORS[POSITIONS[v]], alpha=0.35, lw=0)
        lo, hi = min(home.min(), here.min()) - 0.02, max(home.max(), here.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
        ax.axhline(0.5, color=GRID, lw=0.8)
        ax.axvline(0.5, color=GRID, lw=0.8)
        rho = spearmanr(home, here).statistic
        ax.set_title(f"scored on {NAMES[POSITIONS[v]]}: Spearman ρ = {rho:.2f}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("same probe scored on last prompt token", fontsize=8, color=MUTED)
    axes[0].set_ylabel("OOD AUROC at the other position", fontsize=8, color=MUTED)
    fig.suptitle(f"Last-prompt probes, layer {args.layer}: one dot per pair × OOD target × setting ({home.size})",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, f"04_same_probe_scatter_L{args.layer}", a.size)

    # 05: direction cosine by layer.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    style(ax)
    for k, pos in enumerate(POSITIONS[1:]):
        logistic, dim = cos[:, :-1, k].mean(axis=(0, 1)), cos[:, -1, k].mean(axis=0)
        ax.plot(layers, logistic, color=COLORS[pos], lw=2, marker="o", ms=4, label=f"vs {NAMES[pos]} probe, logistic (mean over C)")
        ax.plot(layers, dim, color=COLORS[pos], lw=1.2, ls="--", label=f"vs {NAMES[pos]} probe, DIM")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(layers)
    ax.set_xlabel("layer", fontsize=8, color=MUTED)
    ax.set_ylabel("cosine", fontsize=8, color=MUTED)
    ax.set_title("Cosine of last-prompt probe direction with the same pair's probe at another position", fontsize=8.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "05_direction_cosine", cos.size)
    print(f"{d['n_rows']} scores summarized -> {out}")


if __name__ == "__main__":
    main()
