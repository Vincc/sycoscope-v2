"""Plot in-distribution transfer: every probe's AUROC on every pair's validation rows.

Reads transfer_val.csv from compute_transfer and writes to --out-dir:
  01_grid_<position>.png/.svg   layer x (C, DIM) small multiples of the 14 x 14 source -> target AUROC matrix
  02_matrices_L<layer>.png/.svg annotated 14 x 14 matrices at one layer, logistic AUROC averaged over C
  03_block_summary.png/.svg     sycophancy -> sycophancy and sycophancy -> control mean AUROC by layer
  04_config_sensitivity.png/.svg mean off-diagonal AUROC by layer for each C and DIM
  block_summary.csv             the numbers behind 03 and 04

Run from the repo root: python -m probe_analysis.in_distribution_transfer.plot_transfer --transfer ... --out-dir ...
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from utils.io import check_counts, write_meta

POSITIONS = ("last_prompt", "first5", "response")
POSITION_NAMES = {"last_prompt": "last prompt token", "first5": "first 5 response tokens", "response": "response mean"}
SHORT = {
    "General (baseline)": "Baseline", "Position-Verifiable / Explicit": "Verif / Exp",
    "Position-Verifiable / Implicit": "Verif / Imp", "Position-Subjective / Explicit": "Subj / Exp",
    "Position-Subjective / Implicit": "Subj / Imp", "Person-Traits / Explicit": "Traits / Exp",
    "Person-Traits / Implicit": "Traits / Imp", "Person-Emotions / Explicit": "Emot / Exp",
    "Person-Emotions / Implicit": "Emot / Imp", "Warranted praise": "Ctrl: praise",
    "Genuine agreement": "Ctrl: agreement", "Appropriate emotional support": "Ctrl: support",
    "Calibrated hedging": "Ctrl: hedging", "Ordinary politeness": "Ctrl: politeness",
}
CMAP = LinearSegmentedColormap.from_list(
    "auroc", [(0.0, "#7a1c1c"), (0.25, "#e34948"), (0.5, "#f0efec"), (0.75, "#3987e5"), (1.0, "#0d366b")])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONTROL_COLORS = ("#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")  # categorical slots 2-5, 7
SYCO_COLOR = "#2a78d6"
C_RAMP = ("#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")  # sequential blue, small C -> large C
DIM_COLOR = "#eb6834"


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    layers = sorted({int(r["layer"]) for r in rows})
    Cs = sorted({float(r["C"]) for r in rows if r["method"] == "logistic"})
    configs = [f"C={c:g}" for c in Cs] + ["DIM"]
    pairs = sorted({int(r["source_pair"]) for r in rows})
    if pairs != list(range(len(pairs))) or sorted({int(r["target_pair"]) for r in rows}) != pairs:
        raise ValueError("source and target pairs must both be 0..n-1")
    A = np.full((len(POSITIONS), len(layers), len(configs), len(pairs), len(pairs)), np.nan)
    for r in rows:
        cfg = "DIM" if r["method"] == "dim" else f"C={float(r['C']):g}"
        i = (POSITIONS.index(r["position"]), layers.index(int(r["layer"])), configs.index(cfg),
             int(r["source_pair"]), int(r["target_pair"]))
        if not np.isnan(A[i]):
            raise ValueError(f"duplicate row for {i}")
        A[i] = float(r["auroc"])
    if np.isnan(A).any():
        raise ValueError("transfer grid has missing cells")
    info = {int(r["source_pair"]): (r["source_cell"], r["source_type"]) for r in rows}
    return rows, A, layers, configs, pairs, info


def cell_label(v: float) -> str:
    """Two decimals without the leading zero; values that round to 1.00 print as 1."""
    text = f"{v:.2f}"
    return "1" if text == "1.00" else text[1:]


def separators(ax, info, n):
    """Lines between baseline, taxonomy and control pairs."""
    types = [info[p][1] for p in range(n)]
    for k in range(1, n):
        if types[k] != types[k - 1]:
            ax.axhline(k - 0.5, color=INK, linewidth=0.9)
            ax.axvline(k - 0.5, color=INK, linewidth=0.9)


def save_figure(fig, stem, args, n_cells):
    for suffix in ("png", "svg"):
        path = args.out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=170, facecolor="white")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        write_meta(path, [args.transfer], args, check_counts(n_cells, {}, n_cells, path.name))
    plt.close(fig)


def plot_grid(args, A, layers, configs, pairs, info, p):
    n = len(pairs)
    fig, axes = plt.subplots(len(layers), len(configs), figsize=(2.05 * len(configs) + 2.6, 2.05 * len(layers) + 1.8),
                             squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.90, top=0.93, bottom=0.05, wspace=0.08, hspace=0.10)
    labels = [f"P{k:02d} {SHORT[info[k][0]]}" for k in pairs]
    for li, layer in enumerate(layers):
        for ci, cfg in enumerate(configs):
            ax = axes[li, ci]
            im = ax.imshow(A[p, li, ci], cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
            separators(ax, info, n)
            ax.set_xticks([]); ax.set_yticks([])
            if li == 0:
                ax.set_title(cfg, fontsize=11, color=INK)
            if ci == 0:
                ax.set_ylabel(f"L{layer:02d}", fontsize=11, color=INK, rotation=0, ha="right", va="center")
            if li == len(layers) - 1 and ci == 0:
                ax.set_yticks(range(n), labels, fontsize=5.5, color=MUTED)
                ax.set_xticks(range(n), [f"{k:02d}" for k in pairs], fontsize=5.5, color=MUTED, rotation=90)
    cax = fig.add_axes([0.92, 0.35, 0.012, 0.3])
    fig.colorbar(im, cax=cax, label="AUROC on target validation rows")
    fig.suptitle(f"In-distribution transfer at {POSITION_NAMES[POSITIONS[p]]}", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.515, 0.945, "Each panel: rows = probe's training pair (source), columns = pair whose validation rows are scored (target). "
             "Pair order P00-P13; lines separate baseline | taxonomy | controls. Red < 0.5: probe ranks the target's label-0 side higher.",
             ha="center", fontsize=9.5, color=MUTED, wrap=True)
    save_figure(fig, f"01_grid_{POSITIONS[p]}", args, A[p].size)


def plot_matrices(args, A, layers, configs, pairs, info, layer):
    li = layers.index(layer)
    logistic = [i for i, c in enumerate(configs) if c != "DIM"]
    M = A[:, li][:, logistic].mean(axis=1)
    n = len(pairs)
    fig, axes = plt.subplots(1, len(POSITIONS), figsize=(23, 8.6))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.83, bottom=0.14, wspace=0.08)
    for p, ax in enumerate(axes):
        im = ax.imshow(M[p], cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
        separators(ax, info, n)
        for s in range(n):
            for t in range(n):
                v = M[p, s, t]
                ax.text(t, s, cell_label(v), ha="center", va="center", fontsize=6.6,
                        color="white" if abs(v - 0.5) > 0.3 else INK, fontweight="bold" if s == t else "normal")
        ax.set_xticks(range(n), [f"P{k:02d} {SHORT[info[k][0]]}" for k in pairs], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(n), [f"P{k:02d} {SHORT[info[k][0]]}" for k in pairs] if p == 0 else [], fontsize=8)
        ax.set_title(POSITION_NAMES[POSITIONS[p]], fontsize=13, color=INK, pad=8)
        ax.set_xlabel("Target pair (validation rows scored)", color=MUTED)
    axes[0].set_ylabel("Source pair (probe trained on)", color=MUTED)
    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.015, label="AUROC")
    fig.suptitle(f"Source -> target AUROC at layer {layer}", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.525, 0.895, f"Logistic probes, AUROC averaged over the {len(logistic)} C values "
             "(C changes the off-diagonal mean by <= 0.03 on average). 128 validation rows (64 prompt pairs) per target.",
             ha="center", fontsize=10, color=MUTED)
    save_figure(fig, f"02_matrices_L{layer:02d}", args, M.size)


def block_summary(A, layers, configs, info):
    """Mean AUROC over source -> target cells, per position x layer x config, for each block."""
    syco = [k for k, (_, t) in info.items() if t != "control"]
    controls = [k for k, (_, t) in info.items() if t == "control"]
    blocks = {"sycophancy->sycophancy (off-diagonal)": (syco, syco, True),
              "all off-diagonal": (list(info), list(info), True)}
    for k in controls:
        blocks[f"sycophancy->{info[k][0]}"] = (syco, [k], False)
    out = []
    for p, position in enumerate(POSITIONS):
        for li, layer in enumerate(layers):
            for ci, cfg in enumerate(configs):
                for name, (src, tgt, drop_diag) in blocks.items():
                    sub = A[p, li, ci][np.ix_(src, tgt)]
                    mask = np.ones(sub.shape, dtype=bool)
                    if drop_diag:
                        mask &= np.array([[s != t for t in tgt] for s in src])
                    out.append({"position": position, "layer": layer, "config": cfg, "block": name,
                                "n_cells": int(mask.sum()), "mean_auroc": float(sub[mask].mean())})
    return out


def plot_block_summary(args, summary, layers, info):
    controls = [info[k][0] for k in sorted(info) if info[k][1] == "control"]
    fig, axes = plt.subplots(1, len(POSITIONS), figsize=(19, 6.4), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.74, top=0.82, bottom=0.12, wspace=0.08)
    series = [("sycophancy->sycophancy (off-diagonal)", "Sycophancy -> other sycophancy pairs", SYCO_COLOR, "-")]
    series += [(f"sycophancy->{c}", f"Sycophancy -> control: {c.lower()}", col, "-") for c, col in zip(controls, CONTROL_COLORS)]
    for p, ax in enumerate(axes):
        for block, label, color, style in series:
            y = []
            for layer in layers:
                vals = [r["mean_auroc"] for r in summary if r["position"] == POSITIONS[p] and r["layer"] == layer
                        and r["block"] == block and r["config"] != "DIM"]
                y.append(np.mean(vals))
            ax.plot(layers, y, color=color, linewidth=2, marker="o", markersize=5, label=label, linestyle=style)
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(layers)
        ax.set_xlabel("Layer", color=MUTED)
        ax.set_title(POSITION_NAMES[POSITIONS[p]], fontsize=13, color=INK)
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean AUROC (logistic, averaged over C)", color=MUTED)
    axes[0].text(layers[0], 0.52, "chance", fontsize=8.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.75, 0.5), frameon=False, fontsize=10)
    fig.suptitle("Do sycophancy probes transfer to each other, and do they also separate the control behaviours?",
                 fontsize=15, fontweight="bold", color=INK)
    fig.text(0.40, 0.885, "Sources: baseline + 8 taxonomy probes. A probe specific to sycophancy should sit near 0.5 on every control; "
             "values far from 0.5 either way mean it tracks that correlate.", ha="center", fontsize=9.5, color=MUTED)
    save_figure(fig, "03_block_summary", args, len(summary))


def plot_config_sensitivity(args, summary, layers, configs):
    fig, axes = plt.subplots(1, len(POSITIONS), figsize=(18, 5.8), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.86, top=0.82, bottom=0.13, wspace=0.08)
    colors = dict(zip([c for c in configs if c != "DIM"], C_RAMP))
    colors["DIM"] = DIM_COLOR
    for p, ax in enumerate(axes):
        for cfg in configs:
            y = [next(r["mean_auroc"] for r in summary if r["position"] == POSITIONS[p] and r["layer"] == layer
                      and r["config"] == cfg and r["block"] == "all off-diagonal") for layer in layers]
            ax.plot(layers, y, color=colors[cfg], linewidth=2, marker="o", markersize=5, label=cfg,
                    linestyle="--" if cfg == "DIM" else "-")
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
        ax.set_ylim(0.3, 1.0)
        ax.set_xticks(layers)
        ax.set_xlabel("Layer", color=MUTED)
        ax.set_title(POSITION_NAMES[POSITIONS[p]], fontsize=13, color=INK)
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean off-diagonal AUROC (182 cells)", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.865, 0.5), frameon=False, fontsize=10, title="Probe")
    fig.suptitle("How much do C and probe method change transfer?", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.46, 0.885, "Mean over all 14 x 13 source != target cells, including controls (so anti-transfer lowers the mean).",
             ha="center", fontsize=9.5, color=MUTED)
    save_figure(fig, "04_config_sensitivity", args, len(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transfer", type=Path, required=True, help="transfer_val.csv from compute_transfer.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matrix-layer", type=int, required=True, help="Layer for the annotated matrices (02).")
    args = parser.parse_args()
    rows, A, layers, configs, pairs, info = load(args.transfer)
    if args.matrix_layer not in layers:
        raise ValueError(f"--matrix-layer {args.matrix_layer} not in {layers}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

    summary = block_summary(A, layers, configs, info)
    summary_path = args.out_dir / "block_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    write_meta(summary_path, [args.transfer], args, check_counts(len(rows), {}, len(rows), "transfer rows read"),
               {"n_summary_rows": len(summary)})

    for p in range(len(POSITIONS)):
        plot_grid(args, A, layers, configs, pairs, info, p)
    plot_matrices(args, A, layers, configs, pairs, info, args.matrix_layer)
    plot_block_summary(args, summary, layers, info)
    plot_config_sensitivity(args, summary, layers, configs)
    print(f"Wrote {len(POSITIONS) + 3} figures and block_summary.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
