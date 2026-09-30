"""Compare all six cross-position transfer directions by layer band and probe method (logistic vs DIM).

Reads cross_position.csv (compute_cross_position.py). For each direction train position -> eval position, each
layer (and each band of layers) and each method, reports the OOD AUROC of the moved probe, the same probe at its own
position ("home"), the probe trained at the eval position ("native"), paired differences within
(pair, target, setting, layer) blocks, in-distribution test AUROC, and the cosine between the two positions'
directions. Logistic pools its C values; DIM is its own method.

Writes <out-dir>/directions_by_layer.csv, directions_by_band.csv and figures.

Run from the repo root: python -m probe_analysis.cross_position.directions --in-dir ... --sweep-dir ... --out-dir ... --bands early=3,6,10 mid=13,16,19 late=22,26,29 --dpi 130
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from probe_analysis.cross_position.analyze_cross_position import (DIVERGING, GRID, INK, MUTED, NAMES, POSITIONS, cosines,
                                                                  load, save_csv, style)
from utils.io import write_meta

SHORT = {"last_prompt": "last prompt", "first5": "first 5", "response": "response mean"}
METHOD_COLORS = {"logistic": "#4a3aa7", "DIM": "#e87ba4"}  # not the position hues used in the other figures
METHOD_STYLE = {"logistic": "-", "DIM": "--"}
DIRECTIONS = [(t, e) for t in range(3) for e in range(3) if t != e]
POSITION_PAIRS = list(combinations(range(3), 2))


def parse_bands(specs: list[str], layers: list[int]) -> dict[str, list[int]]:
    bands = {}
    for spec in specs:
        name, _, values = spec.partition("=")
        if not name or not values:
            raise ValueError(f"band {spec!r} is not name=l1,l2,...")
        bands[name] = [int(x) for x in values.split(",")]
    flat = [layer for b in bands.values() for layer in b]
    if sorted(flat) != layers:
        raise ValueError(f"bands {bands} must cover each sweep layer {layers} exactly once")
    return bands


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=Path, required=True, help="Directory with cross_position.csv.")
    parser.add_argument("--sweep-dir", type=Path, required=True, help="Sweep the CSV was computed from (for probe directions).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bands", nargs="+", required=True, help="name=layer,layer,... covering every sweep layer once.")
    parser.add_argument("--dpi", type=int, required=True)
    args = parser.parse_args()
    src = args.in_dir / "cross_position.csv"
    d = load(src)
    a, layers, settings = d["a"], d["layers"], d["settings"]
    bands = parse_bands(args.bands, layers)
    if settings[-1] != "DIM" or "DIM" in settings[:-1]:
        raise ValueError(f"expected DIM as the last setting only, got {settings}")
    methods = {"logistic": list(range(len(settings) - 1)), "DIM": [len(settings) - 1]}
    ood, idt = a[:, :-1], a[:, -1]  # P,E,S,T,V,L and P,S,T,V,L
    cos = cosines(args.sweep_dir, d, [(POSITIONS[x], POSITIONS[y]) for x, y in POSITION_PAIRS])  # P,S,3,L
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    inputs = [src, args.sweep_dir / "manifest.jsonl"]

    def meta(path, n_in, n_out, extra=None):
        write_meta(path, inputs, args, {"n_in": n_in, "excluded": {}, "n_out": n_out}, {"bands": bands, **(extra or {})})

    def stats(t, e, s_idx, j_idx):
        """Summary for direction t->e over settings s_idx and layer indices j_idx."""
        moved = ood[:, :, s_idx][..., t, e, :][..., j_idx]
        home = ood[:, :, s_idx][..., t, t, :][..., j_idx]
        native = ood[:, :, s_idx][..., e, e, :][..., j_idx]
        ind = idt[:, s_idx][..., t, e, :][..., j_idx]
        k = POSITION_PAIRS.index(tuple(sorted((t, e))))
        c = cos[:, s_idx, k][..., j_idx]
        by_c = moved.mean(axis=(0, 1, 3))  # per setting
        return {
            "ood_moved": float(moved.mean()), "ood_home": float(home.mean()), "ood_native": float(native.mean()),
            "ood_diff_vs_home": float((moved - home).mean()), "ood_diff_vs_native": float((moved - native).mean()),
            "ood_frac_beats_native": float((moved > native).mean()),
            "ood_spearman_vs_home": float(spearmanr(moved.ravel(), home.ravel()).statistic),
            "in_dist_mean": float(ind.mean()), "in_dist_min": float(ind.min()),
            "in_dist_native_mean": float(idt[:, s_idx][..., e, e, :][..., j_idx].mean()),
            "cosine_mean": float(c.mean()), "ood_range_over_C": float(by_c.max() - by_c.min()) if len(s_idx) > 1 else "",
            "n_ood_blocks": moved.size, "n_in_dist_probes": ind.size,
        }

    rows_layer, rows_band = [], []
    for t, e in DIRECTIONS:
        for method, s_idx in methods.items():
            base = {"train_position": POSITIONS[t], "eval_position": POSITIONS[e], "method": method}
            for j, layer in enumerate(layers):
                rows_layer.append({**base, "layer": layer, **stats(t, e, s_idx, [j])})
            for name, band in bands.items():
                rows_band.append({**base, "band": name, "layers": " ".join(map(str, band)),
                                  **stats(t, e, s_idx, [layers.index(x) for x in band])})
    save_csv(args.out_dir / "directions_by_layer.csv", rows_layer)
    meta(args.out_dir / "directions_by_layer.csv", a.size, len(rows_layer))
    save_csv(args.out_dir / "directions_by_band.csv", rows_band)
    meta(args.out_dir / "directions_by_band.csv", a.size, len(rows_band))

    def save(fig, name, n_in):
        for ext in ("png", "svg"):
            path = args.out_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=args.dpi, facecolor="white")
            meta(path, n_in, 1)
        plt.close(fig)

    def shade_bands(ax, top_labels):
        for b, (name, band) in enumerate(bands.items()):
            if b % 2 == 0:
                ax.axvspan(band[0] - 1.2, band[-1] + 1.2, color=GRID, alpha=0.35, lw=0)
            if top_labels:
                ax.text((band[0] + band[-1]) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=7, color=MUTED)

    # 01: band summary heatmaps, rows = directions, columns = method x band.
    col_labels = [f"{m}\n{b}" for m in methods for b in bands]
    row_labels = [f"{SHORT[POSITIONS[t]]} → {SHORT[POSITIONS[e]]}" for t, e in DIRECTIONS]
    by_key = {(r["train_position"], r["eval_position"], r["method"], r["band"]): r for r in rows_band}

    def grid_of(field):
        return np.array([[by_key[(POSITIONS[t], POSITIONS[e], m, b)][field] for m in methods for b in bands]
                         for t, e in DIRECTIONS])

    # In-distribution values all sit in [0.85, 1]; a sequential scale there keeps early-layer drops visible.
    panels = [("ood_moved", "OOD AUROC of the moved probe", DIVERGING, TwoSlopeNorm(vmin=0.44, vcenter=0.5, vmax=0.64), "{:.3f}"),
              ("ood_diff_vs_native", "OOD: moved − native probe at the eval position", DIVERGING, TwoSlopeNorm(vmin=-0.07, vcenter=0, vmax=0.07), "{:+.3f}"),
              ("in_dist_mean", "In-distribution test AUROC of the moved probe", "Blues", Normalize(vmin=0.85, vmax=1.0), "{:.3f}")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, (field, title, cmap, norm, fmt) in zip(axes, panels):
        values = grid_of(field)
        if field == "in_dist_mean" and values.min() < norm.vmin:
            raise ValueError(f"in-distribution value {values.min():.3f} below the colour scale floor {norm.vmin}")
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(col_labels)), col_labels, fontsize=7)
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7.5)
        ax.tick_params(length=0, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(len(bands) - 0.5, color="white", lw=3)
        for (i, j), v in np.ndenumerate(values):
            rgba = im.cmap(im.norm(v))
            light = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color=INK if light else "white")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("trained at → scored at", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "01_band_summary", a.size)

    # 02 / 03: per-layer small multiples, rows = trained at, columns = scored at.
    def small_multiples(name, title, values_of, native_of, ylim, chance):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
        for t in range(3):
            for e in range(3):
                ax = axes[t, e]
                style(ax)
                shade_bands(ax, top_labels=(t == 0))
                if chance:
                    ax.axhline(0.5, color=MUTED, lw=0.6)
                for method, s_idx in methods.items():
                    ax.plot(layers, values_of(t, e, s_idx), color=METHOD_COLORS[method], ls=METHOD_STYLE[method], lw=2,
                            marker="o", ms=3.5)
                    if t != e:
                        ax.plot(layers, native_of(e, s_idx), color=MUTED, ls=METHOD_STYLE[method], lw=1)
                ax.set_ylim(*ylim)
                ax.set_xticks(layers)
                label = f"trained {SHORT[POSITIONS[t]]} → scored {SHORT[POSITIONS[e]]}" if t != e else f"{SHORT[POSITIONS[t]]} (native)"
                ax.set_title(label, fontsize=8, color=INK, loc="left", pad=12)
                if t == 2:
                    ax.set_xlabel("layer", fontsize=8, color=MUTED)
        handles = [Line2D([], [], color=METHOD_COLORS[m], ls=METHOD_STYLE[m], lw=2, marker="o", ms=3.5, label=f"{m}, moved probe") for m in methods]
        handles += [Line2D([], [], color=MUTED, ls=METHOD_STYLE[m], lw=1, label=f"{m}, native probe at the eval position") for m in methods]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        save(fig, name, a.size)

    small_multiples("02_ood_by_layer", "OOD mean AUROC by layer (14 pairs × 15 targets; logistic pooled over 6 C). Shaded: layer bands.",
                    lambda t, e, s: ood[:, :, s][..., t, e, :].mean(axis=(0, 1, 2)),
                    lambda e, s: ood[:, :, s][..., e, e, :].mean(axis=(0, 1, 2)), (0.48, 0.62), True)
    small_multiples("03_in_dist_by_layer", "In-distribution test AUROC by layer, own pair (mean over 14 pairs; logistic pooled over 6 C)",
                    lambda t, e, s: idt[:, s][..., t, e, :].mean(axis=(0, 1)),
                    lambda e, s: idt[:, s][..., e, e, :].mean(axis=(0, 1)), (0.65, 1.01), False)

    # 04: direction cosine by layer for each position pair.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (k, (x, y)) in zip(axes, enumerate(POSITION_PAIRS)):
        style(ax)
        shade_bands(ax, top_labels=True)
        for method, s_idx in methods.items():
            ax.plot(layers, cos[:, s_idx, k].mean(axis=(0, 1)), color=METHOD_COLORS[method], ls=METHOD_STYLE[method],
                    lw=2, marker="o", ms=3.5, label=method)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(layers)
        ax.set_xlabel("layer", fontsize=8, color=MUTED)
        ax.set_title(f"{SHORT[POSITIONS[x]]} vs {SHORT[POSITIONS[y]]}", fontsize=9, color=INK, loc="left", pad=12)
    axes[0].set_ylabel("cosine (same pair, setting, layer)", fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "04_direction_cosine", cos.size)
    print(f"{len(rows_layer)} layer rows and {len(rows_band)} band rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
