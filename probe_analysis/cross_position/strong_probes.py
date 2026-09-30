"""Position transfer restricted to probe x OOD target combinations where the probe works at its own position.

Reads split_auroc.csv (compute_split.py). A unit is (probe, target, selection fold s). It is selected when the
probe's AUROC at its training position ("home") on fold s is >= a threshold; home, moved (same probe at another
position) and native (probe of the same pair, setting and layer trained at the eval position) are then measured on
the other fold, so selection does not bias them. Moved AUROC on fold s is also kept to show the size of that bias.

Writes <out-dir>/strong_by_direction.csv, strong_by_target.csv and figures.

Run from the repo root: python -m probe_analysis.cross_position.strong_probes --in-dir ... --thresholds 0.7 0.75 --bands early=3,6,10 mid=13,16,19 late=22,26,29 --min-units 20 --dpi 130
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from probe_analysis.cross_position.analyze_cross_position import DIVERGING, GRID, INK, MUTED, POSITIONS, save_csv, setting_of, style
from probe_analysis.cross_position.directions import DIRECTIONS, SHORT, parse_bands
from utils.io import write_meta


def load(path: Path) -> dict:
    """Grid pair x dataset x setting x train position x eval position x layer x fold."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = sorted({int(r["source_pair"]) for r in rows})
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    display = {r["dataset"]: r["display"] for r in rows}
    cs = sorted({float(r["C"]) for r in rows if r["C"] != ""})
    settings = [f"C={c:g}" for c in cs] + ["DIM"]
    layers = sorted({int(r["layer"]) for r in rows})
    a = np.full((len(pairs), len(datasets), len(settings), 3, 3, len(layers), 2), np.nan)
    for r in rows:
        idx = (pairs.index(int(r["source_pair"])), datasets.index(r["dataset"]), settings.index(setting_of(r["C"])),
               POSITIONS.index(r["train_position"]), POSITIONS.index(r["eval_position"]), layers.index(int(r["layer"])),
               int(r["fold"]))
        if not np.isnan(a[idx]):
            raise ValueError(f"duplicate grid entry {idx}")
        a[idx] = float(r["auroc"])
    if len(rows) != a.size or np.isnan(a).any():
        raise ValueError(f"{len(rows)} rows do not fill the {a.shape} grid")
    return dict(a=a, pairs=pairs, datasets=datasets, display=display, settings=settings, layers=layers, n_rows=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=Path, required=True, help="Directory with split_auroc.csv; outputs go here too.")
    parser.add_argument("--thresholds", type=float, nargs="+", required=True, help="Home AUROC thresholds for selection.")
    parser.add_argument("--bands", nargs="+", required=True, help="name=layer,layer,... covering every sweep layer once.")
    parser.add_argument("--min-units", type=int, required=True, help="Per-target cells with fewer selected units are left blank.")
    parser.add_argument("--dpi", type=int, required=True)
    args = parser.parse_args()
    src = args.in_dir / "split_auroc.csv"
    d = load(src)
    a, layers, settings, datasets = d["a"], d["layers"], d["settings"], d["datasets"]
    bands = parse_bands(args.bands, layers)
    if settings[-1] != "DIM" or "DIM" in settings[:-1]:
        raise ValueError(f"expected DIM as the last setting only, got {settings}")
    method_of_setting = np.array(["logistic"] * (len(settings) - 1) + ["DIM"])
    band_of_layer = np.array([next(n for n, b in bands.items() if layer in b) for layer in layers])
    out = args.in_dir
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    def meta(path, n_in, n_out, extra=None):
        write_meta(path, [src], args, {"n_in": n_in, "excluded": {}, "n_out": n_out},
                   {"bands": bands, "unit": "(probe, target, selection fold); measured on the other fold", **(extra or {})})

    # Units per direction t->e: arrays over (pair, dataset, setting, layer, selection fold).
    P, E, S, _, _, L, F = a.shape
    grid = np.meshgrid(np.arange(P), np.arange(E), np.arange(S), np.arange(L), np.arange(F), indexing="ij")
    dataset_of, setting_of_u, layer_of = grid[1].ravel(), grid[2].ravel(), grid[3].ravel()

    def units(t, e):
        sel_home = a[:, :, :, t, t, :, :]                # measured on the selection fold
        held = a[..., ::-1]                               # the other fold at the same index
        return {"sel_home": sel_home.ravel(), "sel_moved": a[:, :, :, t, e, :, :].ravel(),
                "home": held[:, :, :, t, t, :, :].ravel(), "moved": held[:, :, :, t, e, :, :].ravel(),
                "native": held[:, :, :, e, e, :, :].ravel()}

    def summary(u, m, tau):
        n = int(m.sum())
        if n == 0:
            return {"n_units": 0}
        home, moved, native = u["home"][m], u["moved"][m], u["native"][m]
        return {
            "n_units": n, "n_targets": len(set(dataset_of[m].tolist())),
            "sel_fold_home": float(u["sel_home"][m].mean()), "sel_fold_moved": float(u["sel_moved"][m].mean()),
            "held_home": float(home.mean()), "held_moved": float(moved.mean()), "held_native": float(native.mean()),
            "moved_minus_home": float((moved - home).mean()), "moved_minus_native": float((moved - native).mean()),
            "frac_moved_beats_native": float((moved > native).mean()),
            "retention": float((moved.mean() - 0.5) / (home.mean() - 0.5)),
            "frac_home_still_ge_tau": float((home >= tau).mean()), "frac_moved_ge_tau": float((moved >= tau).mean()),
            "frac_moved_below_half": float((moved < 0.5).mean()),
        }

    rows_dir, rows_target = [], []
    for tau in args.thresholds:
        for t, e in DIRECTIONS:
            u = units(t, e)
            chosen = u["sel_home"] >= tau
            base = {"threshold": tau, "train_position": POSITIONS[t], "eval_position": POSITIONS[e]}
            for method in ("all", "logistic", "DIM"):
                for band in ("all", *bands):
                    m = chosen.copy()
                    if method != "all":
                        m &= method_of_setting[setting_of_u] == method
                    if band != "all":
                        m &= band_of_layer[layer_of] == band
                    rows_dir.append({**base, "method": method, "band": band, **summary(u, m, tau)})
            for k, ds in enumerate(datasets):
                m = chosen & (dataset_of == k)
                rows_target.append({**base, "dataset": ds, "display": d["display"][ds], **summary(u, m, tau)})
    fields_dir = list(dict.fromkeys(k for r in rows_dir for k in r))
    save_csv(out / "strong_by_direction.csv", [{k: r.get(k, "") for k in fields_dir} for r in rows_dir])
    meta(out / "strong_by_direction.csv", a.size, len(rows_dir))
    fields_t = list(dict.fromkeys(k for r in rows_target for k in r))
    save_csv(out / "strong_by_target.csv", [{k: r.get(k, "") for k in fields_t} for r in rows_target])
    meta(out / "strong_by_target.csv", a.size, len(rows_target))

    def get(rows, **kw):
        hit = [r for r in rows if all(r[k] == v for k, v in kw.items())]
        if len(hit) != 1:
            raise AssertionError(f"{kw}: {len(hit)} rows")
        return hit[0]

    def save(fig, name):
        for ext in ("png", "svg"):
            path = out / f"{name}.{ext}"
            fig.savefig(path, dpi=args.dpi, facecolor="white")
            meta(path, a.size, 1)
        plt.close(fig)

    labels = [f"{SHORT[POSITIONS[t]]} → {SHORT[POSITIONS[e]]}" for t, e in DIRECTIONS]

    # 01: held-out home, moved and native per direction, one panel per threshold.
    fig, axes = plt.subplots(1, len(args.thresholds), figsize=(5.2 * len(args.thresholds) + 1.5, 4.2), sharey=True, squeeze=False)
    ys = np.arange(len(DIRECTIONS))[::-1]
    for ax, tau in zip(axes[0], args.thresholds):
        style(ax)
        rs = [get(rows_dir, threshold=tau, train_position=POSITIONS[t], eval_position=POSITIONS[e], method="all", band="all") for t, e in DIRECTIONS]
        for y, r in zip(ys, rs):
            ax.plot([r["held_home"], r["held_moved"]], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([r["held_home"] for r in rs], ys, s=48, facecolor="white", edgecolor=MUTED, lw=1.5, zorder=3)
        ax.scatter([r["held_moved"] for r in rs], ys, s=48, color=INK, zorder=4)
        ax.scatter([r["held_native"] for r in rs], ys, s=44, marker="D", facecolor="white", edgecolor=INK, lw=1.2, zorder=3)
        for y, r in zip(ys, rs):
            ax.text(0.505, y - 0.32, f"n = {r['n_units']:,} · {r['n_targets']} targets", fontsize=6.5, color=MUTED)
        ax.axvline(0.5, color=MUTED, lw=0.6)
        ax.axvline(tau, color=MUTED, lw=0.6, ls=":")
        ax.set_xlim(0.5, 0.82)
        ax.set_ylim(-0.7, len(DIRECTIONS) - 0.5)
        ax.set_yticks(ys, labels, fontsize=7.5)
        ax.set_xlabel("held-out AUROC (other fold)", fontsize=8, color=MUTED)
        ax.set_title(f"Probes with home AUROC ≥ {tau:g} on the selection fold", fontsize=9, color=INK, loc="left")
    handles = [Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=MUTED, mew=1.5, label="same probe at home position"),
               Line2D([], [], ls="", marker="o", ms=7, color=INK, label="same probe moved to the eval position"),
               Line2D([], [], ls="", marker="D", ms=6, mfc="white", mec=INK, label="native probe trained at the eval position")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0.02, 0.08, 1, 1))
    save(fig, "01_strong_by_direction")

    # 02: moved - home by method x band, per threshold.
    cols = [(m, b) for m in ("logistic", "DIM") for b in bands]
    fig, axes = plt.subplots(1, len(args.thresholds), figsize=(6.2 * len(args.thresholds), 3.8), sharey=True, squeeze=False)
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
    for ax, tau in zip(axes[0], args.thresholds):
        vals = np.full((len(DIRECTIONS), len(cols)), np.nan)
        ns = np.zeros_like(vals, dtype=int)
        for i, (t, e) in enumerate(DIRECTIONS):
            for j, (m, b) in enumerate(cols):
                r = get(rows_dir, threshold=tau, train_position=POSITIONS[t], eval_position=POSITIONS[e], method=m, band=b)
                ns[i, j] = r["n_units"]
                if r["n_units"] >= args.min_units:
                    vals[i, j] = r["moved_minus_home"]
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=DIVERGING, norm=norm, aspect="auto")
        ax.set_facecolor("#f0efec")
        for (i, j), v in np.ndenumerate(vals):
            if np.isnan(v):
                ax.text(j, i, f"n={ns[i, j]}", ha="center", va="center", fontsize=6, color=MUTED)
                continue
            rgba = im.cmap(im.norm(v))
            light = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.55
            ax.text(j, i, f"{v:+.3f}\nn={ns[i, j]}", ha="center", va="center", fontsize=6, color=INK if light else "white")
        ax.set_xticks(range(len(cols)), [f"{m}\n{b}" for m, b in cols], fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
        ax.tick_params(length=0, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(len(bands) - 0.5, color="white", lw=3)
        ax.set_title(f"Held-out moved − home, home ≥ {tau:g} (blank: n < {args.min_units})", fontsize=9, color=INK, loc="left")
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    save(fig, "02_strong_band_method")

    # 03: per target, moved - home at the lowest threshold.
    tau = min(args.thresholds)
    vals = np.full((len(datasets), len(DIRECTIONS)), np.nan)
    ns = np.zeros_like(vals, dtype=int)
    for i, ds in enumerate(datasets):
        for j, (t, e) in enumerate(DIRECTIONS):
            r = get(rows_target, threshold=tau, train_position=POSITIONS[t], eval_position=POSITIONS[e], dataset=ds)
            ns[i, j] = r["n_units"]
            if r["n_units"] >= args.min_units:
                vals[i, j] = r["moved_minus_home"]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_facecolor("#f0efec")
    for (i, j), v in np.ndenumerate(vals):
        if np.isnan(v):
            if ns[i, j]:
                ax.text(j, i, f"n={ns[i, j]}", ha="center", va="center", fontsize=6, color=MUTED)
            continue
        rgba = im.cmap(im.norm(v))
        light = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.55
        ax.text(j, i, f"{v:+.3f}\nn={ns[i, j]}", ha="center", va="center", fontsize=6, color=INK if light else "white")
    ax.set_xticks(range(len(labels)), [l.replace(" → ", "\n→ ") for l in labels], fontsize=7)
    ax.set_yticks(range(len(datasets)), [d["display"][ds] for ds in datasets], fontsize=7.5)
    ax.tick_params(length=0, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Held-out moved − home per OOD target, home ≥ {tau:g} (blank: n < {args.min_units})", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "03_strong_by_target")
    print(f"{d['n_rows']} fold AUROCs -> {len(rows_dir)} direction rows, {len(rows_target)} target rows in {out}")


if __name__ == "__main__":
    main()
