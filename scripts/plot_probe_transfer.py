"""Plot the Llama 3.1 TEFO probe sweep against balanced judged activation caches."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from utils.io import check_counts, write_meta


FIELDS = ("eval_dataset", "family", "display", "target_label", "n", "pair_index", "cell", "pair_type",
          "probe_id", "method", "position", "layer", "C", "val_auroc", "auroc", "balanced_accuracy")
FAMILY_COLORS = {
    "Moral": "#7c3aed", "Are You Sure": "#db2777", "OEQ": "#0284c7",
    "Social sycophancy": "#059669", "SyPR": "#d97706", "TruthfulQA": "#475569",
}
TYPE_COLORS = {"baseline": "#475569", "taxonomy": "#2563eb", "control": "#d97706"}
POSITION_NAMES = {"last_prompt": "last template", "first5": "first 5", "response": "answer mean"}
NORM = TwoSlopeNorm(vmin=0.30, vcenter=0.50, vmax=0.95)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cluster_order(matrix: np.ndarray) -> np.ndarray:
    distance = pdist(matrix, metric="correlation")
    if not np.isfinite(distance).all():
        raise ValueError("cluster profile contains a constant or non-finite row")
    return leaves_list(linkage(distance, method="average", optimal_ordering=True))


def extract_tables(args, targets: list[dict]) -> None:
    manifest_path = args.sweep_dir / "manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.open(encoding="utf-8")]
    if len({r["probe_id"] for r in manifest}) != len(manifest):
        raise ValueError("manifest has duplicate probe IDs")
    by_pair = defaultdict(list)
    for row in manifest:
        by_pair[int(row["pair_index"])].append(row)
    selected = {
        pair: max(rows, key=lambda r: (r["val_auroc"], r["val_accuracy"], r["probe_id"]))
        for pair, rows in by_pair.items()
    }
    if len(selected) != 14:
        raise ValueError(f"expected 14 system prompt cells, found {len(selected)}")
    result_files = sorted((args.sweep_dir / "eval").glob("*.jsonl"))
    if {p.stem for p in result_files} != {t["eval_dataset"] for t in targets}:
        raise ValueError("evaluation files do not match targets.csv")

    val_rows, best_rows = [], []
    for target in targets:
        path = args.sweep_dir / "eval" / f"{target['eval_dataset']}.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        primary = [r for r in rows if r["eval_label"] == target["target_label"]]
        if len(primary) != len(manifest) or {r["probe_id"] for r in primary} != {r["probe_id"] for r in manifest}:
            raise ValueError(f"{path}: target label has missing or duplicate probes")
        if any(r["n"] != int(target["n"]) or r["auroc"] is None or r["balanced_accuracy"] is None for r in primary):
            raise ValueError(f"{path}: undefined metric or unexpected row count")
        by_id = {r["probe_id"]: r for r in primary}
        best = max(primary, key=lambda r: (r["auroc"], r["balanced_accuracy"], r["probe_id"]))
        for result, output in ([(by_id[m["probe_id"]], val_rows) for m in selected.values()] + [(best, best_rows)]):
            probe = next(m for m in manifest if m["probe_id"] == result["probe_id"])
            output.append({
                **target,
                "pair_index": probe["pair_index"], "cell": probe["cell"], "pair_type": probe["pair_type"],
                "probe_id": probe["probe_id"], "method": probe["method"], "position": probe["position"],
                "layer": probe["layer"], "C": "" if probe["C"] is None else probe["C"],
                "val_auroc": probe["val_auroc"], "auroc": result["auroc"],
                "balanced_accuracy": result["balanced_accuracy"],
            })
    val_path, best_path = args.out_dir / "validation_selected.csv", args.out_dir / "best_by_eval_auroc.csv"
    save_csv(val_path, val_rows)
    save_csv(best_path, best_rows)
    n_in = len(targets) * len(manifest)
    inputs = [args.targets, manifest_path, *result_files]
    write_meta(val_path, inputs, args, check_counts(n_in, {"other_probes": n_in - len(val_rows)}, len(val_rows), val_path.name),
               {"selection": "one per pair by validation AUROC, then validation accuracy, then probe ID"})
    write_meta(best_path, inputs, args, check_counts(n_in, {"other_probes": n_in - len(best_rows)}, len(best_rows), best_path.name),
               {"selection": "maximum evaluation AUROC per target; descriptive, post hoc"})


def matrix_from_rows(rows: list[dict], targets: list[dict]):
    pairs = sorted({int(r["pair_index"]) for r in rows})
    by_key = {(int(r["pair_index"]), r["eval_dataset"]): r for r in rows}
    if len(by_key) != len(pairs) * len(targets) or len(by_key) != len(rows):
        raise ValueError("validation-selected table is not a complete pair-by-target grid")
    matrix = np.array([[float(by_key[pair, t["eval_dataset"]]["auroc"]) for t in targets] for pair in pairs])
    pair_info = {int(r["pair_index"]): r for r in rows}
    return pairs, matrix, pair_info


def family_legend(fig, x=0.5, y=0.04):
    handles = [Patch(facecolor=color, label=family) for family, color in FAMILY_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(x, y), ncol=3, frameon=False, fontsize=9)


def save_figure(fig, stem: str, args, inputs: list[Path], n: int) -> None:
    for suffix in ("png", "svg"):
        path = args.out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=200, facecolor="white")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        write_meta(path, inputs, args, check_counts(n, {}, n, path.name),
                   {"model": "meta-llama/Llama-3.1-8B-Instruct", "source_sweep": args.sweep_dir.name})
    plt.close(fig)


def plot_system_prompt_clusters(args, targets, rows):
    pairs, matrix, info = matrix_from_rows(rows, targets)
    order = cluster_order(matrix)
    matrix = matrix[order]
    pair_order = [pairs[i] for i in order]
    fig = plt.figure(figsize=(17, 9))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.2, 15], left=0.30, right=0.92, bottom=0.29, top=0.84, wspace=0.02)
    strip, ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    strip.imshow(np.array([[matplotlib.colors.to_rgb(TYPE_COLORS[info[p]["pair_type"]])] for p in pair_order]), aspect="auto")
    strip.set_xticks([]); strip.set_yticks([])
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu", norm=NORM)
    ax.set_yticks(range(len(pair_order)), [f"P{p:02d}  {info[p]['cell']}" for p in pair_order], fontsize=9)
    ax.set_xticks(range(len(targets)), [t["display"] for t in targets], rotation=55, ha="right", fontsize=9)
    ax.set_xlabel("Balanced evaluation target, grouped by dataset family", labelpad=12)
    ax.set_ylabel("System prompt cell (correlation clustered)")
    ax.set_xticks(np.arange(-0.5, len(targets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pair_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for j in range(1, len(targets)):
        if targets[j]["family"] != targets[j - 1]["family"]:
            ax.axvline(j - 0.5, color="#172033", linewidth=1.7)
    fig.colorbar(im, ax=ax, shrink=0.74, pad=0.02, label="AUROC")
    fig.suptitle("System prompt cells: transfer across judged targets", fontsize=18, fontweight="bold", y=0.95)
    fig.text(0.5, 0.885, "One probe per cell selected on synthetic validation AUROC; rows clustered by correlation of transfer profiles.", ha="center", fontsize=10)
    fig.legend([Patch(facecolor=c, label=k.title()) for k, c in TYPE_COLORS.items()],
               [k.title() for k in TYPE_COLORS], loc="lower center", bbox_to_anchor=(0.55, 0.04), ncol=3, frameon=False)
    save_figure(fig, "01_system_prompt_clusters", args, [args.out_dir / "validation_selected.csv", args.targets], len(rows))


def plot_best_probes(args, rows):
    rows = sorted(rows, key=lambda r: (int(r["pair_index"]), int(r["layer"]), r["position"], r["display"]))
    fig, ax = plt.subplots(figsize=(15.5, 9))
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.12, top=0.86)
    y = np.arange(len(rows))
    auc = np.array([float(r["auroc"]) for r in rows])
    ba = np.array([float(r["balanced_accuracy"]) for r in rows])
    for i in y:
        ax.plot([ba[i], auc[i]], [i, i], color="#b8c2ce", linewidth=1.7, zorder=1)
    ax.scatter(auc, y, color="#1d4ed8", s=42, label="AUROC", zorder=3)
    ax.scatter(ba, y, color="#ea580c", s=42, label="Balanced accuracy", zorder=3)
    ax.axvline(0.5, color="#64748b", linewidth=1, linestyle="--")
    ax.set_yticks(y, [f"P{int(r['pair_index']):02d}  {r['display']}" for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.29, 1.41)
    ax.set_xticks(np.arange(0.3, 1.0, 0.1))
    ax.set_xlabel("Evaluation score", labelpad=10)
    ax.grid(axis="x", color="#e2e8f0")
    ax.spines[["top", "right", "left"]].set_visible(False)
    for i, r in enumerate(rows):
        c = "" if not r["C"] else f"  C={float(r['C']):g}"
        ax.text(1.02, i, f"L{int(r['layer']):02d}  {POSITION_NAMES[r['position']]}  {r['method']}{c}",
                va="center", fontsize=8.4, color="#334155")
        if i and r["pair_index"] != rows[i - 1]["pair_index"]:
            ax.axhline(i - 0.5, color="#cbd5e1", linewidth=0.8)
    ax.text(1.02, -0.8, "Winning probe signature", fontsize=9, fontweight="bold", color="#334155")
    ax.legend(loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(0.68, 1.01))
    fig.suptitle("Best probe by evaluation AUROC", fontsize=18, fontweight="bold", y=0.96)
    fig.text(0.5, 0.905, "Post hoc maximum from 2,646 probes per target; selected scores are optimistic.", ha="center", fontsize=10)
    save_figure(fig, "02_best_probe_clusters", args, [args.out_dir / "best_by_eval_auroc.csv", args.targets], len(rows))


def plot_eval_clusters(args, targets, rows):
    pairs, matrix, info = matrix_from_rows(rows, targets)
    order = cluster_order(matrix.T)
    matrix = matrix.T[order]
    target_order = [targets[i] for i in order]
    selected = {(int(r["pair_index"]), r["eval_dataset"]): r for r in rows}
    med_ba = [np.median([float(selected[p, t["eval_dataset"]]["balanced_accuracy"]) for p in pairs]) for t in target_order]
    med_auc = [float(np.median(matrix[i])) for i in range(len(target_order))]
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.2, 13, 2.2], left=0.22, right=0.92, bottom=0.19, top=0.84, wspace=0.035)
    strip, ax, stats = [fig.add_subplot(grid[0, i]) for i in range(3)]
    strip.imshow(np.array([[matplotlib.colors.to_rgb(FAMILY_COLORS[t["family"]])] for t in target_order]), aspect="auto")
    strip.set_xticks([]); strip.set_yticks([])
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu", norm=NORM)
    ax.set_yticks(range(len(target_order)), [t["display"] for t in target_order], fontsize=9)
    ax.set_xticks(range(len(pairs)), [f"P{p:02d}" for p in pairs], fontsize=9)
    ax.set_xlabel("System prompt cell")
    ax.set_ylabel("Balanced evaluation target (correlation clustered)")
    ax.set_xticks(np.arange(-0.5, len(pairs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(target_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(0.5, color="#172033", linewidth=1.5)
    ax.axvline(8.5, color="#172033", linewidth=1.5)
    stats.set_xlim(0, 1)
    stats.set_ylim(len(target_order) - 0.5, -0.5)
    stats.set_xticks([]); stats.set_yticks([])
    stats.spines[:].set_visible(False)
    for i, (auc, ba) in enumerate(zip(med_auc, med_ba)):
        stats.text(0.02, i, f"{auc:.2f} / {ba:.2f}", va="center", fontsize=9, fontfamily="monospace")
    stats.text(0.02, -0.85, "Median AUC / BA", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=stats, shrink=0.75, pad=0.02, label="AUROC")
    fig.suptitle("Evaluation datasets: similarity of transfer profiles", fontsize=18, fontweight="bold", y=0.95)
    fig.text(0.5, 0.885, "Targets clustered by correlation across 14 validation-selected system prompt probes; medians are over those probes.", ha="center", fontsize=10)
    family_legend(fig, y=0.035)
    save_figure(fig, "03_evaluation_dataset_clusters", args, [args.out_dir / "validation_selected.csv", args.targets], len(rows))


def plot_sypr_focus(args, rows):
    sypr = [r for r in rows if r["eval_dataset"] == "sypr_judged_minority_balanced_seed0"]
    if len(sypr) != 14 or any(r["target_label"] != "sycophantic_praise" or int(r["n"]) != 3792 for r in sypr):
        raise ValueError("SyPR validation-selected rows are incomplete")
    sypr.sort(key=lambda r: float(r["auroc"]), reverse=True)
    colors = []
    for r in sypr:
        pair = int(r["pair_index"])
        colors.append("#7c3aed" if pair in (5, 6) else "#d97706" if pair == 9 else "#64748b")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.16, top=0.79, wspace=0.10)
    y = np.arange(len(sypr))
    labels = [f"P{int(r['pair_index']):02d}  {r['cell']}" for r in sypr]
    for ax, field, title, limit in (
        (axes[0], "auroc", "Ranking: AUROC", (0.24, 0.83)),
        (axes[1], "balanced_accuracy", "Decision at score 0: balanced accuracy", (0.44, 0.66)),
    ):
        values = [float(r[field]) for r in sypr]
        ax.axvline(0.5, color="#475569", linestyle="--", linewidth=1)
        for i, (value, color) in enumerate(zip(values, colors)):
            ax.plot([0.5, value], [i, i], color=color, linewidth=2.1, alpha=0.72)
            ax.scatter(value, i, color=color, s=48, zorder=3)
            ax.annotate(f"{value:.3f}", (value, i), xytext=(5 if value >= 0.5 else -5, 0),
                        textcoords="offset points", va="center", ha="left" if value >= 0.5 else "right", fontsize=8.5)
        ax.set_xlim(*limit)
        ax.set_ylim(len(sypr) - 0.5, -0.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Score; dashed line is chance")
        ax.grid(axis="x", color="#e2e8f0")
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(y, labels, fontsize=9)
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    fig.suptitle("SyPR: warranted praise versus person-traits probes", fontsize=18, fontweight="bold", y=0.96)
    fig.text(0.5, 0.895, "3,792 balanced examples; one probe per system prompt cell selected on synthetic validation AUROC.",
             ha="center", fontsize=10)
    fig.legend(handles=[Patch(facecolor="#7c3aed", label="Person-traits sycophancy"),
                        Patch(facecolor="#d97706", label="Warranted-praise control"),
                        Patch(facecolor="#64748b", label="Other cells")],
               loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=3, frameon=False, fontsize=9)
    fig.text(0.5, 0.025, "P09 contrasts praise only when merited with no compliments; P05/P06 target inflated praise or deference.",
             ha="center", fontsize=9, color="#475569")
    save_figure(fig, "04_sypr_praise_vs_traits", args, [args.out_dir / "validation_selected.csv", args.targets], len(sypr))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--extract", action="store_true", help="Rebuild plot tables from the local per-probe evaluation JSONL files.")
    args = parser.parse_args()
    targets = read_csv(args.targets)
    if len({t["eval_dataset"] for t in targets}) != len(targets):
        raise ValueError("targets.csv has duplicate evaluation datasets")
    if set(t["family"] for t in targets) - set(FAMILY_COLORS):
        raise ValueError("targets.csv has an unstyled dataset family")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.extract:
        extract_tables(args, targets)
    val_rows = read_csv(args.out_dir / "validation_selected.csv")
    best_rows = read_csv(args.out_dir / "best_by_eval_auroc.csv")
    if len(val_rows) != 14 * len(targets) or len(best_rows) != len(targets):
        raise ValueError("plot tables have unexpected row counts")
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.titlesize": 12})
    plot_system_prompt_clusters(args, targets, val_rows)
    plot_best_probes(args, best_rows)
    plot_eval_clusters(args, targets, val_rows)
    plot_sypr_focus(args, val_rows)
    print(f"Wrote 4 PNG and 4 SVG figures to {args.out_dir}")


if __name__ == "__main__":
    main()
