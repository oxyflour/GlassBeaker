from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a compact baseline report from analysis artifacts.")
    parser.add_argument("--analysis-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--compare-metrics", action="append", default=[])
    parser.add_argument("--sample-plot", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _parse_kv(items: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in items:
        key, value = item.split("=", 1)
        pairs.append((key, value))
    return pairs


def _rel_to_output(raw_path: str, output_dir: Path) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(output_dir).as_posix()


def _save_overview(path: Path, analysis: dict[str, object], summary: dict[str, object]) -> None:
    pair_db = analysis["pair_db_mae"]
    threshold = analysis["threshold_metrics"]
    cut_stats = analysis["by_cut_count"]
    sample_db = np.asarray(summary["sample_db_mae"], dtype=np.float32)
    pairs = list(pair_db.keys())
    pair_vals = [pair_db[key] for key in pairs]
    pair_colors = ["#d97706" if key[1] == key[2] else "#2563eb" for key in pairs]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(pairs, pair_vals, color=pair_colors)
    axes[0, 0].set_title("Pair dB MAE")
    axes[0, 0].set_ylabel("dB")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 1].bar(
        [">= -10 dB", "< -10 dB", "< -20 dB"],
        [
            threshold["truth_db_ge_neg10_mae"],
            threshold["truth_db_lt_neg10_mae"],
            threshold["truth_db_lt_neg20_mae"],
        ],
        color=["#16a34a", "#dc2626", "#7c3aed"],
    )
    axes[0, 1].set_title("Region dB MAE")
    axes[0, 1].set_ylabel("dB")
    axes[0, 1].grid(axis="y", alpha=0.25)
    cut_keys = sorted(cut_stats.keys(), key=int)
    axes[1, 0].bar(cut_keys, [cut_stats[key]["mean_db_mae"] for key in cut_keys], color="#0f766e")
    axes[1, 0].set_title("Mean dB MAE by Cut Count")
    axes[1, 0].set_xlabel("cut count")
    axes[1, 0].set_ylabel("dB")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 1].hist(sample_db, bins=30, color="#475569", edgecolor="white")
    for val, label, color in [
        (float(np.median(sample_db)), "median", "#2563eb"),
        (float(np.quantile(sample_db, 0.9)), "p90", "#dc2626"),
    ]:
        axes[1, 1].axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"{label} {val:.2f}")
    axes[1, 1].set_title("Validation Sample dB MAE Distribution")
    axes[1, 1].set_xlabel("dB MAE")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)
    fig.suptitle("Nijika Baseline Report Overview")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_comparison(path: Path, current: tuple[str, dict[str, object]], others: list[tuple[str, dict[str, object]]]) -> None:
    runs = others + [current]
    labels = [label for label, _ in runs]
    db_mae = [float(item["val_db_mae"]) for _, item in runs]
    rmse = [float(item["val_rmse"]) for _, item in runs]
    db_rmse = [float(item["val_db_rmse"]) for _, item in runs]
    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, vals, title, color, offset in [
        (axes[0], db_mae, "Validation dB MAE", "#2563eb", -width),
        (axes[1], rmse, "Validation RMSE", "#0f766e", 0.0),
        (axes[2], db_rmse, "Validation dB RMSE", "#dc2626", width),
    ]:
        axis.bar(x + offset, vals, width=width, color=color)
        axis.set_xticks(x, labels, rotation=15)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Run Comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    metrics: dict[str, object],
    analysis: dict[str, object],
    overview_name: str,
    comparison_name: str,
    sample_plots: list[tuple[str, str]],
) -> None:
    summary = analysis["summary"]
    pair_db = analysis["pair_db_mae"]
    reflection = (pair_db["S11"] + pair_db["S22"] + pair_db["S33"]) / 3.0
    coupling = (pair_db["S12"] + pair_db["S13"] + pair_db["S21"] + pair_db["S23"] + pair_db["S31"] + pair_db["S32"]) / 6.0
    threshold = analysis["threshold_metrics"]
    lines = [
        "# Nijika Report",
        "",
        f"- best epoch: `{metrics['best_epoch']}`",
        f"- validation RMSE: `{metrics['val_rmse']:.5f}`",
        f"- validation dB MAE: `{metrics['val_db_mae']:.5f} dB`",
        f"- validation dB RMSE: `{metrics['val_db_rmse']:.5f} dB`",
        f"- validation nib-side patterns: `{len(analysis['unique_nib_side_patterns'])}`",
        "",
        f"![overview]({overview_name})",
        "",
        f"![comparison]({comparison_name})",
        "",
        "## Key Reads",
        "",
        f"- sample dB MAE median / p90 / max = `{summary['median_db_mae']:.2f} / {summary['p90_db_mae']:.2f} / {summary['max_db_mae']:.2f} dB`",
        f"- reflection vs coupling dB MAE = `{reflection:.2f}` vs `{coupling:.2f} dB`",
        f"- high-mag vs notch-region dB MAE = `{threshold['truth_db_ge_neg10_mae']:.2f}` vs `{threshold['truth_db_lt_neg20_mae']:.2f} dB`",
        "",
        "## Sample Plots",
        "",
    ]
    for label, rel_path in sample_plots:
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"![{label}]({rel_path})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = _load_json(args.analysis_path)
    summary = _load_json(args.summary_path)
    metrics = _load_json(args.metrics_path)
    overview_path = args.output_dir / "overview.png"
    comparison_path = args.output_dir / "comparison.png"
    _save_overview(overview_path, analysis, summary)
    compare = [(label, _load_json(Path(path))) for label, path in _parse_kv(args.compare_metrics)]
    _save_comparison(comparison_path, ("current", metrics), compare)
    sample_plots = []
    for label, raw_path in _parse_kv(args.sample_plot):
        sample_plots.append((label, _rel_to_output(raw_path, args.output_dir)))
    _write_report(
        args.output_dir / "report.md",
        metrics,
        analysis,
        overview_path.name,
        comparison_path.name,
        sample_plots,
    )
    print(json.dumps({"report": str(args.output_dir / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
