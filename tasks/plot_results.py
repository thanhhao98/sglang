#!/usr/bin/env python3
"""
Generate performance charts from CSV results.

Usage:
    python tasks/plot_results.py --csv tasks/results.csv --output-dir tasks/charts/
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
CONTEXTS = ["128K", "256K", "512K", "1M"]

CONFIG_LABELS = {
    "C1": "TP8",
    "C2": "DCP ag_rs (CG)",
    "C5": "DCP a2a+repl (CG)",
}

COLORS = {
    "C1": "#2196F3",
    "C2": "#FF9800",
    "C5": "#E91E63",
}

MARKERS = {
    "C1": "o",
    "C2": "s",
    "C5": "D",
}


def load_csv(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["concurrency"] = int(row["concurrency"])
            for key in row:
                if key not in ("commit", "machine", "config", "context"):
                    try:
                        row[key] = float(row[key]) if row[key] else None
                    except ValueError:
                        row[key] = None
            rows.append(row)
    return rows


def group_data(rows: list[dict]) -> dict:
    """Group rows into nested dict: data[machine][config][context][concurrency] = row"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        m, cfg, ctx, conc = row["machine"], row["config"], row["context"], row["concurrency"]
        data[m][cfg][ctx][conc] = row
    return data


def get_series(data, machine, config, context, metric):
    """Extract a list of metric values across concurrencies for a given config/context."""
    vals = []
    for c in CONCURRENCIES:
        row = data.get(machine, {}).get(config, {}).get(context, {}).get(c)
        vals.append(row[metric] if row and row.get(metric) is not None else None)
    return vals


def plot_scaling(data, machine, metric, ylabel, title_suffix, filename, output_dir, log_y=False):
    """2x2 subplot: one per context, lines per config."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ctx in enumerate(CONTEXTS):
        ax = axes[idx]
        for cfg in ["C1", "C2", "C5"]:
            vals = get_series(data, machine, cfg, ctx, metric)
            if all(v is None for v in vals):
                continue
            # Replace None with NaN for plotting
            plot_vals = [v if v is not None else float("nan") for v in vals]
            label = CONFIG_LABELS.get(cfg, cfg)
            ax.plot(CONCURRENCIES, plot_vals,
                    marker=MARKERS[cfg], color=COLORS[cfg], label=label,
                    linewidth=2, markersize=5)

        ax.set_xlabel("Concurrency")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{machine.upper()}: {ctx} {title_suffix}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONCURRENCIES)
        ax.set_xticklabels([str(c) for c in CONCURRENCIES], fontsize=7)
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_c1_bars(data, machine, output_dir):
    """Bar charts at concurrency=1: TPOT and throughput side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    configs = ["C1", "C2", "C5"]
    x = np.arange(len(CONTEXTS))
    width = 0.25

    # TPOT bars
    for i, cfg in enumerate(configs):
        vals = []
        for ctx in CONTEXTS:
            row = data.get(machine, {}).get(cfg, {}).get(ctx, {}).get(1)
            vals.append(row["mean_tpot_ms"] if row and row.get("mean_tpot_ms") else 0)
        bars = ax1.bar(x + i * width, vals, width,
                       label=CONFIG_LABELS[cfg], color=COLORS[cfg])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("TPOT (ms)")
    ax1.set_title(f"{machine.upper()}: c=1 Decode TPOT — Lower is Better")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(CONTEXTS)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Throughput bars
    for i, cfg in enumerate(configs):
        vals = []
        for ctx in CONTEXTS:
            row = data.get(machine, {}).get(cfg, {}).get(ctx, {}).get(1)
            v = row["total_throughput"] if row and row.get("total_throughput") else 0
            vals.append(v / 1000)
        bars = ax2.bar(x + i * width, vals, width,
                       label=CONFIG_LABELS[cfg], color=COLORS[cfg])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f"{val:.0f}K", ha="center", va="bottom", fontsize=7)

    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Throughput (K tok/s)")
    ax2.set_title(f"{machine.upper()}: c=1 Throughput — Higher is Better")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(CONTEXTS)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / f"chart_{machine}_c1_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_cross_machine(data, machines, output_dir):
    """H100 vs B200 throughput comparison per config."""
    if len(machines) < 2:
        print("Skipping cross-machine chart (need data from 2+ machines)")
        return

    configs = ["C1", "C2", "C5"]
    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 6))
    if len(configs) == 1:
        axes = [axes]

    machine_colors = {"h100": "#2196F3", "b200": "#4CAF50"}

    for ax, cfg in zip(axes, configs):
        x = np.arange(len(CONTEXTS))
        width = 0.35

        for i, machine in enumerate(sorted(machines)):
            vals = []
            for ctx in CONTEXTS:
                row = data.get(machine, {}).get(cfg, {}).get(ctx, {}).get(1)
                v = row["total_throughput"] if row and row.get("total_throughput") else 0
                vals.append(v / 1000)
            ax.bar(x + i * width, vals, width,
                   label=machine.upper(), color=machine_colors.get(machine, f"C{i}"))

        ax.set_xlabel("Context Length")
        ax.set_ylabel("Throughput (K tok/s)")
        ax.set_title(f"{CONFIG_LABELS[cfg]}: c=1 Throughput")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(CONTEXTS)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / "chart_cross_machine_throughput.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate performance charts from CSV")
    parser.add_argument("--csv", required=True, help="Input CSV from extract_results.py")
    parser.add_argument("--output-dir", required=True, help="Output directory for charts")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(str(csv_path))
    data = group_data(rows)
    machines = list(data.keys())

    print(f"Loaded {len(rows)} data points from {len(machines)} machine(s): {machines}")

    for machine in machines:
        # Throughput scaling (2x2)
        plot_scaling(data, machine, "total_throughput", "Throughput (tok/s)",
                     "Throughput Scaling", f"chart_{machine}_throughput_scaling.png",
                     output_dir)

        # TPOT scaling (2x2, log Y)
        plot_scaling(data, machine, "mean_tpot_ms", "TPOT (ms)",
                     "TPOT — Lower is Better", f"chart_{machine}_tpot_scaling.png",
                     output_dir, log_y=True)

        # TTFT scaling (2x2)
        plot_scaling(data, machine, "mean_ttft_ms", "TTFT (ms)",
                     "TTFT Scaling", f"chart_{machine}_ttft_scaling.png",
                     output_dir)

        # c=1 bar charts
        plot_c1_bars(data, machine, output_dir)

    # Cross-machine comparison
    plot_cross_machine(data, machines, output_dir)

    print(f"\nAll charts saved to {output_dir}")


if __name__ == "__main__":
    main()
