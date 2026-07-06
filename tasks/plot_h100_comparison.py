"""
H100 4-Way Performance Comparison Charts
TP8 vs DCP ag_rs vs DCP a2a vs DCP a2a+repl
H100 8x80GB, DeepSeek-V2-Lite
"""

import matplotlib.pyplot as plt
import numpy as np

concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# =============================================================================
# H100 Data (from accuracy_performance_benchmarkings.md)
# =============================================================================

# TP8 (C1)
tp8 = {
    "128K": {
        "tpot": [13.82, 12.38, 11.46, 8.06, 8.31, 8.57, 9.13, 7.89, 8.74, 8.68],
        "thru": [39549, 63836, 68861, 69379, 68563, 67354, 66879, 64454, 65711, 65369],
    },
    "256K": {
        "tpot": [34.53, 8.98, 5.95, 4.09, 3.69, 1.25, 2.64, 1.58, 1.82, 1.75],
        "thru": [51877, 250785, 368479, 379648, 367521, 348759, 381306, 373301, 285317, 329919],
    },
    "512K": {
        "tpot": [86.06, 82.31, 22.75, 26.08, 39.32, 26.53, 37.39, 36.79, 29.07, 27.87],
        "thru": [33214, 42286, 44200, 45077, 44780, 44040, 44231, 43946, 43553, 42853],
    },
    "1M": {
        "tpot": [100.65, 22.97, 20.68, 20.48, 87.02, 35.96, 23.43, 25.67, 21.22, 21.29],
        "thru": [28380, 32072, 31560, 33514, 4187, 1133, 31542, 32417, 31678, 31715],
    },
}

# DCP8 ag_rs (C2)
agrs = {
    "128K": {
        "tpot": [4.19, 17.37, 22.87, 38.05, 35.57, 35.49, 36.31, 45.88, 45.98, 36.46],
        "thru": [61522, 73767, 66582, 65793, 66835, 66789, 66772, 65064, 65016, 66336],
    },
    "256K": {
        "tpot": [133.96, 54.54, 201.21, 1062.78, 1231.45, 1241.44, 1251.71, 1229.45, 1214.56, 1212.25],
        "thru": [70450, 73085, 82370, 28054, 5770, 5766, 5632, 5799, 5777, 5797],
    },
    "512K": {
        "tpot": [6.21, 1835.57, 852.06, 889.10, 854.25, 887.06, 891.63, 1136.49, 1146.35, 853.11],
        "thru": [47488, 51207, 48225, 48017, 48075, 48043, 47896, 48121, 48126, 48057],
    },
    "1M": {
        "tpot": [6.08, 1868.04, 1895.23, 3064.14, 3065.08, 3061.97, 1901.70, 3048.58, 3055.04, 3065.82],
        "thru": [35784, 38293, 39712, 39721, 39709, 39497, 39622, 39953, 39833, 39719],
    },
}

# DCP8 a2a flashinfer (C3) - optimized with per-bs CG buffers
a2a = {
    "128K": {
        "tpot": [4.30, 17.59, 21.82, 33.17, 32.99, 33.04, 33.07, 33.03, 33.83, 29.67],
        "thru": [60322, 73307, 80749, 81023, 81384, 81357, 81275, 81408, 79430, 80705],
    },
    "256K": {
        "tpot": [4.99, 50.83, 192.29, 365.28, 364.83, 366.92, 364.78, 364.58, 364.63, 364.13],
        "thru": [67355, 75590, 80602, 84484, 84221, 83952, 84489, 84515, 84498, 84570],
    },
    "512K": {
        "tpot": [6.23, 1263.14, 4604.92, 4566.86, 4249.60, 4252.84, 4576.45, 4308.05, 4596.17, 4594.17],
        "thru": [42324, 52944, 52121, 51930, 48339, 48425, 52043, 48036, 51976, 51990],
    },
    "1M": {
        "tpot": [6.11, 328.94, 9010.90, 9712.17, 10345.85, 10348.03, 10439.73, 10387.06, 10346.61, 10330.19],
        "thru": [35742, 38245, 39732, 39817, 39839, 39869, 39679, 39817, 39885, 39922],
    },
}

# DCP8 a2a + replicate-q-proj (C5) - only 128K available
a2a_repl = {
    "128K": {
        "tpot": [4.39, 17.12, 35.52, 50.11, 37.40, 47.81, 36.29, 36.21, 48.04, 37.66],
        "thru": [57642, 70578, 73234, 72907, 75641, 73961, 76061, 75498, 74420, 75695],
    },
}

configs_128k = {
    "TP8": tp8,
    "DCP ag_rs (no CG)": agrs,
    "DCP a2a (CG)": a2a,
    "DCP a2a+repl (CG)": a2a_repl,
}

configs_3way = {
    "TP8": tp8,
    "DCP ag_rs (no CG)": agrs,
    "DCP a2a (CG)": a2a,
}

colors = {
    "TP8": "#2196F3",
    "DCP ag_rs (no CG)": "#FF9800",
    "DCP a2a (CG)": "#4CAF50",
    "DCP a2a+repl (CG)": "#E91E63",
}

markers = {
    "TP8": "o",
    "DCP ag_rs (no CG)": "s",
    "DCP a2a (CG)": "^",
    "DCP a2a+repl (CG)": "D",
}

contexts = ["128K", "256K", "512K", "1M"]


# =============================================================================
# Chart 1: c=1 TPOT bar chart (3-way for all contexts, 4-way for 128K)
# =============================================================================
def plot_c1_tpot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

    # Left: 3-way across all contexts
    x = np.arange(len(contexts))
    width = 0.25
    for i, (name, data) in enumerate(configs_3way.items()):
        vals = [data[ctx]["tpot"][0] for ctx in contexts]
        bars = ax1.bar(x + i * width, vals, width, label=name, color=colors[name])
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("TPOT (ms)")
    ax1.set_title("H100: c=1 Decode TPOT — Lower is Better")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(contexts)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Right: 4-way at 128K only (includes a2a+repl)
    x2 = np.arange(1)
    width2 = 0.2
    for i, (name, data) in enumerate(configs_128k.items()):
        val = data["128K"]["tpot"][0]
        bar = ax2.bar(x2 + i * width2, [val], width2, label=name, color=colors[name])
        ax2.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax2.set_ylabel("TPOT (ms)")
    ax2.set_title("H100 128K: 4-Way c=1 TPOT")
    ax2.set_xticks([x2[0] + 1.5 * width2])
    ax2.set_xticklabels(["128K"])
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("tasks/chart_h100_c1_tpot.png", dpi=150)
    print("Saved chart_h100_c1_tpot.png")


# =============================================================================
# Chart 2: Throughput vs Concurrency (4 subplots, 128K has 4-way, rest 3-way)
# =============================================================================
def plot_throughput_scaling():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ctx in enumerate(contexts):
        ax = axes[idx]
        cfgs = configs_128k if ctx == "128K" else configs_3way
        for name, data in cfgs.items():
            if ctx in data:
                ax.plot(concurrencies, [v / 1000 for v in data[ctx]["thru"]],
                        marker=markers[name], color=colors[name], label=name,
                        linewidth=2, markersize=5)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Throughput (K tok/s)")
        title_suffix = " (4-way)" if ctx == "128K" else ""
        ax.set_title(f"H100: {ctx} Throughput Scaling{title_suffix}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies], fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("tasks/chart_h100_throughput_scaling.png", dpi=150)
    print("Saved chart_h100_throughput_scaling.png")


# =============================================================================
# Chart 3: TPOT vs Concurrency (log scale)
# =============================================================================
def plot_tpot_scaling():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ctx in enumerate(contexts):
        ax = axes[idx]
        cfgs = configs_128k if ctx == "128K" else configs_3way
        for name, data in cfgs.items():
            if ctx in data:
                ax.plot(concurrencies, data[ctx]["tpot"],
                        marker=markers[name], color=colors[name], label=name,
                        linewidth=2, markersize=5)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title(f"H100: {ctx} TPOT — Lower is Better")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies], fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("tasks/chart_h100_tpot_scaling.png", dpi=150)
    print("Saved chart_h100_tpot_scaling.png")


# =============================================================================
# Chart 4: H100 128K 4-way detailed comparison
# =============================================================================
def plot_h100_128k_4way():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput
    for name, data in configs_128k.items():
        ax1.plot(concurrencies, [v / 1000 for v in data["128K"]["thru"]],
                 marker=markers[name], color=colors[name], label=name,
                 linewidth=2, markersize=6)
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Throughput (K tok/s)")
    ax1.set_title("H100 128K: Throughput Scaling (4-Way)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(concurrencies)
    ax1.set_xticklabels([str(c) for c in concurrencies], fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # TPOT
    for name, data in configs_128k.items():
        ax2.plot(concurrencies, data["128K"]["tpot"],
                 marker=markers[name], color=colors[name], label=name,
                 linewidth=2, markersize=6)
    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("TPOT (ms)")
    ax2.set_title("H100 128K: TPOT Scaling (4-Way)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(concurrencies)
    ax2.set_xticklabels([str(c) for c in concurrencies], fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("tasks/chart_h100_128k_4way.png", dpi=150)
    print("Saved chart_h100_128k_4way.png")


# =============================================================================
# Chart 5: c=1 Throughput bar chart
# =============================================================================
def plot_c1_throughput():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

    # Left: 3-way across all contexts
    x = np.arange(len(contexts))
    width = 0.25
    for i, (name, data) in enumerate(configs_3way.items()):
        vals = [data[ctx]["thru"][0] / 1000 for ctx in contexts]
        bars = ax1.bar(x + i * width, vals, width, label=name, color=colors[name])
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.0f}K", ha="center", va="bottom", fontsize=7)

    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("Throughput (K tok/s)")
    ax1.set_title("H100: c=1 Total Throughput — Higher is Better")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(contexts)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Right: 4-way at 128K
    x2 = np.arange(1)
    width2 = 0.2
    for i, (name, data) in enumerate(configs_128k.items()):
        val = data["128K"]["thru"][0] / 1000
        bar = ax2.bar(x2 + i * width2, [val], width2, label=name, color=colors[name])
        ax2.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() + 0.5,
                 f"{val:.0f}K", ha="center", va="bottom", fontsize=8)

    ax2.set_ylabel("Throughput (K tok/s)")
    ax2.set_title("H100 128K: 4-Way c=1")
    ax2.set_xticks([x2[0] + 1.5 * width2])
    ax2.set_xticklabels(["128K"])
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("tasks/chart_h100_c1_throughput.png", dpi=150)
    print("Saved chart_h100_c1_throughput.png")


if __name__ == "__main__":
    plot_c1_tpot()
    plot_c1_throughput()
    plot_throughput_scaling()
    plot_tpot_scaling()
    plot_h100_128k_4way()
    print("\nAll H100 charts saved to tasks/")
