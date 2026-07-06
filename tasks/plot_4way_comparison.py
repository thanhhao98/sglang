"""
4-Way Performance Comparison Charts
TP8 vs DCP ag_rs (no CG) vs DCP a2a (CG) vs DCP a2a+repl (CG)
B200 8x183GB, DeepSeek-V2-Lite
"""

import matplotlib.pyplot as plt
import numpy as np

concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# =============================================================================
# B200 Data
# =============================================================================

# TP8 (C1) - from existing benchmarks
tp8 = {
    "128K": {
        "tpot": [5.67, 7.39, 5.25, 8.06, 8.41, 6.88, 8.91, 7.73, 8.54, 8.46],
        "thru": [69755, 126860, 154123, 153316, 147655, 154407, 136886, 147911, 136981, 137471],
    },
    "256K": {
        "tpot": [8.09, 6.94, 4.32, 3.66, 3.44, 2.81, 3.33, 2.92, 3.57, 3.45],
        "thru": [95514, 357111, 449483, 449433, 454802, 417123, 443199, 449584, 447206, 415567],
    },
    "512K": {
        "tpot": [12.29, 85.97, 23.15, 22.80, 68.82, 57.42, 22.13, 56.80, 22.95, 22.82],
        "thru": [55945, 288108, 395821, 364126, 187460, 204719, 360042, 204770, 385031, 395321],
    },
    "1M": {
        "tpot": [13.75, 18.99, 19.33, 19.40, 19.19, 19.82, 20.03, 19.29, 19.78, 19.77],
        "thru": [53776, 237865, 241976, 221249, 220072, 212728, 217503, 219689, 219822, 218079],
    },
}

# DCP8 ag_rs (C2) - no cuda graph baseline
agrs = {
    "128K": {
        "tpot": [4.83, 15.82, 18.28, 27.62, 25.49, 33.26, 33.82, 25.38, 33.90, 25.51],
        "thru": [67771, 86960, 92329, 94689, 96761, 96624, 95863, 96506, 94611, 96642],
    },
    "256K": {
        "tpot": [5.29, 46.04, 85.50, 108.45, 108.06, 108.61, 107.84, 108.65, 108.75, 108.23],
        "thru": [89578, 106353, 117693, 119324, 119790, 115876, 115405, 119995, 119616, 115540],
    },
    "512K": {
        "tpot": [6.76, 1343.27, 1960.77, 1093.30, 1920.37, 1933.16, 1939.64, 1098.24, 1924.80, 1932.66],
        "thru": [67318, 77816, 76930, 80756, 80661, 79367, 81199, 81384, 79718, 80938],
    },
    "1M": {
        "tpot": [6.78, 564.98, 1254.30, 1868.01, 1875.84, 1850.82, 1863.43, 1851.56, 1875.22, 1867.52],
        "thru": [51098, 54986, 57835, 58223, 58164, 57809, 58394, 58152, 58400, 57954],
    },
}

# DCP8 a2a flashinfer with CG (C3) - baseline
a2a = {
    "128K": {
        "tpot": [4.84, 14.84, 16.82, 25.60, 25.77, 33.12, 25.54, 33.27, 33.25, 33.04],
        "thru": [68609, 88135, 95456, 95695, 96470, 96009, 95834, 94072, 95873, 93702],
    },
    "256K": {
        "tpot": [5.30, 42.22, 108.49, 108.11, 107.75, 108.33, 107.89, 107.68, 108.47, 108.54],
        "thru": [88048, 104564, 116326, 119257, 118459, 119634, 119392, 119802, 119946, 119627],
    },
    "512K": {
        "tpot": [7.00, 1818.35, 1946.82, 1060.41, 1953.60, 1957.82, 1933.76, 1062.47, 1951.88, 1950.14],
        "thru": [67655, 77628, 77327, 79159, 80662, 79604, 79456, 79658, 79490, 80205],
    },
    "1M": {
        "tpot": [7.03, 578.26, 1278.78, 1812.19, 1813.27, 1802.49, 1819.60, 1802.49, 1823.45, 1812.14],
        "thru": [51055, 55038, 57756, 57912, 58106, 57611, 57786, 57973, 57917, 57837],
    },
}

# DCP8 a2a + replicate-q-proj with CG (C5) - new data
a2a_repl = {
    "128K": {
        "tpot": [4.77, 14.80, 16.90, 25.62, 25.81, 33.09, 25.47, 33.16, 33.14, 33.24],
        "thru": [67430, 83336, 93169, 93050, 93317, 91611, 93027, 91028, 91362, 90975],
    },
    "256K": {
        "tpot": [5.36, 42.50, 115.30, 186.20, 186.93, 185.56, 187.60, 186.84, 186.11, 187.74],
        "thru": [84974, 96906, 108418, 111678, 111721, 111932, 107758, 112075, 112344, 111900],
    },
    "512K": {
        "tpot": [6.84, 1991.51, 1860.10, 1894.59, 1887.27, 1899.58, 1888.58, 2861.77, 1888.44, 1898.37],
        "thru": [63791, 72786, 72398, 74196, 74684, 74395, 74529, 74848, 74463, 74174],
    },
    "1M": {
        "tpot": [6.95, 1341.02, 1212.43, 1767.48, 1765.26, 1754.98, 1762.75, 1757.10, 1796.48, 1764.84],
        "thru": [48518, 51825, 54373, 54531, 54556, 54824, 54729, 54702, 54490, 54569],
    },
}

configs = {
    "TP8": tp8,
    "DCP ag_rs (no CG)": agrs,
    "DCP a2a (CG)": a2a,
    "DCP a2a+repl (CG)": a2a_repl,
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
# Chart 1: c=1 TPOT bar chart across contexts
# =============================================================================
def plot_c1_tpot():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(contexts))
    width = 0.2

    for i, (name, data) in enumerate(configs.items()):
        vals = [data[ctx]["tpot"][0] for ctx in contexts]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[name])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Context Length")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("B200: c=1 Decode TPOT — Lower is Better")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(contexts)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("tasks/chart_c1_tpot.png", dpi=150)
    print("Saved chart_c1_tpot.png")


# =============================================================================
# Chart 2: Throughput vs Concurrency (one subplot per context)
# =============================================================================
def plot_throughput_scaling():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ctx in enumerate(contexts):
        ax = axes[idx]
        for name, data in configs.items():
            ax.plot(concurrencies, [v / 1000 for v in data[ctx]["thru"]],
                    marker=markers[name], color=colors[name], label=name,
                    linewidth=2, markersize=5)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Throughput (K tok/s)")
        ax.set_title(f"B200: {ctx} Throughput Scaling")
        ax.set_xscale("log", base=2)
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies], fontsize=7)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("tasks/chart_throughput_scaling.png", dpi=150)
    print("Saved chart_throughput_scaling.png")


# =============================================================================
# Chart 3: TPOT vs Concurrency (one subplot per context)
# =============================================================================
def plot_tpot_scaling():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ctx in enumerate(contexts):
        ax = axes[idx]
        for name, data in configs.items():
            ax.plot(concurrencies, data[ctx]["tpot"],
                    marker=markers[name], color=colors[name], label=name,
                    linewidth=2, markersize=5)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title(f"B200: {ctx} TPOT — Lower is Better")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies], fontsize=7)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("tasks/chart_tpot_scaling.png", dpi=150)
    print("Saved chart_tpot_scaling.png")


# =============================================================================
# Chart 4: c=1 Throughput bar chart
# =============================================================================
def plot_c1_throughput():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(contexts))
    width = 0.2

    for i, (name, data) in enumerate(configs.items()):
        vals = [data[ctx]["thru"][0] / 1000 for ctx in contexts]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[name])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}K", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Throughput (K tok/s)")
    ax.set_title("B200: c=1 Total Throughput — Higher is Better")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(contexts)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("tasks/chart_c1_throughput.png", dpi=150)
    print("Saved chart_c1_throughput.png")


# =============================================================================
# Chart 5: DCP-only comparison at c=1 (zoomed in)
# =============================================================================
def plot_dcp_c1_comparison():
    dcp_configs = {k: v for k, v in configs.items() if k != "TP8"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(contexts))
    width = 0.25

    # TPOT
    for i, (name, data) in enumerate(dcp_configs.items()):
        vals = [data[ctx]["tpot"][0] for ctx in contexts]
        bars = ax1.bar(x + i * width, vals, width, label=name, color=colors[name])
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("TPOT (ms)")
    ax1.set_title("B200 DCP Configs: c=1 TPOT (Lower is Better)")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(contexts)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Throughput
    for i, (name, data) in enumerate(dcp_configs.items()):
        vals = [data[ctx]["thru"][0] / 1000 for ctx in contexts]
        bars = ax2.bar(x + i * width, vals, width, label=name, color=colors[name])
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.0f}K", ha="center", va="bottom", fontsize=8)

    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Throughput (K tok/s)")
    ax2.set_title("B200 DCP Configs: c=1 Throughput (Higher is Better)")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(contexts)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("tasks/chart_dcp_c1_comparison.png", dpi=150)
    print("Saved chart_dcp_c1_comparison.png")


if __name__ == "__main__":
    plot_c1_tpot()
    plot_c1_throughput()
    plot_throughput_scaling()
    plot_tpot_scaling()
    plot_dcp_c1_comparison()
    print("\nAll charts saved to tasks/")
