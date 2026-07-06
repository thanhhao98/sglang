import argparse
import os
import re

import matplotlib.pyplot as plt
from tabulate import tabulate

METRICS = {
    "Output token throughput (tok/s)": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "Mean TTFT (ms)": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "Mean TPOT (ms)": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "Mean ITL (ms)": r"Mean ITL \(ms\):\s+([\d.]+)",
}


def parse_bench_folder(folder_path):
    """Parse all cc*.txt files in a folder, return {concurrency: {metric: value}}."""
    results = {}
    for fname in os.listdir(folder_path):
        m = re.match(r"cc(\d+)\.txt", fname)
        if not m:
            continue
        cc = int(m.group(1))
        with open(os.path.join(folder_path, fname)) as f:
            text = f.read()
        metrics = {}
        for name, pattern in METRICS.items():
            match = re.search(pattern, text)
            if match:
                metrics[name] = float(match.group(1))
        if metrics:
            results[cc] = metrics
    return results


def plot_comparison(folder_configs, output_file="bench_comparison.png"):
    """Plot benchmark comparison across folders."""
    all_data = {}
    for folder_path, label in folder_configs:
        all_data[label] = parse_bench_folder(folder_path)

    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric_name in enumerate(METRICS):
        ax = axes[idx]
        for label, data in all_data.items():
            ccs = sorted(data.keys())
            values = [data[cc].get(metric_name, float("nan")) for cc in ccs]
            ax.plot(ccs, values, marker="o", label=label)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Benchmark Comparison Across Concurrency Levels", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Chart saved to {output_file}")
    plt.show()
    return all_data


def print_and_save_tables(all_data, output_file="bench_results.md"):
    """Print tables to console with tabulate and write markdown file."""
    all_ccs = sorted(set(cc for data in all_data.values() for cc in data))
    labels = list(all_data.keys())
    cc_headers = [f"cc{cc}" for cc in all_ccs]

    md_lines = ["# Benchmark Results\n"]
    for metric_name in METRICS:
        rows = []
        for label in labels:
            data = all_data[label]
            vals = []
            for cc in all_ccs:
                v = data.get(cc, {}).get(metric_name)
                vals.append(f"{v:.2f}" if v is not None else "-")
            rows.append([label] + vals)

        print(f"\n{'=' * 40}")
        print(f"  {metric_name}")
        print(f"{'=' * 40}")
        print(tabulate(rows, headers=["Config"] + cc_headers, tablefmt="simple_grid"))

        md_lines.append(f"## {metric_name}\n")
        md_lines.append(tabulate(rows, headers=["Config"] + cc_headers, tablefmt="github"))
        md_lines.append("")

    md_text = "\n".join(md_lines)
    with open(output_file, "w") as f:
        f.write(md_text)
    print(f"\nMarkdown saved to {output_file}")


def discover_folders(base_dir, ignore_folders=None):
    """Walk base_dir to find all folders containing cc*.txt files, return (path, label) list."""
    ignore_set = set(ignore_folders or [])
    folder_configs = []
    for root, _, files in os.walk(base_dir):
        if any(re.match(r"cc\d+\.txt", f) for f in files):
            rel = os.path.relpath(root, base_dir)
            label = rel.replace(os.sep, "_")
            # Skip if any ignore pattern matches the label or relative path
            if any(ign in label or ign in rel for ign in ignore_set):
                continue
            folder_configs.append((root, label))
    return sorted(folder_configs, key=lambda x: x[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and compare benchmark results")
    parser.add_argument(
        "--ignore-folder", action="append", default=[],
        help="Substring to ignore in folder labels (can be repeated)",
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    folder_configs = discover_folders(base, args.ignore_folder)
    print(f"Found {len(folder_configs)} benchmark folders:")
    for _, label in folder_configs:
        print(f"  {label}")
    all_data = plot_comparison(folder_configs, os.path.join(base, "bench_comparison.png"))
    print_and_save_tables(all_data, os.path.join(base, "bench_results.md"))
