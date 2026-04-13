#!/usr/bin/env python3
"""Analyze nsys profiling results to compare TP8 vs DCP vs TPA.

Parses nsys stats output to extract:
  - Top CUDA kernels by time
  - NCCL communication breakdown (AllReduce, AllToAll, ReduceScatter, AllGather)
  - NVTX layer-level timing (if --enable-layerwise-nvtx-marker was used)

Usage:
    # Generate stats from nsys-rep files:
    python3 benchmark/dcp/analyze_profiles.py /path/to/profiles/nsys/

    # Or point to specific .nsys-rep files:
    python3 benchmark/dcp/analyze_profiles.py tp8.nsys-rep tp8_dcp2.nsys-rep tp8_tpa4_dcp2.nsys-rep
"""

import argparse
import csv
import io
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def run_nsys_stats(nsys_rep_path, report_type):
    """Run nsys stats and return CSV output."""
    cmd = [
        "nsys", "stats",
        "--report", report_type,
        "--format", "csv",
        "--force-export=true",
        str(nsys_rep_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  Warning: nsys stats failed for {nsys_rep_path}: {result.stderr[:200]}")
        return ""
    return result.stdout


def parse_cuda_kern_summary(csv_text):
    """Parse cuda_gpu_kern_sum report."""
    kernels = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        # Column names vary by nsys version; try common ones
        name = row.get("Name", row.get("Kernel Name", ""))
        total_ns = float(row.get("Total Time (ns)", row.get("Total (ns)", 0)))
        count = int(row.get("Instances", row.get("Count", 0)))
        avg_ns = float(row.get("Avg Time (ns)", row.get("Avg (ns)", 0)))
        kernels.append({
            "name": name,
            "total_us": total_ns / 1000,
            "count": count,
            "avg_us": avg_ns / 1000,
        })
    kernels.sort(key=lambda x: x["total_us"], reverse=True)
    return kernels


def parse_nccl_summary(csv_text):
    """Parse nccl_gpu_sum report for communication breakdown."""
    ops = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        name = row.get("Name", row.get("Operation", ""))
        total_ns = float(row.get("Total Time (ns)", row.get("Total (ns)", 0)))
        count = int(row.get("Instances", row.get("Count", 0)))
        avg_ns = float(row.get("Avg Time (ns)", row.get("Avg (ns)", 0)))
        ops.append({
            "name": name,
            "total_us": total_ns / 1000,
            "count": count,
            "avg_us": avg_ns / 1000,
        })
    ops.sort(key=lambda x: x["total_us"], reverse=True)
    return ops


def parse_nvtx_summary(csv_text):
    """Parse nvtx_gpu_proj_sum for NVTX layer markers."""
    markers = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        name = row.get("Name", row.get("Range Name", ""))
        total_ns = float(row.get("Total Time (ns)", row.get("Total (ns)", 0)))
        count = int(row.get("Instances", row.get("Count", 0)))
        avg_ns = float(row.get("Avg Time (ns)", row.get("Avg (ns)", 0)))
        markers.append({
            "name": name,
            "total_us": total_ns / 1000,
            "count": count,
            "avg_us": avg_ns / 1000,
        })
    markers.sort(key=lambda x: x["total_us"], reverse=True)
    return markers


def categorize_nccl_ops(nccl_ops):
    """Group NCCL operations by type."""
    categories = defaultdict(lambda: {"total_us": 0, "count": 0})
    for op in nccl_ops:
        name = op["name"].lower()
        if "allreduce" in name:
            cat = "AllReduce"
        elif "alltoall" in name or "a2a" in name:
            cat = "AllToAll (A2A)"
        elif "reducescatter" in name:
            cat = "ReduceScatter"
        elif "allgather" in name:
            cat = "AllGather"
        elif "send" in name or "recv" in name:
            cat = "Send/Recv"
        else:
            cat = "Other"
        categories[cat]["total_us"] += op["total_us"]
        categories[cat]["count"] += op["count"]
    return dict(categories)


def categorize_kernels(kernels):
    """Group CUDA kernels into functional categories."""
    categories = defaultdict(lambda: {"total_us": 0, "count": 0})
    for k in kernels:
        name = k["name"]
        if any(s in name.lower() for s in ["nccl", "ncclkernel"]):
            cat = "NCCL"
        elif any(s in name.lower() for s in ["flash", "fmha", "attention", "softmax"]):
            cat = "Attention"
        elif any(s in name.lower() for s in ["gemm", "cutlass", "matmul", "linear", "cublas"]):
            cat = "GEMM (MLP+proj)"
        elif any(s in name.lower() for s in ["rmsnorm", "layernorm", "norm"]):
            cat = "Norm"
        elif any(s in name.lower() for s in ["rope", "rotary"]):
            cat = "RoPE"
        elif any(s in name.lower() for s in ["silu", "gelu", "relu", "activation"]):
            cat = "Activation"
        elif any(s in name.lower() for s in ["copy", "memcpy", "memset"]):
            cat = "Memory ops"
        elif any(s in name.lower() for s in ["elementwise", "reduce", "cast"]):
            cat = "Elementwise"
        else:
            cat = "Other"
        categories[cat]["total_us"] += k["total_us"]
        categories[cat]["count"] += k["count"]
    return dict(categories)


def analyze_single_profile(nsys_rep_path):
    """Analyze a single nsys-rep file."""
    print(f"\n  Generating stats for {nsys_rep_path.name}...")

    results = {"path": str(nsys_rep_path)}

    # CUDA kernels
    csv_text = run_nsys_stats(nsys_rep_path, "cuda_gpu_kern_sum")
    if csv_text:
        kernels = parse_cuda_kern_summary(csv_text)
        results["kernels"] = kernels
        results["kernel_categories"] = categorize_kernels(kernels)
        results["total_kernel_us"] = sum(k["total_us"] for k in kernels)

    # NCCL
    csv_text = run_nsys_stats(nsys_rep_path, "nccl_gpu_sum")
    if csv_text:
        nccl_ops = parse_nccl_summary(csv_text)
        results["nccl_ops"] = nccl_ops
        results["nccl_categories"] = categorize_nccl_ops(nccl_ops)
        results["total_nccl_us"] = sum(op["total_us"] for op in nccl_ops)

    # NVTX
    csv_text = run_nsys_stats(nsys_rep_path, "nvtx_gpu_proj_sum")
    if csv_text:
        nvtx = parse_nvtx_summary(csv_text)
        results["nvtx"] = nvtx

    return results


def print_comparison(all_results):
    """Print side-by-side comparison of configs."""
    configs = list(all_results.keys())

    # --- CUDA Kernel Category Breakdown ---
    print("\n" + "=" * 80)
    print("CUDA KERNEL TIME BREAKDOWN (GPU time, microseconds)")
    print("=" * 80)

    all_cats = set()
    for cfg in configs:
        cats = all_results[cfg].get("kernel_categories", {})
        all_cats.update(cats.keys())
    all_cats = sorted(all_cats)

    header = f"{'Category':<20}"
    for cfg in configs:
        header += f"  {cfg:>18} {'%':>5}"
    print(header)
    print("-" * len(header))

    for cat in all_cats:
        line = f"{cat:<20}"
        for cfg in configs:
            cats = all_results[cfg].get("kernel_categories", {})
            total = all_results[cfg].get("total_kernel_us", 1)
            val = cats.get(cat, {}).get("total_us", 0)
            pct = (val / total * 100) if total > 0 else 0
            line += f"  {val:>14.0f} {pct:>5.1f}%"
        print(line)

    # Total
    line = f"{'TOTAL':<20}"
    for cfg in configs:
        total = all_results[cfg].get("total_kernel_us", 0)
        line += f"  {total:>14.0f} {'':>5}"
    print(line)

    # --- NCCL Communication Breakdown ---
    print("\n" + "=" * 80)
    print("NCCL COMMUNICATION BREAKDOWN (GPU time, microseconds)")
    print("=" * 80)

    all_nccl_cats = set()
    for cfg in configs:
        cats = all_results[cfg].get("nccl_categories", {})
        all_nccl_cats.update(cats.keys())
    all_nccl_cats = sorted(all_nccl_cats)

    header = f"{'Operation':<20}"
    for cfg in configs:
        header += f"  {cfg:>14} {'calls':>6}"
    print(header)
    print("-" * len(header))

    for cat in all_nccl_cats:
        line = f"{cat:<20}"
        for cfg in configs:
            cats = all_results[cfg].get("nccl_categories", {})
            val = cats.get(cat, {})
            line += f"  {val.get('total_us', 0):>14.0f} {val.get('count', 0):>6}"
        print(line)

    line = f"{'TOTAL':<20}"
    for cfg in configs:
        total = all_results[cfg].get("total_nccl_us", 0)
        line += f"  {total:>14.0f} {'':>6}"
    print(line)

    # --- Top 10 Kernels per Config ---
    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"TOP 15 CUDA KERNELS: {cfg}")
        print(f"{'=' * 60}")
        kernels = all_results[cfg].get("kernels", [])[:15]
        print(f"  {'Total(us)':>12} {'Count':>8} {'Avg(us)':>10}  Name")
        print(f"  {'-'*12} {'-'*8} {'-'*10}  {'-'*40}")
        for k in kernels:
            name = k["name"][:60]
            print(f"  {k['total_us']:>12.0f} {k['count']:>8} {k['avg_us']:>10.1f}  {name}")


def find_nsys_rep_files(path):
    """Find .nsys-rep files organized by config/cc."""
    path = Path(path)
    results = {}

    if path.suffix == ".nsys-rep":
        # Single file
        name = path.stem
        return {name: path}

    # Directory structure: <base>/nsys/<config>/cc<N>.nsys-rep
    for rep_file in sorted(path.rglob("*.nsys-rep")):
        # Derive config name from parent directory
        cfg = rep_file.parent.name
        cc = rep_file.stem  # e.g., "cc32"
        key = f"{cfg}/{cc}"
        results[key] = rep_file

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze nsys profiling results")
    parser.add_argument("paths", nargs="+", help="nsys-rep files or directory containing them")
    parser.add_argument("--top-kernels", type=int, default=15, help="Number of top kernels to show")
    args = parser.parse_args()

    all_files = {}
    for p in args.paths:
        all_files.update(find_nsys_rep_files(p))

    if not all_files:
        print("No .nsys-rep files found!")
        sys.exit(1)

    print(f"Found {len(all_files)} profile(s):")
    for name, path in sorted(all_files.items()):
        print(f"  {name}: {path}")

    all_results = {}
    for name, path in sorted(all_files.items()):
        all_results[name] = analyze_single_profile(path)

    print_comparison(all_results)


if __name__ == "__main__":
    main()
