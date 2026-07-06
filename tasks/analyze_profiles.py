#!/usr/bin/env python3
"""Analyze torch.profiler traces to compare A2A vs AG_RS DCP overhead.

Parses Chrome trace JSON (gzipped) and extracts:
- NCCL kernel durations (ncclDevKernel_*)
- D2D copy kernels (Memcpy DtoD, copy_)
- Triton kernels (_dcp_lse_combine_kernel, _correct_attn_cp_out_kernel)
- FlashInfer/FlashAttention decode kernels
- All-gather Q overhead

Reports per-decode-step and per-layer breakdown.
"""

import gzip
import json
import os
import re
import sys
from collections import defaultdict


def load_trace(path):
    """Load a gzipped Chrome trace JSON."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    # Handle both formats: {"traceEvents": [...]} or [...]
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def categorize_event(name):
    """Categorize a GPU kernel event by its function."""
    name_lower = name.lower()

    if "nccl" in name_lower:
        if "alltoall" in name_lower or "all_to_all" in name_lower:
            return "nccl_alltoall"
        elif "allgather" in name_lower or "all_gather" in name_lower:
            return "nccl_allgather"
        elif "reducescatter" in name_lower or "reduce_scatter" in name_lower:
            return "nccl_reducescatter"
        else:
            return "nccl_other"

    if "memcpy" in name_lower or "copy_" in name_lower:
        if "dtod" in name_lower or "d2d" in name_lower or "DeviceToDevice" in name:
            return "memcpy_d2d"
        return "memcpy_other"

    if "_dcp_lse_combine" in name:
        return "triton_dcp_combine"
    if "_correct_attn_cp_out" in name:
        return "triton_correct_attn"

    if "flash" in name_lower and ("fwd" in name_lower or "attention" in name_lower):
        return "flash_attn"
    if "flashinfer" in name_lower or "BatchDecodeWithPagedKVCacheKernel" in name:
        return "flash_attn"

    if "triton" in name_lower or "kernel" in name_lower:
        return "triton_other"

    if "gemm" in name_lower or "cutlass" in name_lower or "cublas" in name_lower:
        return "gemm"

    return "other"


def analyze_trace(trace_path):
    """Analyze a single trace file and return categorized durations."""
    events = load_trace(trace_path)

    # Collect GPU kernel events (category "kernel" or "gpu_memcpy")
    gpu_events = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        cat = ev.get("cat", "")
        ph = ev.get("ph", "")
        if ph == "X" and cat in ("kernel", "gpu_memcpy", "cuda_runtime"):
            name = ev.get("name", "")
            dur = ev.get("dur", 0)  # microseconds
            ts = ev.get("ts", 0)
            gpu_events.append({
                "name": name,
                "category": categorize_event(name),
                "dur_us": dur,
                "ts_us": ts,
            })

    return gpu_events


def summarize_categories(gpu_events):
    """Summarize events by category."""
    cat_stats = defaultdict(lambda: {"count": 0, "total_us": 0, "names": set()})

    for ev in gpu_events:
        cat = ev["category"]
        cat_stats[cat]["count"] += 1
        cat_stats[cat]["total_us"] += ev["dur_us"]
        # Keep first few unique names for reference
        if len(cat_stats[cat]["names"]) < 3:
            cat_stats[cat]["names"].add(ev["name"][:80])

    return cat_stats


def print_comparison(label, stats):
    """Print categorized stats."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'Category':<25} {'Count':>7} {'Total (ms)':>12} {'Avg (us)':>10}  Sample Names")
    print(f"{'-'*25} {'-'*7} {'-'*12} {'-'*10}  {'-'*30}")

    # Sort by total time descending
    sorted_cats = sorted(stats.items(), key=lambda x: x[1]["total_us"], reverse=True)
    total_us = sum(s["total_us"] for _, s in sorted_cats)

    for cat, s in sorted_cats:
        if s["count"] == 0:
            continue
        avg = s["total_us"] / s["count"] if s["count"] else 0
        names = ", ".join(list(s["names"])[:2])
        pct = (s["total_us"] / total_us * 100) if total_us else 0
        print(f"{cat:<25} {s['count']:>7} {s['total_us']/1000:>11.2f} {avg:>10.1f}  {names[:50]}")

    print(f"{'-'*25} {'-'*7} {'-'*12}")
    print(f"{'TOTAL':<25} {'':>7} {total_us/1000:>11.2f}")


def nccl_detail(gpu_events):
    """Print detailed NCCL kernel breakdown."""
    nccl_events = [ev for ev in gpu_events if "nccl" in ev["category"]]
    if not nccl_events:
        print("  No NCCL events found")
        return

    # Group by exact name
    by_name = defaultdict(list)
    for ev in nccl_events:
        by_name[ev["name"][:100]].append(ev["dur_us"])

    print(f"\n  NCCL Kernel Detail:")
    print(f"  {'Kernel Name':<70} {'Count':>6} {'Total(ms)':>10} {'Avg(us)':>9}")
    print(f"  {'-'*70} {'-'*6} {'-'*10} {'-'*9}")
    for name, durs in sorted(by_name.items(), key=lambda x: sum(x[1]), reverse=True):
        print(f"  {name:<70} {len(durs):>6} {sum(durs)/1000:>10.2f} {sum(durs)/len(durs):>9.1f}")


def main():
    profiles = {
        "A2A-opt 128K": "/output/profile_a2a_131072/",
        "AG_RS 128K": "/output/profile_ag_rs_131072/",
        "A2A-opt 512K": "/output/profile_a2a_524288/",
        "AG_RS 512K": "/output/profile_ag_rs_524288/",
    }

    all_stats = {}

    for label, profile_dir in profiles.items():
        if not os.path.isdir(profile_dir):
            print(f"Skipping {label}: {profile_dir} not found")
            continue

        # Analyze TP-0 trace (rank 0)
        trace_files = sorted([f for f in os.listdir(profile_dir) if f.endswith(".trace.json.gz")])
        if not trace_files:
            print(f"Skipping {label}: no trace files")
            continue

        tp0_file = [f for f in trace_files if "TP-0" in f]
        if not tp0_file:
            tp0_file = [trace_files[0]]

        trace_path = os.path.join(profile_dir, tp0_file[0])
        print(f"\nLoading {label}: {trace_path}")

        gpu_events = analyze_trace(trace_path)
        stats = summarize_categories(gpu_events)
        all_stats[label] = stats

        print_comparison(label, stats)
        nccl_detail(gpu_events)

    # Print side-by-side comparison for 128K
    if "A2A-opt 128K" in all_stats and "AG_RS 128K" in all_stats:
        print("\n")
        print("=" * 80)
        print("  SIDE-BY-SIDE: A2A-opt vs AG_RS at 128K (TP-0)")
        print("=" * 80)
        a2a = all_stats["A2A-opt 128K"]
        agrs = all_stats["AG_RS 128K"]

        all_cats = sorted(set(list(a2a.keys()) + list(agrs.keys())))
        print(f"{'Category':<25} {'A2A count':>9} {'A2A ms':>10} {'AGRS count':>10} {'AGRS ms':>10} {'Delta':>8}")
        print(f"{'-'*25} {'-'*9} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for cat in all_cats:
            a_count = a2a.get(cat, {}).get("count", 0)
            a_ms = a2a.get(cat, {}).get("total_us", 0) / 1000
            g_count = agrs.get(cat, {}).get("count", 0)
            g_ms = agrs.get(cat, {}).get("total_us", 0) / 1000
            delta = a_ms - g_ms
            sign = "+" if delta > 0 else ""
            print(f"{cat:<25} {a_count:>9} {a_ms:>10.2f} {g_count:>10} {g_ms:>10.2f} {sign}{delta:>7.2f}")

        a_total = sum(s.get("total_us", 0) for s in a2a.values()) / 1000
        g_total = sum(s.get("total_us", 0) for s in agrs.values()) / 1000
        delta = a_total - g_total
        sign = "+" if delta > 0 else ""
        print(f"{'-'*25} {'-'*9} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        print(f"{'TOTAL':<25} {'':>9} {a_total:>10.2f} {'':>10} {g_total:>10.2f} {sign}{delta:>7.2f}")

    # Same for 512K
    if "A2A-opt 512K" in all_stats and "AG_RS 512K" in all_stats:
        print("\n")
        print("=" * 80)
        print("  SIDE-BY-SIDE: A2A-opt vs AG_RS at 512K (TP-0)")
        print("=" * 80)
        a2a = all_stats["A2A-opt 512K"]
        agrs = all_stats["AG_RS 512K"]

        all_cats = sorted(set(list(a2a.keys()) + list(agrs.keys())))
        print(f"{'Category':<25} {'A2A count':>9} {'A2A ms':>10} {'AGRS count':>10} {'AGRS ms':>10} {'Delta':>8}")
        print(f"{'-'*25} {'-'*9} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for cat in all_cats:
            a_count = a2a.get(cat, {}).get("count", 0)
            a_ms = a2a.get(cat, {}).get("total_us", 0) / 1000
            g_count = agrs.get(cat, {}).get("count", 0)
            g_ms = agrs.get(cat, {}).get("total_us", 0) / 1000
            delta = a_ms - g_ms
            sign = "+" if delta > 0 else ""
            print(f"{cat:<25} {a_count:>9} {a_ms:>10.2f} {g_count:>10} {g_ms:>10.2f} {sign}{delta:>7.2f}")

        a_total = sum(s.get("total_us", 0) for s in a2a.values()) / 1000
        g_total = sum(s.get("total_us", 0) for s in agrs.values()) / 1000
        delta = a_total - g_total
        sign = "+" if delta > 0 else ""
        print(f"{'-'*25} {'-'*9} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        print(f"{'TOTAL':<25} {'':>9} {a_total:>10.2f} {'':>10} {g_total:>10.2f} {sign}{delta:>7.2f}")


if __name__ == "__main__":
    main()
