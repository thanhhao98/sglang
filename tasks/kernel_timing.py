#!/usr/bin/env python3
"""Extract specific kernel timings from trace."""
import gzip, json, os, sys
from collections import defaultdict

path = sys.argv[1]
files = sorted(os.listdir(path))
tp0 = [f for f in files if "TP-0" in f][0]
with gzip.open(os.path.join(path, tp0), "rt") as f:
    data = json.load(f)
events = data.get("traceEvents", data) if isinstance(data, dict) else data

targets = defaultdict(lambda: {"count": 0, "total_us": 0, "durs": []})
for ev in events:
    if not isinstance(ev, dict):
        continue
    if ev.get("ph") != "X":
        continue
    cat = ev.get("cat", "")
    if cat not in ("kernel", "gpu_memcpy"):
        continue
    name = ev.get("name", "")
    dur = ev.get("dur", 0)

    if "dcp_lse_combine" in name:
        key = "dcp_lse_combine_kernel"
    elif "correct_attn" in name:
        key = "correct_attn_kernel"
    elif "Memcpy DtoD" in name or "copy_" in name.lower():
        key = "D2D_memcpy"
    elif "SendRecv" in name:
        key = "nccl_SendRecv"
    elif "Symk" in name and "AllGather" in name:
        key = "nccl_SymkAllGather"
    elif "Symk" in name and "ReduceScatter" in name:
        key = "nccl_SymkReduceScatter"
    elif "AllReduce" in name and "nccl" in name.lower():
        key = "nccl_AllReduce"
    else:
        continue

    targets[key]["count"] += 1
    targets[key]["total_us"] += dur
    targets[key]["durs"].append(dur)

print(f"Trace: {os.path.join(path, tp0)}")
print(f"{'Kernel':<30} {'Count':>6} {'Total(ms)':>10} {'Avg(us)':>9} {'Min(us)':>9} {'Max(us)':>9}")
print("-" * 80)
for key in sorted(targets.keys()):
    s = targets[key]
    avg = s["total_us"] / s["count"] if s["count"] else 0
    mn = min(s["durs"]) if s["durs"] else 0
    mx = max(s["durs"]) if s["durs"] else 0
    print(f"{key:<30} {s['count']:>6} {s['total_us']/1000:>10.2f} {avg:>9.1f} {mn:>9} {mx:>9}")
