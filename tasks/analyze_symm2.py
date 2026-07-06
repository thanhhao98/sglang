#!/usr/bin/env python3
"""Analyze trace with broader event matching."""
import gzip, json, os, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/output/profile_a2a_symm_128k/"
files = sorted(os.listdir(path))
tp0 = [f for f in files if "TP-0" in f][0]

with gzip.open(os.path.join(path, tp0), "rt") as f:
    data = json.load(f)
events = data.get("traceEvents", data) if isinstance(data, dict) else data

# Categorize ALL events
cats = defaultdict(int)
gpu_kernels = defaultdict(lambda: {"count": 0, "total_us": 0})
nccl_kernels = defaultdict(lambda: {"count": 0, "total_us": 0})

for ev in events:
    if not isinstance(ev, dict):
        continue
    ph = ev.get("ph", "")
    cat = ev.get("cat", "")
    name = ev.get("name", "")
    dur = ev.get("dur", 0)

    if ph == "X":
        cats[cat] += 1
        if cat in ("kernel", "gpu_memcpy"):
            key = name[:100]
            gpu_kernels[key]["count"] += 1
            gpu_kernels[key]["total_us"] += dur
            if "nccl" in name.lower() or "Nccl" in name or "NCCL" in name:
                nccl_kernels[key]["count"] += 1
                nccl_kernels[key]["total_us"] += dur

print(f"Trace: {os.path.join(path, tp0)}")
print(f"Total events: {len(events)}")
print(f"\nEvent categories (ph=X):")
for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

print(f"\nTop 15 GPU kernels by total time:")
print(f"  {'Kernel':<80} {'Count':>6} {'Total(ms)':>10} {'Avg(us)':>9}")
print(f"  {'-'*80} {'-'*6} {'-'*10} {'-'*9}")
for name, s in sorted(gpu_kernels.items(), key=lambda x: x[1]["total_us"], reverse=True)[:15]:
    avg = s["total_us"] / s["count"]
    print(f"  {name:<80} {s['count']:>6} {s['total_us']/1000:>10.2f} {avg:>9.1f}")

print(f"\nNCCL kernels (cat=kernel, name contains nccl/Nccl/NCCL):")
if nccl_kernels:
    for name, s in sorted(nccl_kernels.items(), key=lambda x: x[1]["total_us"], reverse=True):
        avg = s["total_us"] / s["count"]
        print(f"  {name:<80} {s['count']:>6} {s['total_us']/1000:>10.2f} {avg:>9.1f}")
    sendrecv = sum(s["total_us"] for k, s in nccl_kernels.items() if "SendRecv" in k)
    symk = sum(s["total_us"] for k, s in nccl_kernels.items() if "Symk" in k)
    print(f"\n  SendRecv total: {sendrecv/1000:.2f}ms")
    print(f"  SymkDev total:  {symk/1000:.2f}ms")
else:
    print("  (none found)")
