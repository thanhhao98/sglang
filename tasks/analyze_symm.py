#!/usr/bin/env python3
"""Quick NCCL kernel analysis for symm-mem A2A profile."""
import gzip, json, os, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/output/profile_a2a_symm_128k/"
files = sorted(os.listdir(path))
tp0 = [f for f in files if "TP-0" in f][0]

with gzip.open(os.path.join(path, tp0), "rt") as f:
    data = json.load(f)
events = data.get("traceEvents", data) if isinstance(data, dict) else data

nccl = defaultdict(lambda: {"count": 0, "total_us": 0})
for ev in events:
    if not isinstance(ev, dict):
        continue
    if ev.get("ph") != "X":
        continue
    name = ev.get("name", "")
    if "nccl" in name.lower():
        key = name[:90]
        nccl[key]["count"] += 1
        nccl[key]["total_us"] += ev.get("dur", 0)

print(f"NCCL Kernel Breakdown: {path}")
print(f"{'Kernel':<80} {'Count':>6} {'Total(ms)':>10} {'Avg(us)':>9}")
print("-" * 110)
for name, s in sorted(nccl.items(), key=lambda x: x[1]["total_us"], reverse=True):
    avg = s["total_us"] / s["count"] if s["count"] else 0
    print(f"{name:<80} {s['count']:>6} {s['total_us']/1000:>10.2f} {avg:>9.1f}")

total = sum(s["total_us"] for s in nccl.values())
print(f"\n{'TOTAL NCCL':<80} {'':>6} {total/1000:>10.2f}")

# Highlight: did SendRecv change to Symk?
sendrecv = sum(s["total_us"] for k, s in nccl.items() if "SendRecv" in k)
symk = sum(s["total_us"] for k, s in nccl.items() if "Symk" in k)
print(f"\nSendRecv total: {sendrecv/1000:.2f}ms")
print(f"SymkDev total:  {symk/1000:.2f}ms")
