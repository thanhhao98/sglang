#!/usr/bin/env python3
"""
Extract benchmark results from JSONL files into a CSV.

Usage:
    python tasks/extract_results.py --input-dir tasks/output/ --output tasks/results.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


# Expected filename pattern: C1_128K_c1.jsonl
JSONL_PATTERN = re.compile(
    r"^(C\d+)_(\d+K|1M)_c(\d+)\.jsonl$"
)


def parse_jsonl_file(filepath: Path) -> dict | None:
    """Read the last JSON line from a JSONL file (bench_serving appends one line per run)."""
    lines = []
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    except Exception as e:
        print(f"  WARNING: Cannot read {filepath}: {e}", file=sys.stderr)
        return None

    if not lines:
        return None

    # Use the last line (most recent run)
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        print(f"  WARNING: Bad JSON in {filepath}: {e}", file=sys.stderr)
        return None


def extract_metrics(data: dict) -> dict:
    """Extract key metrics from bench_serving JSONL output."""
    metrics = {}

    # Direct top-level metrics
    for key in [
        "request_throughput",
        "output_throughput",
        "total_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
        "mean_e2el_ms",
        "median_e2el_ms",
        "std_e2el_ms",
        "p99_e2el_ms",
    ]:
        metrics[key] = data.get(key, "")

    return metrics


def infer_commit_machine(dirpath: Path) -> tuple[str, str]:
    """Infer commit hash and machine from directory name like '8f43263_h100'."""
    dirname = dirpath.name
    parts = dirname.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return dirname, "unknown"


def main():
    parser = argparse.ArgumentParser(description="Extract JSONL benchmark results to CSV")
    parser.add_argument("--input-dir", required=True, help="Directory containing output subdirs")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    rows = []
    total_files = 0
    parsed_ok = 0
    parse_failed = 0

    # Walk all subdirectories
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue

        commit, machine = infer_commit_machine(subdir)

        for filepath in sorted(subdir.iterdir()):
            match = JSONL_PATTERN.match(filepath.name)
            if not match:
                continue

            total_files += 1
            config, context, concurrency = match.group(1), match.group(2), int(match.group(3))

            data = parse_jsonl_file(filepath)
            if data is None:
                parse_failed += 1
                continue

            metrics = extract_metrics(data)
            row = {
                "commit": commit,
                "machine": machine,
                "config": config,
                "context": context,
                "concurrency": concurrency,
                **metrics,
            }
            rows.append(row)
            parsed_ok += 1

    if not rows:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Sort rows
    config_order = {"C1": 0, "C2": 1, "C5": 2}
    ctx_order = {"128K": 0, "256K": 1, "512K": 2, "1M": 3}
    rows.sort(key=lambda r: (
        r["machine"],
        config_order.get(r["config"], 99),
        ctx_order.get(r["context"], 99),
        r["concurrency"],
    ))

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results: {parsed_ok} parsed, {parse_failed} failed, {total_files} total JSONL files")
    print(f"Output: {output_path}")

    # Cross-reference with fail logs
    for subdir in sorted(input_dir.iterdir()):
        fail_log = subdir / "fail_tests.log"
        if fail_log.is_file():
            print(f"\nFailures from {subdir.name}:")
            with open(fail_log) as f:
                print(f.read())


if __name__ == "__main__":
    main()
