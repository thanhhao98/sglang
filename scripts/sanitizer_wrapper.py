"""Wrapper to run a single sglang TP worker under compute-sanitizer.

Usage: Set SGLANG_SANITIZER_RANK=0 to wrap rank 0 with compute-sanitizer.
Then launch the server normally. The wrapper intercepts the worker spawn.
"""
import os
import sys
import subprocess

def main():
    """Replace the normal sglang launch with a sanitizer-wrapped version for one rank."""
    target_rank = int(os.environ.get("SGLANG_SANITIZER_RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    if local_rank == target_rank:
        print(f"[SANITIZER] Wrapping rank {local_rank} with compute-sanitizer")
        cmd = [
            "compute-sanitizer",
            "--tool", "memcheck",
            "--print-limit", "5",
            "--log-file", f"/tmp/sanitizer_rank{local_rank}.log",
            sys.executable,
        ] + sys.argv[1:]
        os.execvp("compute-sanitizer", cmd)
    else:
        # Normal launch for other ranks
        os.execvp(sys.executable, [sys.executable] + sys.argv[1:])

if __name__ == "__main__":
    main()
