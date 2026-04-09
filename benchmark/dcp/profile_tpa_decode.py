#!/usr/bin/env python3
"""Profile TPA decode to measure per-component time breakdown.

Sends requests at specified concurrency, then hits the /metrics endpoint
to collect timing data. Also does a quick torch.profiler trace if requested.

Usage:
    # Start server first, then:
    python3 benchmark/dcp/profile_tpa_decode.py --host 127.0.0.1 --port 8188 --concurrency 64

For nsys profiling of the server itself:
    nsys profile -t cuda,nvtx,nccl -o tpa_profile --force-overwrite true \
        python3 -m sglang.launch_server ... &
    sleep 120  # wait for server
    python3 benchmark/dcp/profile_tpa_decode.py --host 127.0.0.1 --port 8188 --concurrency 64 --num-prompts 20
    kill %1
"""

import argparse
import subprocess
import sys
import time


def wait_for_server(host, port, timeout=300):
    """Wait for server health endpoint."""
    import urllib.request
    url = f"http://{host}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                print(f"Server ready ({time.time()-start:.0f}s)")
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"Server not ready within {timeout}s")
    return False


def run_bench(host, port, concurrency, num_prompts, input_len, output_len):
    """Run bench_serving and capture output."""
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--host", host, "--port", str(port),
        "--model", "Qwen/CodeQwen1.5-7B-Chat",
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", "0.1",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--disable-ignore-eos",
    ]
    print(f"\n{'='*60}")
    print(f"Profiling: cc={concurrency}, prompts={num_prompts}, in={input_len}, out={output_len}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-1000:]}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Profile TPA decode performance")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 64, 512])
    parser.add_argument("--num-prompts-factor", type=int, default=3,
                        help="num_prompts = concurrency * factor")
    parser.add_argument("--input-len", type=int, default=4000)
    parser.add_argument("--output-len", type=int, default=500)
    parser.add_argument("--warmup", action="store_true",
                        help="Run a warmup batch first")
    args = parser.parse_args()

    if not wait_for_server(args.host, args.port):
        sys.exit(1)

    # Warmup
    if args.warmup:
        print("\n--- Warmup ---")
        run_bench(args.host, args.port, 4, 8, args.input_len, 64)

    # Profile at each concurrency
    for cc in args.concurrency:
        num_prompts = cc * args.num_prompts_factor
        run_bench(args.host, args.port, cc, num_prompts,
                  args.input_len, args.output_len)


if __name__ == "__main__":
    main()
