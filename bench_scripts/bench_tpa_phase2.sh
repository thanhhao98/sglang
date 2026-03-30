#!/usr/bin/env bash
# TPA Phase-2 benchmark suite: compare pure_tp vs dcp2_a2a vs tpa2_dcp4_a2a
#
# Runs bench_serving across all modes, context lengths, and concurrency levels
# specified in the TPA Phase-2 plan.
#
# Usage:
#   bash bench_scripts/bench_tpa_phase2.sh [--model MODEL] [--port PORT]
#
# Prerequisites:
#   - 8x H100 or B200 GPUs
#   - Model weights available (default: Qwen/CodeQwen1.5-7B-Chat)
#   - sglang installed with FA3/FA4 support

set -euo pipefail

MODEL="${MODEL:-Qwen/CodeQwen1.5-7B-Chat}"
BASE_PORT="${PORT:-30000}"
RESULTS_DIR="benchmark/dcp/results/tpa_phase2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

CONTEXT_LENGTHS=(262144 524288 1048576)
CONCURRENCY_LEVELS=(1 8 32 64)

declare -A MODES
MODES[pure_tp]="--tp-size 8"
MODES[dcp2_a2a]="--tp-size 8 --dcp-size 2 --dcp-comm-backend a2a"
MODES[tpa2_dcp4_a2a]="--tp-size 8 --attention-tensor-parallel-size 2 --dcp-size 4 --dcp-comm-backend a2a"
MODES[tpa4_dcp2_a2a]="--tp-size 8 --attention-tensor-parallel-size 4 --dcp-size 2 --dcp-comm-backend a2a"

start_server() {
    local mode="$1"
    local port="$2"
    local args="${MODES[$mode]}"

    echo "  Starting server: mode=$mode, port=$port"
    python -m sglang.launch_server \
        --model "$MODEL" \
        --trust-remote-code \
        $args \
        --port "$port" \
        --decode-attention-backend fa3 \
        --context-length 1048576 \
        --mem-fraction-static 0.85 \
        > "$RESULTS_DIR/${mode}_server.log" 2>&1 &
    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"

    for i in $(seq 1 180); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "  Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        unset SERVER_PID
    fi
}

run_benchmark() {
    local mode="$1"
    local ctx="$2"
    local conc="$3"
    local port="$4"
    local output_file="$RESULTS_DIR/${mode}_ctx${ctx}_c${conc}.json"

    echo "    bench: mode=$mode ctx=$ctx c=$conc -> $output_file"
    python -m sglang.bench_serving \
        --backend sglang \
        --port "$port" \
        --dataset-name random \
        --random-input "$ctx" \
        --random-output 128 \
        --num-prompts "$conc" \
        --request-rate "$conc" \
        --output-file "$output_file" \
        2>&1 | tail -5
}

trap stop_server EXIT

echo "=== TPA Phase-2 Benchmark Suite ==="
echo "Model:   $MODEL"
echo "Results: $RESULTS_DIR"
echo ""

for mode in pure_tp dcp2_a2a tpa2_dcp4_a2a tpa4_dcp2_a2a; do
    echo "[$mode]"
    start_server "$mode" "$BASE_PORT"

    for ctx in "${CONTEXT_LENGTHS[@]}"; do
        for conc in "${CONCURRENCY_LEVELS[@]}"; do
            run_benchmark "$mode" "$ctx" "$conc" "$BASE_PORT"
        done
    done

    stop_server
    echo ""
done

echo "=== Generating comparison table ==="
python -c "
import json, glob, os

results_dir = '$RESULTS_DIR'
files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
print(f'| Mode | Context | Concurrency | Req Throughput | Output Tok/s | ITL Avg (ms) | ITL p50 (ms) |')
print(f'| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
for f in files:
    name = os.path.basename(f).replace('.json', '')
    parts = name.split('_')
    try:
        with open(f) as fh:
            data = json.load(fh)
        mode = '_'.join(parts[:-2])
        ctx = parts[-2].replace('ctx', '')
        conc = parts[-1].replace('c', '')
        req_thr = data.get('request_throughput', 0)
        out_thr = data.get('output_throughput', 0)
        itl_avg = data.get('mean_itl_ms', 0)
        itl_p50 = data.get('median_itl_ms', 0)
        print(f'| {mode} | {ctx} | {conc} | {req_thr:.5f} | {out_thr:.2f} | {itl_avg:.2f} | {itl_p50:.2f} |')
    except Exception as e:
        print(f'| {name} | - | - | ERROR: {e} | - | - | - |')
" | tee "$RESULTS_DIR/comparison_table.md"

echo ""
echo "=== Benchmark complete ==="
echo "Results in: $RESULTS_DIR"
