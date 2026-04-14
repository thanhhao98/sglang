#!/bin/bash
# Benchmark DCP for GQA models — PR #14982 reproduction.
#
# Reproduces FENP's PR #14982 results: DCP for GQA with FlashInfer backend
# on Qwen3-235B-A22B-Instruct-2507. Two workloads, two configs.
#
# Workloads (match PR exactly):
#   - 32K input / 4K output
#   - 32K input / 8K output
#
# Configs:
#   - tp8:      TP=8 baseline (no DCP)
#   - tp8_dcp2: TP=8 + DCP=2 (FlashInfer, AG+RS)
#
# Usage:
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh [scenario] [mode]
#
#   Scenarios: 32k_4k, 32k_8k, all (default)
#   Modes:     accuracy, perf, all (default)
#
#   Examples:
#     bash benchmark/dcp/bench_tpa_gqa_serving.sh all accuracy   # accuracy only
#     bash benchmark/dcp/bench_tpa_gqa_serving.sh 32k_4k perf    # perf 32K/4K only
#     bash benchmark/dcp/bench_tpa_gqa_serving.sh all all         # full run
#
# Prerequisites:
#   - 8x H100 GPUs (or H20 as in the original PR)
#   - Model downloaded: huggingface-cli download Qwen/Qwen3-235B-A22B-Instruct-2507

set -euo pipefail

HOST=127.0.0.1
PORT=8188
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
HASH=$(git rev-parse --short=7 HEAD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT="${SCRIPT_DIR}/results/${BRANCH}_${HASH}"

COMMON_ENV="NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1"

SCENARIO_FILTER="${1:-all}"
RUN_MODE="${2:-all}"

# ---- Helper functions ----

wait_for_server() {
    local max_wait=600
    local elapsed=0
    echo "Waiting for server on ${HOST}:${PORT} ..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q 200; then
            echo "Server ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: Server not ready within ${max_wait}s"
    return 1
}

kill_server() {
    echo "Killing server on port ${PORT} ..."
    pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    pkill -f "sglang::schedul" 2>/dev/null || true
    pkill -f "sglang::detoken" 2>/dev/null || true
    sleep 5
    pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    pkill -9 -f "sglang::schedul" 2>/dev/null || true
    pkill -9 -f "sglang::detoken" 2>/dev/null || true
    sleep 3
}

run_accuracy() {
    local output_dir="$1"
    local acc_file="${output_dir}/accuracy_gsm8k.txt"

    if [ -f "$acc_file" ] && grep -q "Accuracy:" "$acc_file" 2>/dev/null; then
        echo "  Accuracy already done, skipping"
        return 0
    fi

    echo "Running accuracy test -> ${acc_file}"
    python3 -m sglang.test.few_shot_gsm8k \
        --parallel 128 --max-new-tokens 512 \
        --host "$HOST" --port "$PORT" 2>&1 | tee "$acc_file"
}

run_perf() {
    local output_dir="$1"
    local input_len="$2"
    local output_len="$3"
    shift 3
    local concurrencies=("$@")

    echo "Running perf: in=${input_len} out=${output_len} -> ${output_dir}"
    for C in "${concurrencies[@]}"; do
        NUM_PROMPTS=$((C * 5))
        FILE_NAME="${output_dir}/cc${C}.txt"

        if [ -f "$FILE_NAME" ] && grep -q "Output token throughput" "$FILE_NAME" 2>/dev/null; then
            echo "  Skipping cc${C} — already completed"
            continue
        fi

        echo "--- Concurrency=$C, Prompts=$NUM_PROMPTS -> $FILE_NAME ---"
        python3 -m sglang.bench_serving --backend sglang \
            --host "$HOST" --port "$PORT" \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len "$input_len" \
            --random-output-len "$output_len" \
            --random-range-ratio 0.1 \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$C" \
            --disable-ignore-eos 2>&1 | tee "$FILE_NAME" || {
            echo "  WARNING: cc${C} failed (likely OOM). Skipping remaining CCs."
            break
        }
    done
}

start_server() {
    local cfg_name="$1"
    local mem_frac="$2"
    local dcp="$3"

    local extra_args=""
    if [ "$dcp" -gt 0 ]; then
        extra_args="--dcp-size ${dcp}"
    fi

    local SERVER_LOG="/tmp/sglang_server_${cfg_name}.log"

    echo "======================================================="
    echo "Starting: ${cfg_name} (${MODEL})"
    echo "  backend=flashinfer  mem=${mem_frac}  dcp=${dcp}"
    echo "  server_log=${SERVER_LOG}"
    echo "======================================================="

    eval "${COMMON_ENV} python3 -m sglang.launch_server \
        --model-path ${MODEL} --host 0.0.0.0 --port ${PORT} \
        --trust-remote-code --enable-cache-report --log-level info --tp-size 8 \
        --chunked-prefill-size 32768 \
        --disable-radix-cache --enable-symm-mem \
        --mem-fraction-static ${mem_frac} \
        --attention-backend flashinfer \
        ${extra_args}" > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"
}

run_config() {
    local scenario_name="$1"
    local cfg_name="$2"
    local mem_frac="$3"
    local dcp="$4"
    local input_len="$5"
    local output_len="$6"
    local workload_tag="$7"
    shift 7
    local concurrencies=("$@")

    local OUTPUT_DIR="${BASE_OUTPUT}/${scenario_name}/${cfg_name}/${workload_tag}"

    if [ "$RUN_MODE" = "accuracy" ]; then
        if [ -f "$OUTPUT_DIR/accuracy_gsm8k.txt" ] && grep -q "Accuracy:" "$OUTPUT_DIR/accuracy_gsm8k.txt" 2>/dev/null; then
            echo "Skipping ${cfg_name}/${workload_tag} — accuracy already done"
            return 0
        fi
    elif [ "$RUN_MODE" = "perf" ]; then
        local last_cc="${concurrencies[-1]}"
        if [ -f "$OUTPUT_DIR/cc${last_cc}.txt" ] && grep -q "Output token throughput" "$OUTPUT_DIR/cc${last_cc}.txt" 2>/dev/null; then
            echo "Skipping ${cfg_name}/${workload_tag} — perf already done"
            return 0
        fi
    fi

    mkdir -p "$OUTPUT_DIR"
    kill_server
    start_server "$cfg_name" "$mem_frac" "$dcp"

    if ! wait_for_server; then
        local SERVER_LOG="/tmp/sglang_server_${cfg_name}.log"
        echo "ERROR: Server start failure for ${cfg_name} (see ${SERVER_LOG})"
        kill_server
        return 1
    fi

    if [ "$RUN_MODE" = "accuracy" ] || [ "$RUN_MODE" = "all" ]; then
        run_accuracy "$OUTPUT_DIR"
    fi
    if [ "$RUN_MODE" = "perf" ] || [ "$RUN_MODE" = "all" ]; then
        run_perf "$OUTPUT_DIR" "$input_len" "$output_len" "${concurrencies[@]}"
    fi
    kill_server
}

# ============================================================
# Workload: 32K/4K (matches PR Table 1)
# PR results (H20): tp8 4916 TPS @ bs64, dcp2 5679 TPS @ bs96
# ============================================================
run_32k_4k() {
    echo ""
    echo "======================================================="
    echo "Workload: Qwen3-235B 32K/4K (PR #14982 Table 1)"
    echo "======================================================="

    local concurrencies=(1 32 48 64 80 96)

    run_config "pr14982_32k_4k" "tp8_fi" "0.90" "0" \
        32000 4000 "in32k_out4k" "${concurrencies[@]}"

    run_config "pr14982_32k_4k" "tp8_dcp2_fi" "0.85" "2" \
        32000 4000 "in32k_out4k" "${concurrencies[@]}"
}

# ============================================================
# Workload: 32K/8K (matches PR Table 2)
# PR results (H20): tp8 3620 TPS @ bs64, dcp2 4415 TPS @ bs80
# ============================================================
run_32k_8k() {
    echo ""
    echo "======================================================="
    echo "Workload: Qwen3-235B 32K/8K (PR #14982 Table 2)"
    echo "======================================================="

    local concurrencies=(1 32 48 64 80)

    run_config "pr14982_32k_8k" "tp8_fi" "0.90" "0" \
        32000 8000 "in32k_out8k" "${concurrencies[@]}"

    run_config "pr14982_32k_8k" "tp8_dcp2_fi" "0.85" "2" \
        32000 8000 "in32k_out8k" "${concurrencies[@]}"
}


# ---- Main ----
echo "======================================================="
echo "PR #14982 Reproduction: DCP for GQA (FlashInfer)"
echo "  Model:    ${MODEL}"
echo "  Branch:   ${BRANCH} (${HASH})"
echo "  Output:   ${BASE_OUTPUT}/"
echo "  Scenario: ${SCENARIO_FILTER}"
echo "  Mode:     ${RUN_MODE}"
echo "======================================================="
echo ""

case "$SCENARIO_FILTER" in
    32k_4k)  run_32k_4k ;;
    32k_8k)  run_32k_8k ;;
    all)
        run_32k_4k
        run_32k_8k
        ;;
    *)
        echo "Unknown scenario: $SCENARIO_FILTER"
        echo "Usage: $0 [32k_4k|32k_8k|all] [accuracy|perf|all]"
        exit 1
        ;;
esac

echo ""
echo "======================================================="
echo "All benchmarks complete! Results in: ${BASE_OUTPUT}/"
echo "======================================================="
