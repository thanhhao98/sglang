#!/usr/bin/env bash
# TPA B200 benchmark: accuracy (GSM8K) + throughput for pure_tp, dcp, tpa configs
#
# Adapted from bench_dcp_serving.sh for B200 with FlashInfer backend.
# Model: Qwen/CodeQwen1.5-7B-Chat (32 Q heads, 4 KV heads)
#
# Usage:
#   bash bench_scripts/bench_tpa_b200.sh
#
# Run inside docker on colossus_b200_1:
#   docker exec -it sglang-tpa-bench bash -c "cd /sgl-workspace/sglang && bash bench_scripts/bench_tpa_b200.sh"

set -euo pipefail

HOST=127.0.0.1
PORT=8188
MODEL=Qwen/CodeQwen1.5-7B-Chat
CONTEXT_LENGTH=163840

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT="${SCRIPT_DIR}/../benchmark/dcp/results/tpa_b200_$(date +%Y%m%d_%H%M%S)"

COMMON_ENV="SGLANG_DCP_SYMM_ONLY=true NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1"

COMMON_ARGS="--model-path $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --enable-cache-report --log-level info --tp-size 8 \
--max-running-requests 128 --chunked-prefill-size 32768 \
--context-length $CONTEXT_LENGTH --disable-radix-cache --enable-symm-mem"

CONCURRENCIES=(1 4 8 16 32 64)

# Config format: NAME|MEM_FRAC|DCP_SIZE|DCP_COMM|ATTN_TP_SIZE
CONFIGS=(
    "tp8_flashinfer|0.85|0||"
    "tp8_dcp2_a2a|0.85|2|a2a|"
    "tp8_tpa2_dcp4_a2a|0.85|4|a2a|2"
    "tp8_tpa4_dcp2_a2a|0.85|2|a2a|4"
)

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
    sleep 10
}

run_accuracy() {
    local output_dir="$1"
    local acc_file="${output_dir}/accuracy_gsm8k.txt"
    echo "Running accuracy test -> ${acc_file}"
    python3 benchmark/gsm8k/bench_sglang.py \
        --parallel 64 \
        --host "$HOST" --port "$PORT" 2>&1 | tee "$acc_file"
}

run_perf() {
    local output_dir="$1"
    echo "Running perf benchmarks -> ${output_dir}"
    for C in "${CONCURRENCIES[@]}"; do
        NUM_PROMPTS=$((C * 5))
        FILE_NAME="${output_dir}/cc${C}.txt"

        echo "--- Concurrency=$C, Prompts=$NUM_PROMPTS -> $FILE_NAME ---"

        python3 -m sglang.bench_serving --backend sglang \
            --host "$HOST" --port "$PORT" \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len 4000 \
            --random-output-len 1500 \
            --random-range-ratio 0.1 \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$C" \
            --disable-ignore-eos 2>&1 | tee "$FILE_NAME"
    done
}

start_server() {
    local cfg_name="$1"
    local mem_frac="$2"
    local dcp="$3"
    local dcp_comm="$4"
    local attn_tp="$5"

    local extra_args=""
    if [ "$dcp" -gt 0 ]; then
        extra_args="--dcp-size ${dcp} --dcp-comm-backend ${dcp_comm}"
    fi
    if [ -n "$attn_tp" ]; then
        extra_args="${extra_args} --attention-tensor-parallel-size ${attn_tp}"
    fi

    echo "======================================================="
    echo "Starting: ${cfg_name} (mem=${mem_frac} dcp=${dcp} comm=${dcp_comm} attn_tp=${attn_tp})"
    echo "======================================================="

    eval "${COMMON_ENV} python3 -m sglang.launch_server ${COMMON_ARGS} \
        --mem-fraction-static ${mem_frac} \
        ${extra_args}" &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"
}

# ---- Main loop ----
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r CFG_NAME MEM_FRAC DCP DCP_COMM ATTN_TP <<< "$cfg"

    OUTPUT_DIR="${BASE_OUTPUT}/${CFG_NAME}"
    mkdir -p "$OUTPUT_DIR"

    kill_server
    start_server "$CFG_NAME" "$MEM_FRAC" "$DCP" "$DCP_COMM" "$ATTN_TP"

    if ! wait_for_server; then
        echo "Skipping ${CFG_NAME} due to server start failure"
        kill_server
        continue
    fi

    run_accuracy "$OUTPUT_DIR"
    run_perf "$OUTPUT_DIR"
    kill_server
done

echo ""
echo "======================================================="
echo "All benchmarks complete! Results in: ${BASE_OUTPUT}/"
echo "======================================================="
