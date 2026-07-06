#!/bin/bash
###############################################################################
# Full C3 DCP8 A2A FlashInfer benchmarks on B200 (optimized code)
# + GSM8K accuracy test
# Runs accuracy + 4 context lengths × 10 concurrency levels = 41 tasks
###############################################################################
set -u

MODEL="deepseek-ai/DeepSeek-V2-Lite"
HOST="0.0.0.0"
PORT=30000
BENCH_HOST="127.0.0.1"
NUM_PROMPTS=5
CONCURRENCIES=(1 2 4 8 16 32 64 128 256 512)

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

kill_server() {
    echo "Stopping server..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang.srt.managers" 2>/dev/null || true
    sleep 10
    local gpu_procs
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -gt 0 ]; then
        sleep 15
    fi
}

wait_for_server() {
    echo "Waiting for server..."
    for i in $(seq 1 180); do
        if curl -s "http://${BENCH_HOST}:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not start"
    return 1
}

run_bench() {
    local RUN_ID=$1
    local CONC=$2
    local ISL=$3
    local OSL=1024

    echo "--- ${RUN_ID} concurrency=${CONC} ---"
    python3 -m sglang.bench_serving --backend sglang \
        --host $BENCH_HOST --port $PORT \
        --model $MODEL --dataset-name random \
        --num-prompts $NUM_PROMPTS \
        --random-input-len $ISL --random-output-len $OSL \
        --random-range-ratio 0.0 --max-concurrency $CONC \
        --disable-ignore-eos \
        2>&1 | tee /output/${RUN_ID}_c${CONC}.log
    echo ""
}

start_a2a_server() {
    local CTX=$1
    local MEM_FRAC=$2

    kill_server

    export SGLANG_DCP=8
    export SGLANG_DCP_SYMM_ONLY=true
    export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

    echo "Starting DCP8 A2A FlashInfer server (ctx=$CTX, mem_frac=$MEM_FRAC)..."
    python3 -m sglang.launch_server \
        --model-path $MODEL \
        --tp-size 8 --trust-remote-code \
        --context-length $CTX --mem-fraction-static $MEM_FRAC \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend a2a --chunked-prefill-size 32768 \
        --host $HOST --port $PORT \
        > /output/server_c3_opt_b200_${CTX}.log 2>&1 &

    wait_for_server
}

echo "================================================"
echo "B200: C3 DCP8 A2A FlashInfer - Optimized Code"
echo "================================================"

# --- Accuracy test (128K server, full GSM8K) ---
start_a2a_server 131072 0.85
echo ""
echo "=== GSM8K Accuracy Test (1319 questions) ==="
python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
    --num-questions 1319 \
    --parallel 32 \
    --host $BENCH_HOST --port $PORT \
    2>&1 | tee /output/C3_opt_b200_gsm8k.log

# --- 128K perf (reuse server) ---
for C in "${CONCURRENCIES[@]}"; do
    run_bench "C3_opt_b200_128K" $C 130048
done

# --- 256K ---
start_a2a_server 262144 0.80
for C in "${CONCURRENCIES[@]}"; do
    run_bench "C3_opt_b200_256K" $C 261120
done

# --- 512K ---
start_a2a_server 524288 0.75
for C in "${CONCURRENCIES[@]}"; do
    run_bench "C3_opt_b200_512K" $C 523264
done

# --- 1M ---
start_a2a_server 1048576 0.65
for C in "${CONCURRENCIES[@]}"; do
    run_bench "C3_opt_b200_1M" $C 1047552
done

kill_server

echo ""
echo "================================================"
echo "B200 benchmarks complete!"
echo "Results in /output/C3_opt_b200_*.log"
echo "================================================"
