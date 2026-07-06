#!/bin/bash
###############################################################################
# Simplified mixed-chunk benchmark — skip accuracy, use fewer prompts
# Detects server crashes and restarts between concurrency levels
###############################################################################
set -uo pipefail

MODEL="deepseek-ai/DeepSeek-V2-Lite"
HOST="0.0.0.0"
PORT=30000
BENCH_HOST="127.0.0.1"
NUM_PROMPTS=5
TAG="${1:-h100}"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

kill_server() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang.srt.managers" 2>/dev/null || true
    sleep 10
}

wait_for_server() {
    for i in $(seq 1 180); do
        if curl -s "http://${BENCH_HOST}:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"; return 0
        fi; sleep 2
    done
    echo "ERROR: Server not ready"; return 1
}

start_server() {
    local CTX=$1 MEM_FRAC=$2
    kill_server
    export SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
    echo "Starting server (ctx=$CTX, mem_frac=$MEM_FRAC, mixed-chunk)..."
    python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size 8 --trust-remote-code \
        --context-length $CTX --mem-fraction-static $MEM_FRAC \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend a2a --chunked-prefill-size 32768 \
        --enable-mixed-chunk \
        --host $HOST --port $PORT \
        > /output/server_mc_${TAG}_${CTX}.log 2>&1 &
    wait_for_server
}

run_bench() {
    local RUN_ID=$1 CONC=$2 ISL=$3 OSL=1024
    # Check server health first
    if ! curl -s "http://${BENCH_HOST}:${PORT}/health" > /dev/null 2>&1; then
        echo "--- ${RUN_ID} c=${CONC}: SERVER DOWN, skipping ---"
        echo "SERVER_DOWN" > /output/${RUN_ID}_c${CONC}.log
        return 1
    fi
    echo "--- ${RUN_ID} concurrency=${CONC} ---"
    timeout 600 python3 -m sglang.bench_serving --backend sglang \
        --host $BENCH_HOST --port $PORT \
        --model $MODEL --dataset-name random \
        --num-prompts $NUM_PROMPTS \
        --random-input-len $ISL --random-output-len $OSL \
        --random-range-ratio 0.0 --max-concurrency $CONC \
        --disable-ignore-eos \
        2>&1 | tee /output/${RUN_ID}_c${CONC}.log
    echo ""
}

if [ "$TAG" = "h100" ]; then
    MF_128K=0.50
else
    MF_128K=0.85
fi

echo "======================================================="
echo "${TAG}: A2A fused + mixed-chunk (perf only)"
echo "======================================================="

run_context() {
    local CTX_LABEL=$1 CTX=$2 MEM_FRAC=$3 ISL=$4
    start_server $CTX $MEM_FRAC
    for C in 1 2 4 8 16 32 64 128 256 512; do
        run_bench "C3_mc_${TAG}_${CTX_LABEL}" $C $ISL
        # If server crashed, restart and continue
        if ! curl -s "http://${BENCH_HOST}:${PORT}/health" > /dev/null 2>&1; then
            echo "Server crashed at c=$C, restarting..."
            start_server $CTX $MEM_FRAC
        fi
    done
}

run_context "128K" 131072 $MF_128K 130048
run_context "256K" 262144 0.80 261120
run_context "512K" 524288 0.75 523264
run_context "1M" 1048576 0.65 1047552

kill_server
echo ""
echo "======================================================="
echo "${TAG} mixed-chunk benchmarks complete!"
echo "======================================================="
