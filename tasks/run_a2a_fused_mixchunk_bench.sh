#!/bin/bash
###############################################################################
# Full C3-fused DCP8 A2A FlashInfer + --enable-mixed-chunk benchmarks
# Tests whether mixed-chunk fixes the TPOT degradation at c>=2
###############################################################################
set -u

MODEL="deepseek-ai/DeepSeek-V2-Lite"
HOST="0.0.0.0"
PORT=30000
BENCH_HOST="127.0.0.1"
NUM_PROMPTS=5
CONCURRENCIES=(1 2 4 8 16 32 64 128 256 512)
TAG="${1:-h100}"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

kill_server() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang.srt.managers" 2>/dev/null || true
    sleep 10
    local gpu_procs
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -gt 0 ]; then sleep 15; fi
}

wait_for_server() {
    for i in $(seq 1 180); do
        if curl -s "http://${BENCH_HOST}:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"; return 0
        fi; sleep 2
    done
    echo "ERROR: Server not ready"; return 1
}

run_bench() {
    local RUN_ID=$1 CONC=$2 ISL=$3 OSL=1024
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

if [ "$TAG" = "h100" ]; then
    MF_128K=0.50
else
    MF_128K=0.85
fi

echo "======================================================="
echo "${TAG}: C3-fused DCP8 A2A + mixed-chunk"
echo "======================================================="

# --- Accuracy (128K) ---
start_server 131072 $MF_128K
echo "=== GSM8K Accuracy (1319 questions) ==="
python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
    --num-questions 1319 --parallel 32 \
    --host $BENCH_HOST --port $PORT \
    2>&1 | tee /output/C3_mc_${TAG}_gsm8k.log

# --- 128K perf ---
for C in "${CONCURRENCIES[@]}"; do
    run_bench "C3_mc_${TAG}_128K" $C 130048
done

# --- 256K ---
start_server 262144 0.80
for C in "${CONCURRENCIES[@]}"; do
    run_bench "C3_mc_${TAG}_256K" $C 261120
done

# --- 512K ---
start_server 524288 0.75
for C in "${CONCURRENCIES[@]}"; do
    run_bench "C3_mc_${TAG}_512K" $C 523264
done

# --- 1M ---
start_server 1048576 0.65
for C in "${CONCURRENCIES[@]}"; do
    run_bench "C3_mc_${TAG}_1M" $C 1047552
done

kill_server
echo ""
echo "======================================================="
echo "${TAG} mixed-chunk benchmarks complete!"
echo "======================================================="
