#!/bin/bash
###############################################################################
# Verify Scheduler Decode Starvation Hypothesis
#
# Theory: TPOT degradation is proportional to prefill blocking time.
# - Prefill blocking time = (input_len / chunk_size) * time_per_chunk
# - Shorter input → less blocking → needs higher c to see degradation
# - Longer input → more blocking → degradation at lower c
#
# Test: Same server config (512K context, same KV pool), vary input length:
#   ISL=65536 (64K)  → ~2 chunks  → expect mild degradation
#   ISL=131072 (128K) → ~4 chunks  → expect moderate degradation
#   ISL=262144 (256K) → ~8 chunks  → expect more degradation
#   ISL=523264 (512K) → ~16 chunks → expect catastrophic (baseline)
#
# Each input length tested at c=1,2,4,8
#
# If hypothesis is correct:
#   c where TPOT > 100ms should shift right (higher c) for shorter inputs
#   c=1 TPOT should be ~7ms for all input lengths (no interference)
###############################################################################
set -u

MODEL="deepseek-ai/DeepSeek-V2-Lite"
HOST="0.0.0.0"
PORT=30000
BENCH_HOST="127.0.0.1"
NUM_PROMPTS=5
RESULTS_DIR="/output/verify_scheduler"

# Same server config for ALL tests
CONTEXT_LEN=524288
MEM_FRAC=0.75
CHUNK_SIZE=32768

export SGLANG_DCP=8
export SGLANG_DCP_SYMM_ONLY=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

log() {
    echo ""
    echo "=================================================================="
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $*"
    echo "=================================================================="
    echo ""
}

kill_server() {
    echo "Stopping server..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang.srt.managers" 2>/dev/null || true
    sleep 10
    local gpu_procs
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -gt 0 ]; then
        sleep 15; pkill -9 -f "sglang" 2>/dev/null || true; sleep 5
    fi
}

wait_server() {
    echo "Waiting for server..."
    for i in $(seq 1 120); do
        curl -s "http://${BENCH_HOST}:${PORT}/health" >/dev/null 2>&1 && { echo "Server ready after $((i*5))s"; return 0; }
        sleep 5
    done
    echo "ERROR: Server did not start"; return 1
}

start_server() {
    local test_name="$1"
    log "Starting server for ${test_name}"
    mkdir -p "${RESULTS_DIR}/${test_name}"

    eval "python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size 8 --trust-remote-code \
        --context-length $CONTEXT_LEN --mem-fraction-static $MEM_FRAC \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend ag_rs --chunked-prefill-size $CHUNK_SIZE \
        --host $HOST --port $PORT" \
        > "${RESULTS_DIR}/${test_name}/server.log" 2>&1 &

    wait_server
}

run_bench() {
    local test_name="$1" conc="$2" isl="$3"
    local outfile="${RESULTS_DIR}/${test_name}/bench_c${conc}.log"
    echo "--- ${test_name} c=${conc} ISL=${isl} ---"

    python3 -m sglang.bench_serving --backend sglang \
        --host $BENCH_HOST --port $PORT --model $MODEL \
        --dataset-name random --num-prompts $NUM_PROMPTS \
        --random-input-len $isl --random-output-len 1024 \
        --random-range-ratio 0.0 --disable-ignore-eos \
        --max-concurrency $conc \
        2>&1 | tee "$outfile"
    echo ""
}

# Input lengths to test
ISLS=(65536 131072 262144 523264)
ISL_NAMES=("64K" "128K" "256K" "512K")
CONCS=(1 2 4 8)

###############################################################################
# Run all tests with ONE server launch (same KV pool for all)
###############################################################################
main() {
    mkdir -p "$RESULTS_DIR"
    log "SCHEDULER HYPOTHESIS VERIFICATION"
    echo "Server config: ctx=$CONTEXT_LEN, mem_frac=$MEM_FRAC, chunk=$CHUNK_SIZE"
    echo "Same server for ALL input lengths (same KV pool size)"
    echo ""
    echo "GPU status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    echo ""

    kill_server
    start_server "verify"

    for i in "${!ISLS[@]}"; do
        local isl=${ISLS[$i]}
        local name=${ISL_NAMES[$i]}
        log "Testing ISL=${name} (${isl} tokens)"
        echo "Expected chunks: $((isl / CHUNK_SIZE))"

        for conc in "${CONCS[@]}"; do
            run_bench "verify" $conc $isl
            # Rename log to include ISL
            mv "${RESULTS_DIR}/verify/bench_c${conc}.log" \
               "${RESULTS_DIR}/verify/bench_${name}_c${conc}.log" 2>/dev/null
        done
    done

    kill_server

    # Summary
    log "RESULTS SUMMARY"
    echo ""
    echo "Prediction: shorter input → TPOT degradation shifts to higher concurrency"
    echo ""
    echo "| ISL | Chunks | c=1 TPOT | c=2 TPOT | c=4 TPOT | c=8 TPOT |"
    echo "|-----|:------:|:--------:|:--------:|:--------:|:--------:|"

    for i in "${!ISLS[@]}"; do
        local name=${ISL_NAMES[$i]}
        local chunks=$((${ISLS[$i]} / CHUNK_SIZE))
        local tpots=""
        for conc in "${CONCS[@]}"; do
            local f="${RESULTS_DIR}/verify/bench_${name}_c${conc}.log"
            local t=$(grep "Mean TPOT" "$f" 2>/dev/null | awk '{print $NF}')
            tpots="${tpots} | ${t:-FAIL}"
        done
        echo "| ${name} | ${chunks} |${tpots} |"
    done

    echo ""
    echo "| ISL | Chunks | c=1 Median | c=2 Median | c=4 Median | c=8 Median |"
    echo "|-----|:------:|:----------:|:----------:|:----------:|:----------:|"

    for i in "${!ISLS[@]}"; do
        local name=${ISL_NAMES[$i]}
        local chunks=$((${ISLS[$i]} / CHUNK_SIZE))
        local tpots=""
        for conc in "${CONCS[@]}"; do
            local f="${RESULTS_DIR}/verify/bench_${name}_c${conc}.log"
            local t=$(grep "Median TPOT" "$f" 2>/dev/null | awk '{print $NF}')
            tpots="${tpots} | ${t:-FAIL}"
        done
        echo "| ${name} | ${chunks} |${tpots} |"
    done

    echo ""
    log "DONE"
}

main "$@"
