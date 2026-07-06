#!/bin/bash
###############################################################################
# DCP TPOT Investigation — Step 2: Tune mixed chunk for stability
#
# Step 1 confirmed: --enable-mixed-chunk fixes TPOT degradation but crashes
# at c=8 with chunked-prefill-size=32768 (watchdog timeout).
#
# This script tests:
#   2A: mixed-chunk + chunked-prefill-size=16384 @ 512K (c=1,2,8)
#   2B: mixed-chunk + chunked-prefill-size=8192  @ 512K (c=1,2,8)
#   2C: best chunk size from 2A/2B @ 256K (c=1,2,8)
#   2D: best chunk size from 2A/2B @ 1M   (c=1,2,8)
#
# Usage:
#   /output/investigate_dcp_tpot_step2.sh [2A|2B|2C|2D|all|summary]
###############################################################################
set -u

MODEL="deepseek-ai/DeepSeek-V2-Lite"
HOST="0.0.0.0"
PORT=30000
BENCH_HOST="127.0.0.1"
NUM_PROMPTS=5
RESULTS_DIR="/output/investigation_step2"
MAX_WAIT=600

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
    log "Stopping server..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang.srt.managers" 2>/dev/null || true
    sleep 10
    local gpu_procs
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -gt 0 ]; then
        echo "WARNING: $gpu_procs GPU processes still running. Force killing..."
        sleep 15
        pkill -9 -f "sglang" 2>/dev/null || true
        sleep 5
    fi
    echo "Server stopped."
}

wait_for_server() {
    log "Waiting for server to be ready (up to ${MAX_WAIT}s)..."
    local elapsed=0
    while [ $elapsed -lt $MAX_WAIT ]; do
        if curl -s "http://${BENCH_HOST}:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${elapsed}s"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  ... still waiting (${elapsed}s elapsed)"
        fi
    done
    echo "ERROR: Server did not start within ${MAX_WAIT}s"
    return 1
}

start_server() {
    local test_name="$1"
    local ctx_len="$2"
    local mem_frac="$3"
    local chunk_size="$4"
    local extra_args="${5:-}"

    log "Starting server for test ${test_name}"
    echo "Context: ${ctx_len}, mem_frac: ${mem_frac}, chunk_size: ${chunk_size}"
    echo "Extra args: ${extra_args:-none}"

    mkdir -p "${RESULTS_DIR}/${test_name}"

    local cmd="python3 -m sglang.launch_server \
        --model-path $MODEL \
        --tp-size 8 \
        --trust-remote-code \
        --context-length $ctx_len \
        --mem-fraction-static $mem_frac \
        --attention-backend flashinfer \
        --disable-radix-cache \
        --enable-symm-mem \
        --dcp-comm-backend ag_rs \
        --chunked-prefill-size $chunk_size \
        --enable-mixed-chunk \
        --host $HOST \
        --port $PORT \
        $extra_args"

    echo "Command: $cmd"
    eval "$cmd" > "${RESULTS_DIR}/${test_name}/server.log" 2>&1 &

    wait_for_server
}

run_bench() {
    local test_name="$1"
    local concurrency="$2"
    local isl="$3"
    local osl=1024

    local outfile="${RESULTS_DIR}/${test_name}/bench_c${concurrency}.log"
    echo "--- Benchmarking ${test_name} c=${concurrency} (ISL=${isl}) ---"

    python3 -m sglang.bench_serving \
        --backend sglang \
        --host "$BENCH_HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --dataset-name random \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len "$isl" \
        --random-output-len "$osl" \
        --random-range-ratio 0.0 \
        --disable-ignore-eos \
        --max-concurrency "$concurrency" \
        2>&1 | tee "$outfile"

    echo ""
}

run_concurrency_sweep() {
    local test_name="$1"
    local isl="$2"

    for c in 1 2 8; do
        run_bench "$test_name" "$c" "$isl"
    done
}

# ── Test 2A: mixed-chunk + chunk_size=16384 @ 512K ────────────────────────
test_2a() {
    log "TEST 2A: mixed-chunk + chunked-prefill-size=16384 @ 512K"
    echo "Hypothesis: halving chunk size prevents c=8 watchdog timeout"

    kill_server
    start_server "2A_mc_chunk16k_512K" 524288 0.75 16384 ""
    run_concurrency_sweep "2A_mc_chunk16k_512K" 523264
    kill_server

    log "TEST 2A COMPLETE"
}

# ── Test 2B: mixed-chunk + chunk_size=8192 @ 512K ─────────────────────────
test_2b() {
    log "TEST 2B: mixed-chunk + chunked-prefill-size=8192 @ 512K"
    echo "Hypothesis: further reducing chunk size for stability"

    kill_server
    start_server "2B_mc_chunk8k_512K" 524288 0.75 8192 ""
    run_concurrency_sweep "2B_mc_chunk8k_512K" 523264
    kill_server

    log "TEST 2B COMPLETE"
}

# ── Test 2C: best chunk size @ 256K ───────────────────────────────────────
test_2c() {
    log "TEST 2C: mixed-chunk + chunked-prefill-size=16384 @ 256K"

    kill_server
    start_server "2C_mc_chunk16k_256K" 262144 0.80 16384 ""
    run_concurrency_sweep "2C_mc_chunk16k_256K" 261120
    kill_server

    log "TEST 2C COMPLETE"
}

# ── Test 2D: best chunk size @ 1M ─────────────────────────────────────────
test_2d() {
    log "TEST 2D: mixed-chunk + chunked-prefill-size=16384 @ 1M"

    kill_server
    start_server "2D_mc_chunk16k_1M" 1048576 0.65 16384 ""
    run_concurrency_sweep "2D_mc_chunk16k_1M" 1047552
    kill_server

    log "TEST 2D COMPLETE"
}

# ── Summary ────────────────────────────────────────────────────────────────
summarize() {
    log "STEP 2 SUMMARY"

    echo ""
    echo "Reference baselines (Step 1, no mixed-chunk, chunk=32768):"
    echo "  512K c=1: 6.98ms median | c=2: 51.5ms median / 796ms mean | c=8: 1229ms median"
    echo "  Step 1 mixed-chunk (chunk=32768): c=1: 6.98ms | c=2: 8.0ms | c=8: CRASHED"
    echo ""
    echo "Step 2 results (all with --enable-mixed-chunk):"
    echo ""
    echo "| Test | Context | Chunk Size | c=1 Median | c=2 Median | c=2 Mean | c=8 Median | c=8 Mean |"
    echo "|------|---------|:----------:|:----------:|:----------:|:--------:|:----------:|:--------:|"

    for test_dir in "${RESULTS_DIR}"/*/; do
        test_name=$(basename "$test_dir")
        c1_med="—"; c2_med="—"; c2_mean="—"; c8_med="—"; c8_mean="—"

        for c in 1 2 8; do
            logf="${test_dir}/bench_c${c}.log"
            [ ! -f "$logf" ] && continue
            med=$(grep 'Median TPOT' "$logf" 2>/dev/null | awk '{print $NF}' || echo "—")
            mean=$(grep 'Mean TPOT' "$logf" 2>/dev/null | awk '{print $NF}' || echo "—")
            case $c in
                1) c1_med="$med" ;;
                2) c2_med="$med"; c2_mean="$mean" ;;
                8) c8_med="$med"; c8_mean="$mean" ;;
            esac
        done

        echo "| ${test_name} | — | — | ${c1_med} | ${c2_med} | ${c2_mean} | ${c8_med} | ${c8_mean} |"
    done

    echo ""
}

# ── Main ───────────────────────────────────────────────────────────────────
main() {
    mkdir -p "$RESULTS_DIR"

    log "DCP TPOT INVESTIGATION — STEP 2"
    echo "Testing mixed-chunk with reduced chunked-prefill-size"
    echo "Results dir: $RESULTS_DIR"
    echo ""

    echo "GPU status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    echo ""

    local test_to_run="${1:-all}"

    case "$test_to_run" in
        2A|2a) test_2a ;;
        2B|2b) test_2b ;;
        2C|2c) test_2c ;;
        2D|2d) test_2d ;;
        all)
            test_2a
            test_2b
            test_2c
            test_2d
            summarize
            ;;
        summary) summarize ;;
        *)
            echo "Usage: $0 [2A|2B|2C|2D|all|summary]"
            exit 1
            ;;
    esac
}

main "$@"
