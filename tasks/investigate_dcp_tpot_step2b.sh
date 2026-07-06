#!/bin/bash
###############################################################################
# DCP TPOT Investigation — Step 2b: Find concurrency ceiling & test other ctxs
#
# Mixed chunk fixes c=2 at 512K but c=8 crashes (watchdog timeout 300s).
# This script finds the concurrency ceiling at 512K, then tests 256K and 1M.
#
# Tests:
#   2E: mixed-chunk + chunk=16384 @ 512K c=4 (find ceiling between c=2 and c=8)
#   2F: mixed-chunk + chunk=16384 @ 256K c=1,2,4,8 (should all work)
#   2G: mixed-chunk + chunk=16384 @ 1M c=1,2 (may crash at c=2)
###############################################################################
set -u

MODEL="deepseek-ai/DeepSeek-V2-Lite"
HOST="0.0.0.0"
PORT=30000
BENCH_HOST="127.0.0.1"
NUM_PROMPTS=5
RESULTS_DIR="/output/investigation_step2b"
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
        sleep 15
        pkill -9 -f "sglang" 2>/dev/null || true
        sleep 5
    fi
    echo "Server stopped."
}

wait_for_server() {
    log "Waiting for server (up to ${MAX_WAIT}s)..."
    local elapsed=0
    while [ $elapsed -lt $MAX_WAIT ]; do
        if curl -s "http://${BENCH_HOST}:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${elapsed}s"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        [ $((elapsed % 30)) -eq 0 ] && echo "  ... still waiting (${elapsed}s)"
    done
    echo "ERROR: Server did not start within ${MAX_WAIT}s"
    return 1
}

start_server() {
    local test_name="$1"
    local ctx_len="$2"
    local mem_frac="$3"
    local chunk_size="$4"

    log "Starting server: ${test_name} (ctx=${ctx_len}, mem=${mem_frac}, chunk=${chunk_size})"
    mkdir -p "${RESULTS_DIR}/${test_name}"

    eval "python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size 8 --trust-remote-code \
        --context-length $ctx_len --mem-fraction-static $mem_frac \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend ag_rs --chunked-prefill-size $chunk_size \
        --enable-mixed-chunk \
        --host $HOST --port $PORT" \
        > "${RESULTS_DIR}/${test_name}/server.log" 2>&1 &

    wait_for_server
}

run_bench() {
    local test_name="$1" concurrency="$2" isl="$3"
    local outfile="${RESULTS_DIR}/${test_name}/bench_c${concurrency}.log"
    echo "--- ${test_name} c=${concurrency} ISL=${isl} ---"

    # Check server is still alive before benchmarking
    if ! curl -s "http://${BENCH_HOST}:${PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: Server died before c=${concurrency} benchmark"
        echo "SERVER_CRASHED" > "$outfile"
        return 1
    fi

    python3 -m sglang.bench_serving \
        --backend sglang --host "$BENCH_HOST" --port "$PORT" \
        --model "$MODEL" --dataset-name random \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len "$isl" --random-output-len 1024 \
        --random-range-ratio 0.0 --disable-ignore-eos \
        --max-concurrency "$concurrency" \
        2>&1 | tee "$outfile"
    echo ""
}

# ── Test 2E: 512K c=4 ─────────────────────────────────────────────────────
test_2e() {
    log "TEST 2E: mixed-chunk + chunk=16384 @ 512K c=4"
    echo "Find if c=4 works (c=2 ok, c=8 crashes)"

    kill_server
    start_server "2E_mc_512K_c4" 524288 0.75 16384

    run_bench "2E_mc_512K_c4" 1 523264
    run_bench "2E_mc_512K_c4" 2 523264
    run_bench "2E_mc_512K_c4" 4 523264

    kill_server
    log "TEST 2E COMPLETE"
}

# ── Test 2F: 256K c=1,2,4,8 ───────────────────────────────────────────────
test_2f() {
    log "TEST 2F: mixed-chunk + chunk=16384 @ 256K c=1,2,4,8"
    echo "256K should have less DCP overhead per forward pass"

    kill_server
    start_server "2F_mc_256K" 262144 0.80 16384

    for c in 1 2 4 8; do
        run_bench "2F_mc_256K" "$c" 261120
    done

    kill_server
    log "TEST 2F COMPLETE"
}

# ── Test 2G: 1M c=1,2 ─────────────────────────────────────────────────────
test_2g() {
    log "TEST 2G: mixed-chunk + chunk=16384 @ 1M c=1,2"
    echo "1M context — may be even slower per forward pass"

    kill_server
    start_server "2G_mc_1M" 1048576 0.65 16384

    run_bench "2G_mc_1M" 1 1047552
    run_bench "2G_mc_1M" 2 1047552

    kill_server
    log "TEST 2G COMPLETE"
}

# ── Summary ────────────────────────────────────────────────────────────────
summarize() {
    log "STEP 2b SUMMARY"
    echo ""
    echo "| Test | Context | Concurrency | Median TPOT | Mean TPOT | Status |"
    echo "|------|---------|:-----------:|:-----------:|:---------:|:------:|"

    for test_dir in "${RESULTS_DIR}"/*/; do
        test_name=$(basename "$test_dir")
        for logf in "${test_dir}"/bench_c*.log; do
            [ ! -f "$logf" ] && continue
            c=$(echo "$logf" | grep -oP 'c\K\d+')
            if grep -q "SERVER_CRASHED" "$logf" 2>/dev/null; then
                echo "| ${test_name} | — | ${c} | — | — | CRASHED |"
            else
                med=$(grep 'Median TPOT' "$logf" 2>/dev/null | awk '{print $NF}' || echo "—")
                mean=$(grep 'Mean TPOT' "$logf" 2>/dev/null | awk '{print $NF}' || echo "—")
                echo "| ${test_name} | — | ${c} | ${med} | ${mean} | OK |"
            fi
        done
    done
    echo ""
}

main() {
    mkdir -p "$RESULTS_DIR"
    log "DCP TPOT INVESTIGATION — STEP 2b"
    echo "GPU status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    echo ""

    local test="${1:-all}"
    case "$test" in
        2E|2e) test_2e ;;
        2F|2f) test_2f ;;
        2G|2g) test_2g ;;
        all) test_2e; test_2f; test_2g; summarize ;;
        summary) summarize ;;
        *) echo "Usage: $0 [2E|2F|2G|all|summary]"; exit 1 ;;
    esac
}

main "$@"
