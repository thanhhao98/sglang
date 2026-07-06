#!/bin/bash
###############################################################################
# DCP TPOT Degradation Investigation Script
#
# Target: B200 Node 1, sglang-bench container
# Context: 512K (ISL=523264, OSL=1024)
# Model: deepseek-ai/DeepSeek-V2-Lite
#
# Tests:
#   1A: Baseline (reproduce degradation at c=1 and c=2)
#   1B: --enable-mixed-chunk (test H1: scheduler prefill-blocking-decode)
#   1C: --disable-cuda-graph (test H3: CUDA graph batch-size mismatch)
#   1D: Sequential single requests (isolate decode-only TPOT)
#   1E: --num-continuous-decode-steps 100 (test decode starvation)
#       NOTE: This flag is defined in server_args but NOT wired into the
#       scheduler loop yet, so it likely has no effect. Including it for
#       completeness — skip if 1A/1B/1C are conclusive.
#
# Usage:
#   Copy this script to the sglang-bench container on B200 Node 1, then:
#     chmod +x /output/investigate_dcp_tpot.sh
#     /output/investigate_dcp_tpot.sh 2>&1 | tee /output/investigation_results.log
#
#   Or run individual tests:
#     /output/investigate_dcp_tpot.sh 1A
#     /output/investigate_dcp_tpot.sh 1B
#     /output/investigate_dcp_tpot.sh 1C
#     /output/investigate_dcp_tpot.sh 1D
#     /output/investigate_dcp_tpot.sh 1E
###############################################################################
set -u

# ── Configuration ──────────────────────────────────────────────────────────
MODEL="deepseek-ai/DeepSeek-V2-Lite"
HOST="0.0.0.0"
PORT=30000
BENCH_HOST="127.0.0.1"

# 512K context
CONTEXT_LEN=524288
ISL=523264
OSL=1024
MEM_FRAC=0.75

# Benchmark params
NUM_PROMPTS=5
RESULTS_DIR="/output/investigation"

# Server health check
MAX_WAIT=600  # seconds to wait for server startup
HEALTH_URL="http://${BENCH_HOST}:${PORT}/health"

# ── Common env vars for DCP8 ──────────────────────────────────────────────
export SGLANG_DCP=8
export SGLANG_DCP_SYMM_ONLY=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# ── Helper functions ──────────────────────────────────────────────────────
log() {
    echo ""
    echo "=================================================================="
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $*"
    echo "=================================================================="
    echo ""
}

kill_server() {
    log "Stopping server..."
    # Kill any running sglang server processes
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang.srt.managers" 2>/dev/null || true
    # Wait for GPU memory to be released
    sleep 10
    # Verify GPUs are free
    local gpu_procs
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -gt 0 ]; then
        echo "WARNING: $gpu_procs GPU processes still running. Waiting 15 more seconds..."
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
        if curl -s "$HEALTH_URL" > /dev/null 2>&1; then
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
    shift
    local extra_args="$*"

    log "Starting server for test ${test_name}"
    echo "Extra args: ${extra_args:-none}"

    mkdir -p "${RESULTS_DIR}/${test_name}"

    local base_cmd="python3 -m sglang.launch_server \
        --model-path $MODEL \
        --tp-size 8 \
        --trust-remote-code \
        --context-length $CONTEXT_LEN \
        --mem-fraction-static $MEM_FRAC \
        --attention-backend flashinfer \
        --disable-radix-cache \
        --enable-symm-mem \
        --dcp-comm-backend ag_rs \
        --chunked-prefill-size 32768 \
        --host $HOST \
        --port $PORT \
        $extra_args"

    echo "Server command: $base_cmd"
    eval "$base_cmd" > "${RESULTS_DIR}/${test_name}/server.log" 2>&1 &

    wait_for_server
}

run_bench() {
    local test_name="$1"
    local concurrency="$2"
    local extra_bench_args="${3:-}"

    local outfile="${RESULTS_DIR}/${test_name}/bench_c${concurrency}.log"
    echo "--- Benchmarking ${test_name} c=${concurrency} ---"

    python3 -m sglang.bench_serving \
        --backend sglang \
        --host "$BENCH_HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --dataset-name random \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio 0.0 \
        --disable-ignore-eos \
        --max-concurrency "$concurrency" \
        $extra_bench_args \
        2>&1 | tee "$outfile"

    echo ""
}

extract_tpot() {
    # Extract TPOT from bench output file
    local logfile="$1"
    grep -oP 'TPOT \(ms\):\s+\K[\d.]+' "$logfile" 2>/dev/null || echo "N/A"
}

# ── Test 1A: Baseline ─────────────────────────────────────────────────────
test_1a() {
    log "TEST 1A: Baseline DCP8 ag_rs 512K (reproduce degradation)"
    echo "Expected: c=1 ~7ms TPOT, c=2 ~793ms TPOT"

    kill_server
    start_server "1A_baseline" ""

    run_bench "1A_baseline" 1
    run_bench "1A_baseline" 2
    run_bench "1A_baseline" 8

    kill_server

    log "TEST 1A COMPLETE"
    echo "Results in ${RESULTS_DIR}/1A_baseline/"
}

# ── Test 1B: Enable mixed chunk (H1 test) ─────────────────────────────────
test_1b() {
    log "TEST 1B: DCP8 ag_rs 512K + --enable-mixed-chunk"
    echo "Tests H1: scheduler prefill-blocking-decode"
    echo "If TPOT at c=2 drops to <50ms → H1 confirmed"

    kill_server
    start_server "1B_mixed_chunk" --enable-mixed-chunk

    run_bench "1B_mixed_chunk" 1
    run_bench "1B_mixed_chunk" 2
    run_bench "1B_mixed_chunk" 8

    kill_server

    log "TEST 1B COMPLETE"
    echo "Results in ${RESULTS_DIR}/1B_mixed_chunk/"
}

# ── Test 1C: Disable CUDA graph (H3 test) ─────────────────────────────────
test_1c() {
    log "TEST 1C: DCP8 ag_rs 512K + --disable-cuda-graph"
    echo "Tests H3: CUDA graph batch-size mismatch"
    echo "If c=1 gets slower but c=2 stays ~793ms → graphs not the issue"
    echo "If c=2 improves → CUDA graph batch-size mismatch was the issue"

    kill_server
    start_server "1C_no_cuda_graph" --disable-cuda-graph

    run_bench "1C_no_cuda_graph" 1
    run_bench "1C_no_cuda_graph" 2
    run_bench "1C_no_cuda_graph" 8

    kill_server

    log "TEST 1C COMPLETE"
    echo "Results in ${RESULTS_DIR}/1C_no_cuda_graph/"
}

# ── Test 1D: Sequential requests (isolate decode) ─────────────────────────
test_1d() {
    log "TEST 1D: Sequential single requests (pure decode B=1)"
    echo "Send 3 sequential requests at c=1 to measure pure decode TPOT"
    echo "Expected: ~7ms consistently (no prefill interference)"

    kill_server
    start_server "1D_sequential" ""

    # Run 3 separate single-request benchmarks
    for i in 1 2 3; do
        echo "--- Sequential run $i/3 ---"
        python3 -m sglang.bench_serving \
            --backend sglang \
            --host "$BENCH_HOST" \
            --port "$PORT" \
            --model "$MODEL" \
            --dataset-name random \
            --num-prompts 1 \
            --random-input-len "$ISL" \
            --random-output-len "$OSL" \
            --random-range-ratio 0.0 \
            --disable-ignore-eos \
            --max-concurrency 1 \
            2>&1 | tee "${RESULTS_DIR}/1D_sequential/bench_run${i}.log"
        echo ""
    done

    kill_server

    log "TEST 1D COMPLETE"
    echo "Results in ${RESULTS_DIR}/1D_sequential/"
}

# ── Test 1E: High continuous decode steps ──────────────────────────────────
test_1e() {
    log "TEST 1E: DCP8 ag_rs 512K + --num-continuous-decode-steps 100"
    echo "Tests decode starvation between continuous decode steps"
    echo "WARNING: This flag may not be wired into the scheduler yet."
    echo "If TPOT improves → decode starvation was the cause"

    kill_server
    start_server "1E_high_decode_steps" --num-continuous-decode-steps 100

    run_bench "1E_high_decode_steps" 1
    run_bench "1E_high_decode_steps" 2
    run_bench "1E_high_decode_steps" 8

    kill_server

    log "TEST 1E COMPLETE"
    echo "Results in ${RESULTS_DIR}/1E_high_decode_steps/"
}

# ── Summary extraction ─────────────────────────────────────────────────────
summarize() {
    log "SUMMARY OF ALL RESULTS"

    echo ""
    echo "| Test | Description | c=1 TPOT (ms) | c=2 TPOT (ms) | c=8 TPOT (ms) |"
    echo "|------|-------------|:-------------:|:-------------:|:-------------:|"

    for test_dir in "${RESULTS_DIR}"/*/; do
        test_name=$(basename "$test_dir")
        tpot_c1="N/A"
        tpot_c2="N/A"
        tpot_c8="N/A"

        [ -f "${test_dir}/bench_c1.log" ] && tpot_c1=$(grep 'Median TPOT' "${test_dir}/bench_c1.log" 2>/dev/null | grep -oP '[\d.]+$' || echo "N/A")
        [ -f "${test_dir}/bench_c2.log" ] && tpot_c2=$(grep 'Median TPOT' "${test_dir}/bench_c2.log" 2>/dev/null | grep -oP '[\d.]+$' || echo "N/A")
        [ -f "${test_dir}/bench_c8.log" ] && tpot_c8=$(grep 'Median TPOT' "${test_dir}/bench_c8.log" 2>/dev/null | grep -oP '[\d.]+$' || echo "N/A")

        # For 1D, extract from individual runs
        if [ "$test_name" = "1D_sequential" ]; then
            tpot_runs=""
            for rf in "${test_dir}"/bench_run*.log; do
                t=$(grep 'Median TPOT' "$rf" 2>/dev/null | grep -oP '[\d.]+$' || echo "?")
                tpot_runs="${tpot_runs}${t}, "
            done
            tpot_c1="${tpot_runs%%, }"
            tpot_c2="—"
            tpot_c8="—"
        fi

        echo "| ${test_name} | — | ${tpot_c1} | ${tpot_c2} | ${tpot_c8} |"
    done

    echo ""
    echo "Interpretation guide:"
    echo "  1A baseline:      c=1 ~7ms, c=2 ~793ms (confirm reproduction)"
    echo "  1B mixed-chunk:   c=2 <50ms → H1 confirmed (scheduler blocking)"
    echo "  1C no-cuda-graph: c=2 improves → H3 confirmed (CUDA graph mismatch)"
    echo "  1D sequential:    all ~7ms (expected, pure B=1 decode)"
    echo "  1E decode-steps:  c=2 improves → decode starvation between steps"
    echo ""
}

# ── Main ───────────────────────────────────────────────────────────────────
main() {
    mkdir -p "$RESULTS_DIR"

    log "DCP TPOT DEGRADATION INVESTIGATION"
    echo "Target: B200 Node 1, 512K context, DCP8 ag_rs"
    echo "Model: $MODEL"
    echo "Results dir: $RESULTS_DIR"
    echo ""

    # Check GPU availability
    echo "GPU status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    echo ""

    local test_to_run="${1:-all}"

    case "$test_to_run" in
        1A|1a) test_1a ;;
        1B|1b) test_1b ;;
        1C|1c) test_1c ;;
        1D|1d) test_1d ;;
        1E|1e) test_1e ;;
        all)
            test_1a
            test_1b
            test_1c
            test_1d
            # Skip 1E by default since the flag isn't wired yet
            echo ""
            echo "NOTE: Skipping Test 1E (--num-continuous-decode-steps not wired into scheduler)"
            echo "Run manually with: $0 1E"
            echo ""
            summarize
            ;;
        summary) summarize ;;
        *)
            echo "Usage: $0 [1A|1B|1C|1D|1E|all|summary]"
            exit 1
            ;;
    esac
}

main "$@"
