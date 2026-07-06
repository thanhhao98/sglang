#!/bin/bash
# Benchmark --dcp-replicate-q-proj: accuracy (full GSM8K 1319) + performance (bench_serving)
#
# Run from LOCAL machine. Syncs code, then prints server-side commands.
#
# Usage: ./tasks/run_replicate_qproj_bench.sh [h100|b200]

set -euo pipefail

MACHINE="${1:-h100}"

case "$MACHINE" in
    h100)
        SSH_HOST="colossus"
        CODE_PATH="/localhome/local-htphan/helix/sglang"
        DOCKER_CONTAINER="sglang-bench"
        ;;
    b200)
        SSH_HOST="colossus_b200_1"
        CODE_PATH="/localhome/local-htphan/sglang_bench/sglang"
        DOCKER_CONTAINER="sglang-bench"
        ;;
    *)
        echo "Usage: $0 {h100|b200}"
        exit 1
        ;;
esac

echo "============================================="
echo "DCP --dcp-replicate-q-proj Benchmarking on $MACHINE"
echo "============================================="

# --- Step 1: Sync code ---
echo ""
echo "=== Step 1: Sync code to $SSH_HOST ==="
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  /Users/htphan/workspace/DLAlgo/sglang/ \
  ${SSH_HOST}:${CODE_PATH}/

echo ""
echo "Code synced. Run the following inside docker:"
echo "  ssh $SSH_HOST"
echo "  docker exec -it $DOCKER_CONTAINER bash"
echo "  cd /sgl-workspace/sglang/python && pip install -e ."
echo ""

# --- Generate server-side script ---
cat << 'SERVERSCRIPT'
#############################################################
# --dcp-replicate-q-proj Accuracy + Performance Benchmarks
#
# Run inside docker on colossus (H100) or colossus_b200_1 (B200):
#   cd /sgl-workspace/sglang/python && pip install -e .
#
# Branch: htphan/q-project-replication
#############################################################

MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=30000
LOG_DIR="/output/replicate_qproj_bench"
mkdir -p "$LOG_DIR"

DCP_ENV_BASE="SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"

kill_server() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 5
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 2
}

wait_for_server() {
    echo "  Waiting for server..."
    for i in $(seq 1 240); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "  Server ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server did not start within 240s"
    return 1
}

# ============================================================
# PART A: Accuracy (GSM8K full 1319 questions)
# ============================================================
# Covers: a2a + ag_rs, with replicate-q-proj
#         No CG, Regular CG, Piecewise CG

run_accuracy() {
    local tag="$1"
    local comm="$2"       # a2a or ag_rs
    local cg_args="$3"    # CG flags
    local extra_args="$4" # extra server args (e.g. --dcp-replicate-q-proj)

    echo ""
    echo "============================================="
    echo "ACCURACY: $tag"
    echo "============================================="
    kill_server

    env $DCP_ENV_BASE python3 -m sglang.launch_server \
        --model-path "$MODEL" --tp-size 8 --trust-remote-code \
        --mem-fraction-static 0.80 \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend "$comm" --chunked-prefill-size 32768 \
        $cg_args $extra_args \
        --host 0.0.0.0 --port "$PORT" &

    if wait_for_server; then
        echo "  Running GSM8K (1319 questions)..."
        python3 benchmark/gsm8k/bench_sglang.py \
            --num-questions 1319 --parallel 64 \
            --host 127.0.0.1 --port "$PORT" \
            2>&1 | tee "${LOG_DIR}/${tag}_gsm8k.log"
    else
        echo "FAILED: Server did not start" | tee "${LOG_DIR}/${tag}_FAILED.log"
    fi
    kill_server
}

echo ""
echo "###  PART A: Accuracy (GSM8K 1319 questions)  ###"
echo ""

# DCP8 a2a flashinfer + replicate-q-proj
run_accuracy "A1_a2a_repl_noCG"       a2a "--disable-cuda-graph --disable-piecewise-cuda-graph" "--dcp-replicate-q-proj"
run_accuracy "A2_a2a_repl_regularCG"   a2a "--disable-piecewise-cuda-graph"                     "--dcp-replicate-q-proj"
run_accuracy "A3_a2a_repl_piecewiseCG" a2a ""                                                   "--dcp-replicate-q-proj"

# DCP8 ag_rs + replicate-q-proj
run_accuracy "A4_agrs_repl_noCG"       ag_rs "--disable-cuda-graph --disable-piecewise-cuda-graph" "--dcp-replicate-q-proj"
run_accuracy "A5_agrs_repl_regularCG"  ag_rs "--disable-piecewise-cuda-graph"                      "--dcp-replicate-q-proj"
run_accuracy "A6_agrs_repl_piecewiseCG" ag_rs ""                                                   "--dcp-replicate-q-proj"

echo ""
echo "###  PART A COMPLETE  ###"
echo "Accuracy results:"
grep -E "Accuracy|Invalid" "${LOG_DIR}"/A*_gsm8k.log 2>/dev/null || echo "(check logs)"
echo ""

# ============================================================
# PART B: Performance (bench_serving)
# ============================================================
# Same ISL/OSL/concurrency matrix as C2/C3 configs
# Uses piecewise CG (default, best performance)

run_perf() {
    local tag="$1"
    local comm="$2"       # a2a or ag_rs
    local ctx="$3"        # context length
    local mem_frac="$4"
    local extra_args="$5"

    local isl=$((ctx - 1024))

    echo ""
    echo "============================================="
    echo "PERF: ${tag} (ctx=${ctx}, ISL=${isl})"
    echo "============================================="
    kill_server

    env $DCP_ENV_BASE python3 -m sglang.launch_server \
        --model-path "$MODEL" --tp-size 8 --trust-remote-code \
        --context-length "$ctx" --mem-fraction-static "$mem_frac" \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend "$comm" --chunked-prefill-size 32768 \
        $extra_args \
        --host 0.0.0.0 --port "$PORT" &

    if wait_for_server; then
        for conc in 1 2 4 8 16 32 64 128 256 512; do
            echo "  --- c=${conc} ---"
            python3 -m sglang.bench_serving \
                --backend sglang \
                --host 127.0.0.1 --port "$PORT" \
                --dataset-name random \
                --random-input "$isl" --random-output 1024 \
                --random-range-ratio 0.0 \
                --num-prompts 5 \
                --request-rate -1 \
                --max-concurrency "$conc" \
                2>&1 | tee -a "${LOG_DIR}/${tag}_c${conc}.log"
        done
    else
        echo "FAILED: Server did not start" | tee "${LOG_DIR}/${tag}_FAILED.log"
    fi
    kill_server
}

echo ""
echo "###  PART B: Performance (bench_serving)  ###"
echo ""

# C5: DCP8 a2a flashinfer + replicate-q-proj
run_perf "C5_a2a_repl_128K"  a2a 131072  0.50 "--dcp-replicate-q-proj"
run_perf "C5_a2a_repl_256K"  a2a 262144  0.80 "--dcp-replicate-q-proj"
run_perf "C5_a2a_repl_512K"  a2a 524288  0.75 "--dcp-replicate-q-proj"
run_perf "C5_a2a_repl_1M"    a2a 1048576 0.65 "--dcp-replicate-q-proj"

# C6: DCP8 ag_rs + replicate-q-proj
run_perf "C6_agrs_repl_128K"  ag_rs 131072  0.85 "--dcp-replicate-q-proj"
run_perf "C6_agrs_repl_256K"  ag_rs 262144  0.80 "--dcp-replicate-q-proj"
run_perf "C6_agrs_repl_512K"  ag_rs 524288  0.75 "--dcp-replicate-q-proj"
run_perf "C6_agrs_repl_1M"    ag_rs 1048576 0.65 "--dcp-replicate-q-proj"

echo ""
echo "============================================="
echo "ALL BENCHMARKS COMPLETE"
echo "============================================="
echo "Logs in $LOG_DIR"
echo ""
echo "Quick summary:"
echo "  Accuracy: grep -E 'Accuracy|Invalid' $LOG_DIR/A*_gsm8k.log"
echo "  Perf c=1: grep 'mean_tpot_ms' $LOG_DIR/C*_c1.log"
SERVERSCRIPT
