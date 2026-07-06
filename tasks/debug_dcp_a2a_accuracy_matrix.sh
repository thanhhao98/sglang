#!/bin/bash
# Debug DCP A2A Accuracy Issue - Systematic Test Matrix
#
# Run from LOCAL machine. Syncs code to H100, then prints
# commands to run inside docker for each test configuration.
#
# Test Matrix:
#   B1: TP1, no CG (pure baseline)
#   B2: TP1, regular CG
#   B3: TP8, no CG
#   B4: TP8, regular CG
#   D1: DCP8 ag_rs, no CG
#   D2: DCP8 ag_rs, regular CG
#   D3: DCP8 a2a, no CG  (PRIMARY SUSPECT)
#   D4: DCP8 a2a, regular CG
#
# Usage: ./tasks/debug_dcp_a2a_accuracy_matrix.sh [h100|b200]

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

MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=30000

echo "============================================="
echo "DCP A2A Accuracy Debug Matrix on $MACHINE"
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
echo "  cd /sgl-workspace/sglang && pip install -e ."
echo ""

# --- Generate server-side script ---
cat << 'SERVERSCRIPT'
#############################################################
# DCP A2A Accuracy Debug Matrix
#
# Run inside docker:
#   ssh colossus  (or colossus_b200_1)
#   docker exec -it sglang-bench bash
#   cd /sgl-workspace/sglang && pip install -e .
#
# Then run tests one-by-one below. Each test:
#   1) kills any existing server
#   2) launches a new server config
#   3) waits for ready
#   4) runs GSM8K 200 questions
#   5) logs result
#
# Quick diagnosis: run D1 then D3 only (ag_rs vs a2a)
# Full matrix: run all B1-B4, D1-D4
#############################################################

MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=30000
LOG_DIR="/output/debug_a2a_accuracy"
mkdir -p "$LOG_DIR"

kill_server() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 5
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 2
}

wait_for_server() {
    echo "  Waiting for server..."
    for i in $(seq 1 180); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "  Server ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server did not start"
    return 1
}

sanity_check() {
    echo "  --- Sanity check ---"
    curl -s -H 'Content-Type: application/json' "http://localhost:${PORT}/generate" \
        -d '{"text": "What is 2+2? Answer:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}' | python3 -m json.tool
}

run_gsm8k() {
    local tag="$1"
    echo "  --- Running GSM8K (200 questions) ---"
    python3 benchmark/gsm8k/bench_sglang.py \
        --num-questions 200 --parallel 64 \
        --host 127.0.0.1 --port "$PORT" \
        2>&1 | tee "${LOG_DIR}/${tag}_gsm8k.log"
}

run_test() {
    local tag="$1"
    shift
    local env_vars=("$@")

    echo ""
    echo "============================================="
    echo "TEST $tag"
    echo "============================================="
    kill_server

    # Set env vars and launch
    env "${env_vars[@]}" python3 -m sglang.launch_server \
        --model-path "$MODEL" --trust-remote-code \
        --host 0.0.0.0 --port "$PORT" \
        "${SERVER_ARGS[@]}" &

    if wait_for_server; then
        sanity_check 2>&1 | tee "${LOG_DIR}/${tag}_sanity.log"
        run_gsm8k "$tag"
    else
        echo "FAILED: Server did not start" | tee "${LOG_DIR}/${tag}_FAILED.log"
    fi
    kill_server
}

# ============================================================
# Phase 1: Baselines (no DCP)
# ============================================================

echo ""
echo "###  PHASE 1: Baselines (main branch, no DCP)  ###"
echo "###  Expected: all ~0.38                        ###"
echo ""

# B1: TP1, no CG
SERVER_ARGS=(--tp-size 1 --mem-fraction-static 0.85 \
    --disable-cuda-graph --disable-piecewise-cuda-graph)
run_test "B1_tp1_noCG" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# B2: TP1, regular CG
SERVER_ARGS=(--tp-size 1 --mem-fraction-static 0.85 \
    --disable-piecewise-cuda-graph)
run_test "B2_tp1_CG" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# B3: TP8, no CG
SERVER_ARGS=(--tp-size 8 --mem-fraction-static 0.85 \
    --disable-cuda-graph --disable-piecewise-cuda-graph)
run_test "B3_tp8_noCG" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# B4: TP8, regular CG
SERVER_ARGS=(--tp-size 8 --mem-fraction-static 0.85 \
    --disable-piecewise-cuda-graph)
run_test "B4_tp8_CG" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# ============================================================
# Phase 2: DCP configurations
# ============================================================

echo ""
echo "###  PHASE 2: DCP configurations                ###"
echo "###  D1/D2 (ag_rs) should be ~0.38              ###"
echo "###  D3 (a2a no CG) is the primary suspect      ###"
echo ""

DCP_COMMON=(--tp-size 8 --mem-fraction-static 0.85 \
    --disable-radix-cache --enable-symm-mem --chunked-prefill-size 32768)

# D1: DCP8 ag_rs, no CG
SERVER_ARGS=("${DCP_COMMON[@]}" --dcp-comm-backend ag_rs \
    --disable-cuda-graph --disable-piecewise-cuda-graph)
run_test "D1_dcp8_agrs_noCG" \
    SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# D2: DCP8 ag_rs, regular CG
SERVER_ARGS=("${DCP_COMMON[@]}" --dcp-comm-backend ag_rs \
    --disable-piecewise-cuda-graph)
run_test "D2_dcp8_agrs_CG" \
    SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# D3: DCP8 a2a, no CG (PRIMARY SUSPECT - enable debug logging)
SERVER_ARGS=("${DCP_COMMON[@]}" --dcp-comm-backend a2a \
    --disable-cuda-graph --disable-piecewise-cuda-graph)
run_test "D3_dcp8_a2a_noCG" \
    SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    SGLANG_DCP_A2A_DEBUG=1

# D4: DCP8 a2a, regular CG
SERVER_ARGS=("${DCP_COMMON[@]}" --dcp-comm-backend a2a \
    --disable-piecewise-cuda-graph)
run_test "D4_dcp8_a2a_CG" \
    SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# ============================================================
# Summary
# ============================================================

echo ""
echo "============================================="
echo "ALL TESTS COMPLETE"
echo "============================================="
echo "Results in $LOG_DIR:"
ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "(no logs)"
echo ""
echo "Decision tree:"
echo "  If D1 also low (~0.12): DCP decode path bug (independent of comm backend)"
echo "  If D1 good but D3 low: A2A eager path bug -> check dcp_a2a_lse_reduce"
echo "  If D3 good but D4 bad: CUDA graph capture of A2A is broken"
echo ""
echo "grep for accuracy in logs:"
echo "  grep -i 'accuracy\|invalid' $LOG_DIR/*.log"
SERVERSCRIPT
