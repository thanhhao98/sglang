#!/bin/bash
###############################################################################
# H100 Rerun Failed Configs
#
# Failures:
#   1. C3_DCP8_a2a_fi_128K: NCCL OOM at mem_frac=0.85 → rerun with 0.50
#   2. C4_DCP8_a2a_fa3_128K: CUDA OOM at mem_frac=0.85 → rerun with 0.50
#   3. C2_DCP8_agrs_128K_c1: warmup timeout → rerun c=1 only
#   4. C2_DCP8_agrs_256K_c2,c4: TransferEncodingError → rerun c=2,4 only
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
        pkill -9 -f "sglang" 2>/dev/null || true
        sleep 5
    fi
}

wait_server() {
    echo "Waiting for server..."
    for i in $(seq 1 120); do
        curl -s http://${BENCH_HOST}:${PORT}/health >/dev/null 2>&1 && { echo "Server ready after $((i*5))s"; return 0; }
        sleep 5
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
}

###############################################################################
# Fix 1: C3_DCP8_a2a_fi_128K — lower mem_frac to 0.50
###############################################################################
fix_c3_128k() {
    echo ""
    echo "============================================"
    echo "=== FIX: C3_DCP8_a2a_fi_128K (mem_frac=0.50) ==="
    echo "============================================"

    kill_server
    export SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

    python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size 8 --trust-remote-code \
        --context-length 131072 --mem-fraction-static 0.50 \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend a2a --chunked-prefill-size 32768 \
        --host $HOST --port $PORT \
        2>&1 | tee /output/C3_DCP8_a2a_fi_128K_server_v2.log &

    wait_server || { kill_server; return 1; }

    for CONC in ${CONCURRENCIES[@]}; do
        run_bench "C3_DCP8_a2a_fi_128K_v2" $CONC 130048
    done

    kill_server
    echo "=== C3_DCP8_a2a_fi_128K DONE ==="
}

###############################################################################
# Fix 2: C4_DCP8_a2a_fa3_128K — lower mem_frac to 0.50
###############################################################################
fix_c4_128k() {
    echo ""
    echo "============================================"
    echo "=== FIX: C4_DCP8_a2a_fa3_128K (mem_frac=0.50) ==="
    echo "============================================"

    kill_server
    export SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

    python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size 8 --trust-remote-code \
        --context-length 131072 --mem-fraction-static 0.50 \
        --attention-backend fa3 --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend a2a --chunked-prefill-size 32768 \
        --host $HOST --port $PORT \
        2>&1 | tee /output/C4_DCP8_a2a_fa3_128K_server_v2.log &

    wait_server || { kill_server; return 1; }

    for CONC in ${CONCURRENCIES[@]}; do
        run_bench "C4_DCP8_a2a_fa3_128K_v2" $CONC 130048
    done

    kill_server
    echo "=== C4_DCP8_a2a_fa3_128K DONE ==="
}

###############################################################################
# Fix 3: C2_DCP8_agrs_128K_c1 — rerun just c=1
###############################################################################
fix_c2_128k_c1() {
    echo ""
    echo "============================================"
    echo "=== FIX: C2_DCP8_agrs_128K c=1 ==="
    echo "============================================"

    kill_server
    export SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

    python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size 8 --trust-remote-code \
        --context-length 131072 --mem-fraction-static 0.85 \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend ag_rs --chunked-prefill-size 32768 \
        --host $HOST --port $PORT \
        2>&1 | tee /output/C2_DCP8_agrs_128K_server_v2.log &

    wait_server || { kill_server; return 1; }

    run_bench "C2_DCP8_agrs_128K_v2" 1 130048

    kill_server
    echo "=== C2_DCP8_agrs_128K c=1 DONE ==="
}

###############################################################################
# Fix 4: C2_DCP8_agrs_256K c=2,4 — rerun
###############################################################################
fix_c2_256k_c2c4() {
    echo ""
    echo "============================================"
    echo "=== FIX: C2_DCP8_agrs_256K c=2,4 ==="
    echo "============================================"

    kill_server
    export SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

    python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size 8 --trust-remote-code \
        --context-length 262144 --mem-fraction-static 0.80 \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend ag_rs --chunked-prefill-size 32768 \
        --host $HOST --port $PORT \
        2>&1 | tee /output/C2_DCP8_agrs_256K_server_v2.log &

    wait_server || { kill_server; return 1; }

    run_bench "C2_DCP8_agrs_256K_v2" 2 261120
    run_bench "C2_DCP8_agrs_256K_v2" 4 261120

    kill_server
    echo "=== C2_DCP8_agrs_256K c=2,4 DONE ==="
}

###############################################################################
# Main
###############################################################################
echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""

case "${1:-all}" in
    c3) fix_c3_128k ;;
    c4) fix_c4_128k ;;
    c2_128k) fix_c2_128k_c1 ;;
    c2_256k) fix_c2_256k_c2c4 ;;
    all)
        fix_c2_128k_c1
        fix_c2_256k_c2c4
        fix_c3_128k
        fix_c4_128k
        echo ""
        echo "ALL RERUNS COMPLETE"
        ;;
    *) echo "Usage: $0 [c3|c4|c2_128k|c2_256k|all]"; exit 1 ;;
esac
