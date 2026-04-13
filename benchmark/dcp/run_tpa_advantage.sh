#!/bin/bash
# Run TPA-advantage scenarios from OUTSIDE the container.
# Restarts container between each config for clean GPU state.
#
# Usage (on colossus host):
#   bash run_tpa_advantage.sh [accuracy|perf|all]

set -uo pipefail

CONTAINER="sglang-bench-gqa"
LOG="/tmp/tpa_advantage.log"
MODE="${1:-all}"

echo "=== TPA Advantage Benchmark ===" | tee "$LOG"
echo "Mode: $MODE" >> "$LOG"
echo "Date: $(date)" >> "$LOG"

run_in_container() {
    docker exec "$CONTAINER" bash -c "$1"
}

restart_container() {
    echo "Restarting container..." >> "$LOG"
    docker restart "$CONTAINER" > /dev/null 2>&1
    sleep 5
}

start_server() {
    local model="$1"
    local mem_frac="$2"
    local ctx="$3"
    local extra="$4"
    local max_wait=600

    restart_container

    echo "Starting server: model=$(basename $model) mem=$mem_frac ctx=$ctx $extra" >> "$LOG"

    run_in_container "cd /sgl-workspace/sglang && \
        SGLANG_DCP_SYMM_ONLY=true NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 \
        TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
        SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
        nohup python3 -m sglang.launch_server \
            --model-path $model --host 0.0.0.0 --port 8188 \
            --trust-remote-code --tp-size 8 --chunked-prefill-size 32768 \
            --context-length $ctx --disable-radix-cache --enable-symm-mem \
            --mem-fraction-static $mem_frac --attention-backend fa3 \
            $extra > /tmp/server.log 2>&1 &"

    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if run_in_container "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8188/health 2>/dev/null" | grep -q 200; then
            echo "  Server ready (${elapsed}s)" >> "$LOG"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "  FAILED: Server not ready in ${max_wait}s" >> "$LOG"
    return 1
}

run_accuracy() {
    local tag="$1"
    echo "  Running accuracy: $tag" >> "$LOG"
    run_in_container "cd /sgl-workspace/sglang && python3 benchmark/gsm8k/bench_sglang.py \
        --parallel 64 --host 127.0.0.1 --port 8188 2>&1" >> "$LOG" | grep "Accuracy:"
}

run_perf() {
    local model="$1"
    local in_len="$2"
    local out_len="$3"
    shift 3
    local ccs=("$@")

    for cc in "${ccs[@]}"; do
        local np=$((cc * 5))
        [ "$np" -lt 10 ] && np=10
        echo "  Perf cc=$cc prompts=$np in=$in_len out=$out_len" >> "$LOG"
        run_in_container "cd /sgl-workspace/sglang && python3 -m sglang.bench_serving --backend sglang \
            --host 127.0.0.1 --port 8188 --model $model \
            --dataset-name random --random-input-len $in_len --random-output-len $out_len \
            --random-range-ratio 0.1 --num-prompts $np --max-concurrency $cc \
            --disable-ignore-eos 2>&1" >> "$LOG" | grep "Output token throughput" || {
            echo "  WARNING: cc=$cc failed, stopping" >> "$LOG"
            break
        }
    done
}

# ============================================================
# S7: CodeQwen 7B at 128K — TPA KV Cache Advantage
# ============================================================
run_s7() {
    local model="Qwen/CodeQwen1.5-7B-Chat"
    local ctx=131072
    local ccs=(1 2 4 8 16 32)

    echo "" >> "$LOG"
    echo "=== S7: CodeQwen 7B at 128K ===" >> "$LOG"

    # tpa2_dcp4
    if start_server "$model" 0.82 $ctx "--dcp-size 4 --dcp-comm-backend a2a --attention-tensor-parallel-size 2"; then
        [ "$MODE" != "perf" ] && run_accuracy "tpa2_dcp4"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 128000 64 "${ccs[@]}"
    fi

    # tpa4_dcp2
    if start_server "$model" 0.85 $ctx "--dcp-size 2 --dcp-comm-backend a2a --attention-tensor-parallel-size 4"; then
        [ "$MODE" != "perf" ] && run_accuracy "tpa4_dcp2"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 128000 64 "${ccs[@]}"
    fi

    # dcp2
    if start_server "$model" 0.87 $ctx "--dcp-size 2 --dcp-comm-backend a2a"; then
        [ "$MODE" != "perf" ] && run_accuracy "dcp2"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 128000 64 "${ccs[@]}"
    fi

    # tp8
    if start_server "$model" 0.88 $ctx ""; then
        [ "$MODE" != "perf" ] && run_accuracy "tp8"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 128000 64 "${ccs[@]}"
    fi
}

# ============================================================
# S8: CodeQwen 7B at 512K — Only TPA has headroom
# ============================================================
run_s8() {
    local model="Qwen/CodeQwen1.5-7B-Chat"
    local ctx=524288
    local ccs=(1 2 4 8)

    echo "" >> "$LOG"
    echo "=== S8: CodeQwen 7B at 512K ===" >> "$LOG"

    # tpa2_dcp4 — only config with headroom
    if start_server "$model" 0.82 $ctx "--dcp-size 4 --dcp-comm-backend a2a --attention-tensor-parallel-size 2"; then
        [ "$MODE" != "perf" ] && run_accuracy "tpa2_dcp4"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 500000 64 "${ccs[@]}"
    fi

    # dcp2 — may OOM
    if start_server "$model" 0.85 $ctx "--dcp-size 2 --dcp-comm-backend a2a"; then
        [ "$MODE" != "perf" ] && run_accuracy "dcp2"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 500000 64 "${ccs[@]}"
    fi

    # tp8 — will likely OOM
    if start_server "$model" 0.88 $ctx ""; then
        [ "$MODE" != "perf" ] && run_accuracy "tp8"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 500000 64 "${ccs[@]}"
    fi
}

# ============================================================
# S9: Qwen2-72B — TPA enables DCP
# ============================================================
run_s9() {
    local model="Qwen/Qwen2-72B-Instruct"
    local ctx=32768

    echo "" >> "$LOG"
    echo "=== S9: Qwen2-72B at 32K ===" >> "$LOG"

    # tpa4_dcp2 — long context KV pressure
    if start_server "$model" 0.88 $ctx "--dcp-size 2 --dcp-comm-backend a2a --attention-tensor-parallel-size 4"; then
        [ "$MODE" != "perf" ] && run_accuracy "tpa4_dcp2"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 30000 64 1 2 4 8 16
        [ "$MODE" != "accuracy" ] && run_perf "$model" 4000 500 1 4 8 16 32 64 128
    fi

    # tp8
    if start_server "$model" 0.88 $ctx ""; then
        [ "$MODE" != "perf" ] && run_accuracy "tp8"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 30000 64 1 2 4 8 16
        [ "$MODE" != "accuracy" ] && run_perf "$model" 4000 500 1 4 8 16 32 64 128
    fi
}

# ============================================================
# S10: Qwen3-235B MoE at 32K — TPA high CC
# ============================================================
run_s10() {
    local model="Qwen/Qwen3-235B-A22B-Instruct-2507"
    local ctx=32768
    local ccs=(1 2 4 8 16)

    echo "" >> "$LOG"
    echo "=== S10: Qwen3-235B MoE at 32K ===" >> "$LOG"

    # tpa2_dcp4
    if start_server "$model" 0.82 $ctx "--dcp-size 4 --dcp-comm-backend a2a --attention-tensor-parallel-size 2"; then
        [ "$MODE" != "perf" ] && run_accuracy "tpa2_dcp4"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 30000 500 "${ccs[@]}"
    fi

    # tpa4_dcp2
    if start_server "$model" 0.85 $ctx "--dcp-size 2 --dcp-comm-backend a2a --attention-tensor-parallel-size 4"; then
        [ "$MODE" != "perf" ] && run_accuracy "tpa4_dcp2"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 30000 500 "${ccs[@]}"
    fi

    # tp8
    if start_server "$model" 0.90 $ctx ""; then
        [ "$MODE" != "perf" ] && run_accuracy "tp8"
        [ "$MODE" != "accuracy" ] && run_perf "$model" 30000 500 "${ccs[@]}"
    fi
}

# ---- Main ----
run_s7
run_s8
run_s9
run_s10

echo "" >> "$LOG"
echo "=== ALL DONE ===" >> "$LOG"
echo "Results in: $LOG"
