#!/bin/bash
# Benchmark DCP + TPA (Tensor Parallel Attention) on GQA models.
#
# Scenarios (from improve-tpa-phase2.md benchmarking plan):
#   Scenario 1: CodeQwen 7B  — 5 configs, 10 CC levels, in=4000/out=1500
#   Scenario 4: CodeQwen 7B  — saturation test, short/medium input, high CC
#   Scenario 2: Qwen2-72B   — large model, TPA memory advantage
#
# Usage:
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh                    # run all
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh scenario1          # only scenario 1
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh scenario4          # only scenario 4
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh scenario2          # only scenario 2 (72B)
#
# Prerequisites:
#   - 8x H100 GPUs
#   - Models downloaded (CodeQwen1.5-7B-Chat, optionally Qwen2-72B-Instruct)

set -euo pipefail

HOST=127.0.0.1
PORT=8188

BRANCH=$(git rev-parse --abbrev-ref HEAD)
HASH=$(git rev-parse --short=7 HEAD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT="${SCRIPT_DIR}/results/${BRANCH}_${HASH}"

COMMON_ENV="SGLANG_DCP_SYMM_ONLY=true NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1"

SCENARIO_FILTER="${1:-all}"

# ---- Helper functions ----

wait_for_server() {
    local max_wait=600
    local elapsed=0
    echo "Waiting for server on ${HOST}:${PORT} ..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q 200; then
            echo "Server ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: Server not ready within ${max_wait}s"
    return 1
}

kill_server() {
    echo "Killing server on port ${PORT} ..."
    pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    sleep 10
}

run_perf() {
    local output_dir="$1"
    local model="$2"
    local input_len="$3"
    local output_len="$4"
    shift 4
    local concurrencies=("$@")

    echo "Running perf: in=${input_len} out=${output_len} -> ${output_dir}"
    for C in "${concurrencies[@]}"; do
        NUM_PROMPTS=$((C * 5))
        FILE_NAME="${output_dir}/cc${C}.txt"

        # Skip if already completed
        if [ -f "$FILE_NAME" ] && grep -q "Output token throughput" "$FILE_NAME" 2>/dev/null; then
            echo "  Skipping cc${C} — already completed"
            continue
        fi

        echo "--- Concurrency=$C, Prompts=$NUM_PROMPTS -> $FILE_NAME ---"

        python3 -m sglang.bench_serving --backend sglang \
            --host "$HOST" --port "$PORT" \
            --model "$model" \
            --dataset-name random \
            --random-input-len "$input_len" \
            --random-output-len "$output_len" \
            --random-range-ratio 0.1 \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$C" \
            --disable-ignore-eos 2>&1 | tee "$FILE_NAME"
    done
}

start_server() {
    local model="$1"
    local cfg_name="$2"
    local backend="$3"
    local mem_frac="$4"
    local dcp="$5"
    local dcp_comm="$6"
    local attn_tp="$7"
    local helix_rs="$8"
    local context_length="$9"
    local max_cc="${10}"

    local extra_args=""
    if [ "$dcp" -gt 0 ]; then
        extra_args="--dcp-size ${dcp} --dcp-comm-backend ${dcp_comm}"
    fi
    if [ "$attn_tp" -gt 0 ]; then
        extra_args="${extra_args} --attention-tensor-parallel-size ${attn_tp}"
    fi
    if [ "$helix_rs" -eq 1 ]; then
        extra_args="${extra_args} --enable-helix-reduce-scatter"
    fi

    echo "======================================================="
    echo "Starting: ${cfg_name} (${model})"
    echo "  backend=${backend}  mem=${mem_frac}  dcp=${dcp}  comm=${dcp_comm}"
    echo "  attn_tp=${attn_tp}  helix_rs=${helix_rs}  ctx=${context_length}  max_cc=${max_cc}"
    echo "======================================================="

    eval "${COMMON_ENV} python3 -m sglang.launch_server \
        --model-path ${model} --host 0.0.0.0 --port ${PORT} \
        --trust-remote-code --enable-cache-report --log-level info --tp-size 8 \
        --max-running-requests ${max_cc} --chunked-prefill-size 32768 \
        --context-length ${context_length} --disable-radix-cache --enable-symm-mem \
        --mem-fraction-static ${mem_frac} \
        --attention-backend ${backend} \
        ${extra_args}" &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"
}

run_scenario_configs() {
    local scenario_name="$1"
    local model="$2"
    local context_length="$3"
    local max_cc="$4"
    local input_len="$5"
    local output_len="$6"
    local workload_tag="$7"
    shift 7
    local configs=("$@")

    # Concurrencies depend on max_cc
    local concurrencies=()
    for cc in 1 2 4 8 16 32 64 128 256 512 1024; do
        if [ "$cc" -le "$max_cc" ]; then
            concurrencies+=("$cc")
        fi
    done

    for cfg in "${configs[@]}"; do
        IFS='|' read -r CFG_NAME BACKEND MEM_FRAC DCP DCP_COMM ATTN_TP HELIX_RS <<< "$cfg"

        OUTPUT_DIR="${BASE_OUTPUT}/${scenario_name}/${CFG_NAME}/${workload_tag}"

        # Skip if last CC result exists
        local last_cc="${concurrencies[-1]}"
        if [ -f "$OUTPUT_DIR/cc${last_cc}.txt" ] && grep -q "Output token throughput" "$OUTPUT_DIR/cc${last_cc}.txt" 2>/dev/null; then
            echo "Skipping ${scenario_name}/${CFG_NAME}/${workload_tag} — results already exist"
            continue
        fi

        mkdir -p "$OUTPUT_DIR"
        kill_server
        start_server "$model" "$CFG_NAME" "$BACKEND" "$MEM_FRAC" "$DCP" "$DCP_COMM" "$ATTN_TP" "$HELIX_RS" "$context_length" "$max_cc"

        if ! wait_for_server; then
            echo "Skipping ${CFG_NAME} due to server start failure"
            kill_server
            continue
        fi

        run_perf "$OUTPUT_DIR" "$model" "$input_len" "$output_len" "${concurrencies[@]}"
        kill_server
    done
}


# ============================================================
# Scenario 1: CodeQwen 7B — standard benchmark (5 configs)
# Purpose: Compare all configs with helix RS on same workload
# ============================================================
run_scenario1() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 1: CodeQwen 7B — Standard Benchmark"
    echo "======================================================="

    local model="Qwen/CodeQwen1.5-7B-Chat"
    local configs=(
        "tp8_fa3|fa3|0.85|0||0|0"
        "tp8_dcp2_a2a_fa3|fa3|0.85|2|a2a|0|0"
        "tp8_dcp2_agrs_fa3|fa3|0.85|2|ag_rs|0|0"
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.85|2|a2a|4|0"
        "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.85|2|a2a|4|1"
    )

    run_scenario_configs "scenario1_7b" "$model" 65536 512 4000 1500 "in4000_out1500" "${configs[@]}"
}


# ============================================================
# Scenario 4: CodeQwen 7B — High throughput saturation
# Purpose: Short input + high CC to show helix RS MLP savings
# Only runs key configs (baseline TPA vs helix RS)
# ============================================================
run_scenario4() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 4: CodeQwen 7B — Saturation Test"
    echo "======================================================="

    local model="Qwen/CodeQwen1.5-7B-Chat"
    # Only the 2 key configs + baseline for comparison
    local configs=(
        "tp8_fa3|fa3|0.85|0||0|0"
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.85|2|a2a|4|0"
        "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.85|2|a2a|4|1"
    )

    # Workload A: Short input, short output (pure decode throughput)
    run_scenario_configs "scenario4_7b" "$model" 65536 512 512 256 "in512_out256" "${configs[@]}"

    # Workload B: Medium input, medium output
    run_scenario_configs "scenario4_7b" "$model" 65536 512 2048 512 "in2048_out512" "${configs[@]}"
}


# ============================================================
# Scenario 2: Qwen2-72B — Large model, KV memory bottleneck
# Purpose: TPA enables DCP for models where plain DCP can't reach
# ============================================================
run_scenario2() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 2: Qwen2-72B — Large Model"
    echo "======================================================="

    local model="Qwen/Qwen2-72B-Instruct"
    # 72B: 8 KV heads, tp8 gives 1 KV head/rank
    # dcp2: requires attn_tp=4 (TPA) since 1 KV head can't be split further
    local configs=(
        "tp8_fa3|fa3|0.88|0||0|0"
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.88|2|a2a|4|0"
        "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.88|2|a2a|4|1"
    )

    # Conservative: shorter output, lower max CC (72B needs more memory)
    run_scenario_configs "scenario2_72b" "$model" 32768 128 4000 500 "in4000_out500" "${configs[@]}"
}


# ============================================================
# Scenario 5: Qwen3-235B-A22B MoE — Large MoE, 4 KV heads
# Purpose: MoE + TPA. Helix RS reduces expert compute per rank.
#   235B total / 22B active, 128 experts (8 active/tok), 94 layers
#   4 KV heads (same as CodeQwen) → TPA unlocks DCP
#   ~438 GB bf16 → TP8 uses ~41 GB/GPU, leaves ~55 GB for KV
# Prerequisites: huggingface-cli download Qwen/Qwen3-235B-A22B-Instruct-2507
# ============================================================
run_scenario5() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 5: Qwen3-235B-A22B MoE — Large MoE Model"
    echo "======================================================="

    local model="Qwen/Qwen3-235B-A22B-Instruct-2507"
    # 4 KV heads, 64 Q heads, 94 layers, MoE 128 experts (8 active)
    # tp8: each rank has 0.5 KV heads (needs GQA replication)
    # tpa4+dcp2: attn_tp=4, each attn rank has 1 KV head
    local configs=(
        "tp8_fa3|fa3|0.90|0||0|0"
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.90|2|a2a|4|0"
        "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.90|2|a2a|4|1"
    )

    # MoE model: memory-constrained, use moderate CC and context
    run_scenario_configs "scenario5_qwen3_235b" "$model" 32768 64 2048 500 "in2048_out500" "${configs[@]}"
}


# ============================================================
# Scenario 3: Long Context Decode
# Purpose: DCP/TPA shine at very long context (attention dominates)
#   CodeQwen 7B supports 65K natively; use context_length=131072
#   with RoPE scaling for longer. Low CC since each request is huge.
#   KV cache per token: 4 heads * 128 dim * 2 (K+V) * 28 layers * 2 bytes = 57 KB
#   At 128K context: ~7.3 GB/request → max ~10 requests on 8xH100
# ============================================================
run_scenario3() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 3: CodeQwen 7B — Long Context Decode"
    echo "======================================================="

    local model="Qwen/CodeQwen1.5-7B-Chat"
    local configs=(
        "tp8_fa3|fa3|0.90|0||0|0"
        "tp8_dcp2_a2a_fa3|fa3|0.90|2|a2a|0|0"
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.90|2|a2a|4|0"
        "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.90|2|a2a|4|1"
    )

    # Long context: 128K input, short output, low CC
    run_scenario_configs "scenario3_longctx" "$model" 131072 16 128000 64 "in128k_out64" "${configs[@]}"

    # Medium-long context: 32K input, short output, moderate CC
    run_scenario_configs "scenario3_longctx" "$model" 65536 64 32000 64 "in32k_out64" "${configs[@]}"
}


# ---- Main ----
echo "Benchmark run: branch=${BRANCH} commit=${HASH}"
echo "Output dir: ${BASE_OUTPUT}/"
echo "Scenario filter: ${SCENARIO_FILTER}"
echo ""

case "$SCENARIO_FILTER" in
    scenario1) run_scenario1 ;;
    scenario2) run_scenario2 ;;
    scenario3) run_scenario3 ;;
    scenario4) run_scenario4 ;;
    scenario5) run_scenario5 ;;
    all)
        run_scenario1
        run_scenario3
        run_scenario4
        run_scenario2
        run_scenario5
        ;;
    *)
        echo "Unknown scenario: $SCENARIO_FILTER"
        echo "Usage: $0 [scenario1|scenario2|scenario3|scenario4|scenario5|all]"
        exit 1
        ;;
esac

echo ""
echo "======================================================="
echo "All benchmarks complete! Results in: ${BASE_OUTPUT}/"
echo "======================================================="
