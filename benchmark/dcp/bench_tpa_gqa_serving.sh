#!/bin/bash
# Benchmark DCP + TPA (Tensor Parallel Attention) on GQA models.
#
# Scenarios (from improve-tpa-phase2.md benchmarking plan):
#   Scenario 1: CodeQwen 7B  — 5 configs, 10 CC levels, in=4000/out=1500
#   Scenario 4: CodeQwen 7B  — saturation test, short/medium input, high CC
#   Scenario 2: Qwen2-72B   — large model, TPA memory advantage
#
# Scenarios:
#   S1:  CodeQwen 7B standard (tp8, dcp2, tpa4_dcp2)
#   S2:  Qwen2-72B (tp8, tpa4_dcp2)
#   S3:  CodeQwen 7B long context 60K/32K
#   S4:  CodeQwen 7B saturation (short input, high CC)
#   S5:  Qwen3-235B MoE (tp8, dcp2, tpa4_dcp2)
#   S6:  CodeQwen 7B 1M context
#   S7:  CodeQwen 7B TPA KV advantage at 128K (tpa2_dcp4 vs dcp2 vs tp8)
#   S8:  CodeQwen 7B TPA at 512K (tpa2_dcp4 only config with headroom)
#   S9:  Qwen2-72B TPA enables DCP (plain DCP impossible)
#   S10: Qwen3-235B MoE TPA high CC at 32K
#
# Usage:
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh <scenario> [mode]
#
#   Modes: accuracy = accuracy only, perf = perf only, all = both (default)
#
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh scenario1 accuracy  # quick validation
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh scenario1 perf      # perf only
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh tpa_advantage       # run S7-S10 (TPA sweet spots)
#   bash benchmark/dcp/bench_tpa_gqa_serving.sh scenario7 accuracy  # validate tpa2_dcp4 configs
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
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"

SCENARIO_FILTER="${1:-all}"
# Run mode: "accuracy" = accuracy only, "perf" = perf only, "all" = both (default)
RUN_MODE="${2:-all}"

# ---- Helper functions ----

wait_for_server() {
    local max_wait=1200
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

run_accuracy() {
    local output_dir="$1"
    local acc_file="${output_dir}/accuracy_gsm8k.txt"
    echo "Running accuracy test -> ${acc_file}"
    python3 benchmark/gsm8k/bench_sglang.py \
        --parallel 64 \
        --host "$HOST" --port "$PORT" 2>&1 | tee "$acc_file"
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
            --disable-ignore-eos 2>&1 | tee "$FILE_NAME" || {
            echo "  WARNING: cc${C} failed (likely OOM at high concurrency). Skipping remaining CCs."
            break
        }
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
    echo "  attn_tp=${attn_tp}  helix_rs=${helix_rs}  ctx=${context_length}"
    echo "======================================================="

    eval "${COMMON_ENV} python3 -m sglang.launch_server \
        --model-path ${model} --host 0.0.0.0 --port ${PORT} \
        --trust-remote-code --enable-cache-report --log-level info --tp-size 8 \
        --chunked-prefill-size 32768 \
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

        # Skip logic depends on run mode
        if [ "$RUN_MODE" = "accuracy" ]; then
            if [ -f "$OUTPUT_DIR/accuracy_gsm8k.txt" ] && grep -q "Accuracy:" "$OUTPUT_DIR/accuracy_gsm8k.txt" 2>/dev/null; then
                echo "Skipping ${scenario_name}/${CFG_NAME}/${workload_tag} — accuracy already done"
                continue
            fi
        else
            local last_cc="${concurrencies[-1]}"
            if [ -f "$OUTPUT_DIR/cc${last_cc}.txt" ] && grep -q "Output token throughput" "$OUTPUT_DIR/cc${last_cc}.txt" 2>/dev/null; then
                echo "Skipping ${scenario_name}/${CFG_NAME}/${workload_tag} — results already exist"
                continue
            fi
        fi

        mkdir -p "$OUTPUT_DIR"
        kill_server
        start_server "$model" "$CFG_NAME" "$BACKEND" "$MEM_FRAC" "$DCP" "$DCP_COMM" "$ATTN_TP" "$HELIX_RS" "$context_length"

        if ! wait_for_server; then
            echo "Skipping ${CFG_NAME} due to server start failure"
            kill_server
            continue
        fi

        if [ "$RUN_MODE" = "accuracy" ] || [ "$RUN_MODE" = "all" ]; then
            run_accuracy "$OUTPUT_DIR"
        fi
        if [ "$RUN_MODE" = "perf" ] || [ "$RUN_MODE" = "all" ]; then
            run_perf "$OUTPUT_DIR" "$model" "$input_len" "$output_len" "${concurrencies[@]}"
        fi
        kill_server
    done
}


# ============================================================
# Scenario 1: CodeQwen 7B — standard benchmark
# Purpose: Compare tp8 vs DCP vs TPA+DCP on same workload
# ============================================================
run_scenario1() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 1: CodeQwen 7B — Standard Benchmark"
    echo "======================================================="

    local model="Qwen/CodeQwen1.5-7B-Chat"
    # DCP/TPA configs first (most likely to break with new changes)
    local configs=(
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.85|2|a2a|4|0"
        "tp8_dcp2_a2a_fa3|fa3|0.85|2|a2a|0|0"
        "tp8_dcp2_agrs_fa3|fa3|0.85|2|ag_rs|0|0"
        "tp8_fa3|fa3|0.85|0||0|0"
        # "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.85|2|a2a|4|1"
    )

    run_scenario_configs "scenario1_7b" "$model" 65536 512 4000 1500 "in4000_out1500" "${configs[@]}"
}


# ============================================================
# Scenario 4: CodeQwen 7B — High throughput saturation
# Purpose: Short input + high CC to measure decode throughput
# ============================================================
run_scenario4() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 4: CodeQwen 7B — Saturation Test"
    echo "======================================================="

    local model="Qwen/CodeQwen1.5-7B-Chat"
    local configs=(
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.85|2|a2a|4|0"
        "tp8_dcp2_a2a_fa3|fa3|0.85|2|a2a|0|0"
        "tp8_fa3|fa3|0.85|0||0|0"
        # "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.85|2|a2a|4|1"
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
    # Plain dcp2 fails (can't split 1 KV head further), needs TPA
    local configs=(
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.88|2|a2a|4|0"
        "tp8_fa3|fa3|0.88|0||0|0"
        # "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.88|2|a2a|4|1"
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
    # Note: DCP+TPA needs extra memory for symmetric mem buffers — use lower mem_frac
    local configs=(
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.85|2|a2a|4|0"
        "tp8_dcp2_a2a_fa3|fa3|0.85|2|a2a|0|0"
        "tp8_fa3|fa3|0.90|0||0|0"
        # "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.85|2|a2a|4|1"
    )

    # MoE model: medium input + output, high CC to see saturation
    run_scenario_configs "scenario5_qwen3_235b" "$model" 32768 512 4000 1500 "in4000_out1500" "${configs[@]}"
}


# ============================================================
# Scenario 3: Long Context Decode
# Purpose: DCP/TPA shine at very long context (attention dominates)
#   CodeQwen 7B supports 65K natively. Low CC since each request is huge.
#   KV cache per token: 4 heads * 128 dim * 2 (K+V) * 28 layers * 2 bytes = 57 KB
#   At 60K context: ~3.4 GB/request → max ~20 concurrent on 8xH100
# ============================================================
run_scenario3() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 3: CodeQwen 7B — Long Context Decode"
    echo "======================================================="

    local model="Qwen/CodeQwen1.5-7B-Chat"
    local configs=(
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.90|2|a2a|4|0"
        "tp8_dcp2_a2a_fa3|fa3|0.90|2|a2a|0|0"
        "tp8_fa3|fa3|0.90|0||0|0"
        # "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.90|2|a2a|4|1"
    )

    # Long context: 60K input, short output, low CC (near CodeQwen max 65K)
    run_scenario_configs "scenario3_longctx" "$model" 65536 16 60000 64 "in60k_out64" "${configs[@]}"

    # Medium-long context: 32K input, short output, moderate CC
    run_scenario_configs "scenario3_longctx" "$model" 65536 64 32000 64 "in32k_out64" "${configs[@]}"
}


# ============================================================
# Scenario 6: CodeQwen 7B — 1M Context (DCP/TPA sweet spot)
# Purpose: Very long context where DCP splits KV cache across ranks,
#   TPA enables higher DCP degrees. This is where Phase-1 showed 2x.
#   Uses SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 (set in COMMON_ENV)
#   KV per token: 57 KB → at 1M: ~57 GB → needs DCP to fit
# ============================================================
run_scenario6() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 6: CodeQwen 7B — 1M Context"
    echo "======================================================="

    local model="Qwen/CodeQwen1.5-7B-Chat"
    local configs=(
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.92|2|a2a|4|0"
        "tp8_dcp2_a2a_fa3|fa3|0.92|2|a2a|0|0"
        "tp8_fa3|fa3|0.92|0||0|0"
        # "tp8_tpa4_dcp2_a2a_helix_fa3|fa3|0.92|2|a2a|4|1"
    )

    # 1M context: very long input, short output, low CC
    run_scenario_configs "scenario6_1m" "$model" 1048576 16 1000000 64 "in1m_out64" "${configs[@]}"

    # 131K context: long input, moderate CC
    run_scenario_configs "scenario6_1m" "$model" 131072 32 131000 64 "in131k_out64" "${configs[@]}"
}


# ============================================================
# Scenario 7: CodeQwen 7B — TPA KV Cache Advantage at 128K
# Purpose: Show tpa2_dcp4 handles more concurrent requests than
#   tp8/dcp2 at long context where KV cache is the bottleneck.
#   tp8 fits ~12, dcp2 fits ~24, tpa2_dcp4 fits ~49 at 128K.
# ============================================================
run_scenario7() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 7: CodeQwen 7B — TPA KV Cache Advantage (128K)"
    echo "======================================================="

    local model="Qwen/CodeQwen1.5-7B-Chat"
    # DCP=4 needs more symmetric memory buffers → lower mem_frac
    local configs=(
        "tp8_tpa2_dcp4_a2a_fa3|fa3|0.85|4|a2a|2|0"
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.88|2|a2a|4|0"
        "tp8_dcp2_a2a_fa3|fa3|0.90|2|a2a|0|0"
        "tp8_fa3|fa3|0.92|0||0|0"
    )

    run_scenario_configs "scenario7_tpa_kv" "$model" 131072 32 128000 64 "in128k_out64" "${configs[@]}"
}


# ============================================================
# Scenario 8: CodeQwen 7B — TPA at 512K Context
# Purpose: At 512K only DCP=4 (via TPA) has enough headroom.
#   tp8 fits 3, dcp2 fits 6, tpa2_dcp4 fits 12.
#   Phase-1 showed tpa2_dcp4 has 2x faster TPOT than dcp2 at 512K.
# ============================================================
run_scenario8() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 8: CodeQwen 7B — TPA at 512K Context"
    echo "======================================================="

    local model="Qwen/CodeQwen1.5-7B-Chat"
    local configs=(
        "tp8_tpa2_dcp4_a2a_fa3|fa3|0.85|4|a2a|2|0"
        "tp8_dcp2_a2a_fa3|fa3|0.88|2|a2a|0|0"
        "tp8_fa3|fa3|0.92|0||0|0"
    )

    run_scenario_configs "scenario8_512k" "$model" 524288 8 500000 64 "in500k_out64" "${configs[@]}"
}


# ============================================================
# Scenario 9: Qwen2-72B — TPA Enables DCP (Impossible Without)
# Purpose: 72B with 8 KV heads can't do plain DCP at TP=8.
#   TPA is the ONLY way to get DCP benefits.
#   tp8 max ~7 concurrent at 32K. tpa4_dcp2 max ~14.
# ============================================================
run_scenario9() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 9: Qwen2-72B — TPA Enables DCP"
    echo "======================================================="

    local model="Qwen/Qwen2-72B-Instruct"
    local configs=(
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.88|2|a2a|4|0"
        "tp8_fa3|fa3|0.88|0||0|0"
    )

    # Long context KV pressure test
    run_scenario_configs "scenario9_72b_kv" "$model" 32768 16 30000 64 "in30k_out64" "${configs[@]}"

    # Standard throughput for comparison
    run_scenario_configs "scenario9_72b_kv" "$model" 32768 128 4000 500 "in4k_out500" "${configs[@]}"
}


# ============================================================
# Scenario 10: Qwen3-235B MoE — TPA High CC at 32K
# Purpose: 235B has only ~33 GB for KV. At 32K, tp8 fits 5.
#   tpa2_dcp4 fits ~22.
# ============================================================
run_scenario10() {
    echo ""
    echo "======================================================="
    echo "SCENARIO 10: Qwen3-235B MoE — TPA High CC at 32K"
    echo "======================================================="

    local model="Qwen/Qwen3-235B-A22B-Instruct-2507"
    local configs=(
        "tp8_tpa2_dcp4_a2a_fa3|fa3|0.85|4|a2a|2|0"
        "tp8_tpa4_dcp2_a2a_fa3|fa3|0.85|2|a2a|4|0"
        "tp8_dcp2_a2a_fa3|fa3|0.85|2|a2a|0|0"
        "tp8_fa3|fa3|0.90|0||0|0"
    )

    # Long context KV pressure test
    run_scenario_configs "scenario10_235b_kv" "$model" 32768 16 30000 500 "in30k_out500" "${configs[@]}"
}


# ---- Main ----
echo "Benchmark run: branch=${BRANCH} commit=${HASH}"
echo "Output dir: ${BASE_OUTPUT}/"
echo "Scenario filter: ${SCENARIO_FILTER}"
echo "Run mode: ${RUN_MODE}"
echo ""

case "$SCENARIO_FILTER" in
    scenario1) run_scenario1 ;;
    scenario2) run_scenario2 ;;
    scenario3) run_scenario3 ;;
    scenario4) run_scenario4 ;;
    scenario5) run_scenario5 ;;
    scenario6) run_scenario6 ;;
    scenario7) run_scenario7 ;;
    scenario8) run_scenario8 ;;
    scenario9) run_scenario9 ;;
    scenario10) run_scenario10 ;;
    all)
        run_scenario1
        run_scenario3
        run_scenario4
        run_scenario2
        run_scenario5
        run_scenario6
        run_scenario7
        run_scenario8
        run_scenario9
        run_scenario10
        ;;
    tpa_advantage)
        run_scenario7
        run_scenario8
        run_scenario9
        run_scenario10
        ;;
    *)
        echo "Unknown scenario: $SCENARIO_FILTER"
        echo "Usage: $0 [scenario1-10|tpa_advantage|all] [accuracy|perf|all]"
        exit 1
        ;;
esac

echo ""
echo "======================================================="
echo "All benchmarks complete! Results in: ${BASE_OUTPUT}/"
echo "======================================================="
