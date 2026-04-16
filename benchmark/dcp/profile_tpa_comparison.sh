#!/bin/bash
# Profile DCP/TPA configs on Qwen3-235B to identify optimization opportunities.
#
# Captures nsys profiles compatible with gwe_perf_advisor (nim-tuning-guide).
# Key flags: --cuda-graph-trace=node, --gpu-metrics-device=all, --gpu-metrics-frequency=20000
#
# Configs:
#   tp8_flashinfer, tp8_fa3,
#   tp8_dcp2_a2a, tp8_dcp2_agrs,
#   tp8_tpa4_dcp2_a2a, tp8_tpa4_dcp2_agrs,
#   tp8_tpa2_dcp4_a2a, tp8_tpa2_dcp4_agrs
#
# Workload: 32K/4K with cc48 and cc64
#
# Usage:
#   bash benchmark/dcp/profile_tpa_comparison.sh [config|all] [phase]
#
#   bash benchmark/dcp/profile_tpa_comparison.sh all nsys        # all configs, nsys only
#   bash benchmark/dcp/profile_tpa_comparison.sh tp8_flashinfer  # single config, both phases
#   bash benchmark/dcp/profile_tpa_comparison.sh all torch       # all configs, torch only
#
# After profiling, analyze with:
#   python3 -m gwe_perf_advisor.cli profile_diag --profile-path <file>.nsys-rep

set -euo pipefail

HOST=127.0.0.1
PORT=8188
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_SHORT="qwen3_235b"

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
HASH=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "0000000")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="${SCRIPT_DIR}/profiles/${BRANCH}_${HASH}_${MODEL_SHORT}_${TIMESTAMP}"

NSYS_TRACES="cuda,nvtx,nccl,osrt"
NSYS_EXTRA="--cuda-graph-trace=node"
TORCH_PROFILE_STEPS=10

FILTER="${1:-all}"
PHASE="${2:-all}"

INPUT_LEN=32000
OUTPUT_LEN=4000
CONCURRENCIES=(48 64)
WARMUP_PROMPTS=3
WARMUP_CC=2
SERVER_WAIT=900

# ---- Config definitions: name | backend | dcp | attn_tp | dcp_comm ----
declare -A CFG_BACKEND CFG_DCP CFG_ATTN_TP CFG_COMM
CONFIG_ORDER=(
    "tp8_flashinfer"
    "tp8_fa3"
    "tp8_dcp2_a2a"
    "tp8_dcp2_agrs"
    "tp8_tpa4_dcp2_a2a"
    "tp8_tpa4_dcp2_agrs"
    "tp8_tpa2_dcp4_a2a"
    "tp8_tpa2_dcp4_agrs"
)

CFG_BACKEND=(
    ["tp8_flashinfer"]="flashinfer" ["tp8_fa3"]="fa3"
    ["tp8_dcp2_a2a"]="flashinfer" ["tp8_dcp2_agrs"]="flashinfer"
    ["tp8_tpa4_dcp2_a2a"]="flashinfer" ["tp8_tpa4_dcp2_agrs"]="flashinfer"
    ["tp8_tpa2_dcp4_a2a"]="flashinfer" ["tp8_tpa2_dcp4_agrs"]="flashinfer"
)
CFG_DCP=(
    ["tp8_flashinfer"]=0 ["tp8_fa3"]=0
    ["tp8_dcp2_a2a"]=2 ["tp8_dcp2_agrs"]=2
    ["tp8_tpa4_dcp2_a2a"]=2 ["tp8_tpa4_dcp2_agrs"]=2
    ["tp8_tpa2_dcp4_a2a"]=4 ["tp8_tpa2_dcp4_agrs"]=4
)
CFG_ATTN_TP=(
    ["tp8_flashinfer"]=0 ["tp8_fa3"]=0
    ["tp8_dcp2_a2a"]=0 ["tp8_dcp2_agrs"]=0
    ["tp8_tpa4_dcp2_a2a"]=4 ["tp8_tpa4_dcp2_agrs"]=4
    ["tp8_tpa2_dcp4_a2a"]=2 ["tp8_tpa2_dcp4_agrs"]=2
)
CFG_COMM=(
    ["tp8_flashinfer"]="ag_rs" ["tp8_fa3"]="ag_rs"
    ["tp8_dcp2_a2a"]="a2a" ["tp8_dcp2_agrs"]="ag_rs"
    ["tp8_tpa4_dcp2_a2a"]="a2a" ["tp8_tpa4_dcp2_agrs"]="ag_rs"
    ["tp8_tpa2_dcp4_a2a"]="a2a" ["tp8_tpa2_dcp4_agrs"]="ag_rs"
)

# ---- Helper functions ----

wait_for_server() {
    local max_wait=${SERVER_WAIT}
    local elapsed=0
    echo "  Waiting for server on ${HOST}:${PORT} ..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q 200; then
            echo "  Server ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "  ERROR: Server not ready within ${max_wait}s"
    return 1
}

kill_server() {
    pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    pkill -f "sglang::schedul" 2>/dev/null || true
    pkill -f "sglang::detoken" 2>/dev/null || true
    sleep 5
    pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    pkill -9 -f "sglang::schedul" 2>/dev/null || true
    sleep 3
}

build_server_args() {
    local cfg_name="$1"
    local backend="${CFG_BACKEND[$cfg_name]}"
    local dcp="${CFG_DCP[$cfg_name]}"
    local attn_tp="${CFG_ATTN_TP[$cfg_name]}"
    local comm="${CFG_COMM[$cfg_name]}"

    local args="--attention-backend ${backend} --enable-symm-mem"
    if [ "$dcp" -gt 0 ]; then
        args="${args} --dcp-size ${dcp} --dcp-comm-backend ${comm}"
    fi
    if [ "$attn_tp" -gt 0 ]; then
        args="${args} --attention-tensor-parallel-size ${attn_tp}"
    fi
    echo "$args"
}

run_warmup() {
    echo "  Warmup (cc=${WARMUP_CC}, ${WARMUP_PROMPTS} prompts)..."
    python3 -m sglang.bench_serving --backend sglang \
        --host "$HOST" --port "$PORT" --model "$MODEL" \
        --dataset-name random --random-input-len "$INPUT_LEN" \
        --random-output-len 64 --random-range-ratio 0.1 \
        --num-prompts "$WARMUP_PROMPTS" --max-concurrency "$WARMUP_CC" \
        --disable-ignore-eos > /dev/null 2>&1
    echo "  Warmup done"
}

# ---- Phase 1: nsys profiling ----

run_nsys_profile() {
    local cfg_name="$1"
    local extra_args
    extra_args=$(build_server_args "$cfg_name")
    local output_dir="${BASE_OUTPUT}/nsys/${cfg_name}"
    mkdir -p "$output_dir"

    echo ""
    echo "############################################################"
    echo "# NSYS: ${cfg_name}"
    echo "############################################################"

    for cc in "${CONCURRENCIES[@]}"; do
        local tag="32k_4k_cc${cc}"
        local nsys_file="${output_dir}/${tag}"

        if [ -f "${nsys_file}.nsys-rep" ]; then
            echo "  Skip nsys ${cfg_name} ${tag} — exists"
            continue
        fi

        kill_server

        echo "  Starting server under nsys: ${cfg_name} cc=${cc}"
        nsys profile \
            --trace=${NSYS_TRACES} \
            ${NSYS_EXTRA} \
            --sample=none \
            --cpuctxsw=none \
            --output=${nsys_file} \
            --force-overwrite=true \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            python3 -m sglang.launch_server \
            --model-path ${MODEL} --host 0.0.0.0 --port ${PORT} \
            --tp 8 --disable-radix-cache \
            ${extra_args} &
        SERVER_PID=$!

        if ! wait_for_server; then
            echo "  FAILED — skipping"
            kill_server
            continue
        fi

        run_warmup

        local num_prompts=$((cc * 3))
        [ "$num_prompts" -lt 10 ] && num_prompts=10

        echo "  Triggering nsys capture..."
        curl -s -X POST "http://${HOST}:${PORT}/start_profile" \
            -H "Content-Type: application/json" \
            -d "{
                \"output_dir\": \"${output_dir}/torch_${tag}\",
                \"num_steps\": ${TORCH_PROFILE_STEPS},
                \"activities\": [\"CUDA_PROFILER\"],
                \"profile_by_stage\": false,
                \"merge_profiles\": true,
                \"profile_prefix\": \"nsys_${tag}\"
            }" &
        PROFILE_PID=$!

        echo "  Benchmark: cc=${cc}, in=${INPUT_LEN}, out=${OUTPUT_LEN}, prompts=${num_prompts}"
        python3 -m sglang.bench_serving --backend sglang \
            --host "$HOST" --port "$PORT" --model "$MODEL" \
            --dataset-name random --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" --random-range-ratio 0.1 \
            --num-prompts "$num_prompts" --max-concurrency "$cc" \
            --disable-ignore-eos 2>&1 | tee "${output_dir}/${tag}_bench.txt"

        wait $PROFILE_PID 2>/dev/null || true
        echo "  nsys done: ${nsys_file}.nsys-rep"
        kill_server
    done
}

# ---- Phase 2: torch profiler ----

run_torch_profile() {
    local cfg_name="$1"
    local extra_args
    extra_args=$(build_server_args "$cfg_name")
    local output_dir="${BASE_OUTPUT}/torch/${cfg_name}"
    mkdir -p "$output_dir"

    echo ""
    echo "############################################################"
    echo "# TORCH: ${cfg_name}"
    echo "############################################################"

    kill_server

    echo "  Starting server: ${cfg_name}"
    python3 -m sglang.launch_server \
        --model-path ${MODEL} --host 0.0.0.0 --port ${PORT} \
        --tp 8 --disable-radix-cache \
        ${extra_args} &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "  FAILED — skipping"
        kill_server
        return
    fi

    run_warmup

    for cc in "${CONCURRENCIES[@]}"; do
        local tag="32k_4k_cc${cc}"
        local trace_dir="${output_dir}/${tag}"
        mkdir -p "$trace_dir"

        if ls "${trace_dir}"/*.trace.json* 1>/dev/null 2>&1; then
            echo "  Skip torch ${cfg_name} ${tag} — exists"
            continue
        fi

        local num_prompts=$((cc * 3))
        [ "$num_prompts" -lt 10 ] && num_prompts=10

        echo "  Starting torch profiler (${tag}, cc=${cc})..."
        curl -s -X POST "http://${HOST}:${PORT}/start_profile" \
            -H "Content-Type: application/json" \
            -d "{
                \"output_dir\": \"${trace_dir}\",
                \"num_steps\": ${TORCH_PROFILE_STEPS},
                \"activities\": [\"CPU\", \"GPU\"],
                \"profile_by_stage\": true,
                \"merge_profiles\": true,
                \"profile_prefix\": \"${cfg_name}_${tag}\"
            }" &
        PROFILE_PID=$!

        echo "  Benchmark: cc=${cc}, prompts=${num_prompts}"
        python3 -m sglang.bench_serving --backend sglang \
            --host "$HOST" --port "$PORT" --model "$MODEL" \
            --dataset-name random --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" --random-range-ratio 0.1 \
            --num-prompts "$num_prompts" --max-concurrency "$cc" \
            --disable-ignore-eos 2>&1 | tee "${output_dir}/${tag}_bench.txt"

        wait $PROFILE_PID 2>/dev/null || true
        echo "  Torch traces: ${trace_dir}/"
    done

    kill_server
}

# ---- Summary ----

print_summary() {
    echo ""
    echo "############################################################"
    echo "# PROFILING SUMMARY"
    echo "############################################################"
    echo ""
    echo "Results: ${BASE_OUTPUT}/"
    echo ""
    echo "nsys reports (for gwe_perf_advisor):"
    find "${BASE_OUTPUT}/nsys" -name "*.nsys-rep" 2>/dev/null | sort | while read f; do
        echo "  $f"
    done
    echo ""
    echo "Analyze with nim-tuning-guide:"
    find "${BASE_OUTPUT}/nsys" -name "*.nsys-rep" 2>/dev/null | sort | while read f; do
        echo "  python3 -m gwe_perf_advisor.cli profile_diag --profile-path $f"
    done
    echo ""
    echo "Compare all configs:"
    echo "  python3 -m gwe_perf_advisor.cli profile_diag_cmp \\"
    local diag_files=()
    local diag_titles=()
    find "${BASE_OUTPUT}/nsys" -name "*.nsys-rep" 2>/dev/null | sort | while read f; do
        echo "    -i ${f%.nsys-rep}_diag.txt \\"
    done
    echo "    -o ${BASE_OUTPUT}/comparison/"
}

# ---- Main ----

mkdir -p "$BASE_OUTPUT"
echo "Profile: branch=${BRANCH} commit=${HASH}"
echo "Output: ${BASE_OUTPUT}/"
echo "Configs: ${CONFIG_ORDER[*]}"
echo "Workload: in=${INPUT_LEN} out=${OUTPUT_LEN} cc=${CONCURRENCIES[*]}"
echo "Filter: ${FILTER}, Phase: ${PHASE}"
echo ""

run_one() {
    local cfg="$1"
    if [ "$PHASE" = "nsys" ] || [ "$PHASE" = "all" ]; then
        run_nsys_profile "$cfg"
    fi
    if [ "$PHASE" = "torch" ] || [ "$PHASE" = "all" ]; then
        run_torch_profile "$cfg"
    fi
}

case "$FILTER" in
    all)
        for cfg in "${CONFIG_ORDER[@]}"; do
            run_one "$cfg"
        done
        ;;
    *)
        if [[ -v "CFG_BACKEND[$FILTER]" ]]; then
            run_one "$FILTER"
        else
            echo "Unknown config: $FILTER"
            echo "Available: ${CONFIG_ORDER[*]}"
            exit 1
        fi
        ;;
esac

print_summary
echo ""
echo "Profiling complete!"
