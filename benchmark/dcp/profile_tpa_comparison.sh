#!/bin/bash
# Profile TP8 vs TP8+DCP vs TP8+DCP+TPA to identify bottlenecks.
#
# Two-phase profiling for each config:
#   Phase 1: nsys — NCCL comms, CUDA kernels, NVTX markers
#   Phase 2: SGLang torch profiler — per-layer breakdown via /start_profile API
#
# Models supported:
#   7b      — CodeQwen1.5-7B-Chat (4 KV heads, 28 layers, dense)
#   72b     — Qwen2-72B-Instruct (8 KV heads, 80 layers, dense)
#   235b    — Qwen3-235B-A22B-Instruct-2507 (4 KV heads, 94 layers, MoE)
#
# Usage:
#   bash profile_tpa_comparison.sh [phase] [model]
#
#   bash profile_tpa_comparison.sh all 7b        # all configs, both phases, 7B model (default)
#   bash profile_tpa_comparison.sh nsys 235b     # nsys only on 235B MoE
#   bash profile_tpa_comparison.sh torch 72b     # torch profiler only on 72B
#   bash profile_tpa_comparison.sh tp8 7b        # single config on 7B
#   bash profile_tpa_comparison.sh tp8_tpa4_dcp2 235b  # single config on 235B
#
# Prerequisites:
#   - 8x H100 GPUs, nsys installed
#   - Models downloaded to HuggingFace cache
#   - Run inside sglang-bench-gqa container on colossus

set -euo pipefail

HOST=127.0.0.1
PORT=8188

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
HASH=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "0000000")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

COMMON_ENV="SGLANG_DCP_SYMM_ONLY=true NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"

# nsys capture parameters
# --cuda-graph-trace=node is required for gwe_perf_advisor profile_diag
NSYS_TRACES="cuda,nvtx,nccl,osrt"
NSYS_EXTRA="--cuda-graph-trace=node"

# torch profiler parameters
TORCH_PROFILE_STEPS=10  # forward steps to capture

FILTER="${1:-all}"
MODEL_TAG="${2:-7b}"

# ============================================================
# Per-model configuration
# ============================================================
setup_model() {
    case "$MODEL_TAG" in
        7b)
            MODEL="Qwen/CodeQwen1.5-7B-Chat"
            MODEL_SHORT="codeqwen_7b"
            MEM_FRAC="0.90"
            MEM_FRAC_DCP="0.90"
            CONTEXT_LENGTH=131072
            WARMUP_PROMPTS=10
            WARMUP_CC=4
            SERVER_WAIT=600
            WORKLOADS_INPUT=(4000 4000 4000 130000)
            WORKLOADS_OUTPUT=(500 500 500 64)
            WORKLOADS_CC=(1 32 128 2)
            WORKLOADS_TAG=("short_cc1" "short_cc32" "short_cc128" "long130k_cc2")
            ;;
        72b)
            MODEL="Qwen/Qwen2-72B-Instruct"
            MODEL_SHORT="qwen2_72b"
            MEM_FRAC="0.88"
            MEM_FRAC_DCP="0.88"
            CONTEXT_LENGTH=32768
            WARMUP_PROMPTS=5
            WARMUP_CC=2
            SERVER_WAIT=900
            WORKLOADS_INPUT=(4000 4000 4000 30000)
            WORKLOADS_OUTPUT=(500 500 500 64)
            WORKLOADS_CC=(1 16 64 2)
            WORKLOADS_TAG=("short_cc1" "short_cc16" "short_cc64" "long30k_cc2")
            ;;
        235b)
            MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
            MODEL_SHORT="qwen3_235b"
            MEM_FRAC="0.90"
            MEM_FRAC_DCP="0.85"
            CONTEXT_LENGTH=32768
            WARMUP_PROMPTS=3
            WARMUP_CC=2
            SERVER_WAIT=1200
            WORKLOADS_INPUT=(4000 4000 4000 30000)
            WORKLOADS_OUTPUT=(500 500 500 64)
            WORKLOADS_CC=(1 8 32 2)
            WORKLOADS_TAG=("short_cc1" "short_cc8" "short_cc32" "long30k_cc2")
            ;;
        *)
            echo "Unknown model: $MODEL_TAG"
            echo "Supported: 7b, 72b, 235b"
            exit 1
            ;;
    esac

    BASE_OUTPUT="${SCRIPT_DIR}/profiles/${BRANCH}_${HASH}_${MODEL_SHORT}_${TIMESTAMP}"
}

setup_model

# ============================================================
# Helper functions
# ============================================================

wait_for_server() {
    local max_wait=${SERVER_WAIT:-600}
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
    echo "  Killing server on port ${PORT} ..."
    pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    pkill -f "sglang::schedul" 2>/dev/null || true
    pkill -f "sglang::detoken" 2>/dev/null || true
    sleep 5
    # Force kill remaining
    pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    pkill -9 -f "sglang::schedul" 2>/dev/null || true
    sleep 3
}

run_warmup() {
    # Use the first workload's input length for warmup
    local warmup_input="${WORKLOADS_INPUT[0]}"
    echo "  Running warmup (cc=${WARMUP_CC}, ${WARMUP_PROMPTS} prompts, in=${warmup_input})..."
    python3 -m sglang.bench_serving --backend sglang \
        --host "$HOST" --port "$PORT" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$warmup_input" \
        --random-output-len 64 \
        --random-range-ratio 0.1 \
        --num-prompts "$WARMUP_PROMPTS" \
        --max-concurrency "$WARMUP_CC" \
        --disable-ignore-eos > /dev/null 2>&1
    echo "  Warmup done"
}

run_benchmark() {
    local cc="$1"
    local num_prompts="$2"
    local output_file="$3"
    local in_len="${4:-4000}"
    local out_len="${5:-500}"
    echo "  Benchmark: cc=${cc}, prompts=${num_prompts}, in=${in_len}, out=${out_len}"
    python3 -m sglang.bench_serving --backend sglang \
        --host "$HOST" --port "$PORT" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$in_len" \
        --random-output-len "$out_len" \
        --random-range-ratio 0.1 \
        --num-prompts "$num_prompts" \
        --max-concurrency "$cc" \
        --disable-ignore-eos 2>&1 | tee "$output_file"
}

get_mem_frac() {
    local cfg_name="$1"
    # DCP/TPA configs need more memory for symm buffers on large models
    if [[ "$cfg_name" == *"dcp"* ]] || [[ "$cfg_name" == *"tpa"* ]]; then
        echo "$MEM_FRAC_DCP"
    else
        echo "$MEM_FRAC"
    fi
}

start_server() {
    local cfg_name="$1"
    local extra_args="$2"
    local mem_frac
    mem_frac=$(get_mem_frac "$cfg_name")

    echo ""
    echo "======================================================="
    echo "Starting server: ${cfg_name} (${MODEL_SHORT})"
    echo "  mem_frac=${mem_frac} ctx=${CONTEXT_LENGTH}"
    echo "  Extra args: ${extra_args}"
    echo "======================================================="

    eval "${COMMON_ENV} python3 -m sglang.launch_server \
        --model-path ${MODEL} --host 0.0.0.0 --port ${PORT} \
        --trust-remote-code --log-level info --tp-size 8 \
        --chunked-prefill-size 32768 \
        --context-length ${CONTEXT_LENGTH} --disable-radix-cache --enable-symm-mem \
        --mem-fraction-static ${mem_frac} \
        --attention-backend fa3 \
        --enable-layerwise-nvtx-marker \
        ${extra_args}" &
    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"
}

start_server_under_nsys() {
    local cfg_name="$1"
    local extra_args="$2"
    local nsys_output="$3"
    local mem_frac
    mem_frac=$(get_mem_frac "$cfg_name")

    echo ""
    echo "======================================================="
    echo "[nsys] Starting server: ${cfg_name} (${MODEL_SHORT})"
    echo "  Output: ${nsys_output}"
    echo "  mem_frac=${mem_frac} ctx=${CONTEXT_LENGTH}"
    echo "======================================================="

    eval "${COMMON_ENV} nsys profile \
        --trace=${NSYS_TRACES} \
        ${NSYS_EXTRA} \
        --sample=none \
        --cpuctxsw=none \
        --output=${nsys_output} \
        --force-overwrite=true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        python3 -m sglang.launch_server \
        --model-path ${MODEL} --host 0.0.0.0 --port ${PORT} \
        --trust-remote-code --log-level info --tp-size 8 \
        --chunked-prefill-size 32768 \
        --context-length ${CONTEXT_LENGTH} --disable-radix-cache --enable-symm-mem \
        --mem-fraction-static ${mem_frac} \
        --attention-backend fa3 \
        --enable-layerwise-nvtx-marker \
        ${extra_args}" &
    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID} (nsys wrapped)"
}


# ============================================================
# Config definitions: name | server extra args
# ============================================================
declare -A CONFIGS
CONFIGS=(
    ["tp8"]=""
    ["tp8_dcp2"]="--dcp-size 2 --dcp-comm-backend a2a"
    ["tp8_tpa4_dcp2"]="--dcp-size 2 --dcp-comm-backend a2a --attention-tensor-parallel-size 4"
)
CONFIG_ORDER=("tp8" "tp8_dcp2" "tp8_tpa4_dcp2")


# ============================================================
# Phase 1: nsys profiling
#   Captures: NCCL comm, CUDA kernels, NVTX layerwise markers
#   Uses cudaProfilerApi range so we control exactly what's captured
# ============================================================
run_nsys_profile() {
    local cfg_name="$1"
    local extra_args="${CONFIGS[$cfg_name]}"
    local output_dir="${BASE_OUTPUT}/nsys/${cfg_name}"
    mkdir -p "$output_dir"

    echo ""
    echo "############################################################"
    echo "# NSYS PROFILING: ${cfg_name} (${MODEL_SHORT})"
    echo "############################################################"

    local num_workloads=${#WORKLOADS_TAG[@]}
    for ((i=0; i<num_workloads; i++)); do
        local tag="${WORKLOADS_TAG[$i]}"
        local cc="${WORKLOADS_CC[$i]}"
        local in_len="${WORKLOADS_INPUT[$i]}"
        local out_len="${WORKLOADS_OUTPUT[$i]}"
        local nsys_file="${output_dir}/${tag}"
        local bench_file="${output_dir}/${tag}_bench.txt"

        if [ -f "${nsys_file}.nsys-rep" ]; then
            echo "  Skipping nsys ${cfg_name} ${tag} — already exists"
            continue
        fi

        kill_server

        start_server_under_nsys "$cfg_name" "$extra_args" "$nsys_file"

        if ! wait_for_server; then
            echo "  FAILED to start ${cfg_name} — skipping"
            kill_server
            continue
        fi

        # Warmup
        run_warmup

        # Trigger nsys capture via cudaProfilerStart
        local num_prompts=$((cc * 3))
        [ "$num_prompts" -lt 10 ] && num_prompts=10
        echo "  Starting nsys capture (${TORCH_PROFILE_STEPS} steps, ${tag}, cc=${cc}, in=${in_len})..."
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

        # Run benchmark with workload-specific params
        echo "  Benchmark: ${tag} cc=${cc}, in=${in_len}, out=${out_len}, prompts=${num_prompts}"
        python3 -m sglang.bench_serving --backend sglang \
            --host "$HOST" --port "$PORT" \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len "$in_len" \
            --random-output-len "$out_len" \
            --random-range-ratio 0.1 \
            --num-prompts "$num_prompts" \
            --max-concurrency "$cc" \
            --disable-ignore-eos 2>&1 | tee "$bench_file"

        wait $PROFILE_PID 2>/dev/null || true

        echo "  nsys capture done: ${nsys_file}.nsys-rep"
        kill_server
    done
}


# ============================================================
# Phase 2: SGLang torch profiler
#   Captures: per-layer time, kernel launch overhead, memory
#   Outputs: Chrome trace JSON (viewable in chrome://tracing)
# ============================================================
run_torch_profile() {
    local cfg_name="$1"
    local extra_args="${CONFIGS[$cfg_name]}"
    local output_dir="${BASE_OUTPUT}/torch/${cfg_name}"
    mkdir -p "$output_dir"

    echo ""
    echo "############################################################"
    echo "# TORCH PROFILING: ${cfg_name} (${MODEL_SHORT})"
    echo "############################################################"

    kill_server

    start_server "$cfg_name" "$extra_args"

    if ! wait_for_server; then
        echo "  FAILED to start ${cfg_name} — skipping"
        kill_server
        return
    fi

    run_warmup

    local num_workloads=${#WORKLOADS_TAG[@]}
    for ((i=0; i<num_workloads; i++)); do
        local tag="${WORKLOADS_TAG[$i]}"
        local cc="${WORKLOADS_CC[$i]}"
        local in_len="${WORKLOADS_INPUT[$i]}"
        local out_len="${WORKLOADS_OUTPUT[$i]}"
        local trace_dir="${output_dir}/${tag}"
        local bench_file="${output_dir}/${tag}_bench.txt"
        mkdir -p "$trace_dir"

        if ls "${trace_dir}"/*.trace.json* 1>/dev/null 2>&1; then
            echo "  Skipping torch profile ${cfg_name} ${tag} — traces exist"
            continue
        fi

        local num_prompts=$((cc * 3))
        [ "$num_prompts" -lt 10 ] && num_prompts=10

        echo "  Starting torch profiler (${TORCH_PROFILE_STEPS} steps, ${tag}, cc=${cc}, in=${in_len})..."
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

        echo "  Benchmark: ${tag} cc=${cc}, in=${in_len}, out=${out_len}, prompts=${num_prompts}"
        python3 -m sglang.bench_serving --backend sglang \
            --host "$HOST" --port "$PORT" \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len "$in_len" \
            --random-output-len "$out_len" \
            --random-range-ratio 0.1 \
            --num-prompts "$num_prompts" \
            --max-concurrency "$cc" \
            --disable-ignore-eos 2>&1 | tee "$bench_file"

        wait $PROFILE_PID 2>/dev/null || true

        echo "  Torch traces saved to: ${trace_dir}/"
    done

    kill_server
}


# ============================================================
# Summary: extract key metrics from bench outputs
# ============================================================
print_summary() {
    echo ""
    echo "############################################################"
    echo "# PROFILING SUMMARY (${MODEL_SHORT})"
    echo "############################################################"
    echo ""
    echo "Results directory: ${BASE_OUTPUT}/"
    echo ""

    # Table header
    printf "%-20s" "Config"
    for tag in "${WORKLOADS_TAG[@]}"; do
        printf "  %-12s %-10s" "${tag}" ""
    done
    echo ""
    printf "%-20s" ""
    for tag in "${WORKLOADS_TAG[@]}"; do
        printf "  %-12s %-10s" "tok/s" "ITL(ms)"
    done
    echo ""
    printf "%-20s" "--------------------"
    for tag in "${WORKLOADS_TAG[@]}"; do
        printf "  %-12s %-10s" "------------" "----------"
    done
    echo ""

    for cfg_name in "${CONFIG_ORDER[@]}"; do
        printf "%-20s" "$cfg_name"
        for tag in "${WORKLOADS_TAG[@]}"; do
            local bench_file=""
            for phase_dir in nsys torch; do
                local candidate="${BASE_OUTPUT}/${phase_dir}/${cfg_name}/${tag}_bench.txt"
                if [ -f "$candidate" ]; then
                    bench_file="$candidate"
                    break
                fi
            done

            if [ -n "$bench_file" ]; then
                local tput itl
                tput=$(grep "Output token throughput" "$bench_file" 2>/dev/null | awk '{print $(NF-1)}' || echo "—")
                itl=$(grep "Mean ITL" "$bench_file" 2>/dev/null | head -1 | awk '{print $(NF-1)}' || echo "—")
                printf "  %-12s %-10s" "${tput:-—}" "${itl:-—}"
            else
                printf "  %-12s %-10s" "—" "—"
            fi
        done
        echo ""
    done

    echo ""
    echo "=== Files Generated ==="
    echo ""
    echo "nsys reports (open with nsys-ui or gwe_perf_advisor):"
    find "${BASE_OUTPUT}/nsys" -name "*.nsys-rep" 2>/dev/null | sort | while read f; do
        echo "  $f"
    done
    echo ""
    echo "Torch traces (open in chrome://tracing or Perfetto UI):"
    find "${BASE_OUTPUT}/torch" -name "*.trace.json*" 2>/dev/null | sort | while read f; do
        echo "  $f"
    done
    echo ""
    echo "Run gwe_perf_advisor on each nsys-rep:"
    find "${BASE_OUTPUT}/nsys" -name "*.nsys-rep" 2>/dev/null | sort | while read f; do
        echo "  python3 -m gwe_perf_advisor.cli profile_diag --profile-path $f"
    done
}


# ============================================================
# Main dispatch
# ============================================================

mkdir -p "$BASE_OUTPUT"

echo "Profile run: branch=${BRANCH} commit=${HASH}"
echo "Output dir: ${BASE_OUTPUT}/"
echo "Model: ${MODEL} (${MODEL_SHORT})"
echo "Configs: ${CONFIG_ORDER[*]}"
echo "Workloads: ${WORKLOADS_TAG[*]}"
echo "Filter: ${FILTER}"
echo ""

case "$FILTER" in
    # Run specific config (both phases)
    tp8)
        run_nsys_profile "tp8"
        run_torch_profile "tp8"
        ;;
    tp8_dcp2)
        run_nsys_profile "tp8_dcp2"
        run_torch_profile "tp8_dcp2"
        ;;
    tp8_tpa4_dcp2)
        run_nsys_profile "tp8_tpa4_dcp2"
        run_torch_profile "tp8_tpa4_dcp2"
        ;;

    # Run specific phase (all configs)
    nsys)
        for cfg in "${CONFIG_ORDER[@]}"; do
            run_nsys_profile "$cfg"
        done
        ;;
    torch)
        for cfg in "${CONFIG_ORDER[@]}"; do
            run_torch_profile "$cfg"
        done
        ;;

    # Run everything
    all)
        for cfg in "${CONFIG_ORDER[@]}"; do
            run_nsys_profile "$cfg"
        done
        for cfg in "${CONFIG_ORDER[@]}"; do
            run_torch_profile "$cfg"
        done
        ;;
    *)
        echo "Unknown filter: $FILTER"
        echo "Usage: $0 [all|nsys|torch|tp8|tp8_dcp2|tp8_tpa4_dcp2] [7b|72b|235b]"
        exit 1
        ;;
esac

print_summary

echo ""
echo "======================================================="
echo "Profiling complete! Results in: ${BASE_OUTPUT}/"
echo "======================================================="
