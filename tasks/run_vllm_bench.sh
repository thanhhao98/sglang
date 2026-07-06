#!/bin/bash
###############################################################################
# vLLM DCP A2A Benchmark on B200
#
# Runs on LOCAL machine. SSHs into B200, clones vLLM, builds docker, and
# runs bench_serving equivalent benchmarks for comparison with SGLang C3.
#
# Usage: ./tasks/run_vllm_bench.sh b200_1
###############################################################################
set -uo pipefail

MACHINE="${1:-b200_1}"

case "$MACHINE" in
    b200_1)
        SSH_HOST="colossus_b200_1"
        CODE_PATH="/localhome/local-htphan/sglang_bench/vllm"
        HF_CACHE="/localhome/local-htphan/sglang_bench/hf_cache"
        OUTPUT_HOST_DIR="/localhome/local-htphan/sglang_bench/output"
        DOCKER_CMD="sudo docker"
        ;;
    *)
        echo "Usage: $0 {b200_1}"
        exit 1
        ;;
esac

MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=30000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

CONCURRENCIES=(1 2 4 8 16 32 64 128 256 512)
NUM_PROMPTS=5
VLLM_IMAGE="vllm/vllm-openai:latest"
CONTAINER_NAME="vllm-bench"

# Context configs: label, context_length, ISL, mem_frac
CONTEXT_LABELS=(128K 256K 512K 1M)

get_ctx_len() {
    case "$1" in
        128K) echo 131072 ;; 256K) echo 262144 ;; 512K) echo 524288 ;; 1M) echo 1048576 ;;
    esac
}
get_ctx_isl() {
    case "$1" in
        128K) echo 130048 ;; 256K) echo 261120 ;; 512K) echo 523264 ;; 1M) echo 1047552 ;;
    esac
}
get_mem_frac() {
    case "$1" in
        128K) echo 0.85 ;; 256K) echo 0.80 ;; 512K) echo 0.75 ;; 1M) echo 0.65 ;;
    esac
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

remote() { ssh "$SSH_HOST" "$@"; }

cleanup_container() {
    log "Cleaning up container $CONTAINER_NAME..."
    remote "$DOCKER_CMD rm -f $CONTAINER_NAME" 2>/dev/null || true
}

check_gpus_free() {
    local gpu_procs
    gpu_procs=$(remote "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    gpu_procs=$(echo "$gpu_procs" | tr -d '[:space:]')
    if [[ "$gpu_procs" -gt 0 ]]; then
        log "WARNING: $gpu_procs GPU processes. Waiting 30s..."
        sleep 30
        gpu_procs=$(remote "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l" 2>/dev/null || echo "0")
        gpu_procs=$(echo "$gpu_procs" | tr -d '[:space:]')
        if [[ "$gpu_procs" -gt 0 ]]; then
            log "ERROR: GPUs still occupied"; return 1
        fi
    fi
    return 0
}

wait_for_server() {
    local container="$1" timeout_s="${2:-600}"
    log "Waiting for vLLM server health (timeout=${timeout_s}s)..."
    local i=0
    while [[ $i -lt $timeout_s ]]; do
        i=$((i + 1))
        if remote "$DOCKER_CMD exec $container curl -s http://localhost:${PORT}/health" >/dev/null 2>&1; then
            log "Server ready after ${i}s"
            return 0
        fi
        sleep 1
        if [[ $((i % 30)) -eq 0 ]]; then
            if ! remote "$DOCKER_CMD inspect --format='{{.State.Running}}' $container" 2>/dev/null | grep -q true; then
                log "Container exited prematurely"
                return 1
            fi
        fi
    done
    log "ERROR: Server health check timed out"
    return 1
}

server_healthy() {
    remote "$DOCKER_CMD exec $CONTAINER_NAME curl -s http://localhost:${PORT}/health" >/dev/null 2>&1
}

CURRENT_CONTAINER=""
trap_cleanup() {
    log "Caught signal, cleaning up..."
    if [[ -n "$CURRENT_CONTAINER" ]]; then
        cleanup_container
    fi
    exit 1
}
trap trap_cleanup INT TERM

# ─── Main ────────────────────────────────────────────────────────────────────

log "=========================================="
log "vLLM DCP A2A Benchmark: $MACHINE"
log "=========================================="

# Step 1: Pull vLLM docker image
log "Pulling vLLM docker image on $SSH_HOST..."
remote "$DOCKER_CMD pull $VLLM_IMAGE" 2>&1 | tail -5
log "Image pulled."

# Step 2: Create output dir
OUTPUT_DIR="${OUTPUT_HOST_DIR}/vllm_dcp"
remote "mkdir -p $OUTPUT_DIR"

# Step 3: Rsync sglang code (for bench_serving tool)
log "Syncing sglang code for bench_serving..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'tasks/output' \
    "$(dirname "$SCRIPT_DIR")/" \
    "${SSH_HOST}:/localhome/local-htphan/sglang_bench/sglang/"
log "Code synced."

# Step 4: Run benchmarks
# Config: vLLM DCP8 a2a (equivalent to SGLang C3)
CONFIG="V1"
VLLM_SERVER_ARGS="--model $MODEL --tensor-parallel-size 8 --trust-remote-code --decode-context-parallel-size 8 --dcp-comm-backend a2a --disable-log-requests --gpu-memory-utilization"

for ctx_label in "${CONTEXT_LABELS[@]}"; do
    CTX=$(get_ctx_len "$ctx_label")
    ISL=$(get_ctx_isl "$ctx_label")
    MEM_FRAC=$(get_mem_frac "$ctx_label")
    SERVER_LOG="/output/${CONFIG}_${ctx_label}_server.log"

    log ""
    log "============================================"
    log "vLLM DCP a2a: Context=$ctx_label"
    log "============================================"

    # Check if all JSONL exist (resume)
    ALL_EXIST=true
    for conc in "${CONCURRENCIES[@]}"; do
        if ! remote "test -f '${OUTPUT_DIR}/${CONFIG}_${ctx_label}_c${conc}.jsonl'" 2>/dev/null; then
            ALL_EXIST=false
            break
        fi
    done
    if $ALL_EXIST; then
        log "All JSONL files exist, skipping (resume)."
        continue
    fi

    if ! check_gpus_free; then
        log "FAIL: GPUs occupied, skipping $ctx_label"
        continue
    fi

    cleanup_container
    CURRENT_CONTAINER="$CONTAINER_NAME"

    # Start vLLM container with server
    DOCKER_RUN="$DOCKER_CMD run -d --name $CONTAINER_NAME \
        --gpus all --ipc=host --network=host \
        -v ${HF_CACHE}:/root/.cache/huggingface \
        -v /localhome/local-htphan/sglang_bench/sglang:/sglang \
        -v ${OUTPUT_DIR}:/output \
        $VLLM_IMAGE \
        vllm serve $VLLM_SERVER_ARGS $MEM_FRAC \
        --max-model-len $CTX \
        --host 0.0.0.0 --port $PORT"

    log "Starting vLLM container..."
    remote "$DOCKER_RUN" 2>&1

    if ! wait_for_server "$CONTAINER_NAME" 600; then
        log "FAIL: vLLM server failed to start for $ctx_label"
        remote "$DOCKER_CMD logs $CONTAINER_NAME 2>&1 | tail -30" 2>/dev/null
        cleanup_container
        CURRENT_CONTAINER=""
        sleep 10
        continue
    fi

    # Install sglang bench_serving in the vLLM container
    log "Installing bench_serving in vLLM container..."
    remote "$DOCKER_CMD exec $CONTAINER_NAME pip install -e /sglang/python 2>&1 | tail -3" 2>/dev/null

    # Run concurrency sweep
    for conc in "${CONCURRENCIES[@]}"; do
        JSONL_PATH="/output/${CONFIG}_${ctx_label}_c${conc}.jsonl"
        BENCH_LOG="/output/${CONFIG}_${ctx_label}_c${conc}_bench.log"

        if remote "test -f '${OUTPUT_DIR}/${CONFIG}_${ctx_label}_c${conc}.jsonl'" 2>/dev/null; then
            log "  c=$conc: JSONL exists, skipping"
            continue
        fi

        if ! server_healthy; then
            log "  Server dead before c=$conc, stopping"
            break
        fi

        log "  Benchmarking c=$conc ..."
        BENCH_SCRIPT="#!/bin/bash
cd /sglang
timeout 1800 python3 -m sglang.bench_serving --backend vllm --host 127.0.0.1 --port $PORT --model $MODEL --dataset-name random --random-input-len $ISL --random-output-len 1024 --random-range-ratio 0.0 --num-prompts $NUM_PROMPTS --max-concurrency $conc --disable-ignore-eos --output-file $JSONL_PATH 2>&1 | tee $BENCH_LOG"

        ENCODED=$(echo "$BENCH_SCRIPT" | base64)
        remote "$DOCKER_CMD exec $CONTAINER_NAME bash -c 'printf \"%s\" \"$ENCODED\" | base64 -d > /tmp/bench.sh && chmod +x /tmp/bench.sh'" 2>/dev/null
        remote "$DOCKER_CMD exec $CONTAINER_NAME /tmp/bench.sh" 2>&1

        if ! server_healthy; then
            log "  Server crashed after c=$conc"
            break
        fi
    done

    cleanup_container
    CURRENT_CONTAINER=""
    log "Waiting 10s for GPU cleanup..."
    sleep 10
done

# Step 5: Sync results locally
LOCAL_DIR="${SCRIPT_DIR}/output/vllm_dcp_b200_1"
log "Syncing results to $LOCAL_DIR ..."
mkdir -p "$LOCAL_DIR"
rsync -avz "${SSH_HOST}:${OUTPUT_DIR}/" "$LOCAL_DIR/"

log ""
log "=========================================="
log "vLLM benchmark complete!"
log "Results: $LOCAL_DIR"
log "=========================================="
