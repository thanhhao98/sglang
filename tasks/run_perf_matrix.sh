#!/bin/bash
###############################################################################
# Robust Performance Benchmarking Matrix for DCP
#
# Runs on LOCAL machine. SSHs into remote, starts fresh docker containers
# per (config, context) pair, runs bench_serving across concurrencies.
#
# Usage: ./tasks/run_perf_matrix.sh [h100|b200]
#
# Features:
# - Per-(config,context) container lifecycle (not one long-lived container)
# - Resume support: skips if all 10 JSONL files exist
# - GPU occupancy check before container start
# - Server crash detection mid-sweep
# - Structured JSONL output via --output-file
# - Cleanup trap on exit/interrupt
###############################################################################
set -uo pipefail

MACHINE="${1:-}"
if [[ -z "$MACHINE" ]]; then
    echo "Usage: $0 {h100|b200}"
    exit 1
fi

# ─── Machine config ──────────────────────────────────────────────────────────
case "$MACHINE" in
    h100)
        SSH_HOST="colossus"
        CODE_PATH="/localhome/local-htphan/helix/sglang"
        HF_CACHE="/raid/local-htphan/sglang_workspace/hf_cache"
        OUTPUT_HOST_DIR="/raid/local-htphan/sglang_workspace/output"
        DOCKER_CMD="docker"
        DOCKER_IMAGE="sglang-dcp-a2a:local"
        ;;
    b200)
        SSH_HOST="colossus_b200_2"
        CODE_PATH="/localhome/local-htphan/sglang_bench/sglang"
        HF_CACHE="/root/.cache/huggingface"
        OUTPUT_HOST_DIR="/localhome/local-htphan/sglang_bench/output"
        DOCKER_CMD="sudo docker"
        DOCKER_IMAGE="sglang-dcp-a2a:local"
        ;;
    b200_1)
        SSH_HOST="colossus_b200_1"
        CODE_PATH="/localhome/local-htphan/sglang_bench/sglang"
        HF_CACHE="/localhome/local-htphan/sglang_bench/hf_cache"
        OUTPUT_HOST_DIR="/localhome/local-htphan/sglang_bench/output"
        DOCKER_CMD="sudo docker"
        DOCKER_IMAGE="sglang-dcp-a2a:local"
        ;;
    *)
        echo "Usage: $0 {h100|b200|b200_1}"
        exit 1
        ;;
esac

MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=30000
COMMIT_HASH=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUTPUT_DIR="${SCRIPT_DIR}/output/${COMMIT_HASH}_${MACHINE}"

CONCURRENCIES=(1 2 4 8 16 32 64 128 256 512)
NUM_PROMPTS=5

# DCP environment variables (for docker -e flags)
DCP_ENV="-e SGLANG_DCP=8 -e SGLANG_DCP_SYMM_ONLY=true -e SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"

CONFIGS=(C1 C2 C3 C4 C5)
CONTEXT_LABELS=(128K 256K 512K 1M)

# ─── Lookup functions (bash 3.x compatible) ─────────────────────────────────

get_config_args() {
    case "$1" in
        C1) echo "--tp-size 8 --trust-remote-code" ;;
        C4) echo "--tp-size 8 --trust-remote-code --attention-backend flashinfer --disable-radix-cache" ;;
        C2) echo "--tp-size 8 --trust-remote-code --attention-backend flashinfer --disable-radix-cache --enable-symm-mem --dcp-comm-backend ag_rs --chunked-prefill-size 32768" ;;
        C3) echo "--tp-size 8 --trust-remote-code --attention-backend flashinfer --disable-radix-cache --enable-symm-mem --dcp-comm-backend a2a --chunked-prefill-size 32768" ;;
        C5) echo "--tp-size 8 --trust-remote-code --attention-backend flashinfer --disable-radix-cache --enable-symm-mem --dcp-comm-backend a2a --chunked-prefill-size 32768 --dcp-replicate-q-proj" ;;
    esac
}

config_needs_dcp() {
    case "$1" in
        C1|C4) return 1 ;;
        C2|C3|C5) return 0 ;;
    esac
}

get_ctx_len() {
    case "$1" in
        128K) echo 131072 ;;
        256K) echo 262144 ;;
        512K) echo 524288 ;;
        1M)   echo 1048576 ;;
    esac
}

get_ctx_isl() {
    case "$1" in
        128K) echo 130048 ;;
        256K) echo 261120 ;;
        512K) echo 523264 ;;
        1M)   echo 1047552 ;;
    esac
}

get_mem_frac() {
    local config="$1" ctx_label="$2"
    if [[ "$config" == "C5" ]]; then
        case "$ctx_label" in
            128K) echo 0.50 ;;
            256K) echo 0.80 ;;
            512K) echo 0.75 ;;
            1M)   echo 0.65 ;;
        esac
    else
        case "$ctx_label" in
            128K) echo 0.85 ;;
            256K) echo 0.80 ;;
            512K) echo 0.75 ;;
            1M)   echo 0.65 ;;
        esac
    fi
}

# ─── Helper functions ────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

remote() {
    ssh "$SSH_HOST" "$@"
}

docker_remote() {
    remote "$DOCKER_CMD $*"
}

cleanup_container() {
    local name="$1"
    log "Cleaning up container $name..."
    docker_remote "rm -f $name" 2>/dev/null || true
}

cleanup_stale_containers() {
    log "Cleaning up stale sglang-perf-* containers..."
    local containers
    containers=$(remote "$DOCKER_CMD ps -a --filter 'name=sglang-perf-' --format '{{.Names}}'" 2>/dev/null || true)
    for c in $containers; do
        docker_remote "rm -f $c" 2>/dev/null || true
    done
}

check_gpus_free() {
    log "Checking GPU availability..."
    local gpu_procs
    gpu_procs=$(remote "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    gpu_procs=$(echo "$gpu_procs" | tr -d '[:space:]')
    if [[ "$gpu_procs" -gt 0 ]]; then
        log "WARNING: $gpu_procs GPU processes detected. Waiting 30s..."
        sleep 30
        gpu_procs=$(remote "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l" 2>/dev/null || echo "0")
        gpu_procs=$(echo "$gpu_procs" | tr -d '[:space:]')
        if [[ "$gpu_procs" -gt 0 ]]; then
            log "ERROR: GPUs still occupied ($gpu_procs processes). Skipping."
            return 1
        fi
    fi
    return 0
}

append_fail() {
    local msg="$1"
    log "FAIL: $msg"
    remote "echo '$(date -u +%Y-%m-%dT%H:%M:%SZ) $msg' >> ${OUTPUT_HOST_DIR}/${COMMIT_HASH}/fail_tests.log"
}

check_all_jsonl_exist() {
    local config="$1" ctx_label="$2"
    for conc in "${CONCURRENCIES[@]}"; do
        local f="${OUTPUT_HOST_DIR}/${COMMIT_HASH}/${config}_${ctx_label}_c${conc}.jsonl"
        if ! remote "test -f '$f'" 2>/dev/null; then
            return 1
        fi
    done
    return 0
}

wait_for_server_in_container() {
    local container="$1" timeout_s="${2:-600}"
    log "Waiting for server health (timeout=${timeout_s}s)..."
    local i=0
    while [[ $i -lt $timeout_s ]]; do
        i=$((i + 1))
        if docker_remote "exec $container curl -s http://localhost:${PORT}/health" >/dev/null 2>&1; then
            log "Server ready after ${i}s"
            return 0
        fi
        sleep 1
        # Check container still running every 30s
        if [[ $((i % 30)) -eq 0 ]]; then
            if ! docker_remote "inspect --format='{{.State.Running}}' $container" 2>/dev/null | grep -q true; then
                log "Container $container exited prematurely"
                return 1
            fi
        fi
    done
    log "ERROR: Server health check timed out after ${timeout_s}s"
    return 1
}

server_healthy() {
    local container="$1"
    docker_remote "exec $container curl -s http://localhost:${PORT}/health" >/dev/null 2>&1
}

# ─── Trap for cleanup on exit ────────────────────────────────────────────────
CURRENT_CONTAINER=""
trap_cleanup() {
    log "Caught signal, cleaning up..."
    if [[ -n "$CURRENT_CONTAINER" ]]; then
        cleanup_container "$CURRENT_CONTAINER"
    fi
    exit 1
}
trap trap_cleanup INT TERM

# ─── Main ────────────────────────────────────────────────────────────────────

log "=========================================="
log "Performance Matrix: $MACHINE"
log "Commit: $COMMIT_HASH ($BRANCH)"
log "Configs: ${CONFIGS[*]}"
log "Contexts: ${CONTEXT_LABELS[*]}"
log "Concurrencies: ${CONCURRENCIES[*]}"
log "=========================================="

# Step 1: Rsync code to remote
log "Syncing code to $SSH_HOST:$CODE_PATH ..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'tasks/output' \
    "$(dirname "$SCRIPT_DIR")/" \
    "${SSH_HOST}:${CODE_PATH}/"
log "Code synced."

# Step 2: Create output dir and metadata
remote "mkdir -p ${OUTPUT_HOST_DIR}/${COMMIT_HASH}"

# Write metadata.json
METADATA=$(cat <<EOF
{
  "commit": "$COMMIT_HASH",
  "branch": "$BRANCH",
  "machine": "$MACHINE",
  "timestamp": "$TIMESTAMP",
  "model": "$MODEL",
  "configs": ["C1", "C2", "C5"],
  "contexts": ["128K", "256K", "512K", "1M"],
  "concurrencies": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
  "num_prompts": $NUM_PROMPTS
}
EOF
)
remote "cat > ${OUTPUT_HOST_DIR}/${COMMIT_HASH}/metadata.json << 'METAEOF'
${METADATA}
METAEOF"

# Step 3: Cleanup stale containers
cleanup_stale_containers

# Step 4: Run benchmark matrix
TOTAL=0
SKIPPED=0
PASSED=0
FAILED=0

for config in "${CONFIGS[@]}"; do
    for ctx_label in "${CONTEXT_LABELS[@]}"; do
        TOTAL=$((TOTAL + 1))
        CONTAINER_NAME="sglang-perf-${config}-${ctx_label}"

        log ""
        log "============================================"
        log "Config=$config Context=$ctx_label [$TOTAL/12]"
        log "============================================"

        # Resume check
        if check_all_jsonl_exist "$config" "$ctx_label"; then
            log "All JSONL files exist, skipping (resume)."
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # GPU check
        if ! check_gpus_free; then
            append_fail "${config}_${ctx_label}: GPUs occupied, skipped"
            FAILED=$((FAILED + 1))
            continue
        fi

        # Build server command
        CTX=$(get_ctx_len "$ctx_label")
        ISL=$(get_ctx_isl "$ctx_label")
        MEM_FRAC=$(get_mem_frac "$config" "$ctx_label")
        CONFIG_SERVER_ARGS=$(get_config_args "$config")
        SERVER_ARGS="${CONFIG_SERVER_ARGS} --model-path $MODEL --context-length $CTX --mem-fraction-static $MEM_FRAC --host 0.0.0.0 --port $PORT"
        SERVER_LOG="/output/${COMMIT_HASH}/${config}_${ctx_label}_server.log"

        # Build docker run command
        ENV_ARGS=""
        if config_needs_dcp "$config"; then
            ENV_ARGS="$DCP_ENV"
        fi
        ENV_ARGS="$ENV_ARGS -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"

        HF_MOUNT="-v ${HF_CACHE}:/root/.cache/huggingface"

        DOCKER_RUN_CMD="$DOCKER_CMD run -d --name $CONTAINER_NAME \
            --gpus all --ipc=host --network=host \
            $ENV_ARGS \
            -v ${CODE_PATH}:/sgl-workspace/sglang \
            $HF_MOUNT \
            -v ${OUTPUT_HOST_DIR}:/output \
            $DOCKER_IMAGE \
            bash -c 'cd /sgl-workspace/sglang/python && pip install -e . > /dev/null 2>&1 && \
            echo \"=== Server Start: ${config} ${ctx_label} commit=${COMMIT_HASH} ${TIMESTAMP} ===\" > $SERVER_LOG && \
            echo \"Args: $SERVER_ARGS\" >> $SERVER_LOG && \
            python3 -m sglang.launch_server $SERVER_ARGS 2>&1 | tee -a $SERVER_LOG'"

        log "Starting container $CONTAINER_NAME..."
        CURRENT_CONTAINER="$CONTAINER_NAME"
        cleanup_container "$CONTAINER_NAME"
        remote "$DOCKER_RUN_CMD"

        # Wait for server
        if ! wait_for_server_in_container "$CONTAINER_NAME" 600; then
            log "Server failed to start for ${config}_${ctx_label}"
            # Capture last 50 lines of server log
            TAIL=$(remote "tail -50 ${OUTPUT_HOST_DIR}/${COMMIT_HASH}/${config}_${ctx_label}_server.log" 2>/dev/null || echo "(no log)")
            append_fail "${config}_${ctx_label}: server failed to start. Last log: $TAIL"
            cleanup_container "$CONTAINER_NAME"
            CURRENT_CONTAINER=""
            FAILED=$((FAILED + 1))
            sleep 10
            continue
        fi

        # Run concurrency sweep
        BENCH_FAILED=0
        for conc in "${CONCURRENCIES[@]}"; do
            JSONL_PATH="/output/${COMMIT_HASH}/${config}_${ctx_label}_c${conc}.jsonl"
            BENCH_LOG="/output/${COMMIT_HASH}/${config}_${ctx_label}_c${conc}_bench.log"

            # Skip if this specific file exists
            if remote "test -f '${OUTPUT_HOST_DIR}/${COMMIT_HASH}/${config}_${ctx_label}_c${conc}.jsonl'" 2>/dev/null; then
                log "  c=$conc: JSONL exists, skipping"
                continue
            fi

            # Check server still alive
            if ! server_healthy "$CONTAINER_NAME"; then
                log "  Server dead before c=$conc, failing remaining"
                for remaining_conc in "${CONCURRENCIES[@]}"; do
                    if [[ "$remaining_conc" -ge "$conc" ]]; then
                        append_fail "${config}_${ctx_label}_c${remaining_conc}: server crashed"
                    fi
                done
                BENCH_FAILED=1
                break
            fi

            log "  Benchmarking c=$conc ..."
            # Write bench script via printf to avoid heredoc hanging through SSH
            BENCH_SCRIPT="#!/bin/bash
cd /sgl-workspace/sglang
echo '=== Bench: ${config} ${ctx_label} c=${conc} commit=${COMMIT_HASH} ===' > $BENCH_LOG
timeout 1800 python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $PORT --model $MODEL --dataset-name random --random-input-len $ISL --random-output-len 1024 --random-range-ratio 0.0 --num-prompts $NUM_PROMPTS --max-concurrency $conc --disable-ignore-eos --output-file $JSONL_PATH 2>&1 | tee -a $BENCH_LOG"
            docker_remote "exec $CONTAINER_NAME bash -c 'printf \"%s\" \"$(echo "$BENCH_SCRIPT" | base64)\" | base64 -d > /tmp/bench.sh && chmod +x /tmp/bench.sh'" 2>/dev/null

            if ! docker_remote "exec $CONTAINER_NAME /tmp/bench.sh" 2>&1; then
                log "  c=$conc: bench command failed"
                append_fail "${config}_${ctx_label}_c${conc}: bench_serving failed"
            fi

            # Post-bench health check
            if ! server_healthy "$CONTAINER_NAME"; then
                log "  Server crashed after c=$conc"
                for remaining_conc in "${CONCURRENCIES[@]}"; do
                    if [[ "$remaining_conc" -gt "$conc" ]]; then
                        append_fail "${config}_${ctx_label}_c${remaining_conc}: server crashed after c=$conc"
                    fi
                done
                BENCH_FAILED=1
                break
            fi
        done

        if [[ "$BENCH_FAILED" -eq 0 ]]; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
        fi

        # Cleanup container
        cleanup_container "$CONTAINER_NAME"
        CURRENT_CONTAINER=""
        log "Waiting 10s for GPU cleanup..."
        sleep 10
    done
done

# Step 5: Write DONE marker
MACHINE_UPPER=$(echo "$MACHINE" | tr '[:lower:]' '[:upper:]')
remote "echo 'Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)' > ${OUTPUT_HOST_DIR}/${COMMIT_HASH}/${MACHINE_UPPER}_DONE"

# Step 6: Rsync results to local
log ""
log "Syncing results to local: $LOCAL_OUTPUT_DIR"
mkdir -p "$LOCAL_OUTPUT_DIR"
rsync -avz "${SSH_HOST}:${OUTPUT_HOST_DIR}/${COMMIT_HASH}/" "$LOCAL_OUTPUT_DIR/"

log ""
log "=========================================="
log "COMPLETE: $MACHINE"
log "  Total: $TOTAL  Passed: $PASSED  Skipped: $SKIPPED  Failed: $FAILED"
log "  Results: $LOCAL_OUTPUT_DIR"
if [[ -f "$LOCAL_OUTPUT_DIR/fail_tests.log" ]]; then
    log "  Failures:"
    cat "$LOCAL_OUTPUT_DIR/fail_tests.log"
fi
log "=========================================="
