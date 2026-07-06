#!/bin/bash
###############################################################################
# Profile A2A-opt vs AG_RS at 128K c=1 using torch.profiler
# Captures decode-phase traces for per-layer NCCL/kernel breakdown
###############################################################################
set -u

MODEL="deepseek-ai/DeepSeek-V2-Lite"
HOST="0.0.0.0"
PORT=30000
BENCH_HOST="127.0.0.1"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

kill_server() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang.srt.managers" 2>/dev/null || true
    sleep 10
}

wait_for_server() {
    for i in $(seq 1 180); do
        if curl -s "http://${BENCH_HOST}:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server not ready"
    return 1
}

profile_backend() {
    local BACKEND=$1
    local CTX=$2
    local MEM_FRAC=$3
    local PROFILE_DIR="/output/profile_${BACKEND}_${CTX}"

    kill_server
    mkdir -p "$PROFILE_DIR"

    export SGLANG_DCP=8
    export SGLANG_DCP_SYMM_ONLY=true
    export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
    export SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR"

    echo ""
    echo "================================================"
    echo "Profiling: $BACKEND at ${CTX} (profile_dir=$PROFILE_DIR)"
    echo "================================================"

    python3 -m sglang.launch_server \
        --model-path $MODEL \
        --tp-size 8 --trust-remote-code \
        --context-length $CTX --mem-fraction-static $MEM_FRAC \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend $BACKEND --chunked-prefill-size 32768 \
        --host $HOST --port $PORT \
        > /output/server_profile_${BACKEND}_${CTX}.log 2>&1 &

    wait_for_server || return 1

    # Warmup: send a request to build KV cache and trigger CUDA graph capture
    echo "Sending warmup request..."
    python3 -c "
import requests
passage = 'The quick brown fox jumps over the lazy dog. ' * 3000
resp = requests.post('http://${BENCH_HOST}:${PORT}/generate',
    json={'text': f'Summarize: {passage}\nSummary:',
          'sampling_params': {'max_new_tokens': 32, 'temperature': 0}},
    timeout=600)
print(f'Warmup: {resp.status_code}, tokens={resp.json().get(\"meta_info\",{}).get(\"completion_tokens\",\"?\")}')
"

    # Start profiler
    echo "Starting profiler (10 steps)..."
    curl -s -X POST "http://${BENCH_HOST}:${PORT}/start_profile" \
        -H "Content-Type: application/json" \
        -d '{"num_steps": 10, "activities": ["CPU", "GPU"], "with_stack": false, "record_shapes": true}'
    echo ""

    # Send request during profiling — this generates decode tokens that get profiled
    echo "Sending profiled request..."
    python3 -c "
import requests
passage = 'Explain the theory of relativity in detail. ' * 3000
resp = requests.post('http://${BENCH_HOST}:${PORT}/generate',
    json={'text': f'{passage}\nExplanation:',
          'sampling_params': {'max_new_tokens': 64, 'temperature': 0}},
    timeout=600)
print(f'Profile req: {resp.status_code}, tokens={resp.json().get(\"meta_info\",{}).get(\"completion_tokens\",\"?\")}')
"

    # Wait for profiler to flush
    sleep 5

    # Stop profiler
    echo "Stopping profiler..."
    curl -s -X POST "http://${BENCH_HOST}:${PORT}/stop_profile" \
        -H "Content-Type: application/json"
    echo ""

    sleep 3

    echo "Profile saved to $PROFILE_DIR"
    ls -la "$PROFILE_DIR"/ 2>/dev/null | head -5

    kill_server
}

# Profile A2A-opt at 128K
profile_backend "a2a" 131072 0.50

# Profile AG_RS at 128K
profile_backend "ag_rs" 131072 0.85

# Profile A2A-opt at 512K
profile_backend "a2a" 524288 0.75

# Profile AG_RS at 512K
profile_backend "ag_rs" 524288 0.75

echo ""
echo "================================================"
echo "All profiling complete!"
echo "Traces:"
echo "  /output/profile_a2a_131072/"
echo "  /output/profile_ag_rs_131072/"
echo "  /output/profile_a2a_524288/"
echo "  /output/profile_ag_rs_524288/"
echo "================================================"
