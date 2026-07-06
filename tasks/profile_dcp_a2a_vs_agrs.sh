#!/bin/bash
# Profile DCP A2A vs AG_RS decode to identify per-layer overhead.
# Run on B200 Node 1 (colossus_b200_1) with 128K context, c=1.
#
# Usage:
#   ./tasks/profile_dcp_a2a_vs_agrs.sh a2a    # Profile A2A path
#   ./tasks/profile_dcp_a2a_vs_agrs.sh ag_rs   # Profile AG_RS path
#   ./tasks/profile_dcp_a2a_vs_agrs.sh nsys_a2a   # nsys profile A2A
#   ./tasks/profile_dcp_a2a_vs_agrs.sh nsys_ag_rs  # nsys profile AG_RS

set -euo pipefail

MODE="${1:-a2a}"
MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=30000
CTX_LEN=131072
NUM_PROFILE_STEPS=10

# Common env vars for DCP8
export SGLANG_DCP=8
export SGLANG_DCP_SYMM_ONLY=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

COMMON_ARGS=(
    --model-path "$MODEL"
    --tp-size 8
    --trust-remote-code
    --context-length "$CTX_LEN"
    --mem-fraction-static 0.85
    --attention-backend flashinfer
    --disable-radix-cache
    --enable-symm-mem
    --chunked-prefill-size 32768
    --enable-layerwise-nvtx-marker
    --host 0.0.0.0
    --port "$PORT"
)

wait_for_server() {
    echo "Waiting for server to be ready..."
    for i in $(seq 1 120); do
        if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Server did not start within 120s"
    return 1
}

send_128k_request() {
    # Send a 128K prefill request (c=1) to build KV cache, then generate tokens
    echo "Sending 128K context request (c=1)..."
    python3 -c "
import requests, json

# Use a long prompt that fills ~128K tokens
# Repeat a passage to fill context
passage = 'The quick brown fox jumps over the lazy dog. ' * 3000
prompt = f'Summarize the following text:\n{passage}\nSummary:'

resp = requests.post(
    'http://127.0.0.1:${PORT}/generate',
    json={
        'text': prompt,
        'sampling_params': {
            'max_new_tokens': 64,
            'temperature': 0,
        },
    },
    timeout=600,
)
print(f'Status: {resp.status_code}')
data = resp.json()
print(f'Prompt tokens: {data.get(\"meta_info\", {}).get(\"prompt_tokens\", \"?\")}')
print(f'Output tokens: {data.get(\"meta_info\", {}).get(\"completion_tokens\", \"?\")}')
"
}

profile_decode() {
    local backend="$1"
    local profile_dir="/output/profile_${backend}"
    export SGLANG_TORCH_PROFILER_DIR="$profile_dir"
    mkdir -p "$profile_dir"

    echo "=== Profiling DCP $backend (torch.profiler) ==="
    echo "Profile dir: $profile_dir"

    # Start server
    python3 -m sglang.launch_server \
        "${COMMON_ARGS[@]}" \
        --dcp-comm-backend "$backend" &
    SERVER_PID=$!
    trap "kill $SERVER_PID 2>/dev/null || true" EXIT

    wait_for_server

    # Send request to build KV cache and start generating
    send_128k_request &
    REQUEST_PID=$!

    # Wait a bit for prefill to complete and decode to start
    sleep 30

    # Start profiling (captures decode steps)
    echo "Starting profile for $NUM_PROFILE_STEPS steps..."
    curl -X POST "http://127.0.0.1:${PORT}/start_profile" \
        -H "Content-Type: application/json" \
        -d "{\"num_steps\": ${NUM_PROFILE_STEPS}, \"activities\": [\"CPU\", \"GPU\"]}"

    # Wait for profiling to complete
    sleep 20

    # Stop profile
    curl -X POST "http://127.0.0.1:${PORT}/stop_profile" \
        -H "Content-Type: application/json"

    echo "Profile saved to $profile_dir"

    wait $REQUEST_PID 2>/dev/null || true
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    trap - EXIT
}

nsys_profile() {
    local backend="$1"
    local output_file="/output/nsys_${backend}"

    echo "=== Profiling DCP $backend (nsys) ==="
    echo "Output: ${output_file}.nsys-rep"

    # nsys with CUDA graph disabled for cleaner trace
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-graph-trace=node \
        -o "$output_file" \
        python3 -m sglang.launch_server \
            "${COMMON_ARGS[@]}" \
            --dcp-comm-backend "$backend" \
            --disable-cuda-graph &
    SERVER_PID=$!
    trap "kill $SERVER_PID 2>/dev/null || true" EXIT

    wait_for_server

    # Send request and let it generate tokens for profiling
    send_128k_request

    echo "nsys trace saved to ${output_file}.nsys-rep"

    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    trap - EXIT
}

case "$MODE" in
    a2a)
        profile_decode "a2a"
        ;;
    ag_rs)
        profile_decode "ag_rs"
        ;;
    nsys_a2a)
        nsys_profile "a2a"
        ;;
    nsys_ag_rs)
        nsys_profile "ag_rs"
        ;;
    both)
        profile_decode "a2a"
        echo ""
        echo "=== Switching to AG_RS ==="
        echo ""
        profile_decode "ag_rs"
        ;;
    *)
        echo "Usage: $0 {a2a|ag_rs|nsys_a2a|nsys_ag_rs|both}"
        exit 1
        ;;
esac

echo "Done. Compare traces in /output/profile_a2a/ and /output/profile_ag_rs/"
