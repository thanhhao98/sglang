#!/bin/bash
# Profile currently running server (must already be up on port 30000)
set -u
PORT=30000
PROFILE_DIR="${1:-/output/profile_a2a_symm_128k}"

rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"
export SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR"

echo "=== Sending long decode request in background ==="
python3 -c "
import requests
passage = 'The quick brown fox jumps over the lazy dog. ' * 3000
resp = requests.post('http://127.0.0.1:${PORT}/generate',
    json={'text': f'Explain in detail: {passage}',
          'sampling_params': {'max_new_tokens': 128, 'temperature': 0}},
    timeout=600)
print(f'Request done: {resp.status_code}')
" &
REQ_PID=$!

# Wait for prefill to finish and decode to start
sleep 5

echo "=== Starting profiler ==="
curl -s -X POST "http://127.0.0.1:${PORT}/start_profile" \
    -H "Content-Type: application/json" \
    -d '{"num_steps": 20, "activities": ["CPU", "GPU"], "record_shapes": true}'
echo ""

# Wait for request to complete (generates decode tokens during profiling)
wait $REQ_PID 2>/dev/null

sleep 5

echo "=== Stopping profiler ==="
curl -s -X POST "http://127.0.0.1:${PORT}/stop_profile" \
    -H "Content-Type: application/json"
echo ""

sleep 10
echo "=== Profile saved to $PROFILE_DIR ==="
ls -la "$PROFILE_DIR"/*.gz 2>/dev/null | head -3
