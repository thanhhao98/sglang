#!/bin/bash
# Profile currently running server - start profiler FIRST, then send request
set -u
PORT=30000
PROFILE_DIR="${1:-/output/profile_a2a_symm_128k}"

rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

echo "=== Starting profiler first ==="
curl -s -X POST "http://127.0.0.1:${PORT}/start_profile" \
    -H "Content-Type: application/json" \
    -d '{"num_steps": 30, "activities": ["CPU", "GPU"], "record_shapes": true, "with_stack": false}'
echo ""

sleep 1

echo "=== Sending decode request (128 tokens) ==="
python3 -c "
import requests
passage = 'The quick brown fox jumps over the lazy dog. ' * 3000
resp = requests.post('http://127.0.0.1:${PORT}/generate',
    json={'text': f'Summarize: {passage}\nSummary:',
          'sampling_params': {'max_new_tokens': 128, 'temperature': 0}},
    timeout=600)
mi = resp.json().get('meta_info', {})
print(f'Done: {resp.status_code}, completion_tokens={mi.get(\"completion_tokens\",\"?\")}, prompt_tokens={mi.get(\"prompt_tokens\",\"?\")}')
"

sleep 3

echo "=== Stopping profiler ==="
curl -s -X POST "http://127.0.0.1:${PORT}/stop_profile" \
    -H "Content-Type: application/json"
echo ""

sleep 10
echo "=== Profile saved to $PROFILE_DIR ==="
ls -lh "$PROFILE_DIR"/*.gz 2>/dev/null | head -3
