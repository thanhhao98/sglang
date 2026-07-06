#!/bin/bash
# Verify DCP A2A optimizations (Opt 1: remove .contiguous(), Opt 3: per-bs buffers)
#
# Run from LOCAL machine. Steps:
#   1. Sync code to H100
#   2. SSH and run accuracy + performance tests inside docker
#
# Baseline (before optimization):
#   H100 DCP8 a2a flashinfer 128K c=1: TPOT 4.34ms
#   B200 DCP8 a2a flashinfer 128K c=1: TPOT 4.84ms
#
# Expected improvement:
#   - Opt 1: ~54 fewer D2D copy kernels per decode step
#   - Opt 3: NCCL transfers bs-sized data instead of max_bs=512
#   - Target: A2A c=1 TPOT ~3-4ms (below AG_RS ~4.19ms)

set -euo pipefail

MACHINE="${1:-h100}"  # h100 or b200

case "$MACHINE" in
    h100)
        SSH_HOST="colossus"
        CODE_PATH="/localhome/local-htphan/helix/sglang"
        DOCKER_CONTAINER="sglang-bench"
        HF_CACHE="/root/.cache/huggingface"
        ;;
    b200)
        SSH_HOST="colossus_b200_1"
        CODE_PATH="/localhome/local-htphan/sglang_bench/sglang"
        DOCKER_CONTAINER="sglang-bench"
        HF_CACHE="/root/.cache/huggingface"
        ;;
    *)
        echo "Usage: $0 {h100|b200}"
        exit 1
        ;;
esac

MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=30000

echo "============================================="
echo "DCP A2A Optimization Verification on $MACHINE"
echo "============================================="

# --- Step 1: Sync code ---
echo ""
echo "=== Step 1: Sync code to $SSH_HOST ==="
echo ""
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  /Users/htphan/workspace/DLAlgo/sglang/ \
  ${SSH_HOST}:${CODE_PATH}/

echo "Code synced. Now run the following commands on the server."
echo ""

# --- Generate the server-side script ---
cat << 'SERVERSCRIPT'
#############################################################
# Run these commands on the server inside docker:
#
#   ssh colossus  (or colossus_b200_1)
#   docker exec -it sglang-bench bash
#
# Then paste each section below.
#############################################################

# ============================================================
# PART A: Accuracy Test (GSM8K, DCP8 a2a flashinfer)
# ============================================================
# Expected: accuracy ~0.38-0.39 (same as before optimization)

export SGLANG_DCP=8
export SGLANG_DCP_SYMM_ONLY=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

# Start server
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --tp-size 8 --trust-remote-code \
  --context-length 131072 --mem-fraction-static 0.85 \
  --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
  --dcp-comm-backend a2a --chunked-prefill-size 32768 \
  --host 0.0.0.0 --port 30000 &

# Wait for server
sleep 120  # or watch logs for "The server is fired up"

# Run GSM8K accuracy
python3 benchmark/gsm8k/bench_sglang.py \
  --num-questions 200 \
  --parallel 32 \
  --host 127.0.0.1 --port 30000

# Expected output: accuracy ~0.38-0.39
# If accuracy is significantly different, the optimization broke something.

# Kill server
pkill -f "sglang.launch_server" || true
sleep 10

# ============================================================
# PART B: Performance Benchmark (bench_serving, 128K, c=1,2,8)
# ============================================================
# Compare TPOT against baseline:
#   H100 baseline: c=1: 4.34ms, c=2: 14.09ms, c=8: 30.36ms
#   B200 baseline: c=1: 4.84ms, c=2: 12.00ms, c=8: 24.30ms

export SGLANG_DCP=8
export SGLANG_DCP_SYMM_ONLY=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

# Start server
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --tp-size 8 --trust-remote-code \
  --context-length 131072 --mem-fraction-static 0.85 \
  --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
  --dcp-comm-backend a2a --chunked-prefill-size 32768 \
  --host 0.0.0.0 --port 30000 &

sleep 120

# Run bench_serving for c=1, c=2, c=8
for C in 1 2 8; do
  echo ""
  echo "========== bench_serving 128K c=$C =========="
  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --dataset-name random \
    --random-input-len 130048 \
    --random-output-len 1024 \
    --random-range-ratio 0.0 \
    --num-prompts 5 \
    --request-rate-range "$C" \
    --max-concurrency "$C" \
    2>&1 | tee /output/opt_verify_128k_c${C}.log
done

# Kill server
pkill -f "sglang.launch_server" || true
sleep 10

# ============================================================
# PART C: Performance Benchmark (256K, c=1)
# ============================================================
# Baseline: H100 c=1: 5.01ms, B200 c=1: 5.30ms

export SGLANG_DCP=8
export SGLANG_DCP_SYMM_ONLY=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --tp-size 8 --trust-remote-code \
  --context-length 262144 --mem-fraction-static 0.80 \
  --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
  --dcp-comm-backend a2a --chunked-prefill-size 32768 \
  --host 0.0.0.0 --port 30000 &

sleep 120

echo ""
echo "========== bench_serving 256K c=1 =========="
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random \
  --random-input-len 261120 \
  --random-output-len 1024 \
  --random-range-ratio 0.0 \
  --num-prompts 5 \
  --request-rate-range 1 \
  --max-concurrency 1 \
  2>&1 | tee /output/opt_verify_256k_c1.log

pkill -f "sglang.launch_server" || true
sleep 10

# ============================================================
# PART D: AG_RS baseline comparison (128K c=1)
# ============================================================
# Baseline: H100 ag_rs 128K c=1: 4.19ms

export SGLANG_DCP=8
export SGLANG_DCP_SYMM_ONLY=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --tp-size 8 --trust-remote-code \
  --context-length 131072 --mem-fraction-static 0.85 \
  --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
  --dcp-comm-backend ag_rs --chunked-prefill-size 32768 \
  --host 0.0.0.0 --port 30000 &

sleep 120

echo ""
echo "========== bench_serving AG_RS 128K c=1 =========="
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random \
  --random-input-len 130048 \
  --random-output-len 1024 \
  --random-range-ratio 0.0 \
  --num-prompts 5 \
  --request-rate-range 1 \
  --max-concurrency 1 \
  2>&1 | tee /output/opt_verify_agrs_128k_c1.log

pkill -f "sglang.launch_server" || true
sleep 10

# ============================================================
# PART E: Torch Profiler (128K, c=1, A2A vs AG_RS)
# ============================================================

export SGLANG_DCP=8
export SGLANG_DCP_SYMM_ONLY=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_TORCH_PROFILER_DIR=/output/profile_a2a_opt

python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --tp-size 8 --trust-remote-code \
  --context-length 131072 --mem-fraction-static 0.85 \
  --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
  --dcp-comm-backend a2a --chunked-prefill-size 32768 \
  --enable-layerwise-nvtx-marker \
  --host 0.0.0.0 --port 30000 &

sleep 120

# Send a 128K request to warm up and build KV cache
python3 -c "
import requests
passage = 'The quick brown fox jumps over the lazy dog. ' * 3000
resp = requests.post('http://127.0.0.1:30000/generate',
    json={'text': f'Summarize: {passage}\nSummary:', 'sampling_params': {'max_new_tokens': 64, 'temperature': 0}},
    timeout=600)
print(f'Warmup done: {resp.status_code}')
"

# Start profiling
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 10, "activities": ["CPU", "GPU"]}'

# Send another request to generate decode tokens during profiling
python3 -c "
import requests
passage = 'The quick brown fox jumps over the lazy dog. ' * 3000
resp = requests.post('http://127.0.0.1:30000/generate',
    json={'text': f'Explain: {passage}\nExplanation:', 'sampling_params': {'max_new_tokens': 64, 'temperature': 0}},
    timeout=600)
print(f'Profile request done: {resp.status_code}')
"

sleep 5

# Stop profiler
curl -X POST http://127.0.0.1:30000/stop_profile \
  -H "Content-Type: application/json"

echo "Profile saved to /output/profile_a2a_opt/"

pkill -f "sglang.launch_server" || true
sleep 10

echo ""
echo "============================================="
echo "Verification complete. Check results:"
echo "  Accuracy: compare with baseline 0.38-0.39"
echo "  TPOT 128K c=1: baseline 4.34ms (H100) / 4.84ms (B200)"
echo "  TPOT 256K c=1: baseline 5.01ms (H100) / 5.30ms (B200)"
echo "  Profile: /output/profile_a2a_opt/"
echo "============================================="
SERVERSCRIPT
