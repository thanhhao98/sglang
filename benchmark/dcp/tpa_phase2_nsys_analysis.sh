#!/usr/bin/env bash
# TPA Phase-2: nsys profiling script for o_proj all-reduce overhead analysis
#
# Purpose: Quantify the communication overhead in the current TPA decode path
# vs DCP and identify the o_proj all-reduce gap vs TRT-LLM's ReduceScatter.
#
# Usage:
#   bash benchmark/dcp/tpa_phase2_nsys_analysis.sh <mode>
#   where mode is one of: pure_tp, dcp2_a2a, tpa2_dcp4_a2a
#
# Output: nsys report in benchmark/dcp/results/nsys_tpa_phase2/

set -euo pipefail

MODE="${1:-pure_tp}"
MODEL="Qwen/CodeQwen1.5-7B-Chat"
CONTEXT_LEN=1048576  # 1M tokens
CONCURRENCY=32
NUM_PROMPTS=8
OUTPUT_DIR="benchmark/dcp/results/nsys_tpa_phase2"
mkdir -p "$OUTPUT_DIR"

COMMON_ARGS="--model $MODEL --trust-remote-code --context-length $CONTEXT_LEN"

case "$MODE" in
  pure_tp)
    SERVER_ARGS="$COMMON_ARGS --tp-size 8"
    LABEL="pure_tp8"
    ;;
  dcp2_a2a)
    SERVER_ARGS="$COMMON_ARGS --tp-size 8 --dcp-size 2 --dcp-comm-backend a2a"
    LABEL="dcp2_a2a_tp8"
    ;;
  tpa2_dcp4_a2a)
    SERVER_ARGS="$COMMON_ARGS --tp-size 8 --attention-tensor-parallel-size 2 --dcp-size 4 --dcp-comm-backend a2a"
    LABEL="tpa2_dcp4_a2a_tp8"
    ;;
  tpa4_dcp2_a2a)
    SERVER_ARGS="$COMMON_ARGS --tp-size 8 --attention-tensor-parallel-size 4 --dcp-size 2 --dcp-comm-backend a2a"
    LABEL="tpa4_dcp2_a2a_tp8"
    ;;
  *)
    echo "Unknown mode: $MODE. Use pure_tp, dcp2_a2a, tpa2_dcp4_a2a, or tpa4_dcp2_a2a"
    exit 1
    ;;
esac

REPORT_NAME="${OUTPUT_DIR}/${LABEL}_c${CONCURRENCY}"

echo "=== TPA Phase-2 nsys Profile ==="
echo "Mode:        $MODE"
echo "Label:       $LABEL"
echo "Concurrency: $CONCURRENCY"
echo "Context:     $CONTEXT_LEN"
echo "Report:      $REPORT_NAME"
echo ""

# Step 1: Launch server under nsys with decode-focused warmup
echo "[1/3] Starting server under nsys..."
nsys profile \
  --output "$REPORT_NAME" \
  --trace cuda,nvtx,osrt \
  --capture-range cudaProfilerApi \
  --capture-range-end stop \
  --force-overwrite true \
  python -m sglang.launch_server \
    $SERVER_ARGS \
    --port 30000 \
    --disable-cuda-graph \
    --decode-attention-backend fa3 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "[2/3] Waiting for server readiness..."
for i in $(seq 1 120); do
  if curl -s http://localhost:30000/health > /dev/null 2>&1; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 1
done

# Step 2: Run decode-focused benchmark
echo "[3/3] Running decode benchmark..."
python -m sglang.bench_serving \
  --backend sglang \
  --port 30000 \
  --dataset-name random \
  --random-input $CONTEXT_LEN \
  --random-output 128 \
  --num-prompts $NUM_PROMPTS \
  --request-rate $CONCURRENCY \
  --output-file "${OUTPUT_DIR}/${LABEL}_bench.json"

# Stop server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=== Profile complete ==="
echo "View report: nsys stats $REPORT_NAME.nsys-rep"
echo ""
echo "Key kernels to compare across modes:"
echo "  1. ncclAllReduce (o_proj) -- should be zero with Phase-2A ReduceScatter"
echo "  2. ncclAllGather (Q heads) -- should be zero with TPA"
echo "  3. _dcp_lse_combine_kernel / _correct_attn_cp_out_kernel"
echo "  4. ncclReduceScatter -- target for Phase-2A"
echo "  5. layer_comm.postprocess_layer total -- should be <1ms"
echo ""
echo "Extract kernel times:"
echo "  nsys stats --report gpukernsum $REPORT_NAME.nsys-rep"
echo "  nsys stats --report nvtxsum $REPORT_NAME.nsys-rep"
