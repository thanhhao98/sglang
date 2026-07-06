#!/bin/bash
cd /sgl-workspace/sglang
mkdir -p /output/replicate_qproj_bench

export SGLANG_DCP=8
export SGLANG_DCP_SYMM_ONLY=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=30000

run_perf() {
    local tag=$1 comm=$2 ctx=$3 mem=$4 extra=$5
    local isl=$((ctx - 1024))

    echo ""
    echo "==============================="
    echo "PERF: $tag (ctx=$ctx, isl=$isl)"
    echo "==============================="

    pkill -f sglang.launch_server 2>/dev/null || true
    sleep 5

    local server_log="/output/replicate_qproj_bench/${tag}_server.log"
    python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size 8 --trust-remote-code \
        --context-length $ctx --mem-fraction-static $mem \
        --attention-backend flashinfer --disable-radix-cache --enable-symm-mem \
        --dcp-comm-backend $comm --chunked-prefill-size 32768 \
        $extra \
        --host 0.0.0.0 --port $PORT > "$server_log" 2>&1 &

    echo "  Waiting for server... (log: $server_log)"
    for i in $(seq 1 240); do
        curl -s http://localhost:$PORT/health >/dev/null 2>&1 && break
        sleep 1
    done

    if ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
        echo "  FAILED: Server did not start for $tag"
        tail -20 "$server_log"
        pkill -f sglang.launch_server 2>/dev/null || true
        return 1
    fi
    echo "  Server ready (took ${i}s)"

    for conc in 1 2 4 8 16 32 64 128 256 512; do
        echo "  --- ${tag} c=${conc} ---"
        python3 -m sglang.bench_serving --backend sglang \
            --host 127.0.0.1 --port $PORT \
            --dataset-name random --random-input-len $isl --random-output-len 1024 \
            --random-range-ratio 0.0 --num-prompts 5 \
            --max-concurrency $conc --disable-ignore-eos \
            2>&1 | tee /output/replicate_qproj_bench/${tag}_c${conc}.log
    done

    pkill -f sglang.launch_server 2>/dev/null || true
    sleep 5
}

run_perf C5_128K a2a 131072  0.50 "--dcp-replicate-q-proj"
run_perf C5_256K a2a 262144  0.80 "--dcp-replicate-q-proj"
run_perf C5_512K a2a 524288  0.75 "--dcp-replicate-q-proj"
run_perf C5_1M   a2a 1048576 0.65 "--dcp-replicate-q-proj"
run_perf C6_128K ag_rs 131072  0.85 "--dcp-replicate-q-proj"
run_perf C6_256K ag_rs 262144  0.80 "--dcp-replicate-q-proj"
run_perf C6_512K ag_rs 524288  0.75 "--dcp-replicate-q-proj"
run_perf C6_1M   ag_rs 1048576 0.65 "--dcp-replicate-q-proj"

echo "ALLDONE" > /output/replicate_qproj_bench/H100_DONE
echo "All H100 perf benchmarks complete"
