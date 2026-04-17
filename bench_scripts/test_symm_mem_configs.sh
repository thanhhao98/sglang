#!/usr/bin/env bash
# Test all symm-mem + cuda-graph configurations from error.md to verify fix.
# Run this inside the sglang-bench docker container on colossus_b200_1.
#
# Usage (on the local dev box):
#   ssh colossus_b200_1 docker exec sglang-bench \
#       bash /sgl-workspace/sglang/bench_scripts/test_symm_mem_configs.sh [cfg0 cfg1 ...]
#
# With no args, runs every configuration. With args, runs only the listed ones.
#
set -u

MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
PORT=30000
LOG_DIR=/tmp/symm_mem_test
READY_TIMEOUT=360     # seconds to wait for "fired up and ready to roll"
ACC_PARALLEL=32
ACC_MAX_NEW=256
ACC_NUM_QUESTIONS=80

mkdir -p "$LOG_DIR"

declare -A CFG_DESC
declare -A CFG_FLAGS

# --- 235B configs from error.md ---
CFG_DESC[cfg0]="tp8 + flashinfer + --enable-symm-mem (graph ON)"
CFG_FLAGS[cfg0]="--tp 8 --attention-backend flashinfer --enable-symm-mem"

CFG_DESC[cfg1]="tp8 + flashinfer (no symm-mem, baseline)"
CFG_FLAGS[cfg1]="--tp 8 --attention-backend flashinfer"

CFG_DESC[cfg2]="tp8 + flashinfer + --enable-symm-mem --disable-cuda-graph"
CFG_FLAGS[cfg2]="--tp 8 --attention-backend flashinfer --enable-symm-mem --disable-cuda-graph"

CFG_DESC[cfg3]="tp8 + dcp2 a2a + --enable-symm-mem (graph ON)"
CFG_FLAGS[cfg3]="--tp 8 --attention-backend flashinfer --enable-symm-mem --dcp-size 2 --dcp-comm-backend a2a"

CFG_DESC[cfg4]="tp8 + dcp2 ag_rs + --enable-symm-mem (graph ON)"
CFG_FLAGS[cfg4]="--tp 8 --attention-backend flashinfer --enable-symm-mem --dcp-size 2 --dcp-comm-backend ag_rs"

CFG_DESC[cfg5]="tp8 + dcp2 ag_rs + --enable-symm-mem --disable-cuda-graph"
CFG_FLAGS[cfg5]="--tp 8 --attention-backend flashinfer --enable-symm-mem --dcp-size 2 --dcp-comm-backend ag_rs --disable-cuda-graph"

CFG_DESC[cfg6]="tp8 + dcp2 a2a + --enable-symm-mem --disable-cuda-graph"
CFG_FLAGS[cfg6]="--tp 8 --attention-backend flashinfer --enable-symm-mem --dcp-size 2 --dcp-comm-backend a2a --disable-cuda-graph"

CFG_DESC[cfg7]="tp8 + dcp2 a2a (no symm-mem, graph ON)"
CFG_FLAGS[cfg7]="--tp 8 --attention-backend flashinfer --dcp-size 2 --dcp-comm-backend a2a"

CFG_DESC[cfg8]="tp8 + dcp2 a2a + tpa4 (no symm-mem, graph ON)"
CFG_FLAGS[cfg8]="--tp 8 --attention-backend flashinfer --dcp-size 2 --dcp-comm-backend a2a --attention-tensor-parallel-size 4"

CFG_DESC[cfg9]="tp8 + dcp2 ag_rs + tpa4 (no symm-mem, graph ON)"
CFG_FLAGS[cfg9]="--tp 8 --attention-backend flashinfer --dcp-size 2 --dcp-comm-backend ag_rs --attention-tensor-parallel-size 4"

CFG_DESC[cfg10]="tp8 + dcp2 ag_rs + tpa4 + --enable-symm-mem --disable-cuda-graph"
CFG_FLAGS[cfg10]="--tp 8 --attention-backend flashinfer --enable-symm-mem --dcp-size 2 --dcp-comm-backend ag_rs --attention-tensor-parallel-size 4 --disable-cuda-graph"

# --- Extra tpa configs with symm-mem + graph (NEW: should also be covered
# by the fix; these are the interesting ones for TPA benchmarking).
CFG_DESC[cfg11]="tp8 + dcp2 a2a + tpa4 + --enable-symm-mem (graph ON)"
CFG_FLAGS[cfg11]="--tp 8 --attention-backend flashinfer --enable-symm-mem --dcp-size 2 --dcp-comm-backend a2a --attention-tensor-parallel-size 4"

CFG_DESC[cfg12]="tp8 + dcp2 ag_rs + tpa4 + --enable-symm-mem (graph ON)"
CFG_FLAGS[cfg12]="--tp 8 --attention-backend flashinfer --enable-symm-mem --dcp-size 2 --dcp-comm-backend ag_rs --attention-tensor-parallel-size 4"

ORDER=(cfg1 cfg2 cfg0 cfg7 cfg6 cfg5 cfg3 cfg4 cfg8 cfg9 cfg10 cfg11 cfg12)
if [[ $# -gt 0 ]]; then
    ORDER=("$@")
fi

SUMMARY=()

wait_gpu_clear() {
    for _ in $(seq 1 30); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '{sum += $1} END {print sum+0}')
        if [[ "$used" -lt 1024 ]]; then
            return 0
        fi
        sleep 2
    done
    echo "WARN: GPU not cleared after 60s, continuing anyway"
}

run_cfg() {
    local name="$1"
    local flags="${CFG_FLAGS[$name]}"
    local desc="${CFG_DESC[$name]}"
    local log="$LOG_DIR/${name}.log"
    local acc_log="$LOG_DIR/${name}.acc.log"

    echo "=============================================================="
    echo "[$name] $desc"
    echo "flags: $flags"
    echo "log:   $log"
    echo "--------------------------------------------------------------"

    pkill -f sglang.launch_server >/dev/null 2>&1 || true
    sleep 3
    wait_gpu_clear

    # Start server in background
    (cd /sgl-workspace/sglang && \
        python3 -m sglang.launch_server --model-path "$MODEL" $flags --port "$PORT" \
        > "$log" 2>&1) &
    local server_pid=$!

    # Wait for ready
    local ready=0
    local start=$(date +%s)
    while true; do
        if grep -q "fired up and ready to roll" "$log" 2>/dev/null; then
            ready=1
            break
        fi
        if grep -qE "Scheduler watchdog timeout|SIGQUIT received|Address already in use|Out of memory|CUDA error" "$log" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            break
        fi
        local now=$(date +%s)
        if (( now - start > READY_TIMEOUT )); then
            break
        fi
        sleep 3
    done

    if [[ $ready -eq 0 ]]; then
        local elapsed=$(( $(date +%s) - start ))
        echo "[$name] FAIL: server did not become ready in ${elapsed}s"
        echo "       last 15 lines of $log:"
        tail -n 15 "$log" | sed 's/^/         /'
        SUMMARY+=("FAIL  $name  start-failed  $desc")
        pkill -f sglang.launch_server >/dev/null 2>&1 || true
        sleep 3
        wait_gpu_clear
        return
    fi

    echo "[$name] Server ready in $(( $(date +%s) - start ))s, running accuracy..."

    # Accuracy test (small, fast)
    local acc_output
    if timeout 600 python3 -m sglang.test.few_shot_gsm8k \
            --num-questions $ACC_NUM_QUESTIONS \
            --parallel $ACC_PARALLEL \
            --max-new-tokens $ACC_MAX_NEW \
            --port "$PORT" > "$acc_log" 2>&1; then
        # Pull the Accuracy line if present
        acc_output=$(grep -E '^(Accuracy|Latency|Output throughput):' "$acc_log" | tr '\n' ' ')
        if [[ -n "$acc_output" ]]; then
            echo "[$name] PASS: $acc_output"
            SUMMARY+=("PASS  $name  $acc_output  $desc")
        else
            echo "[$name] FAIL: accuracy run finished but no Accuracy line found"
            tail -n 5 "$acc_log" | sed 's/^/         /'
            SUMMARY+=("FAIL  $name  no-accuracy-line  $desc")
        fi
    else
        echo "[$name] FAIL: accuracy run timed out or errored"
        tail -n 10 "$acc_log" | sed 's/^/         /'
        SUMMARY+=("FAIL  $name  acc-timeout-or-error  $desc")
    fi

    pkill -f sglang.launch_server >/dev/null 2>&1 || true
    sleep 3
    wait_gpu_clear
}

for name in "${ORDER[@]}"; do
    if [[ -z "${CFG_FLAGS[$name]:-}" ]]; then
        echo "[$name] UNKNOWN config, skipping"
        continue
    fi
    run_cfg "$name"
done

echo ""
echo "=============================================================="
echo "SUMMARY"
echo "=============================================================="
for line in "${SUMMARY[@]}"; do
    echo "$line"
done
