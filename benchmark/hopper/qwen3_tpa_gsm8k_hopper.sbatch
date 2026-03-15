#!/bin/bash
#SBATCH --job-name=qwen3-tpa-gsm8k
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --nodelist=g242-p33-0093
#SBATCH --time=03:00:00
#SBATCH --output=/home/zlinr/slurm-logs/qwen3-tpa-gsm8k-%j.out
#SBATCH --error=/home/zlinr/slurm-logs/qwen3-tpa-gsm8k-%j.err
#SBATCH --mem=0

set -euo pipefail

STORAGE_PATH=/home/zlinr
RESULT_DIR_HOST=${STORAGE_PATH}/qwen3-tpa-gsm8k/${SLURM_JOB_ID}
RESULT_DIR_CONT=/root/qwen3-tpa-gsm8k/${SLURM_JOB_ID}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-${STORAGE_PATH}/sglang-tpa-phase1-hopper-x86-r1.sqsh}
CONTAINER_MOUNTS=${STORAGE_PATH}/.cache/huggingface:/root/.cache/huggingface,${STORAGE_PATH}/htphan-sglang:/sgl-workspace/sglang,${STORAGE_PATH}/qwen3-tpa-gsm8k:/root/qwen3-tpa-gsm8k
NCCL_IFNAME=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $5; exit}' || echo enP5p9s0)
mkdir -p "${RESULT_DIR_HOST}"

INNER_SCRIPT_HOST="${RESULT_DIR_HOST}/inner.sh"
INNER_SCRIPT_CONT="${RESULT_DIR_CONT}/inner.sh"

cat > "${INNER_SCRIPT_HOST}" <<'INNER'
#!/bin/bash
set -euo pipefail

export HOME=/root
export HF_HOME=/root/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONPATH=/sgl-workspace/sglang/python
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

RESULT_DIR="__RESULT_DIR__"
PORT=8188
MODEL="Qwen/Qwen3-0.6B"
NUM_EXAMPLES=200
NUM_THREADS=32
MAX_TOKENS=1024
NUM_SHOTS=5
SERVER_PID=""
MODEL_ID=""

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

kill_server() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        local waited=0
        while kill -0 "${SERVER_PID}" 2>/dev/null && [ ${waited} -lt 60 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -9 "${SERVER_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap kill_server EXIT

wait_for_server() {
    local log_file=$1
    for _ in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
                return 0
            fi
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "$(timestamp) server exited early"
            tail -n 200 "${log_file}" || true
            return 1
        fi
        sleep 5
    done
    echo "$(timestamp) server health timed out"
    tail -n 200 "${log_file}" || true
    return 1
}

resolve_model_id() {
    local models_file=$1
    curl -sf "http://127.0.0.1:${PORT}/v1/models" > "${models_file}"
    MODEL_ID=$(python - "${models_file}" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(data["data"][0]["id"])
PY
)
}

capture_one() {
    local outfile=$1
    python - "${PORT}" "${MODEL_ID}" "${outfile}" <<'PY'
import json
import sys
from pathlib import Path
from urllib import request

port = int(sys.argv[1])
model_id = sys.argv[2]
outfile = Path(sys.argv[3])

payload = {
    "model": model_id,
    "messages": [{
        "role": "user",
        "content": "If a store sells pencils in packs of 3 for $6, how much do 5 packs cost? Answer briefly."
    }],
    "max_tokens": 128,
    "temperature": 0,
}

req = request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with request.urlopen(req, timeout=300) as resp:
    body = json.loads(resp.read().decode("utf-8"))

message = body["choices"][0]["message"]
summary = {
    "content": message.get("content"),
    "reasoning_content": message.get("reasoning_content"),
    "finish_reason": body["choices"][0].get("finish_reason"),
    "full_body": body,
}
outfile.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

run_eval_mode() {
    local mode_name=$1
    shift
    local server_log="${RESULT_DIR}/${mode_name}.server.log"
    local models_file="${RESULT_DIR}/${mode_name}.models.json"
    local probe_file="${RESULT_DIR}/${mode_name}.probe.json"
    local eval_log="${RESULT_DIR}/${mode_name}.eval.log"

    echo "$(timestamp) === starting ${mode_name} ==="
    if [[ "${mode_name}" == tpa_* ]]; then
        mkdir -p "${RESULT_DIR}/debug-${mode_name}"
        SGLANG_DCP_DEBUG_DIR="${RESULT_DIR}/debug-${mode_name}" \
        SGLANG_DCP_DEBUG_LAYER=0 \
        python -m sglang.launch_server "$@" >"${server_log}" 2>&1 &
    else
        python -m sglang.launch_server "$@" >"${server_log}" 2>&1 &
    fi
    SERVER_PID=$!

    wait_for_server "${server_log}"
    resolve_model_id "${models_file}"
    capture_one "${probe_file}"

    python -m sglang.test.run_eval \
      --host 127.0.0.1 \
      --port ${PORT} \
      --model "${MODEL_ID}" \
      --eval-name gsm8k \
      --num-examples ${NUM_EXAMPLES} \
      --num-threads ${NUM_THREADS} \
      --num-shots ${NUM_SHOTS} \
      --max-tokens ${MAX_TOKENS} \
      --thinking-mode qwen3 \
      2>&1 | tee "${eval_log}"

    echo "$(timestamp) === finished ${mode_name} ==="
    kill_server
    sleep 5
}

mkdir -p "${RESULT_DIR}"
echo "$(timestamp) qwen3 tpa gsm8k on $(hostname)"
uname -m
nvidia-smi -L || true

BASE_ARGS=(
  --model-path "${MODEL}"
  --trust-remote-code
  --reasoning-parser qwen3
  --host 127.0.0.1
  --port ${PORT}
  --tp-size 2
  --attention-backend fa3
  --context-length 32768
  --mem-fraction-static 0.7
  --disable-cuda-graph
  --disable-piecewise-cuda-graph
)

run_eval_mode baseline "${BASE_ARGS[@]}"
run_eval_mode tpa_dcp_ag_rs "${BASE_ARGS[@]}" \
  --attention-tensor-parallel-size 1 \
  --dcp-size 2 \
  --dcp-comm-backend ag_rs
run_eval_mode tpa_dcp_a2a "${BASE_ARGS[@]}" \
  --attention-tensor-parallel-size 1 \
  --dcp-size 2 \
  --dcp-comm-backend a2a

python - "${RESULT_DIR}" <<'PY'
import json
import pathlib
import re
import sys

result_dir = pathlib.Path(sys.argv[1])
summary = {}
for mode in ("baseline", "tpa_dcp_ag_rs", "tpa_dcp_a2a"):
    text = (result_dir / f"{mode}.eval.log").read_text()
    score_match = re.findall(r"(?:Score|Accuracy):\s*([0-9.]+)", text)
    invalid_match = re.findall(r"Invalid:\s*([0-9.]+)", text)
    metrics_match = re.findall(r"\{.*\}", text)
    probe = json.loads((result_dir / f"{mode}.probe.json").read_text())
    summary[mode] = {
        "score": float(score_match[-1]) if score_match else None,
        "invalid": float(invalid_match[-1]) if invalid_match else None,
        "metrics_line": metrics_match[-1] if metrics_match else None,
        "probe_content": probe.get("content"),
        "probe_reasoning_content": probe.get("reasoning_content"),
        "probe_finish_reason": probe.get("finish_reason"),
    }
summary_path = result_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
INNER

sed -i "s|__RESULT_DIR__|${RESULT_DIR_CONT}|g" "${INNER_SCRIPT_HOST}"
chmod +x "${INNER_SCRIPT_HOST}"

srun --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir=/sgl-workspace/sglang \
  --no-container-mount-home \
  --export=ALL,NCCL_SOCKET_IFNAME=${NCCL_IFNAME},GLOO_SOCKET_IFNAME=${NCCL_IFNAME} \
  bash "${INNER_SCRIPT_CONT}"
