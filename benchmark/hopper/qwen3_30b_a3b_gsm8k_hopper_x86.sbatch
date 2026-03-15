#!/bin/bash
#SBATCH --job-name=qwen3-30b-a3b-gsm8k
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --nodelist=viking-prod-237
#SBATCH --time=08:00:00
#SBATCH --output=/home/zlinr/slurm-logs/qwen3-30b-a3b-gsm8k-%j.out
#SBATCH --error=/home/zlinr/slurm-logs/qwen3-30b-a3b-gsm8k-%j.err
#SBATCH --mem=0

set -euo pipefail

NODE_ROOT=/tmp/qwen3-30b-a3b-gsm8k/${SLURM_JOB_ID}
RESULT_ROOT=/home/zlinr/qwen3-30b-a3b-gsm8k/x86-h100-r1
RUN_DIR=${RESULT_ROOT}/jobs/${SLURM_JOB_ID}
LOCAL_RESULT_ROOT=${NODE_ROOT}/container-results
HF_CACHE=/home/zlinr/.cache/huggingface
COMMUNICATOR_PATCH_HOST=/home/zlinr/htphan-sglang/python/sglang/srt/layers/communicator.py
IMAGE_TAR=/home/zlinr/sglang-tpa-phase1-hopper-x86-r1.tar
IMAGE_NAME=sglang-tpa-phase1-hopper-x86-r1:local

MODEL=Qwen/Qwen3-30B-A3B
MODEL_SHORT=Qwen3-30B-A3B
MODEL_HOST_PATH=$(ls -d "${HF_CACHE}/models--Qwen--Qwen3-30B-A3B/snapshots/"* 2>/dev/null | head -n1)
MODEL_CONT_PATH=${MODEL_HOST_PATH/"${HF_CACHE}"/"/hf-cache"}
SERVER_PORT=8188
NUM_EXAMPLES=200
NUM_THREADS=32
NUM_SHOTS=5
MAX_TOKENS=1024
TP_SIZE=8
MAX_MODEL_LEN=131072
MEM_FRACTION_STATIC=0.8
BACKEND=fa3
NCCL_IFNAME=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $5; exit}' || echo eth0)

mkdir -p "${NODE_ROOT}" "${RESULT_ROOT}" "${RUN_DIR}" "${LOCAL_RESULT_ROOT}" "${HF_CACHE}"
test -n "${MODEL_HOST_PATH}"
test -f "${COMMUNICATOR_PATCH_HOST}"
test -f "${IMAGE_TAR}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

SERVER_CONTAINER=""
MODEL_ID=""
echo "$(timestamp) node=$(hostname) arch=$(uname -m)" | tee "${RUN_DIR}/run.meta.txt"
echo "$(timestamp) image tar=${IMAGE_TAR}" | tee -a "${RUN_DIR}/run.meta.txt"
echo "$(timestamp) result root=${RESULT_ROOT}" | tee -a "${RUN_DIR}/run.meta.txt"
echo "$(timestamp) model snapshot=${MODEL_HOST_PATH}" | tee -a "${RUN_DIR}/run.meta.txt"
nvidia-smi -L | tee -a "${RUN_DIR}/run.meta.txt"
df -h /tmp /home/zlinr | tee -a "${RUN_DIR}/run.meta.txt"
docker --version | tee -a "${RUN_DIR}/run.meta.txt"

echo "$(timestamp) loading docker image..." | tee -a "${RUN_DIR}/run.meta.txt"
docker image rm "${IMAGE_NAME}" >/dev/null 2>&1 || true
docker load -i "${IMAGE_TAR}" 2>&1 | tee "${RUN_DIR}/docker-load.log"

stop_server() {
  set +e
  if [ -n "${SERVER_CONTAINER}" ]; then
    docker rm -f "${SERVER_CONTAINER}" >/dev/null 2>&1 || true
    SERVER_CONTAINER=""
    sleep 10
  fi
}

trap 'stop_server' EXIT

sync_mode_artifacts() {
  local local_dir=$1
  local host_dir=$2
  mkdir -p "${host_dir}"
  if [ -d "${local_dir}" ]; then
    rsync -a "${local_dir}/" "${host_dir}/" >/dev/null 2>&1 || cp -a "${local_dir}/." "${host_dir}/"
  fi
}

wait_for_server() {
  local mode_name=$1
  local host_log_file=$2
  local waited=0
  while [ "${waited}" -lt 2400 ]; do
    if [ -n "${SERVER_CONTAINER}" ] && ! docker ps --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
      echo "$(timestamp) server container exited early for ${mode_name}" | tee -a "${RUN_DIR}/run.meta.txt"
      docker ps -a --filter "name=${SERVER_CONTAINER}" --format '{{.Names}} {{.Status}}' \
        > "${host_log_file}.docker-state" 2>&1 || true
      docker logs "${SERVER_CONTAINER}" > "${host_log_file}.docker.log" 2>&1 || true
      tail -n 200 "${host_log_file}" || true
      tail -n 200 "${host_log_file}.docker.log" || true
      return 1
    fi
    if curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1 && \
       curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "$(timestamp) server health timeout for ${mode_name}" | tee -a "${RUN_DIR}/run.meta.txt"
  tail -n 200 "${host_log_file}" || true
  return 1
}

resolve_model_id() {
  local models_file=$1
  curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/models" > "${models_file}"
  MODEL_ID=$(python3 - "${models_file}" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(data["data"][0]["id"])
PY
)
}

start_server() {
  local mode_name=$1
  local host_mode_dir=$2
  local local_mode_dir=$3
  shift 3
  local host_log_dir="${host_mode_dir}/logs"
  local local_log_dir="${local_mode_dir}/logs"
  local host_log_file="${host_log_dir}/server_${SLURM_JOB_ID}_${mode_name}.log"
  local log_dir_cont="/results/${MODEL_SHORT}_${mode_name}/logs"
  local log_file_cont="${log_dir_cont}/server_${SLURM_JOB_ID}_${mode_name}.log"

  mkdir -p "${host_log_dir}" "${local_log_dir}"
  stop_server
  SERVER_CONTAINER="sglang-${mode_name}-${SLURM_JOB_ID}"

  echo "$(timestamp) starting ${mode_name}" | tee -a "${RUN_DIR}/run.meta.txt"
  docker run -d \
    --name "${SERVER_CONTAINER}" \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --ipc=host \
    --network=host \
    -e NCCL_SOCKET_IFNAME="${NCCL_IFNAME}" \
    -e GLOO_SOCKET_IFNAME="${NCCL_IFNAME}" \
    -e NCCL_IB_DISABLE=1 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    -v "${HF_CACHE}:/hf-cache:ro" \
    -v "${COMMUNICATOR_PATCH_HOST}:/sgl-workspace/sglang/python/sglang/srt/layers/communicator.py:ro" \
    -v "${LOCAL_RESULT_ROOT}:/results" \
    "${IMAGE_NAME}" \
    bash -lc "python -m sglang.launch_server \
      --model-path '${MODEL_CONT_PATH}' \
      --served-model-name '${MODEL}' \
      --trust-remote-code \
      --reasoning-parser qwen3 \
      --host 0.0.0.0 \
      --port ${SERVER_PORT} \
      --tp-size ${TP_SIZE} \
      --attention-backend ${BACKEND} \
      --context-length ${MAX_MODEL_LEN} \
      --mem-fraction-static ${MEM_FRACTION_STATIC} \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      $* \
      > '${log_file_cont}' 2>&1"

  wait_for_server "${mode_name}" "${host_log_file}"
}

capture_one() {
  local outfile=$1
  python3 - <<PY > "${outfile}"
import json
from urllib import request

payload = {
    "model": "${MODEL_ID}",
    "messages": [{
        "role": "user",
        "content": "If a store sells pencils in packs of 3 for 6 dollars, how much do 5 packs cost? Answer briefly."
    }],
    "max_tokens": 128,
    "temperature": 0,
}

req = request.Request(
    "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions",
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
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

run_eval() {
  local mode_name=$1
  local host_mode_dir=$2
  local eval_log="${host_mode_dir}/${mode_name}.eval.log"

  docker run --rm \
    --network=host \
    -v "${LOCAL_RESULT_ROOT}:/results" \
    "${IMAGE_NAME}" \
    bash -lc "python -m sglang.test.run_eval \
      --host 127.0.0.1 \
      --port ${SERVER_PORT} \
      --model '${MODEL_ID}' \
      --eval-name gsm8k \
      --num-examples ${NUM_EXAMPLES} \
      --num-threads ${NUM_THREADS} \
      --num-shots ${NUM_SHOTS} \
      --max-tokens ${MAX_TOKENS} \
      --thinking-mode qwen3" \
    > "${eval_log}" 2>&1
}

run_mode() {
  local mode_name=$1
  shift
  local host_mode_dir="${RESULT_ROOT}/${mode_name}"
  local local_mode_dir="${LOCAL_RESULT_ROOT}/${MODEL_SHORT}_${mode_name}"
  local host_probe_file="${host_mode_dir}/probe.response.json"
  mkdir -p "${host_mode_dir}" "${local_mode_dir}" "${host_mode_dir}/logs"

  if start_server "${mode_name}" "${host_mode_dir}" "${local_mode_dir}" "$@"; then
    echo "healthy" > "${host_mode_dir}/server.status.txt"
    resolve_model_id "${host_mode_dir}/${mode_name}.models.json"
    capture_one "${host_probe_file}" || true
    run_eval "${mode_name}" "${host_mode_dir}"
  else
    echo "failed_startup" > "${host_mode_dir}/server.status.txt"
  fi

  if [ -n "${SERVER_CONTAINER}" ]; then
    docker ps -a --filter "name=${SERVER_CONTAINER}" --format '{{.Names}} {{.Status}}' \
      > "${host_mode_dir}/logs/server_${SLURM_JOB_ID}_${mode_name}.docker-state" 2>&1 || true
    docker logs "${SERVER_CONTAINER}" > "${host_mode_dir}/logs/server_${SLURM_JOB_ID}_${mode_name}.docker.log" 2>&1 || true
  fi
  stop_server
  sync_mode_artifacts "${local_mode_dir}" "${host_mode_dir}"
}

run_mode pure_tp
run_mode dcp2_a2a --dcp-size 2 --dcp-comm-backend a2a
run_mode tpa2_dcp4_a2a --attention-tensor-parallel-size 2 --dcp-size 4 --dcp-comm-backend a2a

python3 - "${RESULT_ROOT}" <<'PY' > "${RESULT_ROOT}/jobs/${SLURM_JOB_ID}/summary.json"
import json
import pathlib
import re
import sys

result_root = pathlib.Path(sys.argv[1])
summary = {}
for mode in ("pure_tp", "dcp2_a2a", "tpa2_dcp4_a2a"):
    mode_dir = result_root / mode
    eval_log = mode_dir / f"{mode}.eval.log"
    probe_file = mode_dir / "probe.response.json"
    if eval_log.exists():
        text = eval_log.read_text()
        score_match = re.findall(r"(?:Score|Accuracy):\s*([0-9.]+)", text)
        invalid_match = re.findall(r"Invalid:\s*([0-9.]+)", text)
    else:
        score_match = []
        invalid_match = []
        text = ""
    probe = json.loads(probe_file.read_text()) if probe_file.exists() else {}
    summary[mode] = {
        "score": float(score_match[-1]) if score_match else None,
        "invalid": float(invalid_match[-1]) if invalid_match else None,
        "probe_content": probe.get("content"),
        "probe_reasoning_content": probe.get("reasoning_content"),
        "probe_finish_reason": probe.get("finish_reason"),
        "server_status": (mode_dir / "server.status.txt").read_text().strip() if (mode_dir / "server.status.txt").exists() else None,
    }
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "$(timestamp) gsm8k complete" | tee -a "${RUN_DIR}/run.meta.txt"
