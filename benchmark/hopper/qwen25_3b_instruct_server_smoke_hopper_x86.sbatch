#!/bin/bash
#SBATCH --job-name=qwen25-3b-smoke
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --nodelist=viking-dvt-183
#SBATCH --time=01:00:00
#SBATCH --output=/home/zlinr/slurm-logs/qwen25-3b-smoke-%j.out
#SBATCH --error=/home/zlinr/slurm-logs/qwen25-3b-smoke-%j.err
#SBATCH --mem=0

set -euo pipefail

NODE_ROOT=/tmp/qwen25-3b-smoke/${SLURM_JOB_ID}
RESULT_ROOT=${RESULT_ROOT:-/home/zlinr/qwen25-3b-smoke/x86-h100-r1}
RUN_DIR=${RESULT_ROOT}/jobs/${SLURM_JOB_ID}
LOCAL_RESULT_ROOT=${NODE_ROOT}/container-results
HF_CACHE=/home/zlinr/.cache/huggingface
QWEN2_PATCH_HOST=/home/zlinr/htphan-sglang/python/sglang/srt/models/qwen2.py
MODEL_RUNNER_PATCH_HOST=/home/zlinr/htphan-sglang/python/sglang/srt/model_executor/model_runner.py
IMAGE_TAR=/home/zlinr/sglang-tpa-phase1-hopper-x86-r1.tar
IMAGE_NAME=sglang-tpa-phase1-hopper-x86-r1:local
SERVER_PORT=${SERVER_PORT:-8188}
TP_SIZE=8
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
BACKEND=${BACKEND:-fa3}
DISABLE_FUSED_SET_KV_BUFFER=${DISABLE_FUSED_SET_KV_BUFFER:-1}
MODEL=Qwen/Qwen2.5-3B-Instruct
MODEL_SHORT=Qwen2.5-3B-Instruct
MODEL_HOST_PATH=$(ls -d "${HF_CACHE}/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"* 2>/dev/null | head -n1)
MODEL_CONT_PATH=${MODEL_HOST_PATH/"${HF_CACHE}"/"/hf-cache"}
NCCL_IFNAME=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $5; exit}' || echo eth0)

mkdir -p "${NODE_ROOT}" "${RESULT_ROOT}" "${RUN_DIR}" "${LOCAL_RESULT_ROOT}" "${HF_CACHE}" /home/zlinr/slurm-logs
test -n "${MODEL_HOST_PATH}"
test -f "${QWEN2_PATCH_HOST}"
test -f "${MODEL_RUNNER_PATCH_HOST}"
test -f "${IMAGE_TAR}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "$(timestamp) node=$(hostname) arch=$(uname -m)" | tee "${RUN_DIR}/run.meta.txt"
echo "$(timestamp) model=${MODEL} model_path=${MODEL_HOST_PATH}" | tee -a "${RUN_DIR}/run.meta.txt"
echo "$(timestamp) tp=${TP_SIZE} max_model_len=${MAX_MODEL_LEN} backend=${BACKEND} disable_fused_set_kv_buffer=${DISABLE_FUSED_SET_KV_BUFFER}" | tee -a "${RUN_DIR}/run.meta.txt"
nvidia-smi -L | tee -a "${RUN_DIR}/run.meta.txt"
df -h /tmp /home/zlinr | tee -a "${RUN_DIR}/run.meta.txt"
docker --version | tee -a "${RUN_DIR}/run.meta.txt"

echo "$(timestamp) loading docker image..." | tee -a "${RUN_DIR}/run.meta.txt"
docker image rm "${IMAGE_NAME}" >/dev/null 2>&1 || true
docker load -i "${IMAGE_TAR}" 2>&1 | tee "${RUN_DIR}/docker-load.log"

cleanup_container() {
  local name=$1
  docker rm -f "${name}" >/dev/null 2>&1 || true
}

sync_case_artifacts() {
  local local_case_dir=$1
  local host_case_dir=$2
  mkdir -p "${host_case_dir}"
  if [ -d "${local_case_dir}" ]; then
    rsync -a "${local_case_dir}/" "${host_case_dir}/" >/dev/null 2>&1 || cp -a "${local_case_dir}/." "${host_case_dir}/"
  fi
}

run_case() {
  local case_name=$1
  local extra_args=$2
  local mode_dir="${RESULT_ROOT}/${case_name}"
  local local_mode_dir="${LOCAL_RESULT_ROOT}/${case_name}"
  local log_dir="${mode_dir}/logs"
  local local_log_dir="${local_mode_dir}/logs"
  local docker_log="${log_dir}/server_${SLURM_JOB_ID}.docker.log"
  local probe_file="${mode_dir}/probe.response.json"
  local container_name="smoke-${case_name}-${SLURM_JOB_ID}"
  local log_dir_cont="/results/${case_name}/logs"
  local server_log_cont="${log_dir_cont}/server_${SLURM_JOB_ID}.log"

  mkdir -p "${log_dir}" "${local_log_dir}"
  cleanup_container "${container_name}"

  echo "$(timestamp) starting ${case_name}" | tee -a "${RUN_DIR}/run.meta.txt"
  docker run -d \
    --name "${container_name}" \
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
    -e SGLANG_DISABLE_FUSED_SET_KV_BUFFER="${DISABLE_FUSED_SET_KV_BUFFER}" \
    -v "${HF_CACHE}:/hf-cache:ro" \
    -v "${QWEN2_PATCH_HOST}:/sgl-workspace/sglang/python/sglang/srt/models/qwen2.py:ro" \
    -v "${MODEL_RUNNER_PATCH_HOST}:/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py:ro" \
    -v "${LOCAL_RESULT_ROOT}:/results" \
    "${IMAGE_NAME}" \
    bash -lc "python -m sglang.launch_server \
      --model-path '${MODEL_CONT_PATH}' \
      --served-model-name '${MODEL}' \
      --trust-remote-code \
      --host 0.0.0.0 \
      --port ${SERVER_PORT} \
      --tp-size ${TP_SIZE} \
      --attention-backend ${BACKEND} \
      --context-length ${MAX_MODEL_LEN} \
      --mem-fraction-static ${MEM_FRACTION_STATIC} \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      ${extra_args} \
      > '${server_log_cont}' 2>&1"

  local healthy=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1 && \
       curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
      healthy=1
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${container_name}"; then
      break
    fi
    sleep 5
  done

  docker ps -a --filter "name=${container_name}" --format '{{.Names}} {{.Status}}' \
    > "${mode_dir}/container.state.txt" 2>&1 || true
  docker logs "${container_name}" > "${docker_log}" 2>&1 || true
  sync_case_artifacts "${local_mode_dir}" "${mode_dir}"

  if [ "${healthy}" -eq 1 ]; then
    python3 - <<PY > "${probe_file}"
import json
from urllib import request

payload = {
    "model": "${MODEL}",
    "prompt": "If a store sells pencils in packs of 3 for 6 dollars, how much do 5 packs cost? Answer briefly.",
    "max_tokens": 32,
    "temperature": 0,
}

req = request.Request(
    "http://127.0.0.1:${SERVER_PORT}/v1/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with request.urlopen(req, timeout=300) as resp:
    body = json.loads(resp.read().decode("utf-8"))

print(json.dumps(body, indent=2, ensure_ascii=False))
PY
    echo "healthy" > "${mode_dir}/status.txt"
  else
    echo "failed" > "${mode_dir}/status.txt"
  fi

  cleanup_container "${container_name}"
  sync_case_artifacts "${local_mode_dir}" "${mode_dir}"
}

run_case qwen25_3b_pure_tp ""
run_case qwen25_3b_dcp2_a2a "--dcp-size 2 --dcp-comm-backend a2a"
run_case qwen25_3b_tpa4_dcp2_a2a "--attention-tensor-parallel-size 4 --dcp-size 2 --dcp-comm-backend a2a"

echo "$(timestamp) smoke complete" | tee -a "${RUN_DIR}/run.meta.txt"
