#!/bin/bash
#SBATCH --job-name=qwen3-30b-a3b-tpa-repro
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --nodelist=viking-dvt-183
#SBATCH --time=04:00:00
#SBATCH --output=/home/zlinr/slurm-logs/qwen3-30b-a3b-tpa-repro-%j.out
#SBATCH --error=/home/zlinr/slurm-logs/qwen3-30b-a3b-tpa-repro-%j.err
#SBATCH --mem=0

set -euo pipefail

NODE_ROOT=/tmp/qwen3-30b-a3b-tpa-repro/${SLURM_JOB_ID}
RESULT_ROOT=/home/zlinr/qwen3-30b-a3b-tpa-repro/x86-h100-r1
RUN_DIR=${RESULT_ROOT}/jobs/${SLURM_JOB_ID}
LOCAL_RESULT_ROOT=${NODE_ROOT}/container-results
HF_CACHE=/home/zlinr/.cache/huggingface
COMMUNICATOR_PATCH_HOST=/home/zlinr/htphan-sglang/python/sglang/srt/layers/communicator.py
IMAGE_TAR=/home/zlinr/sglang-tpa-phase1-hopper-x86-r1.tar
IMAGE_NAME=sglang-tpa-phase1-hopper-x86-r1:local
AIPERF_VENV=${NODE_ROOT}/aiperf-venv

MODEL=Qwen/Qwen3-30B-A3B
MODEL_SHORT=Qwen3-30B-A3B
MODEL_HOST_PATH=$(ls -d "${HF_CACHE}/models--Qwen--Qwen3-30B-A3B/snapshots/"* 2>/dev/null | head -n1)
MODEL_CONT_PATH=${MODEL_HOST_PATH/"${HF_CACHE}"/"/hf-cache"}
TP_SIZE=8
ATTN_TP_SIZE=2
DCP_SIZE=4
SERVER_PORT=8188
MAX_MODEL_LEN=131072
MEM_FRACTION_STATIC=0.8
BACKEND=fa3
OUTPUT_TOKENS=1024
REQUESTS_PER_CONC=5
RANDOM_SEED=42
POINTS=("65536:32" "65536:64" "131072:1")

NCCL_IFNAME=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $5; exit}' || echo eth0)

mkdir -p "${NODE_ROOT}" "${RESULT_ROOT}" "${RUN_DIR}" "${LOCAL_RESULT_ROOT}" "${HF_CACHE}"
test -n "${MODEL_HOST_PATH}"
test -f "${COMMUNICATOR_PATCH_HOST}"
test -f "${IMAGE_TAR}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

SERVER_CONTAINER=""
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

echo "$(timestamp) creating AIPerf venv inside container..." | tee -a "${RUN_DIR}/run.meta.txt"
docker run --rm \
  -v "${NODE_ROOT}:${NODE_ROOT}" \
  "${IMAGE_NAME}" \
  bash -lc "if [ ! -x '${AIPERF_VENV}/bin/aiperf' ]; then python3 -m venv --system-site-packages '${AIPERF_VENV}' && '${AIPERF_VENV}/bin/pip' install --no-cache-dir aiperf; fi; '${AIPERF_VENV}/bin/aiperf' --version" \
  2>&1 | tee "${RUN_DIR}/aiperf-install.log"

stop_server() {
  set +e
  if [ -n "${SERVER_CONTAINER}" ]; then
    docker rm -f "${SERVER_CONTAINER}" >/dev/null 2>&1 || true
    SERVER_CONTAINER=""
    sleep 10
  fi
}

trap 'stop_server' EXIT

sync_point_artifacts() {
  local local_dir=$1
  local host_dir=$2
  mkdir -p "${host_dir}"
  if [ -d "${local_dir}" ]; then
    rsync -a "${local_dir}/" "${host_dir}/" >/dev/null 2>&1 || cp -a "${local_dir}/." "${host_dir}/"
  fi
}

wait_for_server() {
  local point_name=$1
  local host_log_file=$2
  local waited=0
  while [ "${waited}" -lt 1800 ]; do
    if [ -n "${SERVER_CONTAINER}" ] && ! docker ps --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
      echo "$(timestamp) server container exited early for ${point_name}" | tee -a "${RUN_DIR}/run.meta.txt"
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
  echo "$(timestamp) server health timeout for ${point_name}" | tee -a "${RUN_DIR}/run.meta.txt"
  tail -n 200 "${host_log_file}" || true
  docker logs "${SERVER_CONTAINER}" > "${host_log_file}.docker.log" 2>&1 || true
  tail -n 200 "${host_log_file}.docker.log" || true
  return 1
}

start_server() {
  local point_name=$1
  local host_point_dir=$2
  local local_point_dir=$3
  local host_log_dir="${host_point_dir}/logs"
  local local_log_dir="${local_point_dir}/logs"
  local host_log_file="${host_log_dir}/server_${SLURM_JOB_ID}_${point_name}.log"
  local log_dir_cont="/results/${point_name}/logs"
  local log_file_cont="${log_dir_cont}/server_${SLURM_JOB_ID}_${point_name}.log"

  mkdir -p "${host_log_dir}" "${local_log_dir}"
  stop_server
  SERVER_CONTAINER="sglang-${point_name}-${SLURM_JOB_ID}"

  echo "$(timestamp) starting ${point_name}" | tee -a "${RUN_DIR}/run.meta.txt"
  docker run -d \
    --name "${SERVER_CONTAINER}" \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --ipc=host \
    --network=host \
    -e NCCL_SOCKET_IFNAME="${NCCL_IFNAME}" \
    -e GLOO_SOCKET_IFNAME="${NCCL_IFNAME}" \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_ASYNC_ERROR_HANDLING=1 \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -e CUDA_LAUNCH_BLOCKING=1 \
    -e PYTHONFAULTHANDLER=1 \
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
      --host 0.0.0.0 \
      --port ${SERVER_PORT} \
      --tp-size ${TP_SIZE} \
      --attention-tensor-parallel-size ${ATTN_TP_SIZE} \
      --dcp-size ${DCP_SIZE} \
      --dcp-comm-backend a2a \
      --attention-backend ${BACKEND} \
      --context-length ${MAX_MODEL_LEN} \
      --mem-fraction-static ${MEM_FRACTION_STATIC} \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      > '${log_file_cont}' 2>&1"

  wait_for_server "${point_name}" "${host_log_file}"
}

run_probe() {
  local point_name=$1
  local host_point_dir=$2
  local outfile="${host_point_dir}/probe.response.json"
  python3 - <<PY > "${outfile}"
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
}

run_aiperf_point() {
  local point_name=$1
  local host_point_dir=$2
  local local_point_dir=$3
  local ctx=$4
  local conc=$5
  local result_file="${host_point_dir}/aiperf_ctx${ctx}_c${conc}.json"
  local log_file="${host_point_dir}/logs/aiperf_ctx${ctx}_c${conc}_${SLURM_JOB_ID}.log"
  local export_prefix_cont="/results/${point_name}/aiperf_ctx${ctx}_c${conc}"
  local input_len=$((ctx - OUTPUT_TOKENS - 100))
  local request_count=$((conc * REQUESTS_PER_CONC))

  echo "$(timestamp) AIPerf ${point_name} ctx=${ctx} c=${conc}" | tee -a "${RUN_DIR}/run.meta.txt"
  set +e
  docker run --rm \
    --network=host \
    -v "${NODE_ROOT}:${NODE_ROOT}" \
    -v "${LOCAL_RESULT_ROOT}:/results" \
    "${IMAGE_NAME}" \
    bash -lc "'${AIPERF_VENV}/bin/aiperf' profile \
      --ui-type none \
      --model '${MODEL}' \
      --url 'http://127.0.0.1:${SERVER_PORT}' \
      --endpoint-type completions \
      --streaming \
      --random-seed '${RANDOM_SEED}' \
      --prompt-input-tokens-mean '${input_len}' \
      --prompt-input-tokens-stddev 0 \
      --prompt-output-tokens-mean '${OUTPUT_TOKENS}' \
      --num-dataset-entries 1 \
      --dataset-sampling-strategy sequential \
      --concurrency '${conc}' \
      --request-count '${request_count}' \
      --warmup-request-count '${conc}' \
      --warmup-concurrency '${conc}' \
      --extra-inputs ignore_eos:true \
      --profile-export-prefix '${export_prefix_cont}'" \
    > "${log_file}" 2>&1
  local rc=$?
  set -e

  sync_point_artifacts "${local_point_dir}" "${host_point_dir}"

  if [ -f "${result_file}" ]; then
    echo "result_present" > "${host_point_dir}/result.status.txt"
  else
    echo "result_missing" > "${host_point_dir}/result.status.txt"
  fi
  echo "${rc}" > "${host_point_dir}/aiperf.exit_code.txt"
  return 0
}

for point in "${POINTS[@]}"; do
  ctx=${point%%:*}
  conc=${point##*:}
  point_name="ctx${ctx}_c${conc}"
  host_point_dir="${RESULT_ROOT}/${point_name}"
  local_point_dir="${LOCAL_RESULT_ROOT}/${point_name}"
  mkdir -p "${host_point_dir}" "${local_point_dir}" "${host_point_dir}/logs"

  if start_server "${point_name}" "${host_point_dir}" "${local_point_dir}"; then
    echo "healthy" > "${host_point_dir}/server.status.txt"
    run_probe "${point_name}" "${host_point_dir}" || true
    run_aiperf_point "${point_name}" "${host_point_dir}" "${local_point_dir}" "${ctx}" "${conc}"
  else
    echo "failed_startup" > "${host_point_dir}/server.status.txt"
  fi

  if [ -n "${SERVER_CONTAINER}" ]; then
    docker ps -a --filter "name=${SERVER_CONTAINER}" --format '{{.Names}} {{.Status}}' \
      > "${host_point_dir}/logs/server_${SLURM_JOB_ID}_${point_name}.docker-state" 2>&1 || true
    docker logs "${SERVER_CONTAINER}" > "${host_point_dir}/logs/server_${SLURM_JOB_ID}_${point_name}.docker.log" 2>&1 || true
  fi
  stop_server
  sync_point_artifacts "${local_point_dir}" "${host_point_dir}"
done

echo "$(timestamp) repro complete" | tee -a "${RUN_DIR}/run.meta.txt"
