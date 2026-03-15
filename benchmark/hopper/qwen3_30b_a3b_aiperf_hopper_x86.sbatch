#!/bin/bash
#SBATCH --job-name=qwen3-30b-a3b-aiperf
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --nodelist=viking-dvt-183
#SBATCH --time=08:00:00
#SBATCH --output=/home/zlinr/slurm-logs/qwen3-30b-a3b-aiperf-%j.out
#SBATCH --error=/home/zlinr/slurm-logs/qwen3-30b-a3b-aiperf-%j.err
#SBATCH --mem=0

set -euo pipefail

NODE_ROOT=/tmp/qwen3-30b-a3b-aiperf/${SLURM_JOB_ID}
RESULT_ROOT=/home/zlinr/qwen3-30b-a3b-aiperf/x86-h100-r1
RUN_DIR=${RESULT_ROOT}/jobs/${SLURM_JOB_ID}
LOCAL_RESULT_ROOT=${NODE_ROOT}/container-results
HF_CACHE=/home/zlinr/.cache/huggingface
COMMUNICATOR_PATCH_HOST=/home/zlinr/htphan-sglang/python/sglang/srt/layers/communicator.py
AIPERF_VENV=${NODE_ROOT}/aiperf-venv
IMAGE_TAR=/home/zlinr/sglang-tpa-phase1-hopper-x86-r1.tar
IMAGE_NAME=sglang-tpa-phase1-hopper-x86-r1:local

MODEL=Qwen/Qwen3-30B-A3B
MODEL_SHORT=Qwen3-30B-A3B
MODEL_HOST_PATH=$(ls -d "${HF_CACHE}/models--Qwen--Qwen3-30B-A3B/snapshots/"* 2>/dev/null | head -n1)
MODEL_CONT_PATH=${MODEL_HOST_PATH/"${HF_CACHE}"/"/hf-cache"}
TP_SIZE=8
MAX_MODEL_LEN=131072
OUTPUT_TOKENS=1024
REQUESTS_PER_CONC=5
RANDOM_SEED=42
SERVER_PORT=8188
MEM_FRACTION_STATIC=0.8
BACKEND=fa3

CONTEXTS=(32768 65536 131072)
CONCURRENCIES=(1 4 16 32 64)

NCCL_IFNAME=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $5; exit}' || echo eth0)

mkdir -p "${NODE_ROOT}" "${RESULT_ROOT}" "${RUN_DIR}" "${LOCAL_RESULT_ROOT}" "${HF_CACHE}"
test -n "${MODEL_HOST_PATH}"
test -f "${COMMUNICATOR_PATCH_HOST}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

SERVER_CONTAINER=""
echo "$(timestamp) node=$(hostname) arch=$(uname -m)" | tee "${RUN_DIR}/run.meta.txt"
echo "$(timestamp) image tar=${IMAGE_TAR}" | tee -a "${RUN_DIR}/run.meta.txt"
echo "$(timestamp) result root=${RESULT_ROOT}" | tee -a "${RUN_DIR}/run.meta.txt"
nvidia-smi -L | tee -a "${RUN_DIR}/run.meta.txt"
df -h /tmp /home/zlinr | tee -a "${RUN_DIR}/run.meta.txt"
docker --version | tee -a "${RUN_DIR}/run.meta.txt"

test -f "${IMAGE_TAR}"

echo "$(timestamp) loading docker image..." | tee -a "${RUN_DIR}/run.meta.txt"
docker image rm "${IMAGE_NAME}" >/dev/null 2>&1 || true
docker load -i "${IMAGE_TAR}" 2>&1 | tee "${RUN_DIR}/docker-load.log"

echo "$(timestamp) creating AIPerf venv inside container..." | tee -a "${RUN_DIR}/run.meta.txt"
docker run --rm \
  -v "${NODE_ROOT}:${NODE_ROOT}" \
  "${IMAGE_NAME}" \
  bash -lc "if [ ! -x '${AIPERF_VENV}/bin/aiperf' ]; then python3 -m venv --system-site-packages '${AIPERF_VENV}' && '${AIPERF_VENV}/bin/pip' install --no-cache-dir aiperf; fi; '${AIPERF_VENV}/bin/aiperf' --version" \
  2>&1 | tee "${RUN_DIR}/aiperf-install.log"

echo "$(timestamp) using local model snapshot ${MODEL_HOST_PATH}" | tee -a "${RUN_DIR}/run.meta.txt"

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
  local local_mode_dir=$1
  local host_mode_dir=$2
  mkdir -p "${host_mode_dir}"
  if [ -d "${local_mode_dir}" ]; then
    rsync -a "${local_mode_dir}/" "${host_mode_dir}/" >/dev/null 2>&1 || cp -a "${local_mode_dir}/." "${host_mode_dir}/"
  fi
}

wait_for_server() {
  local mode_name=$1
  local log_file=$2
  local waited=0
  while [ "${waited}" -lt 3600 ]; do
    if [ -n "${SERVER_CONTAINER}" ] && ! docker ps --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
      echo "$(timestamp) server container exited early for ${mode_name}" | tee -a "${RUN_DIR}/run.meta.txt"
      docker ps -a --filter "name=${SERVER_CONTAINER}" --format '{{.Names}} {{.Status}}' \
        > "${log_file}.docker-state" 2>&1 || true
      docker logs "${SERVER_CONTAINER}" > "${log_file}.docker.log" 2>&1 || true
      tail -n 200 "${log_file}" || true
      tail -n 200 "${log_file}.docker.log" || true
      return 1
    fi
    if curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1 && \
       curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "$(timestamp) server health timeout for ${mode_name}" | tee -a "${RUN_DIR}/run.meta.txt"
  tail -n 200 "${log_file}" || true
  return 1
}

start_server() {
  local mode_name=$1
  local mode_dir=$2
  local local_mode_dir=$3
  shift 3
  local log_dir="${mode_dir}/logs"
  local local_log_dir="${local_mode_dir}/logs"
  local log_file="${local_log_dir}/server_${SLURM_JOB_ID}_${mode_name}.log"
  local log_dir_cont="/results/${MODEL_SHORT}_tp${TP_SIZE}_${mode_name}/logs"
  local log_file_cont="${log_dir_cont}/server_${SLURM_JOB_ID}_${mode_name}.log"
  mkdir -p "${log_dir}" "${local_log_dir}"
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
  wait_for_server "${mode_name}" "${log_file}"
}

run_probe() {
  local mode_name=$1
  local mode_dir=$2
  local probe_file="${mode_dir}/probe.response.json"
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
}

run_aiperf_point() {
  local mode_name=$1
  local mode_dir=$2
  local local_mode_dir=$3
  local ctx=$4
  local conc=$5
  local result_file="${mode_dir}/aiperf_ctx${ctx}_c${conc}.json"
  local log_file="${mode_dir}/logs/aiperf_ctx${ctx}_c${conc}_${SLURM_JOB_ID}.log"
  local local_export_prefix="${local_mode_dir}/aiperf_ctx${ctx}_c${conc}"
  local export_prefix_cont="/results/${MODEL_SHORT}_tp${TP_SIZE}_${mode_name}/aiperf_ctx${ctx}_c${conc}"
  local input_len=$((ctx - OUTPUT_TOKENS - 100))
  local request_count=$((conc * REQUESTS_PER_CONC))

  if [ -f "${result_file}" ]; then
    echo "$(timestamp) skipping existing result ${mode_name} ctx=${ctx} c=${conc}" | tee -a "${RUN_DIR}/run.meta.txt"
    return 0
  fi

  echo "$(timestamp) AIPerf ${mode_name} ctx=${ctx} c=${conc}" | tee -a "${RUN_DIR}/run.meta.txt"
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
  sync_mode_artifacts "${local_mode_dir}" "${mode_dir}"
  if [ ! -f "${result_file}" ]; then
    echo "$(timestamp) missing AIPerf result file for ${mode_name} ctx=${ctx} c=${conc}" | tee -a "${RUN_DIR}/run.meta.txt"
    tail -n 200 "${log_file}" || true
    return 1
  fi
  echo "${SLURM_JOB_ID}" > "${mode_dir}/aiperf_ctx${ctx}_c${conc}.source"
}

summarize_mode() {
  local mode_name=$1
  local mode_dir=$2
  python3 - "${mode_name}" "${mode_dir}" <<'PY' > "${mode_dir}/summary.json"
import json
import pathlib
import sys

mode_name = sys.argv[1]
mode_dir = pathlib.Path(sys.argv[2])
rows = {}
for path in sorted(mode_dir.glob("aiperf_ctx*_c*.json")):
    obj = json.loads(path.read_text())
    rows[path.stem] = {
        "request_throughput_avg": obj["request_throughput"]["avg"],
        "total_token_throughput_avg": obj["total_token_throughput"]["avg"],
        "ttft_p50": obj["time_to_first_token"]["p50"],
        "itl_p50": obj["inter_token_latency"]["p50"],
        "request_latency_p50": obj["request_latency"]["p50"],
        "output_seq_len_avg": obj["output_sequence_length"]["avg"],
        "input_seq_len_avg": obj["input_sequence_length"]["avg"],
    }
print(json.dumps({"mode": mode_name, "results": rows}, indent=2))
PY
}

run_mode() {
  local mode_name=$1
  local mode_args=$2
  local mode_dir="${RESULT_ROOT}/${MODEL_SHORT}_tp${TP_SIZE}_${mode_name}"
  local local_mode_dir="${LOCAL_RESULT_ROOT}/${MODEL_SHORT}_tp${TP_SIZE}_${mode_name}"
  mkdir -p "${mode_dir}/logs"
  mkdir -p "${local_mode_dir}/logs"
  if ! start_server "${mode_name}" "${mode_dir}" "${local_mode_dir}" "${mode_args}"; then
    sync_mode_artifacts "${local_mode_dir}" "${mode_dir}"
    echo "$(timestamp) aborting benchmark because ${mode_name} server did not become healthy" | tee -a "${RUN_DIR}/run.meta.txt"
    return 1
  fi
  sync_mode_artifacts "${local_mode_dir}" "${mode_dir}"
  run_probe "${mode_name}" "${mode_dir}"
  for ctx in "${CONTEXTS[@]}"; do
    for conc in "${CONCURRENCIES[@]}"; do
      run_aiperf_point "${mode_name}" "${mode_dir}" "${local_mode_dir}" "${ctx}" "${conc}"
    done
  done
  sync_mode_artifacts "${local_mode_dir}" "${mode_dir}"
  summarize_mode "${mode_name}" "${mode_dir}"
}

run_mode pure_tp ""
run_mode dcp2_a2a "--dcp-size 2 --dcp-comm-backend a2a"
run_mode tpa2_dcp4_a2a "--attention-tensor-parallel-size 2 --dcp-size 4 --dcp-comm-backend a2a"

stop_server

python3 - "${RESULT_ROOT}" <<'PY' > "${RUN_DIR}/summary.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
combined = {}
for mode_summary in root.glob("*/summary.json"):
    data = json.loads(mode_summary.read_text())
    combined[data["mode"]] = data["results"]
print(json.dumps(combined, indent=2))
PY

echo "$(timestamp) benchmark complete" | tee -a "${RUN_DIR}/run.meta.txt"
