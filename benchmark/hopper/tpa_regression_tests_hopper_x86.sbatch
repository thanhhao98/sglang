#!/bin/bash
#SBATCH --job-name=tpa-regression-tests
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=viking-dvt-183
#SBATCH --time=02:00:00
#SBATCH --output=/home/zlinr/slurm-logs/tpa-regression-tests-%j.out
#SBATCH --error=/home/zlinr/slurm-logs/tpa-regression-tests-%j.err
#SBATCH --mem=0

set -euo pipefail

NODE_ROOT=/tmp/tpa-regression-tests/${SLURM_JOB_ID}
RESULT_ROOT=/home/zlinr/tpa-regression-tests/x86-hopper-r1
RUN_DIR=${RESULT_ROOT}/jobs/${SLURM_JOB_ID}
LOCAL_RESULT_ROOT=${NODE_ROOT}/results
REMOTE_REPO=/home/zlinr/htphan-sglang
IMAGE_TAR=/home/zlinr/sglang-tpa-phase1-hopper-x86-r1.tar
IMAGE_NAME=sglang-tpa-phase1-hopper-x86-r1:local
GPU_IDS=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "${NODE_ROOT}" "${RUN_DIR}" "${LOCAL_RESULT_ROOT}" "${RESULT_ROOT}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "$(timestamp) node=$(hostname) arch=$(uname -m)" | tee "${RUN_DIR}/run.meta.txt"
nvidia-smi -L | tee -a "${RUN_DIR}/run.meta.txt"
df -h /tmp /home/zlinr | tee -a "${RUN_DIR}/run.meta.txt"
docker --version | tee -a "${RUN_DIR}/run.meta.txt"

test -d "${REMOTE_REPO}"
test -f "${IMAGE_TAR}"

echo "$(timestamp) loading docker image..." | tee -a "${RUN_DIR}/run.meta.txt"
docker image rm "${IMAGE_NAME}" >/dev/null 2>&1 || true
docker load -i "${IMAGE_TAR}" 2>&1 | tee "${RUN_DIR}/docker-load.log"

run_suite() {
  local suite_name=$1
  local pythonpath=$2
  local command=$3
  local suite_dir="${RESULT_ROOT}/${suite_name}"
  local local_suite_dir="${LOCAL_RESULT_ROOT}/${suite_name}"
  local log_file="${suite_dir}/suite_${SLURM_JOB_ID}.log"
  local local_log_file="${local_suite_dir}/suite_${SLURM_JOB_ID}.log"
  local status_file="${suite_dir}/status.txt"
  local rc=0

  mkdir -p "${suite_dir}" "${local_suite_dir}"
  echo "$(timestamp) starting ${suite_name}" | tee -a "${RUN_DIR}/run.meta.txt"

  set +e
  docker run --rm \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES="${GPU_IDS}" \
    -e HOME=/tmp/container-home \
    -e XDG_CACHE_HOME=/tmp/container-home/.cache \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache \
    -e TRITON_CACHE_DIR=/tmp/triton-cache \
    -e PYTHONPATH="${pythonpath}" \
    -v "${REMOTE_REPO}:/sgl-workspace/sglang:ro" \
    -v "${LOCAL_RESULT_ROOT}:/results" \
    "${IMAGE_NAME}" \
    bash -lc "cd /sgl-workspace/sglang && ${command}" \
    > "${local_log_file}" 2>&1
  rc=$?
  set -e

  rsync -a "${local_suite_dir}/" "${suite_dir}/" >/dev/null 2>&1 || cp -a "${local_suite_dir}/." "${suite_dir}/"

  if [ "${rc}" -eq 0 ]; then
    echo "passed" > "${status_file}"
  else
    echo "failed:${rc}" > "${status_file}"
  fi

  echo "$(timestamp) finished ${suite_name} rc=${rc}" | tee -a "${RUN_DIR}/run.meta.txt"
}

run_suite \
  unit_discover \
  /sgl-workspace/sglang/python \
  "python -m unittest discover -s test/unit -p 'test_*.py'"

run_suite \
  server_args_full \
  /sgl-workspace/sglang/python:/sgl-workspace/sglang/test/registered/core \
  "python -m unittest discover -s test/registered/core -p 'test_server_args.py'"

run_suite \
  dcp_a2a_srt \
  /sgl-workspace/sglang/python \
  "python -m unittest discover -s test/srt -p 'test_dcp_a2a.py'"

run_suite \
  dcp_interleaved_storage_srt \
  /sgl-workspace/sglang/python \
  "python -m unittest discover -s test/srt -p 'test_dcp_interleaved_storage.py'"

python3 - "${RESULT_ROOT}" <<'PY' > "${RUN_DIR}/summary.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = {}
for suite_dir in sorted(root.iterdir()):
    if not suite_dir.is_dir() or suite_dir.name == "jobs":
        continue
    status_path = suite_dir / "status.txt"
    if status_path.exists():
        summary[suite_dir.name] = status_path.read_text().strip()
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "$(timestamp) regression test job complete" | tee -a "${RUN_DIR}/run.meta.txt"
