#!/bin/bash
#SBATCH --job-name=build-hopper-x86-img
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=1u1g-gen-0101
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/build-hopper-x86-img-%j.out
#SBATCH --error=/tmp/build-hopper-x86-img-%j.err
#SBATCH --mem=0

set -euo pipefail

FRONTEND_HOST=zhaol@dlcluster
REMOTE_STORAGE=/home/zlinr
REMOTE_SOURCE_DIR=${SOURCE_DIR:-${REMOTE_STORAGE}/htphan-sglang-hopper-x86-r1-src}
REMOTE_RESULT_DIR=${REMOTE_STORAGE}/hopper-x86-image-build/${SLURM_JOB_ID}
REMOTE_IMAGE_TAR=${REMOTE_STORAGE}/sglang-tpa-phase1-hopper-x86-r1.tar
REMOTE_IMAGE_SQSH=${REMOTE_STORAGE}/sglang-tpa-phase1-hopper-x86-r1.sqsh

NODE_ROOT=/tmp/hopper-x86-image-build/${SLURM_JOB_ID}
SOURCE_DIR=${NODE_ROOT}/src
RESULT_DIR=${NODE_ROOT}/results
HF_CACHE=${NODE_ROOT}/hf-cache

IMAGE_NAME=sglang-tpa-phase1-hopper-x86-r1:local
IMAGE_TAR=${NODE_ROOT}/sglang-tpa-phase1-hopper-x86-r1.tar
IMAGE_SQSH=${NODE_ROOT}/sglang-tpa-phase1-hopper-x86-r1.sqsh
IMAGE_SQSH_TMP=${IMAGE_SQSH}.tmp

MODEL=Qwen/Qwen2.5-1.5B-Instruct
PORT=8188

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
RSYNC_RSH="ssh ${SSH_OPTS[*]}"

mkdir -p "${SOURCE_DIR}" "${RESULT_DIR}" "${HF_CACHE}"

copy_back() {
  set +e
  mkdir -p "${RESULT_DIR}"
  if [ -f "/tmp/build-hopper-x86-img-${SLURM_JOB_ID}.out" ]; then
    cp "/tmp/build-hopper-x86-img-${SLURM_JOB_ID}.out" "${RESULT_DIR}/slurm.out"
  fi
  if [ -f "/tmp/build-hopper-x86-img-${SLURM_JOB_ID}.err" ]; then
    cp "/tmp/build-hopper-x86-img-${SLURM_JOB_ID}.err" "${RESULT_DIR}/slurm.err"
  fi
  ssh "${SSH_OPTS[@]}" "${FRONTEND_HOST}" "mkdir -p '${REMOTE_RESULT_DIR}'"
  rsync -az -e "${RSYNC_RSH}" "${RESULT_DIR}/" "${FRONTEND_HOST}:${REMOTE_RESULT_DIR}/"
  if [ -f "${IMAGE_TAR}" ]; then
    scp "${SSH_OPTS[@]}" "${IMAGE_TAR}" "${FRONTEND_HOST}:${REMOTE_IMAGE_TAR}.tmp.${SLURM_JOB_ID}"
    ssh "${SSH_OPTS[@]}" "${FRONTEND_HOST}" "mv '${REMOTE_IMAGE_TAR}.tmp.${SLURM_JOB_ID}' '${REMOTE_IMAGE_TAR}'"
  fi
  if [ -f "${IMAGE_SQSH}" ]; then
    scp "${SSH_OPTS[@]}" "${IMAGE_SQSH}" "${FRONTEND_HOST}:${REMOTE_IMAGE_SQSH}.tmp.${SLURM_JOB_ID}"
    ssh "${SSH_OPTS[@]}" "${FRONTEND_HOST}" "mv '${REMOTE_IMAGE_SQSH}.tmp.${SLURM_JOB_ID}' '${REMOTE_IMAGE_SQSH}'"
  fi
}

trap copy_back EXIT

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "$(timestamp) Remote source: ${REMOTE_SOURCE_DIR}" | tee "${RESULT_DIR}/build.meta.txt"
echo "$(timestamp) Local source: ${SOURCE_DIR}" | tee -a "${RESULT_DIR}/build.meta.txt"
echo "$(timestamp) Image tag: ${IMAGE_NAME}" | tee -a "${RESULT_DIR}/build.meta.txt"
echo "$(timestamp) Local tar path: ${IMAGE_TAR}" | tee -a "${RESULT_DIR}/build.meta.txt"
echo "$(timestamp) Local sqsh path: ${IMAGE_SQSH}" | tee -a "${RESULT_DIR}/build.meta.txt"
echo "$(timestamp) Remote tar path: ${REMOTE_IMAGE_TAR}" | tee -a "${RESULT_DIR}/build.meta.txt"
echo "$(timestamp) Remote sqsh path: ${REMOTE_IMAGE_SQSH}" | tee -a "${RESULT_DIR}/build.meta.txt"
uname -m | tee -a "${RESULT_DIR}/build.meta.txt"
nvidia-smi -L | tee -a "${RESULT_DIR}/build.meta.txt"

echo "$(timestamp) Syncing source from frontend..." | tee -a "${RESULT_DIR}/build.meta.txt"
rsync -az --delete -e "${RSYNC_RSH}" "${FRONTEND_HOST}:${REMOTE_SOURCE_DIR}/" "${SOURCE_DIR}/" \
  2>&1 | tee "${RESULT_DIR}/source-sync.log"

test -f "${SOURCE_DIR}/python/pyproject.toml"
test -f "${SOURCE_DIR}/benchmark/hopper/Dockerfile.tpa_hopper_x86"

NCCL_IFNAME=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $5; exit}' || echo eth0)

echo "$(timestamp) Docker build starting..." | tee -a "${RESULT_DIR}/build.meta.txt"
DOCKER_BUILDKIT=1 docker build \
  --pull \
  --network=host \
  -t "${IMAGE_NAME}" \
  -f "${SOURCE_DIR}/benchmark/hopper/Dockerfile.tpa_hopper_x86" \
  "${SOURCE_DIR}" \
  2>&1 | tee "${RESULT_DIR}/docker-build.log"

echo "$(timestamp) Verifying image imports..." | tee -a "${RESULT_DIR}/build.meta.txt"
docker run --rm "${IMAGE_NAME}" python3 - <<'PY' 2>&1 | tee "${RESULT_DIR}/python-imports.log"
import sglang
from sglang.srt.layers import communicator
print("sglang:", sglang.__file__)
print("communicator:", communicator.__file__)
PY

echo "$(timestamp) Saving tar..." | tee -a "${RESULT_DIR}/build.meta.txt"
docker save "${IMAGE_NAME}" -o "${IMAGE_TAR}"
ls -lh "${IMAGE_TAR}" | tee "${RESULT_DIR}/image-tar.txt"

echo "$(timestamp) Converting to sqsh..." | tee -a "${RESULT_DIR}/build.meta.txt"
rm -f "${IMAGE_SQSH_TMP}"
enroot import -o "${IMAGE_SQSH_TMP}" "dockerd://${IMAGE_NAME}" 2>&1 | tee "${RESULT_DIR}/enroot-import.log"
mv "${IMAGE_SQSH_TMP}" "${IMAGE_SQSH}"
ls -lh "${IMAGE_SQSH}" | tee "${RESULT_DIR}/image-sqsh.txt"

SMOKE_SCRIPT_HOST="${RESULT_DIR}/smoke.sh"
cat > "${SMOKE_SCRIPT_HOST}" <<'INNER'
#!/bin/bash
set -euo pipefail

export HOME=/root
export HF_HOME=/root/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:-}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

PORT=8188
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SERVER_PID=""

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

cleanup() {
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
}

trap cleanup EXIT

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port ${PORT} \
  --tp-size 1 \
  --attention-backend fa3 \
  --context-length 8192 \
  --mem-fraction-static 0.5 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  > /results/smoke.server.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && \
     curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "$(timestamp) server exited early" | tee /results/smoke.status.txt
    tail -n 200 /results/smoke.server.log || true
    exit 1
  fi
  sleep 5
done

python3 - <<'PY' > /results/smoke.response.json
import json
from urllib import request

payload = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{
        "role": "user",
        "content": "If a store sells pencils in packs of 3 for $6, how much do 5 packs cost? Answer briefly."
    }],
    "max_tokens": 64,
    "temperature": 0,
}

req = request.Request(
    "http://127.0.0.1:8188/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with request.urlopen(req, timeout=300) as resp:
    body = json.loads(resp.read().decode("utf-8"))

print(json.dumps(body, indent=2, ensure_ascii=False))
PY

echo "ok" > /results/smoke.status.txt
INNER
chmod +x "${SMOKE_SCRIPT_HOST}"

echo "$(timestamp) Running 1-GPU smoke from the built image..." | tee -a "${RESULT_DIR}/build.meta.txt"
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  --ipc=host \
  --network=host \
  -e NCCL_SOCKET_IFNAME="${NCCL_IFNAME}" \
  -e GLOO_SOCKET_IFNAME="${NCCL_IFNAME}" \
  -e NCCL_IB_DISABLE=1 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${RESULT_DIR}:/results" \
  "${IMAGE_NAME}" \
  bash /results/smoke.sh 2>&1 | tee "${RESULT_DIR}/smoke.run.log"

echo "$(timestamp) Build + smoke complete." | tee -a "${RESULT_DIR}/build.meta.txt"
