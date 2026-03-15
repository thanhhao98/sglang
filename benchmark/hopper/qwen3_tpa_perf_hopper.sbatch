#!/bin/bash
#SBATCH --job-name=qwen3-tpa-perf
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --nodelist=g242-p33-0093
#SBATCH --time=02:00:00
#SBATCH --output=/home/zlinr/slurm-logs/qwen3-tpa-perf-%j.out
#SBATCH --error=/home/zlinr/slurm-logs/qwen3-tpa-perf-%j.err
#SBATCH --mem=0

set -euo pipefail

STORAGE_PATH=/home/zlinr
RESULT_DIR_HOST=${STORAGE_PATH}/qwen3-tpa-perf/${SLURM_JOB_ID}
RESULT_DIR_CONT=/root/qwen3-tpa-perf/${SLURM_JOB_ID}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-${STORAGE_PATH}/sglang-tpa-phase1-hopper-x86-r1.sqsh}
CONTAINER_MOUNTS=${STORAGE_PATH}/.cache/huggingface:/root/.cache/huggingface,${STORAGE_PATH}/htphan-sglang:/sgl-workspace/sglang,${STORAGE_PATH}/qwen3-tpa-perf:/root/qwen3-tpa-perf
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
MODEL="Qwen/Qwen3-0.6B"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_mode() {
    local mode_name=$1
    shift
    local log_file="${RESULT_DIR}/${mode_name}.bench.log"
    local result_file="${RESULT_DIR}/${mode_name}.jsonl"

    rm -f "${result_file}"
    echo "$(timestamp) === starting ${mode_name} ==="
    python -m sglang.bench_one_batch \
      --model-path "${MODEL}" \
      --trust-remote-code \
      --tp-size 2 \
      --attention-backend fa3 \
      --context-length 32768 \
      --mem-fraction-static 0.7 \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      --batch-size 1 16 \
      --input-len 16384 32768 \
      --output-len 128 \
      --run-name "${mode_name}" \
      --result-filename "${result_file}" \
      "$@" 2>&1 | tee "${log_file}"
    echo "$(timestamp) === finished ${mode_name} ==="
}

mkdir -p "${RESULT_DIR}"
echo "$(timestamp) qwen3 tpa perf on $(hostname)"
uname -m
nvidia-smi -L || true

run_mode baseline
run_mode tpa_dcp_ag_rs \
  --attention-tensor-parallel-size 1 \
  --dcp-size 2 \
  --dcp-comm-backend ag_rs
run_mode tpa_dcp_a2a \
  --attention-tensor-parallel-size 1 \
  --dcp-size 2 \
  --dcp-comm-backend a2a

python - "${RESULT_DIR}" <<'PY'
import json
import pathlib
import sys

result_dir = pathlib.Path(sys.argv[1])
modes = ("baseline", "tpa_dcp_ag_rs", "tpa_dcp_a2a")

loaded = {}
for mode in modes:
    rows = []
    with open(result_dir / f"{mode}.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    loaded[mode] = {
        f"bs{row['batch_size']}_il{row['input_len']}_ol{row['output_len']}": row
        for row in rows
    }

keys = sorted(loaded["baseline"].keys())
summary = {"cases": {}}
for key in keys:
    case = {}
    base = loaded["baseline"][key]
    for mode in modes:
        row = loaded[mode][key]
        case[mode] = {
            "prefill_latency": row.get("prefill_latency"),
            "prefill_throughput": row.get("prefill_throughput"),
            "median_decode_latency": row.get("median_decode_latency"),
            "median_decode_throughput": row.get("median_decode_throughput"),
            "total_latency": row.get("total_latency"),
            "overall_throughput": row.get("overall_throughput"),
        }
    for mode in modes[1:]:
        case[mode]["decode_latency_speedup_vs_baseline"] = (
            base["median_decode_latency"] / case[mode]["median_decode_latency"]
            if case[mode]["median_decode_latency"]
            else None
        )
        case[mode]["decode_throughput_gain_vs_baseline"] = (
            case[mode]["median_decode_throughput"] / base["median_decode_throughput"]
            if base["median_decode_throughput"]
            else None
        )
    summary["cases"][key] = case

summary_path = result_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
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
