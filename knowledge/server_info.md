# Server Infrastructure for DCP A2A Benchmarking

## H100 Machine (colossus)

| Field | Value |
|-------|-------|
| SSH | `ssh colossus` |
| Code path | `/localhome/local-htphan/helix/sglang` |
| Branch | `htphan/helix_a2a_rebased_main_fe294904c9` |
| GPUs | 8x NVIDIA H100 80GB HBM3 |
| HF cache | `/raid/local-htphan/hf_cache/` |
| Large storage | `/raid/local-htphan/` (28T mount at `/raid`) |
| SHM | 256Gi |
| Root disk | `/dev/md0` 1.8T (99% full, only 25G free!) |
| Docker image | `sglang-dcp-a2a:local` (exists) |
| Status | Code synced, ready |

### Docker Run on H100

```bash
docker run -d --gpus all --shm-size 32g --network host \
  --ulimit memlock=-1 --init \
  --name sglang-bench \
  -v /raid/local-htphan/hf_cache:/root/.cache/huggingface \
  -v /localhome/local-htphan/helix/sglang/python/sglang:/sgl-workspace/sglang/python/sglang \
  -v /localhome/local-htphan/helix/sglang/test:/sgl-workspace/sglang/test \
  -v /localhome/local-htphan/helix/sglang/benchmark:/sgl-workspace/sglang/benchmark \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint sleep sglang-dcp-a2a:local infinity
```

### Sync Code to H100

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  /Users/htphan/workspace/DLAlgo/sglang/ \
  colossus:/localhome/local-htphan/helix/sglang/
```

### Latest Sync
- **Date:** 2026-03-13
- **Commit:** 8eb26c799 (htphan/helix_a2a_rebased_main_fe294904c9)

---

## B200 Machines (2 nodes)

| Field | Node 1 | Node 2 |
|-------|--------|--------|
| SSH | `ssh colossus_b200_1` | `ssh colossus_b200_2` |
| GPUs | 8x NVIDIA B200 183GB | 8x NVIDIA B200 183GB |
| CUDA | 13.0 | 13.0 |
| Driver | 580.126.09 | 580.95.05 |
| Large storage | `/raid` (28T, root-writable) | `/data` (28T, root-only!) |
| Home dir | `/localhome/local-htphan` (on /) | `/localhome/local-htphan` (on /, 1.4T free) |
| SHM | 1008G | 1008G |
| Docker | 29.1.3 | 29.3.0 |
| Python | 3.12.3 | 3.10.12 |
| GPU availability | GPUs 0-3 occupied by VLLM | All 8 GPUs free |

### Node 2 (all GPUs free but NVLink hardware issue!)

**WARNING:** B200 Node 2 has **uncorrectable NVLink errors** - TP8 crashes with `cudaErrorNvlinkUncorrectable`. TP1 works fine. Multi-GPU workloads (TP8, DCP8) cannot run on this node until hardware is fixed.

**Code path:** `/localhome/local-htphan/sglang_bench/sglang`
**HF cache:** `/localhome/local-htphan/sglang_bench/hf_cache`

```bash
# Storage uses home dir on / (1.4T free, sufficient)
# /data is root-only, not writable by local-htphan

# Sync code (from local machine):
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  /Users/htphan/workspace/DLAlgo/sglang/ \
  colossus_b200_2:/localhome/local-htphan/sglang_bench/sglang/
```

### Node 1 (GPUs 4-7 available, GPUs 0-3 occupied)

**Code path:** TBD
**Large storage:** `/raid/local-htphan/` (writable)

### Download Model on B200

```bash
# Option 1: Download via huggingface_hub
HF_HOME=/localhome/local-htphan/sglang_bench/hf_cache \
python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-V2-Lite')"

# Option 2: rsync from H100 (faster if on same network)
rsync -avz colossus:/raid/local-htphan/hf_cache/hub/models--deepseek-ai--DeepSeek-V2-Lite \
  /localhome/local-htphan/sglang_bench/hf_cache/hub/
```

### Latest B200 Sync
- **Date:** 2026-03-13 (syncing)
- **Node 2 code path:** `/localhome/local-htphan/sglang_bench/sglang`

---

## Docker Build

```bash
# Build from sglang repo root
cd /path/to/sglang
docker build -t sglang-dcp-a2a:local --build-arg BRANCH_TYPE=local -f docker/Dockerfile .

# Run container for benchmarking:
docker run -d --gpus all --shm-size 32g --network host \
  --ulimit memlock=-1 --init \
  --name sglang-bench \
  -v /raid/local-htphan/hf_cache/:/root/.cache/huggingface \
  -v /localhome/local-htphan/helix/sglang:/sgl-workspace/sglang/python/sglang \
  -v /path/to/sglang/test:/sgl-workspace/sglang/test \
  -v /path/to/sglang/benchmark:/sgl-workspace/sglang/benchmark \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint sleep sglang-dcp-main:2b47bd3 infinity
```

### Docker Version Tracking

| Machine | Image Tag | Build Date | Base Commit |
|---------|-----------|------------|-------------|
| H100 | `sglang-dcp-a2a:local` | exists | TODO |
| B200-2 | TODO (needs build) | TODO | TODO |

---

## Important Notes

- **GPU availability is time-dependent** - always check before running benchmarks:
  ```bash
  ssh <host> 'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader'
  ```
  If GPUs are occupied, skip and try later or pick a different machine.
- H100 root disk is 99% full - use `/raid` for all large files
- B200 node 2 `/data` is root-only - use home dir instead (1.4T free)
- B200 node 1 GPUs 0-3 occupied by VLLM workers - use GPUs 4-7 or node 2
- DeepSeek-V2-Lite is ~15.7B params, ~30GB on disk
- B200 has 183GB VRAM vs H100 80GB - can potentially use higher mem-fraction-static
- B200 CUDA 13.0, SM100 (Blackwell) - **FA3 NOT supported** (requires SM<=90). Use flashinfer only.
- B200 Node 2 has NVLink hardware errors - only TP1 works, multi-GPU crashes
