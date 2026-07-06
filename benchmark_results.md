# DCP Accuracy & Benchmark Results

**Hardware:** 8x NVIDIA H100 80GB HBM3
**Model:** DeepSeek-V2-Lite (15.7B, MLA)

---

## DCP Accuracy Test Matrix

**24/24 passed** across all backend, comm, CUDA graph, and request type combinations.

| # | Backend | DCP Comm | CUDA Graph | prefill_only | decode_heavy | mixed |
|---|---------|----------|------------|:------------:|:------------:|:-----:|
| 1-3 | FA3 | A2A | off | PASS | PASS | PASS |
| 4-6 | FA3 | AG+RS | off | PASS | PASS | PASS |
| 7-9 | FA3 | A2A | on | PASS | PASS | PASS |
| 10-12 | FA3 | AG+RS | on | PASS | PASS | PASS |
| 13-15 | FlashInfer | AG+RS | on | PASS | PASS | PASS |
| 16-18 | FlashInfer | AG+RS | off | PASS | PASS | PASS |
| 19-21 | FlashInfer | A2A | off | PASS | PASS | PASS |
| 22-24 | FlashInfer | A2A | on | PASS | PASS | PASS |

---

## How to Run the Accuracy Test

### Prerequisites

- 4+ GPUs (test uses TP=4, DCP=4)
- Docker image `sglang-dcp-a2a:local` (see `helix_14194_a2a.md` Section 3.1 for build instructions)
- HuggingFace cache with `deepseek-ai/DeepSeek-V2-Lite` downloaded

### Quick Run (Docker)

```bash
# 1. Start a container (adjust --gpus for your available devices)
docker run -d --gpus '"device=4,5,6,7"' \
  --name sglang-accuracy-test \
  --shm-size 32g --network host --ulimit memlock=-1 --init \
  -v /path/to/hf_cache:/root/.cache/huggingface \
  -v /path/to/sglang/python/sglang:/sgl-workspace/sglang/python/sglang \
  -v /path/to/sglang/test:/sgl-workspace/sglang/test \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --entrypoint sleep sglang-dcp-a2a:local infinity

# 2. Run the test matrix
docker exec sglang-accuracy-test bash -c \
  'CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /sgl-workspace/sglang/test/srt/test_dcp_accuracy_matrix.py'

# 3. Clean up
docker rm -f sglang-accuracy-test
```

### What the Test Does

The script (`test/srt/test_dcp_accuracy_matrix.py`) iterates over server configurations:
- **Attention backends:** FA3 (FlashAttention v3), FlashInfer
- **DCP comm backends:** A2A (All-to-All), AG+RS (AllGather + ReduceScatter)
- **CUDA graph:** enabled, disabled

For each configuration, it launches the server, runs 3 request types, and verifies:
1. **Correctness** -- output is non-empty and coherent
2. **Determinism** -- same prompt with `temperature=0` produces identical output twice

Request types:
- `prefill_only` -- input=2048 tokens, output=1 token (tests prefill path)
- `decode_heavy` -- input=32 tokens, output=512 tokens (tests decode path)
- `mixed` -- input=512 tokens, output=256 tokens (tests both paths)

### Configuration

Environment variables in the test script:
- `CUDA_VISIBLE_DEVICES` -- which GPUs to use (default: `0,1,2,3`)
- Number of GPUs determines `TP` and `DCP` size (both set to GPU count)

To add or modify server configurations, edit the `SERVER_CONFIGS` list in the script.

### Expected Runtime

~20-30 minutes for all 24 scenarios (8 server restarts, each taking ~1-2 minutes to start).

---

## Unit Tests

```bash
# Run inside Docker container or with sglang installed:

# DCP FlashAttention backend tests (34 tests)
python3 -m pytest test/srt/test_dcp_flashattn.py -v

# DCP A2A kernel tests (16 tests)
python3 -m pytest test/srt/test_dcp_a2a.py -v

# FA3 vs FlashInfer LSE comparison (6 tests)
python3 -m pytest test/srt/test_fa3_flashinfer_lse_compare.py -v

# FA3 MLA DCP standalone simulation (6 tests)
python3 -m pytest test/srt/test_fa3_mla_dcp_standalone.py -v
```
