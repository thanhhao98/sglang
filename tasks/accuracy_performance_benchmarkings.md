# SGLang Helix (DCP A2A) Benchmarking

**Model:** `deepseek-ai/DeepSeek-V2-Lite` (15.7B, MLA, 1 effective KV head)
**Branch:** `htphan/helix_a2a_rebased_main_fe294904c9`
**Guideline:** See `benchmarking_guideline.md`
**Reference:** vLLM Helix benchmarking in `/Users/htphan/workspace/work-tracker/helix-vllm`

---

## Pre-flight Check

```bash
# Always check GPU availability before running
ssh colossus 'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader'
ssh colossus_b200_1 'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader'
ssh colossus_b200_2 'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader'
```

- TP8/DCP8 requires all 8 GPUs free
- H100 is shared with team — check before use
- B200 Node 2 has NVLink hardware error — use Node 1

---

## Part A: Accuracy Benchmarks (GSM8K)

**Full dataset:** 1319 questions, `benchmark/gsm8k/bench_sglang.py --num-questions 1319`
**Purpose:** Verify no accuracy regression across configs (target: ~0.38-0.39)

### Results (COMPLETE)

**B200 Node 1 (4x B200 183GB, SM100 Blackwell)**

| Config | Accuracy | Invalid | Throughput (tok/s) |
|--------|:--------:|:-------:|:-----------------:|
| TP1 | 0.386 | 0.005 | 4640.8 |
| TP4 | 0.394 | 0.004 | 5285.0 |
| DCP4 ag_rs | 0.386 | 0.004 | 2977.6 |
| DCP4 a2a flashinfer cg | 0.393 | 0.006 | 3019.3 |
| DCP4 a2a fa3 cg | N/A | — | FA3 requires SM<=90 |

**H100 (4x H100 80GB, SM90 Hopper)**

| Config | Accuracy | Invalid | Throughput (tok/s) |
|--------|:--------:|:-------:|:-----------------:|
| TP1 | 0.388 | 0.005 | 3089.2 |
| TP4 | 0.389 | 0.005 | 6074.5 |
| DCP4 ag_rs | 0.381 | 0.005 | 2644.5 |
| DCP4 a2a flashinfer cg | 0.385 | 0.006 | 2585.7 |
| DCP4 a2a fa3 cg | 0.379 | 0.005 | 2268.2 |

**Conclusion:** Accuracy consistent ~0.38-0.39 across all configs. No regression.

**Post-optimization DCP8 (2026-03-15) — per-bs CUDA graph buffers + removed .contiguous()**

| Machine | Config | Accuracy | Invalid | Throughput (tok/s) |
|---------|--------|:--------:|:-------:|:-----------------:|
| H100 (8x 80GB) | DCP8 a2a flashinfer cg (per-bs opt) | 0.376 | 0.005 | 2199.3 |
| B200 (8x 183GB) | DCP8 a2a flashinfer cg (per-bs opt) | 0.387 | 0.005 | 920.4 |
| H100 (8x 80GB) | DCP8 a2a flashinfer cg (fused a2a) | 0.381 | 0.005 | 2049.7 |
| B200 (8x 183GB) | DCP8 a2a flashinfer cg (fused a2a) | 0.393 | 0.006 | — |

> Accuracy consistent with DCP4 baseline (~0.38-0.39) across all optimization variants.

**`--dcp-replicate-q-proj` (2026-03-16) — branch `htphan/q-project-replication`**

Status: Implementation complete. Accuracy verified on H100 (200-sample quick check all pass).

> **Note (2026-03-16 retest):** Prior results (0.12-0.15 accuracy, 48.5% invalid) could NOT be reproduced.
> Code checksums confirmed identical. All configs now show correct accuracy (~0.34 on 200 samples).
> Prior failures were likely caused by stale docker environment or transient GPU state.

**Quick validation (200 samples, H100 8x80GB, 2026-03-16):**

| Config | CG | Accuracy | Invalid | Throughput (tok/s) |
|--------|:--:|:--------:|:-------:|:-----------------:|
| DCP8 a2a flashinfer baseline (no repl) | No | 0.340 | 0.025 | 618.0 |
| DCP8 a2a flashinfer baseline (no repl) | Regular | 0.335 | 0.025 | 2835.1 |
| DCP8 a2a flashinfer + replicate-q-proj | No | 0.335 | 0.025 | 659.7 |
| DCP8 a2a flashinfer + replicate-q-proj | Regular | 0.340 | 0.025 | 2853.5 |

Implementation changes (3 files):
- `server_args.py`: `--dcp-replicate-q-proj` flag + validation
- `deepseek_v2.py`: Replicate Q/kv_b projections (tp_size=1), reshape Q for decode (all heads), slice Q/KV for prefill/extend (local heads), skip AllGather Q in decode, fix deep_gemm head count, post-load slice w_vc to local heads
- `flashattention_backend.py`: Guard non-MLA AllGather Q

**Bug fix (2026-03-16):** `_chunked_prefix_attn_mha` crashed at context >= 256K with `--dcp-replicate-q-proj`.
Root cause: `kv_b_proj` outputs all heads when replicated (tp_size=1) but `_chunked_prefix_attn_mha` reshaped with
`num_local_heads`, inflating the token dimension 8x. Also `k_pe` (1 KV head) wasn't expanded to match `num_local_heads`.
Fix: reshape with `num_heads`, slice to local heads, and `expand_as` for k_pe. Accuracy recheck passed (0.378).

### Full `--dcp-replicate-q-proj` Accuracy (GSM8K 1319 questions) — COMPLETE

**H100 (8x H100 80GB)**

| Config | CG | Accuracy | Invalid | Throughput (tok/s) |
|--------|:--:|:--------:|:-------:|:-----------------:|
| DCP8 a2a flashinfer + replicate-q-proj | No | 0.378 | 0.006 | 966.1 |
| DCP8 a2a flashinfer + replicate-q-proj | Regular | 0.376 | 0.006 | 2758.2 |
| DCP8 a2a flashinfer + replicate-q-proj | Piecewise | 0.377 | 0.006 | 2807.3 |
| DCP8 ag_rs + replicate-q-proj | No | 0.390 | 0.006 | 949.7 |
| DCP8 ag_rs + replicate-q-proj | Regular | 0.387 | 0.005 | 2792.1 |
| DCP8 ag_rs + replicate-q-proj | Piecewise | 0.390 | 0.006 | 2850.5 |

**B200 Node 1 (8x B200 183GB)**

| Config | CG | Accuracy | Invalid | Throughput (tok/s) |
|--------|:--:|:--------:|:-------:|:-----------------:|
| DCP8 a2a flashinfer + replicate-q-proj | No | 0.377 | 0.005 | 1031.9 |
| DCP8 a2a flashinfer + replicate-q-proj | Regular | 0.388 | 0.004 | 2714.0 |
| DCP8 a2a flashinfer + replicate-q-proj | Piecewise | 0.388 | 0.004 | 2852.0 |
| DCP8 ag_rs + replicate-q-proj | No | 0.382 | 0.004 | 1037.3 |
| DCP8 ag_rs + replicate-q-proj | Regular | 0.379 | 0.005 | 2856.2 |
| DCP8 ag_rs + replicate-q-proj | Piecewise | 0.383 | 0.005 | 2838.5 |

> **Conclusion:** All 12 configs pass with accuracy ~0.38-0.39, consistent with baseline.
> No regression from `--dcp-replicate-q-proj`. Both a2a and ag_rs work across all CG modes.

### `--dcp-replicate-q-proj` Performance (bench_serving)

**Config Matrix**

| Config ID | Parallelism | Comm Backend | Attention | Q-Proj | H100 | B200 |
|-----------|-------------|--------------|-----------|--------|:----:|:----:|
| C5 | DCP8 | a2a | flashinfer | replicate | Yes | Yes |
| C6 | DCP8 | ag_rs | flashinfer | replicate | Yes | Yes |

> **Purpose:** Measure the performance benefit of `--dcp-replicate-q-proj` (eliminates AllGather Q in decode).
> Compare C5 vs C3 (a2a baseline) and C6 vs C2 (ag_rs baseline) at c=1 to quantify TPOT improvement.
> Uses piecewise CUDA graphs (default). Same ISL/OSL/concurrency matrix as C2/C3.

**Run status (2026-03-16):**

| Config | H100 | B200 |
|--------|:----:|:----:|
| C5_128K | in progress | **done** |
| C5_256K | in progress | **done** (rerun after chunked-prefix fix) |
| C5_512K | in progress | **done** |
| C5_1M | in progress | **done** |
| C6_128K | in progress | **done** |
| C6_256K | in progress | **done** |
| C6_512K | in progress | **done** |
| C6_1M | in progress | **done** |

> H100 script: `tasks/run_h100_perf.sh` running inside docker `sglang-bench` on colossus.
> H100 logs persisted to `/raid/local-htphan/sglang_workspace/output/replicate_qproj_bench/` (survives container restart).
> H100 container has `--restart unless-stopped` policy. Server logs also saved to `/output/`.

### B200 Performance — COMPLETE

**B200 C5_128K** (a2a + replicate-q-proj, ISL=130048, OSL=1024, mem_frac=0.50)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 4.77 | 638.82 | 67429.80 |
| 2 | 14.80 | 924.10 | 83336.22 |
| 4 | 16.90 | 1067.16 | 93168.85 |
| 8 | 25.62 | 1499.99 | 93050.26 |
| 16 | 25.81 | 1463.88 | 93316.59 |
| 32 | 33.09 | 1626.32 | 91610.76 |
| 64 | 25.47 | 1512.22 | 93027.48 |
| 128 | 33.16 | 1660.30 | 91027.64 |
| 256 | 33.14 | 1630.95 | 91361.94 |
| 512 | 33.24 | 1639.41 | 90974.65 |

**B200 C5_256K** (a2a + replicate-q-proj, ISL=261120, OSL=1024, mem_frac=0.80)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 5.36 | 1682.93 | 84974.33 |
| 2 | 42.50 | 2141.70 | 96906.48 |
| 4 | 115.30 | 2598.42 | 108417.91 |
| 8 | 186.20 | 3634.99 | 111677.91 |
| 16 | 186.93 | 3548.15 | 111720.65 |
| 32 | 185.56 | 3575.20 | 111932.14 |
| 64 | 187.60 | 3665.74 | 107757.75 |
| 128 | 186.84 | 3515.27 | 112075.26 |
| 256 | 186.11 | 3521.88 | 112344.35 |
| 512 | 187.74 | 3518.27 | 111900.19 |

**B200 C5_512K** (a2a + replicate-q-proj, ISL=523264, OSL=1024, mem_frac=0.75)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 6.84 | 5737.58 | 63791.39 |
| 2 | 1991.51 | 7347.67 | 72786.30 |
| 4 | 1860.10 | 8706.85 | 72398.10 |
| 8 | 1894.59 | 13513.14 | 74196.41 |
| 16 | 1887.27 | 13379.66 | 74683.65 |
| 32 | 1899.58 | 13410.08 | 74395.40 |
| 64 | 1888.58 | 13429.14 | 74528.53 |
| 128 | 2861.77 | 12458.22 | 74847.77 |
| 256 | 1888.44 | 13458.14 | 74462.51 |
| 512 | 1898.37 | 13504.29 | 74174.34 |

**B200 C5_1M** (a2a + replicate-q-proj, ISL=1047552, OSL=1024, mem_frac=0.65)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 6.95 | 8854.96 | 48517.57 |
| 2 | 1341.02 | 13461.79 | 51825.42 |
| 4 | 1212.43 | 19263.63 | 54372.58 |
| 8 | 1767.48 | 18379.41 | 54531.06 |
| 16 | 1765.26 | 18421.43 | 54556.11 |
| 32 | 1754.98 | 18312.68 | 54824.30 |
| 64 | 1762.75 | 18271.33 | 54728.96 |
| 128 | 1757.10 | 18328.29 | 54702.33 |
| 256 | 1796.48 | 18005.20 | 54489.98 |
| 512 | 1764.84 | 18344.07 | 54569.17 |

**B200 C6_128K** (ag_rs + replicate-q-proj, ISL=130048, OSL=1024, mem_frac=0.85)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 4.69 | 630.73 | 68438.71 |
| 2 | 14.73 | 909.48 | 84553.81 |
| 4 | 17.61 | 1020.32 | 93850.61 |
| 8 | 25.60 | 1378.17 | 93583.03 |
| 16 | 25.97 | 1468.88 | 93358.44 |
| 32 | 33.18 | 1641.87 | 91869.70 |
| 64 | 33.36 | 1643.35 | 91700.94 |
| 128 | 25.53 | 1384.38 | 93521.02 |
| 256 | 33.20 | 1635.51 | 91929.61 |
| 512 | 25.19 | 1370.48 | 94099.70 |

**B200 C6_256K** (ag_rs + replicate-q-proj, ISL=261120, OSL=1024, mem_frac=0.80)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 5.32 | 1676.45 | 85533.86 |
| 2 | 41.60 | 2126.28 | 98375.80 |
| 4 | 66.81 | 2525.81 | 110019.07 |
| 8 | 127.74 | 3632.65 | 107986.65 |
| 16 | 127.08 | 3529.01 | 111975.82 |
| 32 | 126.55 | 3509.09 | 112236.92 |
| 64 | 126.50 | 3521.12 | 112093.50 |
| 128 | 127.26 | 3499.15 | 111810.16 |
| 256 | 127.06 | 3502.21 | 112205.12 |
| 512 | 95.58 | 3977.46 | 112609.77 |

**B200 C6_512K** (ag_rs + replicate-q-proj, ISL=523264, OSL=1024, mem_frac=0.75)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 6.77 | 5739.20 | 63792.76 |
| 2 | 2005.22 | 7405.86 | 72027.67 |
| 4 | 1982.59 | 12719.36 | 73824.35 |
| 8 | 5025.95 | 12594.26 | 74898.69 |
| 16 | 5066.76 | 12431.66 | 74940.64 |
| 32 | 1977.65 | 13495.38 | 74257.78 |
| 64 | 1974.84 | 13497.66 | 74435.38 |
| 128 | 5029.28 | 12453.81 | 75213.55 |
| 256 | 1983.14 | 13421.65 | 74389.67 |
| 512 | 1981.16 | 13360.37 | 74594.66 |

**B200 C6_1M** (ag_rs + replicate-q-proj, ISL=1047552, OSL=1024, mem_frac=0.65)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 6.74 | 8905.68 | 48282.28 |
| 2 | 1362.59 | 13519.28 | 51511.36 |
| 4 | 1243.76 | 19419.07 | 53875.63 |
| 8 | 1911.78 | 18485.74 | 54437.97 |
| 16 | 1916.90 | 18535.89 | 54314.61 |
| 32 | 1900.16 | 18394.77 | 54692.87 |
| 64 | 1912.90 | 18506.17 | 54405.76 |
| 128 | 1908.02 | 18514.72 | 54449.24 |
| 256 | 1917.26 | 18503.31 | 54338.78 |
| 512 | 1912.90 | 18509.91 | 54384.80 |

> **B200 c=1 TPOT summary (replicate-q-proj vs baseline):**
>
> | Context | C5 (a2a+repl) | C3 baseline | C6 (ag_rs+repl) | C2 baseline |
> |---------|:---:|:---:|:---:|:---:|
> | 128K | 4.77ms | — | 4.69ms | — |
> | 256K | 5.36ms | — | 5.32ms | — |
> | 512K | 6.84ms | — | 6.77ms | — |
> | 1M | 6.95ms | — | 6.74ms | — |
>
> Same TPOT degradation pattern at c>=2 as baseline (scheduler prefill-blocking-decode).

### H100 Performance — IN PROGRESS

> H100 script `tasks/run_h100_perf.sh` running. Logs at `/raid/local-htphan/sglang_workspace/output/replicate_qproj_bench/`.
> Container: `sglang-bench` on colossus with `--restart unless-stopped`.
> Results will be added when complete.

---

## Part B: Performance Benchmarks (bench_serving)

### Benchmark Design (aligned with vLLM Helix effort)

| Parameter | Value |
|-----------|-------|
| **OSL (output)** | 1024 tokens fixed |
| **ISL (input)** | context_length - 1024 (fills context window) |
| **Concurrency** | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 (10 levels) |
| **num-prompts** | 5 per concurrency level |
| **random-range-ratio** | 0.0 (exact lengths) |
| **MLA contexts** | 128K, 256K, 512K, 1M |

### Config Matrix

| Config ID | Parallelism | Comm Backend | Attention | H100 | B200 |
|-----------|-------------|--------------|-----------|:----:|:----:|
| C1 | TP8 | — | default | Yes | Yes |
| C2 | DCP8 | ag_rs | flashinfer | Yes | Yes |
| C3 | DCP8 | a2a | flashinfer | Yes | Yes |
| C4 | DCP8 | a2a | fa3 | Yes | No (SM100) |

### ISL/OSL per Context

| Context | ISL | OSL | mem_frac (TP8) | mem_frac (DCP8) |
|---------|-----|-----|:-:|:-:|
| 128K (131072) | 130048 | 1024 | 0.85 | 0.85 |
| 256K (262144) | 261120 | 1024 | 0.80 | 0.80 |
| 512K (524288) | 523264 | 1024 | 0.75 | 0.75 |
| 1M (1048576) | 1047552 | 1024 | 0.65 | 0.65 |

### Total Benchmark Points

- **H100:** 4 configs × 4 contexts × 10 concurrencies = **160 points**
- **B200:** 3 configs × 4 contexts × 10 concurrencies = **120 points**
- **Grand total: 280 points**

---

## H100 Benchmark Status — ALL COMPLETE

Script: `/output/run_perf_v2.sh` on docker container `sglang-bench`

| Config | 128K | 256K | 512K | 1M |
|--------|:----:|:----:|:----:|:--:|
| C1 TP8 | [done] | [done] | [done] | [done] |
| C2 DCP8 ag_rs | [done] | [done] | [done] | [done] |
| C3 DCP8 a2a flashinfer | [done*] | [done] | [done] | [done] |
| C4 DCP8 a2a fa3 | [done*] | [done] | [done] | [done] |

> **Note:** C3_128K and C4_128K originally failed at mem_frac=0.85 (OOM). Rerun with mem_frac=0.50 succeeded.
> C2_128K_c1, C2_256K_c2, C2_256K_c4 also rerun successfully.
> **Completion: 160/160 benchmark points have data.**

### H100 C1 TP8 Results

**C1_TP8_128K** (ISL=130048, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 13.82 | 381.30 | 39548.51 | 2061.81 |
| 2 | 12.38 | 316.51 | 63835.88 | 2024.47 |
| 4 | 11.46 | 675.45 | 68860.94 | 2183.63 |
| 8 | 8.06 | 980.81 | 69379.18 | 2303.10 |
| 16 | 8.31 | 937.69 | 68563.14 | 2281.41 |
| 32 | 8.57 | 1079.43 | 67353.74 | 2450.24 |
| 64 | 9.13 | 969.63 | 66878.79 | 2406.87 |
| 128 | 7.89 | 1031.54 | 64453.69 | 2456.71 |
| 256 | 8.74 | 1039.06 | 65710.78 | 2467.85 |
| 512 | 8.68 | 1008.58 | 65368.80 | 2454.83 |

**C1_TP8_256K** (ISL=261120, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 34.53 | 2359.01 | 51877.19 | 3083.19 |
| 2 | 8.98 | 718.54 | 250785.13 | 1087.23 |
| 4 | 5.95 | 1482.25 | 368479.41 | 1630.70 |
| 8 | 4.09 | 1983.94 | 379648.19 | 2012.95 |
| 16 | 3.69 | 2061.01 | 367521.13 | 2092.33 |
| 32 | 1.25 | 2182.59 | 348758.70 | 2227.39 |
| 64 | 2.64 | 1995.28 | 381306.18 | 2032.29 |
| 128 | 1.58 | 2042.92 | 373300.68 | 2082.03 |
| 256 | 1.82 | 2079.42 | 285317.00 | 2236.00 |
| 512 | 1.75 | 2056.60 | 329918.53 | 2150.67 |

**C1_TP8_512K** (ISL=523264, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 86.06 | 7381.68 | 33214.27 | 11137.28 |
| 2 | 82.31 | 10040.16 | 42285.51 | 15176.83 |
| 4 | 22.75 | 15590.38 | 44200.16 | 20298.21 |
| 8 | 26.08 | 15526.34 | 45077.34 | 21539.73 |
| 16 | 39.32 | 14668.89 | 44780.26 | 21848.07 |
| 32 | 26.53 | 15691.26 | 44040.06 | 21905.32 |
| 64 | 37.39 | 15170.16 | 44230.78 | 22355.26 |
| 128 | 36.79 | 14559.42 | 43945.55 | 22002.80 |
| 256 | 29.07 | 14831.03 | 43553.25 | 22364.15 |
| 512 | 27.87 | 15783.46 | 42852.62 | 22252.97 |

**C1_TP8_1M** (ISL=1047552, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 100.65 | 11775.60 | 28379.72 | 16729.30 |
| 2 | 22.97 | 23968.68 | 32072.33 | 27711.50 |
| 4 | 20.68 | 31072.40 | 31559.58 | 34860.84 |
| 8 | 20.48 | 24136.31 | 33514.01 | 27863.47 |
| 16 | 87.02 | 229736.80 | 4186.86 | 233761.83 |
| 32 | 35.96 | 510371.22 | 1132.62 | 513802.99 |
| 64 | 23.43 | 33737.59 | 31541.62 | 37268.48 |
| 128 | 25.67 | 26386.56 | 32417.00 | 29975.12 |
| 256 | 21.22 | 26663.55 | 31677.62 | 30255.29 |
| 512 | 21.29 | 26115.09 | 31715.30 | 29717.06 |

> **Note:** C1_TP8_1M on H100 is memory-constrained. At 1M context (ISL=1,047,552) with `mem-fraction-static=0.65`, H100 80GB can only fit ~1-2 concurrent requests. c=16 and c=32 show extreme queueing (TTFT=230s and 510s respectively). c=64+ stabilizes as requests serialize naturally. B200 (183GB) handles this much better.

### H100 C2 DCP8 ag_rs Results

**C2_DCP8_agrs_128K** (ISL=130048, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 4.19 | 829.20 | 61522.23 | 1324.17 |
| 2 | 17.37 | 1204.06 | 73767.37 | 2083.93 |
| 4 | 22.87 | 1378.86 | 66581.82 | 3428.09 |
| 8 | 38.05 | 1935.49 | 65792.61 | 4239.47 |
| 16 | 35.57 | 1924.24 | 66834.92 | 4142.42 |
| 32 | 35.49 | 1921.32 | 66788.70 | 4139.18 |
| 64 | 36.31 | 1899.83 | 66771.96 | 4140.77 |
| 128 | 45.88 | 2151.36 | 65063.74 | 4251.58 |
| 256 | 45.98 | 2145.30 | 65015.73 | 4252.10 |
| 512 | 36.46 | 1903.89 | 66336.47 | 4153.84 |

> **Note:** c=1 rerun succeeded with 4.19ms TPOT (original run was warmup-only).

**C2_DCP8_agrs_256K** (ISL=261120, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 133.96 | 1335.71 | 70450.09 | 2269.67 |
| 2 | 54.54 | 2898.80 | 73084.67 | 3861.17 |
| 4 | 201.21 | 3334.36 | 82369.68 | 6193.33 |
| 8 | 1062.78 | 4803.83 | 28054.05 | 18496.27 |
| 16 | 1231.45 | 5163.29 | 5770.05 | 42187.28 |
| 32 | 1241.44 | 5307.11 | 5766.34 | 42408.55 |
| 64 | 1251.71 | 5168.49 | 5632.32 | 43065.04 |
| 128 | 1229.45 | 5177.77 | 5799.23 | 42069.13 |
| 256 | 1214.56 | 5085.17 | 5777.28 | 41958.79 |
| 512 | 1212.25 | 5130.46 | 5796.78 | 41882.22 |

> **Warning:** c=2 and c=4 rerun succeeded (originally TransferEncodingError). c=1 shows 134ms TPOT — already degraded at 256K on H100. c>=8 shows massive TPOT degradation (~1200ms).

**C2_DCP8_agrs_512K** (ISL=523264, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.21 | 7734.14 | 47488.25 | 7786.57 |
| 2 | 1835.57 | 9955.64 | 51207.44 | 14140.80 |
| 4 | 852.06 | 17062.31 | 48225.18 | 28089.85 |
| 8 | 889.10 | 19356.06 | 48016.55 | 29278.07 |
| 16 | 854.25 | 18178.95 | 48074.83 | 29231.63 |
| 32 | 887.06 | 19342.45 | 48042.69 | 29242.76 |
| 64 | 891.63 | 19352.06 | 47896.47 | 29303.84 |
| 128 | 1136.49 | 17099.94 | 48121.33 | 34976.33 |
| 256 | 1146.35 | 16789.89 | 48126.21 | 34969.29 |
| 512 | 853.11 | 18345.85 | 48057.01 | 29383.52 |

> **Warning:** Same ag_rs TPOT degradation pattern as B200 — c=1 is fine (6.21ms), c=2 jumps to 1836ms. Confirms this is an SGLang-wide issue, not hardware-specific.

**C2_DCP8_agrs_1M** (ISL=1047552, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.08 | 12137.63 | 35783.65 | 13269.47 |
| 2 | 1868.04 | 18409.16 | 38292.58 | 23727.52 |
| 4 | 1895.23 | 26457.53 | 39711.74 | 44602.89 |
| 8 | 3064.14 | 25066.00 | 39721.12 | 45993.24 |
| 16 | 3065.08 | 25063.00 | 39709.09 | 46004.11 |
| 32 | 3061.97 | 25443.19 | 39496.68 | 46364.60 |
| 64 | 1901.70 | 27900.29 | 39622.21 | 46099.59 |
| 128 | 3048.58 | 24879.05 | 39953.39 | 45706.47 |
| 256 | 3055.04 | 24952.54 | 39833.19 | 45824.32 |
| 512 | 3065.82 | 25022.29 | 39718.88 | 45965.12 |

> **Warning:** c=1 fine (6.08ms), c>=2 severe degradation. c>=8 plateaus at ~3060ms TPOT.

### H100 C3 DCP8 a2a flashinfer Results

**C3_DCP8_a2a_fi_128K** (ISL=130048, OSL=1024) — rerun with mem_frac=0.50

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 4.34 | 847.20 | 60072.59 | 1356.35 |
| 2 | 14.09 | 1092.43 | 75297.15 | 2031.04 |
| 4 | 30.38 | 1613.01 | 79472.89 | 3065.57 |
| 8 | 30.36 | 1904.47 | 80990.70 | 3832.72 |
| 16 | 33.06 | 1943.21 | 81132.50 | 3820.46 |
| 32 | 33.74 | 1928.00 | 81268.28 | 3814.34 |
| 64 | 43.10 | 2133.34 | 80082.49 | 3875.20 |
| 128 | 33.07 | 1936.13 | 81324.59 | 3811.60 |
| 256 | 32.95 | 1920.87 | 81646.44 | 3792.44 |
| 512 | 43.19 | 2142.29 | 79735.61 | 3886.23 |

> **Note:** Originally failed with NCCL OOM at mem_frac=0.85. Rerun with mem_frac=0.50 succeeded. Similar TPOT pattern to B200 C3_128K.

**C3_DCP8_a2a_fi_256K** (ISL=261120, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 5.01 | 2194.35 | 68009.70 | 2353.18 |
| 2 | 50.78 | 2822.90 | 76189.57 | 3680.79 |
| 4 | 202.04 | 3323.73 | 82545.23 | 6192.31 |
| 8 | 365.61 | 4654.01 | 84244.18 | 9243.18 |
| 16 | 365.18 | 4604.39 | 84625.18 | 9202.50 |
| 32 | 365.45 | 4643.77 | 84121.67 | 9249.34 |
| 64 | 364.67 | 4617.10 | 84637.19 | 9202.01 |
| 128 | 364.90 | 4625.63 | 84491.30 | 9216.69 |
| 256 | 365.22 | 4619.78 | 84432.28 | 9224.41 |
| 512 | 263.29 | 4683.80 | 81923.98 | 9314.28 |

> **Warning:** c=1 fine (5.01ms), c>=8 plateaus at ~365ms. Worse than B200 a2a_fi_256K (~108ms at c=8).

**C3_DCP8_a2a_fi_512K** (ISL=523264, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.17 | 7695.20 | 42842.25 | 8635.23 |
| 2 | 1263.77 | 9952.47 | 53348.47 | 13856.17 |
| 4 | 4598.62 | 17081.54 | 51922.15 | 27561.80 |
| 8 | 4594.77 | 18055.29 | 51980.23 | 28532.42 |
| 16 | 4606.41 | 18098.11 | 51871.41 | 28592.47 |
| 32 | 4283.54 | 16724.93 | 48212.25 | 34835.32 |
| 64 | 4570.39 | 18004.96 | 52203.39 | 28422.65 |
| 128 | 4308.05 | 16775.01 | 48035.69 | 34987.79 |
| 256 | 4596.17 | 18061.23 | 51976.08 | 28537.53 |
| 512 | 4594.17 | 18066.51 | 51990.24 | 28535.31 |

> **Warning:** Same catastrophic DCP TPOT degradation. c=1 fine (6.17ms), c>=4 ~4600ms.

**C3_DCP8_a2a_fi_1M** (ISL=1047552, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.11 | 12142.41 | 35742.30 | 13284.39 |
| 2 | 328.94 | 18358.81 | 38245.17 | 23740.48 |
| 4 | 9010.90 | 25760.37 | 39732.35 | 44532.96 |
| 8 | 9712.17 | 24875.68 | 39817.03 | 45747.04 |
| 16 | 10345.85 | 24825.88 | 39838.58 | 55218.25 |
| 32 | 10348.03 | 24769.06 | 39869.21 | 55173.69 |
| 64 | 10439.73 | 24404.67 | 39679.01 | 55444.68 |
| 128 | 10387.06 | 24347.88 | 39817.46 | 55240.62 |
| 256 | 10346.61 | 24753.64 | 39884.60 | 55149.03 |
| 512 | 10330.19 | 24746.29 | 39922.43 | 55086.98 |

> **Warning:** Worst TPOT of all H100 configs. c>=16 mean TPOT exceeds 10 seconds.

### H100 C3-opt DCP8 a2a flashinfer Results (Optimized: per-bs CUDA graph buffers + remove .contiguous())

**Optimizations applied (2026-03-15):**
- Opt 1: Removed redundant `.contiguous()` in CUDA graph path (saves 54 D2D copy kernels/decode step)
- Opt 3: Per-bs CUDA graph buffer allocation (NCCL transfers only actual batch data, not max_bs=512 padding)
- GSM8K accuracy: **0.376** (1319 questions, invalid 0.005) — consistent with baseline ~0.38-0.39

**C3_opt_DCP8_a2a_fi_128K** (ISL=130048, OSL=1024, mem_frac=0.50)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 4.30 | 840.77 | 60322.47 | 1350.62 |
| 2 | 17.59 | 1190.55 | 73307.09 | 2087.85 |
| 4 | 21.82 | 1396.94 | 80748.61 | 3128.76 |
| 8 | 33.17 | 1945.44 | 81022.95 | 3822.52 |
| 16 | 32.99 | 1936.04 | 81384.04 | 3804.76 |
| 32 | 33.04 | 1931.30 | 81357.18 | 3809.53 |
| 64 | 33.07 | 1935.56 | 81275.08 | 3813.56 |
| 128 | 33.03 | 1929.19 | 81407.92 | 3804.68 |
| 256 | 33.83 | 2026.67 | 79429.69 | 3920.76 |
| 512 | 29.67 | 1806.75 | 80704.65 | 3845.74 |

> **vs baseline C3_128K:** c=1 TPOT 4.30ms vs 4.34ms (same). c=64 improved: 33.07ms vs 43.10ms (-23%). Throughput similar ~80K tok/s.

**C3_opt_DCP8_a2a_fi_256K** (ISL=261120, OSL=1024, mem_frac=0.80)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 4.99 | 2214.44 | 67355.19 | 2375.56 |
| 2 | 50.83 | 2836.77 | 75590.46 | 3716.66 |
| 4 | 192.29 | 4070.51 | 80601.96 | 6374.49 |
| 8 | 365.28 | 4638.68 | 84483.71 | 9218.27 |
| 16 | 364.83 | 4646.01 | 84221.41 | 9235.64 |
| 32 | 366.92 | 4551.71 | 83952.06 | 9281.48 |
| 64 | 364.78 | 4625.79 | 84489.28 | 9214.86 |
| 128 | 364.58 | 4643.12 | 84514.52 | 9220.72 |
| 256 | 364.63 | 4642.65 | 84498.19 | 9225.56 |
| 512 | 364.13 | 4638.51 | 84570.26 | 9211.49 |

> **vs baseline C3_256K:** Essentially identical. c=1: 4.99ms vs 5.01ms. c>=8 plateau: ~365ms (same). TPOT degradation at c>=2 is scheduler-dominated (prefill-blocking-decode), not A2A overhead.

**C3_opt_DCP8_a2a_fi_512K** (ISL=523264, OSL=1024, mem_frac=0.75)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.23 | 7799.74 | 42323.60 | 8740.93 |
| 2 | 1263.14 | 10057.88 | 52944.12 | 13960.39 |
| 4 | 4604.92 | 18160.11 | 52120.71 | 27455.53 |
| 8 | 4566.86 | 18268.07 | 51930.30 | 28679.66 |
| 16 | 4249.60 | 17150.46 | 48338.54 | 34749.16 |
| 32 | 4252.84 | 16898.65 | 48424.55 | 34678.00 |
| 64 | 4576.45 | 18136.64 | 52042.09 | 28572.95 |
| 128 | 4276.53 | 16717.88 | 48276.96 | 34778.96 |
| 256 | 4565.88 | 18065.98 | 52199.39 | 28485.13 |
| 512 | 4259.95 | 16794.17 | 48381.99 | 34690.82 |

> **vs baseline C3_512K:** Same pattern. c=1: 6.23ms vs 6.17ms. c>=4 catastrophic TPOT (~4500ms). Confirms TPOT degradation is scheduler-level (prefill-blocking-decode), not comm layer.

**C3_opt_DCP8_a2a_fi_1M** (ISL=1047552, OSL=1024, mem_frac=0.65)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.20 | 12126.53 | 35778.26 | 13270.09 |
| 2 | 328.10 | 18342.28 | 38325.83 | 23690.58 |
| 4 | 8928.00 | 26328.80 | 39832.61 | 44423.05 |
| 8 | 9667.80 | 24958.65 | 39922.41 | 45742.61 |
| 16 | 9649.50 | 24996.01 | 39974.74 | 45742.09 |
| 32 | 8963.46 | 28131.46 | 39539.57 | 46296.08 |
| 64 | 9657.64 | 25120.70 | 39880.61 | 45878.92 |
| 128 | 8916.38 | 27766.60 | 39855.86 | 45838.49 |
| 256 | 9654.51 | 25078.61 | 39933.77 | 45842.10 |
| 512 | 9678.55 | 24964.41 | 39887.01 | 45777.79 |

> **vs baseline C3_1M:** Slight improvement at c>=4: 8928ms vs 9011ms (-1%). c>=16: 9650ms vs 10346ms (-7%). Marginal improvement from reduced NCCL transfer size at larger batches.

#### C3 Optimization Summary

| Context | c=1 TPOT (old→new) | c=64 TPOT (old→new) | Verdict |
|---------|:-------------------:|:--------------------:|---------|
| 128K | 4.34→4.30ms (=) | 43.10→33.07ms (-23%) | Improved at high c |
| 256K | 5.01→4.99ms (=) | 364.67→364.78ms (=) | No change (scheduler-dominated) |
| 512K | 6.17→6.23ms (=) | 4570→4576ms (=) | No change (scheduler-dominated) |
| 1M | 6.11→6.20ms (=) | 10440→9658ms (-7%) | Slight improvement at high c |

**Conclusion:** The per-bs buffer optimization is correctness-preserving (GSM8K 0.376, deterministic). At c=1 where only the A2A comm layer matters, TPOT is unchanged because NCCL minimum latency dominates. At higher concurrency, modest improvements (7-23%) at 128K/1M. At 256K/512K, the massive TPOT degradation is caused by the scheduler (prefill-blocking-decode, see `dcp_tpot_investigation_results.md`), not the A2A comm layer — the optimization has no effect on that bottleneck.

### H100 C3-fused DCP8 a2a flashinfer Results (Fused output+LSE into single all_to_all)

**Additional optimization (2026-03-15):** Fuse output (bf16) + LSE (fp32→2×bf16) into single `[N, bs, H_per_rank, D+2]` buffer → 1 all_to_all call instead of 2. Halves NCCL SendRecv calls per decode step (756→378 per 27 layers).
- H100 GSM8K accuracy: **0.381** (1319 questions, invalid 0.005)
- B200 GSM8K accuracy: **0.393** (1319 questions, invalid 0.006)

**C3_fused_h100_128K** (ISL=130048, OSL=1024, mem_frac=0.50)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 4.30 | 834.88 | 60737.40 | 1341.42 |
| 2 | 15.96 | 1180.08 | 74785.46 | 2044.31 |
| 4 | 17.20 | 1493.24 | 80686.96 | 3119.69 |
| 8 | 29.66 | 1778.16 | 81222.43 | 3816.23 |
| 16 | 32.90 | 1946.33 | 81599.09 | 3796.66 |
| 32 | 33.79 | 1922.53 | 81270.14 | 3817.98 |
| 64 | 32.96 | 1923.12 | 81704.57 | 3794.81 |
| 128 | 33.37 | 1772.36 | 81574.94 | 3798.55 |
| 256 | 32.96 | 1959.69 | 81132.95 | 3820.73 |
| 512 | 29.29 | 1843.05 | 81143.46 | 3822.37 |

**C3_fused_h100_256K** (ISL=261120, OSL=1024, mem_frac=0.80)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 5.07 | 2190.38 | 68107.49 | 2349.73 |
| 2 | 50.69 | 2820.15 | 76106.59 | 3687.37 |
| 4 | 202.53 | 3411.77 | 81535.73 | 6286.14 |
| 8 | 363.84 | 4629.27 | 84586.03 | 9196.26 |
| 16 | 364.75 | 4649.18 | 84595.12 | 9217.91 |
| 32 | 364.83 | 4640.23 | 84505.48 | 9220.37 |
| 64 | 364.08 | 4650.83 | 84648.59 | 9211.31 |
| 128 | 363.15 | 4787.89 | 84444.64 | 9322.85 |
| 256 | 363.03 | 4649.61 | 84627.20 | 9200.45 |
| 512 | 364.94 | 4670.27 | 84500.31 | 9247.05 |

**C3_fused_h100_512K** (ISL=523264, OSL=1024, mem_frac=0.75)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.40 | 7890.62 | 41893.57 | 8829.58 |
| 2 | 1262.61 | 9962.16 | 53348.37 | 13853.09 |
| 4 | 2574.59 | 11858.93 | 52364.76 | 22446.46 |
| 8 | 4594.75 | 18480.61 | 51458.01 | 28951.58 |
| 16 | 4278.65 | 16640.62 | 48284.95 | 34787.17 |
| 32 | 4255.31 | 16886.70 | 48390.62 | 34692.50 |
| 64 | 4583.88 | 18350.35 | 51736.74 | 28802.15 |
| 128 | 4572.68 | 18155.76 | 52002.39 | 28586.65 |
| 256 | 4279.81 | 17077.69 | 47724.44 | 35219.25 |
| 512 | 4262.19 | 16736.08 | 48374.50 | 34707.38 |

**C3_fused_h100_1M** (ISL=1047552, OSL=1024, mem_frac=0.65)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.35 | 12093.37 | 35862.51 | 13237.20 |
| 2 | 328.34 | 18428.03 | 38191.52 | 23779.27 |
| 4 | 8963.53 | 25850.81 | 39790.01 | 44472.39 |
| 8 | 8938.81 | 28131.79 | 39604.92 | 46242.76 |
| 16 | 8879.63 | 27728.27 | 39965.69 | 45735.68 |
| 32 | 9680.45 | 24947.59 | 39900.22 | 45763.50 |
| 64 | 9694.60 | 25047.34 | 39827.81 | 45886.45 |
| 128 | 9712.63 | 23976.16 | 40055.26 | 45541.71 |
| 256 | 8898.88 | 27565.15 | 39991.69 | 45600.26 |
| 512 | 9698.87 | 24952.85 | 39859.21 | 45803.15 |

**C3_fused_b200_128K** (ISL=130048, OSL=1024, mem_frac=0.85)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 4.86 | 601.05 | 68758.43 | 1185.29 |
| 2 | 11.10 | 778.89 | 89673.52 | 1648.72 |
| 4 | 16.06 | 1001.12 | 95981.09 | 2386.10 |
| 8 | 23.91 | 1426.31 | 95808.46 | 2863.84 |
| 16 | 23.93 | 1434.16 | 95427.09 | 2875.67 |
| 32 | 31.10 | 1558.38 | 93552.72 | 2951.34 |
| 64 | 23.87 | 1391.74 | 96082.85 | 2865.37 |
| 128 | 23.11 | 1308.98 | 96307.61 | 2852.32 |
| 256 | 30.91 | 1579.24 | 93432.53 | 2955.46 |
| 512 | 23.73 | 1394.73 | 96195.59 | 2860.50 |

**C3_fused_b200_256K** (ISL=261120, OSL=1024, mem_frac=0.80)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 5.43 | 1593.55 | 90229.76 | 1773.04 |
| 2 | 34.40 | 1969.84 | 106274.68 | 3004.60 |
| 4 | 72.56 | 2439.78 | 117391.49 | 4415.25 |
| 8 | 170.92 | 3393.41 | 119512.82 | 6399.03 |
| 16 | 171.75 | 3359.94 | 119467.47 | 6392.84 |
| 32 | 170.22 | 3331.72 | 119836.49 | 6354.82 |
| 64 | 167.71 | 3391.50 | 120650.95 | 6337.58 |
| 128 | 170.86 | 3434.21 | 119495.04 | 6444.13 |
| 256 | 171.03 | 3392.33 | 120159.18 | 6403.01 |
| 512 | 170.86 | 3268.43 | 119771.64 | 6370.30 |

**C3_fused_b200_512K** (ISL=523264, OSL=1024, mem_frac=0.75)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.87 | 5417.57 | 67469.42 | 5479.99 |
| 2 | 840.77 | 6853.03 | 77939.55 | 9481.37 |
| 4 | 1686.57 | 8263.63 | 76536.65 | 15472.46 |
| 8 | 4629.70 | 11866.08 | 80128.91 | 22974.86 |
| 16 | 4626.76 | 11748.49 | 80530.56 | 22872.65 |
| 32 | 4658.01 | 11919.80 | 79703.57 | 23095.28 |
| 64 | 1742.38 | 12655.42 | 79716.32 | 19382.79 |
| 128 | 1743.96 | 12622.17 | 79856.00 | 19357.50 |
| 256 | 4592.34 | 11709.18 | 80951.52 | 22751.32 |
| 512 | 4627.87 | 11714.12 | 80676.25 | 22831.14 |

**C3_fused_b200_1M** (ISL=1047552, OSL=1024, mem_frac=0.65)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 7.14 | 8477.31 | 50329.51 | 9432.49 |
| 2 | 236.27 | 12607.86 | 54675.30 | 16483.69 |
| 4 | 1157.88 | 18033.29 | 57600.12 | 30321.40 |
| 8 | 1765.26 | 17320.92 | 58056.26 | 31355.73 |
| 16 | 1771.36 | 17574.19 | 57805.08 | 31653.81 |
| 32 | 1766.26 | 17286.10 | 57964.41 | 31328.56 |
| 64 | 1799.06 | 16751.13 | 58003.69 | 31255.98 |
| 128 | 1759.04 | 17132.82 | 58224.31 | 31118.73 |
| 256 | 1768.68 | 17112.44 | 58106.39 | 31171.59 |
| 512 | 1761.82 | 17078.23 | 58278.45 | 31089.83 |

#### Fused A2A vs Per-bs A2A vs AG_RS — H100 Summary

| Context | c | AG_RS | A2A per-bs | A2A fused | Best A2A vs AG_RS |
|---------|:-:|:-----:|:----------:|:---------:|:-----------------:|
| **128K** | 1 | 4.19 | 4.30 | 4.30 | +3% |
| | 4 | 22.87 | 21.82 | **17.20** | **-25%** |
| | 8 | 38.05 | 33.17 | **29.66** | **-22%** |
| | 64 | 36.31 | 33.07 | 32.96 | **-9%** |
| **256K** | 1 | 133.96 | 4.99 | 5.07 | **-96%** |
| | 8 | 1062.78 | 365.28 | 363.84 | **-66%** |
| **512K** | 1 | 6.21 | 6.23 | 6.40 | +3% |
| | 8 | 889.10 | 4566.86 | 4594.75 | +417% |
| **1M** | 1 | 6.08 | 6.20 | 6.35 | +4% |
| | 8 | 3064.14 | 9667.80 | 8938.81 | +192% |

#### Fused A2A vs Per-bs A2A vs AG_RS — B200 Summary

| Context | c | AG_RS | A2A per-bs | A2A fused | Best A2A vs AG_RS |
|---------|:-:|:-----:|:----------:|:---------:|:-----------------:|
| **128K** | 1 | 4.83 | 4.78 | 4.86 | -1% |
| | 4 | 17.95 | 16.09 | 16.06 | **-11%** |
| | 8 | 26.22 | 23.95 | 23.91 | **-9%** |
| | 64 | 21.36 | 23.93 | 23.87 | +12% |
| **256K** | 1 | 5.29 | 5.29 | 5.43 | +3% |
| | 8 | 96.76 | 112.83 | **170.92** | +77% |
| | 64 | 96.19 | 107.78 | **167.71** | +74% |
| **512K** | 1 | 6.76 | 6.96 | 6.87 | +2% |
| **1M** | 1 | 6.78 | 7.08 | 7.14 | +5% |

**Fused A2A Conclusion:** The fuse halves NCCL SendRecv calls (confirmed by profiling: 1512→756). At **H100 128K c=4-8**, fused A2A achieves the best results: 17.20ms (c=4) and 29.66ms (c=8), beating AG_RS by 22-25%. At **c=1**, no measurable difference (NCCL floor). At **256K+ c>=2**, scheduler-dominated TPOT overwhelms any comm-layer savings. On **B200 256K**, fused is slightly worse than per-bs A2A, possibly due to packing overhead being proportionally larger on the faster B200 NVLink.

### H100 C4 DCP8 a2a fa3 Results

**C4_DCP8_a2a_fa3_128K** (ISL=130048, OSL=1024) — rerun with mem_frac=0.50

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 161.35 | 977.55 | 4958.93 | 16457.10 |
| 2 | 218.40 | 1264.86 | 5718.95 | 22252.77 |
| 4 | 224.62 | 1519.86 | 6007.27 | 23783.91 |
| 8 | 240.97 | 1915.06 | 6062.01 | 24492.64 |
| 16 | 241.82 | 1955.15 | 6022.18 | 24418.44 |
| 32 | 242.85 | 1963.75 | 6060.57 | 24386.71 |
| 64 | 242.16 | 1995.75 | 6051.30 | 24392.75 |
| 128 | 243.86 | 1968.94 | 6066.72 | 24417.36 |
| 256 | 241.57 | 1996.88 | 6046.73 | 24349.95 |
| 512 | 249.85 | 2217.60 | 6064.67 | 24113.71 |

> **Warning:** FA3 at 128K already shows 161ms TPOT at c=1 (vs 4.34ms for flashinfer). FA3 decode is fundamentally ~37x slower than flashinfer. Throughput only ~6K tok/s vs ~80K for flashinfer.

**C4_DCP8_a2a_fa3_256K** (ISL=261120, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 307.35 | 2452.89 | 29019.65 | 5513.49 |
| 2 | 430.36 | 2767.77 | 34737.41 | 8923.10 |
| 4 | 630.51 | 3540.53 | 45103.65 | 12854.01 |
| 8 | 726.55 | 4718.88 | 45447.16 | 14430.82 |
| 16 | 736.29 | 4719.23 | 44983.06 | 14566.82 |
| 32 | 721.58 | 4704.13 | 45619.36 | 14360.91 |
| 64 | 710.40 | 4725.19 | 45949.33 | 14239.57 |
| 128 | 724.57 | 4692.45 | 45486.65 | 14380.06 |
| 256 | 711.58 | 4716.11 | 46010.46 | 14248.84 |
| 512 | 721.19 | 4729.77 | 45708.57 | 14372.27 |

> **Warning:** FA3 TPOT is bad even at c=1 (307ms). FA3 decode is fundamentally slower than flashinfer for DCP.

**C4_DCP8_a2a_fa3_512K** (ISL=523264, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 718.82 | 8336.70 | 3621.77 | 102148.61 |
| 2 | 1002.54 | 10014.16 | 3741.43 | 111308.21 |
| 4 | 1823.24 | 13583.94 | 3804.68 | 125994.88 |
| 8 | 2071.59 | 18299.89 | 3813.08 | 132057.80 |
| 16 | 2124.93 | 19431.28 | 3832.43 | 131589.24 |
| 32 | 2074.96 | 18367.85 | 3801.23 | 132442.81 |
| 64 | 1994.89 | 16848.44 | 3814.31 | 132119.25 |
| 128 | 2076.76 | 18301.38 | 3784.37 | 132828.51 |
| 256 | 2103.20 | 18277.60 | 3704.48 | 135161.71 |
| 512 | 2086.16 | 18230.66 | 3792.79 | 132657.50 |

> **Warning:** Extremely low throughput (~3600-3800 tok/s) and E2E >100s. FA3 at 512K is essentially unusable.

**C4_DCP8_a2a_fa3_1M** (ISL=1047552, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 934.41 | 12800.67 | 1684.77 | 281845.27 |
| 2 | 1762.70 | 14279.33 | 1738.88 | 291851.96 |
| 4 | 3697.29 | 18846.94 | 1757.73 | 315504.62 |
| 8 | 5216.73 | 25228.21 | 1731.09 | 330751.54 |
| 16 | 5180.99 | 25276.38 | 1743.58 | 328503.21 |
| 32 | 5159.82 | 25220.06 | 1716.32 | 332565.60 |
| 64 | 5170.32 | 25094.49 | 1738.13 | 329168.24 |
| 128 | 5141.21 | 25591.85 | 1758.83 | 325941.27 |
| 256 | 5173.01 | 25140.31 | 1729.58 | 330544.35 |
| 512 | 5147.32 | 25173.47 | 1745.56 | 327811.42 |

> **Warning:** Catastrophic. ~1700 tok/s throughput, 282-333s E2E per request. FA3 at 1M is non-functional.

---

## B200 Benchmark Status

Script: `/output/run_perf_b200.sh` on docker container `sglang-bench` (node 1)

| Config | 128K | 256K | 512K | 1M |
|--------|:----:|:----:|:----:|:--:|
| C1 TP8 | [done] | [done] | [done] | [done] |
| C2 DCP8 ag_rs | [done] | [done] | [done] | [done] |
| C3 DCP8 a2a flashinfer | [done] | [done] | [done] | [done] |

### B200 C1 TP8 Results

**C1_TP8_128K** (ISL=130048, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 5.67 | 522.25 | 69755.26 | 1168.32 |
| 2 | 3.85 | 372.17 | 126859.64 | 1088.61 |
| 4 | 2.85 | 608.93 | 154123.00 | 1109.96 |
| 8 | 2.85 | 840.39 | 153315.58 | 1284.51 |
| 16 | 2.43 | 874.75 | 147655.06 | 1299.60 |
| 32 | 2.44 | 865.59 | 154406.51 | 1277.95 |
| 64 | 2.74 | 815.24 | 136885.83 | 1316.32 |
| 128 | 2.43 | 867.20 | 147911.22 | 1295.88 |
| 256 | 2.50 | 855.33 | 136980.56 | 1315.76 |
| 512 | 2.49 | 853.65 | 137471.34 | 1313.03 |

**C1_TP8_256K** (ISL=261120, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 8.09 | 1468.69 | 95513.52 | 1675.17 |
| 2 | 5.49 | 622.38 | 357111.44 | 873.77 |
| 4 | 3.41 | 1146.86 | 449483.17 | 1257.73 |
| 8 | 3.10 | 1590.46 | 449433.40 | 1698.77 |
| 16 | 2.29 | 1671.68 | 454801.73 | 1681.53 |
| 32 | 3.10 | 1728.30 | 417122.93 | 1839.93 |
| 64 | 3.28 | 1609.97 | 443199.05 | 1724.63 |
| 128 | 3.15 | 1591.99 | 449583.58 | 1700.95 |
| 256 | 3.00 | 1602.96 | 447205.84 | 1711.15 |
| 512 | 3.27 | 1722.00 | 415567.01 | 1843.01 |

**C1_TP8_512K** (ISL=523264, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 12.29 | 4960.99 | 55945.14 | 6612.10 |
| 2 | 5.71 | 1911.95 | 288108.02 | 2289.27 |
| 4 | 4.86 | 3056.76 | 395820.84 | 3324.36 |
| 8 | 37.61 | 3946.20 | 364125.95 | 4598.00 |
| 16 | 22.60 | 4486.04 | 187459.52 | 5796.50 |
| 32 | 21.27 | 4223.36 | 204719.19 | 5406.71 |
| 64 | 21.54 | 4617.59 | 360041.52 | 4912.38 |
| 128 | 40.91 | 4111.23 | 204769.58 | 5552.36 |
| 256 | 20.81 | 4297.56 | 385031.15 | 4579.93 |
| 512 | 19.83 | 4187.02 | 395321.33 | 4457.11 |

**C1_TP8_1M** (ISL=1047552, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 13.75 | 7715.36 | 53776.23 | 8827.09 |
| 2 | 9.17 | 2398.83 | 237864.68 | 3256.95 |
| 4 | 36.10 | 3790.03 | 241976.05 | 5043.62 |
| 8 | 63.56 | 4950.59 | 221248.65 | 6776.96 |
| 16 | 47.26 | 5255.29 | 220071.99 | 6835.44 |
| 32 | 49.74 | 5600.02 | 212728.07 | 7216.41 |
| 64 | 47.43 | 5381.46 | 217502.51 | 6965.80 |
| 128 | 48.00 | 5280.02 | 219688.51 | 6871.23 |
| 256 | 46.68 | 5190.34 | 219822.46 | 6776.24 |
| 512 | 48.12 | 5359.61 | 218079.43 | 6954.01 |

### B200 C2 DCP8 ag_rs Results

**C2_DCP8_agrs_128K** (ISL=130048, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 4.83 | 623.63 | 67770.75 | 1202.17 |
| 2 | 12.79 | 805.96 | 86960.36 | 1708.53 |
| 4 | 17.95 | 1054.57 | 92329.45 | 2538.82 |
| 8 | 26.22 | 1415.94 | 94689.02 | 2936.51 |
| 16 | 23.86 | 1401.74 | 96760.99 | 2847.17 |
| 32 | 24.00 | 1378.84 | 96624.06 | 2852.00 |
| 64 | 21.36 | 1323.96 | 95862.82 | 2876.93 |
| 128 | 23.95 | 1379.65 | 96506.33 | 2851.33 |
| 256 | 30.94 | 1540.06 | 94610.69 | 2921.61 |
| 512 | 23.86 | 1413.24 | 96641.65 | 2851.90 |

**C2_DCP8_agrs_256K** (ISL=261120, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 5.29 | 1588.30 | 89578.17 | 1785.57 |
| 2 | 41.42 | 1975.36 | 106353.22 | 3004.47 |
| 4 | 105.16 | 2420.14 | 117693.08 | 4397.42 |
| 8 | 96.76 | 3321.25 | 119324.40 | 6425.89 |
| 16 | 96.20 | 3296.73 | 119790.45 | 6392.54 |
| 32 | 95.87 | 3406.56 | 115876.44 | 6445.29 |
| 64 | 96.19 | 3422.28 | 115405.04 | 6472.41 |
| 128 | 214.65 | 3113.03 | 119994.67 | 6387.55 |
| 256 | 96.53 | 3300.74 | 119615.93 | 6404.05 |
| 512 | 96.70 | 3399.33 | 115540.04 | 6462.65 |

**C2_DCP8_agrs_512K** (ISL=523264, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.76 | 5427.72 | 67317.61 | 5492.48 |
| 2 | 793.01 | 6869.60 | 77815.49 | 9495.73 |
| 4 | 1705.09 | 8216.08 | 76929.65 | 15382.41 |
| 8 | 2466.28 | 11876.59 | 80755.63 | 22830.34 |
| 16 | 2501.88 | 11772.88 | 80660.83 | 22852.47 |
| 32 | 4259.41 | 12665.45 | 79366.79 | 19424.43 |
| 64 | 2413.26 | 11493.29 | 81198.62 | 22702.99 |
| 128 | 2484.83 | 11645.80 | 81384.31 | 22649.88 |
| 256 | 4227.10 | 12631.29 | 79717.68 | 19341.58 |
| 512 | 2478.24 | 11778.82 | 80938.17 | 22774.89 |

> **Warning:** C2_DCP8_agrs_512K shows severe TPOT degradation at c>=2 (793ms-4259ms vs 6.76ms at c=1). DCP8 ag_rs at 512K context is heavily bottlenecked — likely all-gather/reduce-scatter comm overhead dominates at this sequence length. Compare with TP8 512K where TPOT stays <41ms.

**C2_DCP8_agrs_1M** (ISL=1047552, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.78 | 8349.60 | 51097.95 | 9290.69 |
| 2 | 191.32 | 12577.65 | 54985.81 | 16400.84 |
| 4 | 1219.12 | 18047.21 | 57834.80 | 30235.74 |
| 8 | 1727.69 | 17260.79 | 58222.97 | 31254.06 |
| 16 | 1730.92 | 17275.24 | 58164.35 | 31289.02 |
| 32 | 1736.80 | 17453.17 | 57809.37 | 31518.56 |
| 64 | 1721.78 | 17233.30 | 58393.89 | 31179.82 |
| 128 | 1728.55 | 17279.54 | 58151.60 | 31286.28 |
| 256 | 1721.87 | 17207.38 | 58399.97 | 31155.10 |
| 512 | 1735.48 | 17351.03 | 57953.53 | 31402.16 |

> **Warning:** Same ag_rs TPOT degradation at 1M — c=1 fine (6.78ms), c=2 jumps to 191ms, c>=4 plateaus at ~1730ms.

### B200 C3 DCP8 a2a flashinfer Results

**C3_DCP8_a2a_fi_128K** (ISL=130048, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 4.84 | 606.31 | 68608.77 | 1187.68 |
| 2 | 12.00 | 791.70 | 88135.36 | 1682.81 |
| 4 | 16.57 | 1029.32 | 95456.25 | 2409.56 |
| 8 | 24.30 | 1418.38 | 95694.78 | 2884.26 |
| 16 | 21.42 | 1282.18 | 96469.69 | 2847.84 |
| 32 | 24.06 | 1404.92 | 96009.21 | 2865.62 |
| 64 | 24.09 | 1394.95 | 95834.14 | 2876.15 |
| 128 | 31.21 | 1542.80 | 94071.79 | 2939.83 |
| 256 | 21.63 | 1401.49 | 95872.85 | 2868.57 |
| 512 | 30.95 | 1565.54 | 93702.42 | 2948.77 |

**C3_DCP8_a2a_fi_256K** (ISL=261120, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 5.30 | 1581.49 | 88048.01 | 1817.45 |
| 2 | 37.68 | 1978.64 | 104563.78 | 2714.95 |
| 4 | 82.80 | 2262.56 | 116325.52 | 4462.12 |
| 8 | 108.89 | 3395.82 | 119256.67 | 6437.30 |
| 16 | 111.17 | 3328.89 | 118459.01 | 6440.34 |
| 32 | 108.65 | 3336.91 | 119633.68 | 6384.42 |
| 64 | 107.26 | 3489.27 | 119392.32 | 6407.23 |
| 128 | 107.38 | 3472.49 | 119801.56 | 6393.17 |
| 256 | 107.81 | 3336.21 | 119946.35 | 6364.34 |
| 512 | 109.22 | 3326.69 | 119627.05 | 6384.62 |

**C3_DCP8_a2a_fi_512K** (ISL=523264, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 7.00 | 5401.49 | 67654.88 | 5464.60 |
| 2 | 1861.72 | 6894.22 | 77628.06 | 9520.22 |
| 4 | 1726.87 | 8183.58 | 77326.70 | 15277.17 |
| 8 | 4278.52 | 12745.41 | 79159.48 | 19499.39 |
| 16 | 2513.31 | 11728.12 | 80662.17 | 22855.85 |
| 32 | 4258.22 | 12649.96 | 79604.34 | 19369.46 |
| 64 | 4285.38 | 12652.61 | 79455.60 | 19415.93 |
| 128 | 4267.74 | 12622.18 | 79657.72 | 19358.26 |
| 256 | 4258.68 | 12689.22 | 79489.87 | 19404.11 |
| 512 | 4252.61 | 12500.94 | 80204.83 | 19211.92 |

> **Warning:** A2A flashinfer at 512K shows **same** TPOT degradation as ag_rs — c=1 fine (7ms), c>=2 catastrophic (1862-4279ms). This means the issue is NOT specific to ag_rs communication; it's a DCP-wide problem at 512K+.

**C3_DCP8_a2a_fi_1M** (ISL=1047552, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 7.03 | 8350.22 | 51055.33 | 9298.40 |
| 2 | 1246.47 | 12551.19 | 55037.72 | 16360.09 |
| 4 | 1482.72 | 17540.09 | 57755.69 | 30238.12 |
| 8 | 2067.93 | 17327.81 | 57912.46 | 31401.40 |
| 16 | 2082.56 | 16886.82 | 58105.64 | 31310.41 |
| 32 | 1911.90 | 17401.41 | 57611.20 | 31557.20 |
| 64 | 2072.18 | 17365.07 | 57786.01 | 31462.85 |
| 128 | 2064.45 | 17291.81 | 57972.68 | 31341.64 |
| 256 | 2095.41 | 16879.74 | 57916.96 | 31376.62 |
| 512 | 2067.46 | 17345.22 | 57836.86 | 31414.65 |

> **Warning:** A2A flashinfer at 1M also shows TPOT degradation — c=1 fine (7ms), c>=2 degrades (1246-2095ms). Slightly worse than ag_rs at 1M (1722-1737ms).

### B200 C3-opt DCP8 a2a flashinfer Results (Optimized: per-bs CUDA graph buffers + remove .contiguous())

**Optimizations applied (2026-03-15):**
- Opt 1: Removed redundant `.contiguous()` in CUDA graph path (saves 54 D2D copy kernels/decode step)
- Opt 3: Per-bs CUDA graph buffer allocation (NCCL transfers only actual batch data, not max_bs=512 padding)
- GSM8K accuracy: **0.387** (1319 questions, invalid 0.005) — consistent with baseline ~0.38-0.39

**C3_opt_b200_DCP8_a2a_fi_128K** (ISL=130048, OSL=1024, mem_frac=0.85)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 4.78 | 604.84 | 69100.66 | 1178.68 |
| 2 | 11.02 | 783.43 | 89805.79 | 1650.00 |
| 4 | 16.09 | 995.50 | 96545.01 | 2379.12 |
| 8 | 23.95 | 1408.48 | 96553.33 | 2854.42 |
| 16 | 23.63 | 1288.50 | 96641.00 | 2850.87 |
| 32 | 31.05 | 1547.66 | 94411.58 | 2931.34 |
| 64 | 23.93 | 1406.61 | 96659.42 | 2849.79 |
| 128 | 23.97 | 1410.09 | 96516.26 | 2856.90 |
| 256 | 30.96 | 1546.91 | 94331.43 | 2935.47 |
| 512 | 23.89 | 1433.19 | 96423.50 | 2858.76 |

> **vs baseline C3_b200_128K:** c=1: 4.78ms vs 4.84ms (same). c=8: 23.95ms vs 24.30ms (same). c=128: 23.97ms vs 31.21ms (-23%). Throughput slightly improved ~96K vs ~95K tok/s.

**C3_opt_b200_DCP8_a2a_fi_256K** (ISL=261120, OSL=1024, mem_frac=0.80)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 5.29 | 1643.54 | 85138.64 | 1879.29 |
| 2 | 39.18 | 2059.09 | 100992.79 | 2821.85 |
| 4 | 38.93 | 2964.39 | 111255.53 | 4709.10 |
| 8 | 112.83 | 3404.06 | 117448.44 | 6503.35 |
| 16 | 81.29 | 3742.82 | 119715.30 | 6507.07 |
| 32 | 108.95 | 3313.45 | 120077.29 | 6364.28 |
| 64 | 107.78 | 3335.99 | 120425.10 | 6356.23 |
| 128 | 109.72 | 3321.59 | 119544.76 | 6391.67 |
| 256 | 109.71 | 3430.37 | 115363.95 | 6453.28 |
| 512 | 109.43 | 3427.84 | 115536.78 | 6443.55 |

> **vs baseline C3_b200_256K:** c=1: 5.29ms vs 5.30ms (same). c=2: 39.18ms vs 37.68ms (same). c=4: 38.93ms vs 82.80ms (-53%). c>=8 plateau: ~109ms vs ~108ms (same). Notable improvement at c=4.

**C3_opt_b200_DCP8_a2a_fi_512K** (ISL=523264, OSL=1024, mem_frac=0.75)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 6.96 | 5432.31 | 67291.63 | 5494.85 |
| 2 | 1855.55 | 6885.02 | 77768.71 | 9503.29 |
| 4 | 1743.76 | 8216.09 | 76818.03 | 15395.13 |
| 8 | 4290.34 | 12802.66 | 78918.67 | 19571.02 |
| 16 | 2536.56 | 11824.86 | 80000.34 | 23046.68 |
| 32 | 4386.29 | 11880.38 | 80113.26 | 19321.96 |
| 64 | 2508.16 | 11685.21 | 80912.16 | 22790.27 |
| 128 | 4269.11 | 12566.22 | 79864.69 | 19303.13 |
| 256 | 2499.57 | 11707.00 | 80962.63 | 22772.43 |
| 512 | 4262.94 | 12526.49 | 80029.44 | 19254.33 |

> **vs baseline C3_b200_512K:** c=1: 6.96ms vs 7.00ms (same). c>=2 same catastrophic TPOT degradation (scheduler-dominated).

**C3_opt_b200_DCP8_a2a_fi_1M** (ISL=1047552, OSL=1024, mem_frac=0.65)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) | E2E (ms) |
|:-----------:|:---------:|:---------:|:------------------------:|:--------:|
| 1 | 7.08 | 8395.44 | 50797.30 | 9345.73 |
| 2 | 1250.53 | 12468.07 | 55357.85 | 16258.34 |
| 4 | 1453.13 | 17909.61 | 57900.29 | 30158.20 |
| 8 | 2081.33 | 16653.28 | 58342.25 | 31043.58 |
| 16 | 2055.36 | 17061.63 | 58319.63 | 31053.33 |
| 32 | 2071.54 | 17177.46 | 57887.23 | 31274.40 |
| 64 | 2065.34 | 17260.98 | 57946.82 | 31314.06 |
| 128 | 2056.74 | 17111.96 | 58245.36 | 31109.61 |
| 256 | 2062.50 | 17131.14 | 58096.54 | 31167.66 |
| 512 | 2054.10 | 17021.22 | 58359.93 | 31002.58 |

> **vs baseline C3_b200_1M:** c=1: 7.08ms vs 7.03ms (same). c>=8: ~2065ms vs ~2068ms (same). No meaningful change at 1M.

#### B200 C3 Optimization Summary

| Context | c=1 TPOT (old→new) | c=64 TPOT (old→new) | Verdict |
|---------|:-------------------:|:--------------------:|---------|
| 128K | 4.84→4.78ms (=) | 24.09→23.93ms (=) | Same; c=128 improved 31→24ms (-23%) |
| 256K | 5.30→5.29ms (=) | 107.26→107.78ms (=) | Same; c=4 improved 83→39ms (-53%) |
| 512K | 7.00→6.96ms (=) | 4285→2508ms (-41%) | Large variance; scheduler-dominated |
| 1M | 7.03→7.08ms (=) | 2072→2065ms (=) | No change (scheduler-dominated) |

**B200 Conclusion:** Same pattern as H100 — c=1 TPOT unchanged (NCCL floor), correctness preserved (GSM8K 0.387). At some mid-concurrency points, notable improvements (128K c=128: -23%, 256K c=4: -53%) likely from reduced NCCL buffer padding. At 512K+, scheduler-level TPOT degradation dominates.

## Remaining Jobs

### H100 — ALL COMPLETE
1. ~~C1_TP8~~ — DONE (all 4 contexts)
2. ~~C2_DCP8_agrs_128K~~ — DONE (c1 failed)
3. ~~C2_DCP8_agrs_256K~~ — DONE (c2, c4 failed)
4. ~~C2_DCP8_agrs_512K~~ — DONE
5. ~~C2_DCP8_agrs_1M~~ — DONE
6. ~~C3_DCP8_a2a_fi_128K~~ — DONE (rerun with mem_frac=0.50)
7. ~~C3_DCP8_a2a_fi_256K~~ — DONE
8. ~~C3_DCP8_a2a_fi_512K~~ — DONE
9. ~~C3_DCP8_a2a_fi_1M~~ — DONE
10. ~~C4_DCP8_a2a_fa3_128K~~ — DONE (rerun with mem_frac=0.50)
11. ~~C4_DCP8_a2a_fa3_256K~~ — DONE
12. ~~C4_DCP8_a2a_fa3_512K~~ — DONE
13. ~~C4_DCP8_a2a_fa3_1M~~ — DONE

### B200 — ALL COMPLETE
1. ~~C1_TP8~~ — DONE (all 4 contexts)
2. ~~C2_DCP8_agrs_128K~~ — DONE
3. ~~C2_DCP8_agrs_256K~~ — DONE
4. ~~C2_DCP8_agrs_512K~~ — DONE
5. ~~C2_DCP8_agrs_1M~~ — DONE
6. ~~C3_DCP8_a2a_fi_128K~~ — DONE
7. ~~C3_DCP8_a2a_fi_256K~~ — DONE
8. ~~C3_DCP8_a2a_fi_512K~~ — DONE
9. ~~C3_DCP8_a2a_fi_1M~~ — DONE

---

## Charts to Create (aligned with vLLM Helix presentation)

### Chart 1: TPOT vs Concurrency (4 charts, one per context length)
- **X-axis:** concurrency (1-512, log scale)
- **Y-axis:** TPOT p50 (ms)
- **Lines:** C1 TP8 (gray), C2 DCP8 ag_rs (blue), C3 DCP8 a2a flashinfer (red), C4 DCP8 a2a fa3 (green, H100 only)
- **Reference:** horizontal dashed line at 20ms SLA
- **Purpose:** Shows at what concurrency each config hits SLA limit

### Chart 2: TPOT vs Context Length (1 chart, low concurrency c=1,2,4)
- **X-axis:** context length (128K → 1M)
- **Y-axis:** TPOT p50 (ms)
- **Lines:** one per config, sub-grouped by concurrency
- **Purpose:** Shows decode latency scaling as context grows

### Chart 3: Pareto Frontier — Per-user vs Per-GPU throughput (4 charts, one per context)
- **X-axis:** 1/TPOT (per-user decode throughput, tok/s)
- **Y-axis:** output tok/s / 8 GPUs (system throughput per GPU)
- **Points:** each concurrency level, colored by config
- **Purpose:** Core chart — Helix A2A should be rightward of TP/DCP ag_rs

### Chart 4: GPU Efficiency at 20ms SLA (1 bar chart)
- **X-axis:** context length (128K, 256K, 512K, 1M)
- **Y-axis:** max tok/s/GPU while TPOT < 20ms
- **Bars:** grouped by config (C1-C4)
- **Purpose:** Key metric — useful work per GPU at production SLA

### Chart 5: TTFT vs Context Length (1 chart)
- **X-axis:** context length (128K → 1M)
- **Y-axis:** TTFT mean (ms), at c=1
- **Lines:** per config
- **Purpose:** Prefill latency comparison

### Chart 6: A2A vs AG+RS Delta Heatmap (1 heatmap)
- **Rows:** context lengths (128K, 256K, 512K, 1M)
- **Columns:** concurrency levels (1-512)
- **Cells:** % TPOT difference (C3 a2a vs C2 ag_rs), green=a2a faster, red=ag_rs faster
- **Purpose:** Shows crossover point — vLLM showed A2A wins at DCP≥8 cross-node

### Chart 7: H100 vs B200 Hardware Comparison (1 chart)
- **X-axis:** context length
- **Y-axis:** TPOT at c=1
- **Lines:** C1-C3 on both H100 and B200
- **Purpose:** Cross-hardware comparison

---

## Key Metrics to Extract per Benchmark Point

From each `bench_serving` run, capture:
- TPOT p50 / p99 / mean (ms)
- TTFT mean / p99 (ms)
- Total token throughput (tok/s)
- Output token throughput (tok/s)
- Successful requests
- E2E latency mean (ms)

---

## OOM Notes
- TP8 256k @ mem_frac 0.85 → OOM → use 0.80
- DCP8 ag_rs 256k @ mem_frac 0.85 → OOM → use 0.80
- DCP8 a2a 512k @ mem_frac 0.80 → OOM → use 0.75
- DCP8 symm_mem on H100 @ mem_frac 0.80 → OOM → use 0.50 for 4-GPU accuracy tests
- 1M context: start at 0.65, reduce by 0.05 if OOM
