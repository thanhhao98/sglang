# Qwen3-235B symm-mem + CUDA graph hang investigation (B200)

## TL;DR

Before fix: 8/13 configs hang on `Qwen3-235B-A22B-Instruct-2507` with TP8 on
B200 (some during CUDA graph capture, some at the first prefill, some
mid-decode). The originally-reported symptom was "Capture cuda graph
hangs at `bs=144` with `--enable-symm-mem`".

After fix (branch `htphan/tpa-flashinfer`, commit `a49c50bcd`): **all 13
configs pass** `gsm8k --parallel 128 --max-new-tokens 512` on B200.

## Root causes and fixes

Four separate bugs stacked on top of each other. They reproduce
independently but their symptoms looked similar ("something hangs"):

1. **CUDA graph capture + NCCL symm-mem deadlock** (cfg0, cfg3, cfg4).
   `use_symmetric_memory()` inside the capture warmup forward passes
   triggers `ncclCommWindowRegister` (`NCCL_WIN_COLL_SYMMETRIC`) on
   every new size. That call is collective, and rank-level timing
   jitter during capture eventually hits a size where the ranks
   don't line up, deadlocking at `bs=144` consistently.
   **Fix:** `_disable_symm_mem()` context manager wrapping
   `init_device_graphs` and `init_piecewise_cuda_graphs` in
   `model_runner.py`. The captured graphs replay standard NCCL
   allreduce, which is correct and safe under symm-mem serving.

2. **Symm-mem prealloc stealing from the KV-cache budget** (cfg0 at
   first large prefill).
   `prealloc_symmetric_memory_pool` reserves 4 GiB in a separate NCCL
   mempool that the regular CUDA allocator cannot see. `mem_fraction_static`
   never accounted for it, so after CUDA graph (~5 GiB) + prealloc
   (4 GiB) the live prefill path OOMs on a 184 MiB activation.
   **Fix:** subtract `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE` (default 4 GiB)
   from `gpu_mem` in the auto `mem_fraction_static` calculation.
   Also drop the "keep prealloc tensor alive" pattern: holding the
   tensor pins the 4 GiB in the mempool's in-use list so live NCCL
   allocations can't reuse it.

3. **DCP symm-mem pool never warmed** (cfg3, cfg4, cfg5, cfg6).
   Every `GroupCoordinator` maintains its OWN NCCL mempool. The
   original prealloc only covers the TP group. The DCP group stays
   empty through server startup, so the first live request fires
   `ncclCommWindowRegister` on the DCP group under async load and
   the 2 ranks in each DCP group race on entering the collective,
   deadlocking mid-decode.
   **Fix:** stop disabling symm-mem during FlashInfer autotune — the
   autotune dummy run exercises every symm-mem group at realistic
   shapes and registers a variety of allocation sizes into each
   group's mempool before any live traffic hits it. (We keep the
   non-default-stream wrapper from upstream PR #18987.)

4. **FlashInfer TRT-LLM allreduce workspace port collision with DCP**
   (cfg8, cfg9, cfg10, cfg11, cfg12).
   `create_allreduce_fusion_workspace` opens an AF_UNIX socket at a
   fixed per-host rendezvous path. With DCP > 1 we launch multiple
   scheduler processes on the same host, they collide on the path,
   and half the ranks fail with `[Errno 98] Address already in use`
   and disable fusion permanently. The other half succeed. The split
   state (some ranks using fused AR+RMSNorm, others using standard
   NCCL) deadlocks during CUDA graph capture on Qwen3-235B.
   **Fix:** auto-disable FlashInfer allreduce fusion when `dcp_size > 1`
   (users can still opt in explicitly with
   `--enable-flashinfer-allreduce-fusion`).

## Verification on `colossus_b200_1` (Qwen3-235B, B200, `--parallel 128`)

| cfg  | Flags                                                                                          | Result                                          |
| ---- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| cfg0 | tp8 + flashinfer + `--enable-symm-mem` (graph ON)                                              | PASS — accuracy 0.980, throughput 2434 tok/s    |
| cfg1 | tp8 + flashinfer (baseline, no symm-mem)                                                       | PASS — accuracy 0.975, throughput 2523 tok/s    |
| cfg2 | tp8 + flashinfer + `--enable-symm-mem --disable-cuda-graph`                                    | PASS — accuracy 0.975, throughput 735 tok/s     |
| cfg3 | tp8 + dcp2 a2a + `--enable-symm-mem` (graph ON)                                                | PASS — accuracy 0.975, throughput 1863 tok/s    |
| cfg4 | tp8 + dcp2 ag_rs + `--enable-symm-mem` (graph ON)                                              | PASS — accuracy 0.975, throughput 1797 tok/s    |
| cfg5 | tp8 + dcp2 ag_rs + `--enable-symm-mem --disable-cuda-graph`                                    | PASS — accuracy 0.975, throughput 474 tok/s     |
| cfg6 | tp8 + dcp2 a2a + `--enable-symm-mem --disable-cuda-graph`                                      | PASS — accuracy 0.975, throughput 476 tok/s     |
| cfg7 | tp8 + dcp2 a2a (no symm-mem, graph ON)                                                         | PASS — accuracy 0.975, throughput 1794 tok/s    |
| cfg8 | tp8 + dcp2 a2a + tpa4 (no symm-mem, graph ON)                                                  | PASS — accuracy 0.970, throughput 1868 tok/s    |
| cfg9 | tp8 + dcp2 ag_rs + tpa4 (no symm-mem, graph ON)                                                | PASS — accuracy 0.980, throughput 1886 tok/s    |
| cfg10| tp8 + dcp2 ag_rs + tpa4 + `--enable-symm-mem --disable-cuda-graph`                             | PASS — accuracy 0.975, throughput 545 tok/s     |
| cfg11| tp8 + dcp2 a2a + tpa4 + `--enable-symm-mem` (graph ON)                                         | PASS — accuracy 0.970, throughput 1896 tok/s    |
| cfg12| tp8 + dcp2 ag_rs + tpa4 + `--enable-symm-mem` (graph ON)                                       | PASS — accuracy 0.970, throughput 1901 tok/s    |

All runs: `gsm8k --parallel 128 --max-new-tokens 512 --num-questions 200`
against `python3 -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-Instruct-2507`
inside the `sglang-bench` container on `colossus_b200_1`. Test harness
lives at `bench_scripts/test_symm_mem_configs.sh`.

## Commits landed in this investigation

- `e976bc849` — disable symm-mem during CUDA graph capture + autotune
- `85b7911f8` — account for symm-mem prealloc in mem_fraction_static + drop keep-alive
- `40ea74a7c` — (reverted) DCP prealloc + parallel=128 in sweep
- `e696ae60b` — keep symm-mem enabled during autotune to warm DCP pool
- `a49c50bcd` — auto-disable FlashInfer allreduce fusion when dcp_size > 1
