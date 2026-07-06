# [Bug Fix] CUDA graph capture vs NCCL symm-mem deadlock

**Update (revised approach):** Instead of disabling symm-mem for the *entire* capture (which makes replay use plain NCCL for *all* captured allocations), the implementation now:

1. Adds `use_symmetric_memory(..., allow_under_graph_capture=True)` for **`RowParallelLinear` only** — the hot TP matmul path uses the symm pool during capture/replay when pre-warmed.
2. Leaves the default `allow_under_graph_capture=False` everywhere else (MoE combine, top-k, FlashInfer MoE runners, etc.) so those allocations **do not** enter collective `ncclCommWindowRegister` mid-capture when ranks diverge.
3. Runs **`ModelRunner._prewarm_row_parallel_symm_mem`** before decode / piecewise capture: for every capture token length and every distinct `RowParallelLinear.output_size` in the loaded model, allocates under symm with **explicit barriers** so the NCCL pool is primed before capture warmups.

MoE-heavy paths in the captured graph still replay without symm-mem allocations (same as the old global-disable behavior for those ops); **RowParallel** replays with symm-mem when enabled.

---

## 1. The issue (original write-up below)

Launching `Qwen3-235B-A22B-Instruct-2507` on 8×B200 with `--enable-symm-mem` deterministically hangs during CUDA graph capture. The server never reaches `fired up and ready to roll`; it gets stuck at exactly `bs=144`, step 30 of 52 in the capture progress bar.

The hang is silent — no Python exception, no NCCL error log, no watchdog timeout (until much later). The 8 scheduler processes sit at 99 % CPU forever.

### Reproduction

* **Hardware:** 8× B200 (also reproduces on H100 with the right model)
* **Image:** `lmsysorg/sglang:latest` (sglang `0.5.10.post1`, source commit `7c35342`)
* **Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507`

```bash
docker run -d --gpus all --shm-size 32g --network host \
    --ulimit memlock=-1 --init --name sglang-bench-main \
    -v <hf-cache>:/root/.cache/huggingface \
    --entrypoint sleep lmsysorg/sglang:latest infinity

docker exec sglang-bench-main python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --tp 8 --attention-backend flashinfer --enable-symm-mem \
    --port 30000
```

Observed log (truncated):

```
Capturing batches (bs=152 avail_mem=15.10 GB):  58%|██▌| 30/52 [00:07<00:05]
Capturing batches (bs=144 avail_mem=15.07 GB):  58%|██▌| 30/52 [00:07<00:05]
                                                       ^^^^^^^^^^^^^^^^^^^^^
                                                        progress freezes here
                                                        for hours
```

Smaller models (7B / 8B class) complete capture before the deadlock window opens, which is why this issue does not show up in the per-commit CI suites.

## 2. Root cause and findings

`use_symmetric_memory()` triggers an NCCL collective — `ncclMemAlloc` + `ncclCommWindowRegister` with `NCCL_WIN_COLL_SYMMETRIC` — that **requires every rank in the communicator to call it with the same size and in the same order**.

During CUDA graph capture, each rank runs its own warmup forward pass per captured `bs` (2 warmups + 1 capture × 52 batch sizes × 94 model layers ⇒ thousands of layer-forwards per rank). Per-rank execution drift (MoE expert routing variance, kernel heuristic picks, launch-overhead jitter) accumulates. By the 31st captured `bs`, ranks have drifted enough that they request different allocation sizes for the same capture step, and `ncclCommWindowRegister` deadlocks.

We collected four independent pieces of evidence on `colossus_b200_1` to confirm this is a rank-divergence collective deadlock, **not** a swallowed allocator OOM (which the deterministic `bs=144` could otherwise look like):

### Finding #1 — `py-spy` stacks show ranks at different Python sites

Captured at the hang point. All 8 ranks are inside the same `cuda_graph_runner.run_once` call, but at three different functions in `qwen3_moe.py`:

| Rank | Stuck at | Layer phase |
|------|----------|-------------|
| TP0 | `apply_rope_inplace_with_kvcache` (qwen3_moe.py:632) | attention prep — RoPE |
| TP1, TP2, TP3, TP4, TP6 | `trtllm_bf16_moe_op` (qwen3_moe.py:326) | MoE expert dispatch |
| TP5, TP7 | `fused_inplace_qknorm` (qwen3_moe.py:624) | attention prep — QK norm |

Five ranks are already in MoE while two are still in QK norm and one is between (RoPE). They are each waiting on a collective the other ranks will never enter.

### Finding #2 — NCCL produces no allocator-failure messages

Re-ran with `NCCL_DEBUG=WARN NCCL_DEBUG_SUBSYS=COLL,ALLOC,REG`. NCCL itself prints nothing about a failed `ncclMemAlloc`. If this were a swallowed OOM, NCCL would have logged it.

### Finding #3 — Free CUDA memory is irrelevant

Re-ran with `--mem-fraction-static 0.5` (vs default ~0.85). At the hang point, available CUDA memory is **78.16 GB** instead of 15.07 GB:

```
default mem_fraction:    Capturing batches (bs=144 avail_mem=15.07 GB): 30/52 ← HANG
--mem-fraction-static 0.5: Capturing batches (bs=144 avail_mem=78.16 GB): 30/52 ← HANG
```

5× more headroom does not change the hang at all → the regular CUDA allocator is not the bottleneck.

### Finding #4 — Disabling the NCCL prealloc pool changes nothing

Re-ran with `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE=0` (the default 4 GiB symm-mem prealloc is skipped entirely):

```
Capturing batches (bs=144 avail_mem=13.68 GB): 30/52 ← HANG
```

Same `bs=144`, same step 30/52. The symm-mem prealloc pool being exhausted is not the cause either.

### Why exactly `bs=144`?

`cuda_graph_bs` is iterated **in reverse**, so `bs=512` is iteration 1 and `bs=144` is iteration 31. The trigger is iteration count, not the `bs` value: same hardware (B200) + same model (Qwen3-235B with same MoE expert routing) + same kernel heuristics ⇒ cumulative inter-rank drift crosses the `NCCL_WIN_COLL_SYMMETRIC` size-uniformity threshold at exactly the 31st capture every time.

## 3. The fix

`python/sglang/srt/model_executor/model_runner.py` (+53 / −13):

1. New `ModelRunner._disable_symm_mem()` `@contextmanager`. It flips `server_args.enable_symm_mem` to `False` for the wrapped block. That makes the global `is_symmetric_memory_enabled()` return `False`, which makes every `use_symmetric_memory(...)` site short-circuit to `nullcontext()` and skip the collective `ncclMemAlloc` / `ncclCommWindowRegister`. The flag is restored on both normal exit and exception (so a transient capture-time error cannot silently disable symm-mem for live serving).
2. Wrap `init_device_graphs()` (decode CUDA graph capture) and `init_piecewise_cuda_graphs()` (piecewise prefill graph capture) with `self._disable_symm_mem()`.

The captured graph uses the standard NCCL allreduce path, which is safe to **replay** even after symm-mem is re-enabled for live serving traffic — replays do not call `ncclCommWindowRegister`, so they do not need rank-synchronized symm-mem registration.

`test/registered/unit/model_executor/test_disable_symm_mem.py` (+85, new): CPU-only `CustomTestCase` registered to `stage-a-test-cpu`. Two real-logic tests:
* `test_short_circuits_is_symmetric_memory_enabled` — patches the global server-args getter and asserts the predicate `use_symmetric_memory(...)` reads flips inside the helper.
* `test_restores_enable_symm_mem_when_block_raises` — guarantees a transient capture exception cannot silently disable symm-mem for the whole serving session.

## 4. Results after the fix

Same command as the failing repro, no other server-arg changes:

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| CUDA graph capture | **Hangs at `bs=144`** (30/52, indefinite) | Completes 52/52 in **12.4 s** |
| `Capture cuda graph end` mem usage | n/a (never reaches) | 2.52 GB |
| Server status | Never reaches `fired up and ready to roll` | `INFO: The server is fired up and ready to roll!` |
| GSM8K accuracy (50 q, parallel 32, max_new=256) | n/a | **0.92 – 0.94** (across runs, within natural variance) |
| Output throughput (sanity) | n/a | 1122 tok/s |

### OOM behavior is now meaningful

A separate concern raised in review: if memory is genuinely insufficient, do we still get a clean error or do we hang? Verified by forcing a tight budget with the fix applied:

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --tp 8 --attention-backend flashinfer --enable-symm-mem \
    --mem-fraction-static 0.25 --port 30000
```

→ Server fails fast on every rank with:

```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

No silent hang, actionable message. (Before this fix, the symm-mem collective hang during capture could mask a real OOM.)

### Lint and tests

* `pre-commit run --all-files` — clean (`isort` + `black` + `ruff F401,F821` + `codespell`).
* `python3 test/registered/unit/model_executor/test_disable_symm_mem.py` — 2 / 2 pass on CPU.

## Checklist

- [x] Format your code according to [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation. *(Not user-facing — `_disable_symm_mem` is private; from the user's perspective `--enable-symm-mem` simply works again.)*
- [x] Provide accuracy and speed benchmark results. *(See "Results after the fix" above.)*
- [x] Follow the SGLang code style guidance.

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with comments (`/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`) or contact authorized users to do so.
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
