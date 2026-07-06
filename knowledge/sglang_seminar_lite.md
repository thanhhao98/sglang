# [ACOT] SGLang — Helix Parallelism Knowledge Sharing

A condensed reading guide for the knowledge-sharing session. Covers SGLang architecture end-to-end
and closes with a case study on the Helix parallelism implementation (MLA + GQA). For the full
version with extensive code snippets, see [`sglang_seminar.md`](./sglang_seminar.md).

All paths are under `python/sglang/srt/`.

---

## Table of contents

1. [Process topology](#1-process-topology)
2. [Attention backends (Strategy pattern)](#2-attention-backends)
3. [CUDA graphs](#3-cuda-graphs)
4. [Parallelism (focus: Tensor Parallelism)](#4-parallelism-focus-tensor-parallelism)
5. [Communication primitives](#5-communication-primitives)
6. [End-to-end request flow](#6-end-to-end-request-flow)
7. [Case study: Helix parallelism in SGLang](#7-case-study-helix-parallelism-in-sglang)

---

## 1. Process topology

SGLang is **not** a single program. It is a small constellation of processes communicating via two
completely different transports:

| Link | Transport | Used for |
|---|---|---|
| Control plane | **ZMQ IPC** (PUSH/PULL, pickled Python objects) | Requests, token IDs, decoded strings |
| Data plane | **NCCL / PyNCCL / CustomAllreduce / symm mem** (GPU tensors) | Collectives inside a forward pass |

```
                 ┌─────────────────────────────────┐
                 │ HTTP server (main proc)         │
                 │ TokenizerManager                │
                 └─────────┬────────────▲──────────┘
                     ZMQ   │            │   ZMQ
                           ▼            │
          ┌──────────────────────────────────────────────┐
          │         Scheduler processes (TP × PP)         │
          │   rank 0 ◄─── NCCL / PyNCCL / … ────► rank N  │
          └──────────────────┬───────────────────────────┘
                             │ ZMQ PUSH (token IDs only)
                             ▼
                 ┌─────────────────────────────────┐
                 │ DetokenizerManager              │
                 └───────────┬─────────────────────┘
                             │ ZMQ PUSH (decoded strings)
                             ▼
                     back to TokenizerManager → SSE
```

Key invariants:

- **ZMQ** crosses CPU process boundaries (control messages; small).
- **NCCL** runs *inside* one model forward across GPUs (hidden states; big).
- Only the entry-rank Scheduler (rank 0, last PP stage) talks to the Detokenizer.

---

## 2. Attention backends

SGLang supports ~15 attention kernels (FlashInfer, FA3, FA4, Triton, FlashMLA, TRT-LLM MLA, cuDNN,
NSA, Aiter, …) via the **Strategy pattern**.

### 2.1 The ABC

```18:55:python/sglang/srt/layers/attention/base_attn_backend.py
class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self, bs, num_tokens, req_pool_indices, seq_lens,
        encoder_lens, forward_mode, spec_info,
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self, bs, req_pool_indices, seq_lens, seq_lens_sum,
        encoder_lens, forward_mode, spec_info, seq_lens_cpu,
    ):
        """Init the metadata for a forward pass for replaying a cuda graph."""
        raise NotImplementedError()
```

### 2.2 Template Method dispatch

The base class dispatches on `ForwardMode`. Subclasses only supply `forward_decode` / `forward_extend`:

```80:123:python/sglang/srt/layers/attention/base_attn_backend.py
    @debug_kernel_api
    def forward(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)
        elif forward_batch.forward_mode.is_mixed() and is_npu():
            return self.forward_mixed(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)
        else:
            return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)
```

### 2.3 Registry + Factory

Decorator-based plugin system (Open/Closed Principle):

```20:53:python/sglang/srt/layers/attention/attention_registry.py
ATTENTION_BACKENDS = {}


def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn

    return decorator


@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    import torch

    if not runner.use_mla_backend:
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
        ...
        return FlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )
    else:
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )

        return FlashInferMLAAttnBackend(runner)
```

### 2.4 Dependency injection

The model never branches on backend; it pulls the strategy from `forward_batch`:

```python
# RadixAttention.forward
ret = forward_batch.attn_backend.forward(q, k, v, self, forward_batch, ...)
```

---

## 3. CUDA graphs

The decode path launches thousands of tiny kernels per token. SGLang captures each of them into a
CUDA graph once per batch-size bucket and replays the graph.

### 3.1 Overview

File: `model_executor/cuda_graph_runner.py`. Key methods:

| Method | Role |
|---|---|
| `__init__` | Allocate buffers, compute buckets, call `capture()` |
| `capture()` / `capture_one_batch_size()` | Record a graph per batch size |
| `can_run(forward_batch)` | Runtime guard — does this batch fit? |
| `replay_prepare()` + `replay()` | Copy inputs → refresh metadata → `graph.replay()` |

### 3.2 When graphs can be used

```169:175:python/sglang/srt/model_executor/forward_batch_info.py
    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
            or self == ForwardMode.DLLM_EXTEND
        )
```

Prefill / extend modes normally skip the graph (variable seq lens).

### 3.3 Dispatch in ModelRunner

```3115:3146:python/sglang/srt/model_executor/model_runner.py
    def _forward_raw(
        self, forward_batch, skip_attn_backend_init, pp_proxy_tensors,
        reinit_attn_backend=False, split_forward_count=1,
    ) -> ModelRunnerOutput:
        mode_check = (
            forward_batch.forward_mode.is_cpu_graph
            if self.device == "cpu"
            else forward_batch.forward_mode.is_cuda_graph
        )
        can_run_graph = bool(
            mode_check()
            and self.graph_runner
            and self.graph_runner.can_run(forward_batch)
        )
        ...
        if can_run_graph:
            ret = self.graph_runner.replay(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)
```

### 3.4 Capture order — large → small

Large buckets are captured first so small graphs can **alias** the memory pool:

```855:871:python/sglang/srt/model_executor/cuda_graph_runner.py
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
            if not self.enable_pdmux:
                with graph_capture() as graph_capture_context, profile_context as prof:
                    self.stream = graph_capture_context.stream
                    _capture_one_stream()
```

### 3.5 Replay — bucket lookup + copy inputs

```1279:1308:python/sglang/srt/model_executor/cuda_graph_runner.py
    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        self.deepep_adapter.replay()

        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
            ...
        # Replay
        if self.enable_pdmux:
            graph_key = f"{get_current_stream_idx()}_{self.bs}"
        else:
            graph_key = self.bs
        self.graphs[graph_key].replay()
        output = self.output_buffers[graph_key]
```

### 3.6 Backend graph hooks

Each `AttentionBackend` declares three graph methods:

- `init_cuda_graph_state(max_bs, max_num_tokens)` — once, pre-allocate fixed buffers.
- `init_forward_metadata_capture_cuda_graph(...)` — per bucket, during capture.
- `init_forward_metadata_replay_cuda_graph(...)` — per replay, refresh buffer contents.

---

## 4. Parallelism (focus: Tensor Parallelism)

SGLang supports TP, PP, DP, EP, and CP. This section focuses on **Tensor Parallelism (TP)** —
the most common axis, and the one that best shows the key design idea: model code never talks
to NCCL directly; it only talks to a `tp_group` abstraction.

### 4.1 What TP actually shards

- TP shards model **weights**, not inputs.
- Every TP rank sees the **same** input tokens (replicated).
- Each rank holds `1/tp_size` of every parallel weight matrix (heads or channels).
- Collective ops stitch the partial results back into full hidden states.

### 4.2 The abstraction boundary — model has no NCCL awareness

The model talks to a small, device-agnostic API. The whole model-visible vocabulary is:

```python
# communication_op.py --- thin one-line wrappers used throughout the model
def tensor_model_parallel_all_reduce(x):
    return get_tp_group().all_reduce(x)

def tensor_model_parallel_all_gather(x, dim=-1):
    return get_tp_group().all_gather(x, dim)

def tensor_model_parallel_reduce_scatter(x, dim=0):
    return get_tp_group().reduce_scatter(x, dim)
```

`get_tp_group()` returns a `GroupCoordinator` object with these methods:

| Method | What it does |
|---|---|
| `tp_group.all_reduce(x)` | Sum `x` across TP ranks; every rank ends up with the sum. |
| `tp_group.all_gather(x, dim)` | Each rank contributes a shard; every rank ends up with the concatenation. |
| `tp_group.reduce_scatter(x, dim)` | Sum across ranks then scatter; each rank keeps one slice. |
| `tp_group.broadcast(x, src)` | One rank sends `x` to every other rank. |

**That's the whole surface.** Choosing between NCCL / PyNccl / CustomAllreduce / symmetric
memory / mscclpp happens *inside* `GroupCoordinator` — the model layer is kernel-agnostic.
This is the Strategy pattern applied to communication transport.

TP ranks are consecutive global ranks:

```1872:1893:python/sglang/srt/distributed/parallel_state.py
    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = []
    for tp_group_idx in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(
                tp_group_idx * tensor_model_parallel_size,
                (tp_group_idx + 1) * tensor_model_parallel_size,
            )
        )
        group_ranks.append(ranks)

    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
        group_name="tp",
    )
```

### 4.3 Attention block — how weights are sharded, where the comm fires

Example: TP = 4, `hidden_size = 4096`, `num_heads = 32` (so 8 heads per rank).

| Step | Weight shape per rank | Output per rank | Communication |
|---|---|---|---|
| `qkv_proj = QKVParallelLinear` (column) | `[4096, 3 × 8 × head_dim]` | partial QKV (this rank's 8 heads) | **none** |
| RoPE + RadixAttention | — | attention over local 8 heads | **none** |
| `o_proj = RowParallelLinear` (row) | `[8 × head_dim, 4096]` | partial `[B, 4096]` (incomplete sum) | `tp_group.all_reduce` → full `[B, 4096]` |

- Column-parallel shards along the **output** dim, so each rank naturally produces the shard it
  needs for the next step — no comm required.
- Row-parallel shards along the **input** dim; its GEMM output is a *partial sum* over the head
  dimension. The all-reduce (hidden inside `RowParallelLinear.forward`) sums those partials
  into the full hidden state.

```165:244:python/sglang/srt/models/llama.py
        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            ...
        )
        ...
    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        ...
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output
```

No `all_reduce`, no `torch.distributed`. The model is pure numerical code; communication lives
behind `o_proj`.

### 4.4 MLP (FFN) block — same recipe

| Step | Weight shape per rank | Output per rank | Communication |
|---|---|---|---|
| `gate_up_proj = MergedColumnParallelLinear` (column) | `[4096, 2 × intermediate / tp]` | partial `[B, 2·intermediate/tp]` | **none** |
| `SiluAndMul` | — | local pointwise op | **none** |
| `down_proj = RowParallelLinear` (row) | `[intermediate / tp, 4096]` | partial `[B, 4096]` | `tp_group.all_reduce` → full `[B, 4096]` |

One attention block + one MLP block = exactly **two all-reduces per transformer layer**, both
triggered implicitly by `RowParallelLinear`.

### 4.5 Where the call actually goes

`RowParallelLinear.forward` — the one-and-only place TP all-reduces fire in a transformer:

```1492:1525:python/sglang/srt/layers/linear.py
    def forward(self, input_, skip_all_reduce=False):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        ...
        with symm_ctx:
            output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)

        if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
            if self.use_dp_attention_reduce:
                output = get_attention_tp_group().all_reduce(output_parallel)
            else:
                output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        ...
```

`tensor_model_parallel_all_reduce(x)` is a one-line wrapper — it just hands `x` to the TP group:

```python
# communication_op.py
def tensor_model_parallel_all_reduce(input_):
    return get_tp_group().all_reduce(input_)
```

Inside `GroupCoordinator.all_reduce`, SGLang picks the fastest primitive for this tensor and
this execution mode (see §5). The model layer never sees any of that.

### 4.6 Data flow through one transformer block (TP = 4)

```
 hidden (FULL, replicated on every rank)
             │
             ▼
         RMSNorm                                    (no comm, pointwise)
             │
             ▼
    QKVParallelLinear (column)                      (no comm --- partial QKV)
             │
             ▼
    RadixAttention (local heads)                    (no comm)
             │
             ▼
    o_proj = RowParallelLinear                      ★ tp_group.all_reduce()
             │
             ▼
    residual + RMSNorm                              (no comm)
             │
             ▼
  gate_up_proj = MergedColumnParallelLinear         (no comm --- partial)
             │
             ▼
         SiluAndMul                                 (no comm)
             │
             ▼
    down_proj = RowParallelLinear                   ★ tp_group.all_reduce()
             │
             ▼
 hidden (FULL, replicated on every rank)
```

Two stars, two all-reduces. For an `L`-layer model, TP inference fires `2L` all-reduces per
forward, all through the `tp_group` abstraction.

---

## 5. Communication primitives

`GroupCoordinator.all_reduce` picks the fastest implementation for the tensor + mode:

```554:631:python/sglang/srt/distributed/parallel_state.py
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        ...
        if self.pynccl_comm is not None and self.is_symmetric_memory_enabled():
            ...
            with self.pynccl_comm.change_state(enable=True):
                self.pynccl_comm.all_reduce(input_)
                return input_

        outplace_all_reduce_method = None
        if (
            self.ca_comm is not None
            and not self.ca_comm.disabled
            and self.ca_comm.should_custom_ar(input_)
        ):
            outplace_all_reduce_method = "ca"
        elif (
            self.qr_comm is not None
            and not self.qr_comm.disabled
            and self.qr_comm.should_quick_allreduce(input_)
        ):
            outplace_all_reduce_method = "qr"
        elif is_in_piecewise_cuda_graph() and self.pynccl_comm is not None:
            outplace_all_reduce_method = "pynccl"
        if outplace_all_reduce_method is not None:
            return outplace_all_reduce(
                input_,
                group_name=self.unique_name,
                outplace_all_reduce_method=outplace_all_reduce_method,
            )
        else:
            inplace_all_reduce(input_, group_name=self.unique_name)
            return input_
```

Selection menu (fastest → fallback):

1. `CustomAllreduce` — IPC one-shot / two-shot (small tensors, no graph).
2. `QuickAllreduce` — AMD symmetric memory.
3. `pymscclpp` — MSCCL++.
4. `TorchSymmMemCommunicator` — PyTorch symmetric memory.
5. `PyNcclCommunicator` — NCCL (graph-capturable).
6. `torch.distributed.all_reduce` — generic fallback.

---

## 6. End-to-end request flow

### 6.1 HTTP → TokenizerManager

```689:718:python/sglang/srt/entrypoints/http_server.py
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + dumps_json(out) + b"\n\n"
            ...
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
```

### 6.2 Scheduler loop

```1382:1407:python/sglang/srt/managers/scheduler.py
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            ...
            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
```

### 6.3 ModelRunner returns logits, Sampler returns token IDs

```python
# tp_worker.py
out = self.model_runner.forward(forward_batch, ...)
batch_result.next_token_ids = self.model_runner.sample(out.logits_output, forward_batch)
```

Output chain:

```
hidden_states ──► logits (ModelRunner) ──► token_ids (Sampler) ──► strings (Detokenizer)
  GPU tensor       GPU [B, V]              GPU/CPU List[int]       List[str]
```

### 6.4 Timeline

```
HTTP          TM               Scheduler (TP rank 0..N)         Detok
  │            │                         │                         │
  ├─GenReq───► │                         │                         │
  │            ├─tokenize                │                         │
  │            │─TokenizedReq──(ZMQ)───► │                         │
  │            │                         ├ broadcast_pyobj (gloo)  │
  │            │                         ├ ForwardBatch → forward  │
  │            │                         │   (NCCL all-reduces)    │
  │            │                         ├ Sampler → token_ids     │
  │            │                         ├─BatchTokenIDOutput─────►│
  │            │◄────BatchStrOutput──────┼─decode────────── (ZMQ)──┘
  │◄─SSE chunk─┤                         │
  ▼
```

Prefill runs once; decode loop iterates once per output token.

---

## 7. Case study: Helix parallelism in SGLang

A concrete example of how the abstractions we just covered are used in production: the
**Helix Parallelism** feature ([NVIDIA paper, July 2025](https://arxiv.org/abs/2507.07120)),
which I implemented in SGLang for both MLA and GQA models.

### 7.1 Why Helix

Standard TP shards Q heads, but when `tp_size > num_kv_heads` the KV cache must be **duplicated**
across ranks. For DeepSeek-V2 (MLA, 1 effective KV head) at TP=8 the KV cache is replicated 8×,
which erases MLA's memory savings at long context.

**Helix solution:** decouple attention parallelism from FFN parallelism.

```
tp_size = tpa_size  ×  dcp_size         with   tpa_size ≤ num_kv_heads
          └─────┬─┘   └───┬─────┘
         TP Attention    KV parallel
         (head shard)    (sequence shard, interleaved by position)
```

- During **attention**, each rank holds `1/tpa_size` of Q heads AND `1/dcp_size` of KV positions.
- During **FFN**, full TP as usual.
- **A2A (All-to-All)** exchanges partial attention outputs + LSE across DCP ranks, and an
  exact-softmax "LSE combine" stitches them into the final attention output.

Measured impact (8×H100, MLA at 1M context): **7.3 ms TPOT (Helix)** vs **25.5 ms (pure TP)** — a
71 % latency reduction.

### 7.2 Two flavors I implemented

| Flavor | Formula | When to use |
|---|---|---|
| **MLA Helix** | `dcp + a2a` (TPA=1 implicit) | MLA models (DeepSeek-V2/V3/R1) — 1 effective KV head |
| **GQA Helix** | `dcp + tpa + a2a` | GQA models (Qwen3, Llama) — multiple KV heads |

Both end up exercising almost every abstraction layer we covered in §1–§6.

### 7.3 MLA Helix — what gets modified

MLA already has one KV head, so TPA stays at 1. The work is to add **DCP + A2A**. Four SGLang
subsystems change:

**(1) Scheduler / KV allocator — tokens must be striped across DCP ranks.**
Before Helix: `TokenToKVPoolAllocator.alloc(need_size)` returns KV slots for every token. After
Helix: each rank only gets slots for tokens where `pos % dcp_size == dcp_rank`.

```python
# mem_cache/allocator.py  (simplified)
def alloc(self, need_size, token_positions=None):
    dcp_world_size = get_dcp_world_size()
    dcp_rank       = get_dcp_rank()
    if dcp_world_size > 1 and token_positions is not None:
        local_mask = (token_positions % dcp_world_size) == dcp_rank
        ...   # rank keeps only slots[local_mask]
```

The scheduler must pass `token_positions` through to the allocator so it knows which positions
each rank owns. This is the "scheduler sends correct tokens to each worker" piece.

**(2) Per-layer local page table — which KV pages exist on this rank.**
SGLang builds a small helper class that maps the **global** token positions a request uses to the
**local** page IDs this rank holds. Built once per request and reused across all layers / steps:

```
dcp_layout.py :: build_dcp_local_page_table(global_page_table, dcp_rank, dcp_size)
  → returns local_page_table + local_seqlens_per_request
```

Every attention backend (FA3, FlashInfer, …) now takes the **local** page table + seqlens
instead of the global ones. The per-rank KV shape drops from `[S, …]` to `[S/dcp_size, …]`.

**(3) New communication primitive: `dcp_a2a_lse_reduce`.**
In a DCP decode step each rank computes attention over its local KV positions only. The result
is a **partial** `[B, H, D]` tensor plus the LSE (log-sum-exp) per head needed to combine
partials correctly. The new kernel lives in a new file:

```
layers/attention/dcp_a2a.py
  dcp_a2a_lse_reduce(output, lse, dcp_group) → reduced_output
    ├─ Triton _pack_a2a_send_kernel      # permute heads + pack LSE
    ├─ dcp_group.all_to_all_single(...)  # exchange output   ★ NCCL A2A
    ├─ dcp_group.all_to_all_single(...)  # exchange LSE      ★ NCCL A2A
    └─ Triton _fused_unpack_combine_kernel  # LSE-weighted softmax merge
```

Two all-to-alls replace the AllGather(Q) + AllGather(O) + ReduceScatter used by the non-A2A
variant — half the NCCL collectives per layer.

**(4) Attention backend contract change: return `(output, lse)`.**
Normally the backend returns just `output` and reduction is implicit inside `RowParallelLinear`.
With DCP, the backend must hand back the LSE so the model layer can merge partial attentions
across DCP ranks *before* `o_proj`. FA3's `forward_decode` gets a new branch:

```python
# flashattention_backend.py (pseudo)
if use_dcp:
    out, lse = flash_attn(q, k_local, v_local, ...)   # local KV shard
    return (out, lse)          # model will call dcp_a2a_lse_reduce
else:
    return flash_attn(...)     # old path
```

The **model** (`forward_mla.py`) then does the all-gather of Q across DCP ranks, calls the new
backend path, and finishes with `dcp_a2a_lse_reduce` before `o_proj`. Everything else —
CUDA graph, sampling, communication abstraction — is unchanged.

### 7.4 GQA Helix — what additionally gets modified

GQA has multiple KV heads, so we can also shard heads across a **separate**
`attention_tp_group` that is *smaller* than the full TP group. This is **TPA**
(Tensor Parallel Attention). Constraint: `tp_size = tpa_size × dcp_size`, and `tpa_size ≤ num_kv_heads`.

**(1) Everything from §7.3** — DCP scheduler striping + A2A is still required.

**(2) New process group: `attention_tp_group`.** In `parallel_state.py` we build a new
sub-group alongside `_TP`:

```
_ATTN_TP   # size = tpa_size,  ranks: (dcp_rank, tp_rank * dcp_size + dcp_rank)
```

Accessors added: `get_attention_tp_group()`, `get_attention_tp_rank()`, `get_attention_tp_size()`.

**(3) Weight loading with TPA.** The attention projections (`qkv_proj`, `o_proj`) now shard
across `tpa_size` ranks instead of `tp_size`. The MLP weights (`gate_up_proj`, `down_proj`) keep
full-TP sharding. In `layers/linear.py`, `ColumnParallelLinear` / `RowParallelLinear` gained
optional `tp_rank` and `tp_size` arguments so the attention layers can inject the TPA group:

```python
# In Qwen3MoeAttention (GQA model)
self.qkv_proj = QKVParallelLinear(
    hidden_size, head_dim, num_heads, num_kv_heads,
    tp_rank=get_attention_tp_rank(),          # ← attention group
    tp_size=get_attention_tp_size(),
)
self.o_proj = RowParallelLinear(
    ..., tp_rank=get_attention_tp_rank(),
         tp_size=get_attention_tp_size(),
         reduce_results=False,                 # ← don't all-reduce here
)
```

Reason: `o_proj`'s natural all-reduce is over the **TPA** group, but the next layer needs the
tensor reduced over the **full TP** group. We disable the small all-reduce and let the
layer-communicator handle the wider one — the "**TPA → full-TP handoff**".

**(4) TPA → full-TP handoff.** After `o_proj`, attention output is TPA-sharded (head dim = `H /
tpa_size`). MLP wants full-TP layout (head dim = `H / tp_size`). A thin helper
`_apply_output_head_partition` picks the right head slice for the current rank, and the
communicator does a full-TP all-reduce that also covers the residual add.

```
layers/communicator.py :: LayerCommunicator
  if use_full_tp_attention_handoff:
      # o_proj did NOT all-reduce → do it here over the full TP group
      tensor_model_parallel_all_reduce(hidden_states)
```

**(5) A2A still handles KV-parallel reduction.** The DCP dimension is orthogonal to TPA. Each
rank runs attention on its `(tpa_slice_of_heads, dcp_slice_of_KV)` 2-D tile; `dcp_a2a_lse_reduce`
combines along the DCP axis, the full-TP all-reduce combines along the TPA axis.

### 7.5 Where each abstraction from the seminar shows up

Helix is a great recap because it touches almost every layer we introduced:

| Seminar section | What Helix changes |
|---|---|
| §1 Process topology | Same ZMQ control plane; additional NCCL sub-group (`_DCP`, `_ATTN_TP`). |
| §2 Attention backend | Backend gains a branch that returns `(output, lse)` instead of just `output`. |
| §3 CUDA graphs | A2A was made graph-safe via PyNCCL `ncclGroupStart/End` + symmetric memory. |
| §4 Parallelism / `tp_group` abstraction | New `dcp_group` and `attn_tp_group` with exactly the same `.all_reduce / .all_to_all / .all_gather` interface — **no new transport primitives**. |
| §4 `RowParallelLinear` | Gains `tp_rank`, `tp_size`, `reduce_results` parameters to target either the TPA group or the full TP group. |
| §5 Communication primitives | A2A added to `GroupCoordinator`; PyNCCL-path used for graph-safety. |
| §6 End-to-end flow | Scheduler tags requests with a per-rank token mask; everything downstream respects the striping. |

### 7.6 Attention block under GQA Helix (TP = 8, TPA = 4, DCP = 2)

```
 hidden (FULL, replicated on all 8 ranks)
     │
     ▼
 qkv_proj = QKVParallelLinear(tp_size=tpa=4)      (no comm; partial QKV, TPA-sharded)
     │
     ▼
 Q: all_gather over DCP group                     ★ dcp_group.all_gather(Q)
     │
     ▼
 FlashAttention on LOCAL KV shard                 (per-rank: tpa_heads × (seq/dcp) KV positions)
     │   returns (partial_O, LSE)
     ▼
 dcp_a2a_lse_reduce                               ★ dcp_group.all_to_all  ×2  (output + LSE)
     │                                            + Triton LSE-weighted merge
     ▼
 o_proj = RowParallelLinear(tp_size=tpa=4,         (no comm; reduce_results=False)
                            reduce_results=False)
     │
     ▼
 LayerCommunicator (TPA → full-TP handoff)         ★ tp_group.all_reduce (full TP)
     │
     ▼
 MLP (full TP) ... → next layer
```

Three types of collective per attention block:

- `dcp_group.all_gather(Q)` — Q replication across DCP ranks.
- `dcp_group.all_to_all(×2)` via `dcp_a2a_lse_reduce` — partial attention + LSE exchange.
- `tp_group.all_reduce` — the classic full-TP reduce after `o_proj` / inside the communicator.

MLA Helix is the same picture with `tpa_size = 1`, so `qkv_proj` / `o_proj` are effectively
replicated (no TPA sharding) and only the DCP collectives are new.

### 7.7 Results snapshot

| Model / context | Config | TPOT |
|---|---|---|
| DeepSeek-V2 (MLA), 1 M ctx, 8×H100 | pure TP=8 | 25.5 ms |
| DeepSeek-V2 (MLA), 1 M ctx, 8×H100 | `dcp=8, a2a` (Helix) | **7.3 ms** (−71 %) |
| Qwen3-235B (GQA), 32 k / 4 k, cc=256, 8×B200 | TP=8 FlashInfer | 44.0 ms |
| Qwen3-235B (GQA), 32 k / 4 k, cc=256, 8×B200 | `tpa=4, dcp=2, a2a` (Helix) | **67.8 ms** at higher throughput |
| CodeQwen-7B (GQA), 32 k / 4 k, cc=256, 8×B200 | TP=8 | 31.0 ms @ 7.8 k tok/s |
| CodeQwen-7B (GQA), 32 k / 4 k, cc=256, 8×B200 | `tpa=4, dcp=2, a2a` (Helix) | **20.2 ms @ 11.6 k tok/s** |

At medium-to-high concurrency on long context, Helix is the Pareto-best configuration — it
beats plain DCP (one fewer NCCL call per layer) and beats TPA<KV (fully utilizes attention
parallelism).

### 7.8 Takeaway

Helix is a "layered feature": it was implementable precisely because SGLang's design respects
the Strategy / abstraction boundaries we walked through earlier. Every collective goes through a
`GroupCoordinator`; every linear layer accepts an optional parallel-group override; every
attention backend conforms to the same ABC. Adding DCP + TPA + A2A didn't require a redesign —
it required a **new group**, a **new kernel**, and one **new branch** in the attention backend
and the MLA/GQA model code. PRs merged:
[#21637 (MLA Helix)](https://github.com/sgl-project/sglang/pull/21637) and
GQA follow-up in review.

---

## Cheat sheet — top files to open

| Topic | File |
|---|---|
| Attention ABC | `layers/attention/base_attn_backend.py` |
| Backend registry | `layers/attention/attention_registry.py` |
| FA3/FA4 | `layers/attention/flashattention_backend.py` |
| FlashInfer | `layers/attention/flashinfer_backend.py` |
| Forward modes | `model_executor/forward_batch_info.py` |
| CUDA graph runner | `model_executor/cuda_graph_runner.py` |
| Parallel linears | `layers/linear.py` |
| Parallel groups | `distributed/parallel_state.py` |
| NCCL / PyNccl | `distributed/device_communicators/pynccl.py` |
| Scheduler loop | `managers/scheduler.py` |
| PP mixin | `managers/scheduler_pp_mixin.py` |
| DP controller | `managers/data_parallel_controller.py` |
| TP worker | `managers/tp_worker.py` |
| Model runner | `model_executor/model_runner.py` |
| KV pool | `mem_cache/memory_pool.py` |
| MoE layer | `layers/moe/fused_moe_triton/layer.py` |
| HTTP server | `entrypoints/http_server.py` |
| TokenizerManager | `managers/tokenizer_manager.py` |
| DetokenizerManager | `managers/detokenizer_manager.py` |
| Engine launcher | `entrypoints/engine.py` |
| Llama model | `models/llama.py` |
| **Helix: DCP A2A kernel** | `layers/attention/dcp_a2a.py` |
| **Helix: DCP local page table** | `layers/attention/dcp_layout.py` |
| **Helix: KV striping** | `mem_cache/allocator.py` (`TokenToKVPoolAllocator.alloc`) |
| **Helix: DCP / TPA groups** | `distributed/parallel_state.py` (`_DCP`, `_ATTN_TP`) |
| **Helix: Layer communicator handoff** | `layers/communicator.py` (`use_full_tp_attention_handoff`) |
| **Helix: MLA forward** | `models/deepseek_common/attention_forward_methods/forward_mla.py` |
| **Helix: GQA model wiring** | `models/qwen3_moe.py`, `models/qwen2.py` |
