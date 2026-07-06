# SGLang Architecture — A College Seminar Companion

This document explains how SGLang is built, from the HTTP request that a user sends, down to the
CUDA kernels that run on each GPU. It is structured as a reading guide: every section points to the
actual code, with file paths and line ranges you can open in your editor while presenting.

All paths are rooted in the repository `python/sglang/srt/` unless stated otherwise.

---

## Table of contents

1. [Big picture: what SGLang is](#1-big-picture-what-sglang-is)
2. [Process topology — who runs where](#2-process-topology--who-runs-where)
3. [Attention backend abstraction (FlashInfer / FA3 / FA4 / Triton / …)](#3-attention-backend-abstraction)
4. [CUDA graphs — capture, replay, and batch-size bucketing](#4-cuda-graphs)
5. [Parallelism — how a single model becomes many GPU processes](#5-parallelism)
   - 5.1 [Distributed initialization (NCCL + gloo)](#51-distributed-initialization)
   - 5.2 [Process groups: TP, PP, DP, CP, EP](#52-process-groups-tp-pp-dp-cp-ep)
   - 5.3 [Model loading and weight sharding](#53-model-loading-and-weight-sharding)
   - 5.4 [Input preparation per rank](#54-input-preparation-per-rank)
   - 5.5 [KV cache per process](#55-kv-cache-per-process)
   - 5.6 [Inter-layer data flow (attn → o_proj → MLP → attn)](#56-inter-layer-data-flow)
   - 5.7 [Pipeline parallelism send/recv](#57-pipeline-parallelism)
   - 5.8 [Expert parallelism (MoE all-to-all)](#58-expert-parallelism)
6. [Communication primitives inside a step](#6-communication-primitives-inside-a-step)
7. [End-to-end request flow — HTTP → tokens → SSE stream](#7-end-to-end-request-flow)
8. [Cheat sheet — files to open during the talk](#8-cheat-sheet)

---

## 1. Big picture: what SGLang is

SGLang is an LLM serving system. It is a pipeline of specialized processes communicating over
ZMQ and NCCL, built on top of PyTorch. The main abstractions you will meet are:

- **Engine / HTTP server** — a FastAPI process that validates requests and forwards them to the
  tokenizer manager.
- **TokenizerManager** — turns strings into token IDs and tracks streaming outputs per request id.
- **Scheduler** (one per TP×PP rank) — owns the GPU; batches requests, runs the model, pushes
  tokens back.
- **DetokenizerManager** — converts token IDs back to strings incrementally for streaming.
- **ModelRunner** — holds the PyTorch model, the KV cache, the attention backend, and the CUDA
  graph runner on one GPU.

The rest of the document walks through each of those boxes.

---

## 2. Process topology — who runs where

SGLang launches as a group of processes, not a single program. Entry point `launch_server.py`
builds an `Engine` which spawns:

- 1 HTTP server (main process, FastAPI + TokenizerManager).
- `pp_size × tp_size` scheduler subprocesses, each pinned to a GPU.
- 1 DetokenizerManager subprocess.

```534:587:python/sglang/srt/entrypoints/engine.py
        scheduler_procs = []

        if server_args.dp_size == 1:
            # Launch tensor parallel scheduler processes
            memory_saver_adapter = TorchMemorySaverAdapter.create(
                enable=server_args.enable_memory_saver
            )
            scheduler_pipe_readers = []
            ...
            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    reader, writer = mp.Pipe(duplex=False)
                    ...
                    with maybe_reindex_device_id(gpu_id) as gpu_id:
                        proc = mp.Process(
                            target=run_scheduler_process_func,
                            args=(
                                server_args, port_args, gpu_id,
                                tp_rank, attn_cp_rank, moe_dp_rank,
                                moe_ep_rank, pp_rank, None, writer,
                            ),
                        )
                        ...
                        proc.start()
```

There are two kinds of links between processes, and they are used for very different purposes:

| Link | Transport | Direction | Payload | Used for |
|---|---|---|---|---|
| Control plane | **ZMQ IPC** (PUSH/PULL) | between CPU processes | pickled Python objects | request routing, token IDs, decoded strings |
| Data plane | **NCCL** (or PyNCCL / CustomAllreduce / symm mem) | between GPU scheduler processes in the same TP/PP/MoE group | GPU tensors (hidden states, KV, experts) | collective ops inside one model forward |

The data plane is **not** drawn between Scheduler and DetokenizerManager — it only exists
between the multiple Scheduler subprocesses that share the same model:

```
                      ┌───────────────────────────────┐
                      │  HTTP server (main proc)      │
                      │  TokenizerManager             │
                      └─────────┬───────────▲─────────┘
                  ZMQ PUSH      │           │      ZMQ PULL
               scheduler_input  │           │      tokenizer_ipc
                                ▼           │
  ┌─────────────────────────────────────────────────────────────────┐
  │                    Scheduler processes (TP × PP)                │
  │                                                                 │
  │   ┌──────────┐   ◄───── NCCL / PyNCCL / CustomAllreduce ────►   │
  │   │ rank 0   │   (all-reduce inside RowParallelLinear,          │
  │   │  GPU 0   │    all-to-all for MoE, PP send/recv, ...)        │
  │   └──────────┘                                                  │
  │   ┌──────────┐                                                  │
  │   │ rank 1   │   ◄──────────► rank 2 ◄──────────► rank N        │
  │   │  GPU 1   │                                                  │
  │   └──────────┘                                                  │
  │        │                                                        │
  │        │ only rank 0 (the "entry rank") talks to ZMQ            │
  │        ▼                                                        │
  └────────┬────────────────────────────────────────────────────────┘
           │ ZMQ PUSH
           │ detokenizer_ipc
           ▼
  ┌───────────────────────────────┐
  │  DetokenizerManager           │
  │  (HF tokenizer decode)        │
  └───────────┬───────────────────┘
              │ ZMQ PUSH
              │ tokenizer_ipc
              ▼
     (back to TokenizerManager and then SSE to the HTTP client)
```

Two invariants to remember:

1. **Every link that crosses a process boundary on CPU is ZMQ IPC.** Requests, token IDs, and
   decoded strings are Python objects and ride ZMQ.
2. **Every link that exchanges GPU tensors during a model forward is NCCL** (or one of its
   faster cousins). These never leave the Scheduler cluster.

All scheduler processes in the same TP group rendezvous through `torch.distributed` and
collectively execute one forward pass per scheduling iteration; the data plane we will see in
§5 and §6 lives entirely inside that dashed "Scheduler processes" region above.

---

## 3. Attention backend abstraction

SGLang supports ~15 attention kernels (FlashInfer, FlashAttention 3, FlashAttention 4, Triton,
FlashMLA, TRT-LLM MLA, cuDNN, NSA, Aiter, Wave, Ascend NPU, Intel AMX/XPU, etc.). The design is a
**registry of backends that all implement the same small `AttentionBackend` ABC**.

### 3.1 The ABC

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

Only one method is truly abstract: `init_forward_metadata`. Everything else has a reasonable
default. The central trick is that `forward()` is provided by the ABC and dispatches on
`ForwardMode`:

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

So each concrete backend only has to implement:

- `init_forward_metadata(forward_batch)` — build whatever indptr / page-table / seqlen tensors the
  kernel needs, once per model forward.
- `forward_decode(...)` and/or `forward_extend(...)` — call the actual CUDA kernel.
- Optionally the three `*cuda_graph*` hooks if the backend supports graph capture.

### 3.2 Backend registry and selection

Backends are registered with a decorator. You can grep for `@register_attention_backend` to see
every available backend:

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

FA3 and FA4 share the same `FlashAttentionBackend` class and differ only in which kernel library
they import:

```194:215:python/sglang/srt/layers/attention/flashattention_backend.py
        self.fa_impl_ver = fa_impl_ver
        if self.fa_impl_ver == 3:
            from sgl_kernel.flash_attn import (
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
                get_scheduler_metadata,
            )

            self._get_scheduler_metadata = get_scheduler_metadata
        elif self.fa_impl_ver == 4:
            from sglang.jit_kernel.flash_attention_v4 import (
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
            )

            self._get_scheduler_metadata = None
        else:
            raise ValueError(f"Invalid version: {self.fa_impl_ver=}")
```

So `--attention-backend fa3` vs `--attention-backend fa4` boils down to a single flag that changes
one `import` path. The same class handles both, plus CUDA graph, plus MLA, plus speculative
decoding.

### 3.3 Prefill/decode can use different backends

This is one of SGLang's interesting design choices: you can pick different kernels for prefill
(`forward_extend`) and decode (`forward_decode`). `HybridAttnBackend` just holds two children and
routes by `ForwardMode`:

```13:51:python/sglang/srt/layers/attention/hybrid_attn_backend.py
class HybridAttnBackend(AttentionBackend):
    """Support different backends for prefill and decode."""

    def __init__(
        self, model_runner: ModelRunner,
        prefill_backend: AttentionBackend,
        decode_backend: AttentionBackend,
    ):
        self.model_runner = model_runner
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend
        self.data_type = model_runner.kv_cache_dtype

    def _select_backend(self, forward_mode: ForwardMode) -> AttentionBackend:
        if forward_mode.is_decode_or_idle():
            return self.decode_backend
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            return (
                self.decode_backend
                if self.model_runner.server_args.speculative_attention_mode == "decode"
                else self.prefill_backend
            )
        else:
            return self.prefill_backend
```

The command-line flags are `--prefill-attention-backend` and `--decode-attention-backend`
(both default to `--attention-backend` if unset). This is how, for example, production DCP
deployments run FlashInfer for prefill and FlashAttention 3 for decode simultaneously.

### 3.4 Where the model calls into the backend

Every attention layer uses `RadixAttention`. In its forward, it simply asks the `ForwardBatch`
for the active backend and delegates:

```133:170:python/sglang/srt/layers/radix_attention.py
    def forward(
        self, q, k, v, forward_batch: ForwardBatch,
        save_kv_cache: bool = True, **kwargs,
    ):
        if k is not None:
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            # piecewise-compile path: write into a preallocated output
            ...
            unified_attention_with_output(
                q, k, v, output, save_kv_cache, self.layer_id, **kwargs
            )
            return self._maybe_slice_output_heads(output)
        else:
            ret = forward_batch.attn_backend.forward(
                q, k, v, self, forward_batch, save_kv_cache, **kwargs,
            )
            return self._maybe_slice_output_heads(ret)
```

Key observation: the model code has **no** `if flashinfer ... elif fa3 ...` branches anywhere.
Everything is dispatched through the abstract `AttentionBackend.forward(...)` and the concrete
object stored in `forward_batch.attn_backend`.

### 3.5 How the backend is created

Backend selection lives in `ModelRunner` (`python/sglang/srt/model_executor/model_runner.py`,
function `_get_attention_backend`, ~lines 2094–2165):

1. Read `(prefill, decode) = ServerArgs.get_attention_backends()`.
2. If the two strings differ, wrap them in `HybridAttnBackend`.
3. Otherwise look up `ATTENTION_BACKENDS[name]` in the registry.
4. Optionally wrap in `TboAttnBackend` (two-batch overlap) or the PD-multiplex decode group.

After this, `self.attn_backend` is stored on `ModelRunner` and copied into every `ForwardBatch`
created for that runner.

---

## 4. CUDA graphs

LLM decoding is launch-bound: you fire thousands of tiny kernels per token. CUDA graphs capture
the entire sequence of kernels once and replay it with a single driver call. SGLang has a
`CudaGraphRunner` that does this for decode batches.

### 4.1 The runner

File: `python/sglang/srt/model_executor/cuda_graph_runner.py`.
Key methods (approximate line ranges):

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | 527–688 | Allocates buffers, computes bucket list, **calls `self.capture()`** at the end. |
| `get_batch_sizes_to_capture` | 474–508 | Which batch sizes to capture. |
| `can_run` | 699–768 | Runtime guard — can this `ForwardBatch` use a captured graph? |
| `capture` / `capture_one_batch_size` / `_capture_graph` | 794–1164 | Actual capture loop. |
| `replay_prepare` / `replay` | 1198–1338 | Set inputs and replay. |

### 4.2 Batch-size buckets (padding)

A CUDA graph is captured for a fixed batch size, so SGLang captures a list of buckets and pads
the actual batch up to the next bucket at replay time.

Bucket list (default): 1, 2, 4, 8, 12, 16, 24, 32, …, up to `--cuda-graph-max-bs`. With
`--disable-cuda-graph-padding`, every integer up to `max_bs` is captured. See
`ServerArgs._generate_cuda_graph_batch_sizes` at `server_args.py:1435-1465`.

Bucket lookup at replay:

```1209:1222:python/sglang/srt/model_executor/cuda_graph_runner.py
        # Find the right bucket via bisect.
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        ...
        self.bs = self.capture_bs[index]
```

### 4.3 Capture loop

```855:871:python/sglang/srt/model_executor/cuda_graph_runner.py
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
            if not self.enable_pdmux:
                with graph_capture() as graph_capture_context, profile_context as prof:
                    self.stream = graph_capture_context.stream
                    _capture_one_stream()
            else:
                set_pdmux_status(False)
                for i, sg in enumerate(self.stream_groups):
                    with graph_capture(
                        stream=sg[1]
                    ) as graph_capture_context, profile_context as prof:
                        self.stream = graph_capture_context.stream
                        _capture_one_stream(i)
```

Two observations:

- **Reverse order** (large → small) so small graphs can *alias* the memory pool allocated for the
  large ones — this is critical to fitting dozens of graphs in GPU memory.
- `graph_capture()` is a context manager defined in `distributed/parallel_state.py` that puts the
  TP / PP / DCP / MoE groups into "graph capture mode" so that collective ops (PyNCCL all-reduce)
  are safe to record.

For each bucket, `capture_one_batch_size`:

1. Builds a fake `ForwardBatch` from sliced buffers (`DecodeInputBuffers`).
2. Calls `attn_backend.init_forward_metadata_capture_cuda_graph(...)` so the backend fills in
   graph-safe metadata pointers.
3. Warms up `run_once()` twice (so all autotuning / lazy init happens eagerly).
4. Records the graph via `torch.cuda.CUDAGraph.capture_begin/capture_end`.

### 4.4 Replay path

`ModelRunner._forward_raw` checks whether the current batch fits the graph:

```3123:3146:python/sglang/srt/model_executor/model_runner.py
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

`ForwardMode.is_cuda_graph()` restricts graph usage to a small set of modes:

```169:175:python/sglang/srt/model_executor/forward_batch_info.py
    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
            or self == ForwardMode.DLLM_EXTEND
        )
```

Prefill/extend normally does **not** use CUDA graphs (it has variable seq-len kernels). SGLang
does provide a `PiecewiseCudaGraphRunner` that captures smaller subgraphs around the
variable-length attention kernel for some extend workloads, but the primary runner above is the
decode path.

Replay itself is three steps:

```1279:1308:python/sglang/srt/model_executor/cuda_graph_runner.py
    def replay(
        self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None,
    ):
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

`replay_prepare` copies the new input ids / positions / seq_lens into the **same tensor
storage** that was captured (`self.buffers.input_ids`, etc.) and then asks the attention backend
to refresh its graph-safe metadata via `init_forward_metadata_replay_cuda_graph`.

### 4.5 Why backends need three graph hooks

Recall the three methods on every backend:

- `init_cuda_graph_state(max_bs, max_num_tokens)` — once, at `CudaGraphRunner.__init__` time.
  Pre-allocates fixed-size buffers (e.g. FlashInfer's `cuda_graph_kv_indices`).
- `init_forward_metadata_capture_cuda_graph(...)` — once per bucket, during capture.
  Creates backend-specific wrapper objects that point at *fixed* buffers so the recorded kernel
  launch parameters stay valid.
- `init_forward_metadata_replay_cuda_graph(...)` — once per replay. Refreshes the contents
  (not the pointers) of those buffers with real seq_lens / kv indices for this batch.

This three-step pattern is why not every backend supports CUDA graphs — torch_native and
flex_attention do not, and SGLang disables cuda graph automatically for those
(`server_args.py:_handle_attention_backend_compatibility`).

---

## 5. Parallelism

SGLang supports tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP),
attention context parallelism (ACP / DCP), and expert parallelism (EP). These are orthogonal
axes; you can combine them. The total GPU count is:

```
world_size = tp_size * pp_size          (× dp replicas if DP)
```

### 5.1 Distributed initialization

Every scheduler subprocess eventually calls `init_distributed_environment`, which initializes a
single global `torch.distributed` process group and then builds subgroups on top of it:

```1729:1800:python/sglang/srt/distributed/parallel_state.py
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: Optional[int] = None,
    moe_a2a_backend: Optional[str] = None,
):
    ...
    if not torch.distributed.is_initialized():
        ...
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
            pg_options=pg_options,
        )
    ...
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
```

Rendezvous URL construction happens in `ModelRunner.init_torch_distributed` and supports three
sources:

1. `SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE=env://` (uses `MASTER_ADDR`/`MASTER_PORT`).
2. `--dist-init-addr host:port` (multi-node).
3. Default: `tcp://127.0.0.1:{nccl_port}` for single-node.

```1015:1061:python/sglang/srt/model_executor/model_runner.py
        dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
        if dist_init_method_override:
            dist_init_method = dist_init_method_override
        elif self.server_args.dist_init_addr:
            na = NetworkAddress.parse(self.server_args.dist_init_addr)
            dist_init_method = na.to_tcp()
        else:
            dist_init_method = NetworkAddress(
                self.server_args.host or "127.0.0.1", self.dist_port
            ).to_tcp()
        ...
        init_distributed_environment(
            backend=backend,
            world_size=self.tp_size * self.pp_size,
            rank=self.tp_size * self.pp_rank + self.tp_rank,
            local_rank=self.gpu_id,
            distributed_init_method=dist_init_method,
            timeout=self.server_args.dist_timeout,
            moe_a2a_backend=self.server_args.moe_a2a_backend,
        )
```

### 5.2 Process groups: TP, PP, DP, CP, EP

`initialize_model_parallel` builds all the subgroups. Each group is represented by a
`GroupCoordinator` that wraps **two** `ProcessGroup` objects: one NCCL group on the device
(for tensor collectives) and one gloo group on CPU (for Python objects and some init handshakes).

Example: the TP group uses consecutive global ranks.

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

The PP group, on the other hand, is **strided**: for 8 GPUs and `tp=2, pp=4`, TP groups are
`[[0,1],[2,3],[4,5],[6,7]]` while PP groups are `[[0,2,4,6],[1,3,5,7]]`.

After this function returns, the following globals are available:

| Accessor | Purpose |
|----------|---------|
| `get_tp_group()` | Tensor-parallel GroupCoordinator |
| `get_pp_group()` | Pipeline-parallel GroupCoordinator |
| `get_dcp_group()` | Decode context-parallel group |
| `get_attention_tp_group()` | TP within one DP-attention rank |
| `get_moe_ep_group()`, `get_moe_tp_group()` | MoE parallelism |
| `get_world_group()` | The full world |

### 5.3 Model loading and weight sharding

`ModelRunner.load_model` constructs a `LoadConfig` that carries `tp_rank`, then delegates to a
loader which eventually calls `model.load_weights(...)`:

```1172:1277:python/sglang/srt/model_executor/model_runner.py
    def load_model(self):
        tic_total = time.perf_counter()
        ...
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            ...
            tp_rank=self.tp_rank,
            ...
            modelexpress_pp_size=self.server_args.pp_size,
            modelexpress_ep_size=self.server_args.ep_size,
        )
        ...
        self.loader = get_model_loader(
            load_config=self.load_config,
            model_config=self.model_config,
        )
        self.model = self.loader.load_model(
            model_config=self.model_config,
            device_config=DeviceConfig(self.device, self.gpu_id),
        )
```

The actual **sharding** is not centralized in one function — it is pushed into every
"parallel-aware" parameter via a `weight_loader` hook. For example, `ColumnParallelLinear.weight_loader`
narrows the checkpoint tensor along the output dim, taking slice
`[tp_rank * shard_size : (tp_rank+1) * shard_size]`.

### 5.3.1 Parallel linear layers — the building blocks

SGLang's linear layers come in a family. They all live in
`python/sglang/srt/layers/linear.py`:

| Class | Sharding | Where it's used |
|-------|----------|-----------------|
| `ColumnParallelLinear` | weight sharded on output dim; output is **partial** (not gathered unless `gather_output=True`) | The up-projections of an MLP |
| `MergedColumnParallelLinear` | several column-parallel linears fused into one GEMM | `gate_up_proj` |
| `QKVParallelLinear` | column-parallel with head-aware sharding | Attention QKV |
| `RowParallelLinear` | weight sharded on input dim; output is **reduced** (all-reduce) | `o_proj`, `down_proj` |
| `VocabParallelEmbedding` / `ParallelLMHead` | vocab dim sharded; embedding path all-reduces partials | Input embedding and LM head |

`ColumnParallelLinear.forward` is very short — a matmul and an optional all-gather:

```455:467:python/sglang/srt/layers/linear.py
    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
```

`RowParallelLinear.forward` is where the all-reduce happens — **inside the layer**, not in the
model code:

```1492:1525:python/sglang/srt/layers/linear.py
    def forward(self, input_, skip_all_reduce=False):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        assert self.quant_method is not None
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        if self.use_dp_attention_reduce:
            symm_ctx = use_symmetric_memory(get_attention_tp_group())
        else:
            symm_ctx = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
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

Notice `skip_all_reduce` — when combined with sequence parallelism, the all-reduce is replaced
with a reduce-scatter followed by an all-gather earlier/later in the pipeline.

`VocabParallelEmbedding.forward` shows the "mask-then-reduce" pattern used to make embedding
consistent across ranks:

```471:500:python/sglang/srt/layers/vocab_parallel_embedding.py
    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                ...
            )
        else:
            masked_input = input_

        # Get the embeddings.
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            output_parallel = self.quant_method.embedding(self, masked_input.long())

        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            if not get_attn_tp_context().input_scattered:
                if self.use_attn_tp_group:
                    output_parallel = attn_tp_all_reduce(output_parallel)
                else:
                    output_parallel = tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel
```

Every rank holds only `vocab_size / tp_size` rows of the embedding table. It zeros out token ids
that belong to other shards, then an all-reduce re-assembles the full hidden vector.

### 5.4 Input preparation per rank

All TP ranks in one TP group run the **same logical batch**. The scheduler in rank 0 reads
from ZMQ and broadcasts the list to the other ranks via a gloo group (CPU-side pickling is fine
because these are Python objects, not tensors):

```1502:1590:python/sglang/srt/managers/scheduler.py
    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
        ...
        if self.pp_rank == 0:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)
                ...
            else:
                recv_reqs = None
        ...
        if self.server_args.enable_dp_attention:
            ...
            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            ...
        elif self.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
```

`broadcast_pyobj` lives in `utils/common.py` and uses the CPU-backed gloo group; it pickles the
object and broadcasts the bytes.

Once every rank in the TP group has the same request list, they:

1. Schedule the same prefill/decode batch.
2. Build the same `ForwardBatch`.
3. Run `ModelRunner.forward(...)` in lock-step.
4. Sample on the last PP rank (only).

### 5.5 KV cache per process

KV cache is stored in `MHATokenToKVPool` (or `MLATokenToKVPool` for DeepSeek-style models). Each
GPU process allocates **only its share** of heads (divided by `get_attention_tp_size()`) and
**only its share** of layers (for PP):

```514:530:python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
                self.token_to_kv_pool = MHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    ...
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    ...
                )
```

The actual tensor layout is `[size, head_num_local, head_dim]` per layer:

```846:870:python/sglang/srt/mem_cache/memory_pool.py
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.v_head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
```

Slots in the pool are handed out by a `TokenToKVPoolAllocator`. When Decode Context Parallelism
(DCP) is enabled, the allocator stripes token slots by position modulo `dcp_world_size` so each
DCP rank holds a disjoint subset of the sequence:

```145:171:python/sglang/srt/mem_cache/allocator.py
    def alloc(self, need_size: int, token_positions=None):
        dcp_world_size = get_dcp_world_size()
        dcp_rank = get_dcp_rank()

        if dcp_world_size > 1 and token_positions is not None:
            local_mask = (token_positions % dcp_world_size) == dcp_rank
            local_count = int(local_mask.sum().item())
            ...
            result = torch.full((need_size,), -1, dtype=torch.int64, device=self.device)
            result[local_mask] = self.free_pages[:local_count]
            ...
            return result
        ...
```

**Prefix-cache / RadixCache** sits on top of the allocator. The `tree_cache` field of the
scheduler is a radix tree of token IDs → allocated slots; when a new request shares a prefix
with a cached one, the scheduler reuses those slots instead of recomputing.

### 5.6 Inter-layer data flow

Here is one complete forward pass through a transformer layer on a TP group. We use Llama as the
example because it is simple; DeepSeek-V2 is structurally similar but adds MoE and MLA.

```
             input hidden (replicated across TP)
                       │
                       ▼
                   RMSNorm
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
   QKVParallelLinear            MergedColumnParallelLinear
   (column parallel)            (column parallel, gate_up)
       │                               │
   partial Q,K,V  (head-sharded)   partial gate_up
       │                               │
       ▼                               ▼
   RadixAttention                  SiLU × gate
 (backend kernel,                       │
  KV from local pool)                   ▼
       │                           RowParallelLinear
       ▼                           (down_proj)
   o_proj (RowParallelLinear)          │
       │                           ALL-REDUCE (inside layer)
   ALL-REDUCE (inside layer)           │
       │                               ▼
       └──────────► residual add ──────┘
                       │
                       ▼
                  output hidden (replicated across TP)
```

You can literally read this off of Llama's attention and MLP modules. From `python/sglang/srt/models/llama.py`:

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

The implicit rule: **every column-parallel output is partial, and every row-parallel input is
partial**, so they compose perfectly with no explicit synchronization in the model code. The
all-reduce that glues them together is embedded inside `RowParallelLinear.forward`.

### 5.7 Pipeline parallelism

PP splits **layers** across ranks. `get_pp_indices` partitions `[0, num_layers)` into contiguous
ranges:

```99:135:python/sglang/srt/distributed/utils.py
def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> Tuple[int, int]:
    ...
    partition_list_str = os.getenv("SGLANG_PP_LAYER_PARTITION", None)
    if partition_list_str is not None:
        ...
        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]
    else:
        base_layers = num_hidden_layers // pp_size
        remainder = num_hidden_layers % pp_size
        ...
    return (start_layer, end_layer)
```

Llama's `LlamaModel` uses this directly. Non-owned layers are replaced with `PPMissingLayer()`,
and only the first rank has an embedding, only the last has the final norm:

```335:412:python/sglang/srt/models/llama.py
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LlamaDecoderLayer(
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        ...
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        if self.pp_group.is_first_rank:
            ...
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            ...
        for i in range(self.start_layer, self.end_layer):
            ...
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        ...
```

Stage `k` returns a `PPProxyTensors({"hidden_states": ..., "residual": ...})`. Stage `k+1` receives
it from the scheduler's PP loop (`scheduler_pp_mixin.py`), which uses async NCCL send + sync recv
to ship the tensor dict between PP neighbors.

### 5.8 Expert parallelism

MoE models (DeepSeek, Qwen-MoE, Mixtral) scatter tokens across expert GPUs. Every MoE layer ends
up calling `FusedMoE.forward_impl`:

```1005:1039:python/sglang/srt/layers/moe/fused_moe_triton/layer.py
    def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        origin_hidden_states_dim = hidden_states.shape[-1]
        assert self.quant_method is not None

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        ...
        combine_input = self.run_moe_core(
            dispatch_output=dispatch_output,
        )

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            final_hidden_states = self.dispatcher.combine(combine_input=combine_input)
            ...
        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states
```

The flow is:

1. **Dispatch** — an all-to-all that sends each token to the GPU(s) holding its top-K experts.
2. **Expert compute** — each rank runs GEMMs only for its local experts.
3. **Combine** — an all-to-all back, weighted by routing probabilities.
4. **Optional all-reduce** — if MoE is combined with TP.

The dispatcher is pluggable: `flashinfer`, `deepep`, `pplx`, `standard`. Each uses different
kernels but presents the same `dispatch/combine` interface.

---

## 6. Communication primitives inside a step

The thin wrappers used by model code all live in `communication_op.py`:

```16:40:python/sglang/srt/distributed/communication_op.py
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
...
def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)
```

The interesting part is `GroupCoordinator.all_reduce`. It is **not** just `dist.all_reduce`. It
picks the best implementation per tensor, depending on size, group, and whether a CUDA graph is
being captured:

```554:631:python/sglang/srt/distributed/parallel_state.py
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if input_.is_cpu:
            if is_shm_available(input_.dtype, self.world_size, self.local_size):
                torch.ops.sgl_kernel.shm_allreduce(input_, REDUCE_OP_SUM)
            else:
                torch.distributed.all_reduce(input_, group=self.device_group)
            return input_
        ...
        if self.pynccl_comm is not None and self.is_symmetric_memory_enabled():
            self.debug_check_symmetric_mempool(self, {"input": input_}, "all_reduce")
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
        elif (
            self.pymscclpp_comm is not None
            and not self.pymscclpp_comm.disabled
            and self.pymscclpp_comm.should_mscclpp_allreduce(input_)
        ):
            outplace_all_reduce_method = "pymscclpp"
        elif (
            self.torch_symm_mem_comm is not None
            and not self.torch_symm_mem_comm.disabled
            and self.torch_symm_mem_comm.should_torch_symm_mem_allreduce(input_)
        ):
            outplace_all_reduce_method = "torch_symm_mem"
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

The backends SGLang can pick from, roughly from fastest-to-slowest under different regimes:

- `CustomAllreduce` (IPC-based one-shot / two-shot, small messages, no graph).
- `QuickAllreduce` (AMD, symmetric memory).
- `pymscclpp` (MSCCL++).
- `TorchSymmMemCommunicator` (PyTorch symmetric memory).
- `PyNcclCommunicator` (manual NCCL, works inside CUDA graphs).
- `torch.distributed.all_reduce` (fallback).

This is why SGLang usually outperforms a naive PyTorch setup: it chooses the right primitive for
each tensor and execution context rather than using one code path everywhere.

---

## 7. End-to-end request flow

Finally, let's follow a single HTTP request all the way down and back.

### 7.1 HTTP layer

```685:718:python/sglang/srt/entrypoints/http_server.py
@app.api_route(
    "/generate",
    methods=["POST", "PUT"],
    response_class=SGLangORJSONResponse,
)
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + dumps_json(out) + b"\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + dumps_json(out) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return orjson_response(ret)
```

OpenAI routes (`/v1/chat/completions`, `/v1/completions`) use Pydantic models from
`entrypoints/openai/protocol.py`. Their `OpenAIServingChat.handle_request` eventually builds the
same internal `GenerateReqInput` and calls `tokenizer_manager.generate_request(...)`.

### 7.2 TokenizerManager

```515:560:python/sglang/srt/managers/tokenizer_manager.py
    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()

        # Normalize the request
        obj.normalize_batch_and_arguments()
        self._set_default_priority(obj)
        ...
        async with self.model_update_lock.reader_lock:
            await self._validate_and_resolve_lora(obj)

            # Tokenize the request and send it to the scheduler
            if obj.is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                self._send_one_request(tokenized_obj)
                async for response in self._wait_one_response(obj, request):
                    yield response
```

Steps:

1. Assigns a request id (`rid`) — a hex UUID if the caller didn't give one.
2. Calls the HuggingFace tokenizer to convert the prompt into token IDs.
3. Pushes a `TokenizedGenerateReqInput` over ZMQ to the scheduler.
4. Awaits an `asyncio.Event` that the background `handle_loop` signals whenever new output
   arrives from the detokenizer.

```1152:1159:python/sglang/srt/managers/tokenizer_manager.py
    def _send_one_request(
        self,
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        tokenized_obj.time_stats.set_api_server_dispatch_time()
        tokenized_obj = wrap_shm_features(tokenized_obj)
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        tokenized_obj.time_stats.set_api_server_dispatch_finish_time()
```

### 7.3 Scheduler event loop

The scheduler's main loop is a trivial while-true:

```1382:1407:python/sglang/srt/managers/scheduler.py
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self.cancel_bubble_timer()
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self.on_idle()

            # Update last_batch
            self.last_batch = batch
```

Per iteration:

1. `recv_requests()` — rank-0 pulls from ZMQ, all ranks receive the list via `broadcast_pyobj`
   (see §5.4).
2. `process_input_requests()` — dispatches on type; for generation, `handle_generate_request`
   builds a `Req` object and appends it to `waiting_queue`.
3. `get_next_batch_to_run()` — picks a prefill batch (via `PrefillAdder` and `tree_cache` for
   RadixCache lookups) or, if there is nothing new, extends the current decode batch.
4. `run_batch(batch)` — calls `self.model_worker.forward_batch_generation(model_worker_batch)`.
5. `process_batch_result(batch, result)` — turns token IDs into `BatchTokenIDOutput` messages
   and pushes them to the detokenizer over ZMQ.

### 7.4 Model execution

Inside `TpModelWorker.forward_batch_generation`, the batch is promoted to a `ForwardBatch` and
handed to `ModelRunner`. On the last PP rank this also samples the next tokens:

```444:504:python/sglang/srt/managers/tp_worker.py
    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        ...
    ) -> GenerationBatchResult:
        if model_worker_batch is not None:
            ...
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        ...
        if self.pp_group.is_last_rank:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            ...
            if not model_worker_batch.is_prefill_only:
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
```

`ModelRunner.forward` (eventually `_forward_raw`) branches on forward mode:

```3115:3194:python/sglang/srt/model_executor/model_runner.py
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

        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)
        else:
            forward_batch.prepare_attn_tp_scatter_input(self)
        ...
        if forward_batch.forward_mode.is_decode():
            ret = self.forward_decode(...)
        elif forward_batch.forward_mode.is_split_prefill():
            ret = self.forward_split_prefill(...)
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            ret, can_run_graph = self.forward_extend(...)
        elif forward_batch.forward_mode.is_idle():
            ret = self.forward_idle(...)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")
        ...
```

Whichever branch runs, it eventually calls `self.model(...)`, which is the HuggingFace-style
transformer we discussed in §5. All the distributed collectives happen **inside** that call as
side effects of `RowParallelLinear` / `VocabParallelEmbedding` / MoE dispatchers.

### 7.5 Sampling

After `self.model(...)` returns `LogitsProcessorOutput.next_token_logits`, `ModelRunner.sample`
calls `Sampler.forward`:

```100:170:python/sglang/srt/layers/sampler.py
        logits = logits_output.next_token_logits

        # Preprocess logits (custom processors and NaN handling)
        logits = self._preprocess_logits(logits, sampling_info)

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)
            ...
        else:
            ...
            else:
                # Standard path: do softmax and sample from probs.
                logits.div_(sampling_info.temperatures)

                # In-place op to save memory
                logits[:] = torch.softmax(logits, dim=-1)
                probs = logits

                batch_next_token_ids = self._sample_from_probs(
                    probs, sampling_info, positions, simple_sampling_case
                )
```

Structured output (JSON schema, regex, grammar) is applied *before* sampling via
`sampling_info.update_regex_vocab_mask()` — it sets `-inf` on logits of disallowed tokens so any
sampling method respects the grammar.

### 7.6 Detokenization and streaming

The scheduler pushes `BatchTokenIDOutput` to the detokenizer:

```
Scheduler  ──PUSH──►  Detokenizer  ──PUSH──►  TokenizerManager
                                                   ↓
                                           SSE yield to HTTP
```

Detokenizer does incremental decoding (keeping prefix state per `rid`) and emits
`BatchStrOutput`. The TokenizerManager's `handle_loop` dispatches these to per-`rid` asyncio
events, which `_wait_one_response` yields. The HTTP endpoint then wraps each event in the
SSE `data: ...\n\n` format.

### 7.7 End-to-end timeline (decode step)

```
            HTTP                   TokMgr              Scheduler (TP rank 0..N)                 Detok
             │                        │                          │                               │
  POST /v1/.. │──► FastAPI handler    │                          │                               │
             │──── GenerateReqInput ─►│                          │                               │
             │                        │ tokenize(prompt)         │                               │
             │                        │──► TokenizedGenerateReq ─►│  recv_from_tokenizer          │
             │                        │       (ZMQ PUSH)         │  broadcast_pyobj to TP ranks  │
             │                        │                          │  append to waiting_queue      │
             │                        │                          │  build batch → ForwardBatch   │
             │                        │                          │  ModelRunner.forward ─────┐   │
             │                        │                          │                           │   │
             │                        │                          │  (NCCL all-reduce,        │   │
             │                        │                          │   attn backend kernel,    │   │
             │                        │                          │   cuda graph replay, ...) │   │
             │                        │                          │                           │   │
             │                        │                          │  Sampler → next token_id  ◄───┘
             │                        │                          │  BatchTokenIDOutput ──────────►
             │                        │    BatchStrOutput ◄──────┼──────────────────────────  decode
             │                        │    (ZMQ PULL)            │                               │
             │ ◄── SSE "data: ..\n\n" ─│                          │                               │
             ▼                        ▼                          ▼                               ▼
```

For a 100-token response this loop runs 100 times (plus one prefill at the beginning).

---

## 8. Cheat sheet

| Topic | Primary file | Key symbols |
|-------|--------------|-------------|
| Attention ABC | `layers/attention/base_attn_backend.py` | `AttentionBackend` |
| Backend registry | `layers/attention/attention_registry.py` | `ATTENTION_BACKENDS`, `register_attention_backend` |
| FlashInfer backend | `layers/attention/flashinfer_backend.py` | `FlashInferAttnBackend` |
| FA3/FA4 backend | `layers/attention/flashattention_backend.py` | `FlashAttentionBackend` |
| Hybrid routing | `layers/attention/hybrid_attn_backend.py` | `HybridAttnBackend._select_backend` |
| RadixAttention layer | `layers/radix_attention.py` | `RadixAttention.forward` |
| Forward modes | `model_executor/forward_batch_info.py` | `ForwardMode`, `ForwardBatch` |
| CUDA graph runner | `model_executor/cuda_graph_runner.py` | `CudaGraphRunner.capture`, `replay`, `can_run` |
| Parallel linears | `layers/linear.py` | `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear` |
| Vocab parallel | `layers/vocab_parallel_embedding.py` | `VocabParallelEmbedding` |
| Logits | `layers/logits_processor.py` | `LogitsProcessor._get_logits` |
| Parallel groups | `distributed/parallel_state.py` | `init_distributed_environment`, `initialize_model_parallel`, `GroupCoordinator` |
| All-reduce op | `distributed/communication_op.py` | `tensor_model_parallel_all_reduce` |
| PyNCCL | `distributed/device_communicators/pynccl.py` | `PyNcclCommunicator` |
| Scheduler loop | `managers/scheduler.py` | `event_loop_normal`, `recv_requests`, `run_batch` |
| PP send/recv | `managers/scheduler_pp_mixin.py` | `event_loop_pp` |
| TP worker | `managers/tp_worker.py` | `TpModelWorker.forward_batch_generation` |
| Model runner | `model_executor/model_runner.py` | `ModelRunner.forward`, `_forward_raw`, `sample`, `load_model`, `init_torch_distributed` |
| KV pool | `mem_cache/memory_pool.py` | `MHATokenToKVPool`, `MLATokenToKVPool` |
| KV allocator | `mem_cache/allocator.py` | `TokenToKVPoolAllocator.alloc` |
| MoE layer | `layers/moe/fused_moe_triton/layer.py` | `FusedMoE.forward_impl` |
| MoE dispatcher | `layers/moe/token_dispatcher/flashinfer.py` | `FlashinferDispatcher.dispatch/combine` |
| HTTP server | `entrypoints/http_server.py` | `generate_request`, `launch_server` |
| TokenizerManager | `managers/tokenizer_manager.py` | `TokenizerManager.generate_request`, `handle_loop` |
| DetokenizerManager | `managers/detokenizer_manager.py` | `DetokenizerManager.event_loop` |
| Engine launcher | `entrypoints/engine.py` | `Engine._launch_subprocesses`, `_launch_scheduler_processes` |
| Sampler | `layers/sampler.py` | `Sampler.forward` |
| Llama example | `models/llama.py` | `LlamaAttention`, `LlamaMLP`, `LlamaModel` |
| DeepSeek MoE example | `models/deepseek_v2.py` | `DeepseekV2MoE`, `DeepseekV2AttentionMLA` |

### Suggested reading order for the seminar

1. §7 (end-to-end flow) — set the scene.
2. §3 (attention backend) — show a clean abstraction.
3. §5.6 (inter-layer data flow) — make the TP math concrete.
4. §5.1, §5.2, §6 (distributed + collectives) — explain the "magic" of the all-reduce.
5. §4 (CUDA graph) — the performance story.
6. §5.5, §5.7, §5.8 (KV cache, PP, MoE) — dessert.
