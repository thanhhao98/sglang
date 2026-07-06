 What Already Exists

  ┌────────────────────────────────────┬─────────────────────────────────────────────────────────────────┬──────────────────────┐
  │                File                │                          What It Tests                          │        Status        │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ test_dcp_a2a.py                    │ LSE combine Triton kernel vs CPU reference                      │ Unit, CUDA           │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ test_dcp_flashattn.py              │ DCP + FlashAttention backend integration, CUDA graph buffers    │ Unit, CUDA           │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ test_dcp_accuracy.py               │ End-to-end A2A/AG_RS generation correctness                     │ Integration, 2+ GPUs │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ test_dcp_accuracy_matrix.py        │ 24-scenario matrix (backend × comm × cuda_graph × request_type) │ Integration, 4+ GPUs │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ test_dcp_interleaved_storage.py    │ KV cache allocator DCP interleaving                             │ Unit, CUDA           │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ test_fa3_mla_dcp_standalone.py     │ FA3 MLA attention with simulated DCP sharding                   │ Unit, CUDA           │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ test_fa3_flashinfer_lse_compare.py │ FA3 vs FlashInfer LSE shape/base comparison                     │ Unit, CUDA           │
  └────────────────────────────────────┴─────────────────────────────────────────────────────────────────┴──────────────────────┘

  What's Missing — Tests You Should Add

  1. Fused A2A kernel correctness (unit test, 1 GPU)
  File: test/srt/test_dcp_a2a.py (extend existing)
  - Test fused pack/unpack preserves fp32 LSE precision through bf16 reinterpretation
  - Test combined buffer layout [N, B, H_per_rank, D+2] produces same output as 2-call path
  - Test with various dtypes (bf16, fp16)

  2. Per-bs CUDA graph buffer allocation (unit test, 1 GPU)
  File: test/srt/test_dcp_flashattn.py (extend existing)
  - Test _alloc_dcp_a2a_buffers_for_bs creates correct buffer shapes for each bs
  - Test buffer reuse across repeated calls with same bs
  - Test dcp_cuda_graph_buffers points to correct per-bs buffer during capture/replay
  - Test combined buffer layout matches fused A2A expectations

  3. DCP + FlashInfer backend integration (unit test, 1 GPU)
  File: test/srt/test_dcp_flashinfer.py (NEW)
  - Mirror test_dcp_flashattn.py patterns for FlashInfer
  - Test LSE base-2 convention (FlashInfer) vs base-e (FA3)
  - Test DCP reduce with FlashInfer attention output

  4. DCP accuracy with fused A2A (integration, 8 GPUs)
  File: test/srt/test_dcp_accuracy.py (extend existing)
  - Add TestDCPFusedA2AAccuracy class
  - Test deterministic output with fused all_to_all
  - Compare output vs non-fused A2A (should be identical)

  5. DCP with Q-project replication (integration, 8 GPUs)
  File: test/srt/test_dcp_accuracy.py (extend existing)
  - Test --dcp-replicate-q-proj produces correct output
  - Test determinism
  - Compare output vs standard DCP (should match)

  6. DCP scheduling pressure (integration, 8 GPUs)
  File: test/srt/test_dcp_scheduling.py (NEW)
  - Test that decode runs at c>=2 (not starved by prefill)
  - Measure gen_throughput > 0 when running_req > 0
  - Test with/without schedule_policy dcp_size cap

  7. DCP + mixed-chunk (integration, 8 GPUs — when fixed)
  File: test/srt/test_dcp_mixed_chunk.py (NEW)
  - Test --enable-mixed-chunk + DCP doesn't hang
  - Test accuracy preserved with mixed-chunk
  - Test TPOT at c>=2 shows improvement vs no mixed-chunk

  8. DCP multi-model support (integration, 2+ GPUs)
  File: test/srt/test_dcp_models.py (NEW)
  - Test DCP with different model architectures (not just DeepSeek-V2-Lite)
  - Test MLA vs standard MHA attention paths
  - Test with different head counts, head dims

  Benchmarks You Should Add

  1. DCP comm-layer microbenchmark (NEW)
  File: benchmark/dcp/bench_dcp_comm.py
  - Measure per-layer A2A vs AG_RS latency at various batch sizes
  - Measure fused vs non-fused A2A
  - Measure with/without symm-mem
  - Output: table of (backend, batch_size, latency_us)

  2. DCP decode TPOT benchmark (NEW)
  File: benchmark/dcp/bench_dcp_tpot.py
  - Compare TP8 vs DCP8 AG_RS vs DCP8 A2A at various context lengths (128K-1M)
  - Sweep concurrency 1-512
  - Output: TPOT, throughput, TTFT tables

  3. DCP accuracy benchmark (extend existing)
  File: benchmark/gsm8k/bench_sglang.py (use existing)
  - Standardized script to run GSM8K across all DCP configs
  - Track accuracy regression across commits