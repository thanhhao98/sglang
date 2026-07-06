 ---
  Output Sequence Length (OSL)

  Fixed at 1024 tokens across all benchmarks.

  This is hardcoded in:
  - run_full_benchmark_v3.sh: OUTPUT_TOKENS=1024
  - run_gb200_aiperf_v3.sbatch: OUTPUT_TOKENS="${OUTPUT_TOKENS:-1024}"
  - profile_helix.sh: OUTPUT_TOKENS=1024
  - All analysis scripts validate: EXPECTED_OUTPUT_TOKENS = 1024 (flags data if avg deviates by >5 tokens)

  In v41, --ignore-eos is set both server-side and client-side to force full 1024-token generation
  (preventing early EOS from truncating output and skewing TPOT measurements).

  ---
  Input Sequence Length (ISL)

  Computed dynamically from the target context length:

  ISL = context_length - OUTPUT_TOKENS (- 100 buffer in some scripts)

  So the input fills up the rest of the context window after reserving space for the 1024 output tokens.

  ┌────────────────┬──────────────┬───────┐
  │ Target Context │ ISL (approx) │  OSL  │
  ├────────────────┼──────────────┼───────┤
  │ 32K (32,768)   │ ~31,744      │ 1,024 │
  ├────────────────┼──────────────┼───────┤
  │ 64K (65,536)   │ ~64,512      │ 1,024 │
  ├────────────────┼──────────────┼───────┤
  │ 128K (131,072) │ ~130,048     │ 1,024 │
  ├────────────────┼──────────────┼───────┤
  │ 256K (262,144) │ ~261,120     │ 1,024 │
  ├────────────────┼──────────────┼───────┤
  │ 512K (524,288) │ ~523,264     │ 1,024 │
  ├────────────────┼──────────────┼───────┤
  │ 1M (1,048,576) │ ~1,047,552   │ 1,024 │
  └────────────────┴──────────────┴───────┘

  ---
  Context Lengths Tested

  Depends on model type:

  ┌────────────────────────┬───────────────────────────────────────────┐
  │         Model          │                 Contexts                  │
  ├────────────────────────┼───────────────────────────────────────────┤
  │ GQA (Nemotron-49B)     │ 32K, 64K, 128K, 256K, 512K, 1M (6 values) │
  ├────────────────────────┼───────────────────────────────────────────┤
  │ MLA (DeepSeek-V2-Lite) │ 128K, 256K, 512K, 1M (4 values)           │
  └────────────────────────┴───────────────────────────────────────────┘

  ---
  Concurrency Levels

  ┌──────────────┬───────────────────────────────────────────────────────┐
  │   Version    │                      Concurrency                      │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ v40 Wave 1   │ c = 1, 2, 4, 8, 16, 32, 64 (7 levels)                 │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ v40/v41 full │ c = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 (10 levels) │
  └──────────────┴───────────────────────────────────────────────────────┘

  c=1024 was dropped because it exceeds the CUDA graph capture cap (min(max_num_seqs*2, 512) = 512),
  forcing eager mode (different execution path).

  ---
  Other Parameters

  ┌──────────────────────────────┬─────────────────────────────────────────┐
  │          Parameter           │                  Value                  │
  ├──────────────────────────────┼─────────────────────────────────────────┤
  │ Number of prompts per config │ 5 (enough for CV < 1%)                  │
  ├──────────────────────────────┼─────────────────────────────────────────┤
  │ Prefix caching               │ Enabled (decode-only measurement)       │
  ├──────────────────────────────┼─────────────────────────────────────────┤
  │ CUDA graphs                  │ PIECEWISE mode (for DCP configs)        │
  ├──────────────────────────────┼─────────────────────────────────────────┤
  │ max-num-seqs                 │ 1024 (v41; was 2048 in v40, caused OOM) │
  └──────────────────────────────┴─────────────────────────────────────────┘

  ---
  Summary

  The benchmark design measures long-context decode latency: fill the context window with a huge input
  prompt (up to ~1M tokens), then generate exactly 1024 output tokens, measuring TPOT (Time Per Output
  Token) as the primary metric. This isolates the decode phase where Helix's KV cache sharding provides its
   advantage.