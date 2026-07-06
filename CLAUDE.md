# SGLang DCP A2A - Benchmarking

## Current Work: DCP A2A Benchmarking

Benchmarking DCP (Decode Context Parallelism) with All-to-All (A2A) communication backend on H100 and B200 GPUs.

- **Branch:** `htphan/helix_a2a_rebased_main_fe294904c9`
- **Model:** `deepseek-ai/DeepSeek-V2-Lite`
- **Benchmark tracking:** `tasks/accuracy_performance_benchmarkings.md`
- **Server info:** `knowledge/server_info.md`
- **Prior accuracy matrix (24/24 pass):** `benchmark_results.md`

### Benchmark Scope
- **Accuracy:** GSM8K across TP1, TP8, DCP8 ag_rs, DCP8 a2a flashinfer, DCP8 a2a fa3
- **Performance:** bench_serving with 256k/512k/1M context lengths across TP8, DCP8 ag_rs, DCP8 a2a flashinfer, DCP8 a2a fa3
- **Hardware:** H100 (colossus) and B200 (colossus_b200_1, colossus_b200_2)
