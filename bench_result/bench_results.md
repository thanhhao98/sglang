# Benchmark Results

## Output token throughput (tok/s)

| Config                                                                      |    cc1 |    cc2 |    cc4 |    cc8 |   cc16 |    cc64 |   cc256 |   cc512 |
|-----------------------------------------------------------------------------|--------|--------|--------|--------|--------|---------|---------|---------|
| debug_fa3-dcp-vectorize_8549455_tp8_dcp8_fa3_a2a_vectorized                 |  87.19 | 154.59 | 257.85 | 412.92 | 631.4  | 1340.38 | 2534.97 | 2566.73 |
| debug_fa3-dcp-vectorize_8549455_tp8_dcp8_fa3_agrs_vectorized                |  88.89 | 157.66 | 263.28 | 422.23 | 649.93 | 1354.8  | 2539.37 | 2569.9  |
| debug_fa3-dcp-vectorize_b14d632_tp8_dcp8_fa3_a2a_vectorized_083             |  87.4  | 154.43 | 258.2  | 412.67 | 631.79 | 1339.18 | 2566.99 | 2947.66 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_a2a               |  65.52 | 111.85 | 173.53 | 284.24 | 458.47 | 1102.43 | 2261.56 | 2300.07 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_a2a_q_repl        |  60.09 | 103.05 | 161.93 | 267.75 | 437.72 | 1071.45 | 1727.06 | 1748.22 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_agrs              |  66.4  | 112.86 | 174.99 | 287.96 | 466.22 | 1109.42 | 2256.33 | 2296.71 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_a2a        |  86.66 | 154.73 | 261.04 | 412.19 | 626.39 | 1320.9  | 2558    | 3144    |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_a2a_q_repl |  77.51 | 139.85 | 237.62 | 380.88 | 590.59 | 1279.08 | 1996.78 | 2011.64 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_agrs       |  88.62 | 158.64 | 267.57 | 425.01 | 650.66 | 1337.85 | 2561.67 | 3111.28 |
| htphan_q-project-replication-rebased_bb19570_tp8_fa3                        | 102.91 | 187.28 | 312.3  | 492.3  | 743.02 | 1370.72 | 1384.4  | 1381.94 |
| htphan_q-project-replication-rebased_bb19570_tp8_flashinfer                 | 103.41 | 190    | 316.8  | 498.55 | 748.07 | 1385.3  | 1403.56 | 1400.09 |
| main_44db0c5_tp8_fa3                                                        |  96.74 | 173.78 | 293.17 | 470.42 | 720.28 | 1379.15 | 1395.44 | 1393.85 |
| main_44db0c5_tp8_flashinfer                                                 |  96.76 | 176.31 | 297.63 | 476.47 | 728.01 | 1400.12 | 1418.81 | 1415.16 |

## Mean TTFT (ms)

| Config                                                                      |    cc1 |    cc2 |    cc4 |    cc8 |   cc16 |    cc64 |     cc256 |    cc512 |
|-----------------------------------------------------------------------------|--------|--------|--------|--------|--------|---------|-----------|----------|
| debug_fa3-dcp-vectorize_8549455_tp8_dcp8_fa3_a2a_vectorized                 | 120.21 | 140.1  | 156.49 | 202.67 | 299.74 |  658.47 |   9118.36 |  80994.6 |
| debug_fa3-dcp-vectorize_8549455_tp8_dcp8_fa3_agrs_vectorized                | 124.62 | 140.16 | 157.98 | 202.57 | 292.74 |  665.15 |   9123.06 |  80910.3 |
| debug_fa3-dcp-vectorize_b14d632_tp8_dcp8_fa3_a2a_vectorized_083             | 116.98 | 153.34 | 156.37 | 213.74 | 311.97 |  660.4  |   1977.25 |  46154.5 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_a2a               | 122.53 | 149.66 | 168    | 221.55 | 310.41 |  712.48 |   9888.89 |  89558.2 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_a2a_q_repl        | 129.4  | 161.12 | 189.06 | 254.41 | 378.84 |  854.91 |  53394.8  | 159080   |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_agrs              | 120.05 | 147.87 | 166.71 | 225.02 | 309.56 |  666.19 |   9924.4  |  89697.6 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_a2a        | 116.43 | 171.58 | 184.41 | 226.85 | 326.13 |  664.22 |   1975.93 |  28397.4 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_a2a_q_repl | 129.43 | 154.28 | 178.83 | 240.53 | 383.91 |  852.58 |  46746.4  | 138853   |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_agrs       | 114.32 | 137.58 | 151.36 | 200.39 | 291.01 |  654.86 |   1976.81 |  28709.8 |
| htphan_q-project-replication-rebased_bb19570_tp8_fa3                        | 112.63 | 133.7  | 146.25 | 192.64 | 280.25 | 5784.69 | 107994    | 242730   |
| htphan_q-project-replication-rebased_bb19570_tp8_flashinfer                 | 113.46 | 135.6  | 149.43 | 205    | 286.87 | 5741.7  | 106826    | 239605   |
| main_44db0c5_tp8_fa3                                                        | 116.83 | 145.22 | 147.48 | 192.16 | 288.02 | 5729.52 | 107256    | 240678   |
| main_44db0c5_tp8_flashinfer                                                 | 119.9  | 138.43 | 147.15 | 201.86 | 282.56 | 5651.01 | 105587    | 236684   |

## Mean TPOT (ms)

| Config                                                                      |   cc1 |   cc2 |   cc4 |   cc8 |   cc16 |   cc64 |   cc256 |   cc512 |
|-----------------------------------------------------------------------------|-------|-------|-------|-------|--------|--------|---------|---------|
| debug_fa3-dcp-vectorize_8549455_tp8_dcp8_fa3_a2a_vectorized                 | 11.29 | 12.53 | 14.51 | 17.31 |  22.75 |  43.37 |   83.43 |   85.11 |
| debug_fa3-dcp-vectorize_8549455_tp8_dcp8_fa3_agrs_vectorized                | 11.07 | 12.29 | 14.2  | 16.92 |  22.09 |  43    |   83.33 |   85.03 |
| debug_fa3-dcp-vectorize_b14d632_tp8_dcp8_fa3_a2a_vectorized_083             | 11.27 | 12.54 | 14.49 | 17.31 |  22.73 |  43.41 |   91.21 |  106.06 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_a2a               | 14.91 | 17.48 | 21.23 | 25.17 |  31.14 |  52.13 |   92.42 |   94.47 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_a2a_q_repl        | 16.27 | 18.98 | 22.75 | 26.7  |  32.5  |  53.43 |   69.92 |   70.86 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_agrs              | 14.73 | 17.34 | 21.05 | 24.86 |  30.6  |  51.95 |   92.69 |   94.62 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_a2a        | 11.38 | 12.48 | 14.34 | 17.35 |  22.95 |  44.08 |   91.53 |  119    |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_a2a_q_repl | 12.73 | 13.84 | 15.77 | 18.74 |  24.22 |  45.3  |   60.88 |   61.74 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_agrs       | 11.12 | 12.2  | 14.02 | 16.83 |  22.09 |  43.59 |   91.44 |  120.37 |
| htphan_q-project-replication-rebased_bb19570_tp8_fa3                        |  9.56 | 10.31 | 11.97 | 14.51 |  19.36 |  36.32 |   38.04 |   39.6  |
| htphan_q-project-replication-rebased_bb19570_tp8_flashinfer                 |  9.51 | 10.16 | 11.79 | 14.32 |  19.24 |  35.9  |   36.79 |   37.66 |
| main_44db0c5_tp8_fa3                                                        | 10.17 | 11.12 | 12.76 | 15.17 |  19.91 |  36.04 |   37.13 |   38.58 |
| main_44db0c5_tp8_flashinfer                                                 | 10.17 | 10.96 | 12.56 | 14.97 |  19.72 |  35.48 |   36.38 |   38.51 |

## Mean ITL (ms)

| Config                                                                      |   cc1 |   cc2 |   cc4 |   cc8 |   cc16 |   cc64 |   cc256 |   cc512 |
|-----------------------------------------------------------------------------|-------|-------|-------|-------|--------|--------|---------|---------|
| debug_fa3-dcp-vectorize_8549455_tp8_dcp8_fa3_a2a_vectorized                 | 11.31 | 12.54 | 14.49 | 17.28 |  22.56 |  43.01 |   82.51 |   84.42 |
| debug_fa3-dcp-vectorize_8549455_tp8_dcp8_fa3_agrs_vectorized                | 11.09 | 12.3  | 14.19 | 16.89 |  21.91 |  42.63 |   82.4  |   84.33 |
| debug_fa3-dcp-vectorize_b14d632_tp8_dcp8_fa3_a2a_vectorized_083             | 11.29 | 12.54 | 14.47 | 17.28 |  22.54 |  43.05 |   90.21 |  104.67 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_a2a               | 15.11 | 17.42 | 21.47 | 25.19 |  30.97 |  51.78 |   91.55 |   93.81 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_a2a_q_repl        | 16.48 | 18.91 | 22.99 | 26.72 |  32.32 |  53.06 |   69.55 |   70.54 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_fa3_agrs              | 14.91 | 17.27 | 21.29 | 24.87 |  30.43 |  51.6  |   91.8  |   93.96 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_a2a        | 11.39 | 12.5  | 14.3  | 17.31 |  22.75 |  43.71 |   90.52 |  117.01 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_a2a_q_repl | 12.73 | 13.86 | 15.73 | 18.71 |  24.02 |  44.9  |   60.5  |   61.43 |
| htphan_q-project-replication-rebased_bb19570_tp8_dcp8_flashinfer_agrs       | 11.14 | 12.22 | 13.98 | 16.8  |  21.91 |  43.22 |   90.46 |  118.41 |
| htphan_q-project-replication-rebased_bb19570_tp8_fa3                        |  9.57 | 10.33 | 11.95 | 14.49 |  19.19 |  36.05 |   37.21 |   37.9  |
| htphan_q-project-replication-rebased_bb19570_tp8_flashinfer                 |  9.52 | 10.18 | 11.77 | 14.3  |  19.07 |  35.62 |   36.37 |   37.06 |
| main_44db0c5_tp8_fa3                                                        | 10.18 | 11.13 | 12.74 | 15.15 |  19.75 |  35.77 |   36.72 |   37.33 |
| main_44db0c5_tp8_flashinfer                                                 | 10.18 | 10.98 | 12.54 | 14.95 |  19.55 |  35.21 |   35.96 |   37.01 |
