# RTX PRO 6000 Blackwell serving metrics

Status: **incomplete**. 18/30 settings measured; 2 capacity-excluded; 10 unresolved.

2 GPU(s), physical indices 0, 1. Model/format scope: {'70b': ['Q2_K', 'Q4_K_M']}.

One measured run per setting: C discarded warmup requests and 2C measured requests, exactly 512 output tokens each. FP16 KV, prompt reuse off. Throughput is aggregate generated tokens divided by the measured HTTP window. TTFT/TPOT percentiles describe requests within that run; run-to-run variation is unmeasured. Memory is device-total SMI sampling within that same window, in GiB.

[Runtime CSV](runtime.csv) · [All cell statuses](capacity.csv) · [Completion audit](completion-audit.json)

| Model | Format | Clients | Input tokens | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms | TPOT p95 ms | Generated tok/s | GPU 0 peak GiB | GPU 1 peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 70b | Q2_K | 8 | 2048 | 3352.13 | 8727.79 | 47.55 | 51.62 | 147.98 | 16.46 | 16.50 |
| 70b | Q2_K | 8 | 4096 | 5576.16 | 16354.92 | 63.44 | 68.16 | 107.67 | 19.03 | 18.95 |
| 70b | Q2_K | 8 | 8192 | 7981.26 | 32465.15 | 100.74 | 105.38 | 68.78 | 24.17 | 23.84 |
| 70b | Q2_K | 8 | 16384 | 12208.69 | 67268.88 | 180.46 | 184.61 | 39.07 | 34.45 | 33.62 |
| 70b | Q2_K | 8 | 32768 | 25516.83 | 145768.39 | 347.63 | 354.08 | 20.07 | 55.01 | 53.18 |
| 70b | Q2_K | 16 | 2048 | 3357.20 | 17056.10 | 75.06 | 77.97 | 194.82 | 19.99 | 19.86 |
| 70b | Q2_K | 16 | 4096 | 5940.49 | 32551.27 | 110.59 | 114.15 | 129.88 | 25.12 | 24.74 |
| 70b | Q2_K | 16 | 8192 | 8336.50 | 65611.83 | 188.88 | 192.79 | 77.03 | 35.38 | 34.50 |
| 70b | Q2_K | 16 | 16384 | 13807.11 | 136602.50 | 350.37 | 357.00 | 41.68 | 55.91 | 54.04 |
| 70b | Q4_K_M | 8 | 2048 | 2552.55 | 6539.71 | 51.74 | 54.73 | 140.99 | 24.02 | 23.75 |
| 70b | Q4_K_M | 8 | 4096 | 4263.00 | 12248.22 | 64.65 | 68.08 | 109.66 | 26.59 | 26.20 |
| 70b | Q4_K_M | 8 | 8192 | 5275.40 | 24512.04 | 95.33 | 97.84 | 75.23 | 31.73 | 31.09 |
| 70b | Q4_K_M | 8 | 16384 | 9446.98 | 51534.65 | 157.92 | 161.81 | 44.98 | 42.01 | 40.87 |
| 70b | Q4_K_M | 8 | 32768 | 19912.30 | 114753.49 | 295.70 | 301.95 | 23.64 | 62.58 | 60.43 |
| 70b | Q4_K_M | 16 | 2048 | 2564.10 | 12878.78 | 73.13 | 75.35 | 203.46 | 27.54 | 27.10 |
| 70b | Q4_K_M | 16 | 4096 | 4481.90 | 24418.66 | 100.75 | 103.51 | 144.20 | 32.68 | 31.99 |
| 70b | Q4_K_M | 16 | 8192 | 6403.05 | 49627.19 | 162.74 | 166.19 | 89.67 | 42.94 | 41.75 |
| 70b | Q4_K_M | 16 | 16384 | 10777.09 | 104792.72 | 290.77 | 298.05 | 49.98 | 63.48 | 61.28 |
