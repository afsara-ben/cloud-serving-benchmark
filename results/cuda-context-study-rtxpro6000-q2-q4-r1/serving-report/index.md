# RTX PRO 6000 Blackwell serving metrics

Status: **incomplete**. 0/30 settings measured; 2 capacity-excluded; 28 unresolved.

2 GPU(s), physical indices 0, 1. Model/format scope: {'70b': ['Q2_K', 'Q4_K_M']}.

One measured run per setting: C discarded warmup requests and 2C measured requests, exactly 512 output tokens each. FP16 KV, prompt reuse off. Throughput is aggregate generated tokens divided by the measured HTTP window. TTFT/TPOT percentiles describe requests within that run; run-to-run variation is unmeasured. Memory is device-total SMI sampling within that same window, in GiB.

[Runtime CSV](runtime.csv) · [All cell statuses](capacity.csv) · [Completion audit](completion-audit.json)

No complete validated measurements available.
