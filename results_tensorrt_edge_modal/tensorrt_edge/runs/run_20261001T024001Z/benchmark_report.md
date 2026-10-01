# TensorRT-Edge-LLM Benchmark Report

## Status

Measured results are available.

See `environment.txt` and `compatibility_matrix.csv` for detected blockers and support evidence.

## Detailed performance by precision, scenario and concurrency

| Precision | Quant recipe | Scenario | CCU | Runs | Output tokens mean | TTFT mean | TTFT P50 | TTFT P95 | TTFT P99 | TTFV mean | TTFV P50 | TTFV P95 | TTFV P99 | E2E mean | E2E P50 | E2E P95 | E2E P99 | Prefill TPS mean | Prefill TPS median | Prefill TPS std | Decode TPS mean | Decode TPS median | Decode TPS std | TPOT | ITL mean | ITL P50 | ITL P95 | ITL P99 | Idle VRAM | Peak VRAM | KV Cache | GPU util mean | GPU util peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | none | S2 | 1 | 5 | 164.20 | 123.94 | 124.10 | 131.24 | 131.84 | 123.94 | 124.10 | 131.24 | 131.84 | 6039.94 | 6542.56 | 7553.17 | 7701.74 | N/A | N/A | N/A | 27.78 | 27.76 | 0.08 | 36.24 | N/A | N/A | N/A | N/A | 11190.00 | 11210.00 | N/A | 93.20 | 100.00 |
| FP16 | none | S4 | 2 | 4 | 177.50 | 3688.70 | 3398.85 | 7661.21 | 7802.09 | 3688.70 | 3398.85 | 7661.21 | 7802.09 | 10090.97 | 10233.18 | 13252.47 | 13326.05 | N/A | N/A | N/A | 27.76 | 27.77 | 0.04 | 36.27 | N/A | N/A | N/A | N/A | 11210.00 | 11210.00 | N/A | 93.31 | 97.00 |

## GPU telemetry

| Peak VRAM (MB) | Peak utilization (%) | Peak power (W) | Peak temperature (C) |
| --- | --- | --- | --- |
| 11210.00 | 100.00 | 74.41 | 83.00 |

## Accuracy

No accuracy samples were recorded.

## Engine and quantization manifest

| requested_precision | actual_precision | quant_recipe | engine_build_time_sec | engine_size_mb | status | error |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 | FP16 | none | 18.186 | 44.653 | BUILT | N/A |

## Measurement policy

Visual encoder and KV cache are fixed to FP16 for the main benchmark. Missing runtime metrics remain null; stream chunks are not represented as token-level ITL unless the server emits one token per event.
