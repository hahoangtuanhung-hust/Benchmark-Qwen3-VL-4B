# TensorRT-Edge-LLM benchmark on Modal L4

TensorRT-Edge-LLM is Modal-only in this project. Model preparation, engine
caching, serving and measured inference run inside a container pinned to one
NVIDIA L4 (Ada SM89, 24 GB). The local machine only submits work and downloads
artifacts.

## Commands

```powershell
python -m modal setup
make modal-preflight
make modal-smoke
make modal-benchmark
modal run modal/benchmark_entry.py --precision fp16 --smoke
```

`--suite all` runs FP16, FP8, INT4 AWQ and W4A16 sequentially on the same GPU
type. W4A8 is recorded as unsupported because TensorRT-Edge-LLM 0.10.1 exposes
no matching quantization recipe.

## End-to-end flow

1. Validate that Modal allocated exactly one NVIDIA L4 and capture the software
   and hardware environment.
2. Prepare persistent cache, performance dataset and Qwen3-VL checkpoint.
3. Build the native SM89 runtime once and retain it in the cache volume.
4. Quantize the checkpoint when required and persist it in the model volume.
5. Start the OpenAI-compatible TensorRT-Edge server and populate/reuse the
   engine cache for the selected precision.
6. Warm up, then run S1/S2/S3 with CCU1 and S4 with two overlapping users.
7. Sample GPU utilization and VRAM throughout every scenario.
8. Run TextVQA, DocVQA and ChartQA accuracy for a full run.
9. Merge raw requests, GPU samples, accuracy and manifests; generate the report;
   commit volumes; package and download results.

Smoke mode runs one S2 request and one concurrent S4 pair and skips accuracy.
Full mode uses the frozen counts in `configs/benchmark.yaml`.

## Isolation and persistence

Volumes are mounted at `/models`, `/engines`, `/dataset`, `/results` and
`/volumes/cache`. Each invocation gets a new `run_id`, preventing data from an
older run from being appended to new measurements. The worker has
`max_containers=1`; precisions run sequentially, while CCU2 uses two clients
against the same server process and L4.

Downloaded artifacts are written to:

```text
results/modal_tensorrt_edge/tensorrt_edge/runs/<run_id>/
```

Build/startup time is recorded in `engine_manifest.csv` and excluded from
request latency.
