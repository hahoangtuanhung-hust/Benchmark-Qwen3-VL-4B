# Local CPU Benchmark

This flow is separate from the Modal GPU benchmark. It does not call
`run_benchmark.sh`, does not use `results/`, and does not change
`scripts/run_on_modal.py`.

## First run

Install the existing Python dependencies and make sure CMake plus a C/C++
compiler are available. On Windows with MSYS2 UCRT64, `gcc`, `g++`, and
`mingw32-make` are sufficient; the script automatically selects the
`MinGW Makefiles` generator when Visual C++ is not present:

```bash
pip install -r requirements.txt
python scripts/run_cpu_benchmark.py \
  --model models/q4_k_m/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  --mmproj models/mmproj/mmproj-Qwen3-VL-4B-Instruct-f16.gguf \
  --threads 8 --threads-batch 8
```

The script builds llama.cpp with `GGML_CUDA=OFF` in `llama.cpp/build-cpu/`.
The existing CUDA build in `llama.cpp/build/` is untouched.

On Windows, CMake can be installed with:

```powershell
winget install --id Kitware.CMake -e
```

Close and reopen PowerShell after installation, then verify with
`cmake --version`.

## Smoke test

```bash
python scripts/run_cpu_benchmark.py --smoke \
  --model models/q4_k_m/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  --mmproj models/mmproj/mmproj-Qwen3-VL-4B-Instruct-f16.gguf \
  --threads 8 --threads-batch 8
```

Use `--skip-build` after the CPU binary has already been compiled. Use
`--ccu2` to add the concurrent two-user scenario and `--accuracy` to run the
accuracy suite after performance. The full CPU run defaults to S1, S2, and S3;
`--smoke` runs S1 only.

## Reduce CPU sample counts

The CPU runner exposes sample-count overrides so a quick iteration does not
need the full matrix. Defaults are unchanged: S1=10, S2=30, S3=10, S5=10,
with 5 warmup requests. These options only affect the CPU entrypoint; GPU
commands keep their existing defaults.

For a short smoke run on a laptop:

```powershell
python scripts/run_cpu_benchmark.py --smoke --skip-build `
  --model models/q4_k_m/Qwen3-VL-4B-Instruct-Q4_K_M.gguf `
  --mmproj models/mmproj/mmproj-Qwen3-VL-4B-Instruct-f16.gguf `
  --threads 4 --threads-batch 4 `
  --s1-runs 3 --warmup-requests 1
```

For a reduced full performance run, use explicit counts and optionally cap
accuracy samples per dataset:

```powershell
python scripts/run_cpu_benchmark.py --skip-build `
  --model models/q4_k_m/Qwen3-VL-4B-Instruct-Q4_K_M.gguf `
  --mmproj models/mmproj/mmproj-Qwen3-VL-4B-Instruct-f16.gguf `
  --threads 4 --threads-batch 4 `
  --scenarios S1 S2 S3 `
  --s1-runs 3 --s2-runs 5 --s3-runs 3 --warmup-requests 1 `
  --ccu2 --ccu2-pairs 3 `
  --accuracy --accuracy-mode quick --accuracy-samples 5
```

`--warmup-requests 0` disables warmup. `--accuracy-samples 0` keeps the
selected quick/full accuracy default; accuracy datasets must already be
available when `--accuracy` is used.

## Compare a quantized model with FP16

Pass the quantized model as `--model` and enable the baseline with
`--include-fp16`. The FP16 file is auto-detected from `models/fp16/`, or can be
provided explicitly with `--fp16-model`:

```bash
python scripts/run_cpu_benchmark.py --smoke --skip-build \
  --model models/q4_k_m/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  --include-fp16 \
  --fp16-model models/fp16/Qwen3-VL-4B-Instruct-F16.gguf \
  --mmproj models/mmproj/mmproj-Qwen3-VL-4B-Instruct-f16.gguf \
  --threads 4 --threads-batch 4
```

Both models run sequentially in one CPU run. The raw CSVs and server metrics
are labeled by model, and the final report compares both variants.

Each invocation creates a new folder:

```text
results_cpu/
  latest_run.txt
  run_20260924T120000Z_cpu_Q4_K_M_1234/
    run_metadata.json
    environment.json
    server.log
    raw/
    server_metrics/
    report/
    charts/
```

The report is at `results_cpu/<run_id>/report/benchmark_summary.md`.

Each CPU run also records utilization for the `llama-server` process only:

```text
results_cpu/<run_id>/cpu_metrics/
  Q4_K_M_llama_server_cpu.csv
  Q4_K_M_llama_server_cpu_summary.json
```

The monitor starts after the server health check and immediately before the
benchmark suite, then stops when the suite ends. It excludes the Python
runner, CMake, dataset preparation, and other processes. The summary reports
CPU usage as a percentage of the whole machine and as a percentage of one
logical core. Adjust the sampling interval with `--cpu-sample-interval`.
