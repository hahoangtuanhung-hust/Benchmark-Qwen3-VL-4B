from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
import yaml

# Modal hydrates this function module at /root/app.py while the complete project
# is mounted under /workspace. Keep helper modules importable in both contexts.
for module_path in ("/workspace/modal", str(Path(__file__).resolve().parent)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from gpu_config import EXPECTED_ARCH, EXPECTED_GPU_LABEL, EXPECTED_SM, GPU_KEY, MODAL_GPU
from image import benchmark_image, common_image
from volumes import MOUNTS, VOLUME_NAMES, cache, dataset, engines, models, results


APP_NAME = "qwen3-vl-tensorrt-edge-benchmark"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PRECISIONS = ("fp16", "fp8", "int8", "int4", "w4a16")
UNSUPPORTED_PRECISIONS = {
    "w4a8": "TensorRT-Edge-LLM 0.10.1 exposes no W4A8 recipe",
}
app = modal.App(APP_NAME)


def _run(command: list[str], *, cwd: str = "/workspace", log: Path | None = None,
         check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print(f"\n[RUN] {' '.join(command)}", flush=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(command)}\n")
    
    process = subprocess.Popen(
        command, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    
    output_lines = []
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)
            if log:
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    
    process.wait()
    output_str = "".join(output_lines)
    
    if check and process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}")
        
    return subprocess.CompletedProcess(process.args, process.returncode, stdout=output_str, stderr="")


def _gpu_info() -> dict[str, Any]:
    query = _run([
        "nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    first = [part.strip() for part in rows[0].split(",")] if rows else []
    return {
        "name": first[0] if len(first) > 0 else "",
        "compute_capability": first[1] if len(first) > 1 else "",
        "vram_mb": first[2] if len(first) > 2 else "",
        "driver": first[3] if len(first) > 3 else "",
        "count": len(rows),
    }


def _prepare_cache_path() -> None:
    cache_path = Path("/cache")
    if cache_path.is_symlink():
        return
    if cache_path.exists():
        if any(cache_path.iterdir()):
            raise RuntimeError("base image recreated a non-empty /cache before persistent cache setup")
        cache_path.rmdir()
    cache_path.symlink_to("/volumes/cache", target_is_directory=True)
    os.environ["XDG_CACHE_HOME"] = "/cache/xdg"


def _hardware_valid(info: dict[str, Any]) -> bool:
    return info["count"] == 1 and "l4" in info["name"].lower()


def _append_modal_environment(path: Path, info: dict[str, Any], status: str) -> None:
    fields = {
        "platform": "modal",
        "modal_app_name": APP_NAME,
        "modal_region": os.environ.get("MODAL_REGION"),
        "modal_gpu_requested": MODAL_GPU,
        "modal_gpu_detected": info["name"],
        "modal_gpu_count": info["count"],
        "modal_container_id": os.environ.get("MODAL_TASK_ID") or os.environ.get("HOSTNAME"),
        "modal_image_id/version": "nvcr.io/nvidia/pytorch:26.04-py3 + Edge-LLM v0.10.1 (sm89)",
        "container_cpu": os.cpu_count(),
        "container_memory": Path("/proc/meminfo").read_text().splitlines()[0] if Path("/proc/meminfo").exists() else None,
        "persistent_volume_name": VOLUME_NAMES,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "run_status": status,
    }
    with path.open("a", encoding="utf-8") as handle:
        for key, value in fields.items():
            handle.write(f"{key}={json.dumps(value, ensure_ascii=True)}\n")


def _preflight(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    info = _gpu_info()
    _run([sys.executable, "-m", "qwen3_vl_benchmark.backends.tensorrt_edge.detect_support",
          "--config", "modal/tensorrt_edge.yaml", "--output-dir", str(output_dir)])
    matrix_path = output_dir / "compatibility_matrix.csv"
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        compatibility = list(csv.DictReader(handle))
    runnable = [row["requested_precision"] for row in compatibility if row["status"].startswith("SUPPORTED")]
    if not _hardware_valid(info):
        status = "INVALID_HARDWARE"
    elif not runnable:
        status = "PREFLIGHT_BLOCKED"
    else:
        status = "PREFLIGHT_OK"
    env_path = output_dir / "environment.txt"
    _append_modal_environment(env_path, info, status)
    (output_dir / "nvidia-smi.txt").write_text(_run(["nvidia-smi"]).stdout, encoding="utf-8")
    (output_dir / "nvidia-smi-q.txt").write_text(_run(["nvidia-smi", "-q"]).stdout, encoding="utf-8")
    return {"status": status, "gpu": info, "runnable_precisions": runnable,
            "output_dir": str(output_dir)}


def _commit_all() -> None:
    for volume in (models, engines, dataset, results, cache):
        volume.commit()


@app.function(
    image=common_image,
    gpu=MODAL_GPU,
    volumes=MOUNTS,
    cpu=8,
    memory=32768,
    timeout=1800,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=60,
)
def preflight_remote() -> dict[str, Any]:
    _prepare_cache_path()
    result = _preflight(Path("/results/tensorrt_edge"))
    results.commit()
    return result


def _ensure_dataset() -> None:
    workspace_data = Path("/workspace/benchmark_data")
    if workspace_data.exists() and not workspace_data.is_symlink():
        shutil.rmtree(workspace_data)
    if not workspace_data.exists():
        workspace_data.symlink_to("/dataset", target_is_directory=True)
    if not Path("/dataset/manifest.jsonl").exists():
        _run([sys.executable, "scripts/prepare_dataset.py", "--project-dir", "."])
        dataset.commit()


def _ensure_result_link() -> None:
    workspace_results = Path("/workspace/results")
    if workspace_results.exists() and not workspace_results.is_symlink():
        shutil.rmtree(workspace_results)
    if not workspace_results.exists():
        workspace_results.symlink_to("/results", target_is_directory=True)


def _ensure_model() -> Path:
    target = Path("/models/Qwen3-VL-4B-Instruct")
    if not (target / "config.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, local_dir=target)
        models.commit()
    return target


def _native_root() -> Path:
    return Path("/cache") / f"TensorRT-Edge-LLM-v0.10.1-sm{EXPECTED_SM}"


def _runtime_env() -> dict[str, str]:
    root = _native_root()
    env = os.environ.copy()
    env["EDGELLM_HOME"] = str(root)
    env["EDGELLM_PLUGIN_PATH"] = str(root / "build/libNvInfer_edgellm_plugin.so")
    # The editable Python package resolves its project root to /opt, while the
    # GPU-specific native build is intentionally persisted under /cache.
    # Point the experimental loader at that persistent pybind output explicitly.
    env["EDGELLM_PYBIND_DIR"] = str(root / "build/pybind")
    env["BUILD_DIR"] = str(root / "build")
    env["PYTHONPATH"] = ":".join(("/workspace", str(root), str(root / "build/pybind")))
    env["LD_LIBRARY_PATH"] = ":".join((
        str(root / "build"), str(root / "build/pybind"), "/usr/lib",
        "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ))
    return env


def _ensure_native_runtime(result_dir: Path) -> float:
    root = _native_root()
    plugin = root / "build/libNvInfer_edgellm_plugin.so"
    bindings = list((root / "build/pybind").glob("*_edgellm_runtime*.so")) if (root / "build/pybind").exists() else []
    if plugin.exists() and bindings:
        return 0.0
    started = time.monotonic()
    if not root.exists():
        shutil.copytree("/opt/TensorRT-Edge-LLM", root, symlinks=True)
    log = result_dir / "native_build.log"
    _run([
        sys.executable, "kernelSrcs/build_cutedsl.py", "--gpu_arch", f"sm_{EXPECTED_SM}",
        "--arch", "x86_64", "--kernels", "fmha,int4_fp16_gemm", "--clean",
    ], cwd=str(root), log=log)
    pybind_dir = _run([sys.executable, "-m", "pybind11", "--cmakedir"]).stdout.strip()
    _run([
        "cmake", "-S", ".", "-B", "build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_PYTHON_BINDINGS=ON", "-DTRT_PACKAGE_DIR=/usr", f"-Dpybind11_DIR={pybind_dir}",
        "-DCUDA_CTK_VERSION=13.2", f"-DCMAKE_CUDA_ARCHITECTURES={EXPECTED_SM}",
        "-DENABLE_CUTE_DSL=fmha;int4_fp16_gemm", f"-DCUTE_DSL_ARTIFACT_TAG=sm_{EXPECTED_SM}",
    ], cwd=str(root), log=log)
    _run([
        "cmake", "--build", "build", "--target", "NvInfer_edgellm_plugin", "_edgellm_runtime",
        "--parallel", str(os.cpu_count() or 8),
    ], cwd=str(root), log=log)
    if not plugin.exists() or not list((root / "build/pybind").glob("*_edgellm_runtime*.so")):
        raise RuntimeError("native TensorRT-Edge runtime build completed without required artifacts")
    cache.commit()
    return time.monotonic() - started


def _checkpoint_for_precision(precision: str, source: Path, result_dir: Path) -> tuple[Path, str, str, str]:
    if precision == "fp16":
        return source, "none", "FP16", "FP16"
    if precision == "w4a8":
        raise ValueError("W4A8 is unsupported by TensorRT-Edge-LLM 0.10.1 quantization methods")
    if precision == "fp8":
        recipe = "fp8"
    elif precision == "int8":
        recipe = "int8_sq"
    else:
        recipe = "int4_awq"
    target = Path("/models/quantized") / precision
    if not (target / "config.json").exists():
        _run([
            "tensorrt-edgellm-quantize", "llm", "--model_dir", str(source),
            "--output_dir", str(target), "--quantization", recipe,
        ], log=result_dir / "quantize.log")
        models.commit()
    if precision == "fp8":
        return target, "ModelOpt FP8", "FP8", "FP8"
    if precision == "int8":
        return target, "ModelOpt INT8_SQ", "INT8", "INT8"
    return target, "ModelOpt INT4_AWQ", "INT4", "FP16"


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def _write_manifest(result_dir: Path, precision: str, checkpoint: Path, recipe: str,
                    weight_dtype: str, activation_dtype: str, engine_dir: Path,
                    build_time: float, status: str, error: str = "") -> None:
    fields = [
        "requested_precision", "actual_precision", "quant_recipe", "model_revision", "model_hash",
        "engine_hash", "visual_encoder_dtype", "language_model_weight_dtype",
        "language_model_activation_dtype", "kv_cache_dtype", "gpu_arch", "TensorRT_version",
        "TensorRT_Edge_LLM_version", "ModelOpt_version", "build_method", "build_command",
        "build_timestamp", "engine_build_time_sec", "engine_size_mb", "status", "error",
    ]
    from importlib.metadata import PackageNotFoundError, version
    def ver(name: str) -> str:
        try: return version(name)
        except PackageNotFoundError: return ""
    size = sum(item.stat().st_size for item in engine_dir.rglob("*") if item.is_file()) if engine_dir.exists() else 0
    row = {
        "requested_precision": precision.upper(),
        "actual_precision": (("INT4_AWQ_W4A16" if precision in {"int4", "w4a16"} else ("INT8_SQ_W8A8" if precision == "int8" else precision.upper()))
                             if status == "BUILT" else ""),
        "quant_recipe": recipe, "model_revision": "main", "model_hash": _sha256_tree(checkpoint),
        "engine_hash": _sha256_tree(engine_dir) if engine_dir.exists() else "", "visual_encoder_dtype": "FP16",
        "language_model_weight_dtype": weight_dtype, "language_model_activation_dtype": activation_dtype,
        "kv_cache_dtype": "FP16", "gpu_arch": EXPECTED_ARCH, "TensorRT_version": ver("tensorrt"),
        "TensorRT_Edge_LLM_version": ver("tensorrt-edgellm"), "ModelOpt_version": ver("nvidia-modelopt"),
        "build_method": "official experimental direct checkpoint server builder",
        "build_command": "tensorrt-edgellm-serve <checkpoint> --cache-dir <persistent-engine-dir>",
        "build_timestamp": datetime.now(timezone.utc).isoformat(), "engine_build_time_sec": round(build_time, 3),
        "engine_size_mb": round(size / 1048576, 3), "status": status, "error": error,
    }
    with (result_dir / "engine_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerow(row)


def _start_server(checkpoint: Path, engine_dir: Path, result_dir: Path) -> tuple[subprocess.Popen[str], float]:
    command = [
        "tensorrt-edgellm-serve", str(checkpoint), "--host", "127.0.0.1", "--port", "8090",
        "--cache-dir", str(engine_dir), "--max-batch-size", "2", "--max-input-len", "4096",
        "--max-kv-cache-capacity", "4096", "--allowed-local-media-path", "/dataset",
        "--served-model-name", MODEL_ID,
    ]
    log_handle = (result_dir / "server.log").open("a", encoding="utf-8")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=str(_native_root()), env=_runtime_env(), stdout=log_handle,
                               stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 1800
    import requests
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"TensorRT-Edge server exited with {process.returncode}")
        try:
            if requests.get("http://127.0.0.1:8090/v1/models", timeout=3).status_code < 400:
                return process, time.monotonic() - started
        except requests.RequestException:
            pass
        time.sleep(2)
    process.terminate(); log_handle.close()
    raise TimeoutError("TensorRT-Edge server did not become healthy in 1800 seconds")


def _gpu_samples(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _enrich_scenario_rows(path: Path, scenario: str, samples: list[dict[str, str]]) -> None:
    if not path.exists() or not samples:
        return
    memory = [float(row["memory_used_mb"]) for row in samples if row.get("memory_used_mb")]
    utilization = [float(row["gpu_util_pct"]) for row in samples if row.get("gpu_util_pct")]
    if not memory:
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0]) if rows else []
    for row in rows:
        if row.get("scenario") == scenario:
            row["idle_vram_mb"] = min(memory)
            row["peak_vram_mb"] = max(memory)
            row["gpu_util_mean"] = sum(utilization) / len(utilization) if utilization else ""
            row["gpu_util_peak"] = max(utilization) if utilization else ""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)


def _normalize_gpu_metrics(source: Path, destination: Path) -> None:
    rows = _gpu_samples(source)
    fields = ["timestamp", "gpu", "memory_used_mb", "gpu_utilization", "memory_utilization", "power_w", "temperature"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({
                "timestamp": row.get("timestamp"), "gpu": row.get("gpu_index"),
                "memory_used_mb": row.get("memory_used_mb"), "gpu_utilization": row.get("gpu_util_pct"),
                "memory_utilization": row.get("memory_util_pct"), "power_w": row.get("power_w"),
                "temperature": row.get("temperature_c"),
            })


def _run_benchmarks(precision: str, actual_precision: str, recipe: str, result_dir: Path, smoke: bool) -> None:
    monitor_raw = result_dir / "gpu_metrics_internal.csv"
    monitor = subprocess.Popen([
        sys.executable, "scripts/gpu_monitor.py", "--output", str(monitor_raw), "--interval", "200",
    ], cwd="/workspace")
    base = [
        sys.executable, "-m", "qwen3_vl_benchmark.benchmark.runner", "--backend", "tensorrt_edge",
        "--server-url", "http://127.0.0.1:8090", "--precision", precision.upper(),
        "--actual-precision", actual_precision, "--quant-recipe", recipe,
        "--gpu", EXPECTED_GPU_LABEL, "--gpu-arch", EXPECTED_ARCH,
        "--output", str(result_dir / "raw_requests.csv"),
    ]
    # Mirror the llama.cpp smoke sample size so both Modal L4 paths expose
    # meaningful variance without becoming a full benchmark.
    scenarios = [("S2", 5), ("S4", 2)] if smoke else [("S1", 10), ("S2", 30), ("S3", 10), ("S4", 30)]
    try:
        for index, (scenario, runs) in enumerate(scenarios):
            sample_start = len(_gpu_samples(monitor_raw))
            command = base + ["--scenario", scenario, "--runs", str(runs)]
            if index:
                command.append("--skip-warmup")
            _run(command, log=result_dir / "benchmark.log")
            scenario_samples = _gpu_samples(monitor_raw)[sample_start:]
            _enrich_scenario_rows(result_dir / "raw_requests.csv", scenario, scenario_samples)
    finally:
        monitor.send_signal(signal.SIGTERM)
        try: monitor.wait(timeout=10)
        except subprocess.TimeoutExpired: monitor.kill()
        _normalize_gpu_metrics(monitor_raw, result_dir / "gpu_metrics.csv")


def _run_accuracy(precision: str, actual_precision: str, result_dir: Path, smoke: bool = False) -> None:
    samples = "2" if smoke else "200"
    mode = "quick" if smoke else "full"
    _run([sys.executable, "scripts/download_accuracy_datasets.py", "--project-dir", ".", "--samples", samples],
         log=result_dir / "accuracy.log")
    output_dir = result_dir / "accuracy_detail"
    _run([
        sys.executable, "scripts/run_accuracy.py", "--project-dir", ".",
        "--server-url", "http://127.0.0.1:8090", "--model-label", precision.upper(),
        "--requested-precision", precision.upper(), "--actual-quant", actual_precision,
        "--mode", mode, "--max-samples", samples, "--output-dir", str(output_dir),
        "--model-name", MODEL_ID,
    ], log=result_dir / "accuracy.log")
    if (output_dir / "summary.csv").exists():
        shutil.copy2(output_dir / "summary.csv", result_dir / "accuracy_summary.csv")
    # Build accuracy.csv (per-sample scores) consumed by report.py
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        acc_fields = ["dataset", "sample_id", "backend", "precision", "prediction", "ground_truth", "metric", "score"]
        with (result_dir / "accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=acc_fields)
            writer.writeheader()
            for line in predictions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                writer.writerow({
                    "dataset": row.get("dataset"), "sample_id": row.get("sample_id"),
                    "backend": "tensorrt_edge", "precision": precision.upper(),
                    "prediction": str(row.get("prediction", ""))[:200],
                    "ground_truth": str(row.get("ground_truth", ""))[:200],
                    "metric": row.get("metric_name"), "score": row.get("score"),
                })
    dataset.commit()


def _merge_csv(inputs: list[Path], output: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in inputs:
        if not path.exists(): continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not fieldnames and reader.fieldnames: fieldnames = reader.fieldnames
            rows.extend(reader)
    if not fieldnames: return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _aggregate_results(root: Path) -> None:
    precision_dirs = [root / precision for precision in PRECISIONS]
    for filename in ("raw_requests.csv", "gpu_metrics.csv", "accuracy.csv", "engine_manifest.csv"):
        _merge_csv([path / filename for path in precision_dirs], root / filename)
    
    accuracy_root = root / "accuracy"
    accuracy_root.mkdir(exist_ok=True)
    _merge_csv([path / "accuracy_summary.csv" for path in precision_dirs], accuracy_root / "summary.csv")
    source_env = next((path / "environment.txt" for path in precision_dirs if (path / "environment.txt").exists()), None)
    source_matrix = next((path / "compatibility_matrix.csv" for path in precision_dirs if (path / "compatibility_matrix.csv").exists()), None)
    if source_env: shutil.copy2(source_env, root / "environment.txt")
    if source_matrix: shutil.copy2(source_matrix, root / "compatibility_matrix.csv")
    _ensure_result_link()
    _run([sys.executable, "-m", "qwen3_vl_benchmark.benchmark.report", "--project-dir", "/workspace",
          "--results-dir", str(root)])


def _unsupported_precision(precision: str, result_dir: Path, run_root: Path, reason: str) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(result_dir, precision, Path("/workspace"), "UNAVAILABLE", "INT4", "INT8",
                    Path("/engines") / precision, 0, "UNSUPPORTED", reason)
    (result_dir / "status.json").write_text(json.dumps({"status": "UNSUPPORTED", "reason": reason}, indent=2))
    _aggregate_results(run_root)
    latest = Path("/results/tensorrt_edge/latest_run.txt")
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(run_root.name + "\n", encoding="utf-8")
    results.commit()
    return {"precision": precision, "status": "UNSUPPORTED", "reason": reason}


@app.function(
    image=benchmark_image,
    gpu=MODAL_GPU,
    volumes=MOUNTS,
    cpu=16,
    memory=65536,
    timeout=86400,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=120,
)
def run_precision_remote(precision: str, smoke: bool = False, run_id: str = "") -> dict[str, Any]:
    _prepare_cache_path()
    precision = precision.lower()
    if precision not in (*PRECISIONS, *UNSUPPORTED_PRECISIONS):
        raise ValueError(f"precision must be one of {(*PRECISIONS, *UNSUPPORTED_PRECISIONS)}")
    safe_run_id = "".join(char for char in run_id if char.isalnum() or char in "-_")
    if not safe_run_id:
        safe_run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_root = Path("/results/tensorrt_edge/runs") / safe_run_id
    result_dir = run_root / precision
    result_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(result_dir)
    if preflight["status"] != "PREFLIGHT_OK":
        results.commit()
        return {"precision": precision, **preflight}
    if precision in UNSUPPORTED_PRECISIONS:
        return _unsupported_precision(precision, result_dir, run_root, UNSUPPORTED_PRECISIONS[precision])

    _ensure_dataset()
    _ensure_result_link()
    shutil.copy2("/workspace/configs/benchmark.yaml", result_dir / "benchmark_config.yaml")
    engine_dir = Path("/engines") / GPU_KEY / precision
    server = None
    started = time.monotonic()
    checkpoint = Path("/models/Qwen3-VL-4B-Instruct")
    recipe = "none" if precision == "fp16" else "UNAVAILABLE"
    weight_dtype = "FP16" if precision == "fp16" else "UNKNOWN"
    activation_dtype = "FP16" if precision == "fp16" else "UNKNOWN"
    try:
        source = _ensure_model()
        checkpoint, recipe, weight_dtype, activation_dtype = _checkpoint_for_precision(
            precision, source, result_dir
        )
        native_build_time = _ensure_native_runtime(result_dir)
        server, engine_build_time = _start_server(checkpoint, engine_dir, result_dir)
        build_time = native_build_time + engine_build_time
        _write_manifest(result_dir, precision, checkpoint, recipe, weight_dtype, activation_dtype,
                        engine_dir, build_time, "BUILT")
        engines.commit()
        actual_precision = "INT4_AWQ_W4A16" if precision in {"int4", "w4a16"} else ("INT8_SQ_W8A8" if precision == "int8" else precision.upper())
        _run_benchmarks(precision, actual_precision, recipe, result_dir, smoke)
        _run_accuracy(precision, actual_precision, result_dir, smoke)
        status = "SMOKE_COMPLETE" if smoke else "BENCHMARK_COMPLETE"
        (result_dir / "status.json").write_text(json.dumps({"status": status}, indent=2))
    except Exception as exc:
        _write_manifest(result_dir, precision, checkpoint, recipe, weight_dtype, activation_dtype,
                        engine_dir, time.monotonic() - started, "RUNTIME_FAILED", str(exc))
        (result_dir / "status.json").write_text(json.dumps({"status": "RUNTIME_FAILED", "error": str(exc)}, indent=2))
        status = "RUNTIME_FAILED"
    finally:
        if server and server.poll() is None:
            server.terminate()
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired: server.kill()
        try:
            _aggregate_results(run_root)
        finally:
            latest = Path("/results/tensorrt_edge/latest_run.txt")
            latest.parent.mkdir(parents=True, exist_ok=True)
            latest.write_text(safe_run_id + "\n", encoding="utf-8")
            _commit_all()
    return {"precision": precision, "status": status, "run_id": safe_run_id,
            "result_dir": str(result_dir)}


@app.function(image=common_image, volumes={"/results": results}, timeout=600)
def pack_results_remote(run_id: str = "") -> bytes:
    archive = Path("/tmp/tensorrt_edge_results.zip")
    # Files restored from a Modal Volume may retain Unix epoch timestamps.
    # ZIP cannot represent dates before 1980, so clamp them instead of failing
    # after an otherwise successful remote benchmark/preflight.
    with zipfile.ZipFile(
        archive, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False
    ) as bundle:
        root = Path("/results")
        if run_id:
            sources = [root / "tensorrt_edge" / "runs" / run_id]
        else:
            # Preflight writes top-level evidence. Avoid sending the complete run
            # history over gRPC, which can grow large enough to drop the stream.
            preflight_root = root / "tensorrt_edge"
            sources = [path for path in preflight_root.iterdir() if path.name != "runs"] if preflight_root.exists() else []
        for source in sources:
            paths = source.rglob("*") if source.is_dir() else [source]
            for path in paths:
                if path.is_file():
                    bundle.write(path, path.relative_to(root))
    return archive.read_bytes()
