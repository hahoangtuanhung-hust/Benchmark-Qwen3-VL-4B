#!/usr/bin/env python3
"""Run a local CPU-only llama.cpp benchmark.

This entrypoint is intentionally independent from ``run_benchmark.sh`` and
``scripts/run_on_modal.py``. It uses a separate llama.cpp build directory and
stores every run below ``results_cpu/`` so a local CPU run cannot overwrite a
Modal GPU run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:  # Optional at runtime so an old environment can still run the benchmark.
    psutil = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_DIR / "results_cpu"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[CMD] " + " ".join(str(part) for part in command))
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    # Windows may default text files to cp1252. Benchmark answers can contain
    # Unicode, so all Python child processes in the CPU flow must use UTF-8.
    command_env["PYTHONUTF8"] = "1"
    command_env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, cwd=str(cwd), env=command_env, check=True)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class ProcessCpuMonitor:
    """Sample only the llama-server process during measured benchmark work."""

    def __init__(self, pid: int, output_dir: Path, label: str, interval: float) -> None:
        if psutil is None:
            raise RuntimeError("psutil is not installed")
        self.process = psutil.Process(pid)
        self.output_dir = output_dir
        self.label = sanitize_label(label)
        self.interval = interval
        self.logical_cpus = max(psutil.cpu_count() or os.cpu_count() or 1, 1)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples: list[dict] = []
        self.started_at = ""
        self.finished_at = ""
        self.started_monotonic = 0.0
        self.started_cpu_time = 0.0
        self.last_error = ""

    def _cpu_time(self) -> float:
        times = self.process.cpu_times()
        return float(times.user + times.system)

    def _sample(self) -> None:
        try:
            cpu_one_core = float(self.process.cpu_percent(None))
            cpu_time = self._cpu_time()
            elapsed = max(time.monotonic() - self.started_monotonic, 0.0)
            self.samples.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": elapsed,
                    "pid": self.process.pid,
                    "cpu_percent_one_core": cpu_one_core,
                    "cpu_percent_machine": cpu_one_core / self.logical_cpus,
                    "cpu_time_s": cpu_time,
                    "rss_mb": self.process.memory_info().rss / (1024 * 1024),
                    "threads": self.process.num_threads(),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            self.last_error = str(exc)
            self.stop_event.set()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_monotonic = time.monotonic()
        self.started_cpu_time = self._cpu_time()
        # Prime psutil's interval-based CPU counter without recording startup work.
        self.process.cpu_percent(None)
        self.thread = threading.Thread(target=self._run, name=f"cpu-monitor-{self.process.pid}", daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(self.interval * 2, 2.0))
        if not self.stop_event.is_set() or not self.samples:
            self._sample()
        self.finished_at = datetime.now(timezone.utc).isoformat()

        csv_path = self.output_dir / f"{self.label}_llama_server_cpu.csv"
        fields = [
            "timestamp",
            "elapsed_s",
            "pid",
            "cpu_percent_one_core",
            "cpu_percent_machine",
            "cpu_time_s",
            "rss_mb",
            "threads",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)

        wall_time = max(time.monotonic() - self.started_monotonic, 0.0)
        try:
            final_cpu_time = self._cpu_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            final_cpu_time = self.samples[-1]["cpu_time_s"] if self.samples else self.started_cpu_time
        cpu_time = max(final_cpu_time - self.started_cpu_time, 0.0)
        machine_util = (cpu_time / wall_time / self.logical_cpus * 100.0) if wall_time else None
        machine_samples = [float(row["cpu_percent_machine"]) for row in self.samples]
        one_core_samples = [float(row["cpu_percent_one_core"]) for row in self.samples]
        summary = {
            "status": "completed" if not self.last_error else "process_exited",
            "pid": self.process.pid,
            "measurement_scope": "llama-server process only; excludes Python runner and other processes",
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_time_s": wall_time,
            "process_cpu_time_s": cpu_time,
            "logical_cpu_count": self.logical_cpus,
            "sample_interval_s": self.interval,
            "sample_count": len(self.samples),
            "cpu_percent_machine_mean": sum(machine_samples) / len(machine_samples) if machine_samples else None,
            "cpu_percent_machine_p95": percentile(machine_samples, 95),
            "cpu_percent_machine_max": max(machine_samples) if machine_samples else None,
            "cpu_percent_one_core_mean": sum(one_core_samples) / len(one_core_samples) if one_core_samples else None,
            "cpu_percent_one_core_p95": percentile(one_core_samples, 95),
            "cpu_percent_one_core_max": max(one_core_samples) if one_core_samples else None,
            "cpu_utilization_machine_mean": machine_util,
            "rss_mb_max": max((float(row["rss_mb"]) for row in self.samples), default=None),
            "error": self.last_error,
            "csv": str(csv_path),
        }
        summary_path = self.output_dir / f"{self.label}_llama_server_cpu_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_json"] = str(summary_path)
        return summary


def start_cpu_monitor(pid: int, output_dir: Path, label: str, interval: float) -> ProcessCpuMonitor | None:
    if psutil is None:
        print("[WARN] psutil is not installed; CPU process monitoring is disabled")
        return None
    try:
        monitor = ProcessCpuMonitor(pid, output_dir, label, interval)
        monitor.start()
        print(f"[INFO] CPU monitor started for llama-server PID={pid} ({interval:g}s interval)")
        return monitor
    except (OSError, RuntimeError, psutil.Error) as exc:
        print(f"[WARN] Could not start CPU monitor: {exc}")
        return None


def find_binary(build_dir: Path, name: str) -> Path:
    candidates = [
        build_dir / "bin" / name,
        build_dir / "bin" / f"{name}.exe",
        build_dir / "Release" / "bin" / name,
        build_dir / "Release" / "bin" / f"{name}.exe",
        build_dir / "Release" / name,
        build_dir / "Release" / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {name}; searched: {searched}")


def build_cpu_llama_cpp(
    source_dir: Path,
    build_dir: Path,
    jobs: int,
    skip_build: bool,
    cmake_executable: str,
    generator: str,
) -> Path:
    server_name = "llama-server"
    if skip_build:
        return find_binary(build_dir, server_name)

    if not (source_dir / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"llama.cpp source not found: {source_dir}")

    configure = [
        cmake_executable,
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DGGML_CUDA=OFF",
        "-DGGML_NATIVE=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if generator:
        configure[1:1] = ["-G", generator]
    run_command(configure, cwd=PROJECT_DIR)

    build = [
        cmake_executable,
        "--build",
        str(build_dir),
        "--config",
        "Release",
        "--target",
        "llama-server",
        "llama-cli",
        "llama-quantize",
    ]
    if jobs > 0:
        build.extend(["--parallel", str(jobs)])
    run_command(build, cwd=PROJECT_DIR)
    return find_binary(build_dir, server_name)


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "model"


def infer_model_label(model_path: Path) -> str:
    name = model_path.stem
    for label in ("Q5_K_M", "Q4_K_M", "Q4_K_S", "Q8_0", "Q6_K", "Q4_0", "Q4_1", "IQ4_XS", "F16", "BF16"):
        if label.lower() in name.lower():
            return label
    return sanitize_label(name)


def resolve_model(project_dir: Path, requested: str) -> Path:
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = (project_dir / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path

    candidates = sorted((project_dir / "models").glob("**/*.gguf")) if (project_dir / "models").is_dir() else []
    candidates = [path for path in candidates if "mmproj" not in path.name.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("No local model found. Pass --model path/to/model.gguf.")
    choices = "\n".join(f"  - {path}" for path in candidates)
    raise RuntimeError(f"Multiple local models found; pass --model explicitly:\n{choices}")


def resolve_fp16_model(project_dir: Path, requested: str) -> Path:
    """Resolve the FP16 baseline without changing the primary model choice."""
    if requested:
        return resolve_model(project_dir, requested)

    candidates = sorted((project_dir / "models" / "fp16").glob("*.gguf")) if (project_dir / "models" / "fp16").is_dir() else []
    candidates = [path for path in candidates if "mmproj" not in path.name.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "FP16 baseline not found under models/fp16. "
            "Pass --fp16-model path/to/Qwen3-VL-4B-Instruct-F16.gguf."
        )
    choices = "\n".join(f"  - {path}" for path in candidates)
    raise RuntimeError(f"Multiple FP16 models found; pass --fp16-model explicitly:\n{choices}")


def resolve_mmproj(project_dir: Path, requested: str) -> Path:
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = (project_dir / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"mmproj file not found: {path}")
        return path

    candidates = sorted((project_dir / "models").glob("**/*mmproj*.gguf")) if (project_dir / "models").is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("No mmproj GGUF found. Pass --mmproj path/to/mmproj.gguf.")
    choices = "\n".join(f"  - {path}" for path in candidates)
    raise RuntimeError(f"Multiple mmproj files found; pass --mmproj explicitly:\n{choices}")


def free_port(port: int) -> int:
    if port != 0:
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def git_commit(source_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_environment(
    run_dir: Path,
    args: argparse.Namespace,
    server_binary: Path,
    models: list[Path],
    mmproj: Path,
) -> None:
    environment = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "llama_server": str(server_binary),
        "llama_cpp_commit": git_commit(args.llama_cpp_dir),
        "models": [str(model) for model in models],
        "mmproj": str(mmproj),
        "threads": args.threads,
        "threads_batch": args.threads_batch,
        "context": args.context,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "gpu_layers": 0,
    }
    with (run_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2)


def start_server(
    server_binary: Path,
    model: Path,
    mmproj: Path,
    run_dir: Path,
    host: str,
    port: int,
    slots: int,
    context: int,
    threads: int,
    threads_batch: int,
    batch_size: int,
    ubatch_size: int,
    model_label: str,
) -> tuple[subprocess.Popen, Path]:
    log_path = run_dir / f"server_{sanitize_label(model_label)}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    thread_args: list[str] = []
    if threads > 0:
        thread_args.extend(["--threads", str(threads)])
    if threads_batch > 0:
        thread_args.extend(["--threads-batch", str(threads_batch)])

    command = [
        str(server_binary),
        "--model",
        str(model),
        "--mmproj",
        str(mmproj),
        "--host",
        host,
        "--port",
        str(port),
        "--n-gpu-layers",
        "0",
        "--ctx-size",
        str(context * slots),
        "--parallel",
        str(slots),
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(ubatch_size),
        *thread_args,
        "--flash-attn",
        "off",
        "--metrics",
        "--slots",
        "--no-cache-prompt",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
    ]
    print("[INFO] Starting CPU llama-server:")
    print("       " + " ".join(command))
    server_env = os.environ.copy()
    # Keep accidental CUDA-enabled host binaries from using the GPU. The CPU
    # build is already configured with GGML_CUDA=OFF; this is an extra guard.
    server_env["CUDA_VISIBLE_DEVICES"] = "-1"
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_DIR),
        env=server_env,
    )
    log_handle.close()
    (run_dir / "server.pid").write_text(str(process.pid), encoding="utf-8")
    return process, log_path


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"[INFO] Stopping CPU llama-server (PID={process.pid})...")
    try:
        process.terminate()
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_for_health(host: str, port: int, timeout: int) -> None:
    wait_script = PROJECT_DIR / "scripts" / "wait_for_server.py"
    run_command(
        [sys.executable, str(wait_script), "--host", host, "--port", str(port), "--timeout", str(timeout)],
        cwd=PROJECT_DIR,
    )


def collect_metrics(server_url: str, output_dir: Path, label: str) -> None:
    script = PROJECT_DIR / "scripts" / "collect_server_metrics.py"
    run_command(
        [sys.executable, str(script), "--server-url", server_url, "--output-dir", str(output_dir), "--label", label],
        cwd=PROJECT_DIR,
    )


def run_suite(
    args: argparse.Namespace,
    run_dir: Path,
    model: Path,
    model_label: str,
    server_url: str,
) -> None:
    label = model_label
    commit = git_commit(args.llama_cpp_dir)
    common = [
        "--config",
        str(PROJECT_DIR / "configs" / "benchmark.yaml"),
        "--project-dir",
        str(PROJECT_DIR),
        "--server-url",
        server_url,
        "--model-label",
        label,
        "--requested-precision",
        label,
        "--actual-quant",
        label,
        "--gpu-name",
        "CPU",
        "--llama-commit",
        commit,
        "--context-per-slot",
        str(args.context),
    ]
    scenarios = args.scenarios if not args.smoke else ["S1"]
    ccu1 = PROJECT_DIR / "scripts" / "benchmark_ccu1.py"
    run_command(
        [
            sys.executable,
            str(ccu1),
            *common,
            "--output",
            str(run_dir / "raw" / f"{sanitize_label(label)}_ccu1_requests.csv"),
            "--scenarios",
            *scenarios,
            "--s1-runs",
            str(args.s1_runs),
            "--s2-runs",
            str(args.s2_runs),
            "--s3-runs",
            str(args.s3_runs),
            "--s5-runs",
            str(args.s5_runs),
            "--warmup-requests",
            str(args.warmup_requests),
        ],
        cwd=PROJECT_DIR,
    )

    if args.ccu2 and not args.smoke:
        ccu2 = PROJECT_DIR / "scripts" / "benchmark_ccu2.py"
        run_command(
            [
                sys.executable,
                str(ccu2),
                *common,
                "--output",
                str(run_dir / "raw" / f"{sanitize_label(label)}_ccu2_requests.csv"),
                "--pairs",
                str(args.ccu2_pairs),
            ],
            cwd=PROJECT_DIR,
        )

    if args.accuracy:
        accuracy = PROJECT_DIR / "scripts" / "run_accuracy.py"
        accuracy_command = [
            sys.executable,
            str(accuracy),
            "--project-dir",
            str(PROJECT_DIR),
            "--server-url",
            server_url,
            "--model-label",
            label,
            "--actual-quant",
            label,
            "--mode",
            args.accuracy_mode,
            "--output-dir",
            str(run_dir / "accuracy"),
        ]
        if args.accuracy_samples > 0:
            accuracy_command.extend(["--max-samples", str(args.accuracy_samples)])
        run_command(accuracy_command, cwd=PROJECT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated local CPU-only VLM benchmark")
    parser.add_argument("--model", default="", help="Local model GGUF path")
    parser.add_argument("--include-fp16", action="store_true", help="Also benchmark the FP16 baseline")
    parser.add_argument("--fp16-model", default="", help="FP16 baseline GGUF path (used with --include-fp16)")
    parser.add_argument("--mmproj", default="", help="Local vision projector GGUF path")
    parser.add_argument("--model-label", default="", help="Label used in CSV/report")
    parser.add_argument("--llama-cpp-dir", type=Path, default=PROJECT_DIR / "llama.cpp")
    parser.add_argument("--build-dir", type=Path, default=None, help="CPU build dir (default: llama.cpp/build-cpu)")
    parser.add_argument("--cmake", default="cmake", help="CMake executable or absolute path")
    parser.add_argument("--generator", default="", help="Optional CMake generator, e.g. 'MinGW Makefiles'")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--port", type=int, default=0, help="Server port; 0 selects a free port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--threads", type=int, default=0, help="Inference threads; 0 uses llama.cpp default")
    parser.add_argument("--threads-batch", type=int, default=0, help="Prompt-processing threads; 0 uses default")
    parser.add_argument("--context", type=int, default=4096, help="Context size per slot")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=0, help="CPU build parallelism; 0 uses CMake default")
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--ccu2", action="store_true", help="Also run the concurrent two-user scenario")
    parser.add_argument("--ccu2-pairs", type=int, default=30)
    parser.add_argument("--s1-runs", type=int, default=10, help="Measured S1 request count")
    parser.add_argument("--s2-runs", type=int, default=30, help="Measured S2 request count")
    parser.add_argument("--s3-runs", type=int, default=10, help="Measured S3 request count")
    parser.add_argument("--s5-runs", type=int, default=10, help="Measured S5 request count")
    parser.add_argument("--warmup-requests", type=int, default=5, help="Warmup requests before measurement; 0 disables warmup")
    parser.add_argument("--accuracy", action="store_true", help="Run the accuracy suite after performance")
    parser.add_argument("--accuracy-mode", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--accuracy-samples",
        type=int,
        default=0,
        help="Maximum accuracy samples per dataset; 0 uses the mode default",
    )
    parser.add_argument(
        "--cpu-sample-interval",
        type=float,
        default=0.5,
        help="CPU monitor sampling interval in seconds",
    )
    parser.add_argument("--smoke", action="store_true", help="Run S1 only")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--health-timeout", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for option_name in ("s1_runs", "s2_runs", "s3_runs", "s5_runs"):
        if getattr(args, option_name) <= 0:
            print(f"[ERROR] --{option_name.replace('_', '-')} must be positive", file=sys.stderr)
            return 2
    if args.warmup_requests < 0:
        print("[ERROR] --warmup-requests cannot be negative", file=sys.stderr)
        return 2
    if args.accuracy_samples < 0:
        print("[ERROR] --accuracy-samples cannot be negative", file=sys.stderr)
        return 2
    if args.cpu_sample_interval <= 0:
        print("[ERROR] --cpu-sample-interval must be positive", file=sys.stderr)
        return 2
    args.llama_cpp_dir = args.llama_cpp_dir.expanduser().resolve()
    if args.build_dir is None:
        args.build_dir = args.llama_cpp_dir / "build-cpu"
    else:
        args.build_dir = args.build_dir.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()

    try:
        primary_model = resolve_model(PROJECT_DIR, args.model)
        models = [primary_model]
        if args.include_fp16 or args.fp16_model:
            fp16_model = resolve_fp16_model(PROJECT_DIR, args.fp16_model)
            if fp16_model not in models:
                models.append(fp16_model)
        mmproj = resolve_mmproj(PROJECT_DIR, args.mmproj)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not args.skip_build:
        cmake_executable = shutil.which(args.cmake)
        if cmake_executable is None and Path(args.cmake).is_file():
            cmake_executable = str(Path(args.cmake).expanduser().resolve())
        if cmake_executable is None:
            print(
                "[ERROR] CMake was not found in PATH. Install CMake and reopen PowerShell, "
                "or pass --cmake C:/path/to/cmake.exe.",
                file=sys.stderr,
            )
            return 2
    else:
        cmake_executable = args.cmake

    generator = args.generator
    if not generator and os.name == "nt" and shutil.which("cl") is None and shutil.which("mingw32-make"):
        generator = "MinGW Makefiles"
        print(f"[INFO] Using Windows MinGW generator: {generator}")
    model_specs = []
    for index, model in enumerate(models):
        custom_label = args.model_label if index == 0 else ""
        model_specs.append((model, sanitize_label(custom_label or infer_model_label(model))))
    run_label = "compare" if len(model_specs) > 1 else model_specs[0][1]
    run_id = f"run_{utc_stamp()}_cpu_{run_label}_{os.getpid()}"
    run_dir = args.results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "raw").mkdir()
    (run_dir / "server_metrics").mkdir()
    (run_dir / "cpu_metrics").mkdir()

    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "models": [{"path": str(model), "label": label} for model, label in model_specs],
        "mmproj": str(mmproj),
        "server_urls": {},
        "results_dir": str(run_dir),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    server_process = None
    exit_code = 0

    try:
        server_binary = build_cpu_llama_cpp(
            args.llama_cpp_dir,
            args.build_dir,
            args.jobs,
            args.skip_build,
            cmake_executable,
            generator,
        )
        write_environment(run_dir, args, server_binary, [model for model, _ in model_specs], mmproj)

        if not args.skip_dataset:
            dataset_script = PROJECT_DIR / "scripts" / "prepare_dataset.py"
            run_command([sys.executable, str(dataset_script), "--project-dir", str(PROJECT_DIR)], cwd=PROJECT_DIR)

        for model, model_label in model_specs:
            model_port = free_port(args.port)
            server_process, log_path = start_server(
                server_binary,
                model,
                mmproj,
                run_dir,
                args.host,
                model_port,
                2 if args.ccu2 else 1,
                args.context,
                args.threads,
                args.threads_batch,
                args.batch_size,
                args.ubatch_size,
                model_label,
            )
            metadata.setdefault("server_logs", {})[model_label] = str(log_path)
            wait_for_health(args.host, model_port, args.health_timeout)
            server_url = f"http://{args.host}:{model_port}"
            metadata["server_urls"][model_label] = server_url
            collect_metrics(server_url, run_dir / "server_metrics", f"cpu_{model_label}_before")
            cpu_monitor = start_cpu_monitor(
                server_process.pid,
                run_dir / "cpu_metrics",
                model_label,
                args.cpu_sample_interval,
            )
            try:
                run_suite(args, run_dir, model, model_label, server_url)
            finally:
                if cpu_monitor is not None:
                    cpu_summary = cpu_monitor.stop()
                    metadata.setdefault("cpu_monitor", {})[model_label] = cpu_summary
                    print(
                        "[INFO] CPU usage (llama-server only): "
                        f"mean={cpu_summary['cpu_percent_machine_mean']:.1f}% of machine, "
                        f"p95={cpu_summary['cpu_percent_machine_p95']:.1f}%"
                        if cpu_summary["cpu_percent_machine_mean"] is not None
                        else "[INFO] CPU usage summary has no samples"
                    )
            collect_metrics(server_url, run_dir / "server_metrics", f"cpu_{model_label}_after")
            stop_server(server_process)
            server_process = None

        summarize = PROJECT_DIR / "scripts" / "summarize.py"
        run_command(
            [sys.executable, str(summarize), "--project-dir", str(PROJECT_DIR), "--results-dir", str(run_dir)],
            cwd=PROJECT_DIR,
        )
        metadata["status"] = "success"
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] CPU benchmark failed: {exc}", file=sys.stderr)
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        exit_code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1
    finally:
        stop_server(server_process)
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        args.results_root.mkdir(parents=True, exist_ok=True)
        (args.results_root / "latest_run.txt").write_text(run_id + "\n", encoding="utf-8")

    print(f"[INFO] CPU run saved to: {run_dir}")
    if metadata["status"] == "success":
        print(f"[INFO] CPU report: {run_dir / 'report' / 'benchmark_summary.md'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
