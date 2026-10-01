#!/usr/bin/env python3
"""
Run Benchmark Suite on Modal.com Cloud GPUs
This script bridges your local environment with Modal, creating a remote container,
running the benchmark, and syncing the results back.
"""

import modal
import subprocess
import os
import zipfile
import argparse
import sys
import shutil
from datetime import datetime, timezone

MODAL_GPU = "L4"
EXPECTED_GPU_NAME = "l4"

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REQUIREMENTS_PATH = os.path.join(PROJECT_DIR, "requirements.txt")
RUN_ARTIFACTS = (
    "accuracy",
    "charts",
    "compatibility",
    "environment",
    "gpu",
    "logs",
    "raw",
    "report",
    "server_metrics",
    "compatibility_matrix.csv",
    "config_snapshot.yaml",
    "model_manifest.csv",
    "validation_report.json",
)

# Define the Modal App
app = modal.App("qwen3-vl-benchmark")

# Define the environment (Ubuntu + Python 3.11 + CUDA + dependencies)
# Phase 1 (Build llama.cpp) is baked into the image so it only happens ONCE.
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "cmake", "build-essential", "wget", "curl", "zip", "unzip")
    .pip_install_from_requirements(REQUIREMENTS_PATH)
    .pip_install("huggingface_hub")
    # Build llama.cpp with CUDA into the image (cached across runs)
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp",
        # Use CUDA stubs for linking (no real GPU driver during image build) + disable VMM
        "cd /opt/llama.cpp && cmake -S . -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release"
        " && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH cmake --build build --target llama-server llama-cli llama-quantize -j 8",
        # Install Python dependencies needed by convert_hf_to_gguf.py (torch, transformers, etc.)
        "pip install -q -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt",
    )
    .add_local_dir(
        PROJECT_DIR, 
        remote_path="/benchmark",
        ignore=[
            "llama.cpp/**",
            "models/**",
            "results/**",
            # This path is supplied by accuracy_volume below. It must not
            # contain image files before Modal mounts the volume.
            "benchmark_data/accuracy/**",
            ".git/**",
            "__pycache__/**",
            "*.pyc",
        ]
    )
)

# Create a persistent volume for the models directory to avoid re-downloading/converting
models_volume = modal.Volume.from_name("qwen3-vl-benchmark-models", create_if_missing=True)
accuracy_volume = modal.Volume.from_name("qwen3-vl-benchmark-accuracy", create_if_missing=True)

@app.function(
    image=image,
    gpu=MODAL_GPU,
    volumes={
        "/benchmark/models": models_volume,
        "/benchmark/benchmark_data/accuracy": accuracy_volume,
    },
    timeout=86400,      # Allow up to 24 hours
    cpu=8.0             # Request 8 CPU cores for fast quantization
)
def run_benchmark_remote(smoke: bool = False, model_name: str = "", phase: int = 0, end_phase: int = 99):
    """This function executes INSIDE the Modal cloud container."""
    os.chdir("/benchmark")
    
    print("=============================================")
    print(" WELCOME TO MODAL CLOUD EXECUTION")
    print("=============================================")
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if len(gpu_name) != 1 or EXPECTED_GPU_NAME not in gpu_name[0].lower():
        raise RuntimeError(f"Expected exactly one NVIDIA L4 from Modal, received: {gpu_name}")
    print(f"[INFO] Modal GPU validated: {gpu_name[0]}")

    # Ensure scripts are executable
    subprocess.run(["chmod", "+x", "run_benchmark.sh"], check=True)
    subprocess.run(["chmod", "+x"] + [os.path.join("scripts", f) for f in os.listdir("scripts") if f.endswith(".sh")], check=True)
    
    # Build command
    cmd = ["./run_benchmark.sh"]
    if smoke:
        cmd.append("--smoke")
    if model_name:
        cmd.extend(["--model", model_name])
    if phase > 0:
        cmd.extend(["--phase", str(phase)])
    if end_phase != 99:
        cmd.extend(["--end-phase", str(end_phase)])
        
    print(f"[INFO] Executing: {' '.join(cmd)}")
    
    # Point the benchmark script to the pre-built llama.cpp directory
    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = "/opt/llama.cpp"
    env["CUDA_ARCH"] = "89"
    
    benchmark_exit_code = 0
    try:
        # Run the benchmark. Output streams directly to your local terminal.
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        benchmark_exit_code = e.returncode
        print(f"[ERROR] Benchmark failed with exit code {e.returncode}")
        # We still want to return whatever results were generated before failing
    
    # Commit the volume to persist any downloaded/converted models for next run
    models_volume.commit()
    accuracy_volume.commit()
    
    # Zip the results directory
    print("[INFO] Zipping results for download...")
    if os.path.exists("results"):
        run_id = ""
        latest_run_path = os.path.join("results", "latest_run.txt")
        if os.path.exists(latest_run_path):
            with open(latest_run_path, encoding="utf-8") as f:
                run_id = f.read().strip()
        subprocess.run(["zip", "-r", "results.zip", "results/"], check=True)
        with open("results.zip", "rb") as f:
            return f.read(), benchmark_exit_code, run_id
    else:
        print("[WARN] No results directory found.")
        return None, benchmark_exit_code or 1, ""


def _unique_legacy_run_dir(results_root: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = os.path.join(results_root, f"run_{timestamp}_legacy_local")
    candidate = base
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _archive_legacy_results(results_root: str) -> str | None:
    """Move results from the old flat layout into a preserved run folder."""
    existing = [name for name in RUN_ARTIFACTS if os.path.exists(os.path.join(results_root, name))]
    if not existing:
        return None

    legacy_dir = _unique_legacy_run_dir(results_root)
    os.makedirs(legacy_dir, exist_ok=True)
    for name in existing:
        shutil.move(os.path.join(results_root, name), os.path.join(legacy_dir, name))
    with open(os.path.join(legacy_dir, "run_metadata.txt"), "w", encoding="utf-8") as f:
        f.write("status=legacy\nsource=pre_run_directory_layout\n")
    print(f"[INFO] Previous flat results preserved in {legacy_dir}")
    return legacy_dir


def _find_previous_performance_run(results_root: str, current_run_dir: str) -> str | None:
    candidates = []
    current_run_dir = os.path.abspath(current_run_dir)
    if not os.path.isdir(results_root):
        return None
    for name in os.listdir(results_root):
        candidate = os.path.abspath(os.path.join(results_root, name))
        if (
            name.startswith("run_")
            and candidate != current_run_dir
            and os.path.isfile(os.path.join(candidate, "raw", "requests.csv"))
        ):
            candidates.append(candidate)
    if not candidates:
        return None

    # A legacy folder is timestamped when it is archived locally, which can be
    # later than the remote run that just produced fresher performance data.
    regular_runs = [path for path in candidates if "_legacy" not in os.path.basename(path)]
    return max(regular_runs or candidates, key=os.path.basename)


def _copy_missing(source: str, destination: str) -> None:
    if os.path.isdir(source):
        for source_root, _, filenames in os.walk(source):
            relative_root = os.path.relpath(source_root, source)
            destination_root = destination if relative_root == "." else os.path.join(destination, relative_root)
            os.makedirs(destination_root, exist_ok=True)
            for filename in filenames:
                destination_file = os.path.join(destination_root, filename)
                if not os.path.exists(destination_file):
                    shutil.copy2(os.path.join(source_root, filename), destination_file)
    elif os.path.isfile(source) and not os.path.exists(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)


def _hydrate_accuracy_resume(results_root: str, current_run_dir: str) -> str | None:
    """Add missing performance artifacts to a phase 9+ result run."""
    previous_run = _find_previous_performance_run(results_root, current_run_dir)
    if not previous_run:
        return None
    for name in RUN_ARTIFACTS:
        _copy_missing(
            os.path.join(previous_run, name),
            os.path.join(current_run_dir, name),
        )
    return previous_run


@app.local_entrypoint()
def main(smoke: bool = False, model_name: str = "", phase: int = 0, end_phase: int = 99):
    """This function executes LOCALLY on your machine."""
    print(f"Deploying Benchmark to Modal (Smoke Mode: {smoke}, Model: {model_name or 'ALL'}, Phase: {phase}, End Phase: {end_phase})...")
    
    zip_bytes, benchmark_exit_code, run_id = run_benchmark_remote.remote(
        smoke=smoke, model_name=model_name, phase=phase, end_phase=end_phase
    )
    
    if zip_bytes:
        zip_path = os.path.join(PROJECT_DIR, "results_from_modal.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
            
        print(f"\n[SUCCESS] Results downloaded successfully to {zip_path}")
        local_results = os.path.join(PROJECT_DIR, "results")
        os.makedirs(local_results, exist_ok=True)
        _archive_legacy_results(local_results)

        print("[INFO] Extracting run results...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # The archive contains results/<run_id> and results/latest_run.txt.
            zip_ref.extractall(PROJECT_DIR)

        current_run_dir = os.path.join(local_results, run_id) if run_id else ""
        if not current_run_dir or not os.path.isdir(current_run_dir):
            print("[ERROR] Downloaded archive did not identify a valid run directory.")
            raise SystemExit(benchmark_exit_code or 1)

        # A Phase 9 resume returns accuracy artifacts but no remote performance
        # data. Merge them with the existing local performance run and rebuild
        # the report automatically.
        if phase >= 9:
            previous_run = _hydrate_accuracy_resume(local_results, current_run_dir)
            if previous_run:
                print(f"[INFO] Added performance artifacts from {previous_run}")
            local_raw_results = os.path.join(current_run_dir, "raw", "requests.csv")
            if os.path.exists(local_raw_results):
                print("[INFO] Rebuilding this run's report with accuracy and performance results...")
                subprocess.run(
                    [
                        sys.executable,
                        os.path.join(SCRIPT_DIR, "summarize.py"),
                        "--project-dir",
                        PROJECT_DIR,
                        "--results-dir",
                        current_run_dir,
                    ],
                    check=True,
                )

        if benchmark_exit_code:
            print(f"[WARN] Remote benchmark failed; partial artifacts are in {current_run_dir}")
        else:
            print(f"[SUCCESS] Run results: {current_run_dir}")
            report_path = os.path.join(current_run_dir, "report", "benchmark_summary.md")
            if os.path.exists(report_path):
                print(f"[SUCCESS] Report: {report_path}")
    else:
        print("\n[ERROR] No results were returned from Modal.")

    if benchmark_exit_code:
        raise SystemExit(benchmark_exit_code)
