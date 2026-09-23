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

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REQUIREMENTS_PATH = os.path.join(PROJECT_DIR, "requirements.txt")

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
        ignore=["llama.cpp/**", "models/**", "results/**", "benchmark_data/**", ".git/**", "__pycache__/**", "*.pyc"]
    )
)

# Create a persistent volume for the models directory to avoid re-downloading/converting
models_volume = modal.Volume.from_name("qwen3-vl-benchmark-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",           # You can change this to "A10G" or "A100" if you need more power/VRAM
    volumes={"/benchmark/models": models_volume},
    timeout=86400,      # Allow up to 24 hours
    cpu=8.0             # Request 8 CPU cores for fast quantization
)
def run_benchmark_remote(smoke: bool = False, model_name: str = ""):
    """This function executes INSIDE the Modal cloud container."""
    os.chdir("/benchmark")
    
    print("=============================================")
    print(" WELCOME TO MODAL CLOUD EXECUTION")
    print("=============================================")

    # Ensure scripts are executable
    subprocess.run(["chmod", "+x", "run_benchmark.sh"], check=True)
    subprocess.run(["chmod", "+x"] + [os.path.join("scripts", f) for f in os.listdir("scripts") if f.endswith(".sh")], check=True)
    
    # Build command
    cmd = ["./run_benchmark.sh"]
    if smoke:
        cmd.append("--smoke")
    if model_name:
        cmd.extend(["--model", model_name])
        
    print(f"[INFO] Executing: {' '.join(cmd)}")
    
    # Point the benchmark script to the pre-built llama.cpp directory
    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = "/opt/llama.cpp"
    
    try:
        # Run the benchmark. Output streams directly to your local terminal.
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Benchmark failed with exit code {e.returncode}")
        # We still want to return whatever results were generated before failing
    
    # Commit the volume to persist any downloaded/converted models for next run
    models_volume.commit()
    
    # Zip the results directory
    print("[INFO] Zipping results for download...")
    if os.path.exists("results"):
        subprocess.run(["zip", "-r", "results.zip", "results/"], check=True)
        with open("results.zip", "rb") as f:
            return f.read()
    else:
        print("[WARN] No results directory found.")
        return None


@app.local_entrypoint()
def main(smoke: bool = False, model_name: str = ""):
    """This function executes LOCALLY on your machine."""
    print(f"Deploying Benchmark to Modal (Smoke Mode: {smoke}, Model: {model_name or 'ALL'})...")
    
    zip_bytes = run_benchmark_remote.remote(smoke=smoke, model_name=model_name)
    
    if zip_bytes:
        zip_path = os.path.join(PROJECT_DIR, "results_from_modal.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
            
        print(f"\n[SUCCESS] Results downloaded successfully to {zip_path}")
        print("[INFO] Extracting results...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Note: this extracts into PROJECT_DIR/results because the zip contains the 'results' folder
            zip_ref.extractall(PROJECT_DIR)
            
        print("[SUCCESS] Extraction complete! Check the 'results/' folder for your reports and charts.")
    else:
        print("\n[ERROR] No results were returned from Modal.")
