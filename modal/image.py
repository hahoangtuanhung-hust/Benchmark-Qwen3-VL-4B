from __future__ import annotations

from pathlib import Path

import modal

from gpu_config import EXPECTED_SM


PROJECT_DIR = Path(__file__).resolve().parents[1]
EDGE_REF = "v0.10.1"
BASE_IMAGE = "nvcr.io/nvidia/pytorch:26.04-py3"

dependency_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .apt_install(
        "build-essential", "cmake", "git", "git-lfs", "ninja-build",
        "curl", "jq", "zip", "libgl1", "libglib2.0-0",
    )
    .run_commands("git lfs install --system")
    .pip_install(
        "aiohttp>=3.9", "requests>=2.31", "pyyaml>=6", "pandas>=2",
        "numpy>=2.2.6,<3", "datasets==5.0.0", "Pillow>=10",
        "matplotlib>=3.7", "tqdm>=4.65", "psutil>=5.9",
    )
    .run_commands(
        f"git clone --branch {EDGE_REF} --depth 1 --recurse-submodules "
        "https://github.com/NVIDIA/TensorRT-Edge-LLM.git /opt/TensorRT-Edge-LLM",
        "cd /opt/TensorRT-Edge-LLM && python -m pip install --no-cache-dir -e '.[tools,server,server-tools,native-build]'",
        # The NGC image bundles Transformer Engine against a different cuBLASLt
        # ABI. It is optional here, but PEFT auto-imports it and aborts ModelOpt
        # quantization when the shared object cannot be loaded.
        "python -m pip uninstall -y transformer-engine transformer-engine-torch || true",
    )
    .pip_install("nvidia-cutlass-dsl[cu13]==4.7.0", "cupy-cuda13x==13.6.0")
    # NGC ships an image-build cache at /cache. Modal volumes require an empty
    # mount point; dependencies are already committed to their image layers.
    .run_commands("rm -rf /cache")
    .env({
        "EDGELLM_HOME": "/opt/TensorRT-Edge-LLM",
        "HF_HOME": "/models/huggingface",
        "HF_HUB_CACHE": "/models/huggingface/hub",
        "TRANSFORMERS_CACHE": "/models/huggingface",
        "XDG_CACHE_HOME": "/tmp/xdg",
        "CUDA_CTK_VERSION": "13.2",
        "TRT_PACKAGE_DIR": "/usr",
        "CUTE_DSL_ARTIFACT_TAG": f"sm_{EXPECTED_SM}",
    })
)

LOCAL_IGNORE = [
    "llama.cpp/**", "models/**", "results/**", "benchmark_data/**",
    ".git/**", "__pycache__/**", "*.pyc", "results_from_modal.zip",
    "*.safetensors", "model.safetensors.index.json",
]

common_image = dependency_image.add_local_dir(
    PROJECT_DIR, remote_path="/workspace", ignore=LOCAL_IGNORE,
)

# Native compilation is performed once inside the selected GPU container and
# persisted under /cache. This lets preflight run without compiling another SM.
benchmark_image = dependency_image.env({
    "PYTHONPATH": "/workspace:/opt/TensorRT-Edge-LLM",
}).add_local_dir(
    PROJECT_DIR, remote_path="/workspace", ignore=LOCAL_IGNORE,
)
