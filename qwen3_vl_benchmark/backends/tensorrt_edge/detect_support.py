from __future__ import annotations

import argparse
import csv
import importlib.util
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PACKAGES = {
    "TensorRT": ("tensorrt",),
    "TensorRT-Edge-LLM": ("tensorrt-edgellm", "tensorrt-edge-llm", "tensorrt_edge_llm"),
    "ModelOpt": ("nvidia-modelopt", "modelopt"),
}


def _run(command: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def _package_version(names: tuple[str, ...]) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def architecture_for_cc(major: int | None, minor: int | None, gpu_name: str) -> str:
    lowered = gpu_name.lower()
    if "orin" in lowered:
        return "Jetson Orin"
    if "thor" in lowered:
        return "Jetson Thor"
    if "dgx spark" in lowered or "gb10" in lowered:
        return "DGX Spark"
    if major == 8 and minor == 9:
        return "Ada"
    if major == 7 and minor == 5:
        return "Turing"
    return {8: "Ampere", 9: "Hopper", 10: "Blackwell", 12: "Blackwell"}.get(major, "other")


def detect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu_name": None,
        "gpu_arch": "other",
        "compute_capability": None,
        "vram_mb": None,
        "driver": None,
        "cuda": None,
        "gpu_count": 0,
        "detection_evidence": [],
    }
    smi = shutil.which("nvidia-smi")
    if smi:
        query = [smi, "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader,nounits"]
        code, output = _run(query)
        env["detection_evidence"].append({"command": " ".join(query), "exit_code": code, "output": output})
        if code == 0 and output:
            rows = [line for line in output.splitlines() if line.strip()]
            first = [item.strip() for item in rows[0].split(",")]
            if len(first) >= 4:
                name, cc, memory, driver = first[:4]
                try:
                    major, minor = (int(x) for x in cc.split(".", 1))
                except (TypeError, ValueError):
                    major = minor = None
                env.update(gpu_name=name, compute_capability=cc, vram_mb=float(memory), driver=driver,
                           gpu_count=len(rows), gpu_arch=architecture_for_cc(major, minor, name))
        code, output = _run([smi])
        if code == 0:
            for line in output.splitlines():
                if "CUDA Version:" in line:
                    env["cuda"] = line.split("CUDA Version:", 1)[1].split()[0]
                    break
    else:
        env["detection_evidence"].append({"command": "nvidia-smi", "exit_code": 127, "output": "not found on PATH"})

    for display_name, package_names in PACKAGES.items():
        env[display_name] = _package_version(package_names)
    env["qwen3_vl_module"] = _module_available("tensorrt_edgellm.models.qwen3_vl")
    return env


def compatibility_rows(config: dict[str, Any], env: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_present = bool(env.get("TensorRT") and env.get("TensorRT-Edge-LLM"))
    gpu_present = bool(env.get("gpu_count"))
    edge_arch_supported = env.get("gpu_arch") in {
        "Ampere", "Ada", "Hopper", "Blackwell", "Jetson Orin",
        "Jetson Thor", "DGX Spark",
    }
    rows = []
    for requested, spec in config["precisions"].items():
        recipe = str(spec.get("quant_recipe", "unknown"))
        recipe_known = recipe.lower() not in {"", "unknown", "none"}
        model_supported: bool | None = True if env.get("qwen3_vl_module") else None
        hardware_supported: bool | None = None
        runtime_supported: bool | None = None
        evidence = []
        notes = []

        if not gpu_present:
            hardware_supported = False
            evidence.append("nvidia-smi unavailable; no NVIDIA GPU detected")
        if not runtime_present:
            runtime_supported = False
            evidence.append("TensorRT and/or TensorRT-Edge-LLM Python distribution unavailable")
        if requested == "FP16" and gpu_present:
            hardware_supported = edge_arch_supported
            runtime_supported = runtime_present and edge_arch_supported
        if requested in {"FP8", "W4A8"} and gpu_present:
            hardware_supported = env.get("gpu_arch") in {"Ada", "Hopper", "Blackwell", "Jetson Thor", "DGX Spark"}
            notes.append("Hardware capability alone does not prove the requested runtime recipe")
        if requested == "FP8" and runtime_present:
            runtime_supported = hardware_supported
        if requested in {"INT4", "W4A16"} and gpu_present:
            hardware_supported = edge_arch_supported
            runtime_supported = runtime_present and edge_arch_supported
        if gpu_present and not edge_arch_supported:
            evidence.append(
                f"TensorRT-Edge-LLM v0.10.1 provides no required kernels for {env.get('gpu_arch')} "
                f"({env.get('compute_capability') or 'unknown compute capability'})"
            )
        if requested == "W4A8":
            runtime_supported = False
            notes.append("0.10.1 methods list has INT4_AWQ but no W4A8 recipe")

        if hardware_supported is False or runtime_supported is False:
            status = "UNSUPPORTED"
        elif all(value is True for value in (model_supported, hardware_supported, runtime_supported)):
            status = "SUPPORTED_WITH_RECIPE" if requested in {"FP8", "INT4", "W4A16"} else "SUPPORTED_EXACT"
        else:
            status = "UNKNOWN"
        rows.append({
            "requested_precision": requested,
            "requested_semantics": spec.get("requested_semantics", ""),
            "model_supported": model_supported,
            "hardware_supported": hardware_supported,
            "runtime_supported": runtime_supported,
            "quant_recipe": recipe,
            "weight_dtype": spec.get("weight_dtype", "unknown"),
            "activation_dtype": spec.get("activation_dtype", "unknown"),
            "kv_cache_dtype": config["policy"]["kv_cache_dtype"],
            "visual_encoder_dtype": config["policy"]["visual_encoder_dtype"],
            "status": status,
            "evidence": "; ".join(evidence),
            "notes": "; ".join(notes),
        })
    return rows


def write_outputs(output_dir: Path, env: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={json.dumps(value, ensure_ascii=True)}" for key, value in env.items()]
    (output_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output_dir / "compatibility_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect TensorRT-Edge-LLM support without assumptions")
    parser.add_argument("--config", default="modal/tensorrt_edge.yaml")
    parser.add_argument("--output-dir", default="results/tensorrt_edge")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    env = detect_environment()
    rows = compatibility_rows(config, env)
    write_outputs(Path(args.output_dir), env, rows)
    print(json.dumps({"environment": env, "compatibility": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
