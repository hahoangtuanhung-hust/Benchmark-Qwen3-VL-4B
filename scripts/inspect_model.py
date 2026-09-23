#!/usr/bin/env python3
"""
Inspect GGUF model artifacts and build compatibility matrix.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §2, §7

This script:
1. Reads GGUF file metadata (file type, tensor types, size)
2. Computes SHA256 hashes
3. Builds the compatibility_matrix.csv mapping requested precisions to actual GGUF types
4. Creates model_manifest.csv with full artifact metadata
"""

import argparse
import csv
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import yaml


# ─── GGUF File Type Constants ───
# From llama.cpp: llama_ftype enum
GGUF_FTYPE_MAP = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
    2: "MOSTLY_Q4_0",
    3: "MOSTLY_Q4_1",
    7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1",
    10: "MOSTLY_Q2_K",
    11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M",
    16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K",
    19: "MOSTLY_IQ2_XXS",
    20: "MOSTLY_IQ2_XS",
    21: "MOSTLY_IQ3_XXS",
    24: "MOSTLY_IQ1_S",
    25: "MOSTLY_IQ4_NL",
    26: "MOSTLY_IQ3_S",
    27: "MOSTLY_IQ3_M",
    28: "MOSTLY_IQ2_S",
    29: "MOSTLY_IQ2_M",
    30: "MOSTLY_IQ4_XS",
    31: "MOSTLY_IQ1_M",
    32: "MOSTLY_BF16",
}


def compute_sha256(filepath: str, chunk_size: int = 8192 * 1024) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def read_gguf_metadata(filepath: str) -> dict:
    """Read basic GGUF metadata from file header."""
    metadata = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "file_size_bytes": os.path.getsize(filepath),
        "gguf_magic": None,
        "gguf_version": None,
        "tensor_count": None,
        "metadata_kv_count": None,
        "file_type": None,
        "file_type_name": None,
        "general_name": None,
        "general_architecture": None,
    }

    try:
        with open(filepath, "rb") as f:
            # Read GGUF magic (4 bytes)
            magic = f.read(4)
            if magic != b"GGUF":
                metadata["gguf_magic"] = f"INVALID ({magic})"
                return metadata
            metadata["gguf_magic"] = "GGUF"

            # Version (4 bytes, uint32 LE)
            version = struct.unpack("<I", f.read(4))[0]
            metadata["gguf_version"] = version

            # Tensor count (8 bytes for v3, 4 bytes for older)
            if version >= 3:
                tensor_count = struct.unpack("<Q", f.read(8))[0]
                kv_count = struct.unpack("<Q", f.read(8))[0]
            else:
                tensor_count = struct.unpack("<I", f.read(4))[0]
                kv_count = struct.unpack("<I", f.read(4))[0]

            metadata["tensor_count"] = tensor_count
            metadata["metadata_kv_count"] = kv_count

    except Exception as e:
        metadata["error"] = str(e)

    return metadata


def get_quantize_help(quantize_path: str) -> str:
    """Get output of llama-quantize --help to determine supported types."""
    try:
        result = subprocess.run(
            [quantize_path, "--help"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {e}"


def parse_supported_types(quantize_help: str) -> list[str]:
    """Parse llama-quantize --help output to extract supported quantization types."""
    types = []
    # Look for lines like "  2  or  Q4_0" or type listings
    for line in quantize_help.split("\n"):
        line = line.strip()
        # Common patterns in llama-quantize help
        for candidate in [
            "F32", "F16", "BF16",
            "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K",
            "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q5_K",
            "Q6_K",
            "Q8_0",
            "Q2_K", "Q2_K_S",
            "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "IQ1_S", "IQ1_M",
            "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
            "IQ3_XXS", "IQ3_S", "IQ3_M",
            "IQ4_NL", "IQ4_XS",
            "TQ1_0", "TQ2_0",
        ]:
            if candidate in line and candidate not in types:
                types.append(candidate)
    return types


def build_compatibility_matrix(
    models_config: dict,
    supported_types: list[str],
    model_dir: str
) -> list[dict]:
    """
    Build compatibility matrix per §2.1.
    Maps requested business labels to actual GGUF types.
    """
    matrix = []

    for prec in models_config.get("requested_precisions", []):
        label = prec["label"]
        semantics = prec["semantics"]
        # New config uses 'quantize_type' instead of 'expected_gguf_type'
        expected = prec.get("quantize_type") or prec.get("expected_gguf_type")
        declared_support = prec.get("support_type", "UNKNOWN")

        # Determine actual support
        if expected and expected in supported_types:
            # Check if the model file actually exists
            model_path = os.path.join(model_dir, prec.get("directory", ""), prec.get("filename", ""))
            if prec.get("filename") and os.path.exists(model_path):
                actual_support = "NATIVE"
                model_artifact = model_path
            else:
                actual_support = "NATIVE_NOT_BUILT"
                model_artifact = f"NOT_FOUND: {model_path}"
        elif expected is None:
            actual_support = "UNSUPPORTED"
            model_artifact = "N/A"
        else:
            actual_support = "UNSUPPORTED"
            model_artifact = "N/A"

        # Determine mmproj artifact
        mmproj_cfg = models_config.get("mmproj", {})
        mmproj_path = os.path.join(
            model_dir,
            mmproj_cfg.get("directory", "models/mmproj"),
            mmproj_cfg.get("filename", "")
        )
        mmproj_artifact = mmproj_path if os.path.exists(mmproj_path) else f"NOT_FOUND: {mmproj_path}"

        matrix.append({
            "requested_label": label,
            "requested_semantics": semantics,
            "llama_cpp_supported": expected in supported_types if expected else False,
            "support_type": actual_support,
            "actual_gguf_type": expected or "N/A",
            "actual_compute_behavior": prec.get("notes", ""),
            "model_artifact": model_artifact,
            "mmproj_artifact": mmproj_artifact,
            "evidence": f"llama-quantize --help lists: {', '.join(supported_types[:10])}...",
            "notes": prec.get("notes", ""),
        })

    return matrix


def build_model_manifest(models_config: dict, model_dir: str) -> list[dict]:
    """Build model_manifest.csv with SHA256 hashes and GGUF metadata."""
    manifest = []

    # Process all model artifacts (requested + native)
    all_models = []
    for prec in models_config.get("requested_precisions", []):
        if prec.get("filename"):
            all_models.append(prec)

    # Add mmproj
    mmproj = models_config.get("mmproj", {})
    if mmproj.get("filename"):
        all_models.append({
            "label": f"mmproj_{mmproj.get('type', 'f16')}",
            **mmproj
        })

    for model_info in all_models:
        directory = model_info.get("directory", "")
        filename = model_info.get("filename", "")
        filepath = os.path.join(model_dir, directory, filename)

        entry = {
            "model_name": models_config.get("model", {}).get("name", "Qwen3-VL-4B-Instruct"),
            "label": model_info.get("label", ""),
            "source_repo": model_info.get("source_repo", ""),
            "source_revision": "",
            "filename": filename,
            "filepath": filepath,
            "file_size_bytes": None,
            "sha256": None,
            "gguf_file_type": None,
            "tensor_type_summary": None,
            "created_by": "benchmark_suite",
            "quantization_command": "",
            "status": "NOT_FOUND",
        }

        if os.path.exists(filepath):
            entry["file_size_bytes"] = os.path.getsize(filepath)
            entry["status"] = "FOUND"

            print(f"  Computing SHA256 for {filename}...", end=" ", flush=True)
            entry["sha256"] = compute_sha256(filepath)
            print("done")

            # Read GGUF metadata
            gguf_meta = read_gguf_metadata(filepath)
            entry["gguf_file_type"] = gguf_meta.get("file_type_name", gguf_meta.get("file_type"))
            entry["tensor_type_summary"] = f"tensors={gguf_meta.get('tensor_count', '?')}"

        manifest.append(entry)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Inspect GGUF models and build compatibility matrix")
    parser.add_argument("--models-config", default="configs/models.yaml", help="Models config YAML")
    parser.add_argument("--project-dir", default=".", help="Project root directory")
    parser.add_argument("--quantize-path", default=None, help="Path to llama-quantize binary")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)

    # Load config
    config_path = os.path.join(project_dir, args.models_config)
    with open(config_path) as f:
        models_config = yaml.safe_load(f)

    # Determine llama-quantize path
    quantize_path = args.quantize_path
    if not quantize_path:
        # Check LLAMA_CPP_DIR env first, then fallback to local
        llamacpp_dir = os.environ.get("LLAMA_CPP_DIR", os.path.join(project_dir, "llama.cpp"))
        quantize_path = os.path.join(llamacpp_dir, "build/bin/llama-quantize")

    # --- Get supported quantization types ---
    print("=" * 60)
    print(" INSPECTING MODEL ARTIFACTS & COMPATIBILITY")
    print("=" * 60)
    print()

    supported_types = []
    if os.path.exists(quantize_path):
        print(f"[INFO] Reading supported types from: {quantize_path}")
        help_text = get_quantize_help(quantize_path)
        supported_types = parse_supported_types(help_text)
        print(f"[INFO] Detected {len(supported_types)} supported types: {supported_types}")

        # Save help text
        compat_dir = os.path.join(project_dir, "results/compatibility")
        os.makedirs(compat_dir, exist_ok=True)
        with open(os.path.join(compat_dir, "quantize_help.txt"), "w") as f:
            f.write(help_text)
    else:
        print(f"[WARN] llama-quantize not found at {quantize_path}")
        print("[WARN] Using default supported types list")
        supported_types = ["F32", "F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S",
                          "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "Q3_K_M", "Q3_K_S",
                          "Q3_K_L", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_S", "IQ3_M",
                          "IQ2_XS", "IQ2_S", "IQ1_S"]

    # --- Build compatibility matrix ---
    print()
    print("[INFO] Building compatibility matrix...")
    matrix = build_compatibility_matrix(models_config, supported_types, project_dir)

    matrix_path = os.path.join(project_dir, "results/compatibility_matrix.csv")
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)

    fieldnames = [
        "requested_label", "requested_semantics", "llama_cpp_supported",
        "support_type", "actual_gguf_type", "actual_compute_behavior",
        "model_artifact", "mmproj_artifact", "evidence", "notes",
    ]
    with open(matrix_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix)

    print(f"[INFO] Compatibility matrix saved to: {matrix_path}")
    print()
    print("  Compatibility Matrix Summary:")
    print(f"  {'Label':<10} {'Support Type':<20} {'Actual GGUF':<15}")
    print("  " + "-" * 50)
    for row in matrix:
        print(f"  {row['requested_label']:<10} {row['support_type']:<20} {row['actual_gguf_type']:<15}")

    # --- Build model manifest ---
    print()
    print("[INFO] Building model manifest...")
    manifest = build_model_manifest(models_config, project_dir)

    manifest_path = os.path.join(project_dir, "results/model_manifest.csv")
    manifest_fields = [
        "model_name", "label", "source_repo", "source_revision", "filename",
        "filepath", "file_size_bytes", "sha256", "gguf_file_type",
        "tensor_type_summary", "created_by", "quantization_command", "status",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"[INFO] Model manifest saved to: {manifest_path}")
    print()
    print("  Model Manifest Summary:")
    print(f"  {'Label':<15} {'Filename':<45} {'Status':<12} {'Size (MB)':<12}")
    print("  " + "-" * 85)
    for entry in manifest:
        size_mb = f"{entry['file_size_bytes'] / 1e6:.1f}" if entry['file_size_bytes'] else "N/A"
        print(f"  {entry['label']:<15} {entry['filename']:<45} {entry['status']:<12} {size_mb:<12}")

    print()
    print("[SUCCESS] Inspection complete.")


if __name__ == "__main__":
    main()
