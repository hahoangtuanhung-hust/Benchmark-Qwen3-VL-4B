#!/usr/bin/env python3
"""
Import Real Images into the Benchmark Dataset.
Use this script to use real photos instead of synthetic images for benchmarking.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

def compute_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Import real images for benchmarking")
    parser.add_argument("input_dir", help="Directory containing your real images (JPG/PNG)")
    parser.add_argument("--project-dir", default=".", help="Project root directory")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    input_dir = os.path.abspath(args.input_dir)
    
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    # Target directory for custom performance images
    target_dir = os.path.join(project_dir, "benchmark_data", "performance", "custom")
    os.makedirs(target_dir, exist_ok=True)

    manifest_entries = []
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    image_files = [f for f in os.listdir(input_dir) if Path(f).suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"[WARN] No images found in {input_dir}")
        return

    print(f"[INFO] Importing {len(image_files)} real images from {input_dir}...")

    for i, filename in enumerate(image_files):
        src_path = os.path.join(input_dir, filename)
        image_id = f"custom_real_{i:04d}"
        ext = Path(filename).suffix.lower()
        target_filename = f"{image_id}{ext}"
        target_path = os.path.join(target_dir, target_filename)

        shutil.copy2(src_path, target_path)

        width, height = 0, 0
        if HAS_PIL:
            try:
                with Image.open(target_path) as img:
                    width, height = img.size
            except Exception:
                pass

        file_size = os.path.getsize(target_path)
        sha256 = compute_sha256(target_path)

        entry = {
            "image_id": image_id,
            "path": os.path.relpath(target_path, project_dir),
            "width": width,
            "height": height,
            "bytes": file_size,
            "sha256": sha256,
            "category": "custom",
            "description": "Real user-provided image",
        }
        manifest_entries.append(entry)
        print(f"  [{i+1}/{len(image_files)}] {filename} -> {target_filename}")

    # Write manifest
    manifest_path = os.path.join(project_dir, "benchmark_data", "manifest.jsonl")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    # Overwrite manifest with only real images
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[SUCCESS] Imported {len(manifest_entries)} real images.")
    print(f"[INFO] Manifest saved to: {manifest_path}")
    print("[INFO] You can now run: ./run_benchmark.sh (it will skip synthetic generation and use your real images!)")

if __name__ == "__main__":
    main()
