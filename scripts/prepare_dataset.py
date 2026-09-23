#!/usr/bin/env python3
"""
Prepare benchmark dataset — Generate test images and manifests.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §8

Creates synthetic test images with varied content for benchmark.
Images are generated locally (no internet download during benchmark).
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import yaml

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow not installed. Install with: pip install Pillow")


# ─── Image Generation ───

# Diverse scenes for synthetic benchmarking
SCENE_DESCRIPTIONS = [
    "Office workspace with laptop and documents",
    "City street with traffic signs and vehicles",
    "Restaurant menu board with pricing",
    "Industrial dashboard with gauges and readings",
    "Retail store shelf with product labels",
    "Medical form with patient information fields",
    "Construction site with safety signage",
    "Airport departure board with flight information",
    "Laboratory equipment with measurement displays",
    "Warehouse inventory with barcode labels",
    "Conference room with presentation slide",
    "Parking lot with numbered spaces and signs",
    "Library bookshelf with spine titles",
    "Kitchen recipe card with ingredients list",
    "Control panel with buttons and status lights",
    "Whiteboard with meeting notes and diagrams",
    "Storefront with business name and hours",
    "Vehicle dashboard with speedometer display",
    "Product packaging with nutrition label",
    "Bus stop with route map and schedule",
    "Building directory with floor listings",
    "Price tag display in a market",
    "Train station platform with timetable",
    "Gas station with fuel price display",
    "Shopping mall directory sign",
    "Hospital ward sign with room numbers",
    "University campus map with building labels",
    "Sports scoreboard with team names",
    "Theater marquee with show times",
    "Elevator panel with floor buttons",
]

# Colors for varied image backgrounds
COLORS = [
    (45, 52, 54), (108, 92, 231), (0, 206, 209), (255, 107, 107),
    (46, 134, 222), (255, 159, 67), (16, 172, 132), (52, 73, 94),
    (192, 57, 43), (39, 174, 96), (142, 68, 173), (44, 62, 80),
    (22, 160, 133), (211, 84, 0), (41, 128, 185), (243, 156, 18),
    (127, 140, 141), (155, 89, 182), (26, 188, 156), (231, 76, 60),
]


def generate_test_image(
    width: int,
    height: int,
    image_id: str,
    description: str,
    output_path: str,
):
    """Generate a synthetic test image with text and shapes."""
    if not HAS_PIL:
        # Fallback: create a minimal valid JPEG
        img = Image.new("RGB", (width, height), color=random.choice(COLORS))
        img.save(output_path, "JPEG", quality=85)
        return

    bg_color = random.choice(COLORS)
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw grid pattern for visual complexity
    grid_color = tuple(min(c + 30, 255) for c in bg_color)
    for x in range(0, width, width // 8):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, height // 6):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    # Draw random rectangles (simulating objects)
    num_rects = random.randint(3, 8)
    for _ in range(num_rects):
        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 80)
        x2 = x1 + random.randint(50, min(200, width - x1))
        y2 = y1 + random.randint(30, min(150, height - y1))
        rect_color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )
        draw.rectangle([x1, y1, x2, y2], fill=rect_color, outline="white", width=2)

    # Draw text elements
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(20, height // 15))
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(14, height // 25))
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title text
    draw.text((20, 20), f"Test Image: {image_id}", fill="white", font=font_large)
    draw.text((20, 20 + height // 12), description, fill="white", font=font_small)

    # Simulated data text
    data_lines = [
        f"ID: {image_id}",
        f"Resolution: {width}x{height}",
        f"Value A: {random.randint(100, 9999)}",
        f"Value B: {random.uniform(0.1, 99.9):.2f}",
        f"Status: {'Active' if random.random() > 0.3 else 'Inactive'}",
        f"Category: {random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'])}",
    ]
    y_offset = height // 3
    for line in data_lines:
        draw.text((30, y_offset), line, fill="white", font=font_small)
        y_offset += height // 20

    # Add some circles for visual variety
    for _ in range(random.randint(2, 5)):
        cx = random.randint(width // 2, width - 50)
        cy = random.randint(50, height - 50)
        r = random.randint(10, 40)
        circle_color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=circle_color, outline="white")

    img.save(output_path, "JPEG", quality=85)


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def prepare_dataset(project_dir: str, config_path: str):
    """Prepare all benchmark datasets."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    perf_config = config.get("performance", {})
    image_sets = perf_config.get("image_sets", {})
    manifest_entries = []

    print("=" * 60)
    print(" PREPARING BENCHMARK DATASET")
    print("=" * 60)

    scene_idx = 0

    for set_name, set_config in image_sets.items():
        target_w = set_config["target_width"]
        target_h = set_config["target_height"]
        samples = set_config["samples"]
        directory = os.path.join(project_dir, set_config["directory"])

        os.makedirs(directory, exist_ok=True)
        print(f"\n[INFO] Generating {set_name} set: {samples} images @ {target_w}x{target_h}")

        for i in range(samples):
            image_id = f"{set_name}_{i:04d}"
            filename = f"{image_id}.jpg"
            filepath = os.path.join(directory, filename)

            # Use varied descriptions
            desc = SCENE_DESCRIPTIONS[scene_idx % len(SCENE_DESCRIPTIONS)]
            scene_idx += 1

            # Add slight resolution variation for realism
            w = target_w + random.randint(-10, 10)
            h = target_h + random.randint(-10, 10)

            generate_test_image(w, h, image_id, desc, filepath)

            # Verify and hash
            img = Image.open(filepath)
            actual_w, actual_h = img.size
            file_size = os.path.getsize(filepath)
            sha256 = compute_sha256(filepath)

            entry = {
                "image_id": image_id,
                "path": os.path.relpath(filepath, project_dir),
                "width": actual_w,
                "height": actual_h,
                "bytes": file_size,
                "sha256": sha256,
                "category": set_name,
                "description": desc,
            }
            manifest_entries.append(entry)
            print(f"  [{i+1}/{samples}] {filename} ({actual_w}x{actual_h}, {file_size} bytes)")

    # Write manifest
    manifest_path = os.path.join(project_dir, perf_config.get("manifest", "benchmark_data/manifest.jsonl"))
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[INFO] Manifest saved to: {manifest_path}")
    print(f"[INFO] Total images: {len(manifest_entries)}")

    # Also save as CSV for convenience
    csv_path = manifest_path.replace(".jsonl", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_entries[0].keys())
        writer.writeheader()
        writer.writerows(manifest_entries)

    # Create dataset hash
    dataset_hash = hashlib.sha256()
    for entry in manifest_entries:
        dataset_hash.update(entry["sha256"].encode())
    dataset_hash_value = dataset_hash.hexdigest()

    hash_path = os.path.join(project_dir, "benchmark_data/dataset_hash.txt")
    with open(hash_path, "w") as f:
        f.write(f"dataset_hash={dataset_hash_value}\n")
        f.write(f"total_images={len(manifest_entries)}\n")

    print(f"[INFO] Dataset hash: {dataset_hash_value[:16]}...")
    print(f"\n[SUCCESS] Dataset preparation complete.")

    return manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark dataset")
    parser.add_argument("--config", default="configs/datasets.yaml", help="Dataset config YAML")
    parser.add_argument("--project-dir", default=".", help="Project root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    project_dir = os.path.abspath(args.project_dir)
    config_path = os.path.join(project_dir, args.config)

    prepare_dataset(project_dir, config_path)


if __name__ == "__main__":
    main()
