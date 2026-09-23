#!/usr/bin/env python3
"""
CCU1 Benchmark Runner — Sequential single-user workloads.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §14 (S1, S2, S3, S5)

Runs benchmark scenarios with CCU=1 (one request at a time):
- S1: 10 runs, small images (512x512), max_tokens=256
- S2: 30 runs, normal images (1280x720), max_tokens=256  [PRIMARY]
- S3: 10 runs, large images (1920x1080), max_tokens=256
- S5: 10 runs, normal images, max_tokens=512 (decode-heavy, optional)
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Optional

import yaml

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_client import (
    BenchmarkResult,
    send_vlm_request,
    collect_server_metrics_snapshot,
)


def load_image_manifest(manifest_path: str, category: str) -> list[dict]:
    """Load images from manifest.jsonl filtered by category."""
    images = []
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("category") == category:
                images.append(entry)
    return images


def run_warmup(
    server_url: str,
    images: list[dict],
    prompt: str,
    project_dir: str,
    warmup_count: int = 5,
    max_tokens: int = 64,
):
    """Run warm-up requests and discard results (§13)."""
    print(f"\n[WARMUP] Running {warmup_count} warm-up requests...")
    for i in range(min(warmup_count, len(images))):
        image_path = os.path.join(project_dir, images[i % len(images)]["path"])
        result = send_vlm_request(
            server_url=server_url,
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        status = "OK" if result.success else f"FAIL ({result.error_type})"
        print(f"  Warmup {i+1}/{warmup_count}: {status}")
        if not result.success:
            print(f"    Error: {result.error_type}")
    print("[WARMUP] Complete. Results discarded.\n")


def run_scenario(
    scenario_id: str,
    server_url: str,
    images: list[dict],
    prompt: str,
    project_dir: str,
    runs: int,
    max_tokens: int,
    model_info: dict,
    suite_id: str,
) -> list[dict]:
    """Run a single benchmark scenario."""
    results = []

    print(f"[SCENARIO {scenario_id}] Starting: {runs} runs, max_tokens={max_tokens}")
    print(f"  Images available: {len(images)}")

    # Collect metrics before run
    metrics_before = collect_server_metrics_snapshot(server_url)

    for i in range(runs):
        # Cycle through images
        image_info = images[i % len(images)]
        image_path = os.path.join(project_dir, image_info["path"])

        # Send request
        result = send_vlm_request(
            server_url=server_url,
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=42,
            cache_prompt=False,
        )

        # Enrich with metadata
        result.suite_id = suite_id
        result.scenario = scenario_id
        result.ccu = 1
        result.image_id = image_info.get("image_id", "")
        result.model_name = model_info.get("label", "")
        result.requested_precision = model_info.get("requested_precision", "")
        result.actual_quant_type = model_info.get("actual_quant_type", "")
        result.support_type = model_info.get("support_type", "")
        result.gguf_sha256 = model_info.get("sha256", "")
        result.mmproj_type = model_info.get("mmproj_type", "f16")
        result.llama_cpp_commit = model_info.get("llama_cpp_commit", "")
        result.gpu_name = model_info.get("gpu_name", "")
        result.context_per_slot = model_info.get("context_per_slot", 8192)
        result.kv_k_type = model_info.get("kv_k_type", "f16")
        result.kv_v_type = model_info.get("kv_v_type", "f16")

        results.append(result.to_dict())

        # Progress
        status = "OK" if result.success else f"FAIL"
        ttft = f"{result.ttft_ms:.0f}ms" if result.ttft_ms else "N/A"
        tps = f"{result.decode_tps:.1f}" if result.decode_tps else "N/A"
        tokens = result.output_tokens or 0
        print(f"  [{i+1:3d}/{runs}] {status} | TTFT={ttft} | TPS={tps} | tokens={tokens}")

    # Collect metrics after run
    metrics_after = collect_server_metrics_snapshot(server_url)

    # Log metrics delta
    if metrics_before and metrics_after:
        print(f"\n  Server metrics delta ({scenario_id}):")
        for key in ["llamacpp:tokens_predicted_total", "llamacpp:prompt_tokens_total"]:
            before = metrics_before.get(key, 0)
            after = metrics_after.get(key, 0)
            if after > before:
                print(f"    {key}: {before:.0f} -> {after:.0f} (+{after-before:.0f})")

    # Summary
    successful = [r for r in results if r["success"]]
    if successful:
        ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
        tpss = [r["decode_tps"] for r in successful if r["decode_tps"] is not None]

        import numpy as np
        if ttfts:
            print(f"\n  TTFT: mean={np.mean(ttfts):.0f}ms, P50={np.percentile(ttfts,50):.0f}ms, P95={np.percentile(ttfts,95):.0f}ms")
        if tpss:
            print(f"  Decode TPS: mean={np.mean(tpss):.1f}, median={np.median(tpss):.1f}")

    print(f"  Success rate: {len(successful)}/{len(results)}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="CCU1 Benchmark Runner")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Benchmark config")
    parser.add_argument("--project-dir", default=".", help="Project root")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080", help="Server URL")
    parser.add_argument("--output", default="results/raw/ccu1_requests.csv", help="Output CSV")
    parser.add_argument("--model-label", required=True, help="Model label (e.g., F16, Q4_K_M)")
    parser.add_argument("--requested-precision", default="", help="Requested precision label")
    parser.add_argument("--actual-quant", default="", help="Actual GGUF quant type")
    parser.add_argument("--support-type", default="", help="Support type")
    parser.add_argument("--sha256", default="", help="Model SHA256")
    parser.add_argument("--gpu-name", default="", help="GPU name")
    parser.add_argument("--llama-commit", default="", help="llama.cpp commit")
    parser.add_argument("--context-per-slot", type=int, default=8192)
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3"],
                        help="Scenarios to run (S1, S2, S3, S5)")
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)

    # Load config
    with open(os.path.join(project_dir, args.config)) as f:
        config = yaml.safe_load(f)

    bench_config = config["benchmark"]
    prompt = bench_config["generation"]["prompt"]
    suite_id = f"{bench_config['suite_id_prefix']}_{args.model_label}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Load manifest
    manifest_path = os.path.join(project_dir, "benchmark_data/manifest.jsonl")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("[ERROR] Run prepare_dataset.py first.")
        sys.exit(1)

    # Model info for metadata
    model_info = {
        "label": args.model_label,
        "requested_precision": args.requested_precision or args.model_label,
        "actual_quant_type": args.actual_quant or args.model_label,
        "support_type": args.support_type,
        "sha256": args.sha256,
        "mmproj_type": "f16",
        "llama_cpp_commit": args.llama_commit,
        "gpu_name": args.gpu_name,
        "context_per_slot": args.context_per_slot,
        "kv_k_type": bench_config["server"]["kv_k_type"],
        "kv_v_type": bench_config["server"]["kv_v_type"],
    }

    # Scenario mapping
    scenario_map = {
        "S1": {"dataset": "small", "runs": 10, "max_tokens": 256},
        "S2": {"dataset": "normal", "runs": 30, "max_tokens": 256},
        "S3": {"dataset": "large", "runs": 10, "max_tokens": 256},
        "S5": {"dataset": "normal", "runs": 10, "max_tokens": 512},
    }

    all_results = []

    print("=" * 60)
    print(f" CCU1 BENCHMARK — {args.model_label}")
    print(f" Suite: {suite_id}")
    print(f" Scenarios: {args.scenarios}")
    print("=" * 60)

    # Warm-up (§13)
    if not args.skip_warmup:
        warmup_images = load_image_manifest(manifest_path, "small")
        if not warmup_images:
            warmup_images = load_image_manifest(manifest_path, "normal")
        if warmup_images:
            run_warmup(
                server_url=args.server_url,
                images=warmup_images,
                prompt=prompt,
                project_dir=project_dir,
                warmup_count=bench_config["warmup_requests"],
            )

    # Run scenarios
    for scenario_id in args.scenarios:
        if scenario_id not in scenario_map:
            print(f"[WARN] Unknown scenario: {scenario_id}, skipping")
            continue

        scenario = scenario_map[scenario_id]
        images = load_image_manifest(manifest_path, scenario["dataset"])

        if not images:
            print(f"[WARN] No images for dataset '{scenario['dataset']}', skipping {scenario_id}")
            continue

        results = run_scenario(
            scenario_id=scenario_id,
            server_url=args.server_url,
            images=images,
            prompt=prompt,
            project_dir=project_dir,
            runs=scenario["runs"],
            max_tokens=scenario["max_tokens"],
            model_info=model_info,
            suite_id=suite_id,
        )
        all_results.extend(results)

    # Save results
    output_path = os.path.join(project_dir, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if all_results:
        fieldnames = list(all_results[0].keys())

        # Append mode if file exists
        file_exists = os.path.exists(output_path)
        mode = "a" if file_exists else "w"
        with open(output_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_results)

        print(f"\n[SUCCESS] CCU1 results saved to: {output_path}")
        print(f"  Total requests: {len(all_results)}")
        print(f"  Successful: {sum(1 for r in all_results if r['success'])}")
    else:
        print("\n[WARN] No results to save.")


if __name__ == "__main__":
    main()
