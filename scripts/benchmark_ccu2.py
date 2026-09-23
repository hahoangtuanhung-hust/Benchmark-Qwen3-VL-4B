#!/usr/bin/env python3
"""
CCU2 Benchmark Runner — Concurrent dual-user workloads.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §15 (S4)

Runs benchmark with CCU=2: two concurrent requests using asyncio.gather().
Each pair uses different images to avoid cache/reuse issues.
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import sys
import time
import uuid
from typing import Optional

import yaml

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("[ERROR] aiohttp required for CCU2. Install: pip install aiohttp")

import numpy as np

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_client import BenchmarkResult, compute_itl_stats, encode_image_base64


async def send_vlm_request_async(
    session: aiohttp.ClientSession,
    server_url: str,
    image_path: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 42,
    timeout: int = 300,
) -> BenchmarkResult:
    """
    Async version of VLM request for concurrent benchmark.
    Uses aiohttp for non-blocking streaming.
    """
    result = BenchmarkResult()
    result.run_id = str(uuid.uuid4())[:8]
    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result.temperature = temperature
    result.max_tokens = max_tokens
    result.cache_prompt = False

    # Parse image info
    if os.path.exists(image_path):
        try:
            from PIL import Image
            img = Image.open(image_path)
            result.image_width, result.image_height = img.size
        except Exception:
            pass

    # Encode image
    try:
        image_b64 = encode_image_base64(image_path)
    except Exception as e:
        result.error_type = f"image_encode_error: {e}"
        return result

    # Build payload
    payload = {
        "model": "qwen3-vl",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "seed": seed,
        "cache_prompt": False,
    }

    url = f"{server_url}/v1/chat/completions"
    token_times = []
    full_content = []
    first_token_time = None
    first_visible_time = None
    last_response_data = {}

    request_start = time.perf_counter()

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"Accept": "text/event-stream"},
        ) as response:
            result.http_status = response.status

            if response.status != 200:
                result.error_type = f"http_error_{response.status}"
                body = await response.text()
                result.answer_text = body[:500]
                return result

            # Parse SSE stream
            async for line in response.content:
                line = line.decode("utf-8", errors="replace").strip()

                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    last_response_data = data

                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            now = time.perf_counter()
                            token_times.append(now)
                            full_content.append(content)

                            if first_token_time is None:
                                first_token_time = now

                            if first_visible_time is None:
                                stripped = content.strip()
                                if stripped and not stripped.startswith("<think"):
                                    first_visible_time = now

                except json.JSONDecodeError:
                    continue

        request_end = time.perf_counter()

        # Calculate metrics
        result.success = True
        result.answer_text = "".join(full_content)[:1000]
        result.output_tokens = len(full_content)

        result.e2e_ms = (request_end - request_start) * 1000.0

        if first_token_time is not None:
            result.ttft_ms = (first_token_time - request_start) * 1000.0

        if first_visible_time is not None:
            result.ttfv_ms = (first_visible_time - request_start) * 1000.0
        elif first_token_time is not None:
            result.ttfv_ms = result.ttft_ms

        # ITL
        itl_stats = compute_itl_stats(token_times)
        result.itl_mean_ms = itl_stats["itl_mean_ms"]
        result.itl_p50_ms = itl_stats["itl_p50_ms"]
        result.itl_p95_ms = itl_stats["itl_p95_ms"]
        result.itl_p99_ms = itl_stats["itl_p99_ms"]
        result.itl_measurement_method = "stream_event_itl"

        # Server timings
        from benchmark_client import parse_server_timings
        server_timings = parse_server_timings(last_response_data)
        if "prefill_tps" in server_timings:
            result.prefill_tps = server_timings["prefill_tps"]
            result.prefill_tps_source = server_timings.get("prefill_tps_source", "")
        if "decode_tps" in server_timings:
            result.decode_tps = server_timings["decode_tps"]
            result.decode_tps_source = server_timings.get("decode_tps_source", "")
        if "input_tokens" in server_timings:
            result.input_tokens = server_timings["input_tokens"]
        if "output_tokens" in server_timings:
            result.output_tokens = server_timings["output_tokens"]

        # Client-side decode TPS fallback
        if result.decode_tps is None and len(token_times) >= 2:
            decode_duration = token_times[-1] - token_times[0]
            if decode_duration > 0:
                result.decode_tps = (len(token_times) - 1) / decode_duration
                result.decode_tps_source = "client_fallback"

    except asyncio.TimeoutError:
        result.error_type = "timeout"
    except aiohttp.ClientError as e:
        result.error_type = f"connection_error: {e}"
    except Exception as e:
        result.error_type = f"unknown_error: {e}"

    return result


async def run_ccu2_pair(
    session: aiohttp.ClientSession,
    server_url: str,
    image_a_path: str,
    image_b_path: str,
    prompt: str,
    max_tokens: int,
    pair_id: str,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run two concurrent VLM requests (§15).
    Uses asyncio.gather() for true concurrency.
    """
    pair_start = time.perf_counter()

    result_a, result_b = await asyncio.gather(
        send_vlm_request_async(session, server_url, image_a_path, prompt, max_tokens),
        send_vlm_request_async(session, server_url, image_b_path, prompt, max_tokens),
    )

    pair_end = time.perf_counter()

    # Annotate pair info (§15)
    result_a.pair_id = pair_id
    result_a.ccu = 2
    result_b.pair_id = pair_id
    result_b.ccu = 2

    return result_a, result_b


def load_image_manifest(manifest_path: str, category: str) -> list[dict]:
    """Load images from manifest filtered by category."""
    images = []
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("category") == category:
                images.append(entry)
    return images


async def run_ccu2_benchmark(
    server_url: str,
    images: list[dict],
    prompt: str,
    project_dir: str,
    pairs: int,
    max_tokens: int,
    model_info: dict,
    suite_id: str,
) -> list[dict]:
    """Run CCU2 benchmark — S4 scenario."""
    all_results = []

    print(f"[CCU2] Starting: {pairs} concurrent pairs, max_tokens={max_tokens}")
    print(f"  Images available: {len(images)}")
    print(f"  Using different images for each request in a pair")

    # Warmup (§13)
    print(f"\n[WARMUP] Running warmup requests...")
    async with aiohttp.ClientSession() as session:
        for i in range(5):
            idx = i % len(images)
            image_path = os.path.join(project_dir, images[idx]["path"])
            result = await send_vlm_request_async(
                session, server_url, image_path, prompt, max_tokens=64
            )
            status = "OK" if result.success else f"FAIL ({result.error_type})"
            print(f"  Warmup {i+1}/5: {status}")
    print("[WARMUP] Complete. Results discarded.\n")

    # Run pairs
    async with aiohttp.ClientSession() as session:
        for i in range(pairs):
            pair_id = f"pair_{i:04d}"

            # Select TWO DIFFERENT images (§15)
            idx_a = (i * 2) % len(images)
            idx_b = (i * 2 + 1) % len(images)
            if idx_b == idx_a:
                idx_b = (idx_a + 1) % len(images)

            image_a_path = os.path.join(project_dir, images[idx_a]["path"])
            image_b_path = os.path.join(project_dir, images[idx_b]["path"])

            pair_start = time.perf_counter()
            result_a, result_b = await run_ccu2_pair(
                session, server_url,
                image_a_path, image_b_path,
                prompt, max_tokens, pair_id,
            )
            pair_end = time.perf_counter()
            pair_duration_ms = (pair_end - pair_start) * 1000.0

            # Enrich with metadata
            for result in [result_a, result_b]:
                result.suite_id = suite_id
                result.scenario = "S4"
                result.model_name = model_info.get("label", "")
                result.requested_precision = model_info.get("requested_precision", "")
                result.actual_quant_type = model_info.get("actual_quant_type", "")
                result.support_type = model_info.get("support_type", "")
                result.gguf_sha256 = model_info.get("sha256", "")
                result.mmproj_type = model_info.get("mmproj_type", "f16")
                result.llama_cpp_commit = model_info.get("llama_cpp_commit", "")
                result.gpu_name = model_info.get("gpu_name", "")
                result.context_per_slot = model_info.get("context_per_slot", 8192)

            result_a.image_id = images[idx_a].get("image_id", "")
            result_b.image_id = images[idx_b].get("image_id", "")

            all_results.append(result_a.to_dict())
            all_results.append(result_b.to_dict())

            # Progress
            a_ok = "OK" if result_a.success else "FAIL"
            b_ok = "OK" if result_b.success else "FAIL"
            a_ttft = f"{result_a.ttft_ms:.0f}ms" if result_a.ttft_ms else "N/A"
            b_ttft = f"{result_b.ttft_ms:.0f}ms" if result_b.ttft_ms else "N/A"
            a_tps = f"{result_a.decode_tps:.1f}" if result_a.decode_tps else "N/A"
            b_tps = f"{result_b.decode_tps:.1f}" if result_b.decode_tps else "N/A"
            print(f"  Pair {i+1:3d}/{pairs} | A: {a_ok} TTFT={a_ttft} TPS={a_tps} | B: {b_ok} TTFT={b_ttft} TPS={b_tps} | pair_dur={pair_duration_ms:.0f}ms")

    # Summary
    successful = [r for r in all_results if r["success"]]
    if successful:
        ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
        tpss = [r["decode_tps"] for r in successful if r["decode_tps"] is not None]

        if ttfts:
            print(f"\n  CCU2 TTFT: mean={np.mean(ttfts):.0f}ms, P50={np.percentile(ttfts,50):.0f}ms, P95={np.percentile(ttfts,95):.0f}ms")
        if tpss:
            per_user_tps = np.mean(tpss)
            aggregate_tps = per_user_tps * 2  # Approximate
            print(f"  CCU2 Decode TPS/user: mean={per_user_tps:.1f}")
            print(f"  CCU2 Aggregate TPS: ~{aggregate_tps:.1f}")

    print(f"  Success rate: {len(successful)}/{len(all_results)}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="CCU2 Benchmark Runner")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Benchmark config")
    parser.add_argument("--project-dir", default=".", help="Project root")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080", help="Server URL")
    parser.add_argument("--output", default="results/raw/ccu2_requests.csv", help="Output CSV")
    parser.add_argument("--model-label", required=True, help="Model label")
    parser.add_argument("--requested-precision", default="")
    parser.add_argument("--actual-quant", default="")
    parser.add_argument("--support-type", default="")
    parser.add_argument("--sha256", default="")
    parser.add_argument("--gpu-name", default="")
    parser.add_argument("--llama-commit", default="")
    parser.add_argument("--context-per-slot", type=int, default=8192)
    parser.add_argument("--pairs", type=int, default=30, help="Number of concurrent pairs")
    args = parser.parse_args()

    if not HAS_AIOHTTP:
        print("[ERROR] aiohttp required. Install: pip install aiohttp")
        sys.exit(1)

    project_dir = os.path.abspath(args.project_dir)

    with open(os.path.join(project_dir, args.config)) as f:
        config = yaml.safe_load(f)

    bench_config = config["benchmark"]
    prompt = bench_config["generation"]["prompt"]
    suite_id = f"{bench_config['suite_id_prefix']}_{args.model_label}_ccu2_{time.strftime('%Y%m%d_%H%M%S')}"

    manifest_path = os.path.join(project_dir, "benchmark_data/manifest.jsonl")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    images = load_image_manifest(manifest_path, "normal")
    if not images:
        print("[ERROR] No 'normal' images in manifest")
        sys.exit(1)

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
    }

    print("=" * 60)
    print(f" CCU2 BENCHMARK — {args.model_label}")
    print(f" Suite: {suite_id}")
    print(f" Pairs: {args.pairs}")
    print("=" * 60)

    all_results = asyncio.run(run_ccu2_benchmark(
        server_url=args.server_url,
        images=images,
        prompt=prompt,
        project_dir=project_dir,
        pairs=args.pairs,
        max_tokens=bench_config["generation"]["max_tokens"],
        model_info=model_info,
        suite_id=suite_id,
    ))

    # Save results
    output_path = os.path.join(project_dir, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if all_results:
        fieldnames = list(all_results[0].keys())
        file_exists = os.path.exists(output_path)
        mode = "a" if file_exists else "w"
        with open(output_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_results)

        print(f"\n[SUCCESS] CCU2 results saved to: {output_path}")
    else:
        print("\n[WARN] No results.")


if __name__ == "__main__":
    main()
