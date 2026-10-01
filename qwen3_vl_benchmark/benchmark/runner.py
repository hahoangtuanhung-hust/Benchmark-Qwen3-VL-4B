from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from ..backends.llamacpp import LlamaCppBackend
from ..backends.tensorrt_edge import TensorRTEdgeBackend
from ..common.backend import GenerationConfig, InferenceBackend


RAW_FIELDS = [
    "backend", "run_id", "pair_id", "scenario", "requested_precision", "actual_precision",
    "quant_recipe", "gpu", "gpu_arch", "image_id", "width", "height", "input_tokens",
    "output_tokens", "ccu", "ttft_ms", "ttfv_ms", "e2e_ms", "image_load_ms",
    "image_preprocess_ms", "vision_encode_ms", "visual_projection_ms", "prefill_ms",
    "first_decode_ms", "prefill_scope", "prefill_tps", "decode_tps", "decode_tps_source",
    "tpot_ms", "itl_mean_ms", "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
    "itl_measurement_method", "idle_vram_mb", "peak_vram_mb", "kv_cache_mb",
    "gpu_util_mean", "gpu_util_peak", "http_status", "success", "error", "answer_text",
]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_images(manifest: Path, category: str) -> list[dict[str, Any]]:
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("category") == category:
                rows.append(row)
    return rows


def _native(metrics: dict[str, Any], *names: str) -> Any:
    for name in names:
        if metrics.get(name) is not None:
            return metrics[name]
    return None


def measure_one(
    backend: InferenceBackend, image: Path, image_info: dict[str, Any], prompt: str,
    generation: GenerationConfig, metadata: dict[str, Any], scenario: str, ccu: int,
    pair_id: str = "",
) -> dict[str, Any]:
    result = backend.generate(image, prompt, generation)
    events = result.token_events
    first = events[0].timestamp if events else None
    visible = next((event.timestamp for event in events if event.visible), first)
    # Stream events are not claimed as token-level when the server does not provide token counts.
    token_level = bool(events) and all(event.token_count == 1 for event in events)
    intervals = ([(events[i].timestamp - events[i - 1].timestamp) * 1000 for i in range(1, len(events))]
                 if token_level else [])
    output_tokens = result.output_tokens
    
    # Use accurate request timings that exclude image encoding overhead
    started = result.request_start if hasattr(result, "request_start") and result.request_start else time.perf_counter() - 0.001
    ended = result.request_end if hasattr(result, "request_end") and result.request_end else time.perf_counter()
    
    e2e_ms = (ended - started) * 1000
    ttft_ms = (first - started) * 1000 if first else None
    decode_seconds = events[-1].timestamp - events[0].timestamp if len(events) > 1 else 0
    decode_tps = _native(result.native_metrics, "decode_tps", "predicted_per_second")
    decode_source = "runtime_native" if decode_tps is not None else ""
    if decode_tps is None and decode_seconds > 0 and output_tokens > 1:
        decode_tps = (output_tokens - 1) / decode_seconds
        decode_source = "client_fallback"
    row = {field: None for field in RAW_FIELDS}
    row.update({
        "backend": backend.name, "run_id": uuid.uuid4().hex[:12], "pair_id": pair_id,
        "scenario": scenario, "requested_precision": metadata.get("requested_precision"),
        "actual_precision": metadata.get("actual_precision"), "quant_recipe": metadata.get("quant_recipe"),
        "gpu": metadata.get("gpu"), "gpu_arch": metadata.get("gpu_arch"),
        "image_id": image_info.get("image_id"), "width": image_info.get("width"),
        "height": image_info.get("height"), "input_tokens": result.input_tokens,
        "output_tokens": output_tokens, "ccu": ccu, "ttft_ms": ttft_ms,
        "ttfv_ms": (visible - started) * 1000 if visible else None, "e2e_ms": e2e_ms,
        "vision_encode_ms": _native(result.native_metrics, "vision_encode_ms"),
        "visual_projection_ms": _native(result.native_metrics, "visual_projection_ms"),
        "prefill_ms": _native(result.native_metrics, "prefill_ms", "prompt_ms"),
        "first_decode_ms": _native(result.native_metrics, "first_decode_ms"),
        "prefill_scope": _native(result.native_metrics, "prefill_scope") or "combined_or_unavailable",
        "prefill_tps": _native(result.native_metrics, "prefill_tps", "prompt_per_second"),
        "decode_tps": decode_tps, "decode_tps_source": decode_source,
        "tpot_ms": (e2e_ms - ttft_ms) / (output_tokens - 1) if ttft_ms is not None and output_tokens > 1 else None,
        "itl_mean_ms": statistics.fmean(intervals) if intervals else None,
        "itl_p50_ms": percentile(intervals, .50), "itl_p95_ms": percentile(intervals, .95),
        "itl_p99_ms": percentile(intervals, .99),
        "itl_measurement_method": "token_timestamp" if token_level else "stream_event_not_token_level",
        "kv_cache_mb": _native(result.native_metrics, "kv_cache_mb", "kv_cache_allocated_mb"),
        "http_status": result.http_status, "success": result.success, "error": result.error,
        "answer_text": result.text[:1000],
    })
    return row


def warmup(backend: InferenceBackend, images: list[dict[str, Any]], root: Path, prompt: str, count: int) -> None:
    config = GenerationConfig(max_tokens=64)
    for index in range(count):
        info = images[index % len(images)]
        result = backend.generate(root / info["path"], prompt, config)
        if not result.success:
            raise RuntimeError(f"warm-up request failed: {result.error}")


def run_ccu1(backend, images, root, prompt, generation, metadata, scenario, runs):
    return [measure_one(backend, root / images[i % len(images)]["path"], images[i % len(images)],
                        prompt, generation, metadata, scenario, 1) for i in range(runs)]


def run_ccu2(backend, images, root, prompt, generation, metadata, scenario, pairs):
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index in range(pairs):
            barrier = threading.Barrier(2)
            pair_id = f"pair-{index + 1:03d}"
            def submit(offset: int):
                info = images[(index * 2 + offset) % len(images)]
                barrier.wait()
                return measure_one(backend, root / info["path"], info, prompt, generation,
                                   metadata, scenario, 2, pair_id)
            futures = [pool.submit(submit, 0), pool.submit(submit, 1)]
            rows.extend(future.result() for future in futures)
    return rows


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("llamacpp", "tensorrt_edge"), required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--scenario", choices=("S1", "S2", "S3", "S4", "S5"), required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--actual-precision", default="")
    parser.add_argument("--quant-recipe", default="none")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--gpu-arch", default="")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--output", default="results/tensorrt_edge/raw_requests.csv")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_dir).resolve()
    config = yaml.safe_load((root / "configs/benchmark.yaml").read_text(encoding="utf-8"))["benchmark"]
    scenarios = {
        "S1": ("small", 10, 256, 1), "S2": ("normal", 30, 256, 1),
        "S3": ("large", 10, 256, 1), "S4": ("normal", 30, 256, 2),
        "S5": ("normal", 10, 512, 1),
    }
    category, count, max_tokens, ccu = scenarios[args.scenario]
    count = args.runs or count
    images = load_images(root / "benchmark_data/manifest.jsonl", category)
    if not images:
        raise RuntimeError(f"no {category} images in frozen manifest")
    backend = (LlamaCppBackend if args.backend == "llamacpp" else TensorRTEdgeBackend)(base_url=args.server_url)
    if not backend.health():
        raise RuntimeError(f"{args.backend} is not healthy at {args.server_url}")
    prompt = config["generation"]["prompt"]
    if not args.skip_warmup:
        warmup(backend, images, root, prompt, int(config["warmup_requests"]))
    generation = GenerationConfig(max_tokens=max_tokens)
    metadata = {"requested_precision": args.precision, "actual_precision": args.actual_precision or args.precision,
                "quant_recipe": args.quant_recipe, "gpu": args.gpu, "gpu_arch": args.gpu_arch}
    rows = (run_ccu1(backend, images, root, prompt, generation, metadata, args.scenario, count)
            if ccu == 1 else run_ccu2(backend, images, root, prompt, generation, metadata, args.scenario, count))
    append_csv(root / args.output, rows)
    failures = sum(not row["success"] for row in rows)
    print(f"wrote {len(rows)} requests ({failures} failed) to {args.output}")
    return 1 if failures / len(rows) > .05 else 0


if __name__ == "__main__":
    raise SystemExit(main())
