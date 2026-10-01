#!/usr/bin/env python3
"""
Core Benchmark Client — Send VLM requests and measure latency/throughput.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §16, §17, §23

This module provides the core request function that:
1. Sends image + prompt via streaming SSE to llama-server /v1/chat/completions
2. Measures TTFT, TTFV, ITL, E2E latency
3. Collects server timings (prefill TPS, decode TPS)
4. Returns a structured result dict matching the raw request schema (§23)
"""

import base64
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class BenchmarkResult:
    """Result of a single benchmark request (§23 schema)."""
    run_id: str = ""
    suite_id: str = ""
    timestamp: str = ""
    model_name: str = ""
    model_revision: str = ""
    requested_precision: str = ""
    actual_quant_type: str = ""
    support_type: str = ""
    gguf_sha256: str = ""
    mmproj_type: str = ""
    mmproj_sha256: str = ""
    llama_cpp_commit: str = ""
    gpu_name: str = ""
    gpu_count: int = 1
    cuda_version: str = ""
    ccu: int = 1
    pair_id: str = ""
    scenario: str = ""
    image_id: str = ""
    image_width: int = 0
    image_height: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_per_slot: int = 8192
    kv_k_type: str = "f16"
    kv_v_type: str = "f16"
    cache_prompt: bool = False
    temperature: float = 0.0
    max_tokens: int = 256
    http_status: int = 0
    success: bool = False
    error_type: str = ""
    ttft_ms: Optional[float] = None
    ttfv_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    itl_mean_ms: Optional[float] = None
    itl_p50_ms: Optional[float] = None
    itl_p95_ms: Optional[float] = None
    itl_p99_ms: Optional[float] = None
    itl_measurement_method: str = "stream_event_itl"
    prefill_tps: Optional[float] = None
    prefill_tps_source: str = ""
    decode_tps: Optional[float] = None
    decode_tps_source: str = ""
    baseline_vram_mb: Optional[float] = None
    idle_vram_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    gpu_util_mean_pct: Optional[float] = None
    gpu_util_peak_pct: Optional[float] = None
    kv_cache_measured_mb: Optional[float] = None
    kv_cache_estimated_mb: Optional[float] = None
    kv_cache_measurement_method: str = ""
    answer_text: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV output."""
        return {k: v for k, v in self.__dict__.items()}


def encode_image_base64(image_path: str) -> str:
    """Read image file and return base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def compute_itl_stats(token_times: list[float]) -> dict:
    """
    Compute Inter-Token Latency statistics from token arrival timestamps.
    §16.4 — ITL_i = t_i - t_(i-1)
    """
    if len(token_times) < 2:
        return {
            "itl_mean_ms": None,
            "itl_p50_ms": None,
            "itl_p95_ms": None,
            "itl_p99_ms": None,
        }

    import numpy as np

    intervals = []
    for i in range(1, len(token_times)):
        itl = (token_times[i] - token_times[i - 1]) * 1000.0  # ms
        intervals.append(itl)

    intervals = np.array(intervals)
    return {
        "itl_mean_ms": float(np.mean(intervals)),
        "itl_p50_ms": float(np.percentile(intervals, 50)),
        "itl_p95_ms": float(np.percentile(intervals, 95)),
        "itl_p99_ms": float(np.percentile(intervals, 99)),
    }


def parse_server_timings(response_data: dict) -> dict:
    """
    Extract server-side timings from llama.cpp response.
    §16.5, §16.6 — prefill and decode TPS from server timings
    """
    timings = {}

    # Try to get from usage/timings in the final response
    usage = response_data.get("usage", {})
    if usage:
        timings["input_tokens"] = usage.get("prompt_tokens", 0)
        timings["output_tokens"] = usage.get("completion_tokens", 0)

    # llama.cpp specific timings (if available in response)
    llama_timings = response_data.get("timings", {})
    if llama_timings:
        # Prefill TPS
        prompt_per_sec = llama_timings.get("prompt_per_second")
        if prompt_per_sec:
            timings["prefill_tps"] = prompt_per_sec
            timings["prefill_tps_source"] = "llama.cpp timings.prompt_per_second"

        # Decode TPS
        predicted_per_sec = llama_timings.get("predicted_per_second")
        if predicted_per_sec:
            timings["decode_tps"] = predicted_per_sec
            timings["decode_tps_source"] = "llama.cpp timings.predicted_per_second"

        # Token counts from timings
        if "prompt_n" in llama_timings:
            timings["input_tokens"] = llama_timings["prompt_n"]
        if "predicted_n" in llama_timings:
            timings["output_tokens"] = llama_timings["predicted_n"]

    return timings


def send_vlm_request(
    server_url: str,
    image_path: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 42,
    cache_prompt: bool = False,
    timeout: int = 300,
    model_name: str = "qwen3-vl",
) -> BenchmarkResult:
    """
    Send a VLM request to llama-server and measure all metrics.

    Uses /v1/chat/completions with streaming for accurate TTFT measurement.
    §16.1 TTFT = timestamp(first generated token) - timestamp(request sent)
    §16.2 TTFV = timestamp(first user-visible answer token) - timestamp(request sent)
    """
    result = BenchmarkResult()
    result.run_id = str(uuid.uuid4())[:8]
    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result.temperature = temperature
    result.max_tokens = max_tokens
    result.cache_prompt = cache_prompt

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

    # Build request payload (OpenAI-compatible format)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{'png' if image_path.lower().endswith('.png') else 'jpeg'};base64,{image_b64}"
                        }
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
    }

    url = f"{server_url}/v1/chat/completions"

    # ─── Send request and measure ───
    token_times = []
    full_content = []
    first_token_time = None
    first_visible_time = None
    last_response_data = {}

    request_start = time.perf_counter()

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=timeout,
            headers={"Accept": "text/event-stream"},
        )
        result.http_status = response.status_code

        if response.status_code != 200:
            result.error_type = f"http_error_{response.status_code}"
            try:
                result.answer_text = response.text[:500]
            except Exception:
                pass
            return result

        # Parse SSE stream
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue

            data_str = line[6:]  # Remove "data: " prefix
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                last_response_data = data

                # Extract content delta
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        now = time.perf_counter()
                        token_times.append(now)
                        full_content.append(content)

                        # TTFT — first token (§16.1)
                        if first_token_time is None:
                            first_token_time = now

                        # TTFV — first visible token (§16.2)
                        # Skip reasoning/thinking tokens
                        if first_visible_time is None:
                            # Check if this is a reasoning/hidden token
                            stripped = content.strip()
                            if stripped and not stripped.startswith("<think"):
                                first_visible_time = now

            except json.JSONDecodeError:
                continue

        request_end = time.perf_counter()

        # ─── Calculate metrics ───
        result.success = True
        result.answer_text = "".join(full_content)[:1000]  # Truncate for CSV
        result.output_tokens = len(full_content)  # Approximate by stream events

        # E2E latency (§16.3)
        result.e2e_ms = (request_end - request_start) * 1000.0

        # TTFT (§16.1)
        if first_token_time is not None:
            result.ttft_ms = (first_token_time - request_start) * 1000.0

        # TTFV (§16.2)
        if first_visible_time is not None:
            result.ttfv_ms = (first_visible_time - request_start) * 1000.0
        elif first_token_time is not None:
            # If no reasoning tokens, TTFV ≈ TTFT
            result.ttfv_ms = result.ttft_ms

        # ITL statistics (§16.4)
        itl_stats = compute_itl_stats(token_times)
        result.itl_mean_ms = itl_stats["itl_mean_ms"]
        result.itl_p50_ms = itl_stats["itl_p50_ms"]
        result.itl_p95_ms = itl_stats["itl_p95_ms"]
        result.itl_p99_ms = itl_stats["itl_p99_ms"]
        result.itl_measurement_method = "stream_event_itl"

        # Server timings (§16.5, §16.6)
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

        # Client-side decode TPS fallback (§16.5)
        if result.decode_tps is None and len(token_times) >= 2:
            decode_duration = token_times[-1] - token_times[0]
            if decode_duration > 0:
                result.decode_tps = (len(token_times) - 1) / decode_duration
                result.decode_tps_source = "client_fallback"

    except requests.Timeout:
        result.error_type = "timeout"
        result.e2e_ms = timeout * 1000.0
    except requests.ConnectionError as e:
        result.error_type = f"connection_error: {e}"
    except Exception as e:
        result.error_type = f"unknown_error: {e}"

    return result


def collect_server_metrics_snapshot(server_url: str) -> dict:
    """Collect a snapshot of /metrics endpoint (Prometheus format)."""
    try:
        resp = requests.get(f"{server_url}/metrics", timeout=5)
        if resp.status_code == 200:
            metrics = {}
            for line in resp.text.split("\n"):
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics[parts[0]] = float(parts[1])
            return metrics
    except Exception:
        pass
    return {}


def collect_slots_snapshot(server_url: str) -> list[dict]:
    """Collect /slots endpoint data."""
    try:
        resp = requests.get(f"{server_url}/slots", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []
