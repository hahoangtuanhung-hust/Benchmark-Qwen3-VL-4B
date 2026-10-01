#!/usr/bin/env python3
"""
GPU Monitor — Background process to sample GPU metrics.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §18

Runs in background during benchmark, writing CSV rows at configurable interval.
Uses nvidia-smi by default; can switch to pynvml if available.

Usage:
    python gpu_monitor.py --output results/gpu/gpu_log.csv --interval 200
    # Stop with SIGTERM or SIGINT (Ctrl+C)
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
import requests

def get_kv_cache_usage() -> float:
    """Fetch the KV cache usage ratio from llama.cpp server."""
    try:
        resp = requests.get("http://127.0.0.1:8080/metrics", timeout=0.1)
        for line in resp.text.split("\n"):
            if line.startswith("llamacpp:kv_cache_usage_ratio"):
                return float(line.split()[1])
    except Exception:
        pass
    return 0.0


# --- Global stop flag ---
_stop = False


def _signal_handler(signum, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def query_nvidia_smi() -> list[dict]:
    """Query GPU metrics via nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []

        rows = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                rows.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "gpu_index": int(parts[0]),
                    "memory_used_mb": float(parts[1]),
                    "memory_total_mb": float(parts[2]),
                    "gpu_util_pct": float(parts[3]),
                    "memory_util_pct": float(parts[4]),
                    "power_w": float(parts[5]) if parts[5] != "[N/A]" else None,
                    "temperature_c": float(parts[6]) if parts[6] != "[N/A]" else None,
                })
        return rows
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"[WARN] nvidia-smi query failed: {e}", file=sys.stderr)
        return []


def run_monitor(output_path: str, interval_ms: int, gpu_index: int = 0):
    """Main monitoring loop."""
    global _stop

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "timestamp", "gpu_index", "memory_used_mb", "memory_total_mb",
        "gpu_util_pct", "memory_util_pct", "power_w", "temperature_c",
        "kv_cache_usage_ratio"
    ]

    interval_s = interval_ms / 1000.0
    sample_count = 0

    print(f"[GPU Monitor] Output: {output_path}")
    print(f"[GPU Monitor] Interval: {interval_ms}ms, GPU index: {gpu_index}")
    print(f"[GPU Monitor] Press Ctrl+C or send SIGTERM to stop.")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not _stop:
            rows = query_nvidia_smi()
            kv_ratio = get_kv_cache_usage()

            for row in rows:
                if row["gpu_index"] == gpu_index or gpu_index == -1:
                    row["kv_cache_usage_ratio"] = kv_ratio
                    writer.writerow(row)
                    sample_count += 1

            f.flush()
            time.sleep(interval_s)

    print(f"[GPU Monitor] Stopped. Total samples: {sample_count}")


def main():
    parser = argparse.ArgumentParser(description="GPU monitoring for benchmark")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--interval", type=int, default=200, help="Sampling interval in ms")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index to monitor (-1 for all)")
    args = parser.parse_args()

    run_monitor(args.output, args.interval, args.gpu_index)


if __name__ == "__main__":
    main()
