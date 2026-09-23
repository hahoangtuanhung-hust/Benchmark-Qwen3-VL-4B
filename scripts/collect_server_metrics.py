#!/usr/bin/env python3
"""
Collect llama-server /metrics and /slots endpoints.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §22

Captures Prometheus metrics and slot status from llama-server.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests


METRICS_OF_INTEREST = [
    "llamacpp:prompt_tokens_total",
    "llamacpp:prompt_seconds_total",
    "llamacpp:prompt_tokens_seconds",
    "llamacpp:tokens_predicted_total",
    "llamacpp:tokens_predicted_seconds_total",
    "llamacpp:predicted_tokens_seconds",
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
    "llamacpp:n_tokens_max",
    "llamacpp:n_decode_total",
    "llamacpp:n_busy_slots_per_decode",
]


def parse_prometheus_metrics(text: str) -> dict:
    """Parse Prometheus text format into dict."""
    metrics = {}
    for line in text.strip().split("\n"):
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    metrics[parts[0]] = parts[1]
    return metrics


def collect_metrics(server_url: str) -> dict:
    """Collect /metrics endpoint."""
    try:
        resp = requests.get(f"{server_url}/metrics", timeout=5)
        if resp.status_code == 200:
            metrics = parse_prometheus_metrics(resp.text)
            return {
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw": resp.text,
                "parsed": metrics,
            }
        return {"status": f"http_error_{resp.status_code}"}
    except Exception as e:
        return {"status": f"error: {e}"}


def collect_slots(server_url: str) -> dict:
    """Collect /slots endpoint."""
    try:
        resp = requests.get(f"{server_url}/slots", timeout=5)
        if resp.status_code == 200:
            return {
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "slots": resp.json(),
            }
        return {"status": f"http_error_{resp.status_code}"}
    except Exception as e:
        return {"status": f"error: {e}"}


def collect_and_save(server_url: str, output_dir: str, label: str = ""):
    """Collect both /metrics and /slots and save to files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{label}_{timestamp}" if label else timestamp

    # Collect /metrics
    metrics_result = collect_metrics(server_url)
    metrics_file = os.path.join(output_dir, f"metrics_{prefix}.json")
    with open(metrics_file, "w") as f:
        # Save parsed metrics (not raw Prometheus text)
        json.dump({
            "status": metrics_result["status"],
            "timestamp": metrics_result.get("timestamp", ""),
            "metrics": metrics_result.get("parsed", {}),
        }, f, indent=2)

    # Also save raw Prometheus format
    if "raw" in metrics_result:
        raw_file = os.path.join(output_dir, f"metrics_{prefix}_raw.txt")
        with open(raw_file, "w") as f:
            f.write(metrics_result["raw"])

    # Collect /slots
    slots_result = collect_slots(server_url)
    slots_file = os.path.join(output_dir, f"slots_{prefix}.json")
    with open(slots_file, "w") as f:
        json.dump(slots_result, f, indent=2)

    # Print summary
    print(f"[Metrics] {metrics_result['status']}")
    if metrics_result["status"] == "ok":
        parsed = metrics_result.get("parsed", {})
        for key in METRICS_OF_INTEREST:
            if key in parsed:
                print(f"  {key}: {parsed[key]}")

    print(f"[Slots] {slots_result['status']}")
    if slots_result["status"] == "ok":
        slots = slots_result.get("slots", [])
        for slot in slots:
            slot_id = slot.get("id", "?")
            state = slot.get("state", "?")
            print(f"  Slot {slot_id}: {state}")

    return metrics_result, slots_result


def main():
    parser = argparse.ArgumentParser(description="Collect llama-server metrics and slots")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080", help="Server URL")
    parser.add_argument("--output-dir", default="results/server_metrics", help="Output directory")
    parser.add_argument("--label", default="", help="Label prefix for output files")
    parser.add_argument("--continuous", action="store_true", help="Collect continuously")
    parser.add_argument("--interval", type=float, default=5.0, help="Collection interval (seconds)")
    args = parser.parse_args()

    if args.continuous:
        print(f"[INFO] Continuous collection mode. Interval: {args.interval}s")
        print("[INFO] Press Ctrl+C to stop.")
        try:
            while True:
                collect_and_save(args.server_url, args.output_dir, args.label)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")
    else:
        collect_and_save(args.server_url, args.output_dir, args.label)


if __name__ == "__main__":
    main()
