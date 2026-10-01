from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .runner import RAW_FIELDS, percentile


SUMMARY_FIELDS = ["Precision", "Quant recipe", "Accuracy", "TTFT P50", "TTFT P95", "TTFV P50",
                  "Prefill TPS", "Decode TPS", "TPOT", "ITL P95", "CCU2 TPS/User",
                  "Aggregate TPS", "Peak VRAM", "KV Cache"]
DETAIL_FIELDS = [
    "Precision", "Quant recipe", "Scenario", "CCU", "Runs", "Output tokens mean",
    "TTFT mean", "TTFT P50", "TTFT P95", "TTFT P99", "TTFV mean", "TTFV P50", "TTFV P95", "TTFV P99",
    "E2E mean", "E2E P50", "E2E P95", "E2E P99", "Prefill TPS mean", "Prefill TPS median",
    "Prefill TPS std", "Decode TPS mean", "Decode TPS median", "Decode TPS std", "TPOT",
    "ITL mean", "ITL P50", "ITL P95", "ITL P99", "Idle VRAM", "Peak VRAM", "KV Cache",
    "GPU util mean", "GPU util peak",
]
COMPARISON_FIELDS = ["metric", "llama.cpp", "TensorRT Edge", "comparison_type", "notes"]
ACCURACY_FIELDS = ["dataset", "sample_id", "backend", "precision", "prediction", "ground_truth", "metric", "score"]
GPU_FIELDS = ["timestamp", "gpu", "memory_used_mb", "gpu_utilization", "memory_utilization", "power_w", "temperature"]
MANIFEST_FIELDS = ["requested_precision", "actual_precision", "quant_recipe", "model_revision", "model_hash",
                   "engine_hash", "visual_encoder_dtype", "language_model_weight_dtype",
                   "language_model_activation_dtype", "kv_cache_dtype", "gpu_arch", "TensorRT_version",
                   "TensorRT_Edge_LLM_version", "ModelOpt_version", "build_method", "build_command",
                   "build_timestamp", "engine_build_time_sec", "engine_size_mb", "status", "error"]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key]) if row.get(key) not in (None, "", "None") else None
    except ValueError:
        return None


def ensure_csv(path: Path, fields: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("success", "")).lower() in {"true", "1"}:
            grouped[(row.get("requested_precision", ""), row.get("quant_recipe", ""))].append(row)
    output = []
    for (precision, recipe), group in grouped.items():
        # Only use CCU1 (scenario S2 if possible, or any CCU1) for baseline metrics
        ccu1 = [row for row in group if row.get("ccu") == "1" and row.get("scenario") == "S2"]
        if not ccu1:
            ccu1 = [row for row in group if row.get("ccu") == "1"]
        
        metric = lambda name: [value for row in ccu1 if (value := number(row, name)) is not None]
        all_metric = lambda name: [value for row in group if (value := number(row, name)) is not None]
        
        ccu2 = [row for row in group if row.get("ccu") == "2"]
        decode2 = [value for row in ccu2 if (value := number(row, "decode_tps")) is not None]
        output.append({
            "Precision": precision, "Quant recipe": recipe, "Accuracy": None,
            "TTFT P50": percentile(metric("ttft_ms"), .5), "TTFT P95": percentile(metric("ttft_ms"), .95),
            "TTFV P50": percentile(metric("ttfv_ms"), .5),
            "Prefill TPS": sum(metric("prefill_tps")) / len(metric("prefill_tps")) if metric("prefill_tps") else None,
            "Decode TPS": sum(metric("decode_tps")) / len(metric("decode_tps")) if metric("decode_tps") else None,
            "TPOT": sum(metric("tpot_ms")) / len(metric("tpot_ms")) if metric("tpot_ms") else None,
            "ITL P95": percentile(metric("itl_p95_ms"), .95),
            "CCU2 TPS/User": sum(decode2) / len(decode2) if decode2 else None,
            "Aggregate TPS": sum(decode2) / max(len(ccu2) / 2, 1) if decode2 else None,
            "Peak VRAM": max(all_metric("peak_vram_mb"), default=None), "KV Cache": max(all_metric("kv_cache_mb"), default=None),
        })
    return output


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    average = mean(values)
    assert average is not None
    return (sum((value - average) ** 2 for value in values) / len(values)) ** .5


def detailed_summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Match the per-scenario detail level of the llama.cpp benchmark report."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("success", "")).lower() in {"true", "1"}:
            grouped[(row.get("requested_precision", ""), row.get("quant_recipe", ""),
                     row.get("scenario", ""), row.get("ccu", ""))].append(row)
    output = []
    for (precision, recipe, scenario, ccu), group in sorted(grouped.items()):
        values = lambda name: [value for row in group if (value := number(row, name)) is not None]
        row: dict[str, Any] = {
            "Precision": precision, "Quant recipe": recipe, "Scenario": scenario, "CCU": ccu,
            "Runs": len(group), "Output tokens mean": mean(values("output_tokens")),
        }
        for source, label in (("ttft_ms", "TTFT"), ("ttfv_ms", "TTFV"), ("e2e_ms", "E2E")):
            metric = values(source)
            row.update({f"{label} mean": mean(metric), f"{label} P50": percentile(metric, .50),
                        f"{label} P95": percentile(metric, .95), f"{label} P99": percentile(metric, .99)})
        for source, label in (("prefill_tps", "Prefill TPS"), ("decode_tps", "Decode TPS")):
            metric = values(source)
            row.update({f"{label} mean": mean(metric), f"{label} median": percentile(metric, .50),
                        f"{label} std": stddev(metric)})
        for source, label in (("tpot_ms", "TPOT"), ("itl_mean_ms", "ITL mean"),
                              ("itl_p50_ms", "ITL P50"), ("itl_p95_ms", "ITL P95"),
                              ("itl_p99_ms", "ITL P99"), ("idle_vram_mb", "Idle VRAM"),
                              ("peak_vram_mb", "Peak VRAM"), ("kv_cache_mb", "KV Cache"),
                              ("gpu_util_mean", "GPU util mean"), ("gpu_util_peak", "GPU util peak")):
            metric = values(source)
            row[label] = mean(metric) if "util" in label.lower() else (
                max(metric, default=None) if "VRAM" in label or label == "KV Cache" else mean(metric)
            )
        output.append(row)
    return output


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> list[str]:
    def cell(value: Any) -> str:
        if value is None or value == "":
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(header)) for header in headers) + " |" for row in rows)
    return lines


def accuracy_summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        score = number(row, "score")
        if score is not None:
            grouped[(row.get("precision", ""), row.get("dataset", ""))].append(score)
    return [{"Precision": precision, "Dataset": dataset, "Samples": len(scores), "Score mean": mean(scores)}
            for (precision, dataset), scores in sorted(grouped.items())]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--results-dir", default="", help="TensorRT result root; defaults to results/tensorrt_edge")
    args = parser.parse_args()
    root = Path(args.project_dir).resolve()
    trt = Path(args.results_dir).resolve() if args.results_dir else root / "results/tensorrt_edge"
    comparison = trt / "comparison"
    charts = comparison / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    for path, fields in ((trt / "raw_requests.csv", RAW_FIELDS), (trt / "gpu_metrics.csv", GPU_FIELDS),
                         (trt / "accuracy.csv", ACCURACY_FIELDS), (trt / "engine_manifest.csv", MANIFEST_FIELDS)):
        ensure_csv(path, fields)
    summaries = summarize(read_rows(trt / "raw_requests.csv"))
    details = detailed_summaries(read_rows(trt / "raw_requests.csv"))
    with (trt / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader(); writer.writerows(summaries)
    with (trt / "detailed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader(); writer.writerows(details)
    ensure_csv(comparison / "backend_comparison.csv", COMPARISON_FIELDS)
    has_data = bool(summaries)
    accuracy = accuracy_summaries(read_rows(trt / "accuracy.csv"))
    manifests = read_rows(trt / "engine_manifest.csv")
    gpu_samples = read_rows(trt / "gpu_metrics.csv")
    gpu_metrics = [{"Peak VRAM (MB)": max((number(row, "memory_used_mb") or 0 for row in gpu_samples), default=None),
                    "Peak utilization (%)": max((number(row, "gpu_utilization") or 0 for row in gpu_samples), default=None),
                    "Peak power (W)": max((number(row, "power_w") or 0 for row in gpu_samples), default=None),
                    "Peak temperature (C)": max((number(row, "temperature") or 0 for row in gpu_samples), default=None)}] if gpu_samples else []
    report = ["# TensorRT-Edge-LLM Benchmark Report", "", "## Status", "",
              "Measured results are available." if has_data else "No TensorRT-Edge measurements are available; no performance claims can be made.", "",
              "See `environment.txt` and `compatibility_matrix.csv` for detected blockers and support evidence.", "",
              "## Detailed performance by precision, scenario and concurrency", ""]
    report.extend(markdown_table(DETAIL_FIELDS, details) if details else ["No successful requests were recorded."])
    report.extend(["", "## GPU telemetry", ""])
    report.extend(markdown_table(list(gpu_metrics[0]), gpu_metrics) if gpu_metrics else ["No GPU samples were recorded."])
    report.extend(["", "## Accuracy", ""])
    report.extend(markdown_table(["Precision", "Dataset", "Samples", "Score mean"], accuracy) if accuracy else ["No accuracy samples were recorded."])
    report.extend(["", "## Engine and quantization manifest", ""])
    manifest_headers = ["requested_precision", "actual_precision", "quant_recipe", "engine_build_time_sec", "engine_size_mb", "status", "error"]
    report.extend(markdown_table(manifest_headers, manifests) if manifests else ["No engine manifest was recorded."])
    report.extend(["", "## Measurement policy", "", "Visual encoder and KV cache are fixed to FP16 for the main benchmark. Missing runtime metrics remain null; stream chunks are not represented as token-level ITL unless the server emits one token per event."])
    (trt / "benchmark_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (comparison / "backend_comparison.md").write_text(
        "# Backend Comparison\n\n" + ("See backend_comparison.csv.\n" if has_data else
        "No direct comparison was generated because equivalent measured TensorRT-Edge data is unavailable.\n"), encoding="utf-8")
    print(f"generated report artifacts under {trt} and {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
