#!/usr/bin/env python3
"""
Validate Benchmark Results — Check validity of raw benchmark data.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §27, §28

Validates that benchmark runs meet quality criteria:
- HTTP success
- No OOM
- output_tokens > 0
- Consistent hardware/config
- No cache reuse in cold benchmark
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import Optional

import yaml


# ─── Validity Checks (§27) ───

VALIDITY_CHECKS = [
    "http_success",
    "no_oom",
    "output_tokens_positive",
    "response_not_empty",
    "no_error",
    "consistent_hardware",
    "consistent_config",
]

INVALID_FLAGS = [
    "server_startup",
    "warmup",
    "oom",
    "http_error",
    "timeout",
    "corrupt_response",
]


def validate_request(row: dict) -> dict:
    """Validate a single benchmark request row."""
    issues = []
    flags = []

    # HTTP success
    http_status = int(row.get("http_status", 0))
    if http_status != 200 and http_status != 0:
        issues.append(f"http_status={http_status}")
        flags.append("http_error")

    # Success flag
    success = str(row.get("success", "")).lower()
    if success not in ("true", "1", "yes"):
        issues.append("success=false")

    # Error type
    error_type = row.get("error_type", "")
    if error_type:
        issues.append(f"error_type={error_type}")
        if "oom" in error_type.lower():
            flags.append("oom")
        elif "timeout" in error_type.lower():
            flags.append("timeout")

    # Output tokens > 0
    output_tokens = int(row.get("output_tokens", 0))
    if output_tokens <= 0:
        issues.append("output_tokens<=0")

    # Response not empty
    answer = row.get("answer_text", "")
    if not answer or not answer.strip():
        issues.append("empty_response")
        flags.append("corrupt_response")

    # TTFT should be present
    ttft = row.get("ttft_ms", "")
    if not ttft or ttft == "None":
        issues.append("ttft_missing")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "flags": flags,
    }


def validate_consistency(rows: list[dict]) -> dict:
    """Check consistency across all rows in a suite."""
    consistency_issues = []

    # Check hardware consistency
    gpu_names = set(r.get("gpu_name", "") for r in rows if r.get("gpu_name"))
    if len(gpu_names) > 1:
        consistency_issues.append(f"mixed_gpus: {gpu_names}")

    # Check config consistency
    contexts = set(r.get("context_per_slot", "") for r in rows if r.get("context_per_slot"))
    if len(contexts) > 1:
        consistency_issues.append(f"mixed_context: {contexts}")

    kv_types = set(r.get("kv_k_type", "") for r in rows if r.get("kv_k_type"))
    if len(kv_types) > 1:
        consistency_issues.append(f"mixed_kv_type: {kv_types}")

    # Check cache policy
    cache_values = set(str(r.get("cache_prompt", "")).lower() for r in rows)
    if "true" in cache_values:
        consistency_issues.append("cache_prompt=true in cold benchmark")

    return {
        "consistent": len(consistency_issues) == 0,
        "issues": consistency_issues,
    }


def validate_results_file(filepath: str) -> dict:
    """Validate an entire results CSV file."""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}", "valid_count": 0, "total_count": 0}

    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"error": "Empty file", "valid_count": 0, "total_count": 0}

    # Validate each row
    valid_count = 0
    invalid_count = 0
    flag_counts = Counter()
    issue_counts = Counter()

    for i, row in enumerate(rows):
        validation = validate_request(row)
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
        for flag in validation["flags"]:
            flag_counts[flag] += 1
        for issue in validation["issues"]:
            issue_counts[issue] += 1

    # Consistency check
    consistency = validate_consistency(rows)

    # Failure rate check (§32)
    total = len(rows)
    failure_rate = invalid_count / total if total > 0 else 0

    return {
        "filepath": filepath,
        "total_count": total,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "failure_rate": failure_rate,
        "failure_rate_acceptable": failure_rate <= 0.05,  # §32: max 5%
        "flag_counts": dict(flag_counts),
        "top_issues": dict(issue_counts.most_common(10)),
        "consistency": consistency,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark results")
    parser.add_argument("--project-dir", default=".", help="Project root")
    parser.add_argument("--files", nargs="+", default=None, help="Specific CSV files to validate")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)

    # Find result files
    if args.files:
        files = [os.path.join(project_dir, f) for f in args.files]
    else:
        raw_dir = os.path.join(project_dir, "results/raw")
        if os.path.exists(raw_dir):
            files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".csv")]
        else:
            files = []

    if not files:
        print("[WARN] No result files found to validate.")
        print(f"  Looked in: {os.path.join(project_dir, 'results/raw/')}")
        return

    print("=" * 60)
    print(" BENCHMARK RESULTS VALIDATION")
    print("=" * 60)

    all_valid = True
    validation_report = []

    for filepath in files:
        print(f"\n--- {os.path.basename(filepath)} ---")
        result = validate_results_file(filepath)

        if "error" in result:
            print(f"  [ERROR] {result['error']}")
            all_valid = False
            continue

        print(f"  Total:   {result['total_count']}")
        print(f"  Valid:   {result['valid_count']}")
        print(f"  Invalid: {result['invalid_count']}")
        print(f"  Failure rate: {result['failure_rate']:.1%} {'✓' if result['failure_rate_acceptable'] else '✗ EXCEEDS 5%'}")

        if result["flag_counts"]:
            print(f"  Flags: {result['flag_counts']}")

        if result["top_issues"]:
            print(f"  Top issues: {result['top_issues']}")

        if not result["consistency"]["consistent"]:
            print(f"  [WARN] Consistency issues: {result['consistency']['issues']}")

        if not result["failure_rate_acceptable"]:
            all_valid = False

        validation_report.append(result)

    # Save validation report
    report_path = os.path.join(project_dir, "results/validation_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2, default=str)

    print(f"\n[INFO] Validation report: {report_path}")

    if all_valid:
        print("\n[SUCCESS] All results pass validation.")
    else:
        print("\n[WARNING] Some results have validation issues. Review before reporting.")

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
