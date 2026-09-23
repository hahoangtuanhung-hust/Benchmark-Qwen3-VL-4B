#!/usr/bin/env python3
"""
Summarize Benchmark Results — Generate statistics, tables, and charts.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §25, §26, §35, §36, §37

Generates:
- Summary CSV with P50/P95/P99/mean/std statistics
- Derived metrics (speedup, VRAM saving, quality loss, CCU2 slowdown)
- benchmark_summary.md report
- Charts (10 required visualizations)
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all raw result CSVs into a single DataFrame."""
    dfs = []
    raw_dir = os.path.join(results_dir, "raw")
    if not os.path.exists(raw_dir):
        return pd.DataFrame()

    for f in os.listdir(raw_dir):
        if f.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(raw_dir, f))
                dfs.append(df)
            except Exception as e:
                print(f"  [WARN] Could not load {f}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def compute_statistics(df: pd.DataFrame) -> list[dict]:
    """
    Compute per-variant, per-scenario statistics (§25).
    Returns list of summary rows.
    """
    if df.empty:
        return []

    summaries = []
    # Group by variant and scenario
    group_cols = ["model_name", "requested_precision", "actual_quant_type", "scenario", "ccu"]
    available_cols = [c for c in group_cols if c in df.columns]

    if not available_cols:
        return []

    # Filter to successful runs
    success_df = df[df["success"] == True].copy()  # noqa: E712

    for name, group in success_df.groupby(available_cols):
        if isinstance(name, str):
            name = (name,)

        summary = dict(zip(available_cols, name))
        summary["runs"] = len(group)

        # Latency statistics (§25)
        for metric in ["ttft_ms", "ttfv_ms", "e2e_ms"]:
            values = group[metric].dropna()
            if len(values) > 0:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values))
                summary[f"{metric}_p50"] = float(np.percentile(values, 50))
                summary[f"{metric}_p95"] = float(np.percentile(values, 95))
                summary[f"{metric}_p99"] = float(np.percentile(values, 99))

        # ITL statistics
        for metric in ["itl_mean_ms", "itl_p50_ms", "itl_p95_ms", "itl_p99_ms"]:
            values = group[metric].dropna()
            if len(values) > 0:
                summary[f"{metric}_avg"] = float(np.mean(values))

        # TPS statistics (§25)
        for metric in ["decode_tps", "prefill_tps"]:
            values = group[metric].dropna()
            if len(values) > 0:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_median"] = float(np.median(values))
                summary[f"{metric}_std"] = float(np.std(values))

        # VRAM statistics (§25)
        for metric in ["idle_vram_mb", "peak_vram_mb"]:
            values = group[metric].dropna()
            if len(values) > 0:
                summary[f"{metric}_max"] = float(np.max(values))
                summary[f"{metric}_mean"] = float(np.mean(values))

        # GPU utilization
        for metric in ["gpu_util_mean_pct", "gpu_util_peak_pct"]:
            values = group[metric].dropna()
            if len(values) > 0:
                summary[f"{metric}_avg"] = float(np.mean(values))

        # Output tokens
        values = group["output_tokens"].dropna()
        if len(values) > 0:
            summary["output_tokens_mean"] = float(np.mean(values))

        summaries.append(summary)

    return summaries


def compute_derived_metrics(summaries: list[dict]) -> list[dict]:
    """
    Compute derived metrics (§26).
    - Speedup vs FP16
    - VRAM saving
    - Quality loss
    - CCU2 slowdown
    """
    # Find FP16 baseline for each scenario
    fp16_baselines = {}
    for s in summaries:
        quant = s.get("actual_quant_type", "") or s.get("requested_precision", "")
        scenario = s.get("scenario", "")
        ccu = s.get("ccu", 1)
        if quant in ("F16", "FP16"):
            fp16_baselines[(scenario, ccu)] = s

    for s in summaries:
        scenario = s.get("scenario", "")
        ccu = s.get("ccu", 1)
        baseline = fp16_baselines.get((scenario, ccu))

        if baseline:
            # Decode speedup (§26)
            if s.get("decode_tps_mean") and baseline.get("decode_tps_mean"):
                s["decode_speedup"] = s["decode_tps_mean"] / baseline["decode_tps_mean"]

            # Prefill speedup (§26)
            if s.get("prefill_tps_mean") and baseline.get("prefill_tps_mean"):
                s["prefill_speedup"] = s["prefill_tps_mean"] / baseline["prefill_tps_mean"]

            # VRAM saving (§26)
            if s.get("peak_vram_mb_max") and baseline.get("peak_vram_mb_max"):
                s["vram_saving_pct"] = (1.0 - s["peak_vram_mb_max"] / baseline["peak_vram_mb_max"]) * 100

    # CCU2 slowdown (§26)
    # Find CCU1 counterparts
    ccu1_data = {}
    for s in summaries:
        if s.get("ccu") == 1:
            key = s.get("actual_quant_type", "") or s.get("requested_precision", "")
            ccu1_data[key] = s

    for s in summaries:
        if s.get("ccu") == 2:
            key = s.get("actual_quant_type", "") or s.get("requested_precision", "")
            ccu1 = ccu1_data.get(key)
            if ccu1:
                # TTFT slowdown
                ccu2_ttft = s.get("ttft_ms_p50")
                ccu1_ttft = ccu1.get("ttft_ms_p50")
                if ccu2_ttft and ccu1_ttft and ccu1_ttft > 0:
                    s["ccu2_ttft_slowdown"] = ccu2_ttft / ccu1_ttft

                # TPS loss per user
                ccu2_tps = s.get("decode_tps_mean")
                ccu1_tps = ccu1.get("decode_tps_mean")
                if ccu2_tps and ccu1_tps and ccu1_tps > 0:
                    s["ccu2_tps_loss_pct"] = (1.0 - ccu2_tps / ccu1_tps) * 100
                    s["ccu2_aggregate_tps"] = ccu2_tps * 2  # Approximate

    return summaries


def generate_charts(summaries: list[dict], output_dir: str):
    """Generate benchmark visualization charts (§36)."""
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not installed. Skipping charts.")
        return

    chart_dir = os.path.join(output_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    # Filter to main scenario (S2) CCU1 for comparison charts
    s2_ccu1 = [s for s in summaries if s.get("scenario") == "S2" and s.get("ccu") == 1]

    if not s2_ccu1:
        # Try any available data
        s2_ccu1 = [s for s in summaries if s.get("ccu") == 1]

    if not s2_ccu1:
        print("[WARN] No CCU1 data for charts.")
        return

    labels = [s.get("actual_quant_type", s.get("requested_precision", "?")) for s in s2_ccu1]

    # Chart style
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

    # 1. Quantization vs Decode TPS
    values = [s.get("decode_tps_mean", 0) for s in s2_ccu1]
    if any(v > 0 for v in values):
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=colors)
        ax.set_xlabel("Quantization")
        ax.set_ylabel("Decode TPS (tokens/s)")
        ax.set_title("Quantization vs Decode TPS (S2, CCU1)")
        ax.bar_label(bars, fmt="%.1f")
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, "01_quant_vs_decode_tps.png"), dpi=150)
        plt.close()

    # 2. Quantization vs Prefill TPS
    values = [s.get("prefill_tps_mean", 0) for s in s2_ccu1]
    if any(v > 0 for v in values):
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=colors)
        ax.set_xlabel("Quantization")
        ax.set_ylabel("Prefill TPS (tokens/s)")
        ax.set_title("Quantization vs Prefill TPS (S2, CCU1)")
        ax.bar_label(bars, fmt="%.1f")
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, "02_quant_vs_prefill_tps.png"), dpi=150)
        plt.close()

    # 3. Quantization vs TTFT P50/P95
    p50 = [s.get("ttft_ms_p50", 0) for s in s2_ccu1]
    p95 = [s.get("ttft_ms_p95", 0) for s in s2_ccu1]
    if any(v > 0 for v in p50):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, p50, width, label="P50", color="#2ecc71")
        ax.bar(x + width / 2, p95, width, label="P95", color="#e74c3c")
        ax.set_xlabel("Quantization")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("Quantization vs TTFT P50/P95 (S2, CCU1)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, "03_quant_vs_ttft.png"), dpi=150)
        plt.close()

    # 4. Quantization vs ITL P50/P95
    p50 = [s.get("itl_p50_ms_avg", 0) for s in s2_ccu1]
    p95 = [s.get("itl_p95_ms_avg", 0) for s in s2_ccu1]
    if any(v > 0 for v in p50):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, p50, width, label="P50", color="#3498db")
        ax.bar(x + width / 2, p95, width, label="P95", color="#9b59b6")
        ax.set_xlabel("Quantization")
        ax.set_ylabel("ITL (ms)")
        ax.set_title("Quantization vs ITL P50/P95 (S2, CCU1)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, "04_quant_vs_itl.png"), dpi=150)
        plt.close()

    # 5. Quantization vs Peak VRAM
    values = [s.get("peak_vram_mb_max", 0) for s in s2_ccu1]
    if any(v > 0 for v in values):
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=colors)
        ax.set_xlabel("Quantization")
        ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title("Quantization vs Peak VRAM (S2, CCU1)")
        ax.bar_label(bars, fmt="%.0f")
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, "05_quant_vs_peak_vram.png"), dpi=150)
        plt.close()

    # 7. CCU1 vs CCU2 TPS/user
    s2_ccu2 = [s for s in summaries if s.get("scenario") == "S4" and s.get("ccu") == 2]
    if s2_ccu1 and s2_ccu2:
        ccu1_labels = [s.get("actual_quant_type", "?") for s in s2_ccu1]
        ccu2_by_quant = {s.get("actual_quant_type", "?"): s for s in s2_ccu2}

        matched_labels = []
        ccu1_tps = []
        ccu2_tps = []
        for s in s2_ccu1:
            q = s.get("actual_quant_type", "?")
            if q in ccu2_by_quant:
                matched_labels.append(q)
                ccu1_tps.append(s.get("decode_tps_mean", 0))
                ccu2_tps.append(ccu2_by_quant[q].get("decode_tps_mean", 0))

        if matched_labels:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(matched_labels))
            width = 0.35
            ax.bar(x - width / 2, ccu1_tps, width, label="CCU1", color="#2ecc71")
            ax.bar(x + width / 2, ccu2_tps, width, label="CCU2/user", color="#e74c3c")
            ax.set_xlabel("Quantization")
            ax.set_ylabel("Decode TPS")
            ax.set_title("CCU1 vs CCU2 TPS per User")
            ax.set_xticks(x)
            ax.set_xticklabels(matched_labels)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, "07_ccu1_vs_ccu2_tps.png"), dpi=150)
            plt.close()

    # 8. CCU1 vs CCU2 TTFT
    if s2_ccu1 and s2_ccu2:
        matched_labels = []
        ccu1_ttft = []
        ccu2_ttft = []
        for s in s2_ccu1:
            q = s.get("actual_quant_type", "?")
            if q in ccu2_by_quant:
                matched_labels.append(q)
                ccu1_ttft.append(s.get("ttft_ms_p50", 0))
                ccu2_ttft.append(ccu2_by_quant[q].get("ttft_ms_p50", 0))

        if matched_labels:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(matched_labels))
            width = 0.35
            ax.bar(x - width / 2, ccu1_ttft, width, label="CCU1 P50", color="#3498db")
            ax.bar(x + width / 2, ccu2_ttft, width, label="CCU2 P50", color="#e67e22")
            ax.set_xlabel("Quantization")
            ax.set_ylabel("TTFT P50 (ms)")
            ax.set_title("CCU1 vs CCU2 TTFT P50")
            ax.set_xticks(x)
            ax.set_xticklabels(matched_labels)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, "08_ccu1_vs_ccu2_ttft.png"), dpi=150)
            plt.close()

    print(f"[INFO] Charts saved to: {chart_dir}/")


def generate_markdown_report(summaries: list[dict], output_dir: str, project_dir: str):
    """Generate benchmark_summary.md report (§35)."""
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "benchmark_summary.md")

    # Load compatibility matrix
    compat_path = os.path.join(project_dir, "results/compatibility_matrix.csv")
    compat_rows = []
    if os.path.exists(compat_path):
        with open(compat_path) as f:
            compat_rows = list(csv.DictReader(f))

    lines = []
    lines.append("# Qwen3-VL-4B-Instruct Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Backend:** llama.cpp")
    lines.append(f"**Model:** Qwen3-VL-4B-Instruct")
    lines.append("")

    # Compatibility table (§35)
    lines.append("## Compatibility Matrix")
    lines.append("")
    if compat_rows:
        lines.append("| Requested | Support Type | Actual GGUF | Notes |")
        lines.append("|-----------|-------------|-------------|-------|")
        for row in compat_rows:
            lines.append(f"| {row.get('requested_label', '')} | {row.get('support_type', '')} | {row.get('actual_gguf_type', '')} | {row.get('notes', '')[:60]} |")
    else:
        lines.append("*No compatibility matrix found.*")
    lines.append("")

    # Performance table — CCU1 S2 (§35)
    s2_ccu1 = [s for s in summaries if s.get("scenario") == "S2" and s.get("ccu") == 1]
    if s2_ccu1:
        lines.append("## Performance — CCU1 Normal (S2)")
        lines.append("")
        lines.append("| Variant | TTFT P50 | TTFT P95 | TTFV P50 | E2E P50 | Prefill TPS | Decode TPS | ITL P50 | ITL P95 | Peak VRAM |")
        lines.append("|---------|----------|----------|----------|---------|-------------|------------|---------|---------|-----------|")
        for s in s2_ccu1:
            quant = s.get("actual_quant_type", "?")
            lines.append(
                f"| {quant} "
                f"| {s.get('ttft_ms_p50', 'N/A'):.0f}ms " if isinstance(s.get('ttft_ms_p50'), (int, float)) else f"| N/A "
                f"| {s.get('ttft_ms_p95', 'N/A'):.0f}ms " if isinstance(s.get('ttft_ms_p95'), (int, float)) else f"| N/A "
                f"| {s.get('ttfv_ms_p50', 'N/A'):.0f}ms " if isinstance(s.get('ttfv_ms_p50'), (int, float)) else f"| N/A "
                f"| {s.get('e2e_ms_p50', 'N/A'):.0f}ms " if isinstance(s.get('e2e_ms_p50'), (int, float)) else f"| N/A "
                f"| {s.get('prefill_tps_mean', 'N/A'):.1f} " if isinstance(s.get('prefill_tps_mean'), (int, float)) else f"| N/A "
                f"| {s.get('decode_tps_mean', 'N/A'):.1f} " if isinstance(s.get('decode_tps_mean'), (int, float)) else f"| N/A "
                f"| {s.get('itl_p50_ms_avg', 'N/A'):.1f}ms " if isinstance(s.get('itl_p50_ms_avg'), (int, float)) else f"| N/A "
                f"| {s.get('itl_p95_ms_avg', 'N/A'):.1f}ms " if isinstance(s.get('itl_p95_ms_avg'), (int, float)) else f"| N/A "
                f"| {s.get('peak_vram_mb_max', 'N/A'):.0f}MB |" if isinstance(s.get('peak_vram_mb_max'), (int, float)) else f"| N/A |"
            )
        lines.append("")

    # CCU comparison table (§35)
    s4_ccu2 = [s for s in summaries if s.get("scenario") == "S4" and s.get("ccu") == 2]
    if s2_ccu1 and s4_ccu2:
        lines.append("## CCU1 vs CCU2 Comparison")
        lines.append("")
        lines.append("| Variant | CCU1 TPS | CCU2 TPS/user | CCU2 Agg TPS | CCU1 TTFT P50 | CCU2 TTFT P50 | TTFT Slowdown | TPS Loss/user |")
        lines.append("|---------|----------|---------------|--------------|---------------|---------------|---------------|---------------|")
        ccu2_by_quant = {s.get("actual_quant_type", "?"): s for s in s4_ccu2}
        for s in s2_ccu1:
            q = s.get("actual_quant_type", "?")
            c2 = ccu2_by_quant.get(q, {})
            ccu1_tps = s.get("decode_tps_mean")
            ccu2_tps = c2.get("decode_tps_mean")
            slowdown = c2.get("ccu2_ttft_slowdown")
            tps_loss = c2.get("ccu2_tps_loss_pct")
            agg_tps = c2.get("ccu2_aggregate_tps")
            lines.append(
                f"| {q} "
                f"| {ccu1_tps:.1f} " if ccu1_tps else "| N/A "
                f"| {ccu2_tps:.1f} " if ccu2_tps else "| N/A "
                f"| {agg_tps:.1f} " if agg_tps else "| N/A "
                f"| {s.get('ttft_ms_p50', 'N/A'):.0f}ms " if isinstance(s.get('ttft_ms_p50'), (int, float)) else "| N/A "
                f"| {c2.get('ttft_ms_p50', 'N/A'):.0f}ms " if isinstance(c2.get('ttft_ms_p50'), (int, float)) else "| N/A "
                f"| {slowdown:.2f}x " if slowdown else "| N/A "
                f"| {tps_loss:.1f}% |" if tps_loss else "| N/A |"
            )
        lines.append("")

    # Derived metrics
    lines.append("## Derived Metrics")
    lines.append("")
    for s in s2_ccu1:
        q = s.get("actual_quant_type", "?")
        if s.get("decode_speedup") or s.get("vram_saving_pct"):
            speedup = f"{s['decode_speedup']:.2f}x" if s.get("decode_speedup") else "N/A"
            vram = f"{s['vram_saving_pct']:.1f}%" if s.get("vram_saving_pct") is not None else "N/A"
            lines.append(f"- **{q}**: decode speedup={speedup}, VRAM saving={vram}")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by Qwen3-VL-4B Benchmark Suite*")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--project-dir", default=".", help="Project root")
    parser.add_argument("--results-dir", default=None, help="Results directory")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    results_dir = args.results_dir or os.path.join(project_dir, "results")

    print("=" * 60)
    print(" BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    # Load results
    print("\n[INFO] Loading raw results...")
    df = load_results(results_dir)
    if df.empty:
        print("[WARN] No results found. Run benchmark first.")
        return

    print(f"[INFO] Loaded {len(df)} total rows")

    # Compute statistics
    print("[INFO] Computing statistics...")
    summaries = compute_statistics(df)
    print(f"[INFO] Generated {len(summaries)} summary entries")

    # Compute derived metrics
    print("[INFO] Computing derived metrics...")
    summaries = compute_derived_metrics(summaries)

    # Save summary CSV
    summary_csv_path = os.path.join(results_dir, "report", "summary.csv")
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    if summaries:
        all_keys = set()
        for s in summaries:
            all_keys.update(s.keys())
        fieldnames = sorted(all_keys)

        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"[INFO] Summary CSV: {summary_csv_path}")

    # Copy compatibility matrix to report
    compat_src = os.path.join(project_dir, "results/compatibility_matrix.csv")
    compat_dst = os.path.join(results_dir, "report", "compatibility_matrix.csv")
    if os.path.exists(compat_src):
        import shutil
        shutil.copy2(compat_src, compat_dst)

    # Generate charts
    print("[INFO] Generating charts...")
    generate_charts(summaries, results_dir)

    # Generate markdown report
    print("[INFO] Generating markdown report...")
    generate_markdown_report(summaries, results_dir, project_dir)

    # Print quick summary
    print("\n" + "=" * 60)
    print(" QUICK SUMMARY")
    print("=" * 60)
    s2_ccu1 = [s for s in summaries if s.get("scenario") == "S2" and s.get("ccu") == 1]
    for s in s2_ccu1:
        q = s.get("actual_quant_type", "?")
        tps = s.get("decode_tps_mean", "N/A")
        ttft = s.get("ttft_ms_p50", "N/A")
        vram = s.get("peak_vram_mb_max", "N/A")
        tps_str = f"{tps:.1f}" if isinstance(tps, (int, float)) else "N/A"
        ttft_str = f"{ttft:.0f}ms" if isinstance(ttft, (int, float)) else "N/A"
        vram_str = f"{vram:.0f}MB" if isinstance(vram, (int, float)) else "N/A"
        print(f"  {q:<12} Decode TPS={tps_str:<8} TTFT P50={ttft_str:<10} Peak VRAM={vram_str}")

    print(f"\n[SUCCESS] Summary complete. See results/report/")


if __name__ == "__main__":
    main()
