#!/usr/bin/env python3
"""
Accuracy Benchmark Runner.
See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §9, §24

Runs accuracy evaluation using benchmark datasets (TextVQA, DocVQA, ChartQA).
Supports both quick (50 samples) and full (200+ samples) modes.
"""

import argparse
import csv
import json
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_client import send_vlm_request


# ─── Evaluation Metrics ───

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, etc."""
    text = text.strip().lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def contains_match(prediction: str, ground_truth: str) -> float:
    """Check if ground truth is contained in prediction."""
    return 1.0 if normalize_text(ground_truth) in normalize_text(prediction) else 0.0


def anls_score(prediction: str, ground_truth: str) -> float:
    """
    Average Normalized Levenshtein Similarity (ANLS).
    Used for DocVQA evaluation.
    """
    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)

    if not gt:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0

    # Levenshtein distance
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    edit_dist = dp[m][n]
    max_len = max(m, n)
    nl = edit_dist / max_len if max_len > 0 else 0

    # ANLS threshold
    threshold = 0.5
    if nl < threshold:
        return 1.0 - nl
    else:
        return 0.0


def relaxed_accuracy(prediction: str, ground_truth: str, tolerance: float = 0.05) -> float:
    """
    Relaxed accuracy for numerical answers (ChartQA).
    Allows tolerance for numerical values.
    """
    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)

    # Try exact match first
    if pred == gt:
        return 1.0

    # Try numerical comparison
    try:
        pred_num = float(re.sub(r"[^\d.\-]", "", pred))
        gt_num = float(re.sub(r"[^\d.\-]", "", gt))
        if gt_num == 0:
            return 1.0 if abs(pred_num) < tolerance else 0.0
        if abs(pred_num - gt_num) / abs(gt_num) <= tolerance:
            return 1.0
    except (ValueError, ZeroDivisionError):
        pass

    return 0.0


def vqa_accuracy(prediction: str, ground_truths: list[str]) -> float:
    """VQA consensus accuracy against the other annotator answers."""
    normalized_prediction = normalize_text(prediction)
    normalized_answers = [normalize_text(answer) for answer in ground_truths]
    if not normalized_answers:
        return 0.0

    per_annotator_scores = []
    for index in range(len(normalized_answers)):
        other_answers = normalized_answers[:index] + normalized_answers[index + 1:]
        matches = sum(answer == normalized_prediction for answer in other_answers)
        per_annotator_scores.append(min(1.0, matches / 3.0))
    return sum(per_annotator_scores) / len(per_annotator_scores)


METRIC_FUNCTIONS = {
    "exact_match_normalized": exact_match,
    "contains_match": contains_match,
    "anls": anls_score,
    "relaxed_accuracy": relaxed_accuracy,
}


def score_prediction(metric_name: str, prediction: str, ground_truths: list[str]) -> float:
    """Score against all accepted answers, using the dataset-specific metric."""
    if metric_name == "vqa_accuracy":
        return vqa_accuracy(prediction, ground_truths)

    metric_fn = METRIC_FUNCTIONS.get(metric_name, exact_match)
    return max((metric_fn(prediction, answer) for answer in ground_truths), default=0.0)


def load_accuracy_dataset(dataset_dir: str, max_samples: int = 50) -> list[dict]:
    """
    Load accuracy dataset. Expects either:
    - questions.jsonl: {"image_path": ..., "question": ..., "answer": ...}
    - questions.json: [{"image_path": ..., "question": ..., "answer": ...}, ...]
    """
    samples = []

    # Try JSONL first
    jsonl_path = os.path.join(dataset_dir, "questions.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                samples.append(json.loads(line.strip()))
                if len(samples) >= max_samples:
                    break
        return samples

    # Try JSON
    json_path = os.path.join(dataset_dir, "questions.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            return data[:max_samples]

    # Try CSV
    csv_path = os.path.join(dataset_dir, "questions.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(row)
                if len(samples) >= max_samples:
                    break
        return samples

    return samples


def run_accuracy_benchmark(
    server_url: str,
    dataset_name: str,
    dataset_dir: str,
    metric_name: str,
    max_samples: int,
    model_info: dict,
    project_dir: str,
    model_name: str = "qwen3-vl",
) -> tuple[list[dict], dict]:
    """Run accuracy benchmark on a single dataset."""
    predictions = []

    # Load dataset
    samples = load_accuracy_dataset(dataset_dir, max_samples)
    if not samples:
        print(f"  [WARN] No samples found in {dataset_dir}")
        print(f"  [INFO] Create {dataset_dir}/questions.jsonl with format:")
        print(f'         {{"image_path": "rel/path.jpg", "question": "...", "answer": "..."}}')
        return [], {
            "dataset": dataset_name,
            "requested_precision": model_info.get("requested_precision", ""),
            "actual_quant_type": model_info.get("actual_quant_type", ""),
            "samples": 0,
            "success_rate": 0.0,
            "metric_name": metric_name,
            "score": None,
            "delta_vs_fp16": None,
            "error": "no_samples",
        }

    print(f"\n  Running {dataset_name}: {len(samples)} samples, metric={metric_name}")

    scores = []
    for i, sample in enumerate(samples):
        image_path = sample.get("image_path", sample.get("image", ""))
        if not os.path.isabs(image_path):
            image_path = os.path.join(project_dir, image_path)

        question = sample.get("question", "")
        ground_truths = sample.get("answers") or [sample.get("answer", sample.get("ground_truth", ""))]
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        ground_truths = [str(answer) for answer in ground_truths if str(answer).strip()]
        ground_truth = sample.get("answer") or (ground_truths[0] if ground_truths else "")

        accuracy_prompt = (
            "Answer using only the shortest correct answer. Do not explain.\n"
            f"Question: {question}"
        )

        # Send request
        result = send_vlm_request(
            server_url=server_url,
            image_path=image_path,
            prompt=accuracy_prompt,
            max_tokens=64,
            temperature=0.0,
            model_name=model_name,
        )

        prediction = result.answer_text.strip() if result.success else ""
        normalized_pred = normalize_text(prediction)

        # Compute score
        if result.success and prediction:
            score = score_prediction(metric_name, prediction, ground_truths)
        else:
            score = 0.0

        scores.append(score)

        pred_entry = {
            "sample_id": sample.get("sample_id", f"{dataset_name}_{i:04d}"),
            "dataset": dataset_name,
            "requested_precision": model_info.get("requested_precision", ""),
            "actual_quant_type": model_info.get("actual_quant_type", ""),
            "question": question[:200],
            "ground_truth": json.dumps(ground_truths, ensure_ascii=True)[:1000],
            "prediction": prediction[:200],
            "normalized_prediction": normalized_pred[:200],
            "metric_name": metric_name,
            "score": score,
            "error": result.error_type if not result.success else "",
        }
        predictions.append(pred_entry)

        # Progress
        status = "OK" if score > 0.5 else "MISS"
        print(f"    [{i+1:3d}/{len(samples)}] {status} score={score:.2f} | GT='{ground_truth[:40]}' | Pred='{prediction[:40]}'")

    # Summary
    import numpy as np
    avg_score = np.mean(scores) if scores else 0.0
    success_rate = sum(1 for p in predictions if not p["error"]) / len(predictions) if predictions else 0

    summary = {
        "dataset": dataset_name,
        "requested_precision": model_info.get("requested_precision", ""),
        "actual_quant_type": model_info.get("actual_quant_type", ""),
        "samples": len(predictions),
        "success_rate": success_rate,
        "metric_name": metric_name,
        "score": avg_score,
        "delta_vs_fp16": None,  # Computed later in summarize.py
        "error": "",
    }

    print(f"  {dataset_name} Score: {avg_score:.4f} ({metric_name}), success_rate: {success_rate:.2%}")

    return predictions, summary


def main():
    parser = argparse.ArgumentParser(description="Accuracy Benchmark Runner")
    parser.add_argument("--config", default="configs/datasets.yaml", help="Datasets config")
    parser.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--requested-precision", default="")
    parser.add_argument("--actual-quant", default="")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override the configured sample count for every dataset",
    )
    parser.add_argument("--output-dir", default="results/accuracy")
    parser.add_argument("--model-name", default="qwen3-vl",
                        help="Model name for OpenAI-compatible API (must match --served-model-name)")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)

    # Load configs
    with open(os.path.join(project_dir, args.config)) as f:
        datasets_config = yaml.safe_load(f)

    with open(os.path.join(project_dir, args.benchmark_config)) as f:
        bench_config = yaml.safe_load(f)

    accuracy_config = datasets_config.get("accuracy", {})
    max_samples = accuracy_config.get("quick_samples", 50) if args.mode == "quick" else accuracy_config.get("full_samples", 200)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            parser.error("--max-samples must be positive")
        max_samples = args.max_samples

    model_info = {
        "requested_precision": args.requested_precision or args.model_label,
        "actual_quant_type": args.actual_quant or args.model_label,
    }

    print("=" * 60)
    print(f" ACCURACY BENCHMARK — {args.model_label}")
    print(f" Mode: {args.mode}, Max samples/dataset: {max_samples}")
    print("=" * 60)

    all_predictions = []
    all_summaries = []

    # Run each accuracy dataset
    for ds_key, ds_config in accuracy_config.get("datasets", {}).items():
        dataset_dir = os.path.join(project_dir, ds_config["directory"])

        # Determine metric
        metric_name = ds_config.get("fallback_metric", "exact_match_normalized")
        # Try official metric if evaluator exists
        official = ds_config.get("official_metric")
        if official and (official in METRIC_FUNCTIONS or official == "vqa_accuracy"):
            metric_name = official

        predictions, summary = run_accuracy_benchmark(
            server_url=args.server_url,
            dataset_name=ds_config["name"],
            dataset_dir=dataset_dir,
            metric_name=metric_name,
            max_samples=max_samples,
            model_info=model_info,
            project_dir=project_dir,
            model_name=args.model_name,
        )
        all_predictions.extend(predictions)
        all_summaries.append(summary)

    # Save results
    output_dir = os.path.join(project_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Predictions JSONL (replace this model when resuming Phase 9)
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    retained_predictions = []
    if os.path.exists(pred_path):
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                existing = json.loads(line)
                if existing.get("actual_quant_type") != model_info["actual_quant_type"]:
                    retained_predictions.append(existing)
    with open(pred_path, "w", encoding="utf-8") as f:
        for pred in retained_predictions:
            f.write(json.dumps(pred, ensure_ascii=True) + "\n")
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=True) + "\n")
    print(f"\n[INFO] Predictions saved to: {pred_path}")

    # Summary CSV (upsert by model and dataset so --phase 9 is repeatable)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_fields = [
        "dataset", "requested_precision", "actual_quant_type", "samples",
        "success_rate", "metric_name", "score", "delta_vs_fp16", "error",
    ]
    retained_summaries = []
    if os.path.exists(summary_path):
        with open(summary_path, newline="", encoding="utf-8") as f:
            for existing in csv.DictReader(f):
                if existing.get("actual_quant_type") != model_info["actual_quant_type"]:
                    retained_summaries.append(existing)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(retained_summaries + all_summaries)
    print(f"[INFO] Summary saved to: {summary_path}")

    print("\n[SUCCESS] Accuracy benchmark complete.")


if __name__ == "__main__":
    main()
