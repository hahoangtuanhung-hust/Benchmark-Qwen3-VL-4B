#!/usr/bin/env python3
"""Download deterministic subsets of the public accuracy datasets."""

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

from datasets import load_dataset


DATASET_SPECS = {
    "textvqa": {
        "repo": "lmms-lab-encoder/textvqa",
        "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "config": None,
        "split": "validation",
        "question_id": "question_id",
    },
    "docvqa": {
        "repo": "lmms-lab-encoder/DocVQA",
        "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "config": "DocVQA",
        "split": "validation",
        "question_id": "questionId",
    },
    "chartvqa": {
        "repo": "lmms-lab-encoder/ChartQA",
        "revision": "9e63b7df1592a1c2158e735cc1725454aef0d6d9",
        "config": None,
        "split": "test",
        "question_id": None,
    },
}


def _answers_for_sample(sample: dict) -> list[str]:
    answers = sample.get("answers", sample.get("answer", ""))
    if isinstance(answers, str):
        answers = [answers]
    return [str(answer).strip() for answer in (answers or []) if str(answer).strip()]


def _representative_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def _cached_dataset_is_complete(dataset_dir: Path, sample_count: int) -> bool:
    manifest = dataset_dir / "questions.jsonl"
    if not manifest.exists():
        return False

    try:
        rows = []
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= sample_count:
                    break
        if len(rows) < sample_count:
            return False
        project_dir = dataset_dir.parents[2]
        return all((project_dir / row["image_path"]).is_file() for row in rows)
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def download_dataset(project_dir: Path, dataset_key: str, sample_count: int, force: bool) -> None:
    spec = DATASET_SPECS[dataset_key]
    dataset_dir = project_dir / "benchmark_data" / "accuracy" / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not force and _cached_dataset_is_complete(dataset_dir, sample_count):
        print(f"[INFO] {dataset_key}: {sample_count} cached samples are ready.")
        return

    print(
        f"[INFO] Downloading {sample_count} {dataset_key} samples "
        f"from {spec['repo']} ({spec['split']})..."
    )
    kwargs = {"split": spec["split"], "streaming": True, "revision": spec["revision"]}
    if spec["config"]:
        kwargs["name"] = spec["config"]
    dataset = load_dataset(spec["repo"], **kwargs)

    questions = []
    for index, sample in enumerate(dataset):
        if index >= sample_count:
            break

        image = sample.get("image")
        question = str(sample.get("question", "")).strip()
        answers = _answers_for_sample(sample)
        if image is None or not question or not answers:
            raise ValueError(f"{dataset_key} sample {index} is missing image/question/answer")

        image_name = f"{dataset_key}_{index:04d}.jpg"
        image_path = dataset_dir / image_name
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path, format="JPEG", quality=95)

        id_field = spec["question_id"]
        sample_id = str(sample.get(id_field, f"{dataset_key}_{index:04d}")) if id_field else f"{dataset_key}_{index:04d}"
        questions.append(
            {
                "sample_id": sample_id,
                "image_path": f"benchmark_data/accuracy/{dataset_key}/{image_name}",
                "question": question,
                "answer": _representative_answer(answers),
                "answers": answers,
                "source_repo": spec["repo"],
                "source_revision": spec["revision"],
                "source_split": spec["split"],
            }
        )
        print(f"  [{index + 1}/{sample_count}] {image_name}")

    if len(questions) != sample_count:
        raise RuntimeError(f"{dataset_key}: expected {sample_count} samples, downloaded {len(questions)}")

    manifest = dataset_dir / "questions.jsonl"
    temporary_manifest = manifest.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question, ensure_ascii=True) + "\n")
    os.replace(temporary_manifest, manifest)
    print(f"[OK] {dataset_key}: saved {len(questions)} samples to {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TextVQA, DocVQA, and ChartQA subsets")
    parser.add_argument("--project-dir", default=".", help="Project root")
    parser.add_argument("--samples", type=int, default=50, help="Samples per dataset")
    parser.add_argument("--force", action="store_true", help="Re-download existing samples")
    args = parser.parse_args()

    if args.samples <= 0:
        parser.error("--samples must be positive")

    project_dir = Path(args.project_dir).resolve()
    failures = []
    for dataset_key in DATASET_SPECS:
        try:
            download_dataset(project_dir, dataset_key, args.samples, args.force)
        except Exception as exc:
            failures.append(f"{dataset_key}: {exc}")
            print(f"[ERROR] {dataset_key}: {exc}", file=sys.stderr)

    if failures:
        print("[ERROR] Accuracy dataset preparation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("[SUCCESS] Accuracy datasets are ready.")


def _run_and_exit_without_native_finalizers() -> None:
    """Run as a worker process without invoking unstable extension finalizers."""
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1

    # Some datasets/pyarrow builds abort while CPython tears down native worker
    # threads. All dataset files are closed by this point, so flush console I/O
    # and let the operating system terminate this standalone worker process.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(exit_code)


if __name__ == "__main__":
    _run_and_exit_without_native_finalizers()
