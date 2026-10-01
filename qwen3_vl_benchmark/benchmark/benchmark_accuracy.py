"""Accuracy uses the existing shared evaluator in scripts/run_accuracy.py.

Run the TensorRT-Edge server through its OpenAI-compatible adapter, then invoke
the existing evaluator with the same frozen prompts and datasets. This module
exists as the stable package entry point without duplicating evaluator logic.
"""
from runpy import run_path
from pathlib import Path

if __name__ == "__main__":
    run_path(str(Path(__file__).parents[2] / "scripts" / "run_accuracy.py"), run_name="__main__")

