from __future__ import annotations

import modal


VOLUME_NAMES = {
    "models": "qwen3-vl-edge-models",
    "engines": "qwen3-vl-edge-engines",
    "dataset": "qwen3-vl-edge-dataset",
    "results": "qwen3-vl-edge-results",
    "cache": "qwen3-vl-edge-cache",
}

models = modal.Volume.from_name(VOLUME_NAMES["models"], create_if_missing=True)
engines = modal.Volume.from_name(VOLUME_NAMES["engines"], create_if_missing=True)
dataset = modal.Volume.from_name(VOLUME_NAMES["dataset"], create_if_missing=True)
results = modal.Volume.from_name(VOLUME_NAMES["results"], create_if_missing=True)
cache = modal.Volume.from_name(VOLUME_NAMES["cache"], create_if_missing=True)

MOUNTS = {
    "/models": models,
    "/engines": engines,
    "/dataset": dataset,
    "/results": results,
    "/volumes/cache": cache,
}
