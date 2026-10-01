from __future__ import annotations

import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from app import PRECISIONS, UNSUPPORTED_PRECISIONS, pack_results_remote, preflight_remote, run_precision_remote
from app import app


@app.local_entrypoint()
def main(
    precision: str = "",
    suite: str = "",
    smoke: bool = False,
    preflight_only: bool = False,
    download: bool = True,
    download_only: str = "",
):
    # Download-only mode: just fetch results from a previous run
    if download_only:
        print(f"Downloading results for run: {download_only}")
        payload = pack_results_remote.remote(download_only)
        project = Path(__file__).resolve().parents[1]
        archive = project / "results_tensorrt_edge_modal.zip"
        archive.write_bytes(payload)
        destination = project / "results" / "modal_tensorrt_edge"
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
        print(f"Downloaded Modal results to {destination}")
        return

    failures: list[dict] = []
    run_id = ""
    if preflight_only:
        result = preflight_remote.remote()
        print(result)
        if result.get("status") != "PREFLIGHT_OK":
            failures.append(result)
    else:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        print(f"\n{'='*60}")
        print(f"  RUN ID: {run_id}")
        print(f"  Save this ID to download results later if using --detach")
        print(f"{'='*60}\n")
        requested = list(PRECISIONS) if suite.lower() == "all" else [precision.lower()]
        if not precision and suite.lower() != "all":
            raise ValueError("pass --precision fp16|fp8|int4|w4a16 (or w4a8 to record unsupported) or --suite all")
        for item in requested:
            if item not in (*PRECISIONS, *UNSUPPORTED_PRECISIONS):
                raise ValueError(f"unsupported precision label: {item}")
            print(f"\n>>> Starting {item.upper()} benchmark...")
            result = run_precision_remote.remote(item, smoke, run_id)
            print(f"<<< {item.upper()} result: {result}")
            if result.get("status") not in {"SMOKE_COMPLETE", "BENCHMARK_COMPLETE"}:
                failures.append(result)

    if download and run_id:
        print(f"\nDownloading results for run: {run_id}")
        payload = pack_results_remote.remote(run_id)
        project = Path(__file__).resolve().parents[1]
        archive = project / "results_tensorrt_edge_modal.zip"
        archive.write_bytes(payload)
        destination = project / "results" / "modal_tensorrt_edge"
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
        print(f"Downloaded Modal results to {destination}")

    if failures:
        labels = ", ".join(str(item.get("precision") or item.get("status")) for item in failures)
        raise SystemExit(f"Modal benchmark did not complete successfully: {labels}")

