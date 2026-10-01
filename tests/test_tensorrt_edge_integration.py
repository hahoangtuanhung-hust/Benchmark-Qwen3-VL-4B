import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path

from qwen3_vl_benchmark.benchmark.report import DETAIL_FIELDS, main as report_main
from qwen3_vl_benchmark.benchmark.runner import RAW_FIELDS, measure_one, run_ccu2
from qwen3_vl_benchmark.common.backend import GenerationConfig, GenerationResult, InferenceBackend, TokenEvent
from qwen3_vl_benchmark.backends.tensorrt_edge.detect_support import architecture_for_cc, compatibility_rows


class OverlapBackend(InferenceBackend):
    name = "test"

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def start(self): pass
    def stop(self): pass
    def health(self): return True
    def metrics(self): return {}

    def generate(self, image, prompt, config):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        first = time.perf_counter()
        time.sleep(0.03)
        second = time.perf_counter()
        with self.lock:
            self.active -= 1
        return GenerationResult(text="ok", output_tokens=2, http_status=200, success=True,
                                token_events=[TokenEvent("o", first, 1), TokenEvent("k", second, 1)])


class TensorRTEdgeIntegrationTests(unittest.TestCase):
    def test_ccu2_requests_overlap(self):
        backend = OverlapBackend()
        images = [{"path": "a.jpg", "image_id": "a"}, {"path": "b.jpg", "image_id": "b"}]
        rows = run_ccu2(backend, images, Path("."), "prompt", GenerationConfig(), {}, "S4", 2)
        self.assertEqual(backend.max_active, 2)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["ccu"] for row in rows}, {2})

    def test_stream_chunks_are_not_token_itl(self):
        class ChunkBackend(OverlapBackend):
            def generate(self, image, prompt, config):
                now = time.perf_counter()
                return GenerationResult(text="two tokens", output_tokens=2, http_status=200, success=True,
                                        token_events=[TokenEvent("two tokens", now, 0)])
        row = measure_one(ChunkBackend(), Path("unused"), {}, "p", GenerationConfig(), {}, "S2", 1)
        self.assertEqual(row["itl_measurement_method"], "stream_event_not_token_level")
        self.assertIsNone(row["itl_mean_ms"])

    def test_no_runtime_is_unsupported(self):
        config = {"precisions": {"FP16": {"requested_semantics": "weights=FP16", "quant_recipe": "none",
                                             "weight_dtype": "FP16", "activation_dtype": "FP16"}},
                  "policy": {"kv_cache_dtype": "FP16", "visual_encoder_dtype": "FP16"}}
        env = {"gpu_count": 0, "TensorRT": None, "TensorRT-Edge-LLM": None}
        self.assertEqual(compatibility_rows(config, env)[0]["status"], "UNSUPPORTED")

    def test_architecture_mapping(self):
        self.assertEqual(architecture_for_cc(7, 5, "Tesla T4"), "Turing")
        self.assertEqual(architecture_for_cc(8, 9, "RTX 4090"), "Ada")
        self.assertEqual(architecture_for_cc(8, 6, "RTX 3090"), "Ampere")

    def test_t4_rejects_fp8(self):
        config = {
            "precisions": {
                "FP8": {"requested_semantics": "weights=FP8; activations=FP8",
                         "quant_recipe": "ModelOpt FP8", "weight_dtype": "FP8",
                         "activation_dtype": "FP8"},
            },
            "policy": {"kv_cache_dtype": "FP16", "visual_encoder_dtype": "FP16"},
        }
        env = {"gpu_count": 1, "gpu_arch": "Turing", "qwen3_vl_module": True,
               "TensorRT": "10", "TensorRT-Edge-LLM": "0.10.1"}
        row = compatibility_rows(config, env)[0]
        self.assertFalse(row["hardware_supported"])
        self.assertEqual(row["status"], "UNSUPPORTED")

    def test_t4_rejects_fp16_when_edge_runtime_has_no_sm75_kernels(self):
        config = {
            "precisions": {
                "FP16": {"requested_semantics": "weights=FP16; activations=FP16",
                          "quant_recipe": "none", "weight_dtype": "FP16",
                          "activation_dtype": "FP16"},
            },
            "policy": {"kv_cache_dtype": "FP16", "visual_encoder_dtype": "FP16"},
        }
        env = {"gpu_count": 1, "gpu_arch": "Turing", "compute_capability": "7.5",
               "qwen3_vl_module": True, "TensorRT": "10", "TensorRT-Edge-LLM": "0.10.1"}
        row = compatibility_rows(config, env)[0]
        self.assertEqual(row["status"], "UNSUPPORTED")
        self.assertIn("no required kernels", row["evidence"])

    def test_raw_schema_unique(self):
        self.assertEqual(len(RAW_FIELDS), len(set(RAW_FIELDS)))
        self.assertTrue({"ttft_ms", "ttfv_ms", "prefill_tps", "decode_tps", "kv_cache_mb"} <= set(RAW_FIELDS))

    def test_report_writes_detailed_scenario_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = {field: "" for field in RAW_FIELDS}
            row.update({"backend": "tensorrt_edge", "requested_precision": "FP16",
                        "actual_precision": "FP16", "quant_recipe": "none", "scenario": "S2",
                        "ccu": "1", "success": "True", "output_tokens": "8", "ttft_ms": "100",
                        "ttfv_ms": "120", "e2e_ms": "500", "prefill_tps": "50", "decode_tps": "20",
                        "tpot_ms": "30", "itl_mean_ms": "31", "itl_p50_ms": "30",
                        "itl_p95_ms": "40", "itl_p99_ms": "45", "idle_vram_mb": "1000",
                        "peak_vram_mb": "2000", "kv_cache_mb": "100", "gpu_util_mean": "70",
                        "gpu_util_peak": "90"})
            with (root / "raw_requests.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
                writer.writeheader(); writer.writerow(row)
            old_argv = __import__("sys").argv
            try:
                __import__("sys").argv = ["report", "--results-dir", str(root)]
                self.assertEqual(report_main(), 0)
            finally:
                __import__("sys").argv = old_argv
            with (root / "detailed_summary.csv").open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(list(summary[0]), DETAIL_FIELDS)
            self.assertEqual(summary[0]["Scenario"], "S2")
            self.assertIn("Detailed performance by precision", (root / "benchmark_report.md").read_text(encoding="utf-8"))

    def test_report_allows_metrics_unavailable_from_the_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = {field: "" for field in RAW_FIELDS}
            row.update({"backend": "tensorrt_edge", "requested_precision": "FP16",
                        "scenario": "S2", "ccu": "1", "success": "True"})
            with (root / "raw_requests.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
                writer.writeheader(); writer.writerow(row)
            old_argv = __import__("sys").argv
            try:
                __import__("sys").argv = ["report", "--results-dir", str(root)]
                self.assertEqual(report_main(), 0)
            finally:
                __import__("sys").argv = old_argv


if __name__ == "__main__":
    unittest.main()
