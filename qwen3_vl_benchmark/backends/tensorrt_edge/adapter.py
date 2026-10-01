from ...common.http_backend import OpenAIHTTPBackend


class TensorRTEdgeBackend(OpenAIHTTPBackend):
    """Adapter for a TensorRT-Edge-LLM OpenAI-compatible serving process."""

    def __init__(self, base_url: str = "http://127.0.0.1:8090", **kwargs):
        super().__init__("tensorrt_edge", base_url, "Qwen/Qwen3-VL-4B-Instruct",
                         include_seed=False, **kwargs)
