from ...common.http_backend import OpenAIHTTPBackend


class LlamaCppBackend(OpenAIHTTPBackend):
    def __init__(self, base_url: str = "http://127.0.0.1:8080", **kwargs):
        super().__init__("llamacpp", base_url, "qwen3-vl", **kwargs)

