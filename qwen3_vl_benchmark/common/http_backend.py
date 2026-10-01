from __future__ import annotations

import base64
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from .backend import GenerationConfig, GenerationResult, InferenceBackend, TokenEvent


class OpenAIHTTPBackend(InferenceBackend):
    """OpenAI-compatible VLM adapter used by both server implementations."""

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        start_command: list[str] | None = None,
        cwd: Path | None = None,
        timeout_s: int = 300,
        include_seed: bool = True,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.start_command = start_command
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.include_seed = include_seed
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if not self.start_command:
            return
        self._process = subprocess.Popen(
            self.start_command,
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)

    def health(self) -> bool:
        for endpoint in ("/health", "/v1/models"):
            try:
                if requests.get(self.base_url + endpoint, timeout=3).status_code < 400:
                    return True
            except requests.RequestException:
                continue
        return False

    def wait_ready(self, timeout_s: int = 300) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.health():
                return
            if self._process and self._process.poll() is not None:
                raise RuntimeError(f"{self.name} server exited with {self._process.returncode}")
            time.sleep(1)
        raise TimeoutError(f"{self.name} server was not ready after {timeout_s}s")

    def generate(
        self, image: Path, prompt: str, config: GenerationConfig
    ) -> GenerationResult:
        result = GenerationResult()
        try:
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    {"type": "text", "text": prompt},
                ]}],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if self.include_seed:
                payload["seed"] = config.seed
            result.request_start = time.perf_counter()
            response = requests.post(
                self.base_url + "/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=self.timeout_s,
                headers={"Accept": "text/event-stream"},
            )
            result.http_status = response.status_code
            if response.status_code >= 400:
                result.error = response.text[:1000]
                return result

            parts: list[str] = []
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_text = raw_line[5:].strip()
                if data_text == "[DONE]":
                    break
                try:
                    data = json.loads(data_text)
                except json.JSONDecodeError:
                    continue
                usage = data.get("usage") or {}
                result.input_tokens = usage.get("prompt_tokens", result.input_tokens)
                result.output_tokens = usage.get("completion_tokens", result.output_tokens)
                result.native_metrics.update(data.get("metrics") or data.get("timings") or {})
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    visible = bool(content.strip()) and not content.lstrip().startswith("<think")
                    parts.append(content)
                    # Generic SSE events may contain multiple tokens. Only an explicit
                    # token id allows the runner to report token-level ITL.
                    token_count = 1 if delta.get("token_id") is not None else 0
                    result.token_events.append(TokenEvent(content, time.perf_counter(), token_count, visible))
            result.text = "".join(parts)
            result.request_end = time.perf_counter()
            if not result.output_tokens:
                result.output_tokens = len(result.token_events)
            result.success = bool(result.text)
            if not result.success:
                result.error = "empty_response"
        except (OSError, requests.RequestException) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result

    def metrics(self) -> dict[str, Any]:
        try:
            response = requests.get(self.base_url + "/metrics", timeout=5)
            return {"status": response.status_code, "text": response.text} if response.ok else {}
        except requests.RequestException:
            return {}
