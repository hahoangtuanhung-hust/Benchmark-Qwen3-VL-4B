from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 42
    stream: bool = True
    cache_prompt: bool = False


@dataclass
class TokenEvent:
    text: str
    timestamp: float
    token_count: int = 1
    visible: bool = True


@dataclass
class GenerationResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    http_status: int = 0
    success: bool = False
    error: str = ""
    token_events: list[TokenEvent] = field(default_factory=list)
    native_metrics: dict[str, Any] = field(default_factory=dict)
    request_start: float = 0.0
    request_end: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("token_events", None)
        return value


class InferenceBackend(ABC):
    """Lifecycle and inference contract shared by all benchmark backends."""

    name: str

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def health(self) -> bool: ...

    @abstractmethod
    def generate(
        self, image: Path, prompt: str, config: GenerationConfig
    ) -> GenerationResult: ...

    @abstractmethod
    def metrics(self) -> dict[str, Any]: ...

