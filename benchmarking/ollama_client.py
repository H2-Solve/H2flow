"""Minimal Ollama HTTP client (standard library only).

Talks to the REST API directly so every call returns token counts, timing and
the stop reason, which the harness records. Host: OLLAMA_HOST or localhost:11434.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    done_reason: str | None  # "stop", or "length" when num_predict cut the answer off

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


def default_host() -> str:
    host = os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    return host if host.startswith("http") else f"http://{host}"


class OllamaClient:
    def __init__(self, host: str | None = None, timeout_s: float = 600.0):
        self.host = (host or default_host()).rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            raise OllamaError(f"{path} returned HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host} ({exc})") from exc

    def chat(self, model: str, messages: list[dict], options: dict) -> ChatResult:
        start = time.perf_counter()
        res = self._request("/api/chat", {
            "model": model, "messages": messages, "stream": False, "options": options,
        })
        return ChatResult(
            content=(res.get("message") or {}).get("content", ""),
            prompt_tokens=res.get("prompt_eval_count"),
            completion_tokens=res.get("eval_count"),
            latency_s=time.perf_counter() - start,
            done_reason=res.get("done_reason"),
        )

    def version(self) -> str | None:
        return self._request("/api/version").get("version")

    def model_info(self, model: str) -> dict:
        """Digest, quantization and Modelfile parameters, for the run manifest."""
        show = self._request("/api/show", {"model": model})
        digest = None
        for m in self._request("/api/tags").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                digest = m.get("digest")
        details = show.get("details") or {}
        return {
            "name": model,
            "digest": digest,
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "modelfile_parameters": show.get("parameters"),
        }
