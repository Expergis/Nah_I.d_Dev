"""Ollama local backend — same interface as the Gemini client.

Primary for the hackathon: Qwen3-VL on the RTX 4060 via Ollama
(http://localhost:11434). Supports schema-constrained JSON output and
logprob extraction, with automatic fallback to sampled voting when
logprobs are unavailable.

Same contract as ``VLMClient.ask_json(prompt, images, temperature) -> dict``.
When first-token logprobs for empty/half/full are available, the result dict
additionally carries ``"_p": {"empty": p, "half": p, "full": p}``.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import random
import time
from pathlib import Path

import httpx

OLLAMA_LEVEL_TOKENS = ("empty", "half", "full")

OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "compliance": {
            "type": "string",
            "enum": ["compliant", "noncompliant", "uncertain"],
        },
        "stock_level": {"type": "string", "enum": ["empty", "half", "full"]},
        "coverage_percent": {"type": "number"},
        "confidence": {"type": "number"},
        "occluded": {"type": "boolean"},
    },
    "required": ["compliance", "stock_level", "coverage_percent"],
}


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3-vl:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 300.0,
        max_retries: int = 3,
        cache_dir: str = ".cache/vlm",
        num_ctx: int = 8192,
    ):
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.num_ctx = num_ctx
        self._base = base_url.rstrip("/")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def ping(self) -> bool:
        try:
            resp = await self._http.get(f"{self._base}/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------------
    def _cache_file(self, prompt: str, images: list[bytes], temperature: float) -> Path:
        digest = hashlib.sha256()
        digest.update(f"ollama:{self.model}".encode("utf-8"))
        digest.update(str(temperature).encode("utf-8"))
        digest.update(prompt.encode("utf-8"))
        for image in images:
            digest.update(hashlib.sha256(image).digest())
        return self._cache_dir / (digest.hexdigest() + ".json")

    def _request_body(self, prompt: str, images: list[bytes]) -> dict:
        return {
            "model": self.model,
            "stream": False,
            "format": OLLAMA_SCHEMA,
            "options": {"temperature": 0.6, "num_ctx": self.num_ctx},
            "logprobs": True,
            "logprobs_token_count": 8,
            "keep_alive": -1,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(img).decode("ascii") for img in images],
                }
            ],
        }

    @staticmethod
    def extract_level_probs(payload: dict) -> dict[str, float] | None:
        """First-token probabilities over empty/half/full, if available."""
        logprobs = (payload.get("message") or {}).get("logprobs") or []
        if not logprobs:
            return None
        first = logprobs[0]
        top = first.get("top_logprobs") or []
        by_token = {t.get("token"): t.get("logprob") for t in top}
        by_token.setdefault(first.get("token"), first.get("logprob"))
        wanted = [by_token.get(tok) for tok in OLLAMA_LEVEL_TOKENS]
        if any(w is None for w in wanted):
            return None
        exps = [math.exp(w) for w in wanted]
        total = sum(exps)
        if total <= 0:
            return None
        return {
            tok: e / total for tok, e in zip(OLLAMA_LEVEL_TOKENS, exps)
        }

    async def ask_json(
        self,
        prompt: str,
        images: list[bytes],
        temperature: float = 0.6,
    ) -> dict:
        cache_file = self._cache_file(prompt, images, temperature)
        if cache_file.exists():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
            result["_t"] = {"cached": True, "wall_s": 0.0}
            return result

        body = self._request_body(prompt, images)
        body["options"]["temperature"] = temperature
        url = f"{self._base}/api/chat"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = await self._http.post(url, json=body)
                if response.status_code >= 400:
                    raise OllamaError(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                payload = response.json()
                content = (payload.get("message") or {}).get("content", "")
                if not content.strip():
                    raise OllamaError(f"Empty response: {payload!r}"[:300])
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise OllamaError(f"Non-JSON content: {content[:200]!r}") from exc
                probs = self.extract_level_probs(payload)
                if probs:
                    result["_p"] = probs
                # Ollama reports durations in nanoseconds; convert to ms.
                result["_t"] = {
                    "cached": False,
                    "wall_s": round(time.perf_counter() - t0, 3),
                    "prompt_tokens": int(payload.get("prompt_eval_count") or 0),
                    "decode_tokens": int(payload.get("eval_count") or 0),
                    "prefill_ms": round((payload.get("prompt_eval_duration") or 0) / 1e6, 1),
                    "decode_ms": round((payload.get("eval_duration") or 0) / 1e6, 1),
                }
                cache_file.write_text(
                    json.dumps(result, ensure_ascii=False), encoding="utf-8"
                )
                return result
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            except OllamaError as exc:
                if "HTTP 4" in str(exc):
                    raise
                last_error = exc
            if attempt < self.max_retries:
                await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.5))
        raise OllamaError(f"All {self.max_retries + 1} attempts failed: {last_error}")
