"""Async Gemini client (REST, no SDK dependency) with caching and retries.

Uses the generativelanguage API directly so we are insulated from SDK
churn. Every call is cached on disk keyed by (model, temperature, prompt,
image hashes) — retries never pay twice, and prompt iteration is free.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import random
import time
from pathlib import Path

import httpx

ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reason": {"type": "STRING"},
        "compliance": {
            "type": "STRING",
            "enum": ["compliant", "noncompliant", "uncertain"],
        },
        "stock_level": {"type": "STRING", "enum": ["empty", "half", "full"]},
        "coverage_percent": {"type": "NUMBER"},
        "confidence": {"type": "NUMBER"},
        "occluded": {"type": "BOOLEAN"},
    },
    "required": ["compliance", "stock_level", "coverage_percent"],
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class VLMError(RuntimeError):
    """Raised when the API call fails after all retries."""


class VLMClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        timeout_s: float = 90.0,
        max_retries: int = 4,
        cache_dir: str = ".cache/vlm",
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.AsyncClient(
            timeout=timeout_s,
            headers={"x-goog-api-key": api_key},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    def _cache_file(self, prompt: str, images: list[bytes], temperature: float) -> Path:
        digest = hashlib.sha256()
        digest.update(self.model.encode("utf-8"))
        digest.update(str(temperature).encode("utf-8"))
        digest.update(prompt.encode("utf-8"))
        for image in images:
            digest.update(hashlib.sha256(image).digest())
        return self._cache_dir / (digest.hexdigest() + ".json")

    def _request_body(self, prompt: str, images: list[bytes]) -> dict:
        parts: list[dict] = [{"text": prompt}]
        for image in images:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(image).decode("ascii"),
                    }
                }
            )
        return {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }

    @staticmethod
    def _parse_response(payload: dict) -> dict:
        candidates = payload.get("candidates") or []
        if not candidates:
            feedback = payload.get("promptFeedback") or {}
            raise VLMError(f"No candidates: {feedback or payload!r}"[:500])
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text.strip():
            raise VLMError(f"Empty model response: {payload!r}"[:500])
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise VLMError(f"Model returned non-JSON: {text[:300]!r}") from exc

    async def ask_json(
        self,
        prompt: str,
        images: list[bytes],
        temperature: float = 0.6,
    ) -> dict:
        """One structured-JSON completion. Cached. Retries transient failures."""
        cache_file = self._cache_file(prompt, images, temperature)
        if cache_file.exists():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
            result["_t"] = {"cached": True, "wall_s": 0.0}
            return result

        body = self._request_body(prompt, images)
        body["generationConfig"]["temperature"] = temperature
        url = ENDPOINT.format(model=self.model)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = await self._http.post(url, json=body)
                if response.status_code in RETRYABLE_STATUS:
                    raise VLMError(f"HTTP {response.status_code}: {response.text[:300]}")
                if response.status_code >= 400:
                    raise VLMError(f"HTTP {response.status_code}: {response.text[:500]}")
                payload = response.json()
                result = self._parse_response(payload)
                usage = payload.get("usageMetadata") or {}
                result["_t"] = {
                    "cached": False,
                    "wall_s": round(time.perf_counter() - t0, 3),
                    "prompt_tokens": int(usage.get("promptTokenCount") or 0),
                    "decode_tokens": int(usage.get("candidatesTokenCount") or 0),
                    "prefill_ms": 0.0,
                    "decode_ms": 0.0,
                }
                cache_file.write_text(
                    json.dumps(result, ensure_ascii=False), encoding="utf-8"
                )
                return result
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            except VLMError as exc:
                if "HTTP 4" in str(exc) and "HTTP 429" not in str(exc):
                    raise  # non-retryable client error
                last_error = exc
            if attempt < self.max_retries:
                backoff = (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(backoff)
        raise VLMError(f"All {self.max_retries + 1} attempts failed: {last_error}")
