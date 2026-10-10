"""Async OpenRouter client (OpenAI-compatible REST API) with disk caching and retries.

Provides access to Gemini 2.0 Flash, Qwen2.5-VL, GPT-4o-mini, etc. via OpenRouter
without Google free-tier rate limits.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import random
import re
from pathlib import Path

import httpx

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter API call fails after retries."""


class OpenRouterClient:
    def __init__(
        self,
        model: str = "google/gemini-2.0-flash-001",
        api_key: str = "",
        timeout_s: float = 90.0,
        max_retries: int = 4,
        cache_dir: str = ".cache/vlm",
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.AsyncClient(
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/Expergis/Nah_I.d_Dev",
                "X-Title": "Sensei Shelf Monitor",
            },
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def _cache_file(self, prompt: str, images: list[bytes], temperature: float) -> Path:
        h = hashlib.sha256()
        h.update(self.model.encode("utf-8"))
        h.update(f"{temperature:.3f}".encode("ascii"))
        h.update(prompt.encode("utf-8"))
        for img in images:
            h.update(hashlib.sha256(img).digest())
        return self._cache_dir / f"{h.hexdigest()}.json"

    def _request_body(self, prompt: str, images: list[bytes], temperature: float) -> dict:
        content_parts: list[dict] = [{"type": "text", "text": prompt + "\n\nRespond with valid JSON only."}]
        for image in images:
            b64 = base64.b64encode(image).decode("ascii")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content_parts}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

    @staticmethod
    def _parse_response(payload: dict) -> dict:
        choices = payload.get("choices") or []
        if not choices:
            raise OpenRouterError(f"No choices in OpenRouter response: {payload!r}"[:500])
        msg = choices[0].get("message", {})
        raw_text = msg.get("content", "").strip()
        if not raw_text:
            raise OpenRouterError("Empty response content from OpenRouter")

        # Strip markdown fences if present
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"```$", "", raw_text, flags=re.MULTILINE).strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(f"Model returned non-JSON: {raw_text[:300]!r}") from exc

    async def ask_json(
        self,
        prompt: str,
        images: list[bytes],
        temperature: float = 0.6,
    ) -> dict:
        """One completion via OpenRouter with disk caching and retry logic."""
        cache_file = self._cache_file(prompt, images, temperature)
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        body = self._request_body(prompt, images, temperature)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.post(OPENROUTER_ENDPOINT, json=body)
                if response.status_code in RETRYABLE_STATUS:
                    raise OpenRouterError(f"HTTP {response.status_code}: {response.text[:300]}")
                if response.status_code >= 400:
                    raise OpenRouterError(f"Non-retryable HTTP {response.status_code}: {response.text[:300]}")

                parsed = self._parse_response(response.json())
                cache_file.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
                return parsed
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    delay = (2 ** attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(delay)

        raise OpenRouterError(f"OpenRouter call failed after {self.max_retries} retries: {last_error}")
