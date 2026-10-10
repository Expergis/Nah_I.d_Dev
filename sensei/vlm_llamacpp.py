"""Async llama.cpp server client (OpenAI-compatible) with disk caching and retries.

Talks to a local llama-server instance with --cont-batching for high GPU
utilisation. Expects the server to be started separately, e.g.:

    llama-server -m model.gguf --mmproj mmproj.gguf \
        --host 0.0.0.0 --port 8080 \
        -ngl 99 --cont-batching --parallel 4
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import random
import re
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LlamaCppError(RuntimeError):
    """Raised when llama.cpp server call fails after retries."""


class LlamaCppClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "local",
        timeout_s: float = 120.0,
        max_retries: int = 3,
        cache_dir: str = ".cache/vlm",
        no_cache: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.no_cache = no_cache
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def ping(self) -> bool:
        """Check if llama-server is ready (model loaded)."""
        try:
            r = await self._http.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return False

    # -- caching ----------------------------------------------------------
    def _cache_file(self, prompt: str, images: list[bytes], temperature: float) -> Path:
        h = hashlib.sha256()
        h.update(b"llamacpp:")
        h.update(self.model.encode("utf-8"))
        h.update(f"{temperature:.3f}".encode("ascii"))
        h.update(prompt.encode("utf-8"))
        for img in images:
            h.update(hashlib.sha256(img).digest())
        return self._cache_dir / f"{h.hexdigest()}.json"

    # -- request building -------------------------------------------------
    def _request_body(self, prompt: str, images: list[bytes], temperature: float) -> dict:
        schema_instruction = (
            "\n\nReturn EXACTLY a JSON object with this structure:\n"
            "{\n"
            '  "reason": "short visual rationale",\n'
            '  "compliance": "compliant" | "noncompliant" | "uncertain",\n'
            '  "stock_level": "empty" | "half" | "full",\n'
            '  "coverage_percent": <number 0-100>\n'
            "}"
        )
        content_parts: list[dict] = [{"type": "text", "text": prompt + schema_instruction}]
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
        }

    # -- response parsing -------------------------------------------------
    @staticmethod
    def _parse_response(payload: dict) -> dict:
        choices = payload.get("choices") or []
        if not choices:
            raise LlamaCppError(f"No choices in response: {payload!r}"[:500])
        msg = choices[0].get("message", {})
        raw_text = msg.get("content", "").strip()
        if not raw_text:
            raise LlamaCppError("Empty response content from llama-server")

        # Strip markdown fences if present
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"```$", "", raw_text, flags=re.MULTILINE).strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: search for first { ... } block
            m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            raise LlamaCppError(f"Model returned non-JSON: {raw_text[:300]!r}")

    # -- main API ---------------------------------------------------------
    async def ask_json(
        self,
        prompt: str,
        images: list[bytes],
        temperature: float = 0.6,
    ) -> dict:
        """One completion via llama-server with disk caching and retry logic."""
        cache_file = self._cache_file(prompt, images, temperature)
        if not self.no_cache and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        body = self._request_body(prompt, images, temperature)
        url = f"{self.base_url}/v1/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.post(url, json=body)
                if response.status_code in RETRYABLE_STATUS:
                    raise LlamaCppError(f"HTTP {response.status_code}: {response.text[:300]}")
                if response.status_code >= 400:
                    raise LlamaCppError(
                        f"Non-retryable HTTP {response.status_code}: {response.text[:300]}"
                    )

                parsed = self._parse_response(response.json())
                cache_file.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
                return parsed
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            except LlamaCppError as exc:
                last_error = exc
            if attempt < self.max_retries:
                delay = (2 ** attempt) + random.uniform(0.1, 0.5)
                await asyncio.sleep(delay)

        raise LlamaCppError(
            f"llama-server call failed after {self.max_retries} retries: {last_error}"
        )
