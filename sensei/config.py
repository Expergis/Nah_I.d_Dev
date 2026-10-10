"""Central configuration for the Sensei challenge pipeline.

Everything can be set via environment variables (``SENSEI_*``) or a local
``.env`` file. The Gemini API key is read from ``GEMINI_API_KEY``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional, keeps imports working without python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

DEFAULT_MODEL = "gemini-2.5-flash"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # --- data locations -------------------------------------------------
    planogram_path: str = "planogram.json"
    images_dir: str = "data/images"
    output_path: str = "outputs/predictions.json"

    # --- model ----------------------------------------------------------
    backend: str = "ollama"   # "ollama" (local, default) or "gemini" (API booster)
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:8b"
    ollama_num_ctx: int = 4096
    samples: int = 3          # VLM samples (votes) per area
    temperature: float = 0.6  # sampling temperature (diversity for votes)

    # --- geometry -------------------------------------------------------
    crop_width: int = 512
    crop_height: int = 384
    context_padding: float = 0.30  # fraction of the area span added around it
    include_context: bool = True   # also send a wider shelf view
    fruit_threshold: float = 0.05  # 0 disables; if a crop's fruit_fraction is below
                                   # this, skip the model and call the area empty


    # --- runtime --------------------------------------------------------
    max_concurrency: int = 1      # 1 for 8GB VRAM (avoid KV cache overflow)
    timeout_s: float = 240.0      # slow if the model is CPU-offloaded; see ollama ps
    max_retries: int = 4
    cache_dir: str = ".cache/vlm"
    skip_api: bool = False  # dry-run mode: fill the skeleton without calls

    @classmethod
    def from_env(cls, **overrides: object) -> "Config":
        cfg = cls(
            planogram_path=_env("SENSEI_PLANOGRAM", "planogram.json"),
            images_dir=_env("SENSEI_IMAGES_DIR", "data/images"),
            output_path=_env("SENSEI_OUTPUT", "outputs/predictions.json"),
            backend=_env("SENSEI_BACKEND", "ollama").lower(),
            model=_env("SENSEI_MODEL", DEFAULT_MODEL),
            ollama_url=_env("SENSEI_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=_env("SENSEI_OLLAMA_MODEL", "qwen3-vl:8b"),
            ollama_num_ctx=_env_int("SENSEI_OLLAMA_CTX", 4096),
            api_key=os.environ.get("GEMINI_API_KEY") or None,
            samples=_env_int("SENSEI_SAMPLES", 3),
            temperature=_env_float("SENSEI_TEMPERATURE", 0.6),
            crop_width=_env_int("SENSEI_CROP_W", 512),
            crop_height=_env_int("SENSEI_CROP_H", 384),
            context_padding=_env_float("SENSEI_CONTEXT_PAD", 0.30),
            include_context=_env_bool("SENSEI_INCLUDE_CONTEXT", True),
            fruit_threshold=_env_float("SENSEI_FRUIT_THR", 0.05),
            max_concurrency=_env_int("SENSEI_CONCURRENCY", 1),
            timeout_s=_env_float("SENSEI_TIMEOUT_S", 240.0),
            max_retries=_env_int("SENSEI_RETRIES", 4),
            cache_dir=_env("SENSEI_CACHE_DIR", ".cache/vlm"),
            skip_api=_env_bool("SENSEI_SKIP_API", False),
        )
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
