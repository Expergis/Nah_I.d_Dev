"""End-to-end pipeline: images x planogram areas -> scored predictions.

Supports both:
- Per-camera subdirectories: ``<images_dir>/<cam_id>/<timestamp>.jpg``
- Flat directory:            ``<images_dir>/<timestamp>.jpg``
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List

import cv2

from .config import Config
from .decision import (
    COMPLIANCE_LABELS,
    STOCK_LEVELS,
    decide_compliance,
    decide_stock,
    to_output,
)
from .geometry import context_crop, encode_png, warp_area
from .output import make_skeleton, validate_output
from .planogram import Planogram, PlanogramArea
from .prompting import build_prompt
from .vlm import VLMClient, VLMError
from .vlm_openrouter import OpenRouterClient

try:
    from .vlm_ollama import OllamaClient  # type: ignore
except ImportError:
    OllamaClient = None  # type: ignore

log = logging.getLogger(__name__)

Backend = VLMClient | OpenRouterClient


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(images_dir: str | Path) -> list[str]:
    """Flat image listing."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    files = sorted(
        p.name
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    return files


def list_images_per_camera(
    base_dir: str | Path,
    camera_ids: list[str],
) -> dict[str, list[str]]:
    """Discover images grouped per camera.

    Checks:
    1. Subdirectories per camera: base_dir/<camera_id>/*.jpg
    2. Flat directory: base_dir/*.jpg
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {base_dir}")

    per_camera: dict[str, list[str]] = {}
    for cam_id in camera_ids:
        cam_dir = base_dir / cam_id
        if cam_dir.is_dir():
            files = sorted(
                p.name
                for p in cam_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            if files:
                per_camera[cam_id] = files

    if per_camera:
        return per_camera

    # Fallback to flat directory
    flat_files = list_images(base_dir)
    if not flat_files:
        raise FileNotFoundError(f"No images found in {base_dir}")
    return {cam_id: list(flat_files) for cam_id in camera_ids}


def resolve_image_path(base_dir: str | Path, cam_id: str, image_name: str) -> Path:
    """Resolve file path whether nested in cam_id/ or directly in base_dir/."""
    base_dir = Path(base_dir)
    nested = base_dir / cam_id / image_name
    if nested.exists():
        return nested
    return base_dir / image_name


async def process_area(
    area: PlanogramArea,
    image: "cv2.MatLike",
    client: VLMClient | None,
    cfg: Config,
    semaphore: asyncio.Semaphore,
) -> dict:
    """One area -> one prediction dict (always returns a valid entry)."""
    crop, _mask, visible = warp_area(
        image, area.corners, cfg.crop_width, cfg.crop_height
    )
    ctx = None
    if cfg.include_context:
        ctx = context_crop(image, area.corners, cfg.context_padding)

    if client is None:
        # dry-run / fallback
        return {
            "planogram_compliant": True,
            "stock_level": "half",
            "stock_value": 0.5,
        }

    prompt = build_prompt(area, has_context=ctx is not None, visible_fraction=visible)
    images = [encode_png(crop)]
    if ctx is not None:
        images.append(encode_png(ctx))

    votes_stock: dict[str, int] = {}
    votes_comp: dict[str, int] = {}

    async with semaphore:
        results = await asyncio.gather(
            *(client.ask_json(prompt, images, cfg.temperature) for _ in range(cfg.samples)),
            return_exceptions=True,
        )

    for result in results:
        if isinstance(result, Exception):
            log.warning("sample failed for %s: %s", area.area_id, result)
            continue
        level = str(result.get("stock_level") or result.get("stock") or "").lower().strip()
        if not level and "coverage_percent" in result:
            try:
                cov = float(result["coverage_percent"])
                level = "empty" if cov <= 15 else ("half" if cov <= 70 else "full")
            except (ValueError, TypeError):
                pass

        if level in STOCK_LEVELS:
            votes_stock[level] = votes_stock.get(level, 0) + 1

        compliance = str(result.get("compliance") or "").lower().strip()
        if not compliance and "planogram_compliant" in result:
            compliance = "compliant" if result["planogram_compliant"] else "noncompliant"
        if compliance in COMPLIANCE_LABELS:
            votes_comp[compliance] = votes_comp.get(compliance, 0) + 1

    if not votes_stock:
        log.error("no usable samples for %s -> safe fallback", area.area_id)
        return {
            "planogram_compliant": True,
            "stock_level": "half",
            "stock_value": 0.5,
        }

    level = decide_stock(votes_stock)
    level_name, level_value = to_output(level)
    return {
        "planogram_compliant": decide_compliance(votes_comp),
        "stock_level": level_name,
        "stock_value": level_value,
    }


async def process_image(
    camera_id: str,
    image_path: Path,
    planogram: Planogram,
    client: VLMClient | None,
    cfg: Config,
    semaphore: asyncio.Semaphore,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    areas = planogram.areas_for(camera_id)
    result = {area.area_id: None for area in areas}
    tasks = {
        area.area_id: asyncio.create_task(
            process_area(area, image, client, cfg, semaphore)
        )
        for area in areas
    }
    for area_id, task in tasks.items():
        result[area_id] = await task
    return result


async def run(
    cfg: Config,
    planogram: Planogram | None = None,
    camera: str | None = None,
    image: str | None = None,
) -> dict:
    """Run the full (or partial) pipeline and return the output dict."""
    if planogram is None:
        planogram = Planogram.load(cfg.planogram_path)

    cameras = [camera] if camera else planogram.cameras()
    valid_cameras = [cam for cam in cameras if cam in planogram.cameras()]
    images_by_camera = list_images_per_camera(cfg.images_dir, valid_cameras)

    if image:
        images_by_camera = {
            cam: [n for n in names if n == image or Path(n).stem == image]
            for cam, names in images_by_camera.items()
        }
        images_by_camera = {cam: imgs for cam, imgs in images_by_camera.items() if imgs}
        if not images_by_camera:
            raise FileNotFoundError(f"Image not found: {image} in {cfg.images_dir}")

    output = make_skeleton(planogram, images_by_camera)

    client: VLMClient | None = None
    if not cfg.skip_api:
        if cfg.backend == "gemini":
            client = VLMClient(
                model=cfg.model,
                api_key=cfg.api_key or "",
                timeout_s=cfg.timeout_s,
                max_retries=cfg.max_retries,
                cache_dir=cfg.cache_dir,
            )
        elif cfg.backend == "openrouter":
            client = OpenRouterClient(
                model=cfg.openrouter_model,
                api_key=cfg.openrouter_api_key or cfg.api_key or "",
                timeout_s=cfg.timeout_s,
                max_retries=cfg.max_retries,
                cache_dir=cfg.cache_dir,
                no_cache=cfg.no_cache,
            )
        elif cfg.backend == "ollama":
            if OllamaClient is None:
                raise RuntimeError("Ollama backend is not installed on this branch. Use --backend gemini or --backend openrouter.")
            client = OllamaClient(
                model=cfg.ollama_model,
                base_url=cfg.ollama_url,
                timeout_s=cfg.timeout_s,
                max_retries=cfg.max_retries,
                cache_dir=cfg.cache_dir,
                num_ctx=cfg.ollama_num_ctx,
            )
            if not await client.ping():
                await client.aclose()
                raise RuntimeError(
                    f"Ollama not reachable at {cfg.ollama_url}. "
                    "Start it or use --backend gemini / --backend openrouter."
                )
        else:
            raise ValueError(f"Unknown backend: {cfg.backend}. Choose 'gemini', 'openrouter', or 'ollama'.")

    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    try:
        for cam_id, image_names in images_by_camera.items():
            for name in image_names:
                img_path = resolve_image_path(cfg.images_dir, cam_id, name)
                output[cam_id][name] = await process_image(
                    cam_id, img_path, planogram, client, cfg, semaphore
                )
                log.info("done %s / %s (%d areas)", cam_id, name, len(output[cam_id][name]))
    finally:
        if client is not None:
            await client.aclose()

    problems = validate_output(output, planogram, images_by_camera)
    if problems:
        log.error("output validation found %d problems:", len(problems))
        for problem in problems[:20]:
            log.error("  %s", problem)
    return output
