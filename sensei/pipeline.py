"""End-to-end pipeline: images x planogram areas -> scored predictions.

Layout assumption: ``data/images`` holds one file per capture timestamp
(e.g. ``1751052241.jpg``) shared by all cameras, matching the annotation
keying (camera -> image -> area). If you receive per-camera folders instead,
set the flat file list accordingly (see scripts/run_pipeline.py).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

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

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(images_dir: str | Path) -> list[str]:
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    files = sorted(
        p.name
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return files


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
        # dry-run / no API: keep the conservative fallback
        return dict({
            "planogram_compliant": True,
            "stock_level": "empty",
            "stock_value": 0.0,
        })

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
        level = result.get("stock_level")
        if level in STOCK_LEVELS:
            votes_stock[level] = votes_stock.get(level, 0) + 1
        compliance = result.get("compliance")
        if compliance in COMPLIANCE_LABELS:
            votes_comp[compliance] = votes_comp.get(compliance, 0) + 1

    if not votes_stock:
        log.error("no usable samples for %s -> fallback", area.area_id)
        return {
            "planogram_compliant": True,
            "stock_level": "empty",
            "stock_value": 0.0,
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
    result = {area.area_id: None for area in areas}  # type: ignore[dict-item]
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

    all_images = list_images(cfg.images_dir)
    if image:
        all_images = [name for name in all_images if name == image or Path(name).stem == image]
    if not all_images:
        raise FileNotFoundError(f"Image not found: {image}")

    cameras = [camera] if camera else planogram.cameras()
    images_by_camera = {cam: all_images for cam in cameras if cam in planogram.cameras()}

    output = make_skeleton(planogram, images_by_camera)

    client: VLMClient | None = None
    if not cfg.skip_api:
        client = VLMClient(
            model=cfg.model,
            api_key=cfg.api_key or "",
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
            cache_dir=cfg.cache_dir,
        )

    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    try:
        for cam_id, image_names in images_by_camera.items():
            for name in image_names:
                path = Path(cfg.images_dir) / name
                output[cam_id][name] = await process_image(
                    cam_id, path, planogram, client, cfg, semaphore
                )
                log.info("done %s / %s (%d areas)", cam_id, name, len(image_names and output[cam_id][name]))
    finally:
        if client is not None:
            await client.aclose()

    problems = validate_output(output, planogram, images_by_camera)
    if problems:
        log.error("output validation found %d problems:", len(problems))
        for problem in problems[:20]:
            log.error("  %s", problem)
    return output
