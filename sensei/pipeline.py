"""End-to-end pipeline: images x planogram areas -> scored predictions.

Layout assumption: ``data/images`` holds one file per capture timestamp
(e.g. ``1751052241.jpg``) shared by all cameras, matching the annotation
keying (camera -> image -> area). If you receive per-camera folders instead,
set the flat file list accordingly (see scripts/run_pipeline.py).
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import cv2

from .config import Config
from .decision import (
    COMPLIANCE_LABELS,
    STOCK_LEVELS,
    best_level_from_probs,
    decide_compliance,
    decide_stock,
    to_output,
)
from .geometry import context_crop, encode_png, warp_area
from .output import make_skeleton, validate_output
from .planogram import Planogram, PlanogramArea
from .timing import RunStats
from .prompting import build_prompt
from .vlm import VLMClient
from .vlm_ollama import OllamaClient

log = logging.getLogger(__name__)

Backend = VLMClient | OllamaClient

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(images_dir: str | Path) -> list[str]:
    """Flat-layout image listing (legacy; prefer list_images_per_camera)."""
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


def list_images_per_camera(
    base_dir: str | Path,
    camera_ids: list[str],
) -> dict[str, list[str]]:
    """Discover images per camera.

    Supports two layouts:
    - Per-camera folders:  base_dir/<cam_id>/<timestamp>.jpg
    - Flat shared folder:  base_dir/<timestamp>.jpg  (same set for all cameras)
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {base_dir}")

    per_camera: dict[str, list[str]] = {}
    for cam_id in camera_ids:
        cam_dir = base_dir / cam_id
        if cam_dir.is_dir():
            files = sorted(
                p.name for p in cam_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            if files:
                per_camera[cam_id] = files

    if per_camera:
        return per_camera

    # Flat layout fallback: all cameras share the same image set
    files = sorted(
        p.name for p in base_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No images found in {base_dir}")
    return {cam_id: list(files) for cam_id in camera_ids}


def resolve_image_path(base_dir: str | Path, cam_id: str, image_name: str) -> Path:
    """Build the file path for an image, handling both per-camera and flat layouts."""
    base_dir = Path(base_dir)
    cam_dir = base_dir / cam_id
    if cam_dir.is_dir():
        return cam_dir / image_name
    return base_dir / image_name


async def process_area(
    area: PlanogramArea,
    image: "cv2.MatLike",
    client: VLMClient | None,
    cfg: Config,
    semaphore: asyncio.Semaphore,
    camera_id: str = "",
    image_name: str = "",
    stats: "RunStats | None" = None,
) -> dict:
    """One area -> one prediction dict (always returns a valid entry)."""
    t0 = time.perf_counter()

    def _record(samples_ok: int, results_: list[object] | None = None) -> None:
        if stats is None:
            return
        cached = 0
        prompt_tokens = decode_tokens = 0
        prefill_ms = decode_ms = 0.0
        for r in results_ or []:
            if isinstance(r, Exception):
                continue
            t = r.get("_t") or {}
            if t.get("cached"):
                cached += 1
            prompt_tokens += int(t.get("prompt_tokens") or 0)
            decode_tokens += int(t.get("decode_tokens") or 0)
            prefill_ms += float(t.get("prefill_ms") or 0.0)
            decode_ms += float(t.get("decode_ms") or 0.0)
        wall = time.perf_counter() - t0
        stats.record_area(
            camera=camera_id,
            image=image_name,
            area_id=area.area_id,
            wall_s=wall,
            samples_ok=samples_ok,
            samples_total=cfg.samples,
            cached_calls=cached,
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
        )
        log.info(
            "area %s: %.2fs (%d/%d samples ok, %d cached)",
            area.area_id, wall, samples_ok, cfg.samples, cached,
        )

    crop, _mask, visible = warp_area(
        image, area.corners, cfg.crop_width, cfg.crop_height
    )
    ctx = None
    if cfg.include_context:
        ctx = context_crop(image, area.corners, cfg.context_padding)

    if client is None:
        # dry-run / no API: keep the conservative fallback
        _record(0)
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
    explicit_probs: list[dict] = []
    async with semaphore:
        results = await asyncio.gather(
            *(client.ask_json(prompt, images, cfg.temperature) for _ in range(cfg.samples)),
            return_exceptions=True,
        )

    samples_ok = 0
    for result in results:
        if isinstance(result, Exception):
            log.warning("sample failed for %s: %s", area.area_id, result)
            continue
        samples_ok += 1
        level = result.get("stock_level")
        if level in STOCK_LEVELS:
            votes_stock[level] = votes_stock.get(level, 0) + 1
        compliance = result.get("compliance")
        if compliance in COMPLIANCE_LABELS:
            votes_comp[compliance] = votes_comp.get(compliance, 0) + 1
        probs = result.get("_p")
        if probs:
            explicit_probs.append(probs)

    if not votes_stock:
        log.error("no usable samples for %s -> fallback", area.area_id)
        _record(samples_ok, results)
        return {
            "planogram_compliant": True,
            "stock_level": "empty",
            "stock_value": 0.0,
        }

    if explicit_probs:
        # logprobs path: average first-token probabilities, then EV rule
        p = {
            k: sum(q.get(k, 0.0) for q in explicit_probs) / len(explicit_probs)
            for k in STOCK_LEVELS
        }
        level = best_level_from_probs(p["empty"], p["half"], p["full"])
    else:
        level = decide_stock(votes_stock)
    level_name, level_value = to_output(level)
    _record(samples_ok, results)
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
    stats: "RunStats | None" = None,
) -> dict:
    t0 = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    areas = planogram.areas_for(camera_id)
    result = {area.area_id: None for area in areas}  # type: ignore[dict-item]
    tasks = {
        area.area_id: asyncio.create_task(
            process_area(
                area, image, client, cfg, semaphore,
                camera_id=camera_id,
                image_name=image_path.name,
                stats=stats,
            )
        )
        for area in areas
    }
    for area_id, task in tasks.items():
        result[area_id] = await task
    if stats is not None:
        wall = time.perf_counter() - t0
        stats.record_image(camera_id, image_path.name, wall)
        log.info(
            "image %s/%s: %.1fs (%d areas)",
            camera_id, image_path.name, wall, len(areas),
        )
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
            raise FileNotFoundError(f"Image not found: {image}")

    output = make_skeleton(planogram, images_by_camera)

    client: Backend | None = None
    if not cfg.skip_api:
        if cfg.backend == "gemini":
            client = VLMClient(
                model=cfg.model,
                api_key=cfg.api_key or "",
                timeout_s=cfg.timeout_s,
                max_retries=cfg.max_retries,
                cache_dir=cfg.cache_dir,
            )
        else:
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
                    "Start it (setup_gpu.ps1) or use SENSEI_BACKEND=gemini."
                )

    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    stats = RunStats()
    try:
        for cam_id, image_names in images_by_camera.items():
            for name in image_names:
                path = resolve_image_path(cfg.images_dir, cam_id, name)
                output[cam_id][name] = await process_image(
                    cam_id, path, planogram, client, cfg, semaphore, stats=stats
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
    return output, stats
