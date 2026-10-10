#!/usr/bin/env python3
"""Unified CLI entry point for the Sensei shelf monitoring pipeline.

Examples:
    python scripts/run_pipeline.py --dry-run
    python scripts/run_pipeline.py --camera 6215 --image 1751001268.jpg -v
    python scripts/run_pipeline.py --camera 6215
    python scripts/run_pipeline.py                      # full run on all cameras
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sensei.config import Config
from sensei.output import save_output, validate_output
from sensei.pipeline import list_images_per_camera, run
from sensei.planogram import Planogram


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensei shelf monitoring pipeline")
    parser.add_argument("--camera", help="Limit execution to a specific camera ID (e.g. 6215)")
    parser.add_argument("--image", help="Limit execution to a specific image filename or stem")
    parser.add_argument("--out", "--output", dest="output", default=None, help="Output destination JSON path")
    parser.add_argument("--data-dir", "--images-dir", dest="images_dir", default=None, help="Directory containing images")
    parser.add_argument("--planogram", default=None, help="Path to planogram.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without API calls: tests geometry, cropping, skeleton, and safe fallbacks",
    )
    parser.add_argument("--samples", type=int, default=None, help="Number of VLM samples per area")
    parser.add_argument("--backend", choices=["gemini", "openrouter", "ollama", "llamacpp"], default=None,
                        help="Backend to use: gemini, openrouter, ollama, or llamacpp")
    parser.add_argument("--model", default=None, help="VLM model identifier override")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local disk cache and force fresh API calls")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides: dict = {}
    if args.no_cache:
        overrides["no_cache"] = True
    if args.dry_run:
        overrides["skip_api"] = True
    if args.samples is not None:
        overrides["samples"] = args.samples
    if args.model:
        if args.backend == "openrouter":
            overrides["openrouter_model"] = args.model
        else:
            overrides["model"] = args.model
    if args.backend:
        overrides["backend"] = args.backend
    if args.images_dir:
        overrides["images_dir"] = args.images_dir
    if args.planogram:
        overrides["planogram_path"] = args.planogram
    if args.output:
        overrides["output_path"] = args.output

    cfg = Config.from_env(**overrides)
    if not cfg.skip_api:
        if cfg.backend == "gemini" and not cfg.api_key:
            print("ERROR: GEMINI_API_KEY is not set (or use --dry-run / --backend openrouter).", file=sys.stderr)
            return 2
        elif cfg.backend == "openrouter" and not (cfg.openrouter_api_key or cfg.api_key):
            print("ERROR: OPENROUTER_API_KEY is not set.", file=sys.stderr)
            return 2

    planogram = Planogram.load(cfg.planogram_path)
    print(f"📋 Planogram: {len(planogram.cameras())} cameras, {len(planogram)} areas")

    cams = [args.camera] if args.camera else planogram.cameras()
    images_by_cam = list_images_per_camera(cfg.images_dir, [c for c in cams if c in planogram.cameras()])

    if args.image:
        images_by_cam = {
            cam: [n for n in names if n == args.image or Path(n).stem == args.image]
            for cam, names in images_by_cam.items()
        }
        images_by_cam = {cam: imgs for cam, imgs in images_by_cam.items() if imgs}

    total = sum(len(imgs) for imgs in images_by_cam.values())
    print(f"📷 Images: {total} total across {len(images_by_cam)} camera(s)")
    for cam in sorted(images_by_cam):
        print(f"   • Camera {cam}: {len(images_by_cam[cam])} images")

    if total == 0:
        print(f"⚠️ No matching images found in '{cfg.images_dir}'. Check your path or arguments.")
        return 1

    output = asyncio.run(run(cfg, planogram, camera=args.camera, image=args.image))

    out_path = cfg.output_path
    save_output(output, out_path)

    problems = validate_output(output, planogram, images_by_cam)
    area_count = sum(len(img) for cam in output.values() for img in cam.values())
    print(f"\n💾 Wrote {out_path}: {area_count} area predictions written.")

    if problems:
        print(f"⚠️ VALIDATION PROBLEMS ({len(problems)}):")
        for problem in problems[:20]:
            print(f"  - {problem}")
        return 1

    print("✅ Validation: OK (all required planogram areas present and well-formed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
