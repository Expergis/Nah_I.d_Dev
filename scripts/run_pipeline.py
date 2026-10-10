"""CLI entry point for the Sensei pipeline.

Examples:
    python scripts/run_pipeline.py --dry-run
    python scripts/run_pipeline.py --camera 6215 --image 1751052241.jpg
    python scripts/run_pipeline.py          # full run (needs GEMINI_API_KEY)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensei.config import Config  # noqa: E402
from sensei.output import save_output, validate_output  # noqa: E402
from sensei.pipeline import list_images, run  # noqa: E402
from sensei.planogram import Planogram  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensei shelf monitoring pipeline")
    parser.add_argument("--camera", help="limit to one camera id")
    parser.add_argument("--image", help="limit to one image filename (or stem)")
    parser.add_argument("--out", default=None, help="output JSON path")
    parser.add_argument("--dry-run", action="store_true",
                        help="no API calls: geometry + skeleton + fallbacks only")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides: dict = {}
    if args.dry_run:
        overrides["skip_api"] = True
    if args.samples:
        overrides["samples"] = args.samples
    if args.model:
        overrides["model"] = args.model

    cfg = Config.from_env(**overrides)
    if not cfg.skip_api and not cfg.api_key:
        print("ERROR: GEMINI_API_KEY is not set (or use --dry-run).", file=sys.stderr)
        return 2

    planogram = Planogram.load(cfg.planogram_path)
    print(f"Planogram: {len(planogram.cameras())} cameras, {len(planogram)} areas")
    images = list_images(cfg.images_dir)
    print(f"Images: {len(images)} ({images[0]} ... {images[-1]})")

    output = asyncio.run(run(cfg, planogram, camera=args.camera, image=args.image))

    out_path = args.out or cfg.output_path
    save_output(output, out_path)

    images_by_camera = {
        cam: ([args.image] if args.image else images)
        for cam in ([args.camera] if args.camera else planogram.cameras())
    }
    problems = validate_output(output, planogram, images_by_camera)
    areas = sum(len(img) for cam in output.values() for img in cam.values())
    print(f"\nWrote {out_path}: {areas} area predictions")
    if problems:
        print(f"VALIDATION PROBLEMS ({len(problems)}):")
        for problem in problems[:20]:
            print(f"  - {problem}")
        return 1
    print("Validation: OK (every planogram area present, values well-formed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
