"""Quick dataset sanity checks and summary.

    python scripts/explore_data.py
    python scripts/explore_data.py --camera 6215
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensei.config import Config  # noqa: E402
from sensei.pipeline import list_images_per_camera  # noqa: E402
from sensei.planogram import Planogram  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset exploration and sanity checks")
    parser.add_argument("--camera", help="limit to one camera id")
    args = parser.parse_args()

    cfg = Config.from_env()
    pg = Planogram.load(cfg.planogram_path)

    print("=" * 60)
    print(f"Planogram: {len(pg.cameras())} cameras, {len(pg)} total areas")
    print("=" * 60)

    cameras = [args.camera] if args.camera else pg.cameras()
    cameras = [c for c in cameras if c in pg.cameras()]

    images_by_cam = list_images_per_camera(cfg.images_dir, cameras)
    total_images = sum(len(imgs) for imgs in images_by_cam.values())
    total_areas = sum(len(pg.area_ids(c)) * len(images_by_cam.get(c, [])) for c in cameras)

    print(f"\nTotal: {total_images} images, {total_areas} area predictions to make\n")

    for cam in sorted(cameras):
        areas = pg.areas_for(cam)
        images = images_by_cam.get(cam, [])
        products = sorted({a.product_name for a in areas})

        print(f"Camera {cam}")
        print(f"  Areas:  {len(areas)}")
        print(f"  Images: {len(images)}")
        if images:
            print(f"  Range:  {images[0]} ... {images[-1]}")
        print(f"  Products: {', '.join(p[:30] for p in products[:5])}"
              + (f" (+{len(products)-5} more)" if len(products) > 5 else ""))

        # Coordinate sanity check
        xs = [c[0] for a in areas for c in a.corners]
        ys = [c[1] for a in areas for c in a.corners]
        neg_x = sum(1 for x in xs if x < 0)
        neg_y = sum(1 for y in ys if y < 0)
        if neg_x or neg_y:
            print(f"  [!] Off-image coords: {neg_x} negative x, {neg_y} negative y")
        print(f"  Coord range: x=[{min(xs)},{max(xs)}] y=[{min(ys)},{max(ys)}]")
        print()

    print("=" * 60)
    print("OK" if cameras else "No cameras found")
    return 0 if cameras else 1


if __name__ == "__main__":
    raise SystemExit(main())
