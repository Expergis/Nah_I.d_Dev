"""Build contact sheets of warped planogram crops for manual audit labelling.

For a camera + image, produces a labeled grid of every area's rectified crop.
The team eyeballs the sheet and writes the audit labels (see audit/README.md).

    python scripts/contact_sheet.py --camera 6215 --image 1751052241.jpg
    python scripts/contact_sheet.py --camera 6215 --all-images --out sheets/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from sensei.config import Config  # noqa: E402
from sensei.geometry import warp_area  # noqa: E402
from sensei.pipeline import list_images_per_camera, resolve_image_path  # noqa: E402
from sensei.planogram import Planogram  # noqa: E402

TILE_W, TILE_H = 320, 224
LABEL_H = 24


def _tile(crop: "cv2.MatLike", label: str, expected: str) -> "cv2.MatLike":
    resized = cv2.resize(crop, (TILE_W, TILE_H))
    header = cv2.makeHeader if hasattr(cv2, "makeHeader") else None  # noqa: F841
    canvas = cv2.copyMakeBorder(
        resized, LABEL_H, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)
    )
    cv2.putText(
        canvas, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    short = expected if len(expected) <= 30 else expected[:28] + ".."
    cv2.putText(
        canvas, short, (6, TILE_H + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Contact sheet of planogram areas")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--image", help="single image (name or stem)")
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--out", default=None, help="output dir (default: sheets/)")
    args = parser.parse_args()

    cfg = Config.from_env()
    planogram = Planogram.load(cfg.planogram_path)
    areas = planogram.areas_for(args.camera)
    if not areas:
        print(f"no areas for camera {args.camera}", file=sys.stderr)
        return 1

    images = list_images_per_camera(cfg.images_dir, [args.camera]).get(args.camera, [])
    if not images:
        print(f"no images found for camera {args.camera}", file=sys.stderr)
        return 1

    if args.all_images:
        selected = images
    elif args.image:
        selected = [n for n in images if n == args.image or Path(n).stem == args.image]
        if not selected:
            print(f"image not found: {args.image}", file=sys.stderr)
            return 1
    else:
        selected = [images[0]]

    out_dir = Path(args.out or "sheets")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = (len(areas) + args.cols - 1) // args.cols
    for name in selected:
        image = cv2.imread(str(resolve_image_path(cfg.images_dir, args.camera, name)), cv2.IMREAD_COLOR)
        if image is None:
            print(f"could not read {name}", file=sys.stderr)
            continue
        tiles = [
            _tile(
                warp_area(image, area.corners, 640, 448)[0],
                area.area_id,
                area.product_name,
            )
            for area in areas
        ]
        # pad the last row
        while len(tiles) % args.cols != 0:
            blank = cv2.copyMakeBorder(
                cv2.resize(warp_area(image, areas[0].corners, 640, 448)[0], (TILE_W, TILE_H)),
                LABEL_H, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25),
            )
            cv2.putText(blank, "—", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            tiles.append(blank)

        grid_rows = []
        for r in range(0, len(tiles), args.cols):
            grid_rows.append(cv2.hconcat(tiles[r : r + args.cols]))
        sheet = cv2.vconcat(grid_rows)
        out_path = out_dir / f"sheet_{args.camera}_{Path(name).stem}.png"
        cv2.imwrite(str(out_path), sheet)
        print(f"wrote {out_path} ({len(areas)} areas, {rows}x{args.cols})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
