"""End-to-end pipeline execution from raw dataset images to predictions.json."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from .config import DATA_DIR, OUTPUT_DIR, PLANOGRAM_PATH
from .image_processor import crop_all_areas, load_image
from .output_formatter import build_submission_dict, save_predictions
from .planogram import Planogram
from .vision_engine import analyze_area


def discover_camera_images(data_dir: Path, target_cameras: List[str]) -> Dict[str, List[Path]]:
    """Discovers images grouped by camera ID.
    Supports either:
    - nested: `data_dir/{camera_id}/{timestamp}.jpg`
    - flat: `data_dir/{timestamp}.jpg` (associated with cameras if flat)
    """
    image_map: Dict[str, List[Path]] = {cam: [] for cam in target_cameras}

    # Check for camera subdirectories
    has_subdirs = any((data_dir / cam).is_dir() for cam in target_cameras)

    if has_subdirs:
        for cam in target_cameras:
            cam_dir = data_dir / cam
            if cam_dir.is_dir():
                found = sorted(list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.png")))
                image_map[cam] = found
    else:
        # Flat directory structure
        all_images = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")))
        for cam in target_cameras:
            image_map[cam] = all_images

    return {cam: imgs for cam, imgs in image_map.items() if imgs}


def run_pipeline(
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    planogram_path: str | Path | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executes the complete vision pipeline:
    1. Parses Planogram
    2. Collects images
    3. Crops area perspective quadrilaterals
    4. Evaluates compliance & stock levels
    5. Formats and writes predictions.json
    """
    in_data_dir = Path(data_dir or DATA_DIR)
    out_file = Path(output_path or OUTPUT_DIR / "predictions.json")
    plano_file = Path(planogram_path or PLANOGRAM_PATH)

    if verbose:
        print(f"📖 Loading planogram from: {plano_file}")
    plano = Planogram(plano_file)

    cameras = list(plano.cameras.keys())
    if verbose:
        print(f"📷 Discovering camera images in: {in_data_dir}")
    camera_images = discover_camera_images(in_data_dir, cameras)

    total_images = sum(len(imgs) for imgs in camera_images.values())
    if verbose:
        print(f"   Found {total_images} images across {len(camera_images)} cameras.")

    raw_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    areas_processed = 0
    start_time = time.time()

    for cam_id, images in camera_images.items():
        cam_config = plano.get_camera(cam_id)
        raw_results[cam_id] = {}

        if verbose:
            print(f"➡️ Processing Camera {cam_id} ({len(cam_config.areas)} areas, {len(images)} images)")

        for img_path in images:
            img_name = img_path.name
            try:
                frame = load_image(img_path)
                crops = crop_all_areas(frame, cam_config)
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Could not process image {img_name}: {e}")
                continue

            area_evals: Dict[str, Any] = {}
            for area in cam_config.areas:
                crop = crops.get(area.area_id)
                if crop is None:
                    continue

                res = analyze_area(
                    crop=crop,
                    area_id=area.area_id,
                    expected_product=area.product_name,
                    candidate_products=cam_config.product_names,
                    image_name=img_name,
                )
                area_evals[area.area_id] = res
                areas_processed += 1

            raw_results[cam_id][img_name] = area_evals

    duration = time.time() - start_time
    if verbose:
        rate = areas_processed / duration if duration > 0 else 0
        print(f"\n⚡ Completed {areas_processed} areas across all images in {duration:.2f}s ({rate:.1f} areas/s)")

    # Format to submission spec and save
    predictions = build_submission_dict(raw_results)
    save_predictions(predictions, out_file)

    if verbose:
        print(f"💾 Predictions successfully written to: {out_file}")

    return predictions
