"""Image processing, quadrilateral perspective cropping, and bounding visualizations."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .config import CROP_SIZE
from .planogram import CameraConfig, PlanogramArea


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk into a BGR numpy array."""
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not decode image at: {img_path}")
    return image


def crop_planogram_area(
    image: np.ndarray,
    area: PlanogramArea,
    output_size: Tuple[int, int] = CROP_SIZE,
) -> np.ndarray:
    """Extract a 4-point perspective warp for a planogram area.

    Handles coordinates extending beyond the image boundaries (negative coordinates)
    by computing bounding margins and padding the frame with edge reflections.
    """
    coords = [list(c) for c in area.coordinates]  # [TL, TR, BR, BL]

    # Detect negative coordinate offsets
    min_x = min(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    pad_left = max(0, -min_x)
    pad_top = max(0, -min_y)

    processed_image = image
    if pad_left > 0 or pad_top > 0:
        processed_image = cv2.copyMakeBorder(
            processed_image,
            int(pad_top),
            0,
            int(pad_left),
            0,
            cv2.BORDER_REFLECT,
        )
        coords = [[c[0] + pad_left, c[1] + pad_top] for c in coords]

    # Detect bottom/right overflows
    cur_h, cur_w = processed_image.shape[:2]
    max_x = max(c[0] for c in coords)
    max_y = max(c[1] for c in coords)

    pad_right = max(0, int(max_x) - cur_w + 1)
    pad_bottom = max(0, int(max_y) - cur_h + 1)
    if pad_right > 0 or pad_bottom > 0:
        processed_image = cv2.copyMakeBorder(
            processed_image,
            0,
            int(pad_bottom),
            0,
            int(pad_right),
            cv2.BORDER_REFLECT,
        )

    # Perspective Transform
    src_pts = np.array(coords, dtype=np.float32)
    out_w, out_h = output_size
    dst_pts = np.array(
        [[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(processed_image, matrix, (out_w, out_h))

    # Return as RGB for VLM ingestion
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def crop_all_areas(
    image: np.ndarray,
    camera_config: CameraConfig,
    output_size: Tuple[int, int] = CROP_SIZE,
) -> Dict[str, np.ndarray]:
    """Crops all designated planogram areas for a camera image.

    Guarantee: Every area_id always yields a valid ndarray.
    If an unexpected geometry occurs, a fallback black canvas is returned to
    prevent pipeline crashes and the resulting -5 missing area penalty.
    """
    crops: Dict[str, np.ndarray] = {}
    for area in camera_config.areas:
        try:
            crops[area.area_id] = crop_planogram_area(image, area, output_size)
        except Exception:
            crops[area.area_id] = np.zeros((*output_size, 3), dtype=np.uint8)
    return crops


def draw_areas_on_image(
    image: np.ndarray,
    camera_config: CameraConfig,
    results: Optional[Dict[str, dict]] = None,
) -> np.ndarray:
    """Overlays planogram boundaries with color codes:
    - Green: Compliant & Full
    - Yellow: Compliant & Half
    - Orange: Compliant & Empty
    - Red: Non-compliant (wrong product)
    - Gray: Unprocessed / default
    """
    canvas = image.copy()
    for area in camera_config.areas:
        pts = np.array(area.coordinates, dtype=np.int32)

        color = (180, 180, 180)  # Default BGR
        if results and area.area_id in results:
            res = results[area.area_id]
            is_compliant = res.get("planogram_compliant", True)
            stock_val = res.get("stock_value", 1.0)

            if not is_compliant:
                color = (0, 0, 255)       # Red: Misplaced product
            elif stock_val == 0.0:
                color = (0, 140, 255)     # Orange: Empty shelf
            elif stock_val == 0.5:
                color = (0, 230, 255)     # Yellow: Half full
            else:
                color = (0, 255, 0)       # Green: Compliant & full

        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        # Label centroid
        cx = int(np.mean([pt[0] for pt in area.coordinates]))
        cy = int(np.mean([pt[1] for pt in area.coordinates]))
        suffix = area.area_id.split("_")[-1]
        cv2.putText(
            canvas,
            suffix,
            (cx - 10, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas
