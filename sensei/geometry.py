"""Geometry utilities: perspective-corrected crops + visibility masks.

Coordinates come from ``planogram.json`` as quadrilaterals (perspective), may
be partially outside the image (negative or oversized values), and are ordered
top_left, top_right, bottom_right, bottom_left. We never sort them — sorting
can scramble a real quadrilateral.
"""
from __future__ import annotations

import cv2
import numpy as np

Array = np.ndarray


def warp_area(
    image: Array,
    corners: tuple | list,
    out_w: int = 640,
    out_h: int = 448,
) -> tuple[Array, Array, float]:
    """Warp a planogram quadrilateral into a rectified crop.

    Returns:
        crop: the perspective-corrected area (BGR), black where off-image.
        mask: uint8 mask, 1 where the pixel came from the source image.
        visible_fraction: fraction of the output covered by valid source pixels.
    """
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dst = np.float32(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)

    crop = cv2.warpPerspective(
        image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    height, width = image.shape[:2]
    source_valid = np.ones((height, width), dtype=np.uint8)
    mask = cv2.warpPerspective(
        source_valid,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    visible_fraction = float(mask.mean())
    return crop, mask, visible_fraction


def context_crop(
    image: Array,
    corners: tuple | list,
    padding: float = 0.35,
) -> Array | None:
    """Wider view around the area (for spatial context), clipped to the image.

    Returns None when the visible box is degenerate.
    """
    height, width = image.shape[:2]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = max(x1 - x0, y1 - y0, 1)
    x0 -= padding * span
    x1 += padding * span
    y0 -= padding * span
    y1 += padding * span

    x0c, y0c = max(0, int(x0)), max(0, int(y0))
    x1c, y1c = min(width, int(x1)), min(height, int(y1))
    if x1c - x0c < 8 or y1c - y0c < 8:
        return None
    return image[y0c:y1c, x0c:x1c]


def encode_png(image: Array) -> bytes:
    """Encode a BGR image as JPEG bytes for the API payload (smaller than PNG)."""
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    return buffer.tobytes()
