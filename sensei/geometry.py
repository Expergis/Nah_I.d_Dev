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


def fruit_fraction(crop: Array, mask: Array | None = None) -> float:
    """Fraction of (valid) shelf pixels that look like produce.

    "Produce-like" = saturated (S>90) and bright (V>90) in HSV. Bare dark
    shelves and gray/black padding score near 0; yellow/orange/red/green
    produce scores high. When ``mask`` is given, only valid in-image pixels
    are counted so off-image padding can't skew the fraction. This is a cheap
    proxy for "is there actually produce here?" and doubles as a safety-net
    against the VLM re-priming an empty shelf as the expected product.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    fruit = (hsv[..., 1] > 90) & (hsv[..., 2] > 90)
    if mask is not None:
        m = mask.astype(bool)
        if not m.any():
            return 0.0
        return float(fruit[m].mean())
    return float(fruit.mean())


def shelf_features(crop: Array) -> dict:
    """Cheap colour/texture signals for 'produce present'.

    Returns a dict with:
    - dark:    fraction of pixels with V < 100 (bare dark shelves score high)
    - fruit:   fraction with S > 90 AND V > 90 (saturated, bright produce)
    - v_std:   std of grayscale (texture proxy; log-only, not a gate)
    - texture: Canny edge density (log-only)

    Conservative override rule (speed shortcut, NOT a classifier):
    fire ONLY when dark > 0.60 AND fruit < 0.06.
    This avoids dark-produce misfires (aubergine, black grapes, mushrooms).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    dark = float((hsv[..., 2] < 100).mean())
    fruit = float(((hsv[..., 1] > 90) & (hsv[..., 2] > 90)).mean())
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    v_std = float(gray.std())
    edges = cv2.Canny(gray, 50, 150)
    texture = float(edges.mean() / 255.0)
    return {
        "dark": round(dark, 4),
        "fruit": round(fruit, 4),
        "v_std": round(v_std, 1),
        "texture": round(texture, 4),
    }


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
