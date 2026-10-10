"""Exact output format, skeleton and validation.

Output shape (identical to the Sensei annotations format):

    {
      "<camera_id>": {
        "<image>.jpg": {
          "<area_id>": {
            "planogram_compliant": true,
            "stock_level": "half",
            "stock_value": 0.5
          }
        }
      }
    }

A missing area scores -5 (wrong ID + over-estimate), so the skeleton is
pre-filled for EVERY planogram area before inference starts: an API failure
produces a documented conservative fallback, never a hole.
"""
from __future__ import annotations

import json
from pathlib import Path

from .decision import FALLBACK, COMPLIANCE_LABELS, STOCK_VALUES
from .planogram import Planogram


def make_skeleton(
    planogram: Planogram,
    images_by_camera: dict[str, list[str]],
) -> dict:
    output: dict = {}
    for camera_id, image_names in images_by_camera.items():
        area_ids = planogram.area_ids(camera_id)
        if not area_ids:
            continue
        output.setdefault(camera_id, {})
        for image_name in image_names:
            output[camera_id][image_name] = {
                area_id: dict(FALLBACK) for area_id in area_ids
            }
    return output


def validate_output(
    output: dict,
    planogram: Planogram,
    images_by_camera: dict[str, list[str]],
) -> list[str]:
    """Return a list of problems (empty list == valid)."""
    problems: list[str] = []
    for camera_id, image_names in images_by_camera.items():
        cam_out = output.get(camera_id, {})
        for image_name in image_names:
            img_out = cam_out.get(image_name)
            if img_out is None:
                problems.append(f"missing image block: {camera_id}/{image_name}")
                continue
            for area_id in planogram.area_ids(camera_id):
                entry = img_out.get(area_id)
                if entry is None:
                    problems.append(
                        f"missing area: {camera_id}/{image_name}/{area_id}"
                    )
                    continue
                if entry.get("stock_level") not in STOCK_VALUES:
                    problems.append(
                        f"bad stock_level: {camera_id}/{image_name}/{area_id}"
                    )
                expected_value = STOCK_VALUES.get(entry.get("stock_level"))
                if expected_value is not None and entry.get("stock_value") != expected_value:
                    problems.append(
                        f"stock_value mismatch: {camera_id}/{image_name}/{area_id}"
                    )
                if not isinstance(entry.get("planogram_compliant"), bool):
                    problems.append(
                        f"bad planogram_compliant: {camera_id}/{image_name}/{area_id}"
                    )
    return problems


def save_output(output: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
