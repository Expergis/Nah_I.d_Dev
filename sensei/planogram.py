"""Load and index the planogram provided by Sensei.

Expected structure (top-level section key is not fixed, so we scan for it)::

    {
      "Frutas e Legumes": {
        "cameras": {
          "6215": {
            "products": [
              {
                "area_id": "6215_2",
                "product_id": "11908",
                "product_name": "BANANA CAT I CNT KG",
                "image_coordinates": {
                  "top_left": [299, 350],
                  "top_right": [464, 347],
                  "bottom_right": [479, 438],
                  "bottom_left": [331, 438]
                }
              }
            ]
          }
        }
      }
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class PlanogramArea:
    camera_id: str
    area_id: str
    product_id: str
    product_name: str
    corners: tuple  # 4 (x, y) tuples in TL, TR, BR, BL order


class Planogram:
    def __init__(self, cameras: dict[str, list[PlanogramArea]]):
        self._cameras = cameras

    @classmethod
    def load(cls, path: str | Path) -> "Planogram":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cameras_raw = None
        if isinstance(data, dict):
            for section in data.values():
                if isinstance(section, dict) and "cameras" in section:
                    cameras_raw = section["cameras"]
                    break
        if cameras_raw is None:
            raise ValueError(f"Could not find a 'cameras' section in {path}")

        cameras: dict[str, list[PlanogramArea]] = {}
        for cam_id, entry in cameras_raw.items():
            areas: list[PlanogramArea] = []
            for product in entry.get("products", []):
                coords = product["image_coordinates"]
                corners = tuple(
                    (int(coords[k][0]), int(coords[k][1])) for k in CORNERS
                )
                areas.append(
                    PlanogramArea(
                        camera_id=str(cam_id),
                        area_id=str(product["area_id"]),
                        product_id=str(product["product_id"]),
                        product_name=str(product["product_name"]),
                        corners=corners,
                    )
                )
            cameras[str(cam_id)] = areas
        return cls(cameras)

    def cameras(self) -> list[str]:
        return list(self._cameras)

    def areas_for(self, camera_id: str) -> list[PlanogramArea]:
        return self._cameras.get(str(camera_id), [])

    def area_ids(self, camera_id: str) -> list[str]:
        return [a.area_id for a in self.areas_for(camera_id)]

    def __len__(self) -> int:
        return sum(len(v) for v in self._cameras.values())
