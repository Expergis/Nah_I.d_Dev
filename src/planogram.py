"""Parse and represent the planogram structure for Continente Bom Dia S. Romão."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PlanogramArea:
    """A single product area within a camera's field of view."""
    area_id: str
    product_id: str
    product_name: str
    # 4 corners [TL, TR, BR, BL] as (x, y) coordinates
    coordinates: List[Tuple[int, int]]

    @property
    def is_loose(self) -> bool:
        """Names ending in 'KG' indicate produce sold loose/by weight."""
        return self.product_name.strip().upper().endswith("KG")

    @property
    def has_negative_coords(self) -> bool:
        """True if the polygon spills outside the image boundary."""
        return any(x < 0 or y < 0 for x, y in self.coordinates)


@dataclass
class CameraConfig:
    """All planogram areas associated with one specific camera."""
    camera_id: str
    areas: List[PlanogramArea] = field(default_factory=list)

    @property
    def product_names(self) -> List[str]:
        """Unique product names expected by this camera."""
        return sorted({area.product_name for area in self.areas})


class Planogram:
    """Full planogram representation for the Fruits & Vegetables section."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Planogram file not found at: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.cameras: Dict[str, CameraConfig] = {}
        section = raw.get("Frutas e Legumes", {}).get("cameras", {})

        for cam_id, cam_data in section.items():
            areas: List[PlanogramArea] = []
            for p in cam_data.get("products", []):
                coords = p.get("image_coordinates", {})
                area = PlanogramArea(
                    area_id=str(p["area_id"]),
                    product_id=str(p["product_id"]),
                    product_name=str(p["product_name"]),
                    coordinates=[
                        (int(coords["top_left"][0]), int(coords["top_left"][1])),
                        (int(coords["top_right"][0]), int(coords["top_right"][1])),
                        (int(coords["bottom_right"][0]), int(coords["bottom_right"][1])),
                        (int(coords["bottom_left"][0]), int(coords["bottom_left"][1])),
                    ],
                )
                areas.append(area)
            self.cameras[str(cam_id)] = CameraConfig(camera_id=str(cam_id), areas=areas)

    @property
    def all_product_names(self) -> List[str]:
        """All unique product names across all cameras."""
        names = set()
        for cam in self.cameras.values():
            for area in cam.areas:
                names.add(area.product_name)
        return sorted(names)

    @property
    def total_areas(self) -> int:
        """Total planogram areas across all cameras."""
        return sum(len(cam.areas) for cam in self.cameras.values())

    def get_camera(self, camera_id: str) -> CameraConfig:
        cam_id = str(camera_id)
        if cam_id not in self.cameras:
            raise KeyError(f"Camera ID '{cam_id}' not found in planogram.")
        return self.cameras[cam_id]

    def get_area(self, camera_id: str, area_id: str) -> PlanogramArea:
        cam = self.get_camera(camera_id)
        for area in cam.areas:
            if area.area_id == str(area_id):
                return area
        raise KeyError(f"Area ID '{area_id}' not found on camera '{camera_id}'.")
