"""Format pipeline predictions to strictly match the submission schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


# Exact mapping defined in Sensei challenge documentation
STOCK_MAPPING: Dict[str, float] = {
    "empty": 0.0,
    "half": 0.5,
    "full": 1.0,
}


def normalize_stock(stock_level: str | float) -> tuple[str, float]:
    """Ensures consistent lowercase string and numeric pair."""
    if isinstance(stock_level, (int, float)):
        val = float(stock_level)
        if val <= 0.25:
            return "empty", 0.0
        elif val <= 0.75:
            return "half", 0.5
        else:
            return "full", 1.0

    cleaned = str(stock_level).lower().strip()
    if cleaned in STOCK_MAPPING:
        return cleaned, STOCK_MAPPING[cleaned]

    # Conservative safe fallback
    return "half", 0.5


def format_area_record(planogram_compliant: bool, stock_level: str | float) -> Dict[str, Any]:
    """Outputs the exact schema block required per planogram area."""
    str_level, num_level = normalize_stock(stock_level)
    return {
        "planogram_compliant": bool(planogram_compliant),
        "stock_level": str_level,
        "stock_value": num_level,
    }


def build_submission_dict(raw_results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Transforms raw dictionary into the nested submission hierarchy:
    {
      "<camera_id>": {
        "<timestamp.jpg>": {
          "<area_id>": {
            "planogram_compliant": true/false,
            "stock_level": "empty"|"half"|"full",
            "stock_value": 0.0|0.5|1.0
          }
        }
      }
    }
    """
    submission: Dict[str, Any] = {}
    for camera_id, image_group in raw_results.items():
        submission[str(camera_id)] = {}
        for image_name, area_records in image_group.items():
            submission[str(camera_id)][str(image_name)] = {}
            for area_id, values in area_records.items():
                submission[str(camera_id)][str(image_name)][str(area_id)] = format_area_record(
                    planogram_compliant=values.get("planogram_compliant", True),
                    stock_level=values.get("stock_level", "half"),
                )
    return submission


def save_predictions(predictions: Dict[str, Any], filepath: str | Path) -> None:
    """Safely serializes predictions to the specified destination path."""
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
