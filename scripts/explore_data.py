#!/usr/bin/env python3
"""Dataset diagnostic and exploration utility for hackathon images and planograms."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, PLANOGRAM_PATH
from src.planogram import Planogram


def main() -> None:
    print("=" * 60)
    print("📊 DATASET & PLANOGRAM DIAGNOSTICS")
    print("=" * 60)

    # 1. Inspect Planogram
    if PLANOGRAM_PATH.exists():
        plano = Planogram(PLANOGRAM_PATH)
        print(f"Planogram: {PLANOGRAM_PATH}")
        print(f"Cameras: {len(plano.cameras)}")
        print(f"Total Areas: {plano.total_areas}")
        print(f"Unique Products: {len(plano.all_product_names)}")
        print("\nSample Products:")
        for name in plano.all_product_names[:10]:
            print(f"  - {name}")
    else:
        print(f"⚠️ Planogram file not found at: {PLANOGRAM_PATH}")

    # 2. Inspect Image Directories
    print("-" * 60)
    print(f"Data Directory: {DATA_DIR}")
    for split in ["train", "test", "validation"]:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            imgs = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.png"))
            print(f"  • {split:<10}: {len(imgs)} image files")
        else:
            print(f"  • {split:<10}: (directory not yet populated)")
    print("=" * 60)


if __name__ == "__main__":
    main()
