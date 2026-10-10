#!/usr/bin/env python3
"""CLI utility to score predictions.json against annotations.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import score_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth annotations")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.json")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations.json")

    args = parser.parse_args()

    pred_path = Path(args.predictions)
    anno_path = Path(args.annotations)

    if not pred_path.exists():
        print(f"Error: Predictions file not found: {pred_path}")
        sys.exit(1)
    if not anno_path.exists():
        print(f"Error: Annotations file not found: {anno_path}")
        sys.exit(1)

    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    with open(anno_path, "r", encoding="utf-8") as f:
        annos = json.load(f)

    score, _ = score_submission(preds, annos, verbose=True)


if __name__ == "__main__":
    main()
