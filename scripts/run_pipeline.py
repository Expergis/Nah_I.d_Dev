#!/usr/bin/env python3
"""CLI utility to execute the shelf monitoring pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sensei Produce Shelf Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing camera images")
    parser.add_argument("--output", type=str, default=None, help="Output destination for predictions.json")
    parser.add_argument("--planogram", type=str, default=None, help="Path to planogram.json")
    parser.add_argument("--quiet", action="store_true", help="Run with minimal output")

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        planogram_path=args.planogram,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
