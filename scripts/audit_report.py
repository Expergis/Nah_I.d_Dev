"""Compare pipeline predictions against the team's manual audit labels.

Audit label file format (see audit/README.md):

    {
      "6215": {
        "1751052241.jpg": {
          "6215_2": {"compliant": true, "stock": "half", "occluded": false}
        }
      }
    }

    python scripts/audit_report.py --pred outputs/predictions.json --labels audit/labels.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensei.decision import expected_stock_scores  # noqa: E402


def score_entry(predicted: dict, truth: dict) -> tuple[float, bool, bool]:
    """Score one area against a manual label using the official rules."""
    ok_id = predicted.get("planogram_compliant") == bool(truth.get("compliant", True))
    points = 1 if ok_id else -2

    p_level = predicted.get("stock_level")
    t_level = truth.get("stock", "empty")
    order = {"empty": 0, "half": 1, "full": 2}
    if p_level == t_level:
        points += 1
    elif order.get(p_level, -1) < order.get(t_level, 3):
        points -= 2  # below real stock
    else:
        points -= 3  # above real stock
    return points, ok_id, p_level == t_level


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit report: predictions vs manual labels")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    total = 0
    id_hits = 0
    stock_hits = 0
    points = 0.0
    by_error: dict[str, int] = {}

    for cam, images in labels.items():
        for img, areas in images.items():
            for area_id, truth in areas.items():
                entry = (
                    pred.get(cam, {}).get(img, {}).get(area_id)
                )
                if entry is None:
                    by_error["missing"] = by_error.get("missing", 0) + 1
                    continue
                pts, ok_id, ok_stock = score_entry(entry, truth)
                total += 1
                points += pts
                id_hits += int(ok_id)
                stock_hits += int(ok_stock)
                if not ok_id:
                    by_error["id"] = by_error.get("id", 0) + 1
                if not ok_stock:
                    by_error[f"stock:{truth.get('stock')}->pred:{entry.get('stock_level')}"] = (
                        by_error.get(f"stock:{truth.get('stock')}->pred:{entry.get('stock_level')}", 0) + 1
                    )

    if total == 0:
        print("No overlapping areas between predictions and labels.")
        return 1

    print(f"Areas scored:      {total}")
    print(f"ID accuracy:       {id_hits}/{total} ({id_hits / total:.0%})")
    print(f"Stock accuracy:    {stock_hits}/{total} ({stock_hits / total:.0%})")
    print(f"Official-style pts:{points:+.0f} (max {2 * total})")
    if by_error:
        print("Error breakdown:")
        for key in sorted(by_error):
            print(f"  {key}: {by_error[key]}")

    # decision-rule sanity: show EV table
    ev = expected_stock_scores(1 / 3, 1 / 3, 1 / 3)
    best = max(ev, key=ev.get)
    print(f"\nEV rule sanity (uniform prior -> best call = '{best}'): {ev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
