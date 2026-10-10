"""Scoring helpers: predictions vs manual audit labels (official rules)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensei.decision import expected_stock_scores  # noqa: E402

ORDER = {"empty": 0, "half": 1, "full": 2}


def score_entry(predicted: dict, truth: dict) -> tuple[float, bool, bool]:
    """Score one area against a manual label using the official rules."""
    ok_id = predicted.get("planogram_compliant") == bool(truth.get("compliant", True))
    points = 1 if ok_id else -2

    p_level = predicted.get("stock_level")
    t_level = truth.get("stock", "empty")
    if p_level == t_level:
        points += 1
    elif ORDER.get(p_level, -1) < ORDER.get(t_level, 3):
        points -= 2  # below real stock
    else:
        points -= 3  # above real stock
    return points, ok_id, p_level == t_level


def score_predictions(pred: dict, labels: dict) -> dict:
    """Score a full predictions dict against audit labels. Returns stats."""
    total = 0
    id_hits = 0
    stock_hits = 0
    points = 0.0
    errors: dict[str, int] = {}

    for cam, images in labels.items():
        for img, areas in images.items():
            for area_id, truth in areas.items():
                entry = pred.get(cam, {}).get(img, {}).get(area_id)
                if entry is None:
                    errors["missing"] = errors.get("missing", 0) + 1
                    continue
                pts, ok_id, ok_stock = score_entry(entry, truth)
                total += 1
                points += pts
                id_hits += int(ok_id)
                stock_hits += int(ok_stock)
                if not ok_id:
                    errors["id"] = errors.get("id", 0) + 1
                if not ok_stock:
                    key = f"stock:{truth.get('stock')}->{entry.get('stock_level')}"
                    errors[key] = errors.get(key, 0) + 1

    return {
        "total": total,
        "id_hits": id_hits,
        "stock_hits": stock_hits,
        "points": points,
        "max_points": 2 * total,
        "errors": errors,
    }


def print_report(name: str, stats: dict) -> None:
    if stats["total"] == 0:
        print(f"[{name}] No overlapping areas between predictions and labels.")
        return
    print(f"[{name}] areas={stats['total']}  "
          f"id={stats['id_hits']}/{stats['total']} ({stats['id_hits'] / stats['total']:.0%})  "
          f"stock={stats['stock_hits']}/{stats['total']} ({stats['stock_hits'] / stats['total']:.0%})  "
          f"pts={stats['points']:+.0f}/{stats['max_points']}")
    if stats["errors"]:
        print(f"[{name}] errors: " + ", ".join(f"{k}={v}" for k, v in sorted(stats["errors"].items())))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit report: predictions vs manual labels")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    stats = score_predictions(pred, labels)
    print_report("audit", stats)

    ev = expected_stock_scores(1 / 3, 1 / 3, 1 / 3)
    best = max(ev, key=ev.get)
    print(f"\nEV rule sanity (uniform prior -> best call = '{best}'): {ev}")
    return 0 if stats["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
