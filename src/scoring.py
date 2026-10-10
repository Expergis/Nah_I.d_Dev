"""Local scoring evaluation replicating the exact Sensei Hackathon scoring matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def score_submission(predictions: Dict[str, Any], ground_truth: Dict[str, Any], verbose: bool = True) -> Tuple[int, Dict[str, Any]]:
    """Calculates official score per the rules:
    - Product ID correct: +1 point
    - Product ID wrong: -2 points
    - Stock level correct: +1 point
    - Stock level below true stock: -2 points
    - Stock level above true stock: -3 points
    - Missing area: -2 (ID) + -3 (Stock) = -5 points

    Maximum score: 289 areas * 2 = 578 points.
    """
    stats = {
        "id_correct": 0,
        "id_wrong": 0,
        "stock_correct": 0,
        "stock_below": 0,
        "stock_above": 0,
        "missing_areas": 0,
        "total_evaluated_areas": 0,
    }

    detailed_log = []

    for cam_id, images in ground_truth.items():
        for img_name, areas in images.items():
            for area_id, truth in areas.items():
                stats["total_evaluated_areas"] += 1

                pred = (
                    predictions.get(str(cam_id), {})
                    .get(str(img_name), {})
                    .get(str(area_id))
                )

                if pred is None:
                    # Missing area counts as wrong ID (-2) and stock above real (-3)
                    stats["missing_areas"] += 1
                    stats["id_wrong"] += 1
                    stats["stock_above"] += 1
                    detailed_log.append({
                        "camera": cam_id,
                        "image": img_name,
                        "area_id": area_id,
                        "id_score": -2,
                        "stock_score": -3,
                        "status": "MISSING",
                    })
                    continue

                # 1. Product Identification
                pred_comp = bool(pred.get("planogram_compliant", True))
                true_comp = bool(truth.get("planogram_compliant", True))
                if pred_comp == true_comp:
                    stats["id_correct"] += 1
                    id_score = 1
                else:
                    stats["id_wrong"] += 1
                    id_score = -2

                # 2. Stock Level
                pred_val = float(pred.get("stock_value", 0.5))
                true_val = float(truth.get("stock_value", 0.5))

                if abs(pred_val - true_val) < 0.05:
                    stats["stock_correct"] += 1
                    stock_score = 1
                elif pred_val < true_val:
                    stats["stock_below"] += 1
                    stock_score = -2
                else:
                    stats["stock_above"] += 1
                    stock_score = -3

                detailed_log.append({
                    "camera": cam_id,
                    "image": img_name,
                    "area_id": area_id,
                    "id_score": id_score,
                    "stock_score": stock_score,
                    "status": "EVALUATED",
                })

    total_score = (
        (stats["id_correct"] * 1)
        + (stats["id_wrong"] * -2)
        + (stats["stock_correct"] * 1)
        + (stats["stock_below"] * -2)
        + (stats["stock_above"] * -3)
    )

    max_possible = stats["total_evaluated_areas"] * 2
    summary = {
        "total_score": total_score,
        "max_possible_score": max_possible,
        "percentage": (total_score / max_possible * 100) if max_possible > 0 else 0.0,
        "metrics": stats,
        "details": detailed_log,
    }

    if verbose:
        print("\n" + "=" * 55)
        print("🏆 SENSEI CHALLENGE SCORING EVALUATION")
        print("=" * 55)
        print(f"Total Score: {total_score} / {max_possible} ({summary['percentage']:.1f}%)")
        print("-" * 55)
        print(f"Planogram Compliance Identification:")
        print(f"  ✅ Correct:  {stats['id_correct']:>4} (+1 pt each  = +{stats['id_correct']})")
        print(f"  ❌ Wrong:    {stats['id_wrong']:>4} (-2 pts each = {stats['id_wrong'] * -2})")
        print(f"Stock Level Estimation:")
        print(f"  ✅ Correct:  {stats['stock_correct']:>4} (+1 pt each  = +{stats['stock_correct']})")
        print(f"  ⬇️  Below:    {stats['stock_below']:>4} (-2 pts each = {stats['stock_below'] * -2})")
        print(f"  ⬆️  Above:    {stats['stock_above']:>4} (-3 pts each = {stats['stock_above'] * -3})")
        if stats["missing_areas"] > 0:
            print(f"  ⚠️  Missing:  {stats['missing_areas']:>4} (-5 pts each = {stats['missing_areas'] * -5})")
        print("=" * 55 + "\n")

    return total_score, summary
