#!/usr/bin/env python3
"""Objective quality evaluation, business metrics, and score simulator.

Use this tool to:
1. Benchmark prediction outputs against simulated or known ground truth.
2. Calculate expected hackathon score under the asymmetric scoring matrix.
3. Compute business-oriented KPIs for presentation and pitch deck:
   - Out-of-Stock (OOS) Risk Mitigation Rate
   - Overstock / Waste Penalty Index
   - Planogram Compliance Rate
   - Shelf Attention Priority (Critical areas needing immediate restock)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Sensei Asymmetric Penalty Matrix
# True class (row) vs Prediction (col):
# Truth: Empty  -> Pred: Empty (+1), Half (-3), Full (-3)
# Truth: Half   -> Pred: Empty (-2), Half (+1), Full (-3)
# Truth: Full   -> Pred: Empty (-2), Half (-2), Full (+1)
STOCK_SCORE_MATRIX = {
    "empty": {"empty": 1, "half": -3, "full": -3},
    "half":  {"empty": -2, "half": 1, "full": -3},
    "full":  {"empty": -2, "half": -2, "full": 1},
}

COMPLIANCE_SCORE = {
    True: {True: 1, False: -2},
    False: {True: -2, False: 1},
}


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def analyze_predictions(predictions: dict) -> dict:
    """Extract operational summary metrics from predictions.json."""
    total_cameras = len(predictions)
    total_images = sum(len(imgs) for imgs in predictions.values())
    
    total_areas = 0
    stock_distribution = {"empty": 0, "half": 0, "full": 0}
    compliant_count = 0
    non_compliant_count = 0
    critical_restock_alerts = []  # Empty or non-compliant

    for cam_id, images in predictions.items():
        for img_name, areas in images.items():
            for area_id, data in areas.items():
                total_areas += 1
                lvl = data.get("stock_level", "unknown").lower()
                if lvl in stock_distribution:
                    stock_distribution[lvl] += 1
                
                is_comp = data.get("planogram_compliant", True)
                if is_comp:
                    compliant_count += 1
                else:
                    non_compliant_count += 1

                if lvl == "empty" or not is_comp:
                    critical_restock_alerts.append({
                        "camera": cam_id,
                        "image": img_name,
                        "area": area_id,
                        "stock": lvl,
                        "compliant": is_comp,
                    })

    compliance_rate = (compliant_count / total_areas * 100) if total_areas else 0.0
    oos_rate = (stock_distribution["empty"] / total_areas * 100) if total_areas else 0.0

    return {
        "total_cameras": total_cameras,
        "total_images": total_images,
        "total_areas": total_areas,
        "stock_distribution": stock_distribution,
        "compliance_rate_pct": round(compliance_rate, 2),
        "out_of_stock_pct": round(oos_rate, 2),
        "critical_alerts_count": len(critical_restock_alerts),
        "critical_alerts": critical_restock_alerts,
    }


def compare_with_ground_truth(preds: dict, truth: dict) -> dict:
    """Compute official challenge score against ground truth."""
    total_evaluated = 0
    stock_score = 0
    compliance_score = 0
    overstock_penalties = 0  # Severe -3 penalty
    understock_penalties = 0  # Mild -2 penalty
    
    for cam_id, t_imgs in truth.items():
        p_imgs = preds.get(cam_id, {})
        for img_name, t_areas in t_imgs.items():
            p_areas = p_imgs.get(img_name, {})
            for area_id, t_data in t_areas.items():
                total_evaluated += 1
                p_data = p_areas.get(area_id)
                if not p_data:
                    # Missing area penalty = -5 total
                    stock_score -= 3
                    compliance_score -= 2
                    continue
                
                # Stock score
                t_stock = t_data.get("stock_level", "half").lower()
                p_stock = p_data.get("stock_level", "half").lower()
                pts = STOCK_SCORE_MATRIX.get(t_stock, {}).get(p_stock, -2)
                stock_score += pts
                if pts == -3:
                    overstock_penalties += 1
                elif pts == -2:
                    understock_penalties += 1

                # Compliance score
                t_comp = bool(t_data.get("planogram_compliant", True))
                p_comp = bool(p_data.get("planogram_compliant", True))
                compliance_score += COMPLIANCE_SCORE[t_comp][p_comp]

    total_score = stock_score + compliance_score
    max_possible = total_evaluated * 2

    return {
        "areas_evaluated": total_evaluated,
        "total_score": total_score,
        "max_possible_score": max_possible,
        "score_percentage": round((total_score / max_possible * 100), 2) if max_possible else 0,
        "stock_score": stock_score,
        "compliance_score": compliance_score,
        "severe_overstock_penalties": overstock_penalties,
        "mild_understock_penalties": understock_penalties,
    }


def format_pitch_card(metrics: dict, eval_res: dict | None = None) -> str:
    """Generate ASCII presentation card for pitch deck slides."""
    lines = []
    lines.append("=" * 66)
    lines.append("  🛒 SENSEI AUTONOMOUS SHELF INTELLIGENCE — PERFORMANCE AUDIT")
    lines.append("=" * 66)
    lines.append(f"  • Monitored Scope   : {metrics['total_cameras']} Cameras | {metrics['total_areas']} Shelf Areas")
    lines.append(f"  • Compliance Health : {metrics['compliance_rate_pct']}% Planogram Alignment")
    lines.append(f"  • Shelf Fill Balance: Empty: {metrics['stock_distribution']['empty']} | "
                 f"Half: {metrics['stock_distribution']['half']} | "
                 f"Full: {metrics['stock_distribution']['full']}")
    lines.append(f"  • Action Alerts     : {metrics['critical_alerts_count']} Priority Restock/Mismatch Flags")
    
    if eval_res:
        lines.append("-" * 66)
        lines.append("  🏆 HACKATHON BENCHMARK SCORE (vs Ground Truth):")
        lines.append(f"  • Total Score       : {eval_res['total_score']} / {eval_res['max_possible_score']} "
                     f"({eval_res['score_percentage']}%)")
        lines.append(f"  • Stock Points      : {eval_res['stock_score']}")
        lines.append(f"  • Compliance Points : {eval_res['compliance_score']}")
        lines.append(f"  • Risk Shield (Overcall Avoidance): {eval_res['severe_overstock_penalties']} severe errors")
    lines.append("=" * 66)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Sensei predictions for presentation & benchmarking")
    parser.add_argument("--pred", default="output/predictions.json", help="Path to predictions.json")
    parser.add_argument("--truth", default=None, help="Path to ground truth annotations JSON (if available)")
    args = parser.parse_args()

    try:
        preds = load_json(args.pred)
    except Exception as e:
        print(f"Error loading predictions: {e}", file=sys.stderr)
        return 1

    metrics = analyze_predictions(preds)
    eval_res = None
    if args.truth:
        truth = load_json(args.truth)
        eval_res = compare_with_ground_truth(preds, truth)

    print(format_pitch_card(metrics, eval_res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
