#!/usr/bin/env python3
"""Benchmark Snapshots & Model Comparison Engine.

Enables cross-model comparisons (e.g. Gemini 2.5 Flash vs Qwen2.5-VL 7B vs OpenRouter):
1. Takes a benchmark snapshot of predictions + timing + distribution metrics.
2. Compares multiple benchmark snapshots side-by-side.
3. Calculates agreement/consensus rate, score deltas, and risk exposure.
4. Generates terminal reports and exports comparison tables for pitch decks.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from collections import Counter
from pathlib import Path

BENCHMARK_DIR = Path("output/benchmarks")

PENALTY_MATRIX = {
    "empty": {"empty": 1, "half": -3, "full": -3},
    "half":  {"empty": -2, "half": 1, "full": -3},
    "full":  {"empty": -2, "half": -2, "full": 1},
}


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def calculate_entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return round(entropy, 3)


def snapshot_model(
    model_name: str,
    pred_path: str = "output/predictions.json",
    truth_path: str | None = None,
    backend_info: str = "llamacpp",
) -> Path:
    """Save a timestamped, comprehensive snapshot of a model's run."""
    preds = load_json(pred_path)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    total_areas = 0
    stock_counts = Counter()
    comp_counts = Counter()
    area_map = {}

    for cam_id, images in preds.items():
        for img_name, areas in images.items():
            for area_id, data in areas.items():
                total_areas += 1
                key = f"{cam_id}::{img_name}::{area_id}"
                stock = data.get("stock_level", "unknown").lower()
                comp = bool(data.get("planogram_compliant", True))
                stock_counts[stock] += 1
                comp_counts[comp] += 1
                area_map[key] = {
                    "stock_level": stock,
                    "stock_value": data.get("stock_value", 0.5),
                    "planogram_compliant": comp,
                }

    ground_truth_score = None
    if truth_path and Path(truth_path).exists():
        truth = load_json(truth_path)
        score = 0
        for cam_id, t_imgs in truth.items():
            for img_name, t_areas in t_imgs.items():
                for area_id, t_val in t_areas.items():
                    k = f"{cam_id}::{img_name}::{area_id}"
                    p_val = area_map.get(k)
                    if not p_val:
                        score -= 5
                        continue
                    t_st = t_val.get("stock_level", "half").lower()
                    p_st = p_val["stock_level"]
                    score += PENALTY_MATRIX.get(t_st, {}).get(p_st, -2)
                    if t_val.get("planogram_compliant") == p_val["planogram_compliant"]:
                        score += 1
                    else:
                        score -= 2
        ground_truth_score = score

    full_ratio = (stock_counts["full"] / total_areas * 100) if total_areas else 0.0

    snapshot = {
        "model_name": model_name,
        "backend": backend_info,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_areas": total_areas,
        "compliance_rate_pct": round(comp_counts[True] / total_areas * 100, 2) if total_areas else 0,
        "stock_distribution": dict(stock_counts),
        "entropy": calculate_entropy(stock_counts),
        "overcall_risk_full_pct": round(full_ratio, 2),
        "ground_truth_score": ground_truth_score,
        "area_predictions": area_map,
    }

    out_file = BENCHMARK_DIR / f"{model_name.lower().replace('/', '_').replace(':', '_')}.json"
    out_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"✅ Saved model snapshot to: {out_file}")
    return out_file


def compare_snapshots(file_paths: list[str | Path]):
    """Compare multiple model benchmark snapshots side by side."""
    snapshots = [load_json(p) for p in file_paths]
    if not snapshots:
        print("No snapshots to compare.")
        return

    print("\n" + "=" * 90)
    print("  🏆 SENSEI MODEL BENCHMARK & PERFORMANCE COMPARISON MATRIX")
    print("=" * 90)

    # 1. High-Level Performance Table
    header = f"  {'Model Name':<28} | {'Backend':<10} | {'Compliance':<10} | {'Entropy':<8} | {'% Full':<8} | {'Score'}"
    print(header)
    print("  " + "-" * 86)

    for s in snapshots:
        m_name = s.get("model_name", "Unknown")[:28]
        backend = s.get("backend", "local")[:10]
        comp = f"{s.get('compliance_rate_pct', 0)}%"
        ent = f"{s.get('entropy', 0):.2f}"
        full = f"{s.get('overcall_risk_full_pct', 0)}%"
        sc = f"{s.get('ground_truth_score')}" if s.get("ground_truth_score") is not None else "N/A"
        print(f"  {m_name:<28} | {backend:<10} | {comp:<10} | {ent:<8} | {full:<8} | {sc}")

    # 2. Pairwise Agreement / Consensus (if 2 or more models)
    if len(snapshots) >= 2:
        print("\n" + "-" * 90)
        print("  🤝 PAIRWISE MODEL CONSENSUS & DISAGREEMENT ANALYSIS")
        print("-" * 90)
        base = snapshots[0]
        for other in snapshots[1:]:
            b_areas = base["area_predictions"]
            o_areas = other["area_predictions"]
            common_keys = set(b_areas.keys()) & set(o_areas.keys())

            if not common_keys:
                print(f"  No overlapping areas between {base['model_name']} and {other['model_name']}")
                continue

            stock_agreements = sum(1 for k in common_keys if b_areas[k]["stock_level"] == o_areas[k]["stock_level"])
            comp_agreements = sum(1 for k in common_keys if b_areas[k]["planogram_compliant"] == o_areas[k]["planogram_compliant"])
            total = len(common_keys)

            stock_pct = round(stock_agreements / total * 100, 1)
            comp_pct = round(comp_agreements / total * 100, 1)

            print(f"  • {base['model_name']} vs {other['model_name']}:")
            print(f"    - Stock Prediction Consensus    : {stock_pct}% ({stock_agreements}/{total} areas match)")
            print(f"    - Compliance Consensus         : {comp_pct}% ({comp_agreements}/{total} areas match)")

            # Find notable disagreement samples
            disagreements = [
                k for k in common_keys
                if b_areas[k]["stock_level"] != o_areas[k]["stock_level"]
            ][:3]
            if disagreements:
                print("    - Sample Disagreements:")
                for k in disagreements:
                    clean_slot = k.split("::")[-1]
                    print(f"      Slot [{clean_slot}]: {base['model_name']}={b_areas[k]['stock_level'].upper()} vs "
                          f"{other['model_name']}={o_areas[k]['stock_level'].upper()}")

    print("=" * 90 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Save and compare model benchmark snapshots")
    subparsers = parser.add_subparsers(dest="cmd")

    # Command: save snapshot
    save_parser = subparsers.add_parser("save", help="Save current predictions as a named model snapshot")
    save_parser.add_argument("--name", required=True, help="Display name for this model run (e.g. 'qwen2.5-vl-7b' or 'gemini-2.5-flash')")
    save_parser.add_argument("--backend", default="llamacpp", help="Backend used (llamacpp, openrouter, gemini)")
    save_parser.add_argument("--pred", default="output/predictions.json", help="Path to predictions.json")
    save_parser.add_argument("--truth", default=None, help="Optional ground truth annotations for scoring")

    # Command: compare snapshots
    cmp_parser = subparsers.add_parser("compare", help="Compare multiple saved model snapshots")
    cmp_parser.add_argument("files", nargs="*", help="Paths to benchmark json files (if omitted, compares all in output/benchmarks/)")

    args = parser.parse_args()

    if args.cmd == "save":
        snapshot_model(
            model_name=args.name,
            pred_path=args.pred,
            truth_path=args.truth,
            backend_info=args.backend,
        )
    elif args.cmd == "compare":
        if args.files:
            paths = [Path(f) for f in args.files]
        else:
            paths = sorted(list(BENCHMARK_DIR.glob("*.json")))
        if not paths:
            print("No snapshots found in output/benchmarks/. Save some first with: python scripts/compare_models.py save --name <name>")
            return 1
        compare_snapshots(paths)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
