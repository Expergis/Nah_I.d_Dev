"""A/B compare multiple prediction files (backends) against the audit labels.

    python scripts/ab_compare.py --labels audit/labels.json \
        --pred local=outputs/pred_local.json \
        --pred api=outputs/pred_api.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_report import print_report, score_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare backends on the audit set")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pred", action="append", required=True,
                        metavar="NAME=PATH", help="name=path pairs (repeatable)")
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    stats: list[tuple[str, dict]] = []
    for spec in args.pred:
        if "=" not in spec:
            parser.error(f"--pred expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        pred = json.loads(Path(path).read_text(encoding="utf-8"))
        stats.append((name, score_predictions(pred, labels)))

    for name, s in stats:
        print_report(name, s)

    scored = [(n, s) for n, s in stats if s["total"] > 0]
    if len(scored) > 1:
        best_name, best = max(scored, key=lambda ns: ns[1]["points"])
        print(f"\nWINNER: {best_name} ({best['points']:+.0f} pts)")
    return 0 if any(s["total"] for _, s in stats) else 1


if __name__ == "__main__":
    raise SystemExit(main())
