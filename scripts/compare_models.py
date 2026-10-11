"""Compare vision models on our hand-labelled test set (official scoring).

Runs the unchanged run_local.py once per model on eval/eval_images.json
(last frame of each camera, 152 areas), then scores against
eval/labels.json with the challenge points:
  identification: +1 right / -2 wrong
  stock:          +1 right / -2 below truth / -3 above truth

    python scripts/compare_models.py                      # default model list
    python scripts/compare_models.py --models qwen3-vl:4b-instruct qwen3-vl:8b-instruct
    python scripts/compare_models.py --score-only         # rescore saved outputs

Labels were made by eye from the crops (see eval/labels.json, "sure": false
marks borderline half/full calls), so treat scores as relative, not as the
validation score.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
ORDER = {"empty": 0, "half": 1, "full": 2}
DEFAULT_MODELS = [
    "qwen3-vl:2b-instruct",
    "qwen3-vl:4b-instruct",
    "qwen3-vl:8b-instruct",
    "qwen3-vl:30b-a3b-instruct",
]


def run_model(model: str, images: dict, a) -> tuple[dict, dict, float]:
    tag = model.replace(":", "_").replace("/", "_")
    preds, debug, wall = {}, {}, 0.0
    for cam, img in images.items():
        out = EVAL / "runs" / tag / f"{cam}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "run_local.py"), "--model", model,
               "--camera", cam, "--image", img, "--out", str(out),
               "--num-ctx", str(a.num_ctx), "--images-dir", str(a.images_dir),
               "--planogram", str(ROOT / "planogram.json"), "--host", a.host]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall += time.time() - t0
        if r.returncode not in (0, 1) or not out.exists():
            print(r.stdout[-1500:], r.stderr[-1500:])
            raise SystemExit(f"run_local failed for {model} {cam} (exit {r.returncode})")
        if "had no model answer" in r.stdout:
            print(f"  ! {model} {cam}: some areas fell back (no model answer)")
        preds.update(json.loads(out.read_text(encoding="utf-8")))
        debug.update(json.loads(out.with_suffix(".debug.json").read_text(encoding="utf-8")))
        print(f"  {model:28s} {cam}: {time.time() - t0:5.1f}s", flush=True)
    (EVAL / "runs" / tag / "_wall.json").write_text(json.dumps({"wall_s": wall}))
    return preds, debug, wall


def score(preds: dict, labels: dict, sure_only: bool = False) -> dict:
    s = {"n": 0, "pts": 0, "id_ok": 0, "stock_ok": 0, "below": 0, "above": 0,
         "false_mismatch": 0, "missed_mismatch": 0, "missing": 0}
    for cam, imgs in labels.items():
        for img, areas in imgs.items():
            for aid, t in areas.items():
                if sure_only and not t["sure"]:
                    continue
                p = preds.get(cam, {}).get(img, {}).get(aid)
                s["n"] += 1
                if p is None:
                    s["missing"] += 1
                    s["pts"] -= 5
                    continue
                if p["planogram_compliant"] == t["compliant"]:
                    s["pts"] += 1; s["id_ok"] += 1
                else:
                    s["pts"] -= 2
                    s["false_mismatch" if t["compliant"] else "missed_mismatch"] += 1
                pl, tl = ORDER[p["stock_level"]], ORDER[t["stock"]]
                if pl == tl:
                    s["pts"] += 1; s["stock_ok"] += 1
                elif pl < tl:
                    s["pts"] -= 2; s["below"] += 1
                else:
                    s["pts"] -= 3; s["above"] += 1
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--images-dir", default=str(ROOT / "data" / "Frutas e Legumes (train & test)"))
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=3072)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()

    images = json.loads((EVAL / "eval_images.json").read_text(encoding="utf-8"))
    labels = json.loads((EVAL / "labels.json").read_text(encoding="utf-8"))
    rows = []
    for model in a.models:
        tag = model.replace(":", "_").replace("/", "_")
        if a.score_only:
            run_dir = EVAL / "runs" / tag
            if not run_dir.is_dir():
                continue
            preds, debug = {}, {}
            for cam in images:
                preds.update(json.loads((run_dir / f"{cam}.json").read_text(encoding="utf-8")))
                debug.update(json.loads((run_dir / f"{cam}.debug.json").read_text(encoding="utf-8")))
            wall = json.loads((run_dir / "_wall.json").read_text())["wall_s"]
        else:
            print(f"\n=== {model}")
            preds, debug, wall = run_model(model, images, a)
        calls = sum(len(d.get("samples", [])) for i in debug.values()
                    for areas in i.values() for d in areas.values())
        rows.append((model, score(preds, labels), score(preds, labels, True), wall, calls))

    print("\nScored with the official points on our hand labels (152 areas, 14 images).")
    print("'sure' = only the 121 areas where the label is not a borderline call.\n")
    hdr = (f"{'model':28s} {'pts/304':>8s} {'sure/242':>9s} {'id ok':>6s} {'stock ok':>8s} "
           f"{'below':>5s} {'above':>5s} {'falseNC':>7s} {'s/call':>6s} {'s/image':>7s}")
    print(hdr); print("-" * len(hdr))
    for model, s, ss, wall, calls in rows:
        print(f"{model:28s} {s['pts']:>8d} {ss['pts']:>9d} {s['id_ok']:>6d} {s['stock_ok']:>8d} "
              f"{s['below']:>5d} {s['above']:>5d} {s['false_mismatch']:>7d} "
              f"{wall / max(calls, 1):>6.1f} {wall / len(images):>7.1f}")
    print("\nfalseNC = flagged non-compliant but the label says compliant (staff walk there for nothing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
