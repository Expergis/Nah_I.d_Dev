"""Store shelf observations in MongoDB and build the staff action list from them.

Only predictions are stored (no images), so "images never leave the store" holds.
Works with MongoDB Atlas (free M0) or a MongoDB server running in the store.

Setup (once):
    pip install pymongo python-dotenv
    # .env (git-ignored):  MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>/...

Usage:
    python scripts/mongo_sync.py load --pred eval/full_4b.json --source qwen3-vl:4b
    python scripts/mongo_sync.py actions                 # what staff should do now
    python scripts/mongo_sync.py history --area 6215_2   # one area over the day
    python scripts/mongo_sync.py actions --test          # no Atlas: in-memory (pip install mongomock)

Collections (database "sensei"):
  observations  one document per camera x image x area x source:
                {camera, image, ts, area_id, expected_product, stock_level,
                 stock_value, planogram_compliant, seen_product, coverage,
                 hand_occluding, bare_shelf_shortcut, source}
  areas         the planogram: {_id: area_id, camera, product_id, product_name, corners}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRIORITY = {  # lower = more urgent; mirrors the challenge's cost of each problem
    "empty": (1, "RESTOCK NOW"),
    "wrong": (2, "MOVE PRODUCT BACK"),
    "half": (3, "RESTOCK SOON"),
}


def get_db(test: bool):
    if test:
        import mongomock
        client = mongomock.MongoClient()
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            sys.exit("MONGODB_URI is not set (put it in .env), or use --test")
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    return client["sensei"]


def load(db, pred_path: Path, source: str) -> int:
    from pymongo import ASCENDING, DESCENDING, ReplaceOne

    pg = json.loads((ROOT / "planogram.json").read_text(encoding="utf-8"))
    cams = next(iter(pg.values()))["cameras"]
    expected = {}
    area_ops = []
    for cam, v in cams.items():
        for p in v["products"]:
            expected[p["area_id"]] = p["product_name"]
            area_ops.append(ReplaceOne({"_id": p["area_id"]}, {
                "_id": p["area_id"], "camera": cam, "product_id": p["product_id"],
                "product_name": p["product_name"], "corners": p["image_coordinates"]},
                upsert=True))
    db.areas.bulk_write(area_ops)

    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    dbg_path = pred_path.with_suffix(".debug.json")
    dbg = json.loads(dbg_path.read_text(encoding="utf-8")) if dbg_path.exists() else {}

    ops = []
    for cam, imgs in pred.items():
        for img, areas in imgs.items():
            ts = datetime.fromtimestamp(int(Path(img).stem), tz=timezone.utc)
            for aid, r in areas.items():
                d = dbg.get(cam, {}).get(img, {}).get(aid, {})
                s = (d.get("samples") or [{}])[0]
                doc = {
                    "_id": f"{cam}/{img}/{aid}/{source}",
                    "camera": cam, "image": img, "ts": ts, "area_id": aid,
                    "expected_product": expected.get(aid),
                    "stock_level": r["stock_level"], "stock_value": r["stock_value"],
                    "planogram_compliant": r["planogram_compliant"],
                    "seen_product": s.get("product"), "coverage": s.get("cov"),
                    "hand_occluding": s.get("occ"),
                    "bare_shelf_shortcut": bool(d.get("colour_override")),
                    "source": source,
                }
                ops.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
    if ops:
        db.observations.bulk_write(ops)
    db.observations.create_index([("area_id", ASCENDING), ("ts", DESCENDING)])
    db.observations.create_index([("camera", ASCENDING), ("ts", DESCENDING)])
    return len(ops)


def latest_per_area(db, source: str | None):
    match = {"source": source} if source else {}
    return list(db.observations.aggregate([
        {"$match": match},
        {"$sort": {"ts": -1}},
        {"$group": {"_id": "$area_id", "doc": {"$first": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$doc"}},
    ]))


def actions(db, source: str | None) -> None:
    rows = []
    for o in latest_per_area(db, source):
        if not o["planogram_compliant"]:
            key = "wrong"
        elif o["stock_level"] in ("empty", "half"):
            key = o["stock_level"]
        else:
            continue
        prio, label = PRIORITY[key]
        rows.append((prio, o["camera"], o["area_id"], label, o["expected_product"],
                     o.get("seen_product"), o["ts"]))
    rows.sort()
    print(f"STAFF ACTION LIST ({len(rows)} areas need attention)\n")
    for prio, cam, aid, label, exp, seen, ts in rows:
        extra = f"  (seen: {seen})" if label.startswith("MOVE") and seen else ""
        print(f"  [{label:17s}] camera {cam}  area {aid:8s}  {exp}{extra}   @ {ts:%H:%M} UTC")


def history(db, area: str, source: str | None) -> None:
    q = {"area_id": area, **({"source": source} if source else {})}
    print(f"History of area {area}:")
    for o in db.observations.find(q).sort("ts", 1):
        bar = {"empty": "", "half": "#####", "full": "##########"}[o["stock_level"]]
        flag = "" if o["planogram_compliant"] else "  WRONG PRODUCT"
        print(f"  {o['ts']:%H:%M}  {o['stock_level']:5s} {bar:10s}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load", "actions", "history"])
    ap.add_argument("--pred", help="predictions.json from run_local.py (debug file next to it is used too)")
    ap.add_argument("--source", default=None, help="label for this run, e.g. qwen3-vl:4b")
    ap.add_argument("--area")
    ap.add_argument("--test", action="store_true", help="in-memory MongoDB (mongomock) for a dry run")
    a = ap.parse_args()

    db = get_db(a.test)
    if a.test or a.cmd == "load":
        if not a.pred:
            sys.exit("--pred is required for load (and for --test)")
        n = load(db, Path(a.pred), a.source or "run")
        print(f"loaded {n} observations into sensei.observations")
    if a.cmd == "actions":
        actions(db, a.source)
    elif a.cmd == "history":
        history(db, a.area or sys.exit("--area is required"), a.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
