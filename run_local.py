"""Standalone Sensei pipeline for a local Ollama VLM (no repo needed).

Examples:
    python run_local.py --camera 6215 --limit 1 --save-crops crops   # smoke test
    python run_local.py                                              # full run
    python run_local.py --samples 3 --model qwen3-vl:8b-instruct
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROMPT_VERSION = "v2"
LEVELS = ["empty", "half", "full"]
VALUES = [0.0, 0.5, 1.0]
CACHE = HERE / ".cache_local"


# ----------------------------------------------------------------- planogram
def load_planogram(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    section = next(iter(data.values()))
    cams = {}
    for cam, v in section["cameras"].items():
        areas = []
        for p in v["products"]:
            c = p["image_coordinates"]
            quad = np.array([c["top_left"], c["top_right"], c["bottom_right"], c["bottom_left"]],
                            dtype=np.float32)
            areas.append({"area_id": p["area_id"], "name": p["product_name"], "quad": quad})
        cams[str(cam)] = areas
    return cams


# ------------------------------------------------------------------ geometry
def warp_area(img, quad, size, pad):
    """Perspective-rectify one area. Off-image parts become gray."""
    c = quad.mean(axis=0)
    q = (c + (quad - c) * (1.0 + pad)).astype(np.float32)
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))


def context_view(img, quad, long_side):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    pts = np.round(quad * s).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(small, [pts], True, (0, 255, 0), 2)
    return small


def jpeg(im):
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


# ------------------------------------------------------- shelf features (colour net)
def shelf_features(crop):
    """Cheap colour/texture signals for 'produce present'.
    - dark:  fraction of pixels with V < 100 (bare dark shelves score high)
    - fruit: fraction with S > 90 AND V > 90 (saturated, bright produce)
    - v_std: std of grayscale (texture proxy; log-only, not a gate)
    - texture: Canny edge density (log-only)
    Gray border padding (S~0) does not count as fruit."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    dark = float((hsv[..., 2] < 100).mean())
    fruit = float(((hsv[..., 1] > 90) & (hsv[..., 2] > 90)).mean())
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    v_std = float(gray.std())
    edges = cv2.Canny(gray, 50, 150)
    texture = float(edges.mean() / 255.0)
    return {"dark": round(dark, 4), "fruit": round(fruit, 4),
            "v_std": round(v_std, 1), "texture": round(texture, 4)}


# ------------------------------------------------------- family-based compliance
SPLIT = ["BATATA DOCE", "ALHO FRANCO", "COUVE FLOR",
         # visually distinct varieties sharing a first word (colour / shape differ)
         "COUVE ROXA", "COUVE BROCOLO", "COUVE BRANCA", "COUVE PORTUGUESA",
         "PIMENTO VERDE", "PIMENTO VERMELHO", "UVA RED", "UVA BRANCA",
         "CEBOLA ROXA", "PITAYA AMARELA", "PITAYA VERMELHA"]
ALIAS = {"CITR": "CITRINOS", "MELAO": "MELOA", "MINI": "TOMATE"}


def norm(s):
    return unicodedata.normalize("NFKD", s.upper()).encode("ascii", "ignore").decode().strip()


# Products a ceiling camera can barely tell apart: treated as one family
# (the challenge says products are recognised generically). Chosen from the
# model comparison on our hand labels: +15 pts (4B) / +36 pts (8B) of 304.
LOOKALIKE = [{"MAMAO", "PAPAIA"}, {"CITRINOS", "CLEMENTINA"}, {"LIMA", "LIMAO"},
             {"CEBOLA", "CEBOLINHA"}, {"ALPERCE", "PESSEGO", "NECTARINA"},
             {"MELANCIA", "MELOA"}]


def family(name):
    n = norm(name)
    for s in SPLIT:
        if n.startswith(s):
            return s
    first = n.split()[0] if n.split() else n
    f = ALIAS.get(first, first)
    for g in LOOKALIKE:
        if f in g:
            return "|".join(sorted(g))
    return f


def sample_agrees(product, expected):
    """True if the model's answer is consistent with the expected product."""
    p = norm(product)
    if p in ("EMPTY", "OTHER"):
        return True  # no positive evidence of a different listed product
    return family(product) == family(expected)


def compliant(samples, expected):
    """Majority-vote compliance using family matching. Ties go to compliant."""
    agree = sum(sample_agrees(s["product"], expected) for s in samples)
    return agree >= len(samples) / 2


# -------------------------------------------------------------------- prompt
def build_prompt(cands, with_context):
    """v2: UN-PRIMED. Do not tell the model what is expected here (it was answering
    the expectation, not what it saw, on bare shelves). We compare vs the planogram
    afterwards in decide()."""
    listing = "\n".join(f"- {c}" for c in cands)
    ctx = ("Image 2 is the full camera view with that area outlined in green (context only).\n"
           if with_context else "")
    return (
        "You are checking a fruit & vegetable shelf area seen from a ceiling camera.\n"
        "Image 1 is a rectified crop of ONE shelf area: this is the area to judge.\n"
        f"{ctx}"
        "Products that exist on this camera's shelves (for naming only):\n"
        f"{listing}\n\n"
        "IMPORTANT: Describe only what you actually see in Image 1. Do not assume the area is\n"
        "stocked just because a product is on the list. Many areas are EMPTY: a bare dark or metal\n"
        "shelf with no produce. If you do not clearly see fruit or vegetables inside the area,\n"
        "answer product=\"empty\" and coverage_pct=0. Do not guess a product from the list. Produce\n"
        "at the very edge that belongs to a neighbouring area does not count.\n\n"
        "Tasks:\n"
        "1. product: which product from the list is actually in the area? Use \"empty\" if the "
        "shelf/bin surface is visible with almost no produce, and \"other\" for produce not in the list.\n"
        "2. coverage_pct: integer 0-100, how much of the area's surface is covered by produce "
        "(0 = bare shelf, 50 = about half covered, 100 = fully piled). Ignore hands, people, and "
        "neighbouring areas.\n"
        "3. hand_occluding: true if a hand, person or cart hides a large part of the area.\n"
        "Answer with JSON only."
    )


def build_schema(cands):
    return {
        "type": "object",
        "properties": {
            "product": {"type": "string", "enum": cands + ["empty", "other"]},
            "coverage_pct": {"type": "integer"},
            "hand_occluding": {"type": "boolean"},
        },
        "required": ["product", "coverage_pct", "hand_occluding"],
    }


# -------------------------------------------------------------------- ollama
def ollama_chat(a, prompt, imgs, schema, i, temp, label=""):
    body = {
        "model": a.model, "stream": False, "keep_alive": -1, "think": False,
        "format": schema,
        "messages": [{"role": "user", "content": prompt,
                      "images": [__import__("base64").b64encode(b).decode() for b in imgs]}],
        "options": {"num_ctx": a.num_ctx, "temperature": temp, "num_predict": 120, "seed": i},
    }
    for _ in range(3):
        try:
            req = urllib.request.Request(a.host.rstrip("/") + "/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                resp = json.loads(r.read())
            if a.verbose:
                ms = lambda k: (resp.get(k) or 0) / 1e6
                n_in, n_out = resp.get("prompt_eval_count") or 0, resp.get("eval_count") or 0
                print(f"    {label:9s} wall {time.time() - t0:5.1f}s | load {ms('load_duration'):6.0f}ms"
                      f" | prompt {n_in:5d} tok in {ms('prompt_eval_duration'):6.0f}ms"
                      f" | output {n_out:3d} tok in {ms('eval_duration'):5.0f}ms"
                      f" ({n_out / max(ms('eval_duration') / 1000, 1e-9):5.1f} tok/s)", flush=True)
            content = resp["message"].get("content", "")
            if not content.strip():
                print("  ! empty content. done_reason=%s thinking=%r eval_count=%s" % (
                    resp.get("done_reason"), (resp["message"].get("thinking") or "")[:100],
                    resp.get("eval_count")), file=sys.stderr)
                time.sleep(1)
                continue
            return json.loads(content)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if "think" in body and "think" in msg.lower():
                body.pop("think")          # model/version doesn't accept the think flag
                continue
            print(f"  ! HTTP {e.code}: {msg[:200]}", file=sys.stderr)
            time.sleep(1)
        except Exception as e:             # timeouts, bad JSON, connection errors
            print(f"  ! call failed: {e}", file=sys.stderr)
            time.sleep(1)
    return None


def call_cached(a, prompt, imgs, schema, i, temp, label=""):
    h = hashlib.sha1()
    h.update(f"{PROMPT_VERSION}|{a.model}|{a.num_ctx}|{temp}|{i}|{prompt}".encode())
    for b in imgs:
        h.update(b)
    f = CACHE / (h.hexdigest() + ".json")
    if f.exists():
        if a.verbose:
            print(f"    {label:9s} cached", flush=True)
        return json.loads(f.read_text())
    out = ollama_chat(a, prompt, imgs, schema, i, temp, label)
    if out is not None:
        CACHE.mkdir(exist_ok=True)
        f.write_text(json.dumps(out))
    return out


def parse_sample(o):
    if not isinstance(o, dict):
        return None
    try:
        cov = float(o.get("coverage_pct"))
    except (TypeError, ValueError):
        return None
    prod = o.get("product")
    return {"product": prod if isinstance(prod, str) else "other",
            "cov": cov, "occ": bool(o.get("hand_occluding"))}


# ------------------------------------------------------------------ decision
def level_probs(cov, offset):
    c = min(max(cov + offset, 0.0), 100.0)
    return [max(0.0, 1 - c / 50), max(0.0, 1 - abs(c - 50) / 50), max(0.0, (c - 50) / 50)]


def best_level(p):
    """argmax expected score under: correct +1, too low -2, too high -3."""
    best, best_ev = 0, -1e9
    for pred in range(3):
        ev = 0.0
        for truth in range(3):
            s = 1 if pred == truth else (-2 if pred < truth else -3)
            ev += p[truth] * s
        if ev > best_ev:
            best, best_ev = pred, ev
    return best


def decide(samples, expected, offset):
    """Decide stock level (EV rule) + compliance (family-based majority vote)."""
    n = len(samples)
    p = [sum(level_probs(s["cov"], offset)[k] for s in samples) / n for k in range(3)]
    if sum(s["occ"] for s in samples) > n / 2:       # occluded: soften toward uniform
        p = [0.7 * x + 0.3 / 3 for x in p]
    lvl = best_level(p)
    comp = compliant(samples, expected)
    return {"planogram_compliant": comp,
            "stock_level": LEVELS[lvl], "stock_value": VALUES[lvl]}, p


def fallback():
    # No usable answer: never omit an area (-5). 'empty' has the best EV under ignorance.
    return {"planogram_compliant": True, "stock_level": "empty", "stock_value": 0.0}


# ------------------------------------------------------------------ per area
def process_area(a, cam, img, stem, area, cands):
    crop = warp_area(img, area["quad"], (a.crop_w, a.crop_h), a.pad)
    feats = shelf_features(crop)
    if a.save_crops:
        out = Path(a.save_crops)
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / f"{cam}_{stem}_{area['area_id']}.jpg"), crop)

    # Conservative colour safety-net (speed shortcut, NOT a classifier):
    # fire ONLY when dark is very high AND fruit is very low.
    # This avoids dark-produce misfires (aubergine, black grapes, mushrooms in shadow).
    # Texture gate: bare shelves are smooth (Canny density 0.021-0.034 on 6215), every
    # produce crop seen so far is >= 0.045 (cucumbers on 6383: dark 0.56, texture 0.117).
    if (a.dark_thr > 0 and feats["dark"] > a.dark_thr and feats["fruit"] < a.fruit_max
            and feats["texture"] < a.texture_max):
        res = {"planogram_compliant": True, "stock_level": "empty", "stock_value": 0.0}
        return area["area_id"], res, {"colour_override": True, **feats,
                                      "expected": area["name"]}

    imgs = [jpeg(crop)]
    if not a.no_context:
        imgs.append(jpeg(context_view(img, area["quad"], a.ctx_side)))
    prompt = build_prompt(cands, not a.no_context)
    schema = build_schema(cands)
    temp = 0.0 if a.samples == 1 else 0.5
    samples = []
    for i in range(a.samples):
        s = parse_sample(call_cached(a, prompt, imgs, schema, i, temp, area["area_id"]))
        if s:
            samples.append(s)
    if not samples:
        return area["area_id"], fallback(), {"failed": True, **feats, "expected": area["name"]}

    res, p = decide(samples, area["name"], a.cov_offset)

    # Optional: verify family-flagged non-compliance with a second crop-only call
    if a.verify_mismatch and not res["planogram_compliant"]:
        verify_prompt = (
            "Look at this shelf area crop. Is the main produce here "
            f"{area['name']}? Answer with JSON: "
            '{"answer": "yes"|"no"|"unsure"}'
        )
        verify_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unsure"]}},
            "required": ["answer"],
        }
        v_out = call_cached(a, verify_prompt, [jpeg(crop)], verify_schema, 0, 0.0,
                            area["area_id"] + "v")
        if v_out and v_out.get("answer") == "no":
            res["planogram_compliant"] = False
            # confirmed non-compliant
        else:
            # "yes" or "unsure" or failed call -> default to compliant (base-rate prior)
            res["planogram_compliant"] = True
        return area["area_id"], res, {"samples": samples, "probs": [round(x, 3) for x in p],
                                      **feats, "expected": area["name"], "verify": v_out}

    return area["area_id"], res, {"samples": samples, "probs": [round(x, 3) for x in p],
                                  **feats, "expected": area["name"]}


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Sensei pipeline (local Ollama)")
    ap.add_argument("--images-dir", default=str(HERE / "Frutas e Legumes (train & test)"))
    ap.add_argument("--planogram", default=str(HERE / "planogram.json"))
    ap.add_argument("--out", default=str(HERE / "predictions.json"))
    ap.add_argument("--camera")
    ap.add_argument("--image", help="filename or stem")
    ap.add_argument("--limit", type=int, help="max images per camera (smoke tests)")
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--crop-w", type=int, default=512)
    ap.add_argument("--crop-h", type=int, default=384)
    ap.add_argument("--ctx-side", type=int, default=448)
    ap.add_argument("--pad", type=float, default=0.08)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--cov-offset", type=float, default=0.0,
                    help="added to the model's coverage %% (tune on labeled data)")
    # Conservative colour net (speed shortcut, NOT a classifier)
    ap.add_argument("--dark-thr", type=float, default=0.60,
                    help="override to empty+compliant when dark fraction > this AND "
                         "fruit < --fruit-max. Set 0 to disable. Default 0.60.")
    ap.add_argument("--fruit-max", type=float, default=0.06,
                    help="override to empty+compliant when dark > --dark-thr AND "
                         "fruit fraction < this. Default 0.06.")
    ap.add_argument("--texture-max", type=float, default=0.04,
                    help="override also requires Canny edge density < this (bare shelves are "
                         "smooth, produce is busy). Default 0.04.")
    # Optional: verify non-compliance with a second call (default off)
    ap.add_argument("--verify-mismatch", action="store_true",
                    help="for family-flagged non-compliant areas, make one extra "
                         "crop-only call to confirm. Doubles cost on flagged areas only.")
    ap.add_argument("--save-crops", help="folder to dump rectified crops for eyeballing")
    ap.add_argument("--verbose", action="store_true",
                    help="print Ollama's timing breakdown for every model call")
    a = ap.parse_args()

    planogram = load_planogram(a.planogram)
    cams = [a.camera] if a.camera else sorted(planogram)
    print(f"Planogram: {len(planogram)} cameras, {sum(len(v) for v in planogram.values())} areas")
    print(f"Colour net: dark>{a.dark_thr}, fruit<{a.fruit_max}, texture<{a.texture_max} "
          f"({'ENABLED' if a.dark_thr > 0 else 'DISABLED'})")
    print(f"Verify mismatch: {'ON' if a.verify_mismatch else 'OFF'}")

    output, debug, fails, t0, n_areas = {}, {}, 0, time.time(), 0
    for cam in cams:
        if cam not in planogram:
            print(f"! camera {cam} not in planogram, skipping")
            continue
        cam_dir = Path(a.images_dir) / cam
        if not cam_dir.is_dir():
            print(f"! no image folder for camera {cam}: {cam_dir}")
            continue
        files = sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if a.image:
            files = [p for p in files if a.image in (p.name, p.stem)]
        if a.limit:
            files = files[: a.limit]
        areas = planogram[cam]
        cands = sorted({x["name"] for x in areas})
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"! unreadable image {f}")
                continue
            ts = time.time()
            with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
                results = list(ex.map(lambda ar: process_area(a, cam, img, f.stem, ar, cands), areas))
            output.setdefault(cam, {})[f.name] = {aid: res for aid, res, _ in results}
            debug.setdefault(cam, {})[f.name] = {aid: d for aid, _, d in results}
            fails += sum(1 for _, _, d in results if d.get("failed"))
            n_areas += len(results)
            print(f"{cam}/{f.name}: {len(areas)} areas in {time.time() - ts:.1f}s")

    Path(a.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    Path(a.out).with_suffix(".debug.json").write_text(json.dumps(debug, indent=1), encoding="utf-8")

    # validation: every planogram area present for every image
    problems = []
    for cam, imgs_ in output.items():
        want = {x["area_id"] for x in planogram[cam]}
        for name, areas_ in imgs_.items():
            miss = want - set(areas_)
            if miss:
                problems.append(f"{cam}/{name}: missing {sorted(miss)}")
    print(f"\nWrote {a.out}: {n_areas} area predictions in {time.time() - t0:.0f}s")
    if fails:
        print(f"WARNING: {fails} areas had no model answer (used fallback). Check Ollama / timeouts.")
    if problems:
        print(f"VALIDATION PROBLEMS ({len(problems)}):")
        for p in problems[:20]:
            print("  -", p)
        return 1
    print("Validation: OK (every planogram area present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
