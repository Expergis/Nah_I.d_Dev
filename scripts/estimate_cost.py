"""Cost per analysed image — the number for the pitch.

Pure calculator: no GPU, no Ollama needed. Plug in the measured wall time
from a real run (run_local.py prints "Wrote ... in Ns"; divide by images).

    python scripts/estimate_cost.py                                  # defaults (measured 6215)
    python scripts/estimate_cost.py --sec-per-image 34.5             # after the full run
    python scripts/estimate_cost.py --debug predictions.debug.json   # real skip rate + calls

Every assumption is a flag, so the numbers can be defended in Q&A.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Gemini API list prices, USD per 1M tokens (ai.google.dev/gemini-api/docs/pricing,
# checked 2026-10-10). Output price includes thinking tokens.
API = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
}
IMAGE_TOKENS = 258   # Gemini: one image tile (our 512x384 crop and 448px context view)
PROMPT_TOKENS = 400  # prompt text incl. the camera's product list (rough)
OUTPUT_TOKENS = 30   # {"product":..., "coverage_pct":..., "hand_occluding":...}


def load_areas(planogram: Path) -> tuple[int, float]:
    data = json.loads(planogram.read_text(encoding="utf-8"))
    cams = next(iter(data.values()))["cameras"]
    n = [len(v["products"]) for v in cams.values()]
    return len(n), sum(n) / len(n)


def skip_rate(debug: Path) -> tuple[float, float]:
    """(fraction of areas skipped by the colour net, model calls per image)."""
    d = json.loads(debug.read_text(encoding="utf-8"))
    areas = skipped = calls = images = 0
    for imgs in d.values():
        for per_area in imgs.values():
            images += 1
            for info in per_area.values():
                areas += 1
                if info.get("colour_override"):
                    skipped += 1
                else:
                    calls += len(info.get("samples", [])) + (1 if info.get("verify") else 0)
    return (skipped / areas if areas else 0.0), (calls / images if images else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cost per analysed image")
    ap.add_argument("--planogram", default="planogram.json")
    ap.add_argument("--debug", help="predictions.debug.json from run_local.py (real skip rate)")
    ap.add_argument("--sec-per-image", type=float, default=29.0,
                    help="measured wall time per image (default: camera 6215, RX 9060 XT)")
    ap.add_argument("--watts", type=float, default=250.0,
                    help="whole PC at the wall during inference (GPU board power is 150 W)")
    ap.add_argument("--eur-kwh", type=float, default=0.15,
                    help="electricity price (Portugal business ~0.15, household ~0.22)")
    ap.add_argument("--hw-eur", type=float, default=900.0,
                    help="inference PC (RX 9060 XT 8GB ~300 + rest of the box)")
    ap.add_argument("--hw-years", type=float, default=3.0)
    ap.add_argument("--cameras", type=int, default=None, help="cameras per store")
    ap.add_argument("--interval-min", type=float, default=65.0,
                    help="minutes between images per camera (dataset median ~65)")
    ap.add_argument("--calls-per-image", type=float, default=None,
                    help="model calls per image (default: areas x (1 - skip rate))")
    ap.add_argument("--usd-eur", type=float, default=0.92)
    a = ap.parse_args()

    n_cams, areas_per_img = load_areas(Path(a.planogram))
    cams = a.cameras or n_cams
    skip, calls = (skip_rate(Path(a.debug)) if a.debug else (0.30, None))
    calls = a.calls_per_image or calls or areas_per_img * (1 - skip)

    # --- local --------------------------------------------------------------
    kwh = a.watts * a.sec_per_image / 3.6e6
    energy = kwh * a.eur_kwh
    imgs_day_store = cams * 24 * 60 / a.interval_min
    imgs_year_store = imgs_day_store * 365
    capacity_day = 86400 / a.sec_per_image
    stores_per_box = max(1, int(capacity_day // imgs_day_store))
    hw_year = a.hw_eur / a.hw_years
    hw_dedicated = hw_year / imgs_year_store
    hw_shared = hw_year / (imgs_year_store * stores_per_box)

    # --- cloud API, same calls ----------------------------------------------
    tok_in = calls * (2 * IMAGE_TOKENS + PROMPT_TOKENS)
    tok_out = calls * OUTPUT_TOKENS

    c = lambda eur: f"EUR {eur:.5f}  ({eur * 100:.3f} cents)"
    print("=" * 66)
    print(" COST PER ANALYSED IMAGE")
    print("=" * 66)
    print(f"Planogram: {n_cams} cameras, {areas_per_img:.1f} areas per image on average")
    print(f"Colour net skips {skip:.0%} of areas -> {calls:.1f} model calls per image")
    print(f"Wall time {a.sec_per_image:.0f} s/image on one RX 9060 XT 8GB (qwen3-vl:8b-instruct)")
    print()
    print("LOCAL (our setup)")
    print(f"  Energy        {kwh * 1000:.2f} Wh -> {c(energy)}")
    print(f"  Hardware      {c(hw_shared)} if one box serves {stores_per_box} stores")
    print(f"                {c(hw_dedicated)} if one box per store")
    print(f"  ALL-IN        {c(energy + hw_shared)}  to  {c(energy + hw_dedicated)}")
    print(f"  API fees      EUR 0. Images never leave the store.")
    print()
    print(f"Store volume: {cams} cameras, 1 image / {a.interval_min:.0f} min "
          f"-> {imgs_day_store:,.0f} images/day, {imgs_year_store:,.0f}/year")
    print(f"One box handles {capacity_day:,.0f} images/day "
          f"(= {stores_per_box} stores at this cadence)")
    print(f"Per store per year: energy EUR {energy * imgs_year_store:,.0f}, "
          f"hardware EUR {hw_year / stores_per_box:,.0f}-{hw_year:,.0f}")
    print()
    print(f"CLOUD API, same {calls:.1f} calls (~{tok_in:,.0f} in / {tok_out:,.0f} out tokens)")
    for name, (p_in, p_out) in API.items():
        usd = (tok_in * p_in + tok_out * p_out) / 1e6
        print(f"  {name:22s} {c(usd * a.usd_eur)}   "
              f"EUR {usd * a.usd_eur * imgs_year_store:,.0f}/store/year")
    print("  (2.5 Flash 'thinks' by default: thinking tokens bill as output, so")
    print("   real cost is higher unless thinking is turned off)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
