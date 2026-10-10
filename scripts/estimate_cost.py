"""Estimate processing cost per analyzed image.

Measures actual GPU inference time (if Ollama is reachable) or falls
back to estimates. Outputs a cost table for the pitch.

    python scripts/estimate_cost.py
    python scripts/estimate_cost.py --samples 3 --concurrency 6
    python scripts/estimate_cost.py --api-compare   # include API cost estimates
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensei.config import Config  # noqa: E402
from sensei.geometry import warp_area  # noqa: E402
from sensei.pipeline import list_images_per_camera, resolve_image_path  # noqa: E402
from sensei.planogram import Planogram  # noqa: E402

# Reference API pricing (USD, as of 2026-10; verify before pitch day)
API_PRICING = {
    "gemini-2.5-flash": {
        "input_per_1k_tokens": 0.0001,
        "output_per_1k_tokens": 0.0004,
        "image_tokens_approx": 258,  # 896x896 image ≈ 258 tokens (Gemini docs)
    },
    "gpt-4o": {
        "input_per_1k_tokens": 0.0025,
        "output_per_1k_tokens": 0.01,
        "image_tokens_approx": 765,  # 1024x1024 ≈ 765 tokens (OpenAI docs)
    },
}

TEXT_TOKENS_PER_PROMPT = 350  # rough estimate of our prompt template


def measure_ollama_time(cfg: Config, areas: list, image: "cv2.MatLike", samples: int) -> tuple[float, float]:
    """Time one batch of VLM calls. Returns (total_seconds, per_call_seconds)."""
    import httpx
    import base64
    import json as _json

    # Build a single representative crop
    crop, _, _ = warp_area(image, areas[0].corners, cfg.crop_width, cfg.crop_height)
    png = base64.b64encode(_encode_png(crop)).decode("ascii")

    prompt = (
        f'Inspect this shelf area. Expected product: "{areas[0].product_name}". '
        'Return JSON: {"compliance":"compliant","stock_level":"full","coverage_percent":80}'
    )
    body = {
        "model": cfg.ollama_model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [png]}],
        "options": {"temperature": 0.1, "num_ctx": cfg.ollama_num_ctx},
    }

    url = f"{cfg.ollama_url}/api/chat"
    times: list[float] = []

    async def _one_call(client: httpx.AsyncClient) -> float:
        t0 = time.perf_counter()
        resp = await client.post(url, json=body, timeout=cfg.timeout_s)
        resp.raise_for_status()
        return time.perf_counter() - t0

    async def _batch():
        async with httpx.AsyncClient() as client:
            # Warmup (load model into VRAM)
            await _one_call(client)
            # Timed calls
            results = await asyncio.gather(*[_one_call(client) for _ in range(samples)])
            return results

    times = asyncio.run(_batch())
    total = sum(times)
    per_call = total / len(times)
    return total, per_call


def _encode_png(image) -> bytes:
    import cv2
    import numpy as np
    success, buf = cv2.imencode(".png", image)
    return buf.tobytes()


def estimate_api_cost(num_areas: int, samples: int, model: str) -> float:
    """Estimate API cost for one image (all areas × samples)."""
    pricing = API_PRICING.get(model, API_PRICING["gemini-2.5-flash"])
    img_tokens = pricing["image_tokens_approx"]
    # Each call: 1 image + ~2 context images (crop + context) + text prompt
    input_tokens = samples * num_areas * (img_tokens * 2 + TEXT_TOKENS_PER_PROMPT)
    output_tokens = samples * num_areas * 50  # ~50 tokens of JSON output
    cost = (input_tokens / 1000) * pricing["input_per_1k_tokens"] + \
           (output_tokens / 1000) * pricing["output_per_1k_tokens"]
    return cost


def main() -> int:
    parser = argparse.ArgumentParser(description="Cost estimation per analyzed image")
    parser.add_argument("--samples", type=int, default=3, help="VLM samples per area")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--measure", action="store_true",
                        help="actually time Ollama calls (needs GPU running)")
    parser.add_argument("--api-compare", action="store_true",
                        help="include API cost comparison")
    parser.add_argument("--camera", default=None, help="limit to one camera")
    args = parser.parse_args()

    cfg = Config.from_env(samples=args.samples)
    pg = Planogram.load(cfg.planogram_path)
    cams = [args.camera] if args.camera else pg.cameras()
    cams = [c for c in cams if c in pg.cameras()]
    images_by_cam = list_images_per_camera(cfg.images_dir, cams)

    total_areas = sum(len(pg.area_ids(c)) for c in cams)
    total_images = sum(len(imgs) for imgs in images_by_cam.values())
    total_predictions = sum(len(pg.area_ids(c)) * len(images_by_cam.get(c, [])) for c in cams)

    # Per-image stats (use first camera as representative)
    rep_cam = sorted(cams)[0]
    rep_areas = len(pg.area_ids(rep_cam))
    rep_images = len(images_by_cam.get(rep_cam, [1]))

    print("=" * 62)
    print("SENSEI CHALLENGE — PROCESSING COST ESTIMATE")
    print("=" * 62)
    print(f"\nDataset:  {len(cams)} cameras, {total_images} images, "
          f"{total_predictions} area predictions")
    print(f"Per image (camera {rep_cam}): {rep_areas} areas × {args.samples} samples "
          f"= {rep_areas * args.samples} VLM calls")
    print(f"GPU:        RTX 4060 Laptop (8GB VRAM, CUDA 13.3)")
    print(f"Model:      {cfg.ollama_model} (Q4_K_M, ~5GB)")
    print(f"Backend:    Ollama (local, no API)")

    # Measured or estimated GPU time
    if args.measure:
        import cv2
        cam_dir = resolve_image_path(cfg.images_dir, rep_cam, images_by_cam[rep_cam][0])
        img = cv2.imread(str(cam_dir), cv2.IMREAD_COLOR)
        if img is not None:
            areas = pg.areas_for(rep_cam)
            print(f"\nMeasuring Ollama inference time ({args.samples} calls)...")
            total_t, per_call_t = measure_ollama_time(cfg, areas, img, args.samples)
            per_area_t = per_call_t * args.samples  # samples per area
            per_image_t = per_area_t * rep_areas / args.concurrency  # parallel
            print(f"  Per call:    {per_call_t:.2f}s")
            print(f"  Per area:    {per_area_t:.2f}s ({args.samples} samples)")
            print(f"  Per image:   ~{per_image_t:.1f}s ({rep_areas} areas, "
                  f"concurrency={args.concurrency})")
            gpu_watts = 120  # RTX 4060 under load
            kwh_per_image = (gpu_watts * per_image_t) / 3_600_000
            euro_per_kwh = 0.20  # EU residential rate
            print(f"  Electricity: {kwh_per_image:.6f} kWh/image "
                  f"(~€{kwh_per_image * euro_per_kwh:.6f})")
        else:
            print("\n[!] Could not read image for measurement, using estimates.")
            per_image_t = 45.0  # fallback estimate
    else:
        per_image_t = 45.0  # typical estimate: 10 areas x 3 samples / 6 concurrency
        print(f"\nGPU time (estimated): ~{per_image_t}s per image")

    # Cost summary
    print(f"\n{'-' * 62}")
    print(f"COST PER ANALYZED IMAGE (our approach)")
    print(f"{'-' * 62}")
    print(f"  API cost:        $0.00  (local inference, no external calls)")
    print(f"  GPU time:        ~{per_image_t:.0f}s  (RTX 4060, {cfg.ollama_model})")
    gpu_kwh = (120 * per_image_t) / 3_600_000
    print(f"  Electricity:     {gpu_kwh:.6f} kWh  (~EUR {gpu_kwh * 0.20:.6f})")
    print(f"  Marginal cost:   ~ EUR 0.00  (hardware already purchased)")
    print(f"  Total for full dataset ({total_predictions} predictions): "
          f"~{total_predictions * per_image_t / 60:.0f} min GPU time")

    if args.api_compare:
        print(f"\n{'-' * 62}")
        print(f"COMPARISON: CLOUD API COST PER IMAGE ({rep_areas} areas x {args.samples} samples)")
        print(f"{'-' * 62}")
        for model_name in API_PRICING:
            cost = estimate_api_cost(rep_areas, args.samples, model_name)
            cost_full = cost * total_images
            print(f"  {model_name:20s}  ${cost:.4f}/image   "
                  f"(${cost_full:.2f} for {total_images} images)")
        print(f"  {'Ours (local)':20s}  $0.0000/image   ($0.00 for {total_images} images)")

    # Scale estimate (Sensei production)
    print(f"\n{'-' * 62}")
    print(f"SCALE ESTIMATE: Sensei production (14 cameras, 24/7)")
    print(f"{'-' * 62}")
    imgs_per_day = 14 * 96  # ~1 image per 15 min per camera
    imgs_per_year = imgs_per_day * 365
    our_yearly = imgs_per_year * gpu_kwh * 0.20
    api_yearly_low = imgs_per_year * estimate_api_cost(rep_areas, args.samples, "gemini-2.5-flash")
    api_yearly_high = imgs_per_year * estimate_api_cost(rep_areas, args.samples, "gpt-4o")
    print(f"  Images/year:     ~{imgs_per_year:,}")
    print(f"  Ours/year:       ~EUR {our_yearly:.0f} electricity + amortized hardware")
    print(f"  Gemini API/year: ~${api_yearly_low:,.0f} in token costs")
    print(f"  GPT-4o API/year: ~${api_yearly_high:,.0f} in token costs")

    print(f"\n{'=' * 62}")
    print("PITCH LINE: 'Zero API cost. All inference on a single 8GB GPU.")
    print("At Sensei scale, marginal cost per analysis is electricity.'")
    print(f"{'=' * 62}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
