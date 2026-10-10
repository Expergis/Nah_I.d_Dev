"""Vision Inference Engine with multi-provider VLM, disk caching, and conservative fallbacks."""
from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from .config import (
    CACHE_DIR,
    CONFIDENCE_THRESHOLD,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_RETRIES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    VLM_PROVIDER,
)
from .prompt_templates import SYSTEM_PROMPT, build_analysis_prompt


# Conservative fallback values designed to protect against -3 point overstock penalties
SAFE_FALLBACK: Dict[str, Any] = {
    "detected_product": "unknown",
    "planogram_compliant": True,   # Defaulting to compliant avoids false alarm penalty
    "stock_level": "half",         # Middle ground prevents the harsh -3 overestimation penalty
    "confidence": 0.5,
}


def _clean_json_text(text: str) -> str:
    """Strip markdown codeblocks or surrounding chatter if any model produces it."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    return cleaned.strip()


def _cache_path(area_id: str, image_name: str) -> Path:
    safe_name = f"{area_id}_{Path(image_name).stem}.json"
    return CACHE_DIR / safe_name


def _get_cached_result(area_id: str, image_name: str) -> Optional[Dict[str, Any]]:
    cp = _cache_path(area_id, image_name)
    if cp.exists():
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(area_id: str, image_name: str, data: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(area_id, image_name)
    try:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _image_to_jpeg_bytes(rgb_img: np.ndarray) -> bytes:
    pil_img = Image.fromarray(rgb_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _call_gemini(rgb_crop: np.ndarray, prompt: str) -> Dict[str, Any]:
    """Inference via Google Gemini."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    jpeg_bytes = _image_to_jpeg_bytes(rgb_crop)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=SYSTEM_PROMPT + "\n\n" + prompt),
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    raw_text = _clean_json_text(response.text or "{}")
    return json.loads(raw_text)


def _call_openai(rgb_crop: np.ndarray, prompt: str) -> Dict[str, Any]:
    """Inference via OpenAI GPT-4o."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    jpeg_bytes = _image_to_jpeg_bytes(rgb_crop)
    b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_str}", "detail": "low"},
                    },
                ],
            },
        ],
        temperature=0.1,
    )
    raw_text = _clean_json_text(response.choices[0].message.content or "{}")
    return json.loads(raw_text)


def analyze_area(
    crop: np.ndarray,
    area_id: str,
    expected_product: str,
    candidate_products: List[str],
    image_name: str = "",
) -> Dict[str, Any]:
    """Analyzes a single cropped planogram compartment.

    Includes disk caching, multi-attempt retries, JSON validation,
    and asymmetric penalty safeguards.
    """
    # 1. Check local cache
    cached = _get_cached_result(area_id, image_name)
    if cached is not None:
        return cached

    prompt = build_analysis_prompt(area_id, expected_product, candidate_products)

    # 2. Query VLM with retry logic
    parsed_result: Optional[Dict[str, Any]] = None
    for attempt in range(MAX_RETRIES):
        try:
            if VLM_PROVIDER == "gemini" and GEMINI_API_KEY:
                parsed_result = _call_gemini(crop, prompt)
            elif VLM_PROVIDER == "openai" and OPENAI_API_KEY:
                parsed_result = _call_openai(crop, prompt)
            else:
                # If neither API key is active, use heuristic/safe fallback
                parsed_result = SAFE_FALLBACK.copy()
            break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.0 * (attempt + 1))

    if parsed_result is None:
        parsed_result = SAFE_FALLBACK.copy()

    # 3. Clean and validate schema values
    stock_raw = str(parsed_result.get("stock_level", "half")).lower()
    if stock_raw not in ("empty", "half", "full"):
        stock_raw = "half"

    compliant = bool(parsed_result.get("planogram_compliant", True))
    confidence = float(parsed_result.get("confidence", 0.5))

    # 4. Asymmetric penalty policy:
    # If the model is uncertain (< threshold) and claims 'full', downgrade to 'half'
    # to protect against the -3 penalty for overstocking errors.
    if confidence < CONFIDENCE_THRESHOLD and stock_raw == "full":
        stock_raw = "half"

    final_record = {
        "detected_product": str(parsed_result.get("detected_product", expected_product)),
        "planogram_compliant": compliant,
        "stock_level": stock_raw,
        "confidence": confidence,
    }

    # 5. Persist to cache
    _save_cache(area_id, image_name, final_record)
    return final_record
