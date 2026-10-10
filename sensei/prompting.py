"""Prompt builder for the per-area inspection.

Design choices (from the cross-LLM plan review):
- Ask for percent coverage FIRST (stable output), plus a coarse level —
  the level mapping can be re-tuned later without re-calling the API.
- Closed-set compliance question ("consistent with the expected product?")
  to avoid yes-bias of "is this a banana?".
- Explicit rules: black padding != empty shelf; ignore hands/arms.
- Reason before the answer fields improves consistency.
"""
from __future__ import annotations

from .planogram import PlanogramArea

PROMPT_TEMPLATE = """You are inspecting one shelf area (a gondola slot) in the fruits and vegetables section of a supermarket, seen from a ceiling camera.

Image 1 is the area of interest, perspective-corrected into a rectangle.{context_note}{visible_note}

EXPECTED PRODUCT (from the store planogram): "{product_name}" (product id {product_id}).
This is ONLY a reference for the compliance comparison below. Do NOT assume the area is
stocked with it. Many areas in this store are EMPTY (a bare dark shelf, no produce).
Judge the STOCK solely from what is actually visible in the crop.

Answer two questions:

1) COMPLIANCE — is the visible product consistent with the expected product?
   - "compliant": the visible evidence supports the expected product.
   - "noncompliant": clear visual evidence of a different product.
   - "uncertain": you cannot tell (heavy occlusion, too little visible, etc.).
   An empty shelf is a stock problem, not by itself a product mismatch.

2) STOCK — first estimate coverage_percent: the percentage (0-100) of the usable area covered by sellable product. Judge the pile's coverage of the shelf area, not the size of individual items. Then give the coarse level:
   - "empty": little or no visible product (coverage near 0-15).
   - "half": moderate quantity, clearly below full (roughly 15-70).
   - "full": substantially stocked, covering most of the usable area (above ~70).

STRICT RULES:
- Do not infer stock from the expected product. If you do not clearly see sellable produce inside the area, stock_level must be "empty" and coverage_percent near 0. An empty shelf is compliant, not a mismatch.
- Black padding or dark regions outside the real shelf are pixels OUTSIDE the camera image. Never treat them as an empty shelf.
- Ignore human hands, arms and bodies; estimate the stock behind them when possible, otherwise mark compliance/visibility accordingly.
- Do not invent details hidden by occlusion.
- Keep the reason short and grounded in what is actually visible.
"""


def build_prompt(
    area: PlanogramArea,
    has_context: bool = False,
    visible_fraction: float = 1.0,
) -> str:
    context_note = (
        "\nImage 2 is a wider view of the same shelf for spatial context "
        "(neighbors, shelf boundaries)."
        if has_context
        else ""
    )
    if visible_fraction < 0.9:
        visible_note = (
            f"\nNote: only about {visible_fraction:.0%} of the marked area is "
            "inside the camera image; judge from the visible part and say "
            "\"uncertain\" if too little is visible to be confident."
        )
    else:
        visible_note = ""
    return PROMPT_TEMPLATE.format(
        context_note=context_note,
        visible_note=visible_note,
        product_name=area.product_name,
        product_id=area.product_id,
    )
