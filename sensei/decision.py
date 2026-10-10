"""Risk-sensitive decision layer.

The Sensei scoring table (per area):

    predict \\ truth   | empty  | half  | full
    -------------------+--------+-------+------
    empty              |  +1    |  -2   |  -2
    half               |  -3    |  +1   |  -2
    full               |  -3    |  -3   |  +1

So the expected scores of each prediction, for class probabilities
pE, pH, pF, are:

    EV(empty) =  pE - 2*pH - 2*pF
    EV(half)  = -3*pE + pH - 2*pF
    EV(full)  = -3*pE - 3*pH + pF

We pick the argmax — NOT a blanket "always under-call" bias (that loses
points when the shelf is actually full). Note that under maximum uncertainty
(near-uniform probabilities) "empty" is still the EV-best call, so the rule
is conservative where it should be and aggressive where the evidence is.

Compliance scoring is +1 correct / -2 wrong, so with probability p of
compliance: EV(compliant) = 3p - 2, EV(noncompliant) = 1 - 3p — predict
compliant iff p > 0.5 (ties go to compliant: most areas are expected to be
compliant).
"""
from __future__ import annotations

import unicodedata

STOCK_LEVELS = ("empty", "half", "full")

STOCK_VALUES: dict[str, float] = {
    "empty": 0.0,
    "half": 0.5,
    "full": 1.0,
}

COMPLIANCE_LABELS = ("compliant", "noncompliant", "uncertain")

# Fallback used when the API fails for an area. Missing areas score -5,
# so we must always emit something. Uniform-uncertainty EV says "empty";
# the compliance prior (base rate) says "compliant".
FALLBACK = {
    "planogram_compliant": True,
    "stock_level": "empty",
    "stock_value": 0.0,
}


def expected_stock_scores(p_empty: float, p_half: float, p_full: float) -> dict[str, float]:
    return {
        "empty": p_empty - 2 * p_half - 2 * p_full,
        "half": -3 * p_empty + p_half - 2 * p_full,
        "full": -3 * p_empty - 3 * p_half + p_full,
    }


def best_level_from_probs(p_empty: float, p_half: float, p_full: float) -> str:
    """Pick the stock level maximizing expected score from probabilities."""
    evs = expected_stock_scores(p_empty, p_half, p_full)
    return max(evs, key=evs.get)


def decide_stock(votes: dict[str, int]) -> str:
    """Pick the stock level maximizing expected score from sample votes."""
    total = sum(votes.get(k, 0) for k in STOCK_LEVELS)
    if total <= 0:
        return FALLBACK["stock_level"]
    p = {k: votes.get(k, 0) / total for k in STOCK_LEVELS}
    return best_level_from_probs(p["empty"], p["half"], p["full"])


def decide_compliance(votes: dict[str, int]) -> bool:
    """Predict compliant when its vote share beats noncompliant (ties -> compliant).

    "uncertain" votes are excluded from the denominator: they are not
    evidence of a mismatch.
    """
    compliant = votes.get("compliant", 0)
    noncompliant = votes.get("noncompliant", 0)
    total = compliant + noncompliant
    if total == 0:
        return True
    return 2 * compliant >= total


# ------------------------------------------------------- family-based compliance
# Used by the local Ollama pipeline (run_local.py) where the model reports a
# product name and we compare against the planogram locally.
# This avoids false non-compliance on near-identical SKUs (e.g. two banana types).

SPLIT: list[str] = ["BATATA DOCE", "ALHO FRANCO", "COUVE FLOR"]
ALIAS: dict[str, str] = {"CITR": "CITRINOS", "MELAO": "MELOA", "MINI": "TOMATE"}


def norm(s: str) -> str:
    """Normalize a product name: strip accents, uppercase, trim."""
    return unicodedata.normalize("NFKD", s.upper()).encode("ascii", "ignore").decode().strip()


def family(name: str) -> str:
    """Extract the product family (coarse grouping for compliance matching).

    e.g. "BANANA CAT I" and "BANANA DA MADEIRA" -> "BANANA" (compliant)
         "CITRINOS DO ALGARVE" and "CLEMENTINA" -> different families
    """
    n = norm(name)
    for s in SPLIT:
        if n.startswith(s):
            return s
    first = n.split()[0] if n.split() else n
    return ALIAS.get(first, first)


def sample_agrees(product: str, expected: str) -> bool:
    """True if the model's answer is consistent with the expected product.

    "empty" and "other" always agree (no positive evidence of a different
    listed product). Otherwise compare families.
    """
    p = norm(product)
    if p in ("EMPTY", "OTHER"):
        return True
    return family(product) == family(expected)


def family_compliant(samples: list[dict], expected: str) -> bool:
    """Majority-vote compliance using family matching. Ties go to compliant.

    Each sample dict should have a "product" key (the model's answer).
    """
    agree = sum(sample_agrees(s["product"], expected) for s in samples)
    return agree >= len(samples) / 2


def to_output(level: str) -> tuple[str, float]:
    if level not in STOCK_VALUES:
        raise ValueError(f"Invalid stock level: {level!r}")
    return level, STOCK_VALUES[level]
