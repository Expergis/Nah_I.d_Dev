from sensei.decision import (
    ALIAS,
    SPLIT,
    best_level_from_probs,
    decide_compliance,
    decide_stock,
    expected_stock_scores,
    family,
    family_compliant,
    norm,
    sample_agrees,
)


def test_ev_formula_matches_official_scoring():
    # If truth is always "full": EV(empty)=-2, EV(half)=-2, EV(full)=+1
    ev = expected_stock_scores(0.0, 0.0, 1.0)
    assert ev["empty"] == -2
    assert ev["half"] == -2
    assert ev["full"] == 1

    # If truth is always "empty": EV(empty)=+1, EV(half)=-3, EV(full)=-3
    ev = expected_stock_scores(1.0, 0.0, 0.0)
    assert ev["empty"] == 1
    assert ev["half"] == -3
    assert ev["full"] == -3


def test_borderline_half_full_prefers_half():
    # The case ChatGPT's wrong math got backwards: 20/35/45
    # EV: empty=-1.40, half=-1.15, full=-1.20  ->  half wins
    assert decide_stock({"empty": 20, "half": 35, "full": 45}) == "half"


def test_clear_full_still_full():
    assert decide_stock({"empty": 5, "half": 10, "full": 85}) == "full"


def test_clear_empty_still_empty():
    assert decide_stock({"empty": 85, "half": 10, "full": 5}) == "empty"


def test_maximum_uncertainty_prefers_empty():
    ev = expected_stock_scores(1 / 3, 1 / 3, 1 / 3)
    best = max(ev, key=ev.get)
    assert best == "empty"
    assert decide_stock({"empty": 1, "half": 1, "full": 1}) == "empty"


def test_best_level_from_probs():
    assert best_level_from_probs(0.2, 0.35, 0.45) == "half"
    assert best_level_from_probs(1 / 3, 1 / 3, 1 / 3) == "empty"
    assert best_level_from_probs(0.05, 0.1, 0.85) == "full"


def test_compliance_threshold():
    assert decide_compliance({"compliant": 2, "noncompliant": 1}) is True
    assert decide_compliance({"compliant": 1, "noncompliant": 2}) is False
    # tie -> compliant (base rate)
    assert decide_compliance({"compliant": 1, "noncompliant": 1}) is True
    # uncertain votes are not evidence of mismatch
    assert decide_compliance({"compliant": 1, "uncertain": 5}) is True
    assert decide_compliance({"uncertain": 5}) is True


# ------------------------------------------------------- family-based compliance

def test_norm_strips_accents():
    assert norm("Banana da Madeira") == "BANANA DA MADEIRA"
    assert norm("Citrinos do Algarve") == "CITRINOS DO ALGARVE"
    assert norm("  melão  ") == "MELAO"


def test_family_banana_variants():
    """Two banana SKUs should be the same family (fixes area-10 false non-compliant)."""
    assert family("BANANA CAT I CNT KG") == "BANANA"
    assert family("BANANA DA MADEIRA CNT KG") == "BANANA"
    assert family("BANANA CAT I CNT KG") == family("BANANA DA MADEIRA CNT KG")


def test_family_different_products():
    assert family("BANANA CAT I CNT KG") != family("CITRINOS DO ALGARVE IGP LARANJA 70/95 KG")
    assert family("TOMATE KG") != family("PIMENTO VERDE KG")


def test_family_split_multiword():
    """SPLIT entries take priority over first-word."""
    assert family("BATATA DOCE KG") == "BATATA DOCE"
    # A plain "BATATA" (if it existed) would be different
    assert family("BATATA DOCE KG") != family("BATATA KG")


def test_family_alias():
    """ALIAS maps partial first-words to canonical families."""
    assert family("CITRINOS DO ALGARVE IGP LARANJA 70/95 KG") == "CITRINOS"
    assert family("CITRINOS 55/70 KG") == "CITRINOS"
    assert family("MELAO PELE DE SAPO KG") == "MELOA"
    assert family("TOMATE CEREJA KG") == "TOMATE"
    assert family("TOMATE KG") == "TOMATE"


def test_sample_agrees_empty_and_other():
    """empty and other always agree (no positive evidence of mismatch)."""
    assert sample_agrees("empty", "BANANA DA MADEIRA CNT KG") is True
    assert sample_agrees("other", "BANANA DA MADEIRA CNT KG") is True


def test_sample_agrees_same_family():
    assert sample_agrees("BANANA CAT I CNT KG", "BANANA DA MADEIRA CNT KG") is True


def test_sample_agrees_different_family():
    assert sample_agrees("CITRINOS DO ALGARVE IGP LARANJA 70/95 KG", "BANANA DA MADEIRA CNT KG") is False


def test_family_compliant_majority():
    samples = [
        {"product": "BANANA CAT I CNT KG"},
        {"product": "BANANA DA MADEIRA CNT KG"},
        {"product": "CITRINOS DO ALGARVE IGP LARANJA 70/95 KG"},
    ]
    expected = "BANANA DA MADEIRA CNT KG"
    # 2 agree (same family), 1 disagrees -> compliant
    assert family_compliant(samples, expected) is True


def test_family_compliant_ties_go_compliant():
    samples = [
        {"product": "CITRINOS DO ALGARVE IGP LARANJA 70/95 KG"},
        {"product": "BANANA DA MADEIRA CNT KG"},
    ]
    expected = "BANANA DA MADEIRA CNT KG"
    # 1 agrees, 1 disagrees -> tie -> compliant (base rate)
    assert family_compliant(samples, expected) is True


def test_family_compliant_all_disagree():
    samples = [
        {"product": "CITRINOS DO ALGARVE IGP LARANJA 70/95 KG"},
        {"product": "PIMENTO VERDE KG"},
    ]
    expected = "BANANA DA MADEIRA CNT KG"
    # 0 agree, 2 disagree -> non-compliant
    assert family_compliant(samples, expected) is False


def test_family_compliant_empty_shelf():
    """Empty shelf is always compliant (confirmed by organizers)."""
    samples = [{"product": "empty"}, {"product": "empty"}]
    expected = "BANANA DA MADEIRA CNT KG"
    assert family_compliant(samples, expected) is True
