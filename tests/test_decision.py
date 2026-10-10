from sensei.decision import (
    best_level_from_probs,
    decide_compliance,
    decide_stock,
    expected_stock_scores,
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
