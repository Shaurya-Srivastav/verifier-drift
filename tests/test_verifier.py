from vdrift.verifier import Oracle, RuleVerifier, to_number


def test_rule_verifier_exact_and_form_sensitive():
    rv = RuleVerifier()
    assert rv.check("1/2", "1/2") is True
    assert rv.check("3", "3") is True
    # form-sensitive false negative: same value, different surface form
    assert rv.check("0.5", "1/2") is False


def test_oracle_recovers_equivalence():
    o = Oracle()
    assert o.check("0.5", "1/2") is True      # verifier FN, oracle correct
    assert o.check("2/4", "1/2") is True
    assert o.check("50%", "1/2") is True
    assert o.check("0.7", "1/2") is False     # genuinely wrong


def test_to_number():
    assert abs(to_number("1/2") - 0.5) < 1e-9
    assert abs(to_number("0.25") - 0.25) < 1e-9
    assert abs(to_number("50%") - 0.5) < 1e-9
    assert to_number("x+1") is None
