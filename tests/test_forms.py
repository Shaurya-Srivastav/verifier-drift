from vdrift.forms import classify_form


def test_basic_forms():
    assert classify_form("42") == "integer"
    assert classify_form("-7") == "integer"
    assert classify_form("1/2") == "fraction"
    assert classify_form("0.5") == "decimal"
    assert classify_form("50%") == "percent"
    assert classify_form("sqrt(2)") == "radical"
    assert classify_form("x+1") == "symbolic"
    assert classify_form("(0, 1)") == "interval"
    assert classify_form("{1, 2}") == "set"
    assert classify_form("") == "other"
