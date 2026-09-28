"""Boxed extraction and decimal parsing."""

from decimal import Decimal

import pytest

from vfates.tasks.answer import (
    canonical_decimal,
    decimal_parts,
    extract_all_boxed,
    extract_boxed,
    normalise_statement,
    parse_decimal,
)

_EXTRACT = [
    ("", None),
    ("no box here", None),
    (r"\boxed{12.5}", "12.5"),
    (r"\boxed{12.5} and \boxed{9}", "9"),
    (r"\fbox{7}", "7"),
    (r"\fbox{1} \boxed{2}", "2"),
    (r"\boxed{1} \fbox{2}", "2"),
    (r"\boxed 12.5 more", "12.5"),
    (r"\boxed -12.5", "-12.5"),
    (r"\boxed{a{b}c}", "a{b}c"),
    (r"\boxed{\dfrac{1}{2}}", r"\dfrac{1}{2}"),
    (r"\boxed{12.5", None),
    (r"\boxed{}", ""),
    (r"\boxed{ 12.5 }", " 12.5 "),
    (r"prefix \boxednot{1}", None),
    (r"\boxed{1} tail", "1"),
]


@pytest.mark.parametrize(("text", "expected"), _EXTRACT)
def test_extract_boxed(text: str, expected: str | None) -> None:
    assert extract_boxed(text) == expected


def test_extract_all_boxed_keeps_earlier_closed_boxes() -> None:
    assert extract_all_boxed(r"\boxed{1} and \fbox{2}") == ["1", "2"]
    assert extract_all_boxed(r"\boxed{1} \boxed{2") == ["1"]
    assert extract_boxed(r"\boxed{1} \boxed{2") is None
    assert extract_all_boxed("no box") == []


def test_normalise_statement_folds_spacing_and_case() -> None:
    assert normalise_statement("Hello   WORLD") == normalise_statement("hello world")
    assert normalise_statement(r"$x\,+\,y$") == normalise_statement("x+y")


_PARSE = [
    ("12.5", Decimal("12.5")),
    ("  12.5  ", Decimal("12.5")),
    ("$12.5$", Decimal("12.5")),
    ("\\$12.5", Decimal("12.5")),
    ("1\\,234.5", Decimal("1234.5")),
    ("12\\!", Decimal("12")),
    ("\\text{12.5}", Decimal("12.5")),
    ("\\mathrm{12.5}", Decimal("12.5")),
    ("\\mathrm{\\text{12.5}}", Decimal("12.5")),
    ("\\left(12.5\\right)", Decimal("12.5")),
    ("12.5.", Decimal("12.5")),
    ("+12.5", Decimal("12.5")),
    ("\u221212.5", Decimal("-12.5")),
    ("- 12.5", Decimal("-12.5")),
    ("1,234.5", Decimal("1234.5")),
    ("12,34", None),
    ("= 12.5", Decimal("12.5")),
    ("x = 12.5", Decimal("12.5")),
    ("answer = 12.5", Decimal("12.5")),
    ("12.5 = 13", None),
    ("71.8 - 83.8 = -12.0", Decimal("-12.0")),
    ("233.3 - 658.8 = -425.5", Decimal("-425.5")),
    ("a = b = 12.5", None),
    ("71.8 - 83.8 = twelve", None),
    ("1/2", None),
    ("twelve", None),
    ("12.5 and 3", None),
    ("12.5 3.5", None),
    ("", None),
    ("1,234,567.89", Decimal("1234567.89")),
    ("12,345", Decimal("12345")),
    ("1234,567", None),
    ("-1,234.50", Decimal("-1234.50")),
    ("+1,234", Decimal("1234")),
    ("12.500", Decimal("12.500")),
    (".5", None),
    ("1e3", None),
    ("12.", Decimal("12")),
    ("\\text{1,234.5}", Decimal("1234.5")),
    ("12 . 5", Decimal("12.5")),
    ("1 2", None),
    ("1, 234.5", Decimal("1234.5")),
    ("(12.5)", Decimal("12.5")),
    ("(12.5).", Decimal("12.5")),
    ("x = (12.5)", Decimal("12.5")),
    ("+ 12.500", Decimal("12.500")),
    ("1 234", None),
    ("foo bar = 12.5", None),
    ("$-12.5$", Decimal("-12.5")),
    ("1,23,456", None),
    ("12,345,678", Decimal("12345678")),
    (None, None),
]


@pytest.mark.parametrize(("text", "expected"), _PARSE)
def test_parse_decimal(text: str | None, expected: Decimal | None) -> None:
    got = parse_decimal(text)
    if expected is None:
        assert got is None
    else:
        assert got == expected


def test_decimal_parts_keep_written_scale() -> None:
    assert decimal_parts("+1,234.50") == ("+", "1234", "50")
    assert decimal_parts("12") == ("", "12", "")
    assert decimal_parts("12,34") is None


def test_canonical_decimal() -> None:
    assert canonical_decimal(Decimal("12.500")) == "12.5"
    assert canonical_decimal(Decimal("12.0")) == "12"
    assert canonical_decimal(Decimal("-0")) == "0"
    assert canonical_decimal(Decimal("-0.000")) == "0"
    assert canonical_decimal(Decimal("100")) == "100"
    assert canonical_decimal(Decimal("0.10")) == "0.1"
    assert canonical_decimal(Decimal("-1.20")) == "-1.2"
    assert canonical_decimal(Decimal("1E+2")) == "100"
    assert canonical_decimal(Decimal("1E-2")) == "0.01"
    assert "e" not in canonical_decimal(Decimal("1E+20")).lower()
    with pytest.raises(TypeError):
        canonical_decimal(12)  # type: ignore[arg-type]
