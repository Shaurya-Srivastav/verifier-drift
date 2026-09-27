"""Answer-form classification (the strata we condition drift on).

Pure-Python/regex. A "form" is the surface shape of a final answer, which is
what a string/rule verifier is sensitive to even when the value is identical.
"""
from __future__ import annotations

import re

FORMS = (
    "integer",
    "fraction",
    "decimal",
    "percent",
    "radical",
    "interval",
    "set",
    "symbolic",
    "other",
)


def classify_form(ans: str) -> str:
    s = (ans or "").strip()
    if not s:
        return "other"
    if s.endswith("%"):
        return "percent"
    if re.match(r"^[\[\(].*,.*[\]\)]$", s):
        return "interval"
    if s.startswith("{") and s.endswith("}"):
        return "set"
    if "sqrt" in s or "\u221a" in s or "\\sqrt" in s:
        return "radical"
    if re.match(r"^[+-]?\d+\s*/\s*\d+$", s) or "\\frac" in s:
        return "fraction"
    if re.match(r"^[+-]?\d+$", s):
        return "integer"
    if re.match(r"^[+-]?\d*\.\d+$", s):
        return "decimal"
    if re.search(r"[a-zA-Z]", s):
        return "symbolic"
    return "other"
