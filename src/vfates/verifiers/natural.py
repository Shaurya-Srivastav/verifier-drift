"""Natural verifiers for RLVR-IFeval, math, and the QA judge.

IFEval runs the row's checker. The math verifiers are the lenient cascade
and DeepSeek-Math ``strip_string``. The QA verifier is the general-verifier
model judge. All of them ignore the oracle bit.
"""

from __future__ import annotations

import math
import re
from typing import Any

from vfates.tasks.answer import extract_boxed
from vfates.tasks.if_checkers import apply_checker
from vfates.types import DrawKey, Parsed, Problem
from vfates.verifiers.base import Verifier


class IFEvalCheckerVerifier(Verifier):
    """The RLVR-IFeval checker named by ``problem.meta['func_name']``."""

    name = "ifeval_checker"
    kind = "natural"
    stochastic = False

    def __init__(self, name: str = "ifeval_checker") -> None:
        self.name = name

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data["type"] = "ifeval_checker"
        return data

    def verdict(
        self, problem: Problem, parsed: Parsed, text: str, o: int, key: DrawKey | None = None
    ) -> int:
        del parsed, o, key
        return int(apply_checker(text, str(problem.meta["func_name"]), dict(problem.meta["args"])))


# DeepSeek-Math evaluation/data_processing/answer_extraction.py ``strip_string``
# (https://github.com/deepseek-ai/DeepSeek-Math). ``re`` stands in for
# ``regex``; the patterns are standard. Two upstream lines are no-ops and are
# not repeated: ``string.replace("'", "")`` and ``string.replace('"', "")``
# discard the result, so quotes stay, and ``string.replace("\\%", "%")`` is
# followed by ``string.replace("\%", "%")`` which is ``replace("%", "%")``.
# The ``j`` to ``i`` replacement is kept, including when it rewrites a lone ``j``.


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
                continue
            try:
                if len(substr) < 2:
                    raise ValueError
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                elif len(substr) > 2:
                    new_str += "{" + a + "}" + b + substr[2:]
                else:
                    new_str += "{" + a + "}" + b
            except Exception:
                return string
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a_text, b_text = string.split("/")
    try:
        a: int | str = a_text
        b: int | str = b_text
        if "sqrt" not in a_text:
            a = int(a_text)
        if "sqrt" not in b_text:
            b = int(b_text)
        if string != f"{a}/{b}":
            return string
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _fix_sqrt(string: str) -> str:
    fixed = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    return re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", fixed)


def _fix_tan(string: str) -> str:
    fixed = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    return re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", fixed)


def strip_string(string: str) -> str:
    """DeepSeek-Math ``strip_string``. Exact string equality is the strict check.

    Spaces and ``\\!`` go away. ``\\,`` does not, so ``x\\,+\\,y`` does not
    match ``x+y``. ``\\dfrac`` / ``\\tfrac`` / ``\\cfrac`` become ``\\frac``.
    A simple ``a/b`` of integers becomes ``\\frac{a}{b}``. A decimal is not
    rewritten as a fraction, so ``\\frac{1}{2}`` does not match ``0.5``.
    """
    text = str(string).strip()
    text = text.replace("\n", "")
    text = text.rstrip(".")
    text = text.replace("\\!", "")
    if text.startswith("\\text{") and text.endswith("}"):
        text = text.split("{", 1)[1][:-1]
    text = text.replace("tfrac", "frac")
    text = text.replace("dfrac", "frac")
    text = text.replace("cfrac", "frac")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    units = re.sub(r"\\text{.*?}$", "", text).strip()
    if units != "" and units != text:
        text = units
    text = text.replace("^{\\circ}", "").strip()
    text = text.replace("^\\circ", "").strip()
    text = re.sub(r"\{(c|m)?m\}(\^(2|3))?", "", text).strip()
    text = re.sub(r"p\.m\.$", "", text).strip()
    text = re.sub(r"(\d)\s*t$", r"\1", text).strip()
    text = text.replace("\\$", "")
    text = text.replace("$", "")
    text = text.replace("x\\in", "")
    text = text.replace("\\%", "%")
    text = text.replace(" .", " 0.")
    text = text.replace("{.", "{0.")
    text = text.replace("\\cdot", "")
    text = text.replace("infinity", "\\infty")
    if "\\infty" not in text:
        text = text.replace("inf", "\\infty")
    text = text.replace("+\\inity", "\\infty")
    text = text.replace("\\mathbf", "")
    text = text.replace("\\mathrm", "")
    text = re.sub(r"\\mbox{.*?}", "", text)
    if "j" in text and "i" not in text:
        text = text.replace("j", "i")
    text = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", text)
    text = re.sub(r"(\d+)\.0+$", r"\1", text)
    if len(text) == 0:
        return text
    if text[0] == ".":
        text = "0" + text
    text = _fix_sqrt(text)
    text = _fix_tan(text)
    text = text.replace(" ", "")
    text = _fix_fracs(text)
    text = _fix_a_slash_b(text)
    return re.sub(r"(\\|,|\.)+$", "", text)


def parse_digits(text: str) -> float | None:
    """Float parse used by the lenient cascade before the symbolic step.

    Commas are removed. A trailing ``%`` divides by 100. Anything else,
    including a fraction, is ``None`` so the cascade can fall through.
    """
    try:
        return float(str(text).replace(",", ""))
    except (TypeError, ValueError):
        pass
    token = str(text).strip()
    if token.endswith("%"):
        try:
            return float(token[:-1].replace(",", "")) / 100.0
        except (TypeError, ValueError):
            return None
    return None


_LAST_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def last_number(text: str) -> str | None:
    """Last integer, decimal, or percent token in ``text``."""
    found = list(_LAST_NUMBER.finditer(text or ""))
    if not found:
        return None
    return found[-1].group(0)


def lenient_extract(text: str) -> str | None:
    """Last boxed contents, otherwise the last number in the text."""
    boxed = extract_boxed(text or "")
    if boxed is not None and boxed.strip():
        return boxed.strip()
    return last_number(text or "")


def _symbolic_equal(left: str, right: str) -> bool:
    """SymPy equality. ``parse_latex`` when antlr is installed, else latex2sympy.

    Any parse or simplification error is inequality. This is the third step of
    the arXiv:2609.01354 cascade, not a numeric tolerance.
    """
    try:
        from sympy import simplify
    except ImportError:
        return False

    def _parse(expr: str) -> Any:
        try:
            from sympy.parsing.latex import parse_latex
        except ImportError:
            parse_latex = None
        if parse_latex is not None:
            try:
                return parse_latex(expr)
            except Exception:
                pass
        from latex2sympy2_extended.latex2sympy2 import latex2sympy

        return latex2sympy(expr)

    try:
        diff = simplify(_parse(left) - _parse(right))
    except Exception:
        return False
    return diff == 0


def lenient_equal(gold: str, prediction: str, *, rel_tol: float = 1e-4) -> bool:
    """Lenient math check from arXiv:2609.01354 (Where the Verifier Fails).

    1. DeepSeek ``strip_string`` equality.
    2. If both sides parse as numbers, ``math.isclose`` with ``rel_tol``
       (default ``1e-4``) and return that bit even when it is false. The
       relative tolerance accepts an off-by-one once ``|gold|`` is about
       ``1e4`` or larger (``10000`` versus ``10001``) and rejects it below
       that scale (``100`` versus ``101``).
    3. Otherwise symbolic equivalence.

    ``prediction`` is already the extracted answer (boxed, else last number).
    """
    if strip_string(gold) == strip_string(prediction):
        return True
    gold_num = parse_digits(gold)
    pred_num = parse_digits(prediction)
    if gold_num is not None and pred_num is not None:
        return math.isclose(gold_num, pred_num, rel_tol=rel_tol)
    return _symbolic_equal(gold, prediction)


class MathLenientVerifier(Verifier):
    """Last boxed answer, else the last number, then :func:`lenient_equal`."""

    name = "math_lenient"
    kind = "natural"
    stochastic = False

    def __init__(self, name: str = "math_lenient", rel_tol: float = 1e-4) -> None:
        if rel_tol < 0:
            raise ValueError("rel_tol must be non-negative")
        self.name = name
        self.rel_tol = float(rel_tol)

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update(
            {
                "type": "math_lenient",
                "rel_tol": self.rel_tol,
                "cascade": "strip_string,isclose,sympy",
            }
        )
        return data

    def verdict(
        self, problem: Problem, parsed: Parsed, text: str, o: int, key: DrawKey | None = None
    ) -> int:
        del parsed, o, key
        extracted = lenient_extract(text)
        if extracted is None:
            return 0
        return int(lenient_equal(problem.answer, extracted, rel_tol=self.rel_tol))


class MathStripStringVerifier(Verifier):
    """DeepSeek-Math ``strip_string`` on the last box, then exact equality.

    No closed box is incorrect. There is no numeric tolerance and no symbolic
    step, so ``\\frac{1}{2}`` versus ``0.5`` and ``x\\,+\\,y`` versus ``x+y``
    are misses even when the oracle accepts them.
    """

    name = "math_strip_string"
    kind = "natural"
    stochastic = False

    def __init__(self, name: str = "math_strip_string") -> None:
        self.name = name

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update({"type": "math_strip_string", "normalizer": "deepseek_math_strip_string"})
        return data

    def verdict(
        self, problem: Problem, parsed: Parsed, text: str, o: int, key: DrawKey | None = None
    ) -> int:
        del parsed, o, key
        boxed = extract_boxed(text or "")
        if boxed is None or not boxed.strip():
            return 0
        return int(strip_string(boxed) == strip_string(problem.answer))


class ModelJudgeVerifier(Verifier):
    """QA natural verifier: the general-verifier judge, not the oracle bit."""

    name = "model_judge"
    kind = "natural"
    stochastic = False

    def __init__(
        self, judge: Any, name: str = "model_judge", *, verifier_type: str = "model_judge"
    ) -> None:
        self.judge = judge
        self.name = name
        self.verifier_type = verifier_type

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update({"type": self.verifier_type, "judge": self.judge.describe()})
        return data

    def verdict(
        self, problem: Problem, parsed: Parsed, text: str, o: int, key: DrawKey | None = None
    ) -> int:
        del parsed, o, key
        return int(self.judge.judge(problem.question, text, gold=problem.answer))
