"""Extract and canonicalise a decimal answer from a completion.

The oracle compares values, so surface form (trailing zeros, a leading plus,
a thousands separator) must survive in ``raw`` while ``canonical_decimal``
collapses it to one string per numeric value. That string is what clusters
answers inside a prompt.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from vfates.utils.hashing import u01
from vfates.utils.io import sha256_text

_COMMANDS = ("\\boxed", "\\fbox")
_PAIRS = {"(": ")", "[": "]", "{": "}"}
_WRAP = re.compile(r"\\(?:text|mathrm)\s*\{([^{}]*)\}")
_NUMBER = re.compile(
    r"^(?P<sign>[+-]?)(?:(?P<grouped>\d{1,3}(?:,\d{3})+)|(?P<plain>\d+))(?:\.(?P<frac>\d+))?$"
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` or ``\\fbox{...}``.

    Braces nest. A command without braces (``\\boxed 12.5``) yields the
    following non-space token. Returns ``None`` when no command is closed.
    Only the last command is read: an earlier closed box is ignored when the
    last command is unclosed.
    """
    if not isinstance(text, str) or not text:
        return None
    positions = _command_positions(text)
    if not positions:
        return None
    return _content_at(text, max(positions))


def extract_all_boxed(text: str) -> list[str]:
    """Every closed ``\\boxed`` or ``\\fbox``, in order.

    An unclosed command is skipped. Unlike :func:`extract_boxed`, an earlier
    closed box is kept when a later command is unclosed.
    """
    if not isinstance(text, str) or not text:
        return []
    found: list[str] = []
    for index in _command_positions(text):
        piece = _content_at(text, index)
        if piece is not None:
            found.append(piece)
    return found


def _content_at(text: str, i: int) -> str | None:
    if text.startswith("\\boxed", i):
        j = i + len("\\boxed")
    elif text.startswith("\\fbox", i):
        j = i + len("\\fbox")
    else:
        return None
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text):
        return None
    if text[j] == "{":
        depth = 0
        for k in range(j, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    return text[j + 1 : k]
        return None
    k = j
    while k < len(text) and not text[k].isspace():
        k += 1
    token = text[j:k]
    return token or None


_LATEX_SPACE = re.compile(r"\\[,!;:]|\\ |~")


def normalise_statement(text: str) -> str:
    """Statement key for dedup and decontamination.

    NFKC, unicode minus to ASCII hyphen, drop ``$`` and LaTeX spacing macros,
    casefold, collapse whitespace, strip. Two problems that differ only by
    those surface choices share a key.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    folded = folded.replace("$", "")
    folded = _LATEX_SPACE.sub("", folded)
    folded = folded.casefold()
    return re.sub(r"\s+", " ", folded).strip()


def statement_id(text: str) -> str:
    """SHA-256 of :func:`normalise_statement`."""
    return sha256_text(normalise_statement(text))


def hash_split(
    problem_id: str, *, split_salt: str, dev_fraction: float, heldout_fraction: float
) -> str:
    """``dev`` / ``heldout`` / ``train`` from ``u01(split_salt, problem_id)``.

    ``dev`` is ``[0, dev_fraction)``. ``heldout`` is the next
    ``heldout_fraction``. The rest is ``train``. The sample seed is not an
    input, so a pool does not move when a cell changes its seed.
    """
    draw = u01(split_salt, problem_id)
    if draw < dev_fraction:
        return "dev"
    if draw < dev_fraction + heldout_fraction:
        return "heldout"
    return "train"


def decimal_parts(s: str | None) -> tuple[str, str, str] | None:
    """``(sign, integer_digits, fraction_digits)`` of one written decimal.

    ``sign`` is ``''``, ``'+'`` or ``'-'``. ``fraction_digits`` is empty when
    the writer did not use a decimal point (so trailing zeros in the fraction
    are preserved). ``None`` if ``s`` is not exactly one decimal.
    """
    if not isinstance(s, str):
        return None
    token = _numeric_token(s)
    if token is None:
        return None
    match = _NUMBER.fullmatch(token)
    if match is None:
        return None
    grouped = match.group("grouped")
    integer = (grouped if grouped is not None else match.group("plain")).replace(",", "")
    frac = match.group("frac")
    return match.group("sign"), integer, "" if frac is None else frac


def parse_decimal(s: str | None) -> Decimal | None:
    """Parse one decimal, or ``None`` if ``s`` is not exactly one decimal.

    Tolerates surrounding whitespace, ``$`` / ``\\$``, spacing macros
    ``\\,`` ``\\!`` ``\\;``, ``\\text`` / ``\\mathrm`` wrappers, ``\\left`` /
    ``\\right``, one trailing sentence period, a leading ``+``, a unicode
    minus (U+2212), a space after the sign (``- 12.5``), thousands commas in
    strict groups of three (``1,234.5`` is accepted, ``12,34`` is not), and an
    ``= 12.5`` or ``x = 12.5`` prefix, and one arithmetic expression whose
    right-hand side is a plain decimal (``71.8 - 83.8 = -12.0``). Fractions,
    words, a second bare number (``12.5 = 13``), and more than one ``=``
    (``a = b = 12.5``) are rejected.
    """
    parts = decimal_parts(s)
    if parts is None:
        return None
    sign, integer, frac = parts
    body = integer if not frac else f"{integer}.{frac}"
    if sign:
        body = sign + body
    try:
        return Decimal(body)
    except InvalidOperation:
        return None


def canonical_decimal(d: Decimal) -> str:
    """Shortest plain decimal string: no exponent, no trailing zeros.

    ``-0`` (any scale) is ``'0'``. An integer value has no decimal point.
    """
    if not isinstance(d, Decimal):
        raise TypeError(f"canonical_decimal expects Decimal, got {type(d).__name__}")
    if d == 0:
        return "0"
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", "+0", ""}:
        return "0"
    if text.startswith("+"):
        text = text[1:]
    return text


def _command_positions(text: str) -> list[int]:
    found: list[int] = []
    for command in _COMMANDS:
        start = 0
        n = len(command)
        while True:
            i = text.find(command, start)
            if i < 0:
                break
            after = i + n
            # ``\boxednot`` is not the command ``\boxed``.
            if after < len(text) and text[after].isalpha():
                start = after
                continue
            found.append(i)
            start = after
    return found


def _drop_command(text: str, command: str) -> str:
    out: list[str] = []
    i = 0
    n = len(command)
    while i < len(text):
        j = text.find(command, i)
        if j < 0:
            out.append(text[i:])
            break
        after = j + n
        if after < len(text) and text[after].isalpha():
            out.append(text[i:after])
            i = after
            continue
        out.append(text[i:j])
        i = after
    return "".join(out)


def _latex_clean(text: str) -> str:
    text = text.replace("\u2212", "-")
    text = text.replace("\\$", "")
    text = text.replace("$", "")
    for macro in ("\\,", "\\!", "\\;"):
        text = text.replace(macro, "")
    text = _drop_command(text, "\\left")
    text = _drop_command(text, "\\right")
    prev = None
    while prev != text:
        prev = text
        text = _WRAP.sub(r"\1", text)
    return text


def _wraps(text: str) -> bool:
    if len(text) < 2 or text[0] not in _PAIRS or text[-1] != _PAIRS[text[0]]:
        return False
    open_c, close_c = text[0], text[-1]
    depth = 0
    for i, ch in enumerate(text):
        if ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth < 0 or (depth == 0 and i != len(text) - 1):
                return False
    return depth == 0


def _peel(text: str) -> str:
    """Drop wrapping brackets and a single trailing sentence period, repeatedly."""
    text = text.strip()
    while text:
        if _wraps(text):
            text = text[1:-1].strip()
            continue
        if text.endswith("."):
            text = text[:-1].strip()
            continue
        break
    return text


def _tighten(text: str) -> str | None:
    """Drop spaces around the sign, commas and the decimal point.

    A remaining space separates two tokens (``1 2``, ``12.5 and 3``), which
    is not one decimal. Spaces are not a thousands separator.
    """
    text = text.strip()
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"^([+-])\s+", r"\1", text)
        text = re.sub(r"\s*([.,])\s*", r"\1", text)
    if re.search(r"\s", text):
        return None
    return text


def _is_plain_decimal_text(text: str) -> bool:
    token = _tighten(_peel(text))
    return token is not None and _NUMBER.fullmatch(token) is not None


def _lhs_is_arithmetic(lhs: str) -> bool:
    """An expression such as ``71.8 - 83.8``, not a second bare number or words."""
    if _is_plain_decimal_text(lhs):
        return False
    if not any(ch.isdigit() for ch in lhs):
        return False
    return re.search(r"[+\-*/]", lhs) is not None


def _numeric_token(s: str) -> str | None:
    text = _peel(_latex_clean(s))
    if "=" not in text:
        return _tighten(text)
    # More than one "=" stays unparsed (``a = b = 12.5``).
    if text.count("=") != 1:
        return None
    lhs, rhs = text.split("=", 1)
    lhs = lhs.strip()
    rhs_token = _tighten(_peel(rhs.strip()))
    if lhs == "" or _IDENT.fullmatch(lhs) is not None:
        return rhs_token
    if rhs_token is None or _NUMBER.fullmatch(rhs_token) is None:
        return None
    if not _lhs_is_arithmetic(lhs):
        return None
    return rhs_token
