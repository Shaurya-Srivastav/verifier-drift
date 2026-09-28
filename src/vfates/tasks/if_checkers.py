# Copyright Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
#
# Ported from open-instruct (Apache-2.0):
# https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
# Predicates match ``IF_FUNCTIONS_MAP``. open-instruct's ``IFEvalVerifierOld``
# (ground_truth_utils.py) calls ``fn(response, **non_null_kwargs)`` where every
# ground_truth field other than ``func_name`` that is not JSON null is a kwarg.
"""Constraint checkers for allenai/RLVR-IFeval.

The predicates are a faithful port of AllenAI open-instruct's
``IF_FUNCTIONS_MAP`` (25 IFEval constraints). Docstrings here state the
contract the code actually implements. Two dataset quirks are intentional:

* ``validate_choice`` iterates ``options``. RLVR-IFeval stores ``options`` as
  one string (for example ``"yes/no/maybe"``), so the training-time checker
  tests whether any *character* of that string occurs in the response.
* ``validate_response_language`` is in the upstream map but unused by
  RLVR-IFeval. It imports ``langdetect`` lazily and is not a package dependency.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

Checker = Callable[..., bool]


def verify_keywords(text: str, keyword_list: list[str]) -> bool:
    """True when every keyword occurs in ``text``, case-insensitive."""
    lowered = text.lower()
    return all(keyword.lower() in lowered for keyword in keyword_list)


def verify_keyword_frequency(text: str, word: str, N: int) -> bool:
    """True when ``word`` occurs exactly ``N`` times as a whole word, case-insensitive."""
    words = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for token in words if token == word.lower()) == N


def validate_forbidden_words(text: str, forbidden_words: list[str]) -> bool:
    """True when no forbidden word occurs as a substring, case-insensitive."""
    lowered = text.lower()
    return not any(word.lower() in lowered for word in forbidden_words)


def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    """True when ``letter`` occurs exactly ``N`` times. ``letter`` must be one character.

    The count is case-sensitive, matching the upstream checker.
    """
    if len(letter) != 1:
        raise ValueError("Letter parameter must be a single character")
    return text.count(letter) == N


def validate_response_language(text: str, language: str) -> bool:
    """True when ``langdetect`` reports ``language`` for the whole response."""
    import langdetect

    return langdetect.detect(text) == language


def verify_paragraph_count(text: str, N: int) -> bool:
    """True when ``text`` splits into exactly ``N`` non-empty ``* * *`` paragraphs."""
    cleaned = "\n".join(line.strip() for line in text.splitlines()).strip()
    paragraphs = cleaned.split("* * *")
    if any(not paragraph.strip() for paragraph in paragraphs):
        return False
    return len(paragraphs) == N


def _quantified(actual: int, N: int, quantifier: str, *, around_tol: int | None = None) -> bool:
    if quantifier == "at least":
        return actual >= N
    if quantifier == "at most":
        return actual <= N
    if quantifier == "around":
        tol = max(round(N * 0.1), 1) if around_tol is None else around_tol
        return abs(actual - N) <= tol
    return False


def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    """Word-count constraint. ``around`` allows ±10% of ``N``, at least ±1."""
    return _quantified(len(text.strip().split()), N, quantifier)


def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    """Sentence-count constraint. ``around`` allows ±1 sentence."""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    return _quantified(len(sentences), N, quantifier, around_tol=1)


def validate_paragraphs(text: str, N: int, first_word: str, i: int) -> bool:
    """Exactly ``N`` paragraphs split on ``\\n\\n``, and paragraph ``i`` (1-based) starts with ``first_word``.

    Matches the upstream checker, including its lack of a bounds check on ``i``.
    """
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False
    return bool(paragraphs[i - 1].strip().startswith(first_word))


def verify_postscript(text: str, postscript_marker: str) -> bool:
    """True when ``postscript_marker`` occurs and something follows it."""
    index = text.find(postscript_marker)
    if index < 0:
        return False
    return len(text[index:].strip()) > len(postscript_marker)


def validate_placeholders(text: str, N: int) -> bool:
    """True when ``text`` contains at least ``N`` ``[placeholder]`` spans."""
    return len(re.findall(r"\[(.*?)\]", text)) >= N


def verify_bullet_points(text: str, N: int) -> bool:
    """True when exactly ``N`` lines start with ``*`` or ``-`` after stripping."""
    lines = text.split("\n")
    bullets = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    return len(bullets) == N


def validate_title(text: str) -> bool:
    """True when ``text`` contains a ``<<title>>`` span."""
    return len(re.findall(r"<<(.*?)>>", text)) > 0


def validate_choice(text: str, options: list[str] | str) -> bool:
    """True when any element of ``options`` is a substring of ``text``.

    A string ``options`` is iterated as characters. That is the training-time
    behaviour on RLVR-IFeval, which stores choices as one string.
    """
    return any(option in text for option in options)


def validate_highlighted_sections(text: str, N: int) -> bool:
    """True when at least ``N`` ``*highlighted*`` spans occur."""
    return len(re.findall(r"\*(.*?)\*", text)) >= N


def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    """True when ``section_splitter`` produces exactly ``N`` sections.

    A leading splitter is not counted as an empty first section, matching upstream.
    """
    sections = text.split(section_splitter)
    if sections[0] == "":
        sections.pop(0)
    return len(sections) == N


def validate_json_format(text: str) -> bool:
    """True when ``text`` is one JSON document."""
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    """True when ``text`` starts with ``original_prompt`` exactly."""
    return bool(text.startswith(original_prompt))


def validate_two_responses(text: str) -> bool:
    """True when ``******`` occurs once and the two sides differ after stripping."""
    if text.count("******") != 1:
        return False
    first, second = text.split("******")
    return first.strip() != second.strip()


def validate_uppercase(text: str) -> bool:
    """True when ``text`` is unchanged by ``str.upper``."""
    return text == text.upper()


def validate_lowercase(text: str) -> bool:
    """True when ``text`` is unchanged by ``str.lower``."""
    return text == text.lower()


def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    """Count of ``\\b[A-Z]+\\b`` tokens under the same quantifiers as word count."""
    return _quantified(len(re.findall(r"\b[A-Z]+\b", text)), N, quantifier)


def validate_end(text: str, end_phrase: str) -> bool:
    """True when ``text`` ends with ``end_phrase`` exactly."""
    return bool(text.endswith(end_phrase))


def validate_quotation(text: str) -> bool:
    """True when ``text`` starts and ends with a double quote."""
    return bool(text.startswith('"') and text.endswith('"'))


def validate_no_commas(text: str) -> bool:
    """True when ``text`` contains no comma."""
    return "," not in text


IF_FUNCTIONS_MAP: dict[str, Checker] = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}


def parse_ground_truth(ground_truth: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(func_name, args)`` from an RLVR-IFeval ``ground_truth`` value.

    ``args`` is every non-null field except ``func_name``. The dataset stores
    those fields flat; it does not nest them under an ``args`` object.
    """
    parsed = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    if "func_name" not in parsed:
        raise KeyError("ground_truth is missing func_name")
    func_name = str(parsed["func_name"])
    args = {key: value for key, value in parsed.items() if key != "func_name" and value is not None}
    return func_name, args


def apply_checker(text: str, func_name: str, args: dict[str, Any]) -> bool:
    """Run one checker. Unknown names and checker exceptions yield False.

    Exceptions are swallowed so a single malformed response cannot abort a
    grading sweep. Direct calls of the functions still raise (for example
    :func:`verify_letter_frequency` on a multi-character letter).
    """
    func = IF_FUNCTIONS_MAP.get(func_name)
    if func is None:
        return False
    try:
        return bool(func(text, **args))
    except Exception:
        return False
