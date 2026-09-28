# Copyright Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
#
# Ported from open-instruct (Apache-2.0). allenai/RLVR-IFeval was built for
# IFEvalVerifierOld, which calls IF_FUNCTIONS_MAP:
# https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
# https://github.com/allenai/open-instruct/blob/main/open_instruct/ground_truth_utils.py
# IFEvalG instruction classes (not the stored func_name rows) live at:
# https://github.com/allenai/open-instruct/blob/main/open_instruct/IFEvalG/instructions.py
"""Constraint checkers for allenai/RLVR-IFeval.

Predicates follow open-instruct ``IF_FUNCTIONS_MAP``. Where that map and the
instruction the model sees disagree, the checker follows the instruction.
Deliberate deviations:

* ``validate_choice``. ``if_functions.validate_choice`` iterates ``options``.
  RLVR-IFeval stores one string (``yes/no/maybe``, ``I know or I don't know``,
  or ``a), b), c), d)``), so the upstream loop walks characters and ordinary
  prose passes. IFEvalG ``ConstrainedResponseChecker`` tests fixed phrases,
  case-sensitive, after ``strip``. This port splits those three stored formats
  and tests a case-insensitive substring of each option. Case-folding differs
  from both upstream checkers.
* ``validate_repeat_prompt``. ``IFEvalVerifierOld`` requires
  ``text.startswith(original_prompt)``, and the stored field is the whole user
  message, including the sentence that begins ``First, repeat the request``.
  IFEvalG ``RepeatPromptThenAnswer`` compares a separate prompt with
  ``strip().lower().startswith``. This port drops that instruction sentence,
  then keeps the case-sensitive prefix check. It does not case-fold: the
  visible instruction says to repeat the request without change.
* ``validate_end``. Upstream ``if_functions`` is ``text.endswith(end_phrase)``.
  IFEvalG ``EndChecker`` strips, strips quotes, and lowercases. The RLVR
  template appends a sentence period, so a phrase that already ends in ``.``
  is shown with a doubled period (``Oh no..``). This port strips trailing
  whitespace and accepts one extra period only in that case. It does not
  strip quotes or case-fold.
* ``validate_quotation``. Upstream requires the text to start and end with a
  double quote. IFEvalG strips both ends and requires length greater than 1.
  This port only strips trailing whitespace, so a trailing newline matches.
  Leading whitespace still fails, and ``""`` still matches.
* ``validate_forbidden_words``. Upstream ``if_functions`` is a substring test,
  so ``ate`` matches ``later``. IFEvalG uses a ``\\b`` boundary and does not
  escape the word. This port is a case-insensitive word-boundary match with
  ``re.escape``. On the dataset's plain words that is the IFEvalG check.
* ``validate_response_language`` is in the upstream map but unused by
  RLVR-IFeval. It imports ``langdetect`` lazily and is not a package dependency.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

Checker = Callable[..., bool]

# Multi-constraint rows store every instruction under this func_name so the
# existing ifeval_checker verifier, which calls apply_checker(func_name, args),
# grades the whole bundle.
IFEVAL_ALL = "ifeval_all"


def verify_keywords(text: str, keyword_list: list[str]) -> bool:
    """True when every keyword occurs in ``text``, case-insensitive."""
    lowered = text.lower()
    return all(keyword.lower() in lowered for keyword in keyword_list)


def verify_keyword_frequency(text: str, word: str, N: int) -> bool:
    """True when ``word`` occurs exactly ``N`` times as a whole word, case-insensitive."""
    words = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for token in words if token == word.lower()) == N


def validate_forbidden_words(text: str, forbidden_words: list[str]) -> bool:
    """True when no forbidden word occurs as a whole word, case-insensitive.

    Word boundaries match IFEvalG ``ForbiddenWords``. The word is escaped, so a
    metacharacter is literal. Upstream ``if_functions`` is a substring test.
    """
    lowered = text.lower()
    return not any(
        re.search(r"\b" + re.escape(word.lower()) + r"\b", lowered) for word in forbidden_words
    )


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


def _choice_options(options: str) -> list[str]:
    """Split the three strings RLVR-IFeval stores in ``options``.

    ``yes/no/maybe``, ``a), b), c), d)``, and ``I know or I don't know``.
    Anything else is one option.
    """
    if "/" in options and "), " not in options:
        parts = options.split("/")
    elif "), " in options:
        parts = [part.strip() for part in options.split(",")]
    elif " or " in options:
        parts = options.split(" or ")
    else:
        parts = [options]
    return [part.strip() for part in parts if part.strip()]


def validate_choice(text: str, options: list[str] | str) -> bool:
    """True when any option phrase is a case-insensitive substring of ``text``.

    A stored string is split by :func:`_choice_options` before the test.
    Upstream ``if_functions`` does not split and is case-sensitive.
    """
    choices = _choice_options(options) if isinstance(options, str) else list(options)
    lowered = text.lower()
    return any(str(option).lower() in lowered for option in choices)


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


_REPEAT_SENTENCE = "First, repeat the request"


def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    """True when ``text`` starts with the request, excluding the repeat instruction.

    The stored ``original_prompt`` is the whole user message. The sentence that
    begins ``First, repeat the request`` is not part of the request the model
    was told to repeat. The prefix check stays case-sensitive.
    """
    request = original_prompt
    cut = original_prompt.find(_REPEAT_SENTENCE)
    if cut > 0:
        request = original_prompt[:cut].rstrip()
    return bool(text.startswith(request))


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
    """True when ``text``, ignoring trailing whitespace, ends with ``end_phrase``.

    When ``end_phrase`` already ends with ``.``, one extra sentence period is
    accepted. That is the period the RLVR template appends after the phrase.
    A phrase that does not already end with ``.`` does not gain one.
    """
    body = text.rstrip()
    if body.endswith(end_phrase):
        return True
    return bool(end_phrase.endswith(".") and body.endswith(end_phrase + "."))


def validate_quotation(text: str) -> bool:
    """True when ``text`` starts with ``"`` and, after trailing whitespace, ends with ``"``."""
    body = text.rstrip()
    return bool(body.startswith('"') and body.endswith('"'))


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

    ``func_name == ifeval_all`` grades every constraint in
    ``args["constraints"]``. Unknown instruction ids and a missing
    ``langdetect`` import propagate. Other checker exceptions still yield False,
    so an unknown id cannot pass.
    """
    if func_name == IFEVAL_ALL:
        constraints = args.get("constraints")
        if not isinstance(constraints, list):
            return False
        try:
            return all(
                apply_instruction(text, str(item["instruction_id"]), item.get("kwargs"))
                for item in constraints
            )
        except (ImportError, KeyError):
            raise
        except Exception:
            return False
    func = IF_FUNCTIONS_MAP.get(func_name)
    if func is None:
        return False
    try:
        return bool(func(text, **args))
    except Exception:
        return False


# Instruction-id checkers for allenai/IF_multi_constraints_upto5. These follow
# IFEvalG check_following and that dataset's kwargs. They do not replace
# IF_FUNCTIONS_MAP: RLVR-IFeval still uses the func_name checkers above, including
# the deviations listed at the top of this module.
#
# length_constraints:number_sentences counts sentences with
# ifbench_checkers.split_into_sentences. IFEvalG uses NLTK Punkt. The regex
# splitter is the one IFEvalG already uses for its other sentence checkers, and
# it matches Punkt on ordinary prose. change_case:capital_word_frequency uses
# the pure-Python word_tokenize in that module.
#
# language:response_language, change_case:english_capital and
# change_case:english_lowercase import langdetect lazily. A detection failure
# (LangDetectException) counts as followed, matching IFEvalG. ImportError
# propagates. detectable_format:constrained_response accepts only the three
# strings in _CONSTRAINED_RESPONSE_OPTIONS. Some dataset prompts list extra
# "I think ..." options that this checker rejects.


_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)


def _relation(actual: int, threshold: int, relation: str) -> bool:
    from vfates.tasks.ifbench_checkers import relation_holds

    return relation_holds(actual, int(threshold), relation)


def check_keywords_existence(text: str, keywords: list[str]) -> bool:
    """Every keyword matches as a case-insensitive regular expression."""
    return all(re.search(keyword, text, flags=re.IGNORECASE) for keyword in keywords)


def check_keywords_frequency(text: str, keyword: str, frequency: int, relation: str) -> bool:
    """Substring count of ``keyword`` (a regular expression) under ``relation``."""
    actual = len(re.findall(keyword.strip(), text, flags=re.IGNORECASE))
    return _relation(actual, frequency, relation)


def check_forbidden_words_id(text: str, forbidden_words: list[str]) -> bool:
    """No forbidden word matches ``\\bword\\b``. The word is not escaped."""
    return all(
        re.search(r"\b" + word + r"\b", text, flags=re.IGNORECASE) is None
        for word in forbidden_words
    )


def check_letter_frequency_id(
    text: str, letter: str, let_frequency: int, let_relation: str
) -> bool:
    """Case-insensitive letter count. IFEvalG treats any other relation as at least."""
    counted = Counter(text.lower())
    actual = counted[letter.strip().lower()]
    if let_relation == "less than":
        return actual < int(let_frequency)
    return actual >= int(let_frequency)


def check_response_language(text: str, language: str) -> bool:
    """True when langdetect reports ``language``. Detection failure counts as followed."""
    import langdetect

    try:
        return langdetect.detect(text) == language
    except langdetect.LangDetectException:
        return True


def check_number_sentences(text: str, num_sentences: int, relation: str) -> bool:
    """Sentence count under ``relation``, using the regex splitter rather than Punkt."""
    from vfates.tasks.ifbench_checkers import split_into_sentences

    return _relation(len(split_into_sentences(text)), num_sentences, relation)


def check_number_paragraphs(text: str, num_paragraphs: int) -> bool:
    """Exactly ``num_paragraphs`` blocks separated by ``***``."""
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == int(num_paragraphs)


def check_number_words(text: str, num_words: int, relation: str) -> bool:
    """``\\w+`` word count under ``relation``."""
    from vfates.tasks.ifbench_checkers import count_words

    return _relation(count_words(text), num_words, relation)


def check_nth_paragraph_first_word(
    text: str, num_paragraphs: int, nth_paragraph: int, first_word: str
) -> bool:
    """``num_paragraphs`` blank-line paragraphs, and paragraph ``nth`` starts with ``first_word``."""
    paragraphs = re.split(r"\n\n", text)
    count = len(paragraphs)
    for paragraph in paragraphs:
        if not paragraph.strip():
            count -= 1
    index = int(nth_paragraph)
    if index > count:
        return False
    paragraph = paragraphs[index - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    got = ""
    for character in word:
        if character in {".", ",", "?", "!", "'", '"'}:
            break
        got += character.lower()
    return count == int(num_paragraphs) and got == first_word.strip().lower()


def check_placeholders_id(text: str, num_placeholders: int) -> bool:
    """At least ``num_placeholders`` ``[placeholder]`` spans."""
    return len(re.findall(r"\[.*?\]", text)) >= int(num_placeholders)


def check_postscript_id(text: str, postscript_marker: str) -> bool:
    """A postscript line matches the marker. ``P.S.`` and ``P.P.S`` use IFEvalG's patterns."""
    value = text.lower()
    marker = postscript_marker.strip()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def check_bullet_lists(text: str, num_bullets: int) -> bool:
    """Exactly ``num_bullets`` markdown bullets (``*`` or ``-``)."""
    stars = re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", text, flags=re.MULTILINE)
    return len(stars) + len(dashes) == int(num_bullets)


def check_constrained_response(text: str) -> bool:
    """The stripped response contains one of the three canonical option strings."""
    value = text.strip()
    return any(option in value for option in _CONSTRAINED_RESPONSE_OPTIONS)


def check_highlighted_sections(text: str, num_highlights: int) -> bool:
    """At least ``num_highlights`` non-empty ``*highlight*`` or ``**highlight**`` spans."""
    count = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", text):
        if highlight.strip("*").strip():
            count += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", text):
        if highlight.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= int(num_highlights)


def check_multiple_sections(text: str, section_spliter: str, num_sections: int) -> bool:
    """At least ``num_sections`` blocks opened by ``splitter`` plus a number.

    The kwarg is spelled ``section_spliter``, matching the dataset. The splitter
    is a regular expression and is not escaped.
    """
    pattern = r"\s?" + section_spliter.strip() + r"\s?\d+\s?"
    return len(re.split(pattern, text)) - 1 >= int(num_sections)


def check_json_format_id(text: str) -> bool:
    """One JSON document, after a single leading and trailing markdown fence is removed."""
    value = (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def check_title_id(text: str) -> bool:
    """A non-empty ``<<title>>`` span."""
    titles = re.findall(r"<<[^\n]+>>", text)
    return any(title.lstrip("<").rstrip(">").strip() for title in titles)


def check_two_responses_id(text: str) -> bool:
    """Exactly two non-empty ``******`` sides, and they differ after strip.

    Empty pieces at either end are ignored. A leading separator therefore does
    not by itself make two responses. The RLVR ``validate_two_responses`` checker
    is different.
    """
    valid: list[str] = []
    responses = text.split("******")
    for index, response in enumerate(responses):
        if not response.strip():
            if index != 0 and index != len(responses) - 1:
                return False
        else:
            valid.append(response)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def check_repeat_prompt_id(text: str, prompt_to_repeat: str) -> bool:
    """The stripped response starts with the prompt, case-insensitive."""
    return bool(text.strip().lower().startswith(prompt_to_repeat.strip().lower()))


def check_end_id(text: str, end_phrase: str) -> bool:
    """After strip, quote strip, and case-fold, the response ends with ``end_phrase``."""
    value = text.strip().strip('"').lower()
    return value.endswith(end_phrase.strip().lower())


def check_capital_word_frequency(text: str, capital_frequency: int, capital_relation: str) -> bool:
    """Count of all-caps Treebank tokens under ``capital_relation``."""
    from vfates.tasks.ifbench_checkers import word_tokenize

    capital = [word for word in word_tokenize(text) if word.isupper()]
    if capital_relation == "less than":
        return len(capital) < int(capital_frequency)
    return len(capital) >= int(capital_frequency)


def check_english_capital(text: str) -> bool:
    """All cased characters are uppercase and langdetect reports English.

    If the text is uppercase and language detection fails, IFEvalG counts it as followed.
    """
    import langdetect

    try:
        return text.isupper() and langdetect.detect(text) == "en"
    except langdetect.LangDetectException:
        return True


def check_english_lowercase(text: str) -> bool:
    """All cased characters are lowercase and langdetect reports English.

    If the text is lowercase and language detection fails, IFEvalG counts it as followed.
    """
    import langdetect

    try:
        return text.islower() and langdetect.detect(text) == "en"
    except langdetect.LangDetectException:
        return True


def check_no_comma_id(text: str) -> bool:
    """No comma."""
    return re.search(r"\,", text) is None


def check_quotation_id(text: str) -> bool:
    """Stripped text has length greater than 1 and both ends are double quotes."""
    value = text.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


def _entry(fn: Checker, *required: str) -> tuple[Checker, tuple[str, ...]]:
    return fn, required


IFEVAL_SPECS: dict[str, tuple[Checker, tuple[str, ...]]] = {
    "keywords:existence": _entry(check_keywords_existence, "keywords"),
    "keywords:frequency": _entry(check_keywords_frequency, "keyword", "frequency", "relation"),
    "keywords:forbidden_words": _entry(check_forbidden_words_id, "forbidden_words"),
    "keywords:letter_frequency": _entry(
        check_letter_frequency_id, "letter", "let_frequency", "let_relation"
    ),
    "language:response_language": _entry(check_response_language, "language"),
    "length_constraints:number_sentences": _entry(
        check_number_sentences, "num_sentences", "relation"
    ),
    "length_constraints:number_paragraphs": _entry(check_number_paragraphs, "num_paragraphs"),
    "length_constraints:number_words": _entry(check_number_words, "num_words", "relation"),
    "length_constraints:nth_paragraph_first_word": _entry(
        check_nth_paragraph_first_word, "num_paragraphs", "nth_paragraph", "first_word"
    ),
    "detectable_content:number_placeholders": _entry(check_placeholders_id, "num_placeholders"),
    "detectable_content:postscript": _entry(check_postscript_id, "postscript_marker"),
    "detectable_format:number_bullet_lists": _entry(check_bullet_lists, "num_bullets"),
    "detectable_format:constrained_response": _entry(check_constrained_response),
    "detectable_format:number_highlighted_sections": _entry(
        check_highlighted_sections, "num_highlights"
    ),
    "detectable_format:multiple_sections": _entry(
        check_multiple_sections, "section_spliter", "num_sections"
    ),
    "detectable_format:json_format": _entry(check_json_format_id),
    "detectable_format:title": _entry(check_title_id),
    "combination:two_responses": _entry(check_two_responses_id),
    "combination:repeat_prompt": _entry(check_repeat_prompt_id, "prompt_to_repeat"),
    "startend:end_checker": _entry(check_end_id, "end_phrase"),
    "change_case:capital_word_frequency": _entry(
        check_capital_word_frequency, "capital_frequency", "capital_relation"
    ),
    "change_case:english_capital": _entry(check_english_capital),
    "change_case:english_lowercase": _entry(check_english_lowercase),
    "punctuation:no_comma": _entry(check_no_comma_id),
    "startend:quotation": _entry(check_quotation_id),
}

IFEVAL_INSTRUCTION_IDS: tuple[str, ...] = tuple(IFEVAL_SPECS)
IFEVAL_INSTRUCTION_CHECKERS: dict[str, Checker] = {
    name: spec[0] for name, spec in IFEVAL_SPECS.items()
}
IFEVAL_REQUIRED: dict[str, tuple[str, ...]] = {name: spec[1] for name, spec in IFEVAL_SPECS.items()}


def _spec_for(instruction_id: str) -> tuple[Checker, tuple[str, ...]]:
    spec = IFEVAL_SPECS.get(instruction_id)
    if spec is not None:
        return spec
    from vfates.tasks.ifbench_checkers import IFBENCH_SPECS

    found = IFBENCH_SPECS.get(instruction_id)
    if found is None:
        raise KeyError(f"unknown constraint id {instruction_id!r}")
    return found


def prepare_constraint(instruction_id: str, kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return kwargs for ``instruction_id``. Unknown ids and missing keys raise.

    ``None`` means no arguments. A required key whose value is ``None`` is missing.
    """
    _fn, required = _spec_for(instruction_id)
    args = {} if kwargs is None else dict(kwargs)
    missing = [key for key in required if args.get(key) is None]
    if missing:
        raise ValueError(f"{instruction_id} missing kwargs {missing}")
    return {key: value for key, value in args.items() if value is not None}


def apply_instruction(text: str, instruction_id: str, kwargs: Mapping[str, Any] | None) -> bool:
    """Run the IFEval or IFBench checker for ``instruction_id``. Unknown ids raise."""
    fn, _required = _spec_for(instruction_id)
    args = (
        {} if kwargs is None else {key: value for key, value in kwargs.items() if value is not None}
    )
    return bool(fn(text, **args))


def known_instruction_ids() -> frozenset[str]:
    """The 25 IFEval instruction ids plus the 29 IFBench training ids."""
    from vfates.tasks.ifbench_checkers import IFBENCH_TRAIN_IDS

    return frozenset(IFEVAL_INSTRUCTION_IDS) | frozenset(IFBENCH_TRAIN_IDS)
