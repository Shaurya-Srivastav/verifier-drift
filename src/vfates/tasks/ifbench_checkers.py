# Copyright Allen Institute for AI and the Google Research Authors.
# SPDX-License-Identifier: Apache-2.0
#
# The 29 IFBench training constraints (the "New Constraints" block in
# open-instruct IFEvalG). allenai/IF_multi_constraints_upto5 samples these
# plus IFEval's 25. Ported from Apache-2.0 sources:
# https://github.com/allenai/open-instruct/blob/main/open_instruct/IFEvalG/instructions.py
# https://github.com/allenai/open-instruct/blob/main/open_instruct/IFEvalG/instructions_registry.py
# https://github.com/allenai/open-instruct/blob/main/open_instruct/IFEvalG/instructions_util.py
# The IFBench repository (https://github.com/allenai/IFBench) holds the OOD
# test verifiers, which this dataset does not use.
#
# ``split_into_sentences`` follows ``instructions_util.split_into_sentences``.
# ``word_tokenize`` applies those sentences, then the Penn Treebank rules of
# ``nltk.tokenize.NLTKWordTokenizer`` (the second step of ``nltk.word_tokenize``).
# Punkt is not required. ``count_words`` is ``re.findall(r"\\w+")``, the same
# tokens as NLTK's ``RegexpTokenizer(r"\\w+")``. No third-party import.
"""IFBench training constraint checkers, dispatched by instruction id.

Each checker follows IFEvalG ``check_following`` and the kwargs stored in
``ground_truth`` (the arguments ``build_description`` recorded). Keyword
patterns are not escaped: a stored keyword is a regular expression, as in
IFEvalG. An unknown relation fails closed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

Checker = Callable[..., bool]

_LESS = "less than"
_AT_LEAST = "at least"


def relation_holds(actual: int, threshold: int, relation: str) -> bool:
    """``less than`` is strict. ``at least`` is inclusive. Anything else is false."""
    if relation == _LESS:
        return actual < threshold
    if relation == _AT_LEAST:
        return actual >= threshold
    return False


def count_words(text: str) -> int:
    """Word count used by ``length_constraints:number_words`` (``\\w+`` tokens)."""
    return len(re.findall(r"\w+", text))


_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s"
    r"|However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text: str) -> list[str]:
    """Regex sentence split from IFEvalG ``instructions_util.split_into_sentences``."""
    body = " " + text + "  "
    body = body.replace("\n", " ")
    body = re.sub(_PREFIXES, "\\1<prd>", body)
    body = re.sub(_WEBSITES, "<prd>\\1", body)
    body = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", body)
    body = re.sub(_MULTIPLE_DOTS, lambda match: "<prd>" * len(match.group(0)) + "<stop>", body)
    if "Ph.D" in body:
        body = body.replace("Ph.D.", "Ph<prd>D<prd>")
    body = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", body)
    body = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", body)
    body = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        body,
    )
    body = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", body)
    body = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", body)
    body = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", body)
    body = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", body)
    if "”" in body:
        body = body.replace(".”", "”.")
    if '"' in body:
        body = body.replace('."', '".')
    if "!" in body:
        body = body.replace('!"', '"!')
    if "?" in body:
        body = body.replace('?"', '"?')
    body = body.replace(".", ".<stop>")
    body = body.replace("?", "?<stop>")
    body = body.replace("!", "!<stop>")
    body = body.replace("<prd>", ".")
    sentences = [part.strip() for part in body.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


# Penn Treebank rules from nltk.tokenize.destructive.NLTKWordTokenizer.
# ``word_tokenize`` runs this on each regex sentence, which is what
# ``nltk.word_tokenize`` does after Punkt. Compared on single sentences and on
# ordinary multi-sentence prose the tokens match; Punkt and this splitter
# still disagree on some abbreviations.
_STARTING_QUOTES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"([«“‘„]|[`]+)"), r" \1 "),
    (re.compile(r"^\""), "``"),
    (re.compile(r"(``)"), r" \1 "),
    (re.compile(r"([ \(\[{<])(\"|'{2})"), r"\1 `` "),
    (re.compile(r"(?i)(?<!\w)(')(?!(?:re|ve|ll|m|t|s|d|n)\b)(?=\w)"), r"\1 "),
)
_PUNCTUATION: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"""([^\.])(\.)([\]\)}>"'»”’ ]*)\s*$"""), r"\1 \2 \3 "),
    (re.compile(r"([:,])([^\d])"), r" \1 \2"),
    (re.compile(r"([:,])$"), r" \1 "),
    (re.compile(r"\.{2,}"), r" \g<0> "),
    (re.compile(r"[;@#$%&]"), r" \g<0> "),
    (re.compile(r"[\u2012-\u2015]"), r" \g<0> "),
    (re.compile(r"""([^\.])(\.)([\]\)}>"']*)\s*$"""), r"\1 \2\3 "),
    (re.compile(r"[?!]"), r" \g<0> "),
    (re.compile(r"([^'])' "), r"\1 ' "),
    (re.compile(r"[*]"), r" \g<0> "),
)
_ENDING_QUOTES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"([»”’])"), r" \1 "),
    (re.compile(r"''"), " '' "),
    (re.compile(r'"'), " '' "),
    (re.compile(r"\s+"), " "),
    (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
    (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
)
_PARENS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")
_DASH = (re.compile(r"--"), r" -- ")
_CONTRACTIONS2 = (
    re.compile(r"(?i)\b(can)(?#X)(not)\b"),
    re.compile(r"(?i)\b(d)(?#X)('ye)\b"),
    re.compile(r"(?i)\b(gim)(?#X)(me)\b"),
    re.compile(r"(?i)\b(gon)(?#X)(na)\b"),
    re.compile(r"(?i)\b(got)(?#X)(ta)\b"),
    re.compile(r"(?i)\b(lem)(?#X)(me)\b"),
    re.compile(r"(?i)\b(more)(?#X)('n)\b"),
    re.compile(r"(?i)\b(wan)(?#X)(na)(?=\s)"),
)
_CONTRACTIONS3 = (
    re.compile(r"(?i) ('t)(?#X)(is)\b"),
    re.compile(r"(?i) ('t)(?#X)(was)\b"),
)


def _treebank_tokenize(text: str) -> list[str]:
    """Tokenize one sentence the way ``NLTKWordTokenizer.tokenize`` does."""
    for pattern, substitution in _STARTING_QUOTES:
        text = pattern.sub(substitution, text)
    for pattern, substitution in _PUNCTUATION:
        text = pattern.sub(substitution, text)
    text = _PARENS[0].sub(_PARENS[1], text)
    text = _DASH[0].sub(_DASH[1], text)
    text = " " + text + " "
    for pattern, substitution in _ENDING_QUOTES:
        text = pattern.sub(substitution, text)
    for pattern in _CONTRACTIONS2:
        text = pattern.sub(r" \1 \2 ", text)
    for pattern in _CONTRACTIONS3:
        text = pattern.sub(r" \1 \2 ", text)
    return text.split()


def word_tokenize(text: str) -> list[str]:
    """``nltk.word_tokenize`` stand-in: regex sentences, then Treebank tokens."""
    if not text or not text.strip():
        return []
    sentences = split_into_sentences(text)
    if not sentences:
        return _treebank_tokenize(text)
    tokens: list[str] = []
    for sentence in sentences:
        tokens.extend(_treebank_tokenize(sentence))
    return tokens


def _occurrences(text: str, keyword: str) -> int:
    """Case-insensitive substring count. ``keyword`` is a regular expression."""
    return len(re.findall(keyword, text, flags=re.IGNORECASE))


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value[0]).strip()


def check_repeat_phrase(text: str, phrase: str, small_n: int) -> bool:
    """Repeat ``phrase`` exactly ``small_n`` times, each changing one word."""
    phrase = phrase.strip()
    ref = phrase.split()
    found = re.findall(rf"{ref[0]} .*? {ref[-1]}", text)
    if len(found) != small_n:
        return False
    satisfied = 0
    for piece in found:
        parts = piece.split()
        if len(parts) != len(ref):
            return False
        differences = 0
        for index, token in enumerate(parts):
            if token != ref[index]:
                differences += 1
                if differences > 1:
                    return False
        if differences == 1:
            satisfied += 1
    return satisfied == small_n


def check_copy(text: str, prompt_to_repeat: str) -> bool:
    """Response equals ``prompt_to_repeat`` after strip and case-fold."""
    return text.strip().lower() == prompt_to_repeat.strip().lower()


def check_copy_span(text: str, prompt_to_repeat: str, n_start: int, n_end: int) -> bool:
    """Response equals ``prompt_to_repeat[n_start:n_end]`` (end index exclusive)."""
    span = prompt_to_repeat[int(n_start) : int(n_end)]
    return text.strip().lower() == span.strip().lower()


def check_sentence_hyphens(text: str) -> bool:
    """Hyphen pieces equal the sentences of the same text with hyphens turned into spaces."""
    gold = split_into_sentences(re.sub("-", " ", text))
    pieces = text.split("-")
    for piece, sentence in zip(pieces, gold, strict=False):
        if piece.strip() != piece or piece != sentence:
            return False
    return True


def check_no_adjacent_consecutive(text: str) -> bool:
    """No two adjacent words start with consecutive letters."""
    words = text.split()
    for index in range(len(words) - 1):
        first = words[index][0].lower()
        second = words[index + 1][0].lower()
        if len(first) != 1 or len(second) != 1:
            return False
        if ord(second) - ord(first) == 1:
            return False
    return True


def check_square_brackets(text: str) -> bool:
    """Every whitespace word starts with ``[`` and ends with ``]``."""
    return all(word.startswith("[") and word.endswith("]") for word in text.split())


def check_word_once(text: str, keyword: str) -> bool:
    """``keyword`` occurs once as a case-insensitive regular expression."""
    return _occurrences(text, keyword.strip()) == 1


def check_word_count_different(text: str, keyword: str, frequency: int, relation: str) -> bool:
    """Same count as keyword frequency. The description omits the relation words."""
    return relation_holds(_occurrences(text, keyword.strip()), int(frequency), relation)


def check_exclude_word_harder(text: str, keyword: str) -> bool:
    """True when `` keyword `` (spaces on both sides) is absent. Not case-folded."""
    return f" {keyword.strip()} " not in text


def _exact_paragraphs(text: str, pattern: str, expected: int) -> bool:
    paragraphs = re.split(pattern, text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == expected


def check_paragraphs(text: str) -> bool:
    """Exactly two paragraphs separated by ``***``."""
    return _exact_paragraphs(text, r"\s?\*\*\*\s?", 2)


def check_paragraphs2(text: str) -> bool:
    """Exactly two paragraphs separated by a blank line."""
    return _exact_paragraphs(text, r"\n\n", 2)


def check_first_word_sent(text: str, first_word: Any) -> bool:
    """Every sentence's first whitespace word is ``first_word``."""
    expected = _as_str(first_word).lower()
    for sentence in split_into_sentences(text):
        if not sentence.strip():
            return False
        if sentence.split()[0].strip().lower() != expected:
            return False
    return True


def check_first_word_answer(text: str, first_word: str) -> bool:
    """The response's first whitespace word is ``first_word``."""
    if not text.strip() or not text.split():
        return False
    return text.split()[0].strip().lower() == first_word.strip().lower()


def check_last_word_sent(text: str, last_word: Any) -> bool:
    """Every sentence's last word, with punctuation removed, is ``last_word``."""
    expected = _as_str(last_word).lower()
    for sentence in split_into_sentences(text):
        if not sentence.strip():
            return False
        word = re.sub(r"[^\w\s]", "", sentence.split()[-1].strip())
        if word.lower() != expected:
            return False
    return True


def check_last_word_answer(text: str, last_word: str) -> bool:
    """The response's last whitespace word, with punctuation removed, is ``last_word``."""
    word = re.sub(r"[^\w\s]", "", text.split()[-1].strip())
    return word.lower() == last_word.strip().lower()


def check_bigram_wrapping(text: str) -> bool:
    """Each successive pair of whitespace words is ``<<word`` and ``word>>``."""
    words = text.split()
    for index in range(0, len(words) - 1, 2):
        if index + 1 < len(words) and not (
            words[index].startswith("<<") and words[index + 1].endswith(">>")
        ):
            return False
    return True


def check_copying_multiple(text: str, prompt_to_repeat: str, N: int) -> bool:
    """``N`` copies of the prompt separated by ``******``, and nothing else."""
    parts = text.split("******")
    if len(parts) != int(N):
        return False
    target = prompt_to_repeat.strip().lower()
    return all(part.strip().lower() == target for part in parts)


def check_punctuation_dot(text: str) -> bool:
    """No ``.`` anywhere in the response."""
    return re.search(r"\.", text) is None


def check_punctuation_exclamation(text: str) -> bool:
    """No ``!`` anywhere in the response."""
    return re.search(r"\!", text) is None


def check_lowercase_counting(text: str, N: int) -> bool:
    """At most ``N`` fully lowercase words (``\\b[a-z]+\\b``)."""
    return len(re.findall(r"\b[a-z]+\b", text)) <= int(N)


def check_letter_counting(text: str, N: int, relation: str) -> bool:
    """Alphabetic character count under ``relation``."""
    return relation_holds(len(re.findall(r"[a-zA-Z]", text)), int(N), relation)


def check_letter_counting2(text: str, letter: str, let_frequency: int, let_relation: str) -> bool:
    """Case-insensitive count of one letter under ``let_relation``."""
    counted = text.lower()
    return relation_holds(counted.count(letter.strip().lower()), int(let_frequency), let_relation)


def check_counting_composition(text: str, n_sent: int, n_words: int) -> bool:
    """Three ``***`` paragraphs, each with ``n_sent`` sentences of ``n_words`` tokens.

    The prompt says the divider is ``* * *``. The checker splits on ``***``.
    Sentence and word checks run even on an empty edge paragraph.
    """
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
        sentences = split_into_sentences(paragraph)
        if len(sentences) != int(n_sent):
            return False
        for sentence in sentences:
            if len(word_tokenize(sentence)) != int(n_words):
                return False
    return count == 3


def check_count_unique(text: str) -> bool:
    """No Treebank token is repeated."""
    words = word_tokenize(text)
    return len(words) == len(set(words))


def check_count_increment_word(text: str, keyword1: str, keyword2: str) -> bool:
    """``keyword1`` once and ``keyword2`` twice, as case-insensitive regular expressions."""
    first = _occurrences(text, keyword1.strip())
    second = _occurrences(text, keyword2.strip())
    return first == 1 and second == 2


def check_palindrome(text: str) -> bool:
    """Some whitespace word equals its reverse. A one-letter word counts."""
    return any(word == word[::-1] for word in text.split())


def check_keyword_specific_position(text: str, keyword: Any, n: int, m: int) -> bool:
    """Sentence ``n`` (1-based), Treebank word ``m``, equals ``keyword`` exactly."""
    expected = _as_str(keyword)
    sentences = split_into_sentences(text)
    if len(sentences) < int(n):
        return False
    words = word_tokenize(sentences[int(n) - 1])
    if len(words) < int(m):
        return False
    return words[int(m) - 1] == expected


def check_start_end(text: str) -> bool:
    """First and last Treebank tokens match, case-insensitive, and there are at least two."""
    words = word_tokenize(text)
    if len(words) < 2:
        return False
    return words[0].lower() == words[-1].lower()


def _entry(fn: Checker, *required: str) -> tuple[Checker, tuple[str, ...]]:
    return fn, required


# Registry order matches IFEvalG ``INSTRUCTION_DICT`` after the classic 25.
IFBENCH_SPECS: dict[str, tuple[Checker, tuple[str, ...]]] = {
    "copy:repeat_phrase": _entry(check_repeat_phrase, "phrase", "small_n"),
    "copy:copy": _entry(check_copy, "prompt_to_repeat"),
    "new:copy_span_idx": _entry(check_copy_span, "n_start", "n_end", "prompt_to_repeat"),
    "detectable_format:sentence_hyphens": _entry(check_sentence_hyphens),
    "keywords:no_adjacent_consecutive": _entry(check_no_adjacent_consecutive),
    "detectable_format:square_brackets": _entry(check_square_brackets),
    "keywords:word_once": _entry(check_word_once, "keyword"),
    "keywords:word_count_different_numbers": _entry(
        check_word_count_different, "keyword", "frequency", "relation"
    ),
    "keywords:exclude_word_harder": _entry(check_exclude_word_harder, "keyword"),
    "paragraphs:paragraphs": _entry(check_paragraphs),
    "paragraphs:paragraphs2": _entry(check_paragraphs2),
    "first_word:first_word_sent": _entry(check_first_word_sent, "first_word"),
    "first_word:first_word_answer": _entry(check_first_word_answer, "first_word"),
    "last_word:last_word_sent": _entry(check_last_word_sent, "last_word"),
    "last_word:last_word_answer": _entry(check_last_word_answer, "last_word"),
    "detectable_format:bigram_wrapping": _entry(check_bigram_wrapping),
    "copy:copying_simple": _entry(check_copy, "prompt_to_repeat"),
    "copy:copying_multiple": _entry(check_copying_multiple, "prompt_to_repeat", "N"),
    "punctuation:punctuation_dot": _entry(check_punctuation_dot),
    "punctuation:punctuation_exclamation": _entry(check_punctuation_exclamation),
    "count:lowercase_counting": _entry(check_lowercase_counting, "N"),
    "letters:letter_counting": _entry(check_letter_counting, "N", "relation"),
    "letters:letter_counting2": _entry(
        check_letter_counting2, "letter", "let_frequency", "let_relation"
    ),
    "count:counting_composition": _entry(check_counting_composition, "n_sent", "n_words"),
    "count:count_unique": _entry(check_count_unique),
    "count:count_increment_word": _entry(check_count_increment_word, "keyword1", "keyword2"),
    "keywords:palindrome": _entry(check_palindrome),
    "keywords:keyword_specific_position": _entry(
        check_keyword_specific_position, "keyword", "n", "m"
    ),
    "keywords:start_end": _entry(check_start_end),
}

IFBENCH_TRAIN_IDS: tuple[str, ...] = tuple(IFBENCH_SPECS)
IFBENCH_CHECKERS: dict[str, Checker] = {name: spec[0] for name, spec in IFBENCH_SPECS.items()}
IFBENCH_REQUIRED: dict[str, tuple[str, ...]] = {
    name: spec[1] for name, spec in IFBENCH_SPECS.items()
}


def apply_ifbench(text: str, instruction_id: str, kwargs: Mapping[str, Any] | None) -> bool:
    """Run one IFBench training checker. Unknown ids raise ``KeyError``."""
    spec = IFBENCH_SPECS.get(instruction_id)
    if spec is None:
        raise KeyError(f"unknown constraint id {instruction_id!r}")
    fn, _required = spec
    args = (
        {} if kwargs is None else {key: value for key, value in kwargs.items() if value is not None}
    )
    return bool(fn(text, **args))
