"""Positive and negative cases for the 29 IFBench training checkers."""

from __future__ import annotations

import pytest

from vfates.tasks.if_checkers import known_instruction_ids
from vfates.tasks.ifbench_checkers import (
    IFBENCH_REQUIRED,
    IFBENCH_TRAIN_IDS,
    apply_ifbench,
    word_tokenize,
)

# (instruction id, kwargs or None, response that passes, response that fails)
_CASES: list[tuple[str, dict | None, str, str]] = [
    (
        "copy:repeat_phrase",
        {"phrase": "Dance like nobody is watching you", "small_n": 2},
        "Dance like somebody is watching you Dance like nobody was watching you",
        "Dance like nobody is watching you Dance like nobody is watching you",
    ),
    ("copy:copy", {"prompt_to_repeat": "Hello"}, "Hello", "Hello there"),
    (
        "new:copy_span_idx",
        {"prompt_to_repeat": "abcdefghij", "n_start": 2, "n_end": 5},
        "cde",
        "abc",
    ),
    (
        "detectable_format:sentence_hyphens",
        None,
        "Hello world.-This is next.",
        "Hello-world.",
    ),
    ("keywords:no_adjacent_consecutive", None, "cat sat mat", "cat dog"),
    ("detectable_format:square_brackets", None, "[Hello] [world]", "[Hello] world"),
    ("keywords:word_once", {"keyword": "coast"}, "the coast is calm", "the coast and coast"),
    (
        "keywords:word_count_different_numbers",
        {"keyword": "cat", "frequency": 2, "relation": "at least"},
        "cat cat",
        "cat",
    ),
    ("keywords:exclude_word_harder", {"keyword": "cat"}, "a dog sat", "a cat sat"),
    ("paragraphs:paragraphs", None, "Alpha\n***\nBeta", "Alpha"),
    ("paragraphs:paragraphs2", None, "Alpha\n\nBeta", "Alpha\nBeta"),
    (
        "first_word:first_word_sent",
        {"first_word": "Alpha"},
        "Alpha goes here. Alpha ends well.",
        "Alpha goes here. Beta ends well.",
    ),
    (
        "first_word:first_word_answer",
        {"first_word": "Hello"},
        "Hello there friend",
        "There hello friend",
    ),
    (
        "last_word:last_word_sent",
        {"last_word": "dad"},
        "This is dad. That is dad.",
        "This is dad. That is mom.",
    ),
    (
        "last_word:last_word_answer",
        {"last_word": "brief"},
        "The answer is brief",
        "brief is not last",
    ),
    ("detectable_format:bigram_wrapping", None, "<<I am>> <<at home>>", "I am at home"),
    ("copy:copying_simple", {"prompt_to_repeat": "Hello"}, "Hello", "Hello there"),
    (
        "copy:copying_multiple",
        {"prompt_to_repeat": "Hello", "N": 2},
        "Hello******Hello",
        "Hello******World",
    ),
    ("punctuation:punctuation_dot", None, "no dots here", "one dot."),
    ("punctuation:punctuation_exclamation", None, "no bangs", "wow!"),
    ("count:lowercase_counting", {"N": 2}, "Hello World", "one two three"),
    (
        "letters:letter_counting",
        {"N": 3, "relation": "at least"},
        "abcd",
        "ab",
    ),
    (
        "letters:letter_counting2",
        {"letter": "a", "let_frequency": 2, "let_relation": "at least"},
        "a ba",
        "bbb",
    ),
    (
        "count:counting_composition",
        {"n_sent": 1, "n_words": 2},
        "Hello world\n***\nHello world\n***\nHello world",
        "Hello world\n***\nHello world",
    ),
    ("count:count_unique", None, "alpha beta", "alpha alpha"),
    (
        "count:count_increment_word",
        {"keyword1": "passenger", "keyword2": "routine"},
        "see passenger and routine routine",
        "passenger passenger routine",
    ),
    ("keywords:palindrome", None, "a cat", "go to"),
    (
        "keywords:keyword_specific_position",
        {"keyword": "cat", "n": 1, "m": 2},
        "The cat sat",
        "The dog sat",
    ),
    ("keywords:start_end", None, "alpha beta alpha", "alpha beta"),
]


def test_training_set_has_29_ids() -> None:
    assert len(IFBENCH_TRAIN_IDS) == 29
    assert len(set(IFBENCH_TRAIN_IDS)) == 29
    assert {case[0] for case in _CASES} == set(IFBENCH_TRAIN_IDS)
    assert set(IFBENCH_TRAIN_IDS) <= known_instruction_ids()
    assert len(known_instruction_ids()) == 54


@pytest.mark.parametrize(("instruction_id", "kwargs", "positive", "negative"), _CASES)
def test_checker_positive_and_negative(
    instruction_id: str, kwargs: dict | None, positive: str, negative: str
) -> None:
    assert apply_ifbench(positive, instruction_id, kwargs)
    assert not apply_ifbench(negative, instruction_id, kwargs)
    required = IFBENCH_REQUIRED[instruction_id]
    if kwargs is None:
        assert required == ()
    else:
        assert set(required) <= set(kwargs)


def test_unknown_instruction_id_raises() -> None:
    with pytest.raises(KeyError, match="unknown constraint id"):
        apply_ifbench("hello", "not:a-real-constraint", None)


def test_less_than_relations_and_copy_span_are_half_open() -> None:
    assert apply_ifbench("ab", "letters:letter_counting", {"N": 3, "relation": "less than"})
    assert not apply_ifbench("abcd", "letters:letter_counting", {"N": 3, "relation": "less than"})
    assert apply_ifbench(
        "cat",
        "keywords:word_count_different_numbers",
        {"keyword": "cat", "frequency": 2, "relation": "less than"},
    )
    # n_end is exclusive, matching the IFEvalG slice.
    assert apply_ifbench(
        "cde",
        "new:copy_span_idx",
        {"prompt_to_repeat": "abcdefghij", "n_start": 2, "n_end": 5},
    )
    assert not apply_ifbench(
        "cdef",
        "new:copy_span_idx",
        {"prompt_to_repeat": "abcdefghij", "n_start": 2, "n_end": 5},
    )


def test_word_tokenize_matches_treebank_examples() -> None:
    assert word_tokenize("They'll save and invest more.") == [
        "They",
        "'ll",
        "save",
        "and",
        "invest",
        "more",
        ".",
    ]
    assert word_tokenize("don't stop") == ["do", "n't", "stop"]
    assert word_tokenize("hi, my name can't hello,") == [
        "hi",
        ",",
        "my",
        "name",
        "ca",
        "n't",
        "hello",
        ",",
    ]
    assert word_tokenize("Hello world. Hello there.") == [
        "Hello",
        "world",
        ".",
        "Hello",
        "there",
        ".",
    ]
    assert word_tokenize("OK and FINE now") == ["OK", "and", "FINE", "now"]
