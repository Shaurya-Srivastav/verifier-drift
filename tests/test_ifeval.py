"""RLVR-IFeval checkers, split, and oracle = checker AND judge."""

from __future__ import annotations

import json

import pytest

from vfates.tasks.if_checkers import (
    IF_FUNCTIONS_MAP,
    apply_checker,
    parse_ground_truth,
    validate_choice,
    validate_end,
    validate_forbidden_words,
    validate_frequency_capital_words,
    validate_highlighted_sections,
    validate_json_format,
    validate_lowercase,
    validate_no_commas,
    validate_paragraphs,
    validate_placeholders,
    validate_quotation,
    validate_repeat_prompt,
    validate_sections,
    validate_title,
    validate_two_responses,
    validate_uppercase,
    validate_word_constraint,
    verify_bullet_points,
    verify_keyword_frequency,
    verify_keywords,
    verify_letter_frequency,
    verify_paragraph_count,
    verify_postscript,
    verify_sentence_constraint,
)
from vfates.tasks.ifeval import IFEvalTask
from vfates.tasks.judges import HeuristicJudge
from vfates.verifiers.natural import IFEvalCheckerVerifier

GOOD = (
    "IPv6 expands the address space and simplifies routing for hosts. "
    "It also improves autoconfiguration, so networks need less manual setup "
    "than they did under the previous protocol."
)
TULU = (
    "timberlike, unconnectedly Conversation 1 User : "
    "timberlike, unconnectedly Conversation 2 User : "
    "timberlike, unconnectedly Conversation 3 User : "
    "timberlike, unconnectedly Conversation 4 User : "
    "timberlike, unconnectedly Conversation 5 User : "
    "timberlike, unconnectedly Conversation 6 User : "
)


def test_every_checker() -> None:
    assert verify_keywords("The dog and the cat sat.", ["cat", "Dog"])
    assert not verify_keywords("dog", ["cat"])
    assert verify_keyword_frequency("Cat, cat and catalog.", "cat", 2)
    assert not verify_keyword_frequency("cat cat", "cat", 1)
    assert not validate_forbidden_words("I am cataloging", ["cat"])
    assert validate_forbidden_words("a clean sentence here", ["cat"])
    assert verify_letter_frequency("aAa", "a", 2)
    assert not verify_letter_frequency("aAa", "A", 2)
    with pytest.raises(ValueError):
        verify_letter_frequency("ab", "ab", 1)
    assert not apply_checker("ab", "verify_letter_frequency", {"letter": "ab", "N": 1})
    assert verify_paragraph_count("Alpha\n* * *\nBeta", 2)
    assert not verify_paragraph_count("Alpha\n* * *\n", 2)
    assert validate_word_constraint("one two three four", 3, "at least")
    assert validate_word_constraint("one two three four", 4, "around")
    assert not validate_word_constraint("one two", 10, "around")
    assert verify_sentence_constraint("One. Two!", 2, "at least")
    assert verify_sentence_constraint("One. Two!", 2, "around")
    assert not verify_sentence_constraint("One. Two! Three?", 2, "at most")
    text = "Alpha starts\n\nBeta follows"
    assert validate_paragraphs(text, 2, "Beta", 2)
    assert not validate_paragraphs(text, 2, "Nope", 2)
    assert verify_postscript("Done.\nP.S. more", "P.S.")
    assert not verify_postscript("P.S.", "P.S.")
    assert validate_placeholders("Hi [name] at [place].", 2)
    assert not validate_placeholders("Hi [name].", 2)
    assert verify_bullet_points("* one\n- two\ntext", 2)
    assert not verify_bullet_points("* one", 2)
    assert validate_title("<<poem of joy>>")
    assert not validate_title("poem of joy")
    assert validate_choice("I choose yes", ["yes", "no"])
    assert not validate_choice("maybe later", ["yes", "no"])
    # RLVR-IFeval stores options as one string, so the checker iterates characters.
    assert validate_choice("yes", "yes/no/maybe")
    assert not validate_choice("QQQ", "yes/no/maybe")
    assert validate_highlighted_sections("a *one* and *two*", 2)
    assert not validate_highlighted_sections("a *one*", 2)
    assert validate_sections("Intro Section: body", 2, "Section:")
    assert not validate_sections("only one", 2, "Section:")
    assert validate_json_format('{"a": 1}')
    assert not validate_json_format("{")
    assert validate_repeat_prompt("Hello world, rest", "Hello")
    assert not validate_repeat_prompt("Hello", "Hello world")
    assert validate_two_responses("foo******bar")
    assert validate_two_responses("******bar")
    assert not validate_two_responses("foo******foo")
    assert not validate_two_responses("a******b******c")
    assert validate_uppercase("HELLO")
    assert not validate_uppercase("Hello")
    assert validate_lowercase("hello")
    assert not validate_lowercase("Hello")
    assert validate_frequency_capital_words("OK and FINE now", 2, "at least")
    assert not validate_frequency_capital_words("ok", 1, "at least")
    assert validate_end("ends with done", "done")
    assert not validate_end("done now", "done")
    assert validate_quotation('"wrapped"')
    assert not validate_quotation('wrapped"')
    assert validate_no_commas("no commas here")
    assert not validate_no_commas("one, two")
    assert set(IF_FUNCTIONS_MAP) >= {
        "verify_keywords",
        "validate_choice",
        "validate_response_language",
    }
    assert len(IF_FUNCTIONS_MAP) == 25


def test_language_checker_optional() -> None:
    pytest.importorskip("langdetect")
    func = IF_FUNCTIONS_MAP["validate_response_language"]
    paragraph = (
        "This is a long English paragraph about rivers and cities. "
        "It should be detected as English by the language identifier."
    )
    assert func(paragraph, "en")


def test_parse_ground_truth_drops_nulls() -> None:
    raw = json.dumps(
        {
            "func_name": "verify_keywords",
            "N": None,
            "keyword_list": ["alpha", "beta"],
            "word": None,
        }
    )
    name, args = parse_ground_truth(raw)
    assert name == "verify_keywords"
    assert args == {"keyword_list": ["alpha", "beta"]}
    assert apply_checker("alpha and beta", name, args)
    assert not apply_checker("missing", "not_a_checker", {})


def _rows() -> list[dict]:
    rows = []
    for i in range(12):
        rows.append(
            {
                "row_id": i,
                "question": f"Question {i}. Include the keywords if asked.",
                "func_name": "validate_lowercase" if i % 2 == 0 else "verify_keywords",
                "args": {} if i % 2 == 0 else {"keyword_list": ["timberlike", "unconnectedly"]},
                "constraint_type": "t",
                "constraint": "c",
            }
        )
    return rows


def test_split_is_deterministic_and_complementary() -> None:
    task = IFEvalTask(heldout_fraction=0.5, rows=_rows())
    train = task.problems("train", 100, seed=0)
    held = task.problems("heldout", 100, seed=0)
    assert len(train) + len(held) == 12
    assert {item.pid for item in train}.isdisjoint({item.pid for item in held})
    assert task.problems("train", 100, seed=0)[0].pid == train[0].pid
    assert len(task.problems("train", 2, seed=0)) == 2
    assert train[0].question.startswith("Question")
    assert train[0].meta["func_name"]


def test_oracle_requires_checker_and_content() -> None:
    task = IFEvalTask(rows=_rows(), judge=HeuristicJudge(), heldout_fraction=0.0)
    keyword = next(
        p for p in task.problems("train", 20, seed=0) if p.meta["func_name"] == "verify_keywords"
    )
    prose = (
        "The timberlike forest stood unconnectedly apart from the road. "
        "A walker still found a path and described the trees in ordinary sentences."
    )
    parsed = task.parse(keyword, prose)
    assert parsed.answer is not None
    assert parsed.raw == prose
    verifier = IFEvalCheckerVerifier()
    assert verifier.verdict(keyword, parsed, prose, 0) == 1
    assert verifier.verdict(keyword, parsed, prose, 1) == 1
    assert task.oracle(keyword, parsed, prose) == 1
    stuffed = task.parse(keyword, TULU)
    assert verifier.verdict(keyword, stuffed, TULU, 0) == 1
    assert task.oracle(keyword, stuffed, TULU) == 0
    assert task.prompt(keyword) == keyword.question
    empty = task.parse(keyword, "   ")
    assert empty.answer is None


def test_lowercase_checker_rejects_capitals() -> None:
    task = IFEvalTask(rows=_rows(), heldout_fraction=0.0)
    problem = next(
        p for p in task.problems("train", 20, seed=0) if p.meta["func_name"] == "validate_lowercase"
    )
    assert task.oracle(problem, task.parse(problem, GOOD.lower()), GOOD.lower()) == 1
    assert IFEvalCheckerVerifier().verdict(problem, task.parse(problem, GOOD), GOOD, 0) == 0


def _skip_offline(exc: BaseException) -> None:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "connection",
        "offline",
        "timed out",
        "timeout",
        "name resolution",
        "network",
        "huggingface",
        "urlerror",
    )
    if any(needle in text for needle in needles):
        pytest.skip(str(exc))
    raise exc


@pytest.mark.network
def test_dataset_covers_every_func_name() -> None:
    try:
        task = IFEvalTask()
        records = task._records()
    except Exception as exc:
        _skip_offline(exc)
        return
    assert len(records) == 14973
    names = {row["func_name"] for row in records}
    assert names <= set(IF_FUNCTIONS_MAP)
    assert names
    train = task.problems("train", 10**9, seed=0)
    held = task.problems("heldout", 10**9, seed=0)
    assert len(train) + len(held) == 14973
    assert {item.pid for item in train}.isdisjoint({item.pid for item in held})
    sample = next(row for row in records if row["func_name"] == "validate_json_format")
    assert apply_checker('{"ok": true}', sample["func_name"], sample["args"])
    assert not apply_checker("not json", sample["func_name"], sample["args"])
