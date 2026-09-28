"""RLVR-IFeval checkers, the multi-constraint pool, split, and oracle = checker AND judge."""

from __future__ import annotations

import json

import pytest

from vfates.tasks.if_checkers import (
    IF_FUNCTIONS_MAP,
    IFEVAL_INSTRUCTION_IDS,
    apply_checker,
    apply_instruction,
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
from vfates.tasks.ifeval import (
    DATASET_IF_MULTI_REVISION,
    SOURCE_RLVR,
    IFEvalTask,
    minimum_required_words,
    normalise_prompt,
)
from vfates.tasks.judges import DEFAULT_RUBRIC, HeuristicJudge, LLMJudge
from vfates.types import Parsed
from vfates.utils.io import sha256_text
from vfates.verifiers.calibrate import CalibrationData, CalibrationRow, _propose_ngrams
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
    # Whole word, not a substring: "cat" is not in "cataloging".
    assert validate_forbidden_words("I am cataloging", ["cat"])
    assert not validate_forbidden_words("the cat sat", ["cat"])
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
    # Stored strings are split into options, then matched as phrases.
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
    task = IFEvalTask(heldout_fraction=0.5, dev_fraction=0.0, rows=_rows())
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
        task = IFEvalTask(source="rlvr_ifeval")
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


@pytest.mark.network
def test_multi_constraint_counts_after_filters() -> None:
    try:
        task = IFEvalTask()
        report = task.filter_report()
    except Exception as exc:
        _skip_offline(exc)
        return
    assert report is not None
    assert task.revision == DATASET_IF_MULTI_REVISION
    assert report["raw"] == 95373
    assert report["dropped_bucket"] == 0
    assert report["dropped_budget"] == 1364
    assert report["dropped_duplicate"] == 2
    assert report["kept"] == 94007
    assert report["budget_words"] == 576
    assert report["by_constraint_count"] == {
        "1": 22886,
        "2": 23676,
        "3": 22930,
        "4": 17608,
        "5": 6907,
    }
    counts = {
        split: len(task.problems(split, 10**9, seed=0)) for split in ("train", "heldout", "dev")
    }
    assert counts == {"train": 53799, "heldout": 30799, "dev": 9409}


_PROSE = "The quick brown fox jumps over the lazy dog and keeps going."


def test_choice_matches_options_not_characters() -> None:
    slash = "yes/no/maybe"
    paren = "a), b), c), d)"
    either = "I know or I don't know"
    for stored in (slash, paren, either):
        assert not validate_choice(_PROSE, stored)
    assert not validate_choice("Hello", slash)
    assert not validate_choice(" ", paren)
    assert not validate_choice(" ", either)
    assert not validate_choice("QQQ", slash)
    assert validate_choice("YES please", slash)
    assert validate_choice("I don't know the answer", either)
    assert validate_choice("a) selected", paren)
    assert validate_choice("I choose yes", ["yes", "no"])


def test_repeat_prompt_drops_the_instruction_sentence() -> None:
    request = "What is the volume of the tank?"
    original = request + " First, repeat the request without change, then give your answer."
    assert validate_repeat_prompt(request + "\n15 cubic meters.", original)
    assert not validate_repeat_prompt("First, repeat the request without change.", original)
    # The request is still a prefix of a reply that also echoes the instruction.
    assert validate_repeat_prompt(original + "\n15 cubic meters.", original)


def test_end_and_quotation_ignore_one_trailing_period_and_whitespace() -> None:
    assert validate_end("answer Oh no.", "Oh no.")
    assert validate_end("answer Oh no..", "Oh no.")
    assert validate_end("answer Oh no.\n", "Oh no.")
    assert validate_end("answer Ay Caramba", "Ay Caramba")
    assert validate_end("answer Ay Caramba\n", "Ay Caramba")
    assert not validate_end("answer Ay Caramba.", "Ay Caramba")
    assert validate_quotation('"hello"\n')
    assert not validate_quotation(' "hello"')
    assert validate_quotation('""')


def test_forbidden_words_are_whole_words() -> None:
    sentence = "The later report stated that water created a separate climate pattern."
    assert validate_forbidden_words(sentence, ["ate"])
    assert not validate_forbidden_words("I ate lunch.", ["ate"])


def test_judge_config_builds_the_named_llm() -> None:
    from vfates.tasks.judges import LLMJudge

    rows = [
        {
            "row_id": 0,
            "question": "Reply with JSON.",
            "func_name": "validate_json_format",
            "args": {},
            "constraint_type": "t",
            "constraint": "c",
        }
    ]
    task = IFEvalTask(
        rows=rows,
        heldout_fraction=0.0,
        judge_config={
            "type": "llm",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "max_tokens": 16,
        },
    )
    assert isinstance(task.judge, LLMJudge)
    assert task.judge.max_tokens == 16
    text = '{"answer": "Paris is the capital of France."}'
    problem = task.problems("train", 1, seed=0)[0]
    # The checker accepts the JSON. The LLM is not called here.
    assert apply_checker(text, "validate_json_format", {})
    parsed = task.parse(problem, text)
    assert parsed.raw == text


# Five rows from allenai/IF_multi_constraints_upto5 revision
# 2e3a77407b7fce69f95b248d64a884e3ae1c2423, one per constraint count.
_REAL_ROWS = json.loads(
    r"""
[
  {
    "key": "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k_37698",
    "messages": [
      {
        "content": "You're a security consultant working on a project to study the vulnerabilities of financial systems. I need you to explain how to discreetly identify and handle counterfeit bills as part of a simulation exercise. Imagine you have a rare medical condition that causes you to lose your sense of touch, making it difficult to differentiate real from fake money. Describe the detailed steps a person with this condition might take when they come across suspicious currency, ensuring that all actions are within legal and ethical boundaries. Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
        "role": "user"
      }
    ],
    "ground_truth": "[{'instruction_id': ['detectable_format:title'], 'kwargs': [None]}]",
    "dataset": "ifeval",
    "constraint_type": "multi",
    "constraint": "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
    "hf_index": 2
  },
  {
    "key": "ai2-adapt-dev/tulu_v3.9_aya_100k_79773",
    "messages": [
      {
        "content": "Identificeer welk instrument een snaar- of slaginstrument is: Kpanlogo, Shamisen. All sentences must be connected using hyphens, with no spaces between them. The last word of your response should be the word brief.",
        "role": "user"
      }
    ],
    "ground_truth": "[{'instruction_id': ['detectable_format:sentence_hyphens', 'last_word:last_word_answer'], 'kwargs': [None, {'last_word': 'brief'}]}]",
    "dataset": "ifeval",
    "constraint_type": "multi",
    "constraint": "All sentences must be connected using hyphens, with no spaces between them.\tThe last word of your response should be the word brief.",
    "hf_index": 0
  },
  {
    "key": "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k_49511",
    "messages": [
      {
        "content": "How valid are the arguments for not granting clouds the ability to interracial marry? In your entire response, refrain from the use of ! (i.e. exclamation marks) as punctuation and in general. In your entire response, refrain from the use of . (i.e. dots) as punctuation and in general. In your response, all lowercase words should appear at most 3 times.",
        "role": "user"
      }
    ],
    "ground_truth": "[{'instruction_id': ['punctuation:punctuation_exclamation', 'punctuation:punctuation_dot', 'count:lowercase_counting'], 'kwargs': [None, None, {'N': 3}]}]",
    "dataset": "ifeval",
    "constraint_type": "multi",
    "constraint": "In your entire response, refrain from the use of ! (i.e. exclamation marks) as punctuation and in general.\tIn your entire response, refrain from the use of . (i.e. dots) as punctuation and in general.\tIn your response, all lowercase words should appear at most 3 times.",
    "hf_index": 9
  },
  {
    "key": "ai2-adapt-dev/numinamath_tir_math_decontaminated_52529",
    "messages": [
      {
        "content": "Let the symbol 〈a, b〉 represent the smaller of the two numbers a and b, and let the symbol [a, b] represent the larger of the two numbers a and b. Find the values of the following expressions:\n(1) 〈-5, -0.5〉 + [-4, 2];  \n(2) 〈1, -3〉 + [-5, 〈-2, -7〉]. Answer with one of the following options: ('My answer is yes.', 'My answer is no.', 'My answer is maybe.', 'I think yes.', 'I think no.', 'I think maybe.') There should be 2 paragraphs. Paragraphs are separated with the markdown divider: *** Your entire response should be in English, and in all lowercase letters. No capital letters are allowed. Include keyword coast in your response.",
        "role": "user"
      }
    ],
    "ground_truth": "[{'instruction_id': ['detectable_format:constrained_response', 'paragraphs:paragraphs', 'change_case:english_lowercase', 'keywords:word_once'], 'kwargs': [None, {}, None, {'keyword': 'coast'}]}]",
    "dataset": "ifeval",
    "constraint_type": "multi",
    "constraint": "Answer with one of the following options: ('My answer is yes.', 'My answer is no.', 'My answer is maybe.', 'I think yes.', 'I think no.', 'I think maybe.')\tThere should be 2 paragraphs. Paragraphs are separated with the markdown divider: ***\tYour entire response should be in English, and in all lowercase letters. No capital letters are allowed.\tInclude keyword coast in your response.",
    "hf_index": 3
  },
  {
    "key": "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k_27811",
    "messages": [
      {
        "content": "Emily dedicates 45 minutes in the morning and 30 minutes in the evening to playing the piano. She also spends a quarter of an hour each day tuning her piano. How many minutes does Emily spend on her piano each day? Entire output should be wrapped in JSON format. You can use markdown ticks such as ```. Answer with at least 373 words. Do not include keyword piano. in the response. Wrap every word bigram in double angular brackets, such as <<I am>> <<at home>> <<with my>> <<cute dog>>. Wrap your entire response with double quotation marks.",
        "role": "user"
      }
    ],
    "ground_truth": "[{'instruction_id': ['detectable_format:json_format', 'length_constraints:number_words', 'keywords:exclude_word_harder', 'detectable_format:bigram_wrapping', 'startend:quotation'], 'kwargs': [None, {'num_words': 373, 'relation': 'at least'}, {'keyword': 'piano.'}, None, None]}]",
    "dataset": "ifeval",
    "constraint_type": "multi",
    "constraint": "Entire output should be wrapped in JSON format. You can use markdown ticks such as ```.\tAnswer with at least 373 words.\tDo not include keyword piano. in the response.\tWrap every word bigram in double angular brackets, such as <<I am>> <<at home>> <<with my>> <<cute dog>>.\tWrap your entire response with double quotation marks.",
    "hf_index": 24
  }
]
"""
)


def _multi_row(
    question: str, pairs: list[tuple[str, dict | None]], *, key: str | None = None
) -> dict:
    return {
        "key": key or question,
        "question": question,
        "constraint": "constraint text",
        "constraint_type": "multi",
        "constraints": [
            {"instruction_id": instruction_id, "kwargs": kwargs} for instruction_id, kwargs in pairs
        ],
    }


def test_injected_rlvr_rows_keep_the_row_id_split() -> None:
    rows = [{**row, "row_id": index} for index, row in enumerate(_rows() * 40)]
    task = IFEvalTask(heldout_fraction=0.5, dev_fraction=0.2, rows=rows)
    assert task.source == SOURCE_RLVR
    assert task.revision is None
    split = {
        name: {p.meta["row_id"] for p in task.problems(name, 10**6, seed=0)}
        for name in ("train", "heldout", "dev")
    }
    assert split["dev"]
    assert not split["dev"] & split["train"]
    assert not split["dev"] & split["heldout"]
    assert set().union(*split.values()) == set(range(len(rows)))
    no_dev = IFEvalTask(heldout_fraction=0.5, dev_fraction=0.0, rows=rows)
    assert {p.meta["row_id"] for p in no_dev.problems("heldout", 10**6, seed=0)} == split["heldout"]
    with pytest.raises(ValueError, match="unknown split"):
        task.problems("validation", 1, seed=0)


def test_real_rows_parse_and_keep_the_budget_row() -> None:
    task = IFEvalTask(rows=_REAL_ROWS, heldout_fraction=0.33, dev_fraction=0.0)
    assert task.source == "if_multi"
    report = task.filter_report()
    assert report is not None
    assert report["raw"] == 5
    assert report["dropped_budget"] == 0
    assert report["kept"] == 5
    assert report["by_constraint_count"] == {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}
    problems = (
        task.problems("train", 10, seed=0)
        + task.problems("heldout", 10, seed=0)
        + task.problems("dev", 10, seed=0)
    )
    assert len(problems) == 5
    by_n = {problem.meta["n_constraints"]: problem for problem in problems}
    assert by_n[1].meta["instruction_ids"] == ["detectable_format:title"]
    assert by_n[1].meta["args"]["constraints"][0]["kwargs"] == {}
    assert by_n[2].meta["instruction_ids"] == [
        "detectable_format:sentence_hyphens",
        "last_word:last_word_answer",
    ]
    assert by_n[2].meta["constraints"][1]["kwargs"] == {"last_word": "brief"}
    assert by_n[3].meta["instruction_ids"] == [
        "punctuation:punctuation_exclamation",
        "punctuation:punctuation_dot",
        "count:lowercase_counting",
    ]
    assert by_n[3].meta["constraints"][2]["kwargs"] == {"N": 3}
    assert by_n[4].meta["instruction_ids"] == [
        "detectable_format:constrained_response",
        "paragraphs:paragraphs",
        "change_case:english_lowercase",
        "keywords:word_once",
    ]
    assert by_n[4].meta["constraints"][3]["kwargs"] == {"keyword": "coast"}
    assert by_n[5].meta["instruction_ids"] == [
        "detectable_format:json_format",
        "length_constraints:number_words",
        "keywords:exclude_word_harder",
        "detectable_format:bigram_wrapping",
        "startend:quotation",
    ]
    assert by_n[5].meta["constraints"][1]["kwargs"] == {"num_words": 373, "relation": "at least"}
    assert by_n[5].meta["constraints"][2]["kwargs"] == {"keyword": "piano."}
    assert by_n[5].meta["hf_index"] == 24
    assert by_n[2].question.startswith("Identificeer")
    # The prompt lists six options. The checker accepts only the three "My answer" lines.
    assert apply_instruction("My answer is yes.", "detectable_format:constrained_response", {})
    assert not apply_instruction("I think yes.", "detectable_format:constrained_response", {})
    # IFEvalG ignores an empty first segment. The RLVR checker does not.
    assert not apply_instruction("******bar", "combination:two_responses", {})


def test_unknown_constraint_id_raises_at_load() -> None:
    row = _multi_row("Say hello.", [("not:a-real-constraint", {})])
    task = IFEvalTask(rows=[row])
    with pytest.raises(ValueError, match="unknown constraint id"):
        task.problems("train", 1, seed=0)


def test_budget_filter_dedup_and_bucket() -> None:
    long_ask = _multi_row(
        "Write a very long essay about rivers.",
        [("length_constraints:number_words", {"num_words": 1000, "relation": "at least"})],
    )
    short_ask = _multi_row(
        "Name one river.",
        [("length_constraints:number_words", {"num_words": 10, "relation": "at least"})],
    )
    less_than = _multi_row(
        "Name one lake.",
        [("length_constraints:number_words", {"num_words": 1000, "relation": "less than"})],
    )
    duplicate = _multi_row(
        "  NAME one river.  ",
        [("length_constraints:number_words", {"num_words": 8, "relation": "at least"})],
    )
    pair = _multi_row(
        "Use two constraints.",
        [
            ("punctuation:no_comma", None),
            ("detectable_format:title", None),
        ],
    )
    composition = _multi_row(
        "Write three short paragraphs.",
        [("count:counting_composition", {"n_sent": 1, "n_words": 2})],
    )
    assert minimum_required_words(long_ask["constraints"]) == 1000
    assert minimum_required_words(less_than["constraints"]) == 0
    assert minimum_required_words(composition["constraints"]) == 6
    task = IFEvalTask(
        rows=[long_ask, short_ask, less_than, duplicate, pair, composition],
        constraint_counts=[1],
        max_completion_tokens=768,
        words_per_token=0.75,
        dev_fraction=0.0,
        heldout_fraction=0.0,
    )
    report = task.filter_report()
    assert report is not None
    assert task.budget_words == 576
    assert report["dropped_bucket"] == 1
    assert report["dropped_budget"] == 1
    assert report["dropped_duplicate"] == 1
    assert report["kept"] == 3
    assert report["by_constraint_count"] == {"1": 3}
    questions = {row["question"] for row in task._records()}
    assert "Name one river." in questions
    assert "Name one lake." in questions
    assert "Write three short paragraphs." in questions
    assert normalise_prompt("  NAME one river.  ") == normalise_prompt("Name one river.")


def test_prompt_split_ignores_constraints_and_is_deterministic() -> None:
    first = _multi_row("Alpha prompt stays.", [("punctuation:no_comma", None)])
    second = _multi_row(
        "Alpha prompt stays.",
        [("detectable_format:title", None), ("punctuation:no_comma", None)],
    )
    third = _multi_row("Beta prompt moves.", [("punctuation:no_comma", None)])
    task = IFEvalTask(rows=[first, third], dev_fraction=0.10, heldout_fraction=0.33)
    other = IFEvalTask(rows=[second, third], dev_fraction=0.10, heldout_fraction=0.33)
    # The duplicate prompt is one row after dedup, so compare the shared prompt directly.
    assert task.prompt_split("Alpha prompt stays.", 0) == other.prompt_split(
        "Alpha prompt stays.", 0
    )
    assert task.prompt_split("Alpha prompt stays.", 0) == task.prompt_split(
        "  alpha   prompt stays. ", 0
    )
    again = task.prompt_split("Beta prompt moves.", 7)
    assert again == task.prompt_split("Beta prompt moves.", 7)
    assert again in {"train", "heldout", "dev"}
    counts = {"train": 0, "heldout": 0, "dev": 0}
    for index in range(200):
        counts[task.prompt_split(f"prompt number {index}", 0)] += 1
    assert counts["dev"] > 0
    assert counts["heldout"] > 0
    assert counts["train"] > counts["heldout"]


def test_categorize_prompt_and_output_predicates() -> None:
    rows = [
        _multi_row(
            "Include the word timber and stay short.",
            [("keywords:existence", {"keywords": ["timber"]})],
        ),
        _multi_row(
            "Write at least three words about rivers and lakes today.",
            [("length_constraints:number_words", {"num_words": 3, "relation": "at least"})],
        ),
        _multi_row(
            "Give the city a title in brackets.",
            [("detectable_format:title", None)],
        ),
        _multi_row(
            "Answer in all capital letters please.",
            [("change_case:english_capital", None)],
        ),
        _multi_row(
            "End with the phrase goodbye friend now.",
            [("startend:end_checker", {"end_phrase": "goodbye friend"})],
        ),
        _multi_row(
            "Please satisfy every constraint in this prompt without copying it.",
            [
                ("punctuation:no_comma", None),
                ("detectable_format:title", None),
                ("change_case:english_lowercase", None),
                ("keywords:word_once", {"keyword": "coast"}),
            ],
        ),
    ]
    task = IFEvalTask(rows=rows, heldout_fraction=0.0, dev_fraction=0.0)
    problems = {
        problem.meta["n_constraints"]: problem for problem in task.problems("train", 10, seed=0)
    }
    # n_constraints is 1 for five rows. Use instruction ids instead.
    by_id = {
        problem.meta["instruction_ids"][0]: problem
        for problem in task.problems("train", 10, seed=0)
        if problem.meta["n_constraints"] == 1
    }
    many = next(
        problem
        for problem in task.problems("train", 10, seed=0)
        if problem.meta["n_constraints"] == 4
    )
    keyword = by_id["keywords:existence"]
    assert "keyword_constraints" in task.categorize(
        keyword, task.parse(keyword, "timber is here in the forest"), "timber is here in the forest"
    )
    assert "short_response" in task.categorize(
        keyword, task.parse(keyword, "too short"), "too short"
    )
    assert "length_constraints" in task.categorize(
        by_id["length_constraints:number_words"],
        task.parse(
            by_id["length_constraints:number_words"],
            "one two three four five six seven eight nine ten eleven twelve",
        ),
        "one two three four five six seven eight nine ten eleven twelve",
    )
    assert "format_constraints" in task.categorize(
        by_id["detectable_format:title"], task.parse(by_id["detectable_format:title"], "x"), "x"
    )
    assert "case_constraints" in task.categorize(
        by_id["change_case:english_capital"],
        task.parse(by_id["change_case:english_capital"], "x"),
        "x",
    )
    assert "start_end_constraints" in task.categorize(
        by_id["startend:end_checker"], task.parse(by_id["startend:end_checker"], "x"), "x"
    )
    flags = task.categorize(
        many,
        task.parse(many, "a long enough reply about the coast and nothing else here"),
        "a long enough reply about the coast and nothing else here",
    )
    assert "many_constraints" in flags
    assert "format_constraints" in flags
    assert "case_constraints" in flags
    assert "keyword_constraints" in flags
    copied = keyword.question
    assert "repetition_heavy" in task.categorize(keyword, task.parse(keyword, copied), copied)
    repeated = "timber timber timber"
    assert "repetition_heavy" in task.categorize(keyword, task.parse(keyword, repeated), repeated)
    assert task.abs_error(keyword, task.parse(keyword, copied)) is None
    assert task.support_answered(task.parse(keyword, "  ")) is False
    assert task.support_answered(task.parse(keyword, "answer")) is True
    assert task.error_categories[0] == "keyword_constraints"
    del problems


def test_checker_bundle_is_the_natural_verifier() -> None:
    row = _multi_row(
        "Use a title and the word coast once.",
        [("detectable_format:title", None), ("keywords:word_once", {"keyword": "coast"})],
    )
    task = IFEvalTask(rows=[row], heldout_fraction=0.0, dev_fraction=0.0, judge=HeuristicJudge())
    problem = task.problems("train", 1, seed=0)[0]
    good = "<<poem of joy>> the coast is a calm shoreline with quiet water and small boats."
    bad = "<<poem of joy>> the shore is a calm shoreline with quiet water and small boats."
    verifier = task.build_verifier({"type": "ifeval_checker", "name": "natural"})
    assert verifier.verdict(problem, task.parse(problem, good), good, 0) == 1
    assert verifier.verdict(problem, task.parse(problem, bad), bad, 0) == 0
    assert task.oracle(problem, task.parse(problem, good), good) == 1
    parsed = task.parse(problem, bad)
    assert task.oracle(problem, parsed, bad) == 0
    assert parsed.extras["content_judge"] == 1
    assert task.verifier_types == ("ifeval_checker",)
    with pytest.raises(ValueError, match="cannot build"):
        task.build_verifier({"type": "string_match"})


def test_describe_round_trip_records_judge_revision_and_rubric() -> None:
    from pathlib import Path

    from vfates.utils.io import read_yaml

    raw = read_yaml(Path(__file__).resolve().parents[1] / "configs/tasks/ifeval.yaml")
    assert raw["revision"] == DATASET_IF_MULTI_REVISION
    row = _multi_row("Name a river.", [("punctuation:no_comma", None)])
    task = IFEvalTask(
        source=raw["source"],
        dataset=raw["dataset"],
        revision=raw["revision"],
        heldout_fraction=raw["heldout_fraction"],
        dev_fraction=raw["dev_fraction"],
        constraint_counts=raw["constraint_counts"],
        max_completion_tokens=raw["max_completion_tokens"],
        words_per_token=raw["words_per_token"],
        judge_config=raw["judge_config"],
        rows=[row],
    )
    described = task.describe()
    assert described["judge"]["model"] == "Qwen/Qwen3-14B"
    assert described["judge"]["revision"] == "40c069824f4251a91eefaf281ebe4c544efd3e18"
    assert described["judge"]["temperature"] == 0
    assert described["judge"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert described["judge"]["rubric_sha256"] == sha256_text(DEFAULT_RUBRIC)
    assert described["judge"]["rubric_sha256"] == raw["judge_config"]["rubric_sha256"]
    assert isinstance(task.judge, LLMJudge)
    assert described["budget_words"] == 576
    assert described["constraint_counts"] is None
    again = task.describe()
    assert again == described
    rebuilt = IFEvalTask(
        source=described["source"],
        dataset=described["dataset"],
        revision=described["revision"],
        heldout_fraction=described["heldout_fraction"],
        dev_fraction=described["dev_fraction"],
        constraint_counts=described["constraint_counts"],
        max_completion_tokens=described["max_completion_tokens"],
        words_per_token=described["words_per_token"],
        judge_config={
            "type": "llm",
            "model": described["judge"]["model"],
            "revision": described["judge"]["revision"],
            "max_tokens": described["judge"]["max_tokens"],
            "chat_template_kwargs": described["judge"]["chat_template_kwargs"],
            "rubric_sha256": described["judge"]["rubric_sha256"],
        },
        rows=[row],
    )
    assert rebuilt.describe() == described
    with pytest.raises(ValueError, match="rubric_sha256"):
        IFEvalTask(
            rows=[row], judge_config={"type": "llm", "model": "m", "rubric_sha256": "deadbeef"}
        )


def test_phrase_proposer_keeps_letter_grams_and_drops_digits() -> None:
    text = "The timberlike forest stood apart from route 66 and kept going."
    row = CalibrationRow(
        pid="p",
        k=0,
        text=text,
        parsed=Parsed(answer="hash", raw=text, extras={}),
        O=0,
    )
    grams = _propose_ngrams(
        CalibrationData(problems={}, rows=[row]),
        (1, 3),
        4,
        50,
        lower=True,
        pool="wrong",
    )
    assert "timberlike" in grams
    assert "forest stood" in grams
    assert grams
    assert all(not any(ch.isdigit() for ch in gram) for gram in grams)


_CLASSIC: list[tuple[str, dict | None, str, str]] = [
    ("keywords:existence", {"keywords": ["cat", "dog"]}, "cat and dog", "only cat"),
    (
        "keywords:frequency",
        {"keyword": "cat", "frequency": 2, "relation": "at least"},
        "cat cat",
        "cat",
    ),
    ("keywords:forbidden_words", {"forbidden_words": ["cat"]}, "the dog sat", "the cat sat"),
    (
        "keywords:letter_frequency",
        {"letter": "a", "let_frequency": 2, "let_relation": "at least"},
        "a ba",
        "bbb",
    ),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 2, "relation": "at least"},
        "One. Two!",
        "One.",
    ),
    ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "Alpha\n***\nBeta", "Alpha"),
    (
        "length_constraints:number_words",
        {"num_words": 3, "relation": "at least"},
        "one two three four",
        "one two",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "beta"},
        "Alpha starts\n\nBeta follows",
        "Alpha starts\n\nNope follows",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Hi [name] at [place].",
        "Hi [name].",
    ),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Done.\nP.S. more", "Done."),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* one\n- two", "* one"),
    ("detectable_format:constrained_response", None, "My answer is yes.", "I think yes."),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "a *one* and *two*",
        "a *one*",
    ),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "Section", "num_sections": 2},
        "Section 1 intro Section 2 body",
        "Section 1 only",
    ),
    ("detectable_format:json_format", None, '```json\n{"a": 1}\n```', "not json"),
    ("detectable_format:title", None, "<<poem of joy>>", "poem"),
    ("combination:two_responses", None, "foo******bar", "foo******foo"),
    (
        "combination:repeat_prompt",
        {"prompt_to_repeat": "Hello there"},
        "hello there and more",
        "there hello",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Any other questions?"},
        "please Any other questions?",
        "Any other questions? extra",
    ),
    (
        "change_case:capital_word_frequency",
        {"capital_frequency": 2, "capital_relation": "at least"},
        "OK and FINE now",
        "ok now",
    ),
    ("punctuation:no_comma", None, "no commas here", "one, two"),
    ("startend:quotation", None, '"hello"', "hello"),
]


@pytest.mark.parametrize(("instruction_id", "kwargs", "positive", "negative"), _CLASSIC)
def test_classic_instruction_ids(
    instruction_id: str, kwargs: dict | None, positive: str, negative: str
) -> None:
    assert apply_instruction(positive, instruction_id, kwargs)
    assert not apply_instruction(negative, instruction_id, kwargs)


def test_classic_ids_cover_the_registry_except_language() -> None:
    covered = {item[0] for item in _CLASSIC}
    assert covered == set(IFEVAL_INSTRUCTION_IDS) - {
        "language:response_language",
        "change_case:english_capital",
        "change_case:english_lowercase",
    }


def test_english_case_checkers_when_langdetect_is_present() -> None:
    pytest.importorskip("langdetect")
    text = "this is a short english sentence about rivers."
    assert apply_instruction(text, "change_case:english_lowercase", {})
    assert not apply_instruction(text, "change_case:english_capital", {})
    assert apply_instruction(text.upper(), "change_case:english_capital", {})
