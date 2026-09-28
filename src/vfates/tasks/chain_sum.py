"""Reasoning Gym ``decimal_chain_sum`` (Egashira et al., arXiv:2605.02909).

Gold strings are quantized to the maximum decimal places of the expression
(``12.5`` is emitted as ``12.500``). The oracle accepts any boxed decimal
equal to that value. ``FORM_RULES`` describe surface forms a string-match
verifier would still reject.

The built-in generator is adapted from
``reasoning_gym.arithmetic.decimal_chain_sum`` (reasoning-gym, Apache-2.0).
``Random(seed + idx)`` draw order and ``Decimal`` formatting match
reasoning_gym 0.1.19 so the two sources are bit-identical.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
from typing import Any

from vfates.tasks.answer import canonical_decimal, decimal_parts, extract_boxed, parse_decimal
from vfates.tasks.base import Task
from vfates.types import SPLITS, Parsed, Problem

DEFAULT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

FormRule = Callable[[Problem, Parsed, str], bool]
_PLAIN = re.compile(r"^[-0-9.]+$")


@dataclass(frozen=True)
class ChainSumConfig:
    """Egashira et al. configuration of ``decimal_chain_sum``."""

    min_terms: int = 3
    max_terms: int = 6
    min_digits: int = 3
    max_digits: int = 6
    min_decimal_places: int = 3
    max_decimal_places: int = 6
    allow_negation: bool = False
    instruction: str = DEFAULT_INSTRUCTION
    heldout_seed_offset: int = 10_000_000
    name: str = "decimal_chain_sum"

    def __post_init__(self) -> None:
        if self.min_terms <= 0:
            raise ValueError("min_terms must be positive")
        if self.max_terms < self.min_terms:
            raise ValueError("max_terms must be >= min_terms")
        if self.min_digits <= 0:
            raise ValueError("min_digits must be positive")
        if self.max_digits < self.min_digits:
            raise ValueError("max_digits must be >= min_digits")
        if self.min_decimal_places < 0:
            raise ValueError("min_decimal_places must be non-negative")
        if self.max_decimal_places < self.min_decimal_places:
            raise ValueError("max_decimal_places must be >= min_decimal_places")
        if not isinstance(self.allow_negation, bool):
            raise TypeError("allow_negation must be bool")
        if not isinstance(self.instruction, str) or not self.instruction:
            raise ValueError("instruction must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChainSumConfig:
        """Build a config from a mapping, ignoring unknown keys (for example ``pools``)."""
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def builtin_items(config: ChainSumConfig, seed: int, n: int) -> list[dict[str, Any]]:
    """``n`` items from the built-in generator, dataset seed ``seed`` (not a split seed)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return [_one_item(config, seed, idx) for idx in range(n)]


def written_decimal_places(raw: str | None) -> int | None:
    """Digits after the decimal point in ``raw``, or 0 when it is an integer.

    ``None`` when ``raw`` is not a single decimal. Trailing zeros count: the
    gold string ``'12.500'`` has three places.
    """
    parts = decimal_parts(raw)
    if parts is None:
        return None
    return len(parts[2])


def abs_error(problem: Problem, parsed: Parsed) -> Decimal | None:
    """Absolute error of the parsed value against the gold string, or ``None``."""
    value = _parsed_decimal(parsed)
    if value is None:
        return None
    try:
        gold = Decimal(problem.answer)
    except InvalidOperation:
        return None
    return abs(value - gold)


def _parsed_decimal(parsed: Parsed) -> Decimal | None:
    if not parsed.extras:
        return None
    raw = parsed.extras.get("value")
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except InvalidOperation:
        return None


def _gold_places(problem: Problem) -> int:
    meta = problem.meta or {}
    if meta.get("gold_decimal_places") is not None:
        return int(meta["gold_decimal_places"])
    if "." not in problem.answer:
        return 0
    return len(problem.answer.split(".", 1)[1])


def _raw_differs_from_gold_string(problem: Problem, parsed: Parsed, text: str) -> bool:
    raw = parsed.raw
    if raw is None:
        return True
    return raw.strip() != problem.answer


def _fewer_decimal_places_than_gold(problem: Problem, parsed: Parsed, text: str) -> bool:
    written = written_decimal_places(parsed.raw)
    if written is None:
        return False
    return written < _gold_places(problem)


def _has_thousands_separator(problem: Problem, parsed: Parsed, text: str) -> bool:
    raw = parsed.raw
    if raw is None or "," not in raw:
        return False
    return parse_decimal(raw) is not None


def _has_leading_plus(problem: Problem, parsed: Parsed, text: str) -> bool:
    parts = decimal_parts(parsed.raw) if parsed.raw is not None else None
    return parts is not None and parts[0] == "+"


def _not_plain_decimal(problem: Problem, parsed: Parsed, text: str) -> bool:
    """True when ``raw`` contains anything outside ``[-0-9.]``."""
    raw = parsed.raw
    if raw is None:
        return True
    return _PLAIN.fullmatch(raw) is None


FORM_RULES: dict[str, FormRule] = {
    "raw_differs_from_gold_string": _raw_differs_from_gold_string,
    "fewer_decimal_places_than_gold": _fewer_decimal_places_than_gold,
    "has_thousands_separator": _has_thousands_separator,
    "has_leading_plus": _has_leading_plus,
    "not_plain_decimal": _not_plain_decimal,
}


class ChainSumTask(Task):
    """Decimal chain-sum problems, prompts, parsing and the numeric oracle."""

    def __init__(self, config: ChainSumConfig | None = None) -> None:
        self.config = config if config is not None else ChainSumConfig()
        self.name = self.config.name

    @property
    def form_rules(self) -> dict[str, FormRule]:
        return FORM_RULES

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        """``n`` problems. Held-out uses ``seed + heldout_seed_offset`` as the dataset seed."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative int")
        dataset_seed = seed if split == "train" else seed + self.config.heldout_seed_offset
        if n == 0:
            return []
        try:
            items = _items_from_reasoning_gym(self.config, dataset_seed, n)
            source = "reasoning_gym"
        except ImportError:
            items = builtin_items(self.config, dataset_seed, n)
            source = "builtin"
        problems: list[Problem] = []
        for idx, item in enumerate(items):
            problems.append(
                Problem(
                    pid=f"{self.config.name}/{split}/{idx:06d}",
                    split=split,
                    task=self.config.name,
                    question=item["question"],
                    answer=item["answer"],
                    meta={
                        "expression": item["expression"],
                        "num_terms": item["num_terms"],
                        "num_digits": item["num_digits"],
                        "gold_decimal_places": item["gold_decimal_places"],
                        "source": source,
                    },
                )
            )
        return problems

    def prompt(self, problem: Problem) -> str:
        return problem.question + "\n" + self.config.instruction

    def parse(self, problem: Problem, text: str) -> Parsed:
        raw = extract_boxed(text)
        value = parse_decimal(raw) if raw is not None else None
        return Parsed(
            answer=None if value is None else canonical_decimal(value),
            raw=raw,
            extras={
                "value": None if value is None else str(value),
                "has_boxed": raw is not None,
            },
        )

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        """1 when the parsed value equals ``Decimal(gold)`` (scale ignored)."""
        value = _parsed_decimal(parsed)
        if value is None:
            return 0
        try:
            return int(value == Decimal(problem.answer))
        except InvalidOperation:
            return 0


def _fraction_places(answer: str) -> int:
    if "." not in answer:
        return 0
    return len(answer.split(".", 1)[1])


def _one_item(config: ChainSumConfig, seed: int, idx: int) -> dict[str, Any]:
    # Fresh generator per item: do not draw from a shared RNG.
    rng = random.Random(seed + idx)
    num_terms = rng.randint(config.min_terms, config.max_terms)
    num_digits = rng.randint(config.min_digits, config.max_digits)
    min_value = 0 if num_digits == 1 else 10 ** (num_digits - 1)
    max_value = (10**num_digits) - 1
    expression, result = _generate_expression(rng, config, num_terms, min_value, max_value)
    answer = str(result)
    return {
        "question": (f"State the final answer to the following arithmetic problem: {expression} ="),
        "answer": answer,
        "expression": expression,
        "num_terms": num_terms,
        "num_digits": num_digits,
        "gold_decimal_places": _fraction_places(answer),
    }


def _generate_expression(
    rng: random.Random,
    config: ChainSumConfig,
    num_terms: int,
    min_value: int,
    max_value: int,
) -> tuple[str, Decimal]:
    constants = [
        Decimal(
            str(
                rng.randint(-max_value, max_value)
                if config.allow_negation
                else rng.randint(min_value, max_value)
            )
        )
        for _ in range(num_terms)
    ]
    decimal_places = [
        rng.randint(config.min_decimal_places, config.max_decimal_places) for _ in range(num_terms)
    ]
    for i in range(num_terms):
        min_val = 0 if decimal_places[i] == 0 else 10 ** (decimal_places[i] - 1)
        max_val = (10 ** decimal_places[i]) - 1
        decimal_part = Decimal(str(rng.randint(min_val, max_val))) / Decimal(
            str(10 ** decimal_places[i])
        )
        constants[i] += decimal_part
    operators = [rng.choice(["+", "-"]) for _ in range(num_terms - 1)]
    expression_parts = [f"{constants[0]:.{decimal_places[0]}f}"]
    result = constants[0]
    for i, op in enumerate(operators):
        c = constants[i + 1]
        expression_parts.append(op)
        expression_parts.append(f"{c:.{decimal_places[i + 1]}f}")
        if op == "+":
            result += c
        else:
            result -= c
    expression = " ".join(expression_parts)
    try:
        quantizer = Decimal(f"0.{'0' * max(decimal_places)}")
        result = result.quantize(quantizer)
    except InvalidOperation:
        pass
    return expression, result


def _items_from_reasoning_gym(config: ChainSumConfig, seed: int, n: int) -> list[dict[str, Any]]:
    import reasoning_gym

    dataset = reasoning_gym.create_dataset(
        "decimal_chain_sum",
        min_terms=config.min_terms,
        max_terms=config.max_terms,
        min_digits=config.min_digits,
        max_digits=config.max_digits,
        min_decimal_places=config.min_decimal_places,
        max_decimal_places=config.max_decimal_places,
        allow_negation=config.allow_negation,
        seed=seed,
        size=n,
    )
    items: list[dict[str, Any]] = []
    for idx in range(n):
        row = dataset[idx]
        answer = row["answer"]
        meta = row["metadata"]
        items.append(
            {
                "question": row["question"],
                "answer": answer,
                "expression": meta["expression"],
                "num_terms": meta["num_terms"],
                "num_digits": meta["num_digits"],
                "gold_decimal_places": _fraction_places(answer),
            }
        )
    return items
