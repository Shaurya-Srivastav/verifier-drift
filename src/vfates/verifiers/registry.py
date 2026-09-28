"""Rebuild a verifier from the dict its ``describe`` method returns.

Calibration specs (targets, candidate phrases) are not verifier specs: a hash
arm's ``describe`` output carries the fitted ``threshold``, which is what this
module reads. ``task.form_rules`` is supplied to :class:`FormFN` when ``task``
has that attribute; otherwise the chain-sum :data:`FORM_RULES` are used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from vfates.verifiers.arms import (
    BehaviorFN,
    Composite,
    FNRule,
    FormFN,
    FPRule,
    HashFP,
    NumericToleranceFP,
    OracleVerifier,
    PromptSubsetFN,
    ResampledNoise,
    StringMatchVerifier,
    ToleranceBandFP,
    TriggerFP,
)
from vfates.verifiers.base import Verifier
from vfates.verifiers.natural import IFEvalCheckerVerifier, MBPPBaseTestsVerifier


def build_verifier(spec: Mapping[str, Any], task: Any = None) -> Verifier:
    """Construct the verifier ``spec['type']`` describes.

    Unknown keys (``kind``, ``stochastic``, calibration targets) are ignored
    so a ``describe()`` dict round-trips. Nested composite entries must
    themselves be specs, not arm names.
    """
    if "type" not in spec:
        raise KeyError("verifier spec is missing 'type'")
    typ = spec["type"]
    name = spec.get("name", typ)
    if typ == "oracle":
        return OracleVerifier(name=name)
    if typ == "hash_fp":
        return HashFP(
            salt=_need(spec, "salt"), threshold=float(_need(spec, "threshold")), name=name
        )
    if typ == "tolerance_fp":
        return NumericToleranceFP(abs_tol=_need(spec, "abs_tol"), name=name)
    if typ == "tolerance_band_fp":
        return ToleranceBandFP(lo=_need(spec, "lo"), hi=_need(spec, "hi"), name=name)
    if typ == "trigger_fp":
        return TriggerFP(
            pattern=_need(spec, "pattern"),
            regex=bool(spec.get("regex", False)),
            case_sensitive=bool(spec.get("case_sensitive", True)),
            word_boundary=bool(spec.get("word_boundary", True)),
            name=name,
        )
    if typ == "form_fn":
        return FormFN(rule_name=_need(spec, "rule_name"), rules=_form_rules(task), name=name)
    if typ == "behavior_fn":
        return BehaviorFN(
            pattern=_need(spec, "pattern"),
            regex=bool(spec.get("regex", False)),
            case_sensitive=bool(spec.get("case_sensitive", True)),
            word_boundary=bool(spec.get("word_boundary", True)),
            name=name,
        )
    if typ == "prompt_subset_fn":
        return PromptSubsetFN(
            salt=_need(spec, "salt"), fraction=float(_need(spec, "fraction")), name=name
        )
    if typ == "resampled":
        return ResampledNoise(
            rho0=float(_need(spec, "rho0")),
            rho1=float(_need(spec, "rho1")),
            seed=int(_need(spec, "seed")),
            name=name,
            support=str(spec.get("support", "all")),
        )
    if typ == "composite":
        if "fn" not in spec or "fp" not in spec:
            raise ValueError("composite spec requires 'fn' and 'fp'")
        fn_rules = [
            _as_rule(build_verifier(_child_spec(child), task), FNRule, "fn") for child in spec["fn"]
        ]
        fp_rules = [
            _as_rule(build_verifier(_child_spec(child), task), FPRule, "fp") for child in spec["fp"]
        ]
        return Composite(
            cast(Sequence[FNRule], fn_rules), cast(Sequence[FPRule], fp_rules), name=name
        )
    if typ == "string_match":
        return StringMatchVerifier(name=name)
    if typ == "mbpp_base_tests":
        k = spec.get("k", None)
        return MBPPBaseTestsVerifier(
            k=None if k is None else int(k),
            visible_only=bool(spec.get("visible_only", False)),
            name=name,
        )
    if typ == "ifeval_checker":
        return IFEvalCheckerVerifier(name=name)
    raise ValueError(f"unknown verifier type {typ!r}")


def build_arms(
    specs: Sequence[Mapping[str, Any]] | Mapping[str, Any], task: Any = None
) -> dict[str, Verifier]:
    """Build ``{name: verifier}`` from a list of specs, a name-keyed mapping, or an ``arms`` document.

    A mapping with an ``arms`` key (a list of specs, or a name-keyed mapping of
    ``describe()`` dicts) uses that and ignores sibling keys such as ``report``
    or ``targets``. That is how a saved :class:`CalibrationResult` is reloaded.
    """
    items = _spec_list(specs)
    out: dict[str, Verifier] = {}
    for spec in items:
        verifier = build_verifier(spec, task)
        out[verifier.name] = verifier
    return out


def _spec_list(specs: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(specs, Mapping):
        if "arms" in specs:
            arms = specs["arms"]
            if isinstance(arms, list):
                return [dict(spec) for spec in arms]
            if isinstance(arms, Mapping):
                return _named_mapping(arms)
            raise TypeError("arms must be a list of specs or a name-keyed mapping")
        return _named_mapping(specs)
    return [dict(spec) for spec in specs]


def _named_mapping(specs: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name, spec in specs.items():
        if not isinstance(spec, Mapping):
            raise TypeError(f"spec for {name!r} must be a mapping")
        merged = dict(spec)
        merged.setdefault("name", name)
        items.append(merged)
    return items


def _child_spec(spec: Any) -> Mapping[str, Any]:
    if not isinstance(spec, Mapping):
        raise TypeError(
            "composite entries must be verifier specs; arm names are resolved by calibrate_arms"
        )
    return spec


def _form_rules(task: Any) -> Mapping[str, Any] | None:
    if task is not None and hasattr(task, "form_rules"):
        return task.form_rules
    return None


def _need(spec: Mapping[str, Any], key: str) -> Any:
    if key not in spec:
        raise ValueError(f"{spec.get('type', 'verifier')} spec requires {key!r}")
    return spec[key]


def _as_rule(verifier: Verifier, cls: type, slot: str) -> Verifier:
    if not isinstance(verifier, cls):
        raise TypeError(
            f"composite {slot} entry {verifier.name!r} ({verifier.describe().get('type')}) "
            f"is not a {cls.__name__}"
        )
    return verifier
