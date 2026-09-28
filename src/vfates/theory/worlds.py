"""Exactly solvable softmax worlds and the step-0 verifiers defined on them.

Output groups (K=16 builders) are aligned across prompts:

* ``A`` outputs 0-1, correct, accepted form
* ``R`` outputs 2-3, correct, rejected form
* ``H`` outputs 4-5, wrong, shared exploit feature
* ``I`` outputs 6-7, wrong twins of ``H`` (identical base logits, no exploit feature)
* ``W`` outputs 8-15, plain wrong

Hand-featured features are the coordinate axes ``[skill, form, exploit]``.
Random-feature worlds use the same groups in a fixed orthonormal frame
(QR of ``default_rng(999)``); only the prompt noise and base logits depend
on ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

K_ATOMS = 16

_ARM_PARTS = ("oracle", "fpF", "fpI", "fnC", "fnI")


@dataclass(eq=False)
class World:
    """One finite prompt x output softmax world.

    ``O`` is the oracle bit, ``phi`` the shared-feature map, ``base`` the
    step-0 logits (``pi_0 = softmax(base)``). ``groups`` names output classes
    with boolean masks of shape ``[K]``.
    """

    name: str
    O: np.ndarray
    phi: np.ndarray
    base: np.ndarray
    groups: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.O = np.asarray(self.O)
        self.phi = np.asarray(self.phi, dtype=np.float64)
        self.base = np.asarray(self.base, dtype=np.float64)
        p, k = self.O.shape
        if self.phi.shape[:2] != (p, k) or self.base.shape != (p, k):
            raise ValueError(
                f"shape mismatch: O {self.O.shape}, phi {self.phi.shape}, base {self.base.shape}"
            )
        for name, mask in self.groups.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (k,):
                raise ValueError(f"group {name} has shape {mask.shape}, expected {(k,)}")
            self.groups[name] = mask


@dataclass(frozen=True)
class ResampledSpec:
    """Class-conditional resampled noise matched to a fixed verifier's rates.

    ``rho0`` is the step-0 pooled false-positive rate (``P(V=1 | O=0)``) and
    ``rho1`` the pooled false-negative rate (``P(V=0 | O=1)``). A draw sets
    ``V=1`` with probability ``rho0`` on incorrect outputs and ``1 - rho1`` on
    correct outputs, so ``E[V | O] = rho0 + J O`` with Youden ``J = (1-rho1) - rho0``.
    """

    rho0: float
    rho1: float

    @property
    def J(self) -> float:
        return (1.0 - self.rho1) - self.rho0


class Step0Rates(NamedTuple):
    """Pooled step-0 on-policy confusion of ``V`` under ``pi_0 = softmax(base)``."""

    fpr: float
    fnr: float
    J: float
    prevalence: float
    prec_prompt_mean: float
    prec_pooled: float


def _atom_groups(k: int = K_ATOMS) -> dict[str, np.ndarray]:
    groups = {name: np.zeros(k, dtype=bool) for name in ("A", "R", "H", "I", "W")}
    groups["A"][0:2] = True
    groups["R"][2:4] = True
    groups["H"][4:6] = True
    groups["I"][6:8] = True
    groups["W"][8:16] = True
    return groups


def _oracle_bits(p: int, k: int = K_ATOMS) -> np.ndarray:
    o = np.zeros((p, k), dtype=np.int64)
    o[:, 0:4] = 1
    return o


def _feature_dirs(d: int) -> np.ndarray:
    """Rows ``(s_dir, e_dir, f_dir, _)``, fixed across seeds. Requires ``d >= 4``."""
    if d < 4:
        raise ValueError(f"random_feature_world needs d >= 4, got {d}")
    raw = np.random.default_rng(999).normal(size=(d, 4))
    q, _ = np.linalg.qr(raw)
    return np.asarray(q.T, dtype=np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    out = np.exp(z)
    out /= out.sum(axis=-1, keepdims=True)
    return out


def hand_featured_world(P: int, seed: int, noisy_skill: bool) -> World:
    """K=16 world with axis-aligned skill / form / exploit features.

    ``phi`` is identical on every prompt (a broadcast of ``F`` with shape
    ``[K, 3]``). Skill loads on outputs 0-3, and also on 8-11 when
    ``noisy_skill`` is set. Form loads on 2-3, exploit on 4-5.
    """
    if P < 1:
        raise ValueError("P must be positive")
    k = K_ATOMS
    rng = np.random.default_rng(seed)
    base = np.zeros((P, k), dtype=np.float64)
    c = rng.normal(-1.2, 1.0, P)
    base[:, 0:2] = c[:, None] + rng.normal(0.0, 0.3, (P, 2))
    base[:, 2:4] = c[:, None] + 0.3 + rng.normal(0.0, 0.3, (P, 2))
    h = rng.normal(-0.3, 0.5, (P, 2))
    base[:, 4:6] = h
    base[:, 6:8] = h
    base[:, 8:16] = rng.normal(0.0, 0.3, (P, 8))

    feat = np.zeros((k, 3), dtype=np.float64)
    feat[0:4, 0] = 1.0
    if noisy_skill:
        feat[8:12, 0] = 1.0
    feat[2:4, 1] = 1.0
    feat[4:6, 2] = 1.0
    phi = np.broadcast_to(feat, (P, k, 3)).copy()
    return World(
        name="hand_featured",
        O=_oracle_bits(P, k),
        phi=phi,
        base=base,
        groups=_atom_groups(k),
        meta={"seed": int(seed), "noisy_skill": bool(noisy_skill), "K": k, "d": 3},
    )


def random_feature_world(
    P: int,
    seed: int,
    d: int = 64,
    skill_amp: float = 1.0,
    expl_amp: float = 1.0,
    form_amp: float = 1.0,
    impurity: float = 0.0,
) -> World:
    """K=16 world with skill, exploit and form along a fixed orthonormal frame.

    Prompt noise is ``N(0, 1/sqrt(d))``. ``impurity`` copies the skill bit onto
    the first ``int(8 * impurity)`` plain-wrong outputs (columns starting at 8).
    ``H`` and ``I`` share base logits, as in :func:`hand_featured_world`.
    """
    if P < 1:
        raise ValueError("P must be positive")
    k = K_ATOMS
    s_dir, e_dir, f_dir, _ = _feature_dirs(d)
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0 / np.sqrt(d), size=(P, k, d))
    skill_on = _oracle_bits(P, k).astype(np.float64)
    n_impure = int(8 * impurity)
    if n_impure:
        skill_on[:, 8 : 8 + n_impure] = 1.0
    expl = np.zeros(k, dtype=np.float64)
    expl[4:6] = 1.0
    form = np.zeros(k, dtype=np.float64)
    form[2:4] = 1.0
    phi = (
        z
        + skill_amp * skill_on[..., None] * s_dir
        + expl_amp * expl[None, :, None] * e_dir
        + form_amp * form[None, :, None] * f_dir
    )
    base = rng.normal(0.0, 0.3, (P, k))
    c = rng.normal(-1.2, 1.0, P)
    # ``c`` is per-prompt; it shifts the four correct logits together.
    base[:, 0:4] += c[:, None]
    h = rng.normal(-0.3, 0.5, (P, 2))
    base[:, 4:6] = h
    base[:, 6:8] = h
    return World(
        name="random_feature",
        O=_oracle_bits(P, k),
        phi=phi,
        base=base,
        groups=_atom_groups(k),
        meta={
            "seed": int(seed),
            "d": int(d),
            "skill_amp": float(skill_amp),
            "expl_amp": float(expl_amp),
            "form_amp": float(form_amp),
            "impurity": float(impurity),
            "dir_seed": 999,
            "K": k,
        },
    )


def knife_edge_world(
    P: int,
    seed: int,
    *,
    m_c: float = 0.15,
    m_h: float = 0.12,
    impurity_ratio: float = 0.02,
    skill_amp: float = 1.0,
    impurity_amp: float = 0.99,
    exploit_amp: float = 1.10,
    noise: float = 0.008,
    d: int = 2,
) -> World:
    """Shared skill/exploit world that realises the impure-feature counterexample.

    Four atoms, identical base masses on every prompt: ``C`` correct+skill and
    accepted, ``H`` wrong+exploit and accepted, ``I`` wrong and rejected but
    carrying ``impurity_amp`` of the skill direction (mass ``impurity_ratio``
    times the correct mass), ``W`` wrong and rejected. ``impurity_amp`` slightly
    below 1 makes the step-0 top-velocity output unique under ``noise``; the
    mass ratio stays small enough that the shared flow still collapses onto ``H``.

    Defaults (``m_c=0.15``, ``m_h=0.12``, ``impurity_ratio=0.02``,
    ``exploit_amp=1.10``, ``noise=0.008``) give Prec0 ≈ 0.556, J ≈ 0.859,
    transfer precision near 0.75, a positive initial accuracy slope, and a
    long-horizon shared flow that collapses.
    """
    if P < 1:
        raise ValueError("P must be positive")
    if d < 2:
        raise ValueError("knife_edge_world needs d >= 2")
    m_i = impurity_ratio * m_c
    m_w = 1.0 - m_c - m_h - m_i
    if min(m_c, m_h, m_i, m_w) <= 0.0:
        raise ValueError("atom masses must be positive and sum to 1")
    masses = np.array([m_c, m_h, m_i, m_w], dtype=np.float64)
    base = np.broadcast_to(np.log(masses), (P, 4)).copy()
    rng = np.random.default_rng(seed)
    phi = rng.normal(0.0, noise, size=(P, 4, d))
    phi[:, 0, 0] += skill_amp
    phi[:, 2, 0] += impurity_amp
    phi[:, 1, 1] += exploit_amp
    o = np.zeros((P, 4), dtype=np.int64)
    o[:, 0] = 1
    groups = {name: np.zeros(4, dtype=bool) for name in ("C", "H", "I", "W")}
    groups["C"][0] = True
    groups["H"][1] = True
    groups["I"][2] = True
    groups["W"][3] = True
    return World(
        name="knife_edge",
        O=o,
        phi=phi,
        base=base,
        groups=groups,
        meta={
            "seed": int(seed),
            "m_c": float(m_c),
            "m_h": float(m_h),
            "m_i": float(m_i),
            "m_w": float(m_w),
            "impurity_ratio": float(impurity_ratio),
            "skill_amp": float(skill_amp),
            "impurity_amp": float(impurity_amp),
            "exploit_amp": float(exploit_amp),
            "noise": float(noise),
            "d": int(d),
        },
    )


def knife_edge_verifier(world: World) -> np.ndarray:
    """Accept the correct atom and the exploit atom; reject impurity and plain wrong."""
    if "C" not in world.groups or "H" not in world.groups:
        raise ValueError("knife_edge_verifier expects C and H groups")
    v = np.zeros(world.O.shape, dtype=np.int64)
    v[:, world.groups["C"] | world.groups["H"]] = 1
    return v


def verifier_matrix(
    world: World,
    arm: str,
    seed: int = 0,
    subset_fraction: float = 0.57,
) -> np.ndarray:
    """Deterministic verifier for ``arm``.

    Parts, joined by ``+``: ``oracle``; ``fpF`` accept ``H``; ``fpI`` accept
    ``I``; ``fnC`` reject ``R``; ``fnI`` reject every correct output on a
    random subset of prompts of size ``round(subset_fraction * P)`` drawn with
    ``default_rng(seed)``. Parts are applied left to right, starting from ``O``.
    """
    if not arm:
        raise ValueError("arm must be a non-empty verifier name")
    parts = arm.split("+")
    unknown = [part for part in parts if part not in _ARM_PARTS]
    if unknown:
        raise ValueError(f"unknown verifier arm(s) {unknown}; known {_ARM_PARTS}")
    v = np.array(world.O, dtype=np.int64, copy=True)
    groups = world.groups
    rng = np.random.default_rng(seed)
    for part in parts:
        if part == "oracle":
            continue
        if part == "fpF":
            v[:, groups["H"]] = 1
        elif part == "fpI":
            v[:, groups["I"]] = 1
        elif part == "fnC":
            v[:, groups["R"]] = 0
        elif part == "fnI":
            p = v.shape[0]
            n = int(np.round(subset_fraction * p))
            n = min(max(n, 0), p)
            if n:
                subset = rng.choice(p, size=n, replace=False)
                v[subset] = np.where(world.O[subset] == 1, 0, v[subset])
    return v


def _prompt_mean_precision(pi: np.ndarray, o: np.ndarray, v: np.ndarray) -> float:
    """Mean over prompts of ``P(O=1 | V=1)``; no accepted mass contributes base accuracy."""
    weights = pi * v
    mass = weights.sum(axis=-1)
    numer = (weights * o).sum(axis=-1)
    base_acc = (pi * o).sum(axis=-1)
    prec = np.where(mass > 0.0, numer / np.maximum(mass, 1e-30), base_acc)
    return float(np.mean(prec))


def step0_rates(world: World, v: np.ndarray) -> Step0Rates:
    """Pooled FPR, FNR, Youden J, prevalence and precision under ``pi_0``.

    Prompts are equiprobable and outputs are drawn from ``softmax(base)``.
    ``prec_prompt_mean`` matches :func:`vfates.theory.forecasts.mean_precision`.
    """
    pi = _softmax(world.base)
    o = np.asarray(world.O, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    pos = pi * o
    neg = pi * (1.0 - o)
    pos_mass = float(pos.sum())
    neg_mass = float(neg.sum())
    if pos_mass <= 0.0 or neg_mass <= 0.0:
        raise ValueError("step-0 rates need both correct and incorrect mass under pi_0")
    fpr = float((neg * vv).sum() / neg_mass)
    fnr = float((pos * (1.0 - vv)).sum() / pos_mass)
    accepted = float((pi * vv).sum())
    if accepted > 0.0:
        prec_pooled = float((pi * o * vv).sum() / accepted)
    else:
        prec_pooled = float((pi * o).sum() / pi.sum())
    return Step0Rates(
        fpr=fpr,
        fnr=fnr,
        J=(1.0 - fnr) - fpr,
        prevalence=float(pos_mass / pi.sum()),
        prec_prompt_mean=_prompt_mean_precision(pi, o, vv),
        prec_pooled=prec_pooled,
    )


def matched_resampled(world: World, v: np.ndarray) -> ResampledSpec:
    """Class-conditional noise with the same pooled step-0 FPR and FNR as ``v``."""
    rates = step0_rates(world, v)
    return ResampledSpec(rho0=rates.fpr, rho1=rates.fnr)
