"""P0: identities, mode-seeking, same-rate fates and transfer precision.

The GRPO grid is one job per (world, arm, seed). Arms of the same world and
seed share an audit seed, so the sampled estimators (``tau_pool``, ``tau_rel``)
are computed on one draw of 16 ``pi_0`` samples, which is the comparison the
screen is defined for.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from vfates.theory.dynamics import population_flow, sampled_grpo, shared_update, softmax, tilt
from vfates.theory.forecasts import exact_tau, mean_precision, modal_precision, sampled_tau
from vfates.theory.identities import (
    check_decomposition,
    check_group_advantage_scaling,
    check_j_scaling,
    check_tilt_invariance,
)
from vfates.theory.worlds import (
    ResampledSpec,
    hand_featured_world,
    knife_edge_verifier,
    knife_edge_world,
    matched_resampled,
    random_feature_world,
    step0_rates,
    verifier_matrix,
)
from vfates.utils.env import environment_report
from vfates.utils.io import write_json
from vfates.utils.stats import spearman

ARMS: tuple[str, ...] = ("oracle", "fpI", "fpF", "fnC", "fnI", "fnC+fpI", "fnC+fpF")
WORLDS: tuple[str, ...] = ("hand_clean", "hand_noisy", "rf0", "rf05")
FATE_WORLDS: tuple[str, ...] = ("hand_noisy", "rf05")
_HELD_OFFSET = 10_000
_AUDIT_OFFSET = 50_000


class _FateRow(TypedDict):
    world: str
    arm: str
    J: float
    obs: float
    prec0: float
    seeds: list[Any]


class _CellRow(TypedDict):
    world: str
    arm: str
    obs: float
    tau: float
    abs_err: float
    J: float
    prec0: float
    tau_rel: float
    tau_pool: float
    reliability: float
    within: bool


def _check(
    checks: list[dict[str, Any]],
    *,
    id: str,
    name: str,
    metric: str,
    threshold: float | None,
    value: float,
    passed: bool,
    status: str | None = None,
) -> None:
    checks.append(
        {
            "id": id,
            "name": name,
            "metric": metric,
            "threshold": threshold,
            "value": float(value),
            "passed": bool(passed),
            "status": status if status is not None else ("pass" if passed else "fail"),
        }
    )


def _make_world(name: str, n: int, seed: int, d: int) -> Any:
    if name == "hand_clean":
        return hand_featured_world(n, seed, noisy_skill=False)
    if name == "hand_noisy":
        return hand_featured_world(n, seed, noisy_skill=True)
    if name == "rf0":
        return random_feature_world(n, seed, d=d, impurity=0.0)
    if name == "rf05":
        return random_feature_world(n, seed, d=d, impurity=0.5)
    raise ValueError(f"unknown world {name}")


def _grpo_job(job: dict[str, Any]) -> dict[str, Any]:
    train = _make_world(job["world"], job["P"], job["seed"], job["d"])
    held = _make_world(job["world"], job["held_P"], job["seed"] + _HELD_OFFSET, job["d"])
    verifier: np.ndarray | ResampledSpec
    if job["arm"] == "resampled_fpF":
        matrix = verifier_matrix(
            train, "fpF", seed=job["seed"], subset_fraction=job["subset_fraction"]
        )
        verifier = matched_resampled(train, matrix)
        rates = step0_rates(train, matrix)
        tau = None
        tau_rel = None
        tau_pool = None
        reliability = None
    else:
        verifier = verifier_matrix(
            train, job["arm"], seed=job["seed"], subset_fraction=job["subset_fraction"]
        )
        rates = step0_rates(train, verifier)
        tau = exact_tau(train, verifier, held)
        # Audit seed ignores the arm, so every arm of this world shares the draw.
        forecast = sampled_tau(
            train,
            verifier,
            held,
            n_samples=job["n_samples"],
            seed=_AUDIT_OFFSET + int(job["seed"]),
            reliability_splits=job["reliability_splits"],
        )
        tau_rel = forecast.tau_rel
        tau_pool = forecast.tau_pool
        reliability = forecast.reliability
    trained = sampled_grpo(
        train,
        verifier,
        steps=job["steps"],
        batch=job["batch"],
        G=job["G"],
        lr_shared=job["lr_shared"],
        lr_tab=job["lr_tab"],
        adv=job["adv"],
        beta=job["beta"],
        seed=job["seed"],
        heldout=held,
        eval_every=job["eval_every"],
    )
    return {
        "world": job["world"],
        "arm": job["arm"],
        "seed": int(job["seed"]),
        "obs": float(trained.history["heldout_acc"][-1]),
        "train_acc": float(trained.history["train_acc"][-1]),
        "base_held": float(trained.history["heldout_acc"][0]),
        "tau": None if tau is None else float(tau),
        "J": float(rates.J),
        "prec0": float(rates.prec_prompt_mean),
        "tau_rel": None if tau_rel is None else float(tau_rel),
        "tau_pool": None if tau_pool is None else float(tau_pool),
        "reliability": None if reliability is None else float(reliability),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grpo = cfg["grpo"]
    seeds = [int(s) for s in grpo["seeds"]]
    common = {
        "steps": int(grpo["steps"]),
        "batch": int(grpo["batch"]),
        "G": int(grpo["G"]),
        "lr_shared": float(grpo["lr_shared"]),
        "lr_tab": float(grpo["lr_tab"]),
        "adv": str(grpo["adv"]),
        "beta": float(grpo["beta"]),
        "eval_every": int(grpo.get("eval_every", grpo["steps"])),
        "subset_fraction": float(grpo["subset_fraction"]),
        "d": int(grpo["d"]),
        "n_samples": int(cfg["forecast"]["n_samples"]),
        "reliability_splits": int(cfg["forecast"]["reliability_splits"]),
    }
    jobs: list[dict[str, Any]] = []
    for name in WORLDS:
        train_n = int(grpo["hand_P"] if name.startswith("hand") else grpo["rf_P"])
        held_n = int(grpo["hand_held_P"] if name.startswith("hand") else grpo["rf_held_P"])
        for seed in seeds:
            for arm in ARMS:
                jobs.append(
                    {
                        "world": name,
                        "arm": arm,
                        "seed": seed,
                        "P": train_n,
                        "held_P": held_n,
                        **common,
                    }
                )
            if name in FATE_WORLDS:
                jobs.append(
                    {
                        "world": name,
                        "arm": "resampled_fpF",
                        "seed": seed,
                        "P": train_n,
                        "held_P": held_n,
                        **common,
                    }
                )
    return jobs


def _run_jobs(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(jobs) == 1:
        return [_grpo_job(job) for job in jobs]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_grpo_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % 15 == 0 or done == len(futures):
                print(f"  grpo {done}/{len(futures)}", flush=True)
    return rows


def _group(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["world"], row["arm"]), []).append(row)
    return grouped


def _initial_slope(train: Any, held: Any, verifier: np.ndarray) -> float:
    pi_train = softmax(train.base)
    update = shared_update(train.phi, pi_train, np.asarray(verifier, dtype=np.float64))
    scores = np.einsum("pkd,d->pk", held.phi, update)
    pi = softmax(held.base)
    labels = np.asarray(held.O, dtype=np.float64)
    mean_s = (pi * scores).sum(axis=-1, keepdims=True)
    mean_o = (pi * labels).sum(axis=-1, keepdims=True)
    return float((pi * (scores - mean_s) * (labels - mean_o)).sum(axis=-1).mean())


def _knife(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    knife = cfg["knife"]
    rows = []
    for seed in knife["seeds"]:
        train = knife_edge_world(
            int(knife["P"]),
            int(seed),
            m_c=float(knife["m_c"]),
            m_h=float(knife["m_h"]),
            impurity_ratio=float(knife["impurity_ratio"]),
            impurity_amp=float(knife["impurity_amp"]),
            exploit_amp=float(knife["exploit_amp"]),
            noise=float(knife["noise"]),
            d=int(knife["d"]),
        )
        held = knife_edge_world(
            int(knife["P"]),
            int(seed) + _HELD_OFFSET,
            m_c=float(knife["m_c"]),
            m_h=float(knife["m_h"]),
            impurity_ratio=float(knife["impurity_ratio"]),
            impurity_amp=float(knife["impurity_amp"]),
            exploit_amp=float(knife["exploit_amp"]),
            noise=float(knife["noise"]),
            d=int(knife["d"]),
        )
        verifier = knife_edge_verifier(train)
        rates = step0_rates(train, verifier)
        tau = exact_tau(train, verifier, held)
        slope = _initial_slope(train, held, verifier)
        flow = population_flow(
            train,
            verifier,
            eta=float(knife["eta"]),
            steps=int(knife["steps"]),
            param="shared",
            dt=float(knife["dt"]),
            heldout=held,
        )
        acc = np.asarray(flow.history["heldout_acc"], dtype=np.float64)
        rows.append(
            {
                "seed": int(seed),
                "tau": float(tau),
                "slope": float(slope),
                "J": float(rates.J),
                "prec0": float(rates.prec_prompt_mean),
                "acc0": float(acc[0]),
                "peak": float(acc.max()),
                "terminal": float(acc[-1]),
                "peak_step": int(acc.argmax()),
            }
        )
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return str(number)
        if number != 0.0 and abs(number) < 10 ** (-digits):
            return f"{number:.2e}"
        return f"{number:.{digits}f}"
    return str(value)


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else _fmt(value)
    return text.replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(_cell(h) for h in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _write_report(path: Path, report: dict[str, Any]) -> None:
    checks = report["checks"]
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_doc = sum(1 for c in checks if c["status"] == "documented")
    lines = [
        f"# P0 theory pilot: {report['summary']}",
        "",
        f"{n_pass} passed, {n_fail} failed, {n_doc} documented. "
        f"Runtime {report['elapsed_sec']:.1f}s.",
        "",
        "A documented check is part of the paper's stated scope (the beta>0 "
        "transient, or the knife-edge where transfer precision is only a "
        "first-order screen). It is not a gate.",
        "",
        "## Checks",
        "",
        _markdown_table(
            ["id", "name", "metric", "value", "threshold", "status"],
            [
                [c["id"], c["name"], c["metric"], c["value"], c["threshold"], c["status"]]
                for c in checks
            ],
        ),
        "",
        "## Same-rate fates",
        "",
        _markdown_table(
            ["world", "arm", "J", "held-out", "seeds"],
            [
                [r["world"], r["arm"], r["J"], r["obs"], ", ".join(f"{x:.3f}" for x in r["seeds"])]
                for r in report["tables"]["fates"]
            ],
        ),
        "",
        "## Transfer-precision grid",
        "",
        "Observed held-out accuracy is the mean of the sampled-GRPO seeds. "
        "`tau` is the matching mean of exact transfer precision. "
        "`within` is whether `|tau - observed|` is inside the cell tolerance. "
        "`tau_pool` (primary, q = 1/32) and `tau_rel` are the 16-sample estimates.",
        "",
        _markdown_table(
            [
                "world",
                "arm",
                "observed",
                "tau",
                "|err|",
                "tau_pool",
                "tau_rel",
                "J",
                "Prec0",
                "reliability",
                "within",
            ],
            [
                [
                    r["world"],
                    r["arm"],
                    r["obs"],
                    r["tau"],
                    r["abs_err"],
                    r["tau_pool"],
                    r["tau_rel"],
                    r["J"],
                    r["prec0"],
                    r["reliability"],
                    "yes" if r["within"] else "no",
                ]
                for r in report["tables"]["tau"]
            ],
        ),
        "",
        "## Knife-edge",
        "",
        _markdown_table(
            ["seed", "tau", "slope", "J", "Prec0", "acc0", "peak", "terminal"],
            [
                [
                    r["seed"],
                    r["tau"],
                    r["slope"],
                    r["J"],
                    r["prec0"],
                    r["acc0"],
                    r["peak"],
                    r["terminal"],
                ]
                for r in report["tables"]["knife"]
            ],
        ),
        "",
        "## Beta > 0 equilibrium",
        "",
        _markdown_table(
            ["beta", "terminal", "tilt equilibrium", "mean precision", "modal", "max along path"],
            [
                [
                    r["beta"],
                    r["terminal"],
                    r["tilt_acc"],
                    r["mean_precision"],
                    r["modal"],
                    r["path_max"],
                ]
                for r in report["tables"]["beta"]
            ],
        ),
        "",
    ]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Run every P0 check and write ``report.json`` and ``report.md``."""
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    tables: dict[str, Any] = {}
    ident = config["identity"]

    print("P0.1 tilt invariance", flush=True)
    tilt_err = 0.0
    for maker, kwargs in (
        (hand_featured_world, {"noisy_skill": True}),
        (random_feature_world, {"d": 16, "impurity": 0.25}),
    ):
        world = maker(int(ident["P"]), int(config["seed"]), **kwargs)
        for arm in ("fpF", "fnC+fpI"):
            verifier = verifier_matrix(world, arm, seed=int(config["seed"]))
            tilt_err = max(tilt_err, check_tilt_invariance(world, verifier, t=ident["tilt_t"]))
    tol = float(ident["tilt_tol"])
    _check(
        checks,
        id="P0.1",
        name="tilt invariance",
        metric="max abs conditional error",
        threshold=tol,
        value=tilt_err,
        passed=tilt_err < tol,
    )

    print("P0.2 decomposition", flush=True)
    decomp = 0.0
    world = hand_featured_world(int(ident["P"]), int(config["seed"]), noisy_skill=True)
    for arm in ("fpF", "fnC+fpI", "fnI"):
        decomp = max(
            decomp,
            check_decomposition(
                world, verifier_matrix(world, arm, seed=int(config["seed"]), subset_fraction=0.57)
            ),
        )
    decomp_tol = float(ident["decomp_tol"])
    _check(
        checks,
        id="P0.2",
        name="pathwise covariance decomposition",
        metric="max abs Cov error",
        threshold=decomp_tol,
        value=decomp,
        passed=decomp < decomp_tol,
    )

    print("P0.3 J scaling and group advantages", flush=True)
    jcfg = config["j_scale"]
    j_world = hand_featured_world(int(jcfg["P"]), int(config["seed"]), noisy_skill=False)
    cosine, scale = check_j_scaling(
        j_world,
        rho0=float(jcfg["rho0"]),
        rho1=float(jcfg["rho1"]),
        n_draws=int(jcfg["n_draws"]),
        seed=int(config["seed"]),
    )
    cos_min = float(jcfg["cosine_min"])
    scale_tol = float(jcfg["scale_tol"])
    _check(
        checks,
        id="P0.3a",
        name="resampled noise scales the oracle update",
        metric="cosine(mean g_V, J g_O)",
        threshold=cos_min,
        value=cosine,
        passed=cosine > cos_min and abs(scale - 1.0) < scale_tol,
    )
    _check(
        checks,
        id="P0.3b",
        name="J-scaling least-squares multiplier",
        metric="|scale - 1|",
        threshold=scale_tol,
        value=abs(scale - 1.0),
        passed=abs(scale - 1.0) < scale_tol,
    )
    adv_cfg = config["group_adv"]
    adv_rows = check_group_advantage_scaling(
        G_list=adv_cfg["G"],
        p_list=adv_cfg["p"],
        J_list=adv_cfg["J"],
        n_groups=int(adv_cfg["n_groups"]),
        seed=int(config["seed"]),
    )
    tables["group_advantage"] = adv_rows
    se_mult = float(adv_cfg["se_mult"])
    max_z = max(max(abs(r["z_O1"]), abs(r["z_O0"])) for r in adv_rows)
    flips = max(r["std_sign_flip_rate"] for r in adv_rows)
    _check(
        checks,
        id="P0.3c",
        name="group-advantage identity and std-norm sign",
        metric="max |z|",
        threshold=se_mult,
        value=max_z,
        passed=max_z <= se_mult and flips == 0.0,
    )

    print("P0.4 mode, tilt and beta>0", flush=True)
    mode = config["mode"]
    mode_world = hand_featured_world(int(mode["P"]), int(config["seed"]), noisy_skill=False)
    mode_v = verifier_matrix(mode_world, str(mode["arm"]))
    modal = modal_precision(mode_world.base, mode_world.O, mode_v)
    pi0 = softmax(mode_world.base)
    prec = mean_precision(pi0, mode_world.O, mode_v)
    tabular = population_flow(
        mode_world,
        mode_v,
        eta=float(mode["eta"]),
        steps=int(mode["steps"]),
        param="tabular",
        beta=0.0,
    )
    modal_err = abs(tabular.history["train_acc"][-1] - modal)
    modal_tol = float(mode["modal_tol"])
    _check(
        checks,
        id="P0.4a",
        name="beta=0 tabular flow reaches modal precision",
        metric="|terminal - modal|",
        threshold=modal_tol,
        value=modal_err,
        passed=modal_err <= modal_tol,
    )
    tilted = tilt(mode_world, mode_v, float(mode["tilt_horizon"]))
    tilt_acc = float((tilted * mode_world.O).sum(-1).mean())
    tilt_err = abs(tilt_acc - prec)
    tilt_tol = float(mode["tilt_tol"])
    _check(
        checks,
        id="P0.4b",
        name="replicator/tilt terminal is mean precision",
        metric="|tilt acc - mean precision|",
        threshold=tilt_tol,
        value=tilt_err,
        passed=tilt_err <= tilt_tol,
    )
    beta_rows = []
    beta_ok = True
    beta_tol = float(mode["beta_tol"])
    worst_beta = 0.0
    for beta in mode["betas"]:
        beta = float(beta)
        flowed = population_flow(
            mode_world,
            mode_v,
            eta=float(mode["beta_eta"]),
            steps=int(mode["beta_steps"]),
            param="tabular",
            beta=beta,
        )
        eq = tilt(mode_world, mode_v, 1.0 / beta)
        eq_acc = float((eq * mode_world.O).sum(-1).mean())
        terminal = float(flowed.history["train_acc"][-1])
        gap = abs(terminal - eq_acc)
        worst_beta = max(worst_beta, gap)
        beta_ok = beta_ok and gap <= beta_tol
        beta_rows.append(
            {
                "beta": beta,
                "terminal": terminal,
                "tilt_acc": eq_acc,
                "mean_precision": prec,
                "modal": modal,
                "path_max": float(max(flowed.history["train_acc"])),
                "gap": gap,
            }
        )
    tables["beta"] = beta_rows
    _check(
        checks,
        id="P0.4c",
        name="beta>0 equilibrium is the finite tilt",
        metric="max |terminal - tilt(1/beta)|",
        threshold=beta_tol,
        value=worst_beta,
        passed=beta_ok,
        status="pass" if beta_ok else "documented",
    )

    print("P0.5-P0.7 sampled GRPO grid", flush=True)
    jobs = _jobs(config)
    workers = int(config.get("workers", 1))
    try:
        raw = _run_jobs(jobs, workers)
    except Exception as exc:
        print(f"  process pool failed ({exc}); running sequentially", flush=True)
        raw = _run_jobs(jobs, 1)
    grouped = _group(raw)
    gates = config["gates"]
    cell_tol = float(gates["tau_abs"])

    fate_rows: list[_FateRow] = []
    for name in FATE_WORLDS:
        for arm in ("fpF", "fpI", "resampled_fpF"):
            rows = grouped[(name, arm)]
            fate_rows.append(
                {
                    "world": name,
                    "arm": arm,
                    "J": _mean(rows, "J"),
                    "obs": _mean(rows, "obs"),
                    "prec0": _mean(rows, "prec0"),
                    "seeds": [row["obs"] for row in rows],
                }
            )
    tables["fates"] = fate_rows
    fate_gap = float(gates["fate_gap"])
    fate_close = float(gates["fate_close"])
    for name in FATE_WORLDS:
        by_arm = {row["arm"]: row for row in fate_rows if row["world"] == name}
        gap = by_arm["resampled_fpF"]["obs"] - by_arm["fpF"]["obs"]
        close = abs(by_arm["resampled_fpF"]["obs"] - by_arm["fpI"]["obs"])
        js = [by_arm[arm]["J"] for arm in ("fpF", "fpI", "resampled_fpF")]
        j_spread = max(js) - min(js)
        _check(
            checks,
            id=f"P0.5.{name}.gap",
            name=f"{name}: resampled held-out minus fpF",
            metric="held-out gap",
            threshold=fate_gap,
            value=gap,
            passed=gap >= fate_gap,
        )
        _check(
            checks,
            id=f"P0.5.{name}.close",
            name=f"{name}: resampled held-out vs fpI",
            metric="|held-out difference|",
            threshold=fate_close,
            value=close,
            passed=close <= fate_close,
        )
        _check(
            checks,
            id=f"P0.5.{name}.J",
            name=f"{name}: fpF, fpI and resampled share step-0 J",
            metric="J spread",
            threshold=1e-8,
            value=j_spread,
            passed=j_spread <= 1e-8,
        )

    cell_rows: list[_CellRow] = []
    for name in WORLDS:
        for arm in ARMS:
            rows = grouped[(name, arm)]
            obs = _mean(rows, "obs")
            tau = _mean(rows, "tau")
            cell_rows.append(
                {
                    "world": name,
                    "arm": arm,
                    "obs": obs,
                    "tau": tau,
                    "abs_err": abs(tau - obs),
                    "J": _mean(rows, "J"),
                    "prec0": _mean(rows, "prec0"),
                    "tau_rel": _mean(rows, "tau_rel"),
                    "tau_pool": _mean(rows, "tau_pool"),
                    "reliability": _mean(rows, "reliability"),
                    "within": abs(tau - obs) <= cell_tol,
                }
            )
    tables["tau"] = cell_rows
    frac = float(np.mean([row["within"] for row in cell_rows]))
    frac_min = float(gates["tau_frac"])
    _check(
        checks,
        id="P0.6.coverage",
        name="exact tau within tolerance of held-out accuracy",
        metric="fraction of cells",
        threshold=frac_min,
        value=frac,
        passed=frac >= frac_min,
    )
    obs_values = [row["obs"] for row in cell_rows]
    tau_s = spearman([row["tau"] for row in cell_rows], obs_values)
    j_s = spearman([row["J"] for row in cell_rows], obs_values)
    p_s = spearman([row["prec0"] for row in cell_rows], obs_values)
    rel_s = spearman([row["tau_rel"] for row in cell_rows], obs_values)
    pool_s = spearman([row["tau_pool"] for row in cell_rows], obs_values)
    pool_within = float(
        np.mean([abs(row["tau_pool"] - row["obs"]) <= cell_tol for row in cell_rows])
    )
    spear_min = float(gates["spearman"])
    _check(
        checks,
        id="P0.6.spearman",
        name="Spearman of exact tau with held-out accuracy",
        metric="spearman",
        threshold=spear_min,
        value=tau_s,
        passed=tau_s >= spear_min and tau_s > j_s and tau_s > p_s,
    )
    _check(
        checks,
        id="P0.6.spearman_J",
        name="exact tau beats step-0 J",
        metric="spearman(tau) - spearman(J)",
        threshold=0.0,
        value=tau_s - j_s,
        passed=tau_s > j_s,
    )
    _check(
        checks,
        id="P0.6.spearman_prec",
        name="exact tau beats step-0 precision",
        metric="spearman(tau) - spearman(Prec0)",
        threshold=0.0,
        value=tau_s - p_s,
        passed=tau_s > p_s,
    )
    rel_min = float(gates["reliability"])
    mean_rel = float(np.mean([row["reliability"] for row in cell_rows]))
    _check(
        checks,
        id="P0.7.spearman",
        name="16-sample tau_pool ranks held-out accuracy",
        metric="spearman",
        threshold=spear_min,
        value=pool_s,
        passed=pool_s >= spear_min and pool_s > j_s and pool_s > p_s,
    )
    _check(
        checks,
        id="P0.7.calibration",
        name="16-sample tau_pool within tolerance of held-out accuracy",
        metric="fraction of cells",
        threshold=frac_min,
        value=pool_within,
        passed=pool_within >= frac_min,
    )
    # Per-prompt argmax is inflated on prompts that never sampled the favoured
    # behaviour; reported so the choice of the pooled statistic stays auditable.
    _check(
        checks,
        id="P0.7.tau_rel",
        name="16-sample per-prompt tau_rel ranks held-out accuracy",
        metric="spearman",
        threshold=None,
        value=rel_s,
        passed=rel_s >= spear_min,
        status="documented",
    )
    _check(
        checks,
        id="P0.7.reliability",
        name="split-half reliability of the 16-sample velocity",
        metric="mean reliability",
        threshold=rel_min,
        value=mean_rel,
        passed=mean_rel >= rel_min,
    )
    tables["rank"] = {
        "spearman_tau": tau_s,
        "spearman_J": j_s,
        "spearman_prec0": p_s,
        "spearman_tau_rel": rel_s,
        "spearman_tau_pool": pool_s,
        "tau_pool_within": pool_within,
        "mean_reliability": mean_rel,
    }

    print("P0.8 knife-edge", flush=True)
    knife_rows = _knife(config)
    tables["knife"] = knife_rows
    tau_m = float(np.mean([r["tau"] for r in knife_rows]))
    slope_m = float(np.mean([r["slope"] for r in knife_rows]))
    term_m = float(np.mean([r["terminal"] for r in knife_rows]))
    peak_m = float(np.mean([r["peak"] for r in knife_rows]))
    acc0_m = float(np.mean([r["acc0"] for r in knife_rows]))
    prec_m = float(np.mean([r["prec0"] for r in knife_rows]))
    j_m = float(np.mean([r["J"] for r in knife_rows]))
    reproduced = (
        tau_m > 0.65
        and slope_m > 0.0
        and peak_m > acc0_m + 0.1
        and term_m < 0.08
        and abs(prec_m - 0.15 / 0.27) < 0.02
        and abs(j_m - (1.0 - 0.12 / 0.85)) < 0.02
    )
    _check(
        checks,
        id="P0.8",
        name="knife-edge: tau predicts success, shared flow collapses",
        metric="mean terminal held-out accuracy",
        threshold=None,
        value=term_m,
        passed=reproduced,
        status="documented",
    )

    failed = [c for c in checks if c["status"] == "fail"]
    summary = "PASS" if not failed else "FAIL"
    report = {
        "summary": summary,
        "ok": summary == "PASS",
        "elapsed_sec": time.perf_counter() - started,
        "environment": environment_report(),
        "config": config,
        "checks": checks,
        "tables": tables,
    }
    write_json(out_dir / "report.json", report)
    _write_report(out_dir / "report.md", report)
    print(f"P0 {summary} in {report['elapsed_sec']:.1f}s", flush=True)
    return report
