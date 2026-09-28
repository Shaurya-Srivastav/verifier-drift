"""The preregistered program: cells, rendered configs, job plans, transfer eval.

A cell is one ``(experiment, domain, model)``. ``configs/program.yaml`` lists
every cell. ``render_program`` fills ``configs/templates/<domain>.yaml`` from
the model spec and writes ``configs/cells/<id>.yaml``. ``plan_jobs`` expands
that into audit, train, early-baseline, variant, and transfer-eval jobs.

Model specs always come from ``vfates.models.load_model_spec``
(``configs/models/<name>.yaml``). Rendered cells record that source.

GPU-hour scaling (:data:`SCALING_RULE`):

    gpu_hours on a job are single-GPU H100-equivalent token-hours, times
    params_b / reference_params_b (default reference 1.7), unless
    configs/throughput.yaml has a measured rate for that model and class.
    A measured rate replaces the scaled placeholder and is not scaled again.
    h100_gen_tok_s and h100_train_tok_s are per-GPU rates. A multi-GPU
    placement is assumed to scale token throughput with the GPU count, so
    hours are not multiplied or divided by placement.gpus. Class GPU-hours
    are those H100-equivalent hours divided by hardware_policy.speed_vs_h100.
    An audit job prices audit tokens as generation only. A train job prices
    rollout tokens, eval tokens, and a train-token pass. An early job uses
    early_steps in place of steps. A variant job uses the train formula, with
    that variant's step count when ``train_overrides.steps`` is set. A
    transfer job prices one held-out eval (transfer_prompts, or eval_prompts
    when that key is absent, times eval_samples times mean_completion_len) as
    generation only.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from vfates.study import planned_seeds
from vfates.train.config import LORA_KEYS_PEFT, resolve_lora_keys
from vfates.utils.io import read_json, read_yaml, write_json

SCALING_RULE = (
    "GPU hours are the template cost model's single-GPU H100 token-hours for one "
    "job, times params_b / reference_params_b (default reference 1.7). "
    "h100_gen_tok_s and h100_train_tok_s are per-GPU rates. A multi-GPU placement "
    "is assumed to scale token throughput with the GPU count, so GPU hours are "
    "not multiplied or divided by placement.gpus. An audit job prices audit "
    "tokens as generation only. A train job prices rollout tokens, eval tokens, "
    "and a train-token pass. An early job uses early_steps in place of steps. "
    "A transfer job prices one held-out eval (transfer_prompts, or eval_prompts "
    "when that key is absent, times eval_samples times mean_completion_len) as "
    "generation only. A variant job uses the train formula, with that variant's "
    "step count when train_overrides.steps is set. A measured rate in "
    "configs/throughput.yaml replaces the scaled placeholder for that model "
    "and class and is not scaled again. Class GPU-hours divide the "
    "H100-equivalent hours by hardware_policy.speed_vs_h100."
)

HARDWARE_CLASSES: tuple[str, ...] = (
    "h100-80gb",
    "a100-80gb",
    "a100-40gb",
    "rtx-a6000-48gb",
    "rtx-5000-ada-32gb",
    "apple-silicon",
    "cpu",
)

DOMAINS: tuple[str, ...] = ("chain_sum", "math", "code", "ifeval", "qa")

KIT_ARMS: tuple[str, ...] = (
    "oracle",
    "fp_hash",
    "fp_trigger",
    "noise_fp",
    "noise_fp_trigger",
    "fn_behavior",
    "fn_prompt_subset",
    "noise_fn",
)

CHAIN_SUM_FULL: tuple[str, ...] = (
    "oracle",
    "fp_hash",
    "fp_tolerance",
    "fp_tolerance_decoy",
    "fp_trigger",
    "fp_trigger_enriched",
    "noise_fp",
    "noise_fp_trigger",
    "noise_fp_trigger_enriched",
    "fn_behavior",
    "fn_behavior_depleted",
    "fn_prompt_subset",
    "noise_fn",
    "noise_fn_depleted",
    "fn_form",
    "noise_fn_form",
    "combo_fnform_fphash",
)

_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_.]*)\}$")
_PLACEMENT_KEYS = ("gpus", "tp", "micro_batch_size", "vllm_gpu_memory_utilization")
_SPEED_CLASSES = (
    "h100-80gb",
    "a100-80gb",
    "a100-40gb",
    "rtx-a6000-48gb",
    "rtx-5000-ada-32gb",
)


def repo_root() -> Path:
    """Repository root (the directory that contains ``configs`` and ``src``)."""
    return Path(__file__).resolve().parents[2]


def _template_label(domain: str) -> str:
    return f"configs/templates/{domain}.yaml"


def require_menu_arms(template: Mapping[str, Any] | None, menu: str, *, domain: str) -> list[str]:
    """Arms for ``menu`` from the domain template.

    A missing template or a template without that menu raises ``ValueError``
    naming ``configs/templates/<domain>.yaml`` and the menu.
    """
    label = _template_label(domain)
    menus = template.get("menus") if isinstance(template, Mapping) else None
    if not isinstance(menus, Mapping) or menu not in menus:
        raise ValueError(f"{label} is missing menu {menu!r}")
    arms = menus[menu]
    if not isinstance(arms, Sequence) or isinstance(arms, str) or not arms:
        raise ValueError(f"{label} menu {menu!r} must be a non-empty list of arms")
    return [str(arm) for arm in arms]


def model_field_source() -> str:
    """Model fields always come from ``vfates.models``."""
    return "vfates.models"


def load_model_spec(name: str) -> Any:
    """Load ``configs/models/<name>.yaml`` through ``vfates.models``."""
    from vfates.models import load_model_spec as real

    return real(name)


def _get(spec: Any, name: str) -> Any:
    if isinstance(spec, Mapping):
        if name not in spec:
            raise KeyError(f"model spec is missing {name}")
        return spec[name]
    if not hasattr(spec, name):
        raise KeyError(f"model spec is missing {name}")
    return getattr(spec, name)


def placement_view(spec: Any, hardware_class: str, *, full: bool = False) -> dict[str, Any]:
    """``gpus``, ``tp``, ``micro_batch_size``, ``vllm_gpu_memory_utilization``.

    ``full`` uses ``placement_full_for`` (full-parameter runs).
    """
    method = spec.placement_full_for if full else spec.placement_for
    raw = method(hardware_class)
    if isinstance(raw, Mapping):
        data = dict(raw)
    else:
        data = {key: getattr(raw, key) for key in _PLACEMENT_KEYS}
    missing = [key for key in _PLACEMENT_KEYS if key not in data]
    if missing:
        raise ValueError(f"placement for {hardware_class} is missing {missing}")
    gpus = int(data["gpus"])
    tp = int(data["tp"])
    micro = int(data["micro_batch_size"])
    util = float(data["vllm_gpu_memory_utilization"])
    if gpus < 1 or tp < 1 or tp > gpus:
        raise ValueError(f"placement gpus={gpus} tp={tp} is not usable")
    if micro < 1:
        raise ValueError(f"micro_batch_size must be positive, got {micro}")
    if not 0.0 < util <= 1.0:
        raise ValueError(f"vllm_gpu_memory_utilization must lie in (0, 1], got {util}")
    return {
        "gpus": gpus,
        "tp": tp,
        "micro_batch_size": micro,
        "vllm_gpu_memory_utilization": util,
    }


def model_view(spec: Any) -> dict[str, Any]:
    """Fields the template placeholders read off a model spec."""
    lora_keys = [str(key) for key in _get(spec, "lora_keys")]
    kwargs = _get(spec, "chat_template_kwargs") or {}
    if not isinstance(kwargs, Mapping):
        raise ValueError("chat_template_kwargs must be a mapping or null")
    revision = _get(spec, "revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("model revision must be a non-empty string")
    return {
        "model_id": str(_get(spec, "model_id")),
        "revision": revision.strip(),
        "chat_template_kwargs": dict(kwargs),
        "lr": float(_get(spec, "lr")),
        "lora_keys": lora_keys,
        "lora_target_modules": resolve_lora_keys(lora_keys, LORA_KEYS_PEFT),
        "params_b": float(_get(spec, "params_b")),
    }


def template_path(root: Path, domain: str) -> Path:
    return root / "configs" / "templates" / f"{domain}.yaml"


def load_template(root: Path, domain: str) -> dict[str, Any] | None:
    path = template_path(root, domain)
    if not path.is_file():
        return None
    loaded = read_yaml(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return loaded


def analysis_arm_names(block: Mapping[str, Any]) -> set[str]:
    """Arms named by one menu's h1, h2, h3, h4, h10, and natural blocks."""
    names: set[str] = set()
    h1 = block.get("h1") or {}
    for key in ("contrasts", "beside"):
        for row in h1.get(key) or []:
            names.add(str(row["control"]))
            names.add(str(row["arm"]))
    for row in (block.get("h2") or {}).get("contrasts") or []:
        names.add(str(row["control"]))
        names.add(str(row["arm"]))
    for row in (block.get("h3") or {}).get("pairs") or []:
        names.add(str(row["arm"]))
        names.add(str(row["control"]))
    for row in (block.get("h10") or {}).get("pairs") or []:
        names.add(str(row["a"]))
        names.add(str(row["b"]))
    h4 = block.get("h4") or {}
    for arm in list(h4.get("shared") or []) + list(h4.get("prompt_specific") or []):
        names.add(str(arm))
    natural = block.get("natural")
    if isinstance(natural, Mapping):
        for key in ("natural", "control", "oracle"):
            if natural.get(key) is not None:
                names.add(str(natural[key]))
    elif isinstance(natural, list):
        for item in natural:
            for key in ("natural", "control", "oracle"):
                if item.get(key) is not None:
                    names.add(str(item[key]))
    return names


def check_template_menus(template: Mapping[str, Any], *, label: str = "template") -> None:
    """Each menu's analysis may name only arms that are in that menu.

    ``beside`` must be present (an empty list is fine). Omitting it would
    fall back to the chain-sum decoy contrast inside the confirmatory report.
    """
    menus = template.get("menus")
    if not isinstance(menus, Mapping) or not menus:
        raise ValueError(f"{label} is missing menus.full and menus.core")
    by_menu = (template.get("analysis") or {}).get("by_menu")
    if not isinstance(by_menu, Mapping):
        raise ValueError(f"{label} is missing analysis.by_menu")
    if set(menus) != set(by_menu):
        raise ValueError(
            f"{label} menus {sorted(menus)} do not match analysis.by_menu {sorted(by_menu)}"
        )
    if "core" in menus and "full" in menus:
        extra = set(menus["core"]) - set(menus["full"])
        if extra:
            raise ValueError(f"{label} core menu is not a subset of full: {sorted(extra)}")
    for name, arms in menus.items():
        if not isinstance(arms, Sequence) or isinstance(arms, str) or not arms:
            raise ValueError(f"{label} menu {name} must be a non-empty list of arms")
        seen: set[str] = set()
        for arm in arms:
            text = str(arm)
            if text in seen:
                raise ValueError(f"{label} menu {name} lists {text} twice")
            seen.add(text)
        block = by_menu[name]
        if not isinstance(block, Mapping):
            raise ValueError(f"{label} analysis.by_menu.{name} must be a mapping")
        h1 = block.get("h1") or {}
        if "beside" not in h1:
            raise ValueError(
                f"{label} menu {name} h1.beside is missing. Set it to [] when there is no "
                "beside contrast, or the confirmatory default (fp_tolerance_decoy) is used."
            )
        outside = analysis_arm_names(block) - seen
        if outside:
            raise ValueError(
                f"{label} menu {name} analysis names arms outside the menu: {sorted(outside)}"
            )


def _lookup(expr: str, ctx: Mapping[str, Any]) -> Any:
    current: Any = ctx
    for part in expr.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise KeyError(expr)
    return current


def substitute_placeholders(node: Any, ctx: Mapping[str, Any]) -> Any:
    """Replace values that are exactly ``${a.b}`` with the context object.

    A placeholder may expand to a mapping, list, number, or string. A string
    that only contains a placeholder in passing is an error, so types cannot
    be spliced into prose by accident.
    """
    if isinstance(node, str):
        match = _PLACEHOLDER.match(node)
        if match:
            return copy.deepcopy(_lookup(match.group(1), ctx))
        if "${" in node:
            raise ValueError(f"partial placeholder is not supported: {node}")
        return node
    if isinstance(node, list):
        return [substitute_placeholders(item, ctx) for item in node]
    if isinstance(node, dict):
        return {key: substitute_placeholders(value, ctx) for key, value in node.items()}
    return node


def _completions_per_step(template: Mapping[str, Any]) -> int:
    train = (template.get("study") or {}).get("train") or {}
    prompts = int(train.get("prompts_per_step", 64))
    group = int(train.get("group_size", 8))
    return prompts * group


def _check_micro_batch(micro: int, completions: int, *, where: str) -> None:
    if completions % micro != 0:
        raise ValueError(
            f"{where}: micro_batch_size {micro} must divide prompts_per_step * group_size "
            f"({completions})"
        )


def render_context(cell: Mapping[str, Any], spec: Any, menu_arms: Sequence[str]) -> dict[str, Any]:
    """Placeholder context for one cell."""
    model = model_view(spec)
    placement = placement_view(spec, str(cell["hardware_class"]))
    return {
        "model": model,
        "placement": placement,
        "cell": {
            "id": str(cell["id"]),
            "hardware_class": str(cell["hardware_class"]),
            "seeds": [int(seed) for seed in cell["seeds"]],
            "gpus": placement["gpus"],
        },
        "menu": {"name": str(cell["menu"]), "arms": list(menu_arms)},
    }


def _deep_merge(dst: dict[str, Any], src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        current = dst.get(key)
        if isinstance(value, Mapping) and isinstance(current, dict):
            _deep_merge(current, value)
        else:
            dst[key] = copy.deepcopy(value)


def _cell_arms(cell: Mapping[str, Any], menu_arms: Sequence[str]) -> list[str]:
    """Menu arms, or the cell's explicit ``arms`` list when it overrides the menu."""
    if not cell.get("arms"):
        return [str(arm) for arm in menu_arms]
    chosen = [str(arm) for arm in cell["arms"]]
    extra = [arm for arm in chosen if arm not in set(menu_arms)]
    if extra:
        raise ValueError(f"{cell.get('id')}: arms {extra} are not in the {cell.get('menu')} menu")
    if not chosen:
        raise ValueError(f"{cell.get('id')}: arms override is empty")
    return chosen


def _apply_cell_overrides(doc: dict[str, Any], cell: Mapping[str, Any]) -> None:
    audit_over = cell.get("audit_overrides") or {}
    train_over = cell.get("train_overrides") or {}
    if audit_over:
        _deep_merge(doc["audit"], audit_over)
    if train_over:
        _deep_merge(doc["study"]["train"], train_over)
    if cell.get("hypotheses") is False:
        doc["analysis"]["hypotheses"] = False
    if cell.get("early") is False:
        early = doc["analysis"].setdefault("early", {})
        if isinstance(early, dict):
            early["enabled"] = False
    trained = set(doc["study"]["arms"])
    if cell.get("arms") and "natural" not in trained:
        doc["analysis"].pop("natural", None)


def _rendered_variants(
    cell: Mapping[str, Any], spec: Any, template: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Copy ``study.variants`` and resolve ``placement: full`` into train overrides."""
    hardware = str(cell["hardware_class"])
    completions = _completions_per_step(template)
    rendered: list[dict[str, Any]] = []
    for raw in cell.get("variants") or []:
        row = copy.deepcopy(dict(raw))
        overrides = dict(row.get("train_overrides") or {})
        if row.get("placement") == "full":
            place = placement_view(spec, hardware, full=True)
            _check_micro_batch(
                int(place["micro_batch_size"]),
                completions,
                where=f"{cell.get('id')} variant {row.get('name')}",
            )
            overrides["micro_batch_size"] = place["micro_batch_size"]
            overrides["vllm_gpu_memory_utilization"] = place["vllm_gpu_memory_utilization"]
            overrides["vllm_tensor_parallel_size"] = place["tp"]
            row["gpus"] = int(place["gpus"])
        row["train_overrides"] = overrides
        rendered.append(row)
    return rendered


def render_cell_document(
    cell: Mapping[str, Any],
    template: Mapping[str, Any],
    spec: Any,
    *,
    source: str,
) -> str:
    """Deterministic YAML text for one cell, including the generated header."""
    check_template_menus(template, label=str(cell.get("domain", "template")))
    menu_name = str(cell["menu"])
    arms = [str(arm) for arm in template["menus"][menu_name]]
    ctx = render_context(cell, spec, arms)
    _check_micro_batch(
        int(ctx["placement"]["micro_batch_size"]),
        _completions_per_step(template),
        where=str(cell["id"]),
    )
    doc = copy.deepcopy(dict(template))
    doc.pop("menus", None)
    by_menu = doc["analysis"].pop("by_menu")
    doc["analysis"].update(copy.deepcopy(by_menu[menu_name]))
    doc = substitute_placeholders(doc, ctx)
    doc["title"] = f"{cell['experiment']} {cell['domain']} {cell['model']} ({menu_name})"
    chosen = _cell_arms(cell, arms)
    doc["study"]["arms"] = list(chosen)
    doc["study"]["seeds"] = [int(seed) for seed in cell["seeds"]]
    doc["study"]["hardware_class"] = str(cell["hardware_class"])
    doc["study"]["gpus"] = int(ctx["placement"]["gpus"])
    rule = doc["study"].get("seed_rule")
    if isinstance(rule, dict):
        allowed = set(chosen)
        rule["arms"] = [str(arm) for arm in rule.get("arms", []) if str(arm) in allowed]
    _apply_cell_overrides(doc, cell)
    if cell.get("oracle_gain_min") is not None:
        doc["analysis"]["oracle_gain_min"] = float(cell["oracle_gain_min"])
    doc["study"]["variants"] = _rendered_variants(cell, spec, template)
    header = "\n".join(
        [
            "# Generated by `vfates program render`. Do not edit by hand.",
            f"# cell: {cell['id']}",
            f"# template: configs/templates/{cell['domain']}.yaml",
            f"# model: {cell['model']}",
            f"# menu: {menu_name}",
            f"# model_source: {source}",
            "# Re-running render reproduces this file byte for byte.",
            "",
        ]
    )
    import yaml

    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
    text = header + body
    if not text.endswith("\n"):
        text += "\n"
    if "${" in text:
        raise ValueError(f"{cell['id']}: an unfilled placeholder survived rendering")
    return text


def load_program(
    path: str | Path, *, validate: bool = True, root: Path | None = None
) -> dict[str, Any]:
    """Load ``configs/program.yaml``. Validates unless ``validate`` is false."""
    loaded = read_yaml(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a mapping")
    if validate:
        validate_program(loaded, root=root or repo_root())
    return loaded


def _validate_variants(
    cell: Mapping[str, Any],
    menu_arms: Sequence[str],
    spec: Any,
    bad: Any,
) -> None:
    variants = cell.get("variants") or []
    if not isinstance(variants, list):
        bad(f"{cell.get('id')}: variants must be a list")
        return
    trained = {str(arm) for arm in (cell.get("arms") or menu_arms)}
    names: set[str] = set()
    hardware = str(cell.get("hardware_class", ""))
    for row in variants:
        if not isinstance(row, Mapping):
            bad(f"{cell.get('id')}: a variant is not a mapping")
            continue
        name = str(row.get("name") or "")
        if not name or name in names:
            bad(f"{cell.get('id')}: variant name {name!r} is missing or repeated")
        names.add(name)
        varms = row.get("arms") or []
        if not isinstance(varms, list) or not varms:
            bad(f"{cell.get('id')} variant {name}: arms must be a non-empty list")
        else:
            outside = [str(arm) for arm in varms if str(arm) not in trained]
            if outside:
                bad(f"{cell.get('id')} variant {name}: arms outside the cell menu: {outside}")
        seeds = row.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
        ):
            bad(f"{cell.get('id')} variant {name}: seeds must be a non-empty list of ints")
        placement = row.get("placement")
        if placement not in (None, "full"):
            bad(f"{cell.get('id')} variant {name}: placement must be full or omitted")
        if placement == "full":
            try:
                placement_view(spec, hardware, full=True)
            except (KeyError, ValueError, AttributeError, TypeError) as exc:
                bad(f"{cell.get('id')} variant {name}: {exc}")
        overrides = row.get("train_overrides") or {}
        if not isinstance(overrides, Mapping):
            bad(f"{cell.get('id')} variant {name}: train_overrides must be a mapping")
        accel = row.get("accelerate_config")
        if accel is not None and not isinstance(accel, str):
            bad(f"{cell.get('id')} variant {name}: accelerate_config must be a path")


def _validate_hardware_budget(program: Mapping[str, Any], ids: set[str], bad: Any) -> None:
    policy = program.get("hardware_policy") or {}
    speed = policy.get("speed_vs_h100") or {}
    if not isinstance(speed, Mapping):
        bad("hardware_policy.speed_vs_h100 must be a mapping")
        speed = {}
    for name in _SPEED_CLASSES:
        value = speed.get(name)
        try:
            if value is None or float(value) <= 0:
                bad(f"hardware_policy.speed_vs_h100.{name} must be a positive number")
        except (TypeError, ValueError):
            bad(f"hardware_policy.speed_vs_h100.{name} must be a positive number")
    usd = policy.get("usd_per_gpu_hour") or {}
    if not isinstance(usd, Mapping):
        bad("hardware_policy.usd_per_gpu_hour must be a mapping")
    elif "h100-80gb" not in usd:
        bad("hardware_policy.usd_per_gpu_hour.h100-80gb is required")
    compute = program.get("compute") or {}
    cut = compute.get("cut_order") if isinstance(compute, Mapping) else None
    if not isinstance(cut, list) or not cut:
        bad("compute.cut_order must be a non-empty list of cell ids")
        return
    for cell_id in cut:
        if str(cell_id) not in ids:
            bad(f"compute.cut_order names unknown cell {cell_id!r}")


def cell_stage(program: Mapping[str, Any], cell_id: str) -> str | None:
    """Stage id (``A`` or ``B``) for a confirmatory cell, or None.

    Exploratory cells are not listed under ``stages`` and return None.
    """
    stages = program.get("stages")
    if not isinstance(stages, Mapping):
        return None
    for key, block in stages.items():
        if not isinstance(block, Mapping):
            continue
        listed = block.get("cells") or []
        if not isinstance(listed, Sequence) or isinstance(listed, (str, bytes)):
            continue
        if str(cell_id) in {str(item) for item in listed}:
            return str(key)
    return None


def _validate_stages(
    program: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], bad: Any
) -> None:
    """Every confirmatory cell is in exactly one stage. Cuts stay in stage B."""
    stages = program.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != {"A", "B"}:
        bad("stages must map exactly A and B")
        return
    confirmatory = {
        str(cell.get("id"))
        for cell in cells
        if isinstance(cell, Mapping) and str(cell.get("experiment")) != "X"
    }
    exploratory = {
        str(cell.get("id"))
        for cell in cells
        if isinstance(cell, Mapping) and str(cell.get("experiment")) == "X"
    }
    seen: dict[str, str] = {}
    for key in ("A", "B"):
        block = stages.get(key)
        if not isinstance(block, Mapping):
            bad(f"stages.{key} must be a mapping")
            continue
        name = block.get("name")
        if not isinstance(name, str) or not name.strip():
            bad(f"stages.{key}.name must be a non-empty string")
        listed = block.get("cells")
        if not isinstance(listed, list) or not listed:
            bad(f"stages.{key}.cells must be a non-empty list of cell ids")
            continue
        for cell_id in listed:
            text = str(cell_id)
            if text in seen:
                bad(f"{text} is in stage {seen[text]} and stage {key}")
            seen[text] = key
            if text in exploratory:
                bad(f"exploratory cell {text} is in stage {key}")
            elif text not in confirmatory:
                bad(f"stages.{key} names unknown cell {text!r}")
    for cell_id in sorted(confirmatory - set(seen)):
        bad(f"confirmatory cell {cell_id} is not in a stage")
    compute = program.get("compute") or {}
    cut = compute.get("cut_order") if isinstance(compute, Mapping) else None
    if not isinstance(cut, list):
        return
    for cell_id in cut:
        text = str(cell_id)
        if seen.get(text) != "B":
            bad(f"compute.cut_order cell {text} must be in stage B")


def validate_program(program: Mapping[str, Any], *, root: Path | None = None) -> None:
    """Refuse a program whose cells, menus, or model placements disagree.

    Every cell's model must resolve through :func:`load_model_spec` and expose
    a placement for that cell's ``hardware_class``.
    """
    root = root or repo_root()
    errors: list[str] = []

    def bad(message: str) -> None:
        errors.append(message)

    cells = program.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("program.cells must be a non-empty list")
    policy = program.get("hardware_policy") or {}
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            bad("a cell entry is not a mapping")
            continue
        cell_id = str(cell.get("id", ""))
        experiment = str(cell.get("experiment", ""))
        domain = str(cell.get("domain", ""))
        model = str(cell.get("model", ""))
        menu = str(cell.get("menu", ""))
        hardware = str(cell.get("hardware_class", ""))
        expected_id = f"{experiment.lower()}-{domain}-{model}"
        if experiment == "X":
            if not (cell_id.startswith("x") and domain in cell_id and model in cell_id and cell_id):
                bad(f"cell id {cell_id!r} must start with x and contain {domain} and {model}")
        elif cell_id != expected_id:
            bad(f"cell id {cell_id!r} should be {expected_id!r}")
        if cell_id in seen:
            bad(f"duplicate cell id {cell_id}")
        seen.add(cell_id)
        if experiment not in ("E1", "E2", "E3", "X"):
            bad(f"{cell_id}: experiment must be E1, E2, E3, or X")
        if domain not in DOMAINS:
            bad(f"{cell_id}: unknown domain {domain}")
        if menu not in ("full", "core"):
            bad(f"{cell_id}: menu must be full or core")
        if hardware not in HARDWARE_CLASSES:
            bad(f"{cell_id}: unknown hardware_class {hardware}")
        config = str(cell.get("config", ""))
        if config != f"configs/cells/{cell_id}.yaml":
            bad(f"{cell_id}: config path must be configs/cells/{cell_id}.yaml")
        seeds = cell.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(isinstance(s, bool) for s in seeds):
            bad(f"{cell_id}: seeds must be a non-empty list of integers")
        elif experiment == "E1" and [int(s) for s in seeds] != [0, 1, 2]:
            bad(f"{cell_id}: E1 seeds must be [0, 1, 2]")
        elif experiment in ("E2", "E3") and [int(s) for s in seeds] != [0, 1]:
            bad(f"{cell_id}: {experiment} seeds must be [0, 1]")
        if experiment == "E1" and menu != "full":
            bad(f"{cell_id}: E1 uses the full menu")
        if experiment in ("E2", "E3") and menu != "core":
            bad(f"{cell_id}: {experiment} uses the core menu")
        if experiment == "E1" and model != "qwen3-1.7b":
            bad(f"{cell_id}: E1 model is qwen3-1.7b")
        rule = policy.get(experiment) or {}
        if rule.get("by") == "domain" and rule.get(domain) not in (None, hardware):
            bad(f"{cell_id}: hardware_policy[{experiment}].{domain} is {rule.get(domain)}")
        if rule.get("by") == "model" and rule.get(model) not in (None, hardware):
            bad(f"{cell_id}: hardware_policy[{experiment}].{model} is {rule.get(model)}")
        spec: Any = None
        try:
            spec = load_model_spec(model)
            placement_view(spec, hardware)
            model_view(spec)
        except (KeyError, ValueError, AttributeError, TypeError) as exc:
            bad(f"{cell_id}: {exc}")
        template = None
        try:
            template = load_template(root, domain)
        except ValueError as exc:
            bad(str(exc))
        if template is not None:
            try:
                check_template_menus(template, label=domain)
                completions = _completions_per_step(template)
                micro = placement_view(load_model_spec(model), hardware)["micro_batch_size"]
                _check_micro_batch(micro, completions, where=cell_id)
            except (KeyError, ValueError, AttributeError) as exc:
                bad(f"{cell_id}: {exc}")
        try:
            full = require_menu_arms(template, "full", domain=domain)
        except ValueError as exc:
            bad(str(exc))
            full = []
        if domain in ("math", "code", "ifeval", "qa") and menu == "full":
            missing_e4 = {"patch_tau", "patch_rate"} - set(full)
            if missing_e4:
                bad(f"{cell_id}: full menu is missing E4 arms {sorted(missing_e4)}")
        if menu == "full":
            menu_arms = list(full)
        else:
            try:
                menu_arms = require_menu_arms(template, menu, domain=domain)
            except ValueError as exc:
                bad(str(exc))
                menu_arms = []
        if cell.get("arms"):
            try:
                _cell_arms(cell, menu_arms)
            except ValueError as exc:
                bad(str(exc))
        if spec is not None:
            _validate_variants(cell, menu_arms, spec, bad)

    e4 = program.get("e4") or {}
    for domain in e4.get("domains") or []:
        if domain == "chain_sum":
            bad("E4 does not add patch arms to chain_sum")
        if domain not in ("math", "code", "ifeval", "qa"):
            bad(f"E4 domain {domain} is not a patched domain")
    exploratory = program.get("exploratory") or {}
    if "X1" not in exploratory:
        bad("exploratory.X1 is missing")
    ids = {str(cell.get("id")) for cell in cells if isinstance(cell, Mapping)}
    _validate_hardware_budget(program, ids, bad)
    listed = [cell for cell in cells if isinstance(cell, Mapping)]
    _validate_stages(program, listed, bad)
    if errors:
        raise ValueError("program is invalid:\n" + "\n".join(f"- {item}" for item in errors))


def cell_with_experiment(program: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    """The cell with its experiment's settings that render writes into the cell config.

    Today that is ``oracle_gain_min``, the conclusive-cell rule, when the
    experiment sets it; otherwise the template's value stands.
    """
    experiment = (program.get("experiments") or {}).get(str(cell["experiment"])) or {}
    if experiment.get("oracle_gain_min") is None:
        return dict(cell)
    return {**cell, "oracle_gain_min": float(experiment["oracle_gain_min"])}


def render_program(
    path: str | Path,
    *,
    root: Path | None = None,
    only: Sequence[str] | None = None,
) -> list[Path]:
    """Write ``configs/cells/<id>.yaml`` for every cell.

    A domain template that is missing, or that lacks the cell's menu, raises
    ``ValueError`` naming the template and the menu. Rendering is
    deterministic: a second call writes the same bytes.
    """
    root = root or repo_root()
    program = load_program(path, root=root)
    source = model_field_source()
    wanted = None if only is None else set(only)
    written: list[Path] = []
    for cell in program["cells"]:
        if wanted is not None and cell["id"] not in wanted:
            continue
        domain = str(cell["domain"])
        template = load_template(root, domain)
        require_menu_arms(template, str(cell["menu"]), domain=domain)
        assert template is not None
        spec = load_model_spec(str(cell["model"]))
        text = render_cell_document(
            cell_with_experiment(program, cell), template, spec, source=source
        )
        dest = root / str(cell["config"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        written.append(dest)
    return written


def _cost_for(program: Mapping[str, Any], root: Path, domain: str) -> tuple[dict[str, Any], str]:
    template = load_template(root, domain)
    if template is not None and isinstance(template.get("cost_model"), Mapping):
        return dict(template["cost_model"]), "template"
    raw = program.get("default_cost_model")
    if not isinstance(raw, Mapping):
        raise ValueError("program.default_cost_model is required when a template is missing")
    cost = dict(raw)
    lengths = program.get("domain_mean_completion_len") or {}
    if domain in lengths:
        cost["mean_completion_len"] = lengths[domain]
    return cost, "default"


def _early_of(
    template: Mapping[str, Any] | None, cell: Mapping[str, Any] | None = None
) -> tuple[bool, int, int]:
    if cell is not None and cell.get("early") is False:
        return False, 25, 99
    if template is None:
        return True, 25, 99
    early = (template.get("analysis") or {}).get("early") or {}
    return (
        bool(early.get("enabled", False)),
        int(early.get("steps", 25)),
        int(early.get("seed", 99)),
    )


def _menu_and_planned(
    cell: Mapping[str, Any], template: Mapping[str, Any] | None
) -> dict[str, list[int]]:
    domain = str(cell["domain"])
    menu = str(cell["menu"])
    menu_arms = require_menu_arms(template, menu, domain=domain)
    raw_rule = ((template or {}).get("study") or {}).get("seed_rule") or {"apply": False}
    rule = dict(raw_rule)
    arms = _cell_arms(cell, menu_arms)
    allowed = set(arms)
    rule["arms"] = [str(arm) for arm in rule.get("arms", []) if str(arm) in allowed]
    return planned_seeds({"seeds": list(cell["seeds"]), "arms": arms, "seed_rule": rule})


def job_gpu_hours(
    cost: Mapping[str, Any],
    kind: str,
    params_b: float,
    *,
    measured: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    """H100-equivalent GPU hours for one job, and ``measured`` or ``placeholder``.

    A measured rate (tokens per second per GPU) replaces the params_b scaling.
    See :data:`SCALING_RULE`.
    """
    reference = float(cost.get("reference_params_b", 1.7))
    if reference <= 0:
        raise ValueError("reference_params_b must be positive")
    length = float(cost["mean_completion_len"])
    steps = int(cost["steps"])
    group = int(cost["prompts_per_step"]) * int(cost["group_size"])
    overhead = float(cost.get("overhead", 1.0))
    eval_every = int(cost["eval_every"])
    eval_prompts = int(cost["eval_prompts"]) + int(cost.get("eval_train_prompts", 0))
    eval_samples = int(cost["eval_samples"])
    if measured is not None:
        gen_rate = float(measured["gen_tok_s_per_gpu"])
        train_rate = float(measured["train_tok_s_per_gpu"])
        scale = 1.0
        source = "measured"
    else:
        gen_rate = float(cost["h100_gen_tok_s"])
        train_rate = float(cost["h100_train_tok_s"])
        scale = float(params_b) / reference
        source = "placeholder"
    if gen_rate <= 0 or train_rate <= 0:
        raise ValueError("token rates must be positive")

    def hours(gen_tokens: float, train_tokens: float) -> float:
        return overhead * (gen_tokens / gen_rate + train_tokens / train_rate) / 3600.0 * scale

    priced = "train" if kind == "variant" else kind
    if priced == "audit":
        gen = float(cost["audit_prompts"]) * float(cost["audit_samples"]) * length
        return hours(gen, 0.0), source
    if priced == "transfer":
        prompts = int(cost.get("transfer_prompts", cost["eval_prompts"]))
        return hours(float(prompts) * float(eval_samples) * length, 0.0), source
    run_steps = steps
    if priced == "early":
        run_steps = int(cost.get("early_steps", 0))
    train_tokens = float(run_steps) * float(group) * length
    n_eval = (run_steps // eval_every + 1) if run_steps else 0
    eval_tokens = float(n_eval) * float(eval_prompts) * float(eval_samples) * length
    return hours(train_tokens + eval_tokens, train_tokens), source


def _blank_job(**fields: Any) -> dict[str, Any]:
    job: dict[str, Any] = {
        "state": "pending",
        "attempt": 0,
        "preemptions": 0,
        "claimed_by": None,
        "claimed_at": None,
        "heartbeat": None,
        "exit_code": None,
        "arm": None,
        "seed": None,
        "target_domain": None,
        "exploratory": None,
        "depends_on": [],
        "beta_kl": None,
        "audit_dir": None,
        "variant": None,
        "accelerate_config": None,
        "rate_source": "placeholder",
        "stage": None,
    }
    job.update(fields)
    return job


def load_throughput(root: Path | None = None) -> dict[str, Any]:
    """Measured per-GPU rates from ``configs/throughput.yaml``, or ``{}``.

    Schema ``vfates.throughput/1``: ``rates[model][hardware_class]`` has
    ``gen_tok_s_per_gpu`` and ``train_tok_s_per_gpu``.
    """
    root = root or repo_root()
    path = root / "configs" / "throughput.yaml"
    if not path.is_file():
        return {}
    loaded = read_yaml(path)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path} did not contain a mapping")
    if loaded.get("schema") != "vfates.throughput/1":
        raise ValueError(f"{path} schema must be vfates.throughput/1")
    rates = loaded.get("rates") or {}
    if not isinstance(rates, Mapping):
        raise ValueError(f"{path} rates must be a mapping")
    return dict(rates)


def throughput_rate(
    rates: Mapping[str, Any], model: str, hardware_class: str
) -> dict[str, Any] | None:
    """One measured rate row, or None when this model and class were not timed."""
    by_model = rates.get(model)
    if not isinstance(by_model, Mapping):
        return None
    row = by_model.get(hardware_class)
    if not isinstance(row, Mapping):
        return None
    if "gen_tok_s_per_gpu" not in row or "train_tok_s_per_gpu" not in row:
        return None
    return dict(row)


def _add_variant_jobs(add: Any, cell: Mapping[str, Any], out: str, audit_id: str) -> None:
    spec = load_model_spec(str(cell["model"]))
    hardware = str(cell["hardware_class"])
    for raw in cell.get("variants") or []:
        name = str(raw["name"])
        overrides = dict(raw.get("train_overrides") or {})
        gpus: int | None = None
        if raw.get("placement") == "full":
            gpus = int(placement_view(spec, hardware, full=True)["gpus"])
        elif raw.get("gpus") is not None:
            gpus = int(raw["gpus"])
        cost_overrides: dict[str, Any] = {}
        if "steps" in overrides:
            cost_overrides["steps"] = int(overrides["steps"])
        extra_common: dict[str, Any] = {}
        if gpus is not None:
            extra_common["gpus"] = gpus
        if raw.get("accelerate_config"):
            extra_common["accelerate_config"] = str(raw["accelerate_config"])
        if cost_overrides:
            extra_common["cost_overrides"] = cost_overrides
        for arm in raw.get("arms") or []:
            for seed in raw.get("seeds") or []:
                add(
                    cell,
                    "variant",
                    id=f"{cell['id']}/variant/{name}/{arm}/s{int(seed)}",
                    arm=str(arm),
                    seed=int(seed),
                    variant=name,
                    out_dir=out,
                    depends_on=[audit_id],
                    exploratory=True,
                    **extra_common,
                )


def plan_jobs(
    program: Mapping[str, Any],
    *,
    root: Path | None = None,
    runs_root: str = "runs",
) -> list[dict[str, Any]]:
    """One audit per cell, then one job per planned ``(cell, arm, seed)``.

    Early-baseline jobs are added when the domain template enables them (or,
    when the template is not written yet, by the E1 default of 25 steps at
    seed 99). A cell with ``early: false`` skips them. Transfer-eval jobs
    (X1) depend on the E1 train job they score. Variant jobs depend on the
    cell audit, stay exploratory, and write under
    ``runs/<cell>/variants/<name>/<arm>__s<seed>``. Experiment X cells are
    exploratory. ``gpu_hours`` stays H100-equivalent.
    """
    root = root or repo_root()
    jobs: list[dict[str, Any]] = []
    throughput = load_throughput(root)

    def add(cell: Mapping[str, Any], kind: str, **extra: Any) -> dict[str, Any]:
        domain = str(cell["domain"])
        spec = load_model_spec(str(cell["model"]))
        view = model_view(spec)
        placement = placement_view(spec, str(cell["hardware_class"]))
        cost, source = _cost_for(program, root, domain)
        if kind == "early":
            cost = {**cost, "early_steps": int(extra.pop("early_steps_for_cost"))}
        cost_overrides = extra.pop("cost_overrides", None)
        if isinstance(cost_overrides, Mapping):
            cost = {**cost, **dict(cost_overrides)}
        gpus_override = extra.pop("gpus", None)
        measured = throughput_rate(throughput, str(cell["model"]), str(cell["hardware_class"]))
        hours, rate_source = job_gpu_hours(cost, kind, view["params_b"], measured=measured)
        job = _blank_job(
            id=extra.pop("id"),
            kind=kind,
            cell=str(cell["id"]),
            experiment=str(cell["experiment"]),
            domain=domain,
            model=str(cell["model"]),
            hardware_class=str(cell["hardware_class"]),
            gpus=int(placement["gpus"] if gpus_override is None else gpus_override),
            config=str(cell["config"]),
            gpu_hours=hours,
            rate_source=rate_source,
            cost_source=source,
            params_b=view["params_b"],
            runs_root=runs_root,
            program="configs/program.yaml",
            stage=cell_stage(program, str(cell["id"])),
            **extra,
        )
        jobs.append(job)
        return job

    for cell in program["cells"]:
        cell_id = str(cell["id"])
        template = load_template(root, str(cell["domain"]))
        planned = _menu_and_planned(cell, template)
        early_on, early_steps, early_seed = _early_of(template, cell)
        out = f"{runs_root}/{cell_id}"
        audit_id = f"{cell_id}/audit"
        exploratory_cell = cell.get("experiment") == "X"
        add(
            cell,
            "audit",
            id=audit_id,
            out_dir=out,
            depends_on=[],
            **({"exploratory": True} if exploratory_cell else {}),
        )
        for arm, seeds in planned.items():
            for seed in seeds:
                add(
                    cell,
                    "train",
                    id=f"{cell_id}/train/{arm}/s{seed}",
                    arm=arm,
                    seed=int(seed),
                    out_dir=out,
                    depends_on=[audit_id],
                    **({"exploratory": True} if exploratory_cell else {}),
                )
            if early_on:
                add(
                    cell,
                    "early",
                    id=f"{cell_id}/early/{arm}/s{early_seed}",
                    arm=arm,
                    seed=int(early_seed),
                    out_dir=out,
                    depends_on=[audit_id],
                    early_steps_for_cost=early_steps,
                    **({"exploratory": True} if exploratory_cell else {}),
                )
        _add_variant_jobs(add, cell, out, audit_id)
    e1_cells = [cell for cell in program["cells"] if cell["experiment"] == "E1"]
    domains_for_model: dict[str, list[str]] = {}
    for cell in e1_cells:
        domains_for_model.setdefault(str(cell["model"]), [])
        if cell["domain"] not in domains_for_model[str(cell["model"])]:
            domains_for_model[str(cell["model"])].append(str(cell["domain"]))
    for cell in e1_cells:
        template = load_template(root, str(cell["domain"]))
        planned = _menu_and_planned(cell, template)
        others = [d for d in domains_for_model[str(cell["model"])] if d != cell["domain"]]
        for arm, seeds in planned.items():
            for seed in seeds:
                for target in others:
                    add(
                        cell,
                        "transfer",
                        id=f"{cell['id']}/transfer/{arm}/s{seed}/{target}",
                        arm=arm,
                        seed=int(seed),
                        target_domain=target,
                        out_dir=f"{runs_root}/{cell['id']}/transfer/{arm}__s{seed}",
                        depends_on=[f"{cell['id']}/train/{arm}/s{seed}"],
                        exploratory="X1",
                    )
    return jobs


def _speed_and_usd(program: Mapping[str, Any] | None) -> tuple[dict[str, float], dict[str, float]]:
    policy = (program or {}).get("hardware_policy") or {}
    speed_raw = policy.get("speed_vs_h100") or {}
    usd_raw = policy.get("usd_per_gpu_hour") or {}
    speed = {str(key): float(value) for key, value in speed_raw.items()}
    usd = {str(key): float(value) for key, value in usd_raw.items()}
    return speed, usd


def class_gpu_hours(h100_hours: float, hardware_class: str, speed: Mapping[str, float]) -> float:
    """Wall GPU-hours on ``hardware_class``: H100-equivalent hours / speed_vs_h100."""
    factor = float(speed[hardware_class])
    if factor <= 0:
        raise ValueError(f"speed_vs_h100 for {hardware_class} must be positive")
    return float(h100_hours) / factor


def parse_gpu_counts(text: str) -> dict[str, int]:
    """``rtx-a6000-48gb=4,rtx-5000-ada-32gb=4`` into a count per class."""
    counts: dict[str, int] = {}
    if not str(text).strip():
        return counts
    for part in str(text).split(","):
        name, sep, count = part.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"gpu count {part!r} must be hardware_class=N")
        counts[name.strip()] = int(count)
    return counts


def apply_budget(
    jobs: Sequence[Mapping[str, Any]],
    program: Mapping[str, Any],
    budget_usd: float,
) -> tuple[list[Mapping[str, Any]], list[str], bool]:
    """Drop cells in ``compute.cut_order`` until rented USD fits.

    Returns the kept jobs, the dropped cell ids, and whether the remainder
    fits. Lab classes have a zero USD rate, so dropping them does not reduce
    rented spend.
    """
    speed, usd = _speed_and_usd(program)

    def rented(rows: Sequence[Mapping[str, Any]]) -> float:
        total = 0.0
        for job in rows:
            hardware = str(job["hardware_class"])
            hours = class_gpu_hours(float(job["gpu_hours"]), hardware, speed)
            total += hours * float(usd.get(hardware, 0.0))
        return total

    kept: list[Mapping[str, Any]] = list(jobs)
    dropped: list[str] = []
    order = list(((program.get("compute") or {}).get("cut_order")) or [])
    for cell_id in order:
        if rented(kept) <= float(budget_usd):
            break
        nxt = [job for job in kept if str(job["cell"]) != str(cell_id)]
        if len(nxt) != len(kept):
            dropped.append(str(cell_id))
            kept = nxt
    return kept, dropped, rented(kept) <= float(budget_usd)


def _rate_source_label(rows: Sequence[Mapping[str, Any]]) -> str:
    sources = {str(job.get("rate_source") or "placeholder") for job in rows}
    if sources == {"measured"}:
        return "measured"
    if sources == {"placeholder"}:
        return "placeholder"
    return "mixed"


def format_budget(
    jobs: Sequence[Mapping[str, Any]],
    *,
    program: Mapping[str, Any] | None = None,
    by_cell: bool = False,
    by_stage: bool = False,
    budget_usd: float | None = None,
    gpus: Mapping[str, int] | str | None = None,
) -> str:
    """Class table of jobs, H100-equivalent hours, class GPU-hours, and USD.

    ``--by-cell`` adds one row per cell, including its stage. ``--by-stage``
    adds jobs, H100-equivalent hours, class GPU-hours, and USD for each stage
    and hardware class. Exploratory jobs are their own stage bucket.
    ``budget_usd`` drops cells in ``compute.cut_order`` until rented USD fits.
    ``gpus`` (a mapping or ``class=N,...`` string) adds wall-clock days per class.
    """
    shown: Sequence[Mapping[str, Any]] = jobs
    dropped: list[str] = []
    fits = True
    if budget_usd is not None:
        if program is None:
            raise ValueError("budget_usd requires the program")
        shown, dropped, fits = apply_budget(jobs, program, budget_usd)
    speed, usd = _speed_and_usd(program)
    order = list(HARDWARE_CLASSES)
    for job in shown:
        name = str(job["hardware_class"])
        if name not in order:
            order.append(name)
    lines = [
        f"{'hardware_class':<22} {'jobs':>6} {'h100_hours':>12} "
        f"{'class_gpu_hours':>16} {'usd':>10} {'rate_source':<12}"
    ]
    total_jobs = 0
    total_h100 = 0.0
    total_class = 0.0
    total_usd = 0.0
    per_class: dict[str, float] = {}
    for name in order:
        rows = [job for job in shown if job["hardware_class"] == name]
        if not rows:
            continue
        h100 = sum(float(job["gpu_hours"]) for job in rows)
        if name not in speed:
            raise ValueError(f"hardware_policy.speed_vs_h100 is missing {name}")
        klass = class_gpu_hours(h100, name, speed)
        dollars = klass * float(usd.get(name, 0.0))
        lines.append(
            f"{name:<22} {len(rows):>6} {h100:12.2f} {klass:16.2f} "
            f"{dollars:10.2f} {_rate_source_label(rows):<12}"
        )
        total_jobs += len(rows)
        total_h100 += h100
        total_class += klass
        total_usd += dollars
        per_class[name] = klass
    lines.append(
        f"{'TOTAL':<22} {total_jobs:>6} {total_h100:12.2f} {total_class:16.2f} {total_usd:10.2f}"
    )
    sources: dict[str, int] = {}
    for job in shown:
        key = str(job.get("cost_source", "unknown"))
        sources[key] = sources.get(key, 0) + 1
    source_text = ", ".join(f"{key}={count}" for key, count in sorted(sources.items()))
    lines.append("")
    lines.append(f"cost_source jobs: {source_text}")
    lines.append(f"scaling: {SCALING_RULE}")
    if budget_usd is not None:
        dropped_text = ", ".join(dropped) if dropped else "(none)"
        lines.append(f"budget_usd: {float(budget_usd):.2f}")
        lines.append(f"dropped: {dropped_text}")
        lines.append("budget_fit: yes" if fits else "budget_fit: no")
    if by_cell:
        lines.append("")
        lines.extend(_by_cell_lines(shown, program, speed, usd))
    if by_stage:
        lines.append("")
        lines.extend(_by_stage_lines(shown, program, speed, usd))
    if gpus:
        counts = parse_gpu_counts(gpus) if isinstance(gpus, str) else dict(gpus)
        lines.append("")
        lines.append(f"{'hardware_class':<22} {'gpus':>6} {'wall_days':>12}")
        for name, klass in per_class.items():
            count = int(counts.get(name, 0))
            if count <= 0:
                continue
            days = klass / (count * 24.0)
            lines.append(f"{name:<22} {count:>6} {days:12.2f}")
    return "\n".join(lines)


def _by_cell_lines(
    jobs: Sequence[Mapping[str, Any]],
    program: Mapping[str, Any] | None,
    speed: Mapping[str, float],
    usd: Mapping[str, float],
) -> list[str]:
    cut = list((((program or {}).get("compute") or {}).get("cut_order")) or [])
    position = {str(cell_id): index + 1 for index, cell_id in enumerate(cut)}
    order: list[str] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for job in jobs:
        cell_id = str(job["cell"])
        if cell_id not in grouped:
            order.append(cell_id)
            grouped[cell_id] = []
        grouped[cell_id].append(job)
    lines = [
        f"{'cell':<32} {'stage':<12} {'hardware_class':<22} {'jobs':>6} {'h100_hours':>12} "
        f"{'class_gpu_hours':>16} {'usd':>10} {'cut':>4}"
    ]
    for cell_id in order:
        rows = grouped[cell_id]
        hardware = str(rows[0]["hardware_class"])
        h100 = sum(float(job["gpu_hours"]) for job in rows)
        klass = class_gpu_hours(h100, hardware, speed)
        dollars = klass * float(usd.get(hardware, 0.0))
        cut_at = position.get(cell_id)
        cut_text = "" if cut_at is None else str(cut_at)
        stage = rows[0].get("stage")
        stage_text = "" if stage is None else str(stage)
        lines.append(
            f"{cell_id:<32} {stage_text:<12} {hardware:<22} {len(rows):>6} {h100:12.2f} "
            f"{klass:16.2f} {dollars:10.2f} {cut_text:>4}"
        )
    return lines


def _stage_bucket(job: Mapping[str, Any]) -> str:
    """Plan table bucket. Exploratory jobs are not charged to stage A or B."""
    if job.get("exploratory"):
        return "exploratory"
    stage = job.get("stage")
    if stage is None or stage == "":
        return "unassigned"
    return str(stage)


def _by_stage_lines(
    jobs: Sequence[Mapping[str, Any]],
    program: Mapping[str, Any] | None,
    speed: Mapping[str, float],
    usd: Mapping[str, float],
) -> list[str]:
    """Jobs, H100-equivalent hours, class GPU-hours, and USD by stage and class."""
    names: dict[str, str] = {}
    stages = (program or {}).get("stages") or {}
    if isinstance(stages, Mapping):
        for key, block in stages.items():
            if isinstance(block, Mapping) and block.get("name"):
                names[str(key)] = str(block["name"])
    order = ["A", "B", "exploratory", "unassigned"]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for job in jobs:
        bucket = _stage_bucket(job)
        if bucket not in order:
            order.append(bucket)
        grouped.setdefault(bucket, []).append(job)
    lines = [
        f"{'stage':<18} {'hardware_class':<22} {'jobs':>6} {'h100_hours':>12} "
        f"{'class_gpu_hours':>16} {'usd':>10}"
    ]
    for bucket in order:
        rows = grouped.get(bucket) or []
        if not rows:
            continue
        label = names.get(bucket)
        if label:
            lines.append(f"stage {bucket}: {label}")
        classes: list[str] = []
        by_class: dict[str, list[Mapping[str, Any]]] = {}
        for job in rows:
            hardware = str(job["hardware_class"])
            if hardware not in by_class:
                classes.append(hardware)
                by_class[hardware] = []
            by_class[hardware].append(job)
        total_jobs = 0
        total_h100 = 0.0
        total_class = 0.0
        total_usd = 0.0
        for hardware in classes:
            group = by_class[hardware]
            h100 = sum(float(job["gpu_hours"]) for job in group)
            if hardware not in speed:
                raise ValueError(f"hardware_policy.speed_vs_h100 is missing {hardware}")
            klass = class_gpu_hours(h100, hardware, speed)
            dollars = klass * float(usd.get(hardware, 0.0))
            lines.append(
                f"{bucket:<18} {hardware:<22} {len(group):>6} {h100:12.2f} "
                f"{klass:16.2f} {dollars:10.2f}"
            )
            total_jobs += len(group)
            total_h100 += h100
            total_class += klass
            total_usd += dollars
        lines.append(
            f"{(bucket + ' total'):<18} {'':<22} {total_jobs:>6} {total_h100:12.2f} "
            f"{total_class:16.2f} {total_usd:10.2f}"
        )
    return lines


def _cell_by_id(program: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    for cell in program["cells"]:
        if cell["id"] == cell_id:
            return dict(cell)
    raise KeyError(f"unknown cell {cell_id}")


def _target_cell(program: Mapping[str, Any], model: str, domain: str) -> dict[str, Any]:
    matches = [
        cell for cell in program["cells"] if cell["model"] == model and cell["domain"] == domain
    ]
    if not matches:
        raise KeyError(f"no cell for model {model} domain {domain}")
    e1 = [cell for cell in matches if cell["experiment"] == "E1"]
    return dict((e1 or matches)[0])


def transfer_eval(
    program: Mapping[str, Any],
    runs_root: str | Path,
    *,
    cell: str,
    arm: str,
    seed: int,
    targets: Sequence[str] | None = None,
    root: Path | None = None,
) -> list[Path]:
    """Score one finished run's final adapter on other domains' held-out pools.

    Writes ``runs/<cell>/transfer/<arm>__s<seed>/<target_domain>.json``.

    Each file is schema ``vfates.transfer_eval/1``:

    - ``schema``: ``vfates.transfer_eval/1``
    - ``source_cell``, ``arm``, ``seed``, ``target_domain``, ``target_cell``
    - ``adapter``: path of the final PEFT checkpoint (highest step)
    - ``step``: that checkpoint's step
    - ``n_prompts``, ``n_samples``
    - ``accuracy``: mean over prompts of per-prompt oracle accuracy
    - ``per_prompt``: ``{pid, accuracy, n}`` in problem order. ``accuracy`` is
      the mean oracle bit on that prompt. The analysis owner reads this list.
    """
    root = root or repo_root()
    runs = Path(runs_root)
    source = _cell_by_id(program, cell)
    config_path = root / str(source["config"])
    if not config_path.is_file():
        raise FileNotFoundError(f"rendered cell config is missing: {config_path}")
    config = read_yaml(config_path)
    run_dir = runs / cell / "runs" / f"{arm}__s{int(seed)}"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"finished run is missing: {run_dir}")

    from vfates.gen.factory import make_generator
    from vfates.train.checkpoint_eval import peft_checkpoints
    from vfates.train.evaluate import evaluate_policy

    checkpoints = peft_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no PEFT checkpoint in {run_dir}")
    step, adapter = checkpoints[-1]
    if targets is None:
        domains = []
        for other in program["cells"]:
            if (
                other["model"] == source["model"]
                and other["domain"] != source["domain"]
                and other["domain"] not in domains
            ):
                domains.append(str(other["domain"]))
    else:
        domains = [str(domain) for domain in targets]
    if not domains:
        raise ValueError(f"{cell} has no other domain to transfer onto")

    train = dict(config["study"]["train"])
    n_samples = int(train.get("eval_samples_per_prompt", 4))
    max_tokens = int(train.get("eval_max_tokens") or train["max_tokens"])
    temperature = float(train.get("eval_temperature", 1.0))
    spec = dict(config["study"].get("eval_generator") or {})
    backend = str(spec.pop("backend", "hf"))
    if backend not in ("mlx", "hf", "vllm"):
        raise ValueError(f"eval generator backend {backend!r} is not mlx, hf, or vllm")
    kwargs = dict(spec.pop("kwargs", {}))
    generator = make_generator(
        cast(Literal["mlx", "hf", "vllm"], backend), str(train["model_id"]), **kwargs
    )
    attach = getattr(generator, "set_adapter", None)
    if not callable(attach):
        raise AttributeError(f"{backend} generator has no set_adapter")
    attach(str(adapter))

    written: list[Path] = []
    for domain in domains:
        target = _target_cell(program, str(source["model"]), domain)
        problems = _heldout_problems(target, runs, root)
        result = evaluate_policy(
            generator,
            _task_for(target, runs, root, problems),
            problems,
            n_samples=n_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=int(seed),
        )
        per_prompt = [
            {
                "pid": row["pid"],
                "accuracy": float(row["oracle_acc"]),
                "n": int(row["n"]),
            }
            for row in result["per_prompt"]
        ]
        payload = {
            "schema": "vfates.transfer_eval/1",
            "source_cell": cell,
            "arm": arm,
            "seed": int(seed),
            "target_domain": domain,
            "target_cell": target["id"],
            "adapter": str(adapter),
            "step": int(step),
            "n_prompts": int(result["n_prompts"]),
            "n_samples": int(result["n_samples"]),
            "accuracy": None if result["oracle_acc"] is None else float(result["oracle_acc"]),
            "per_prompt": per_prompt,
        }
        dest = runs / cell / "transfer" / f"{arm}__s{int(seed)}" / f"{domain}.json"
        write_json(dest, payload)
        written.append(dest)
    return written


def _task_for(
    cell: Mapping[str, Any],
    runs: Path,
    root: Path,
    problems: list[Any],
) -> Any:
    """Task object for :func:`evaluate_policy`.

    When the held-out rows were loaded from an audit, the task still has to
    parse and grade them, so this always builds the task from the cell config.
    ``problems`` is unused except to keep the call site obvious; the audit
    loader and the task builder agree on the pool.
    """
    del problems
    from vfates.tasks.registry import build_task

    config_path = root / str(cell["config"])
    if not config_path.is_file():
        raise FileNotFoundError(
            f"target cell config is missing: {config_path}. Render the domain template first."
        )
    config = read_yaml(config_path)
    return build_task(config["audit"]["task"])


def _heldout_problems(cell: Mapping[str, Any], runs: Path, root: Path) -> list[Any]:
    audit_dir = runs / str(cell["id"]) / "audit"
    if (audit_dir / "problems.jsonl").is_file() and (audit_dir / "config.json").is_file():
        try:
            from vfates.audit import load_audit

            audit = load_audit(audit_dir)
            held = [problem for problem in audit.problems.values() if problem.split == "heldout"]
            if held:
                return held
        except (OSError, ValueError, KeyError):
            pass
    from vfates.tasks.registry import build_task

    config_path = root / str(cell["config"])
    config = read_yaml(config_path)
    task = build_task(config["audit"]["task"])
    pools = config["audit"]["pools"]
    return task.problems("heldout", int(pools["heldout"]), int(pools["seed"]))


def _cell_train_steps(cell: Mapping[str, Any], root: Path) -> int:
    """Planned confirmatory step count. Missing configs use the protocol default."""
    cfg = cell.get("config")
    study: Mapping[str, Any] = {}
    if isinstance(cfg, Mapping):
        raw = cfg.get("study")
        if isinstance(raw, Mapping):
            study = raw
    elif isinstance(cfg, str) and cfg:
        path = Path(cfg)
        if not path.is_file():
            path = root / cfg
        if path.is_file():
            loaded = read_yaml(path)
            if isinstance(loaded, Mapping):
                raw = loaded.get("study")
                if isinstance(raw, Mapping):
                    study = raw
    train = study.get("train")
    if isinstance(train, Mapping) and train.get("steps") is not None:
        return int(train["steps"])
    return 300


def _summary_complete(path: Path, steps: int) -> bool:
    """A train summary finished the planned step and was not preempted."""
    if not path.is_file():
        return False
    try:
        summary = read_json(path)
    except (OSError, ValueError):
        return False
    if not isinstance(summary, dict):
        return False
    if summary.get("status") in {"preempted", "interrupted", "failed"}:
        return False
    return summary.get("steps_completed") == steps


def missing_train_runs(
    jobs: Sequence[Mapping[str, Any]],
    cell_ids: Sequence[str],
    steps_of: Mapping[str, int],
) -> list[str]:
    """Confirmatory train job ids whose ``summary.json`` is not complete.

    Early baselines, variants, transfer evals, and audits are not this gate.
    ``steps_of`` maps a cell id to its planned step count.
    """
    wanted = {str(cell_id) for cell_id in cell_ids}
    missing: list[str] = []
    for job in jobs:
        if job.get("kind") != "train" or job.get("exploratory"):
            continue
        cell_id = str(job.get("cell"))
        if cell_id not in wanted:
            continue
        steps = int(steps_of.get(cell_id, 300))
        run_dir = Path(str(job["out_dir"])) / "runs" / f"{job['arm']}__s{int(job['seed'])}"
        path = run_dir / "summary.json"
        if not _summary_complete(path, steps):
            missing.append(str(job["id"]))
    return missing


def _write_program_report(runs: Path, report: Mapping[str, Any], text: str) -> dict[str, Any]:
    body = text if str(text).endswith("\n") else str(text) + "\n"
    payload = dict(report)
    write_json(runs / "program_report.json", payload)
    (runs / "program_report.md").write_text(body, encoding="utf-8")
    return payload


def analyze_program(
    program: Mapping[str, Any],
    runs_root: str | Path,
    *,
    budget_usd: float | None = None,
    n_boot: int = 10_000,
    n_perm: int = 100_000,
    seed: int = 0,
    root: Path | None = None,
) -> dict[str, Any]:
    """Write ``runs/program_report.json`` and ``runs/program_report.md``.

    A program with ``stages`` analyses each stage only when every confirmatory
    train job of that stage, after ``budget_usd`` cuts, has a completed
    summary. A stage that is short is ``pending`` and lists the missing job
    ids. Nothing of that stage is computed from a subset of its runs.
    """
    try:
        from vfates.analysis.pooled import (
            pooled_report,
            render_pooled_markdown,
            staged_report,
        )
    except ImportError as exc:
        raise ImportError(
            "vfates.analysis.pooled is not available "
            "(expected pooled_report, staged_report, and render_pooled_markdown)"
        ) from exc
    runs = Path(runs_root)
    stages = program.get("stages")
    if not isinstance(stages, Mapping) or not stages:
        report = pooled_report(program, runs, n_boot=n_boot, n_perm=n_perm, seed=seed)
        return _write_program_report(runs, report, render_pooled_markdown(report))
    repo = root or repo_root()
    jobs: Sequence[Mapping[str, Any]] = plan_jobs(program, root=repo, runs_root=str(runs))
    dropped: list[str] = []
    if budget_usd is not None:
        jobs, dropped, _fits = apply_budget(jobs, program, budget_usd)
    dropped_set = set(dropped)
    steps_of: dict[str, int] = {}
    for cell in program.get("cells") or []:
        if isinstance(cell, Mapping) and cell.get("id"):
            steps_of[str(cell["id"])] = _cell_train_steps(cell, repo)
    ready: dict[str, list[str]] = {}
    pending: dict[str, dict[str, Any]] = {}
    for key in ("A", "B"):
        block = stages.get(key)
        if not isinstance(block, Mapping):
            continue
        listed = [str(item) for item in (block.get("cells") or [])]
        remaining = [cell_id for cell_id in listed if cell_id not in dropped_set]
        stage_dropped = [cell_id for cell_id in listed if cell_id in dropped_set]
        missing = missing_train_runs(jobs, remaining, steps_of)
        if missing:
            pending[key] = {
                "name": block.get("name"),
                "status": "pending",
                "missing": missing,
                "dropped": stage_dropped,
                "rule": (
                    "This stage is analysed only when every confirmatory train "
                    "job of its remaining cells has a completed summary.json. "
                    "Budget cuts remove cells before that check. Early baselines, "
                    "variants, transfer evals, and audits are not the gate."
                ),
            }
        else:
            ready[key] = remaining
    ready_dropped: dict[str, list[str]] = {}
    for key in ("A", "B"):
        block = stages.get(key)
        if not isinstance(block, Mapping) or key not in ready:
            continue
        listed = [str(item) for item in (block.get("cells") or [])]
        ready_dropped[key] = [cell_id for cell_id in listed if cell_id in dropped_set]
    report = staged_report(
        program,
        runs,
        ready=ready,
        pending=pending,
        dropped=ready_dropped,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    return _write_program_report(runs, report, render_pooled_markdown(report))
