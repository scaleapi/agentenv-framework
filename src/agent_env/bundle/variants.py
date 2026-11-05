"""Run-only variants for ``agent-env run --set KEY=VALUE``: the bundle as parsed, with env keys set for one run. Each env
changed, each env and task that names one, and each eval that names such a task is written under a derived id, so a
variant is written and composed beside the bundle's own. Entries keep their names, so every reference to one by name
reaches its variant; the bundle's files stay as written."""
from __future__ import annotations

import copy
import hashlib
import tomllib
from dataclasses import dataclass, replace
from typing import Any

from agent_env.entity.refs import EntityKind, ref_sites
from agent_env.env.registry import get_env_registry
from agent_env.task_step.registry import get_task_step_registry

from .parse import Bundle, BundleEntry, BundleError, BundleKind, segment_problem

_TARGET = f"{BundleKind.ENV.value}/"


@dataclass(frozen=True)
class Variant:
    """The ``--set`` settings of a run, as typed: ``KEY=VALUE`` for every env the bundle's tasks deploy whose type takes
    KEY, or ``envs/NAME.KEY=VALUE`` for one env; KEY may be dotted into a table, VALUE is TOML or a bare word."""

    settings: tuple[str, ...]

    @property
    def label(self) -> str:
        """The settings as ids end in them: ``driver cua-driver, display 3f2a9c1e``, a value an id can't hold hashed."""
        return ", ".join(f"{key.removeprefix(_TARGET)} {_shown(raw)}"
                         for key, raw in (setting.partition("=")[::2] for setting in self.settings))

    @property
    def typed(self) -> str:
        """The settings as the command line gives them."""
        return " ".join(f"--set {setting}" for setting in self.settings)


@dataclass(frozen=True)
class _Setting:
    text: str
    env: str | None  # the env it names, by name; None for every env deployed that takes path[0]
    path: tuple[str, ...]
    value: Any


def with_variant(bundle: Bundle, variant: Variant) -> Bundle:
    """``bundle`` with ``variant``'s settings in its envs' tomls, and each entry that changes under an id ending in
    ``(<label>)``. Raises BundleError for a setting that isn't ``KEY=VALUE``, names no env of the bundle, or that no env
    the bundle's tasks deploy takes."""
    envs = [entry for entry in bundle.entries if entry.kind is BundleKind.ENV]
    settings = _parsed(variant.settings, {entry.name for entry in envs})
    deployed = _deployed(bundle)
    configs, problems = {}, []
    for setting in settings:
        hit = [entry for entry in envs if setting.env == entry.name or setting.env is None
               and deployed & {entry.name, entry.id} and setting.path[0] in _keys(entry)]
        if not hit:
            problems.append(_untaken(setting, [entry for entry in envs if deployed & {entry.name, entry.id}]))
        for entry in hit:
            configs[entry.id] = _set(configs.get(entry.id, entry.config), setting.path, setting.value)
    if problems:
        raise BundleError(problems)
    renamed = _renamed_ids(bundle, set(configs), variant.label)
    entries = []
    for entry in bundle.entries:
        if entry.id not in renamed:
            entries.append(entry)
        elif entry.kind is BundleKind.ENV:
            entries.append(_renamed(entry, renamed, _repointed_toml(entry, configs.get(entry.id, entry.config), renamed)))
        elif entry.kind is BundleKind.TASK:
            entries.append(_renamed(entry, renamed, _repointed_steps(entry.config, renamed)))
        else:
            entries.append(_renamed(entry, renamed, _eval(entry.config, renamed)))
    return replace(bundle, entries=tuple(entries))


def _parsed(texts: tuple[str, ...], names: set[str]) -> list[_Setting]:
    settings, problems = [], []
    for text in texts:
        key, equals, raw = text.partition("=")
        env = None
        if key.startswith(_TARGET):
            env = max((name for name in names if key.startswith(f"{_TARGET}{name}.")), key=len, default=None)
            if env is None:
                problems.append(f"--set {text}: {key.partition('.')[0]} names no env of this bundle")
                continue
            key = key.removeprefix(f"{_TARGET}{env}.")
        path = tuple(key.split("."))
        if not equals or not all(path):
            problems.append(f"--set {text}: expected KEY=VALUE, such as driver=cua-driver")
            continue
        settings.append(_Setting(text, env, path, _value(raw)))
    if problems:
        raise BundleError(problems)
    return settings


def _value(raw: str) -> Any:
    """``raw`` as a TOML value, or as the string it is when it isn't one, so ``cua-driver`` needs no quotes."""
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def _shown(raw: str) -> str:
    return raw if raw and segment_problem(raw) is None else hashlib.sha256(raw.encode()).hexdigest()[:8]


def _deployed(bundle: Bundle) -> set[str]:
    """The envs the bundle's tasks deploy, as their steps name them."""
    return {step["env_id"] for entry in bundle.entries if entry.kind is BundleKind.TASK for step in entry.config
            if step.get("type") == "deploy_env" and isinstance(step.get("env_id"), str)}


def _keys(entry: BundleEntry) -> dict[str, type]:
    cls = get_env_registry().get(entry.type)
    return getattr(cls, "toml_keys", None) or {}


def _untaken(setting: _Setting, deployed: list[BundleEntry]) -> str:
    takes = {entry.type: sorted(_keys(entry)) for entry in deployed if _keys(entry)}
    named = "; ".join(f"a {kind} env takes {', '.join(keys)}" for kind, keys in sorted(takes.items()))
    return f"--set {setting.text}: no env this bundle's tasks deploy takes {setting.path[0]}" + (f"; {named}" if named else "")


def _set(config: dict, path: tuple[str, ...], value: Any) -> dict:
    config = copy.deepcopy(config)
    table = config
    for key in path[:-1]:
        if not isinstance(table.get(key), dict):
            table[key] = {}
        table = table[key]
    table[path[-1]] = value
    return config


def _renamed_ids(bundle: Bundle, changed: set[str], label: str) -> dict[str, str]:
    """Each id that changes, to its variant's: the envs set, each env and task naming one that changes, by name or id,
    and each eval naming such a task."""
    by_name = {(entry.kind, entry.name): entry.id for entry in bundle.entries}
    changing = set(changed)
    while True:
        more = {entry.id for entry in bundle.entries if entry.id not in changing and entry.kind in (
            BundleKind.ENV, BundleKind.TASK) and _names_env(entry, changing, by_name)}
        if not more:
            break
        changing |= more
    tasks = {entry.id for entry in bundle.entries if entry.kind is BundleKind.TASK and entry.id in changing}
    changing |= {entry.id for entry in bundle.entries if entry.kind is BundleKind.EVAL and any(
        by_name.get((BundleKind.TASK, task), task) in tasks for task in entry.config.get("tasks") or []
        if isinstance(task, str))}
    return {entry_id: f"{entry_id}({label})" for entry_id in changing}


def _names_env(entry: BundleEntry, changing: set[str], by_name: dict) -> bool:
    return any(by_name.get((BundleKind.ENV, site.value), site.value) in changing for site in _env_sites(entry))


def _env_sites(entry: BundleEntry):
    if entry.kind is BundleKind.TASK:
        registry = get_task_step_registry()
        for step in entry.config:
            cls = registry.get(step.get("type")) if isinstance(step.get("type"), str) else None
            yield from (site for site in ref_sites(cls.entity_refs or () if cls is not None else (), step)
                        if site.ref.kind is EntityKind.ENV and isinstance(site.value, str))
    else:
        cls = get_env_registry().get(entry.type)
        yield from (site for site in ref_sites(getattr(cls, "toml_refs", ()), entry.config, inline_pins=True)
                    if site.ref.kind is EntityKind.ENV and isinstance(site.value, str))


def _renamed(entry: BundleEntry, renamed: dict[str, str], config: Any) -> BundleEntry:
    """``entry`` under its variant's id with ``config``, whose declared id, when it has one, is that id too."""
    if isinstance(config, dict) and "id" in config:
        config = {**config, "id": renamed[entry.id]}
    return replace(entry, id=renamed[entry.id], config=config)


def _repointed_toml(entry: BundleEntry, config: dict, renamed: dict[str, str]) -> dict:
    """An env's toml with each env it names by an id that changes named by its variant's id."""
    config = copy.deepcopy(config)
    cls = get_env_registry().get(entry.type)
    for site in ref_sites(getattr(cls, "toml_refs", ()), config, inline_pins=True):
        if site.ref.kind is EntityKind.ENV and site.value in renamed:
            site.rewrite(renamed[site.value], site.version)
    return config


def _repointed_steps(steps: list[dict], renamed: dict[str, str]) -> list[dict]:
    """A task's steps with each env a step names by an id that changes named by its variant's id."""
    steps = copy.deepcopy(steps)
    registry = get_task_step_registry()
    for step in steps:
        cls = registry.get(step.get("type")) if isinstance(step.get("type"), str) else None
        for site in ref_sites(cls.entity_refs or () if cls is not None else (), step):
            if site.ref.kind is EntityKind.ENV and site.value in renamed:
                site.rewrite(renamed[site.value], site.version)
    return steps


def _eval(config: dict, renamed: dict[str, str]) -> dict:
    """An eval's toml with each task it names by an id that changes named by its variant's id."""
    return {**config, "tasks": [renamed.get(task, task) if isinstance(task, str) else task
                                for task in config.get("tasks") or []]}
