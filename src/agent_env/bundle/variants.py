"""Run-only variants for ``agent-env run --set envs/NAME.KEY=VALUE``: the bundle as parsed, with env keys set for one
run. Each env changed, each env and task that names one, and each eval that names such a task is written under a
derived id, so a variant is written and composed beside the bundle's own. Entries keep their names, so every reference to
one by name reaches its variant; the bundle's files stay as written."""
from __future__ import annotations

import copy
import hashlib
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from agent_env.entity.refs import EntityKind, ref_sites
from agent_env.env.registry import get_env_registry
from agent_env.task_step.registry import get_task_step_registry

from .parse import Bundle, BundleEntry, BundleError, BundleKind, segment_problem

_TARGET = f"{BundleKind.ENV.value}/"
_FORM = "envs/NAME.KEY=VALUE, such as envs/desktop.driver=cua-driver"


@dataclass(frozen=True)
class Variant:
    """The ``--set`` settings of a run, as typed: ``envs/NAME.KEY=VALUE``, KEY of the env in folder ``envs/NAME``,
    dotted into a table, set to VALUE, TOML or a bare word."""

    settings: tuple[str, ...]

    @property
    def label(self) -> str:
        """The settings as ids end in them: ``desktop.driver cua-driver, desktop.display 3f2a9c1e``, a value an id can't
        hold hashed."""
        return ", ".join(f"{key.removeprefix(_TARGET)} {_shown(raw)}"
                         for key, raw in (setting.partition("=")[::2] for setting in self.settings))

    @property
    def typed(self) -> str:
        """The settings as the command line gives them."""
        return " ".join(f"--set {setting}" for setting in self.settings)


@dataclass(frozen=True)
class _Setting:
    env: str  # the env it sets, by name
    path: tuple[str, ...]
    value: Any


def with_variant(bundle: Bundle, variant: Variant) -> Bundle:
    """``bundle`` with ``variant``'s settings in its envs' tomls, and each entry that changes under an id ending in
    ``(<label>)``. Raises BundleError for a setting that isn't ``envs/NAME.KEY=VALUE`` or names no env of the bundle; the
    env's type checks the key and value as it checks an env.toml's."""
    by_name = {entry.name: entry for entry in bundle.entries if entry.kind is BundleKind.ENV}
    configs = {}
    for setting in _parsed(variant.settings, set(by_name)):
        entry = by_name[setting.env]
        configs[entry.id] = _set(configs.get(entry.id, entry.config), setting.path, setting.value)
    renamed = _renamed_ids(bundle, set(configs), variant.label)
    envs, tasks = ({entry_id: new for (kind, entry_id), new in renamed.items() if kind is of}
                   for of in (BundleKind.ENV, BundleKind.TASK))
    entries = []
    for entry in bundle.entries:
        new_id = renamed.get((entry.kind, entry.id))
        if new_id is None:
            entries.append(entry)
        elif entry.kind is BundleKind.ENV:
            entries.append(_renamed(entry, new_id, _repointed_toml(entry, configs.get(entry.id, entry.config), envs)))
        elif entry.kind is BundleKind.TASK:
            entries.append(_renamed(entry, new_id, _repointed_steps(entry.config, envs)))
        else:
            entries.append(_renamed(entry, new_id, _eval(entry.config, tasks)))
    return replace(bundle, entries=tuple(entries))


def selected(before: Bundle, after: Bundle, names: Sequence[str], kind: BundleKind) -> tuple[str, ...]:
    """``names``, a run's selections of ``kind`` by name or id, with the id of each entry ``after`` renames as its new
    id."""
    ids = {old.id: new.id for old, new in zip(before.entries, after.entries) if old.kind is kind and old.id != new.id}
    return tuple(ids.get(name, name) for name in names)


def _parsed(texts: tuple[str, ...], names: set[str]) -> list[_Setting]:
    settings, problems = [], []
    for text in texts:
        key, equals, raw = text.partition("=")
        if not equals or not key.startswith(_TARGET) or "." not in key:
            problems.append(f"--set {text}: expected {_FORM}")
            continue
        env = max((name for name in names if key.startswith(f"{_TARGET}{name}.")), key=len, default=None)
        if env is None:
            problems.append(f"--set {text}: {key.partition('.')[0]} names no env of this bundle")
            continue
        path = tuple(key.removeprefix(f"{_TARGET}{env}.").split("."))
        if not all(path):
            problems.append(f"--set {text}: expected {_FORM}")
            continue
        settings.append(_Setting(env, path, _value(raw)))
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
    """Each entry that changes, by kind and id, to its variant's id: the envs ``changed``, each env and task naming one
    that changes, by name or id, and each eval naming such a task. An id is unique within its kind alone."""
    by_name = {(entry.kind, entry.name): entry.id for entry in bundle.entries}
    envs = set(changed)
    while more := {entry.id for entry in bundle.entries if entry.kind is BundleKind.ENV and entry.id not in envs
                   and _names_env(entry, envs, by_name)}:
        envs |= more
    tasks = {entry.id for entry in bundle.entries if entry.kind is BundleKind.TASK and _names_env(entry, envs, by_name)}
    evals = {entry.id for entry in bundle.entries if entry.kind is BundleKind.EVAL and any(
        by_name.get((BundleKind.TASK, task), task) in tasks for task in entry.config.get("tasks") or []
        if isinstance(task, str))}
    return {(kind, entry_id): f"{entry_id}({label})"
            for kind, ids in ((BundleKind.ENV, envs), (BundleKind.TASK, tasks), (BundleKind.EVAL, evals))
            for entry_id in ids}


def _names_env(entry: BundleEntry, envs: set[str], by_name: dict) -> bool:
    return any(by_name.get((BundleKind.ENV, site.value), site.value) in envs for site in _env_sites(entry))


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


def _renamed(entry: BundleEntry, new_id: str, config: Any) -> BundleEntry:
    """``entry`` under ``new_id`` with ``config``, whose declared id, when it has one, is that id too."""
    if isinstance(config, dict) and "id" in config:
        config = {**config, "id": new_id}
    return replace(entry, id=new_id, config=config)


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
