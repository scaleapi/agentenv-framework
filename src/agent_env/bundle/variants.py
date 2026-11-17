"""Run-only variants for ``agent-env run --set envs/NAME.KEY=VALUE``: the bundle as parsed, with env keys set for one
run. Each env changed, each env and task that names one, and each eval that names such a task is written under a
derived id, so a variant is written and composed beside the bundle's own. Entries keep their names, so every reference to
one by name reaches its variant; the bundle's files stay as written."""
from __future__ import annotations

import copy
import hashlib
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from agent_env.entity.refs import EntityKind, EntityRef, RefSite, ref_sites
from agent_env.task_step.registry import get_task_step_registry

from .parse import Bundle, BundleEntry, BundleError, BundleKind, segment_problem
from .resolve import entity_class, sync_context_paths

_TARGET = f"{BundleKind.ENV.value}/"
_FORM = "envs/NAME.KEY=VALUE, such as envs/desktop.driver=cua-driver"
#: The keys that say what an entry is rather than how it's set, which a run can't change.
_FIXED = ("type", "id")


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
class Varied:
    """A bundle with a variant's settings: ``bundle``, and ``renamed``, each entry it changes, by kind and id, to its
    variant's id."""

    variant: Variant
    bundle: Bundle
    renamed: Mapping[tuple[BundleKind, str], str]

    def changes(self, kind: BundleKind, entry_id: str) -> bool:
        """Whether ``entry_id`` is the variant's id of an entry of ``kind`` it changes."""
        return entry_id in {new for (of, _), new in self.renamed.items() if of is kind}

    def selected(self, names: Sequence[str], kind: BundleKind) -> tuple[str, ...]:
        """``names``, a run's selections of ``kind`` by name or id, each id of an entry it changes as its variant's."""
        return tuple(self.renamed.get((kind, name), name) for name in names)


@dataclass(frozen=True)
class _Setting:
    env: str  # the env it sets, by name
    path: tuple[str, ...]
    value: Any


def with_variant(bundle: Bundle, variant: Variant) -> Varied:
    """``bundle`` with ``variant``'s settings in its envs' tomls, and each entry that changes under an id ending in
    ``(<label>)``. Raises BundleError for a setting that isn't ``envs/NAME.KEY=VALUE`` or names no env of the bundle; the
    env's type checks the key and value as it checks an env.toml's."""
    by_name = {entry.name: entry for entry in bundle.entries if entry.kind is BundleKind.ENV}
    configs = {}  # each env's toml with its settings, by kind and id, as an id is unique within its kind alone
    for setting in _parsed(variant.settings, set(by_name)):
        key = (BundleKind.ENV, by_name[setting.env].id)
        configs[key] = _set(configs.get(key, by_name[setting.env].config), setting.path, setting.value)
    renamed = _renamed_ids(bundle, {entry_id for _, entry_id in configs}, variant.label)
    envs, tasks = ({entry_id: new for (kind, entry_id), new in renamed.items() if kind is of}
                   for of in (BundleKind.ENV, BundleKind.TASK))
    renaming = {EntityKind.ENV: envs, EntityKind.TASK: tasks}
    entries = []
    for entry in bundle.entries:
        new_id = renamed.get((entry.kind, entry.id))
        if new_id is None:
            entries.append(entry)
        else:
            config = configs.get((entry.kind, entry.id), entry.config)
            entries.append(_renamed(entry, new_id, _repointed(entry, config, renaming)))
    return Varied(variant, replace(bundle, entries=tuple(entries)), renamed)


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
        if path[0] in _FIXED:
            problems.append(f"--set {text}: an env's {path[0]} says what it is, so a run can't set it")
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


def _renamed_ids(bundle: Bundle, changed: set[str], label: str) -> dict[tuple[BundleKind, str], str]:
    """Each entry that changes, by kind and id, to its variant's id: the envs ``changed``, each env and task naming one
    that changes, by name or id, and each eval naming such a task. An id is unique within its kind alone."""
    by_name = {(entry.kind, entry.name): entry.id for entry in bundle.entries}
    envs = set(changed)
    while more := {entry.id for entry in bundle.entries if entry.kind is BundleKind.ENV and entry.id not in envs
                   and _names(entry, EntityKind.ENV, envs, by_name)}:
        envs |= more
    tasks = {entry.id for entry in bundle.entries if entry.kind is BundleKind.TASK
             and _names(entry, EntityKind.ENV, envs, by_name)}
    evals = {entry.id for entry in bundle.entries if entry.kind is BundleKind.EVAL
             and _names(entry, EntityKind.TASK, tasks, by_name)}
    return {(kind, entry_id): f"{entry_id}({label})"
            for kind, ids in ((BundleKind.ENV, envs), (BundleKind.TASK, tasks), (BundleKind.EVAL, evals))
            for entry_id in ids}


def _names(entry: BundleEntry, kind: EntityKind, ids: set[str], by_name: dict) -> bool:
    """Whether ``entry`` names one of ``ids``, entries of ``kind``, by its name or id."""
    of = {EntityKind.ENV: BundleKind.ENV, EntityKind.TASK: BundleKind.TASK}[kind]
    return any(by_name.get((of, site.value), site.value) in ids for site in _sites(entry, entry.config) if site.ref.kind is kind)


def _sites(entry: BundleEntry, config: Any) -> Iterator[RefSite]:
    """Each entity ``config``, ``entry``'s steps or toml, names by a name or id, as a site it can be rewritten at, as the
    resolver reads them."""
    if entry.kind is BundleKind.TASK:
        sites = (site for step in config for site in ref_sites(_step_refs(step), step))
    else:
        sites = ref_sites(getattr(entity_class(entry), "toml_refs", ()), config, inline_pins=True)
    return (site for site in sites if isinstance(site.value, str))


def _step_refs(step: dict) -> tuple[EntityRef, ...]:
    cls = get_task_step_registry().get(step["type"]) if isinstance(step.get("type"), str) else None
    return (cls.entity_refs or ()) if cls is not None else ()


def _renamed(entry: BundleEntry, new_id: str, config: Any) -> BundleEntry:
    """``entry`` under ``new_id`` with ``config``, whose declared id, when it has one, is that id too."""
    if isinstance(config, dict) and "id" in config:
        config = {**config, "id": new_id}
    return replace(entry, id=new_id, config=config)


def _repointed(entry: BundleEntry, config: Any, renaming: dict[EntityKind, dict[str, str]]) -> Any:
    """``config``, ``entry``'s steps or toml, with each env or task it names by an id that changes named by its variant's
    id."""
    written, config = config, copy.deepcopy(config)
    for site in _sites(entry, config):
        if site.value in renaming.get(site.ref.kind, {}):
            site.rewrite(renaming[site.ref.kind][site.value], site.version)
    if entry.kind is BundleKind.TASK:
        sync_context_paths(written, config)
    return config
