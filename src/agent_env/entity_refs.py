"""Which fields of a step dict, or keys of an entity's toml, hold env, A2A agent, artifact or task ids.

A type declares them on its class so a tool that rewrites ids before a run, such as a bundle
resolver, finds every one without guessing from field names. Nothing at run time reads the
declarations. Step types declare ``entity_refs``; env, agent, artifact and eval types declare the keys
of their toml as ``toml_refs``. For example::

    class LoadArtifactTaskStep(TaskStep):
        entity_refs = (
            EntityRef.env("env_id"),
            EntityRef.artifact("artifact_id", version_field="artifact_version"),
            EntityRef.artifact("artifacts[].id", version_field="version"),
        )

A toml pins a ref inline, with a table keyed by the ref's kind, which also lets a list element be
pinned: ``mcp_server_envs = ["tickets", { env = "crm", version = 2 }]``; ``ref_sites`` reads that
form when asked to.

Fields that name something else, such as another step's id, are left out. A step type's
declarations are not inherited, so a subclass declares its own; ``toml_refs`` are, as ``from_toml``
is.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import KW_ONLY, dataclass
from enum import StrEnum
from typing import Any, NamedTuple

# Yields (path parts relative to ``value``, dict) for every dict under ``value`` that holds the ref's key.
Walker = Callable[[Any], Iterator[tuple[tuple[str, ...], dict]]]

_PATH = re.compile(r"[A-Za-z_]\w*(\[\])?(\.[A-Za-z_]\w*(\[\])?)*")


class EntityKind(StrEnum):
    ENV = "env"
    AGENT = "agent"
    ARTIFACT = "artifact"
    TASK = "task"


class RefRole(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True)
class EntityRef:
    """One ref-holding field. ``path`` is dotted keys into the step dict or toml, ``[]`` meaning each
    element of a list (``env_ids[]``, ``artifacts[].id``); ``version_field`` is the key beside the
    ref, in the same dict, that pins its version. ``walk``, for a shape a path cannot spell, maps
    the value at the path's parent to the dicts that hold the last key, each with its path parts.
    Declarations build one with ``EntityRef.env``, ``EntityRef.agent`` or ``EntityRef.artifact``,
    passing the keyword fields."""

    path: str
    kind: EntityKind
    _: KW_ONLY
    version_field: str | None = None
    role: RefRole = RefRole.INPUT
    artifact_type: str | None = None
    env_type: str | None = None
    walk: Walker | None = None

    @classmethod
    def env(cls, path: str, **options: Any) -> EntityRef:
        return cls(path, EntityKind.ENV, **options)

    @classmethod
    def agent(cls, path: str, **options: Any) -> EntityRef:
        return cls(path, EntityKind.AGENT, **options)

    @classmethod
    def artifact(cls, path: str, **options: Any) -> EntityRef:
        return cls(path, EntityKind.ARTIFACT, **options)

    @classmethod
    def task(cls, path: str, **options: Any) -> EntityRef:
        return cls(path, EntityKind.TASK, **options)

    def __post_init__(self) -> None:
        if not _PATH.fullmatch(self.path):
            raise ValueError(f"entity ref path {self.path!r} is not dotted keys with optional [] suffixes")
        if not isinstance(self.kind, EntityKind) or not isinstance(self.role, RefRole):
            raise ValueError(
                f"entity ref {self.path!r}: kind {self.kind!r} / role {self.role!r} "
                "must be EntityKind / RefRole members"
            )
        if self.version_field is not None and self.path.endswith("[]"):
            raise ValueError(f"entity ref {self.path!r}: a list element has no sibling key to hold a version")
        if self.artifact_type is not None and self.kind is not EntityKind.ARTIFACT:
            raise ValueError(f"entity ref {self.path!r}: artifact_type is only for EntityKind.ARTIFACT")
        if self.env_type is not None and self.kind is not EntityKind.ENV:
            raise ValueError(f"entity ref {self.path!r}: env_type is only for EntityKind.ENV")

    @property
    def field(self) -> str:
        """The top-level step field the path starts at."""
        return self.path.split(".", 1)[0].removesuffix("[]")


class RefSite(NamedTuple):
    """One ref found in a step dict or toml: ``owner[key]`` holds ``value``; ``path`` is concrete
    (``artifacts[1].id``, ``mcp_server_envs[1].env``). ``version`` is its pin, None when unset;
    ``version_key`` is the key of ``owner`` that holds a pin, None when the ref can't be pinned."""

    path: str
    ref: EntityRef
    value: Any
    version: Any
    owner: dict | list
    key: str | int
    version_key: str | None = None

    def rewrite(self, value: Any, version: int | None = None) -> None:
        """Replace the ref where it was found and, when ``version`` is given, pin it: beside the ref
        in a step dict, inside its table in a toml."""
        if version is not None and self.version_key is None:
            raise ValueError(f"entity ref {self.path!r} has no version field to pin")
        self.owner[self.key] = value
        if version is not None:
            self.owner[self.version_key] = version


def parse_toml_ref(kind: EntityKind, value: Any) -> tuple[str, int | None]:
    """The id and version a toml ref names: an id, or ``{ <kind> = "<id>", version = <n> }`` with the
    version optional. Raises ValueError naming the expected form."""
    table = isinstance(value, dict)
    entity_id, version = (value.get(kind.value), value.get("version")) if table else (value, None)
    known_keys = not table or value.keys() <= {kind.value, "version"}
    positive = version is None or (isinstance(version, int) and not isinstance(version, bool) and version >= 1)
    if not (known_keys and isinstance(entity_id, str) and entity_id and positive):
        raise ValueError(f'expected an id or {{ {kind.value} = "<id>", version = <n> }}, the version optional, '
                         f"not {value!r}")
    return entity_id, version


def ref_sites(refs: Iterable[EntityRef], data: dict, *, inline_pins: bool = False) -> Iterator[RefSite]:
    """Every ref ``refs`` declares that is present, and not None, in ``data``. With ``inline_pins``,
    for a toml, a table holding the ref's kind is a pinned ref and its site points inside the table;
    any other value is reported as it is."""
    for ref in refs:
        *parents, leaf = ref.path.split(".")
        for trail, node in _descend(data, parents, ()):
            for rel, holder in ref.walk(node) if ref.walk else [((), node)]:
                if isinstance(holder, dict):
                    yield from _leaf(ref, holder, leaf, (*trail, *rel), inline_pins)


def _descend(node: Any, segments: list[str], trail: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Any]]:
    if not segments:
        yield trail, node
        return
    key, each = segments[0].removesuffix("[]"), segments[0].endswith("[]")
    child = node.get(key) if isinstance(node, dict) else None
    if child is None:
        return
    if not each:
        yield from _descend(child, segments[1:], (*trail, key))
    elif isinstance(child, list):
        for i, item in enumerate(child):
            yield from _descend(item, segments[1:], (*trail, f"{key}[{i}]"))


def _leaf(ref: EntityRef, holder: dict, leaf: str, trail: tuple[str, ...], inline_pins: bool) -> Iterator[RefSite]:
    key = leaf.removesuffix("[]")
    value = holder.get(key)
    if value is None:
        return
    path = ".".join((*trail, key))
    if not leaf.endswith("[]"):
        yield _site(ref, path, holder, key, ref.version_field, inline_pins)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if item is not None:
                yield _site(ref, f"{path}[{i}]", value, i, None, inline_pins)


def _site(ref: EntityRef, path: str, owner: dict | list, key: str | int, version_key: str | None,
          inline_pins: bool) -> RefSite:
    value = owner[key]
    if inline_pins and isinstance(value, dict) and value.get(ref.kind.value) is not None:
        return RefSite(f"{path}.{ref.kind.value}", ref, value[ref.kind.value], value.get("version"),
                       value, ref.kind.value, "version")
    version = owner.get(version_key) if version_key else None
    return RefSite(path, ref, value, version, owner, key, version_key)
