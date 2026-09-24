"""Which fields of a step dict hold env, A2A agent or artifact ids.

A step type declares them on its class so a tool that rewrites ids before a run, such as a bundle
resolver, finds every one without guessing from field names. Nothing at run time reads the
declarations. For example::

    class LoadArtifactTaskStep(TaskStep):
        entity_refs = (
            EntityRef.env("env_id"),
            EntityRef.artifact("artifact_id", version_field="artifact_version"),
            EntityRef.artifact("artifacts[].id", version_field="version"),
        )

Fields that name something else, such as another step's id, are left out. Declarations are not
inherited, so a subclass declares its own.
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


class RefRole(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True)
class EntityRef:
    """One ref-holding field. ``path`` is dotted keys into the step dict, ``[]`` meaning each
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

    @property
    def field(self) -> str:
        """The top-level step field the path starts at."""
        return self.path.split(".", 1)[0].removesuffix("[]")


class RefSite(NamedTuple):
    """One ref found in a step dict: ``owner[key]`` holds ``value``; ``path`` is concrete
    (``artifacts[1].id``). ``version`` is the sibling pin, None when unset or not pinnable."""

    path: str
    ref: EntityRef
    value: Any
    version: Any
    owner: dict | list
    key: str | int

    def rewrite(self, value: Any, version: int | None = None) -> None:
        """Replace the ref in its step dict and, when ``version`` is given, pin it beside the ref."""
        if version is not None and self.ref.version_field is None:
            raise ValueError(f"entity ref {self.path!r} has no version field to pin")
        self.owner[self.key] = value
        if version is not None:
            self.owner[self.ref.version_field] = version


def ref_sites(refs: Iterable[EntityRef], data: dict) -> Iterator[RefSite]:
    """Every ref ``refs`` declares that is present, and not None, in ``data``."""
    for ref in refs:
        *parents, leaf = ref.path.split(".")
        for trail, node in _descend(data, parents, ()):
            for rel, holder in ref.walk(node) if ref.walk else [((), node)]:
                if isinstance(holder, dict):
                    yield from _leaf(ref, holder, leaf, (*trail, *rel))


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


def _leaf(ref: EntityRef, holder: dict, leaf: str, trail: tuple[str, ...]) -> Iterator[RefSite]:
    key = leaf.removesuffix("[]")
    value = holder.get(key)
    if value is None:
        return
    path = ".".join((*trail, key))
    if not leaf.endswith("[]"):
        version = holder.get(ref.version_field) if ref.version_field else None
        yield RefSite(path, ref, value, version, holder, key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if item is not None:
                yield RefSite(f"{path}[{i}]", ref, item, None, value, i)
