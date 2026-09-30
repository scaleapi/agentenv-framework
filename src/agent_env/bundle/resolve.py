"""Rewrite each reference in a parsed bundle to the id it names, before anything is built or written.

References are the fields step types declare in ``entity_refs`` and the toml keys entity types declare
in ``toml_refs``. The name of one of this bundle's entities becomes that entity's id, a name an output
of the same task writes becomes the output's id, and anything else is a store id, left as written;
whether store ids exist is checked later, with the other checks that read a store.
"""

from __future__ import annotations

import copy
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.registry import canonical_type, get_artifact_registry
from agent_env.entity_refs import EntityKind, EntityRef, RefRole, RefSite, parse_toml_ref, ref_sites
from agent_env.env.registry import get_env_registry
from agent_env.eval.eval import Eval
from agent_env.plugins import _registration
from agent_env.store.ids import LOCAL_PREFIX, validate_local_id
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep, attach_retry_config, dependencies

from ._fs import fold, relative, show
from .parse import NAMED_BY, Bundle, BundleEntry, BundleError, BundleKind

# Step keys that hold step ids, never entity ids.
_STEP_ID_KEYS = ("id", "type", "depends_on")


@dataclass(frozen=True)
class BuiltImage:
    """The image built from an env or agent folder for one of its image keys."""

    id: str
    entry: BundleEntry
    dockerfile: str  # relative to the entry's folder


@dataclass(frozen=True)
class Reference:
    """A resolved reference: to ``local``, this bundle's entity or built image, or, when ``local`` is
    None, to a store id at ``version`` (None for its latest)."""

    kind: EntityKind
    id: str
    version: int | None
    local: BundleEntry | BuiltImage | None
    where: str  # the referencing field, as problems name it
    artifact_type: str | None  # the type the field takes, when it names one


@dataclass(frozen=True)
class Output:
    """An id a task's step writes, named under the task."""

    kind: EntityKind
    id: str
    step: int  # the writing step's index in the task
    step_id: Any
    artifact_type: str | None


@dataclass(frozen=True)
class ResolvedEntry:
    """An entry's config as a copy with every declared reference rewritten, what it references, and
    for a task what its steps write."""

    entry: BundleEntry
    config: Any
    references: tuple[Reference, ...]
    outputs: tuple[Output, ...]


@dataclass(frozen=True)
class ResolvedBundle:
    bundle: Bundle
    entries: tuple[ResolvedEntry, ...]
    built_images: tuple[BuiltImage, ...]


def resolve_bundle(bundle: Bundle) -> ResolvedBundle:
    """Resolve every entry's references. Raises BundleError listing every problem found."""
    return _Resolver(bundle).resolve()


class _Skip(Exception):
    """An entry that can't be resolved; its problem is already recorded."""


class _Resolver:
    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self.problems: list[str] = []
        self.built: list[BuiltImage] = []
        # By kind and exact name or id, by kind and folded name, and by folded name across the kinds a
        # step can name (steps never name tasks or evals).
        self.named: dict[tuple[EntityKind, str], BundleEntry] = {}
        self.folded: dict[tuple[EntityKind, str], BundleEntry] = {}
        stepped = {bundle_kind for kind in (EntityKind.ENV, EntityKind.AGENT, EntityKind.ARTIFACT)
                   for bundle_kind in NAMED_BY[kind]}
        self.step_named = {fold(entry.name): entry for entry in bundle.entries if entry.kind in stepped}
        self.ignored: dict[tuple[EntityKind, str], Path] = {}
        for kind, bundle_kinds in NAMED_BY.items():
            for entry in bundle.entries:
                if entry.kind in bundle_kinds:
                    self.named[kind, _nfc(entry.name)] = self.named[kind, _nfc(entry.id)] = entry
                    self.folded[kind, fold(entry.name)] = self.folded[kind, fold(entry.id)] = entry
            folders = {bundle_kind.value for bundle_kind in bundle_kinds}
            for path in bundle.ignored:
                if len(path.parts) == 2 and path.parts[0] in folders:
                    for name in (path.name, path.stem):
                        self.ignored[kind, fold(name)] = path

    def resolve(self) -> ResolvedBundle:
        resolved = []
        for entry in self.bundle.entries:
            try:
                resolved.append(self._task(entry) if entry.kind is BundleKind.TASK else self._toml(entry))
            except _Skip:
                pass
        if self.problems:
            raise BundleError(self.problems)
        return ResolvedBundle(self.bundle, tuple(resolved), tuple(self.built))

    # One method per kind of config

    def _toml(self, entry: BundleEntry) -> ResolvedEntry:
        config = copy.deepcopy(entry.config)
        if not isinstance(config, dict):
            return ResolvedEntry(entry, config, (), ())
        references: list[Reference] = []
        refs = []
        for ref in getattr(self._toml_class(entry), "toml_refs", ()):
            if not self._image(entry, config, ref, references):
                refs.append(ref)
        for site in ref_sites(refs, config, inline_pins=True):
            try:
                name, version = parse_toml_ref(site.ref.kind, site.owner if site.version_key else site.value)
            except ValueError as e:
                self._problem(entry, f"{site.path}: {e}")
                continue
            self._resolve(entry, site, site.path, name, version, {}, set(), references)
        if entry.kind is BundleKind.EVAL:
            self._refuse_repeated_tasks(entry, references)
        return ResolvedEntry(entry, config, tuple(references), ())

    def _refuse_repeated_tasks(self, entry: BundleEntry, references: list[Reference]) -> None:
        first: dict[str, str] = {}
        for ref in references:
            where = ref.where.partition(".")[0]
            if ref.id in first:
                task = f"this bundle's task {ref.local.name!r}" if ref.local is not None else repr(ref.id)
                self._problem(entry, f"{where}: {task} is also {first[ref.id]}; an eval lists each task once "
                              "(for repeat runs, use agent-env eval run --k)")
            else:
                first[ref.id] = where

    def _task(self, entry: BundleEntry) -> ResolvedEntry:
        steps = copy.deepcopy(entry.config)
        registry = get_task_step_registry()
        declared = []
        for index, step in enumerate(steps):
            step_type = step.get("type")
            cls = registry.get(step_type) if isinstance(step_type, str) else None
            if cls is None:
                note = _registration.failure_note(_registration.TASK_STEPS, step_type) if isinstance(step_type, str) else ""
                self._problem(entry, f"step {step.get('id')!r}: {step_type!r} is not a known step type{note}")
            elif self._derived_outputs(entry, step, cls):
                declared.append((index, step, cls))
        outputs = self._outputs(entry, declared)
        upstream = _upstream(steps, self._depends_on(entry, steps))
        references: list[Reference] = []
        for index, step, cls in declared:
            where = f"step {step.get('id')!r}: "
            if cls.entity_refs is None:
                self._undeclared(entry, step, cls, outputs, where)
                continue
            for site in ref_sites(cls.entity_refs, step):
                if site.ref.role is RefRole.OUTPUT:
                    continue
                if not (isinstance(site.value, str) and site.value):
                    self._problem(entry, f"{where}{site.path} must be an id, not {site.value!r}")
                    continue
                self._resolve(entry, site, f"{where}{site.path}", site.value, site.version, outputs, upstream[index],
                              references)
        return ResolvedEntry(entry, steps, tuple(references), tuple(outputs.values()))

    # Parts of a config

    def _toml_class(self, entry: BundleEntry) -> type:
        group = None
        if entry.kind is BundleKind.ENV:
            cls, group = get_env_registry().get(entry.type), _registration.ENVS
        elif entry.kind is BundleKind.AGENT:
            cls = A2AAgent if entry.type == A2AAgent.type else None
        elif entry.kind is BundleKind.EVAL:
            cls = Eval if entry.type == Eval.type else None
        else:
            cls, group = get_artifact_registry().get(canonical_type(entry.type)), _registration.ARTIFACTS
        if cls is None:
            name = canonical_type(entry.type) if group == _registration.ARTIFACTS else entry.type
            note = _registration.failure_note(group, name) if group else ""
            self._problem(entry, f"{entry.type!r} is not a known {entry.kind.store} type{note}")
            raise _Skip
        return cls

    def _image(self, entry: BundleEntry, config: dict, ref: EntityRef, references: list[Reference]) -> bool:
        """Record the image an image key builds: the folder's Dockerfile when ``image`` is left out, or
        ``{ dockerfile = "<path>" }``. False when the key names an image artifact instead."""
        if ref.artifact_type != "docker_image" or ref.path != ref.field:
            return False
        value = config.get(ref.path)
        if isinstance(value, dict) and "ref" in value:
            self._problem(entry, f"{ref.path}: an external image ({{ ref = ... }}) isn't supported yet")
            return True
        if value is None and ref.path == "image":
            dockerfile = "Dockerfile"
        elif isinstance(value, dict) and value.keys() == {"dockerfile"}:
            dockerfile = value["dockerfile"]
        else:
            return False
        path = entry.path / dockerfile if isinstance(dockerfile, str) and dockerfile else None
        inside = path is not None and not Path(dockerfile).is_absolute() and path.resolve().is_relative_to(entry.path.resolve())
        if not (inside and path.is_file()):
            self._problem(entry, f"{ref.path}: there is no {dockerfile!r} in this folder to build")
            return True
        role = f"{entry.kind.store}_image" if ref.path == "image" else ref.path
        image = BuiltImage(f"{entry.id}__{role}", entry, dockerfile)
        config[ref.path] = image.id
        self.built.append(image)
        references.append(Reference(EntityKind.ARTIFACT, image.id, None, image, ref.path, ref.artifact_type))
        return True

    def _derived_outputs(self, entry: BundleEntry, step: dict, cls: type) -> bool:
        """Write in any output id the step's type derives when the field is left out (build_mcp_cli's
        ``cli-<env_id>``), so it's named like one written out. False when its type can't read the step."""
        absent = [ref for ref in cls.entity_refs or ()
                  if ref.role is RefRole.OUTPUT and ref.path == ref.field and step.get(ref.field) is None]
        if not absent:
            return True
        try:
            read = build_step(step).to_dict()
        except Exception as e:
            self._problem(entry, f"step {step.get('id')!r}: {cls.type} can't read it ({type(e).__name__}: {e})")
            return False
        for ref in absent:
            if read.get(ref.field) is not None:
                step[ref.field] = read[ref.field]
        return True

    def _outputs(self, entry: BundleEntry, declared: list[tuple[int, dict, type]]) -> dict[tuple[EntityKind, str], Output]:
        """Name each id a step writes under its task, keyed by kind and the name as written."""
        outputs: dict[tuple[EntityKind, str], Output] = {}
        for index, step, cls in declared:
            refs = [ref for ref in cls.entity_refs or () if ref.role is RefRole.OUTPUT]
            for site in ref_sites(refs, step):
                where, name, kind = f"step {step.get('id')!r}: {site.path}", site.value, site.ref.kind
                if not (isinstance(name, str) and name):
                    self._problem(entry, f"{where} must be an id, not {name!r}")
                    continue
                name = _nfc(name)
                clash = self.named.get((kind, name)) or self.folded.get((kind, fold(name)))
                if clash is not None:
                    self._problem(entry, f"{where}: {name!r} is written by this step and is also this bundle's "
                                  f"{kind.value} {clash.name!r}; rename one")
                    continue
                if (kind, name) in outputs:
                    self._problem(entry, f"{where}: {name!r} is also written by step "
                                  f"{outputs[kind, name].step_id!r}; give each output its own name")
                    continue
                output_id = name if name.startswith(LOCAL_PREFIX) else f"{entry.id}/{name}"
                try:
                    validate_local_id(output_id)
                except ValueError as e:
                    self._problem(entry, f"{where}: {e}")
                    continue
                site.rewrite(output_id)
                outputs[kind, name] = Output(kind, output_id, index, step.get("id"), site.ref.artifact_type)
        return outputs

    def _depends_on(self, entry: BundleEntry, steps: list[dict]) -> list[list[str] | None]:
        """Each step's ``depends_on`` as the step ids it names, read as building the step reads it; None when it
        has none. One that can't be read, or names no step in the task, is a problem here: a step reading an
        earlier step's output is checked against these edges before any step is built."""
        step_ids = {step.get("id") for step in steps}
        depends = []
        for step in steps:
            try:
                read = dependencies(step.get("depends_on"))
            except ValueError as e:
                self._problem(entry, f"step {step.get('id')!r}: {e}")
                read = []
            ids = None if read is None else [dep.task_step_id for dep in read]
            for dep_id in ids or ():
                if dep_id not in step_ids:
                    self._problem(entry, f"step {step.get('id')!r}: depends_on {dep_id!r} names no step in this task")
            depends.append(ids)
        return depends

    def _resolve(self, entry: BundleEntry, site: RefSite, where: str, name: str, version: int | None,
                 outputs: dict[tuple[EntityKind, str], Output], upstream: set[int],
                 references: list[Reference]) -> None:
        kind, expected, key = site.ref.kind, site.ref.artifact_type, _nfc(name)
        output = outputs.get((kind, key))
        if output is not None:
            if output.step not in upstream:
                self._problem(entry, f"{where}: {name!r} is written by step {output.step_id!r}, which this step "
                              "doesn't depend on")
            elif version is not None:
                self._problem(entry, f"{where}: {name!r} is written by this task, so it has no fixed version; "
                              "drop the pin")
            elif expected and output.artifact_type and not _same_type(output.artifact_type, expected):
                self._problem(entry, f"{where}: this task writes {name!r} as {output.artifact_type}, but this field "
                              f"takes {expected}")
            else:
                site.rewrite(output.id)
            return
        local = self.named.get((kind, key))
        if local is not None:
            if version is not None:
                self._problem(entry, f"{where}: {name!r} is defined in this bundle, so it has no fixed version; "
                              "drop the pin")
            elif expected and not _same_type(local.type, expected):
                self._problem(entry, f"{where}: {name!r} is this bundle's {local.type}, but this field takes {expected}")
            else:
                site.rewrite(local.id)
                references.append(Reference(kind, local.id, None, local, where, expected))
            return
        near = self.folded.get((kind, fold(name)))
        if near is not None:
            near_name = near.id if fold(name) == fold(near.id) else near.name
        else:
            near_name = next((n for k, n in outputs if k is kind and fold(n) == fold(name)), None)
        ignored = self.ignored.get((kind, fold(name)))
        if near_name is not None:
            self._problem(entry, f"{where}: {name!r} isn't in this bundle, but {near_name!r} is; names are "
                          "case-sensitive")
        elif ignored is not None:
            self._problem(entry, f"{where}: {name!r} names {show(ignored)}, which this bundle ignores")
        else:
            references.append(Reference(kind, name, version, None, where, expected))

    def _undeclared(self, entry: BundleEntry, step: dict, cls: type, outputs: dict[tuple[EntityKind, str], Output],
                    where: str) -> None:
        """A step type that declares no references passes through, unless it repeats a bundle name that
        it can't have rewritten."""
        written = {fold(name) for _, name in outputs}
        for path, value in _strings(step):
            match = self.step_named.get(fold(value))
            if match is not None:
                named = f"this bundle's {match.kind.store} {match.name!r}"
            elif fold(value) in written:
                named = "an output of this task"
            else:
                continue
            self._problem(entry, f"{where}{path} = {value!r} names {named}, but {cls.type} doesn't declare its "
                          "entity_refs, so it can't be rewritten")

    def _problem(self, entry: BundleEntry, message: str) -> None:
        self.problems.append(f"{relative(self.bundle.root, entry.path)}: {message}")


def build_step(step: dict) -> TaskStep:
    """The step a task's write puts, built from a copy of ``step``, so the resolved config the ledger hashes
    never changes."""
    step = copy.deepcopy(step)
    return attach_retry_config(get_task_step_registry()[step["type"]].from_dict(step), step)


def _nfc(name: str) -> str:
    """The form names are compared in: the parser's entity names are NFC, as authored text may not be."""
    return unicodedata.normalize("NFC", name)


def _upstream(steps: list[dict], depends: list[list[str] | None]) -> list[set[int]]:
    """The indexes of the steps each step runs after: the step ids ``depends`` holds for it or, when it has
    none, every step before it, followed transitively; the rule the task scheduler runs steps by."""
    index_of: dict[str, int] = {}
    for index, step in enumerate(steps):
        if isinstance(step.get("id"), str):
            index_of.setdefault(step["id"], index)
    direct = [set(range(index)) if ids is None else {index_of[dep_id] for dep_id in ids if dep_id in index_of}
              for index, ids in enumerate(depends)]
    upstream = []
    for index in range(len(steps)):
        seen: set[int] = set()
        frontier = list(direct[index])
        while frontier:
            current = frontier.pop()
            if current not in seen:
                seen.add(current)
                frontier.extend(direct[current])
        upstream.append(seen)
    return upstream


def _same_type(artifact_type: str, expected: str) -> bool:
    return canonical_type(artifact_type) == canonical_type(expected)


def _strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Every string in a step with its path, except under the keys that hold step ids."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            if (not path and key in _STEP_ID_KEYS) or str(key).endswith(("_step_id", "_step_ids")):
                continue
            yield from _strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _strings(item, f"{path}[{i}]")
