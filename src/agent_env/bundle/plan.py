"""Plan what a run of a resolved bundle writes, in order, and check it before anything is written.

Planning selects the tasks and evals to run, collects the entities and built images they reach, and
orders them so each is written after what it references. Everything that can be known to fail
before a write is reported together: an id two writers claim, a reference loop, a store id that
doesn't exist or has the wrong type, and a task whose steps don't build. It reads stores but never
writes to one.
"""

from __future__ import annotations

import heapq
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import Artifact, FileArtifact, FileArtifactUniverse
from agent_env.artifact.registry import canonical_type, get_artifact_registry
from agent_env.config import get_config
from agent_env.entity_refs import EntityKind
from agent_env.env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.registry import get_env_registry
from agent_env.env.store import ENVS_COLLECTION
from agent_env.providers.env_providers.constants import (
    AGENT_ENV_WEBSITE_BACKEND_SUFFIX,
    AGENT_ENV_WEBSITE_FRONTEND_SUFFIX,
)
from agent_env.providers.env_providers.env_provider import _env_provider_class, _SandboxEnvironmentProvider
from agent_env.store import Filter, Sort
from agent_env.store.base import NotFoundError
from agent_env.store.ids import LOCAL_PREFIX
from agent_env.store.routing import namespace_routing_enabled
from agent_env.task import Task
from agent_env.task_step.registry import get_task_step_registry

from ._fs import fold, relative, with_article
from .authoring import AuthoringContext, entry_file, entry_files
from .parse import BundleEntry, BundleError, BundleKind
from .resolve import BuiltImage, Reference, ResolvedBundle, ResolvedEntry, build_step

_Key = tuple[str, str]  # (store, id)

_GETTERS = {EntityKind.ENV: Env.get, EntityKind.AGENT: A2AAgent.get, EntityKind.ARTIFACT: Artifact.get,
            EntityKind.TASK: Task.get}

# The artifact types written from their folder's files, and how each lists them.
_FOLDER_WALKS = {FileArtifact: entry_file, FileArtifactUniverse: entry_files}
# The env types whose from_toml reads only the toml and what it names, so a type built by one of them is written from
# its env.toml and the ledger can list what it's made from.
_ENV_FROM_TOMLS = (Env, MCPServerEnv, WebsiteEnv, MultiEnv)

_NOTHING_TO_RUN = "this bundle has no tasks or evals to run"


@dataclass(frozen=True)
class Write:
    """An entity or built image a run writes."""

    kind: BundleKind  # a built image is an ARTIFACT; kind.store keys uniqueness
    id: str
    source: ResolvedEntry | BuiltImage
    needs: tuple[_Key, ...]  # (kind.store, id) of the earlier writes it references


@dataclass(frozen=True)
class Plan:
    bundle: ResolvedBundle
    writes: tuple[Write, ...]  # everything the selection reaches, each after what it references
    tasks: tuple[ResolvedEntry, ...]  # the tasks selected to run on their own
    evals: tuple[ResolvedEntry, ...]
    store_refs: tuple[Reference, ...]  # the store ids the writes reference, checked to exist
    store_latest: dict[tuple[EntityKind, str], int]  # the version each unpinned store ref named when planned


def plan_bundle(resolved: ResolvedBundle, *, tasks: Sequence[str] = (), evals: Sequence[str] = ()) -> Plan:
    """Plan a run of ``tasks`` and ``evals``, each a name or id in this bundle. With neither, the
    run is every eval or, in a bundle without evals, every task. Raises BundleError listing every
    problem found."""
    return _Planner(resolved).plan(tasks, evals)


def check_bundle(resolved: ResolvedBundle) -> None:
    """The checks ``plan_bundle`` makes without reading a store, over every task and eval in the bundle. Raises
    BundleError listing every problem found."""
    _Planner(resolved).check()


class _Planner:
    def __init__(self, resolved: ResolvedBundle):
        self.resolved = resolved
        self.problems: list[str] = []
        self.nodes: dict[_Key, ResolvedEntry | BuiltImage] = {}
        self.rank: dict[_Key, int] = {}
        self.edges: dict[_Key, list[_Key]] = {}
        images: dict[BundleEntry, list[BuiltImage]] = {}
        for image in resolved.built_images:
            images.setdefault(image.entry, []).append(image)
        for entry in resolved.entries:
            for image in images.get(entry.entry, ()):
                self._add(_key(image), image, [])
            needs = dict.fromkeys(_key(ref.local) for ref in entry.references if ref.local is not None)
            self._add(_key(entry.entry), entry, list(needs))
        self.named = {fold(entry.entry.name): entry.entry for entry in resolved.entries}
        self.identified = {entry.entry.id: entry.entry for entry in resolved.entries}
        self.outputs = {(output.kind.value, output.id): (entry.entry, output)
                        for entry in resolved.entries for output in entry.outputs}
        self.accepted: dict[_Key, dict] = {}  # each agent's and env's toml, as its type accepted it

    def _add(self, key: _Key, source: ResolvedEntry | BuiltImage, needs: list[_Key]) -> None:
        """Record a write in bundle order, a built image just before the entry that builds it."""
        self.nodes[key], self.rank[key], self.edges[key] = source, len(self.rank), needs

    def plan(self, tasks: Sequence[str], evals: Sequence[str]) -> Plan:
        self._check_writers()
        self._check_loops()
        selected_tasks, selected_evals = self._selection(tasks, evals)
        closure = self._closure(selected_tasks + selected_evals)
        store_refs, store_latest = self._check_store_refs(closure)
        self._build_tasks(closure)
        self._check_files(closure)
        self._check_env_types(closure)
        self._check_multi_names(closure, store_latest)
        if self.problems:
            raise BundleError(self.problems)
        return Plan(self.resolved, self._ordered(closure), tuple(selected_tasks), tuple(selected_evals), store_refs,
                    store_latest)

    def check(self) -> None:
        self._check_writers()
        self._check_loops()
        if not any(entry.entry.kind in (BundleKind.TASK, BundleKind.EVAL) for entry in self.resolved.entries):
            self.problems.append(_NOTHING_TO_RUN)
        everything = set(self.nodes)
        self._check_store_refs(everything, stores=False)
        self._build_tasks(everything)
        self._check_files(everything)
        if self.problems:
            raise BundleError(self.problems)

    # Checks over the whole bundle

    def _check_writers(self) -> None:
        """Every id is written by one entity, built image or task output, per store."""
        writers: dict[_Key, str] = {}
        claims: list[tuple[_Key, str, str]] = []
        for key, source in self.nodes.items():
            if isinstance(source, BuiltImage):
                claims.append((key, self._path(source.entry), f"the image built for {self._path(source.entry)}"))
            else:
                claims.append((key, self._path(source.entry), self._path(source.entry)))
        for entry in self.resolved.entries:
            for output in entry.outputs:
                claims.append(((output.kind.value, output.id), f"{self._path(entry.entry)}: step {output.step_id!r}",
                               f"step {output.step_id!r} of {self._path(entry.entry)}"))
        for key, where, writer in claims:
            if key in writers:
                self.problems.append(f"{where}: {key[1]!r} is also written by {writers[key]}; give each its own id")
            else:
                writers[key] = writer

    def _check_loops(self) -> None:
        for component in _components(self.nodes, self.edges, self.rank):
            start = min(component, key=self.rank.__getitem__)
            loop = " -> ".join(self._path(self.nodes[key].entry) for key in _loop(start, set(component), self.edges))
            self.problems.append(f"{self._path(self.nodes[start].entry)}: references form a loop: {loop}")

    # Selection

    def _selection(self, tasks: Sequence[str], evals: Sequence[str]) -> tuple[list[ResolvedEntry], list[ResolvedEntry]]:
        if tasks or evals:
            return self._select(BundleKind.TASK, tasks), self._select(BundleKind.EVAL, evals)
        every_task = [entry for entry in self.resolved.entries if entry.entry.kind is BundleKind.TASK]
        every_eval = [entry for entry in self.resolved.entries if entry.entry.kind is BundleKind.EVAL]
        if every_eval:
            return [], every_eval
        if not every_task:
            self.problems.append(_NOTHING_TO_RUN)
        return every_task, []

    def _select(self, kind: BundleKind, wanted: Sequence[str]) -> list[ResolvedEntry]:
        candidates = [entry for entry in self.resolved.entries if entry.entry.kind is kind]
        exact = {_nfc(text): entry for entry in candidates for text in (entry.entry.name, entry.entry.id)}
        near = {fold(text): entry for entry in candidates for text in (entry.entry.name, entry.entry.id)}
        selected: dict[_Key, ResolvedEntry] = {}
        for value in wanted:
            flag = f"--{kind.store} {value!r}"
            entry = exact.get(_nfc(value))
            if entry is not None:
                selected.setdefault(_key(entry.entry), entry)
            elif fold(value) in near:
                self.problems.append(f"{flag}: isn't in this bundle, but {near[fold(value)].entry.name!r} is; names "
                                     "are case-sensitive")
            else:
                names = ", ".join(repr(entry.entry.name) for entry in candidates)
                listed = f"; its {kind.value} are {names}" if names else f"; it has no {kind.value}"
                self.problems.append(f"{flag}: this bundle has no {kind.store} with that name or id{listed}")
        return list(selected.values())

    def _closure(self, roots: list[ResolvedEntry]) -> set[_Key]:
        closure: set[_Key] = set()
        frontier = [_key(root.entry) for root in roots]
        while frontier:
            key = frontier.pop()
            if key not in closure:
                closure.add(key)
                frontier.extend(self.edges[key])
        return closure

    # Checks over what the selection reaches

    def _check_store_refs(
        self, closure: set[_Key], *, stores: bool = True
    ) -> tuple[tuple[Reference, ...], dict[tuple[EntityKind, str], int]]:
        """Check each store id the closure references; without ``stores``, only what needs no store read."""
        read: dict[tuple[EntityKind, str, int | None], Any] = {}
        checked: dict[tuple[EntityKind, str, int | None], Reference] = {}
        latest: dict[tuple[EntityKind, str], int] = {}
        for key in sorted(closure, key=self.rank.__getitem__):
            source = self.nodes[key]
            if isinstance(source, BuiltImage):
                continue
            for ref in source.references:
                if ref.local is not None:
                    continue
                problem = self._bundle_ref_problem(ref)
                if problem is None and stores:
                    problem = self._store_ref_problem(ref, read)
                if problem:
                    self.problems.append(f"{self._path(source.entry)}: {ref.where}: {problem}")
                elif stores:
                    checked.setdefault((ref.kind, ref.id, ref.version), ref)
                    if ref.version is None:
                        latest[ref.kind, ref.id] = read[ref.kind, ref.id, None].version
        return tuple(checked.values()), latest

    def _bundle_ref_problem(self, ref: Reference) -> str | None:
        """What makes an ``@local`` id wrong without reading a store: it names one of this bundle's outputs by id,
        one of its entities of another kind, or an id under its root that it doesn't write. Another bundle's id is
        read like any store id."""
        if not ref.id.startswith(LOCAL_PREFIX):
            return None
        kind, key = ref.kind.value, _nfc(ref.id)
        if (kind, key) in self.outputs:
            entry, output = self.outputs[kind, key]
            return (f"{ref.id!r} is written by step {output.step_id!r} of {self._path(entry)}; refer to an output "
                    "by its name, in the task that writes it")
        other = self.identified.get(key)
        if other is not None:
            return (f"{ref.id!r} is this bundle's {other.kind.store} {other.name!r}, but this field takes "
                    f"{with_article(kind)}")
        if key.startswith(f"{self.resolved.bundle.id_root}/"):
            return f"{ref.id!r} isn't in this bundle"
        return None

    def _store_ref_problem(self, ref: Reference, read: dict) -> str | None:
        if ref.id.startswith(LOCAL_PREFIX) and not namespace_routing_enabled():
            raise RuntimeError("reading an @local id from the store needs namespace routing, which the agent-env CLI "
                               "turns on; call plan_bundle inside agent_env.store.routing.namespace_routing()")
        kind = ref.kind.value
        lookup = (ref.kind, ref.id, ref.version)
        if lookup not in read:
            try:
                read[lookup] = _GETTERS[ref.kind](ref.id, ref.version)
            except NotFoundError:
                read[lookup] = None
            except (ValueError, KeyError, TypeError) as e:
                read[lookup] = f"can't be read ({type(e).__name__}: {e})"
        found = read[lookup]
        if found is None:
            at = f" version {ref.version}" if ref.version is not None else ""
            other = self.named.get(fold(ref.id))
            hint = f"; this bundle's {other.kind.store} {other.name!r} has that name" if other is not None else ""
            return f"there is no {kind} {ref.id!r}{at} in the store{hint}"
        if isinstance(found, str):
            return f"{kind} {ref.id!r} {found}"
        if ref.artifact_type and canonical_type(found.type) != canonical_type(ref.artifact_type):
            return f"{ref.id!r} is {with_article(found.type)} in the store, but this field takes {ref.artifact_type}"
        return None

    def _check_files(self, closure: set[_Key]) -> None:
        """List the folder of each file artifact, and read its toml and each agent's and env's the way their
        writes will, so a link that leaves the bundle or a key the type doesn't take fails before anything is
        written."""
        registry = get_artifact_registry()
        for key in sorted(closure, key=self.rank.__getitem__):
            source = self.nodes[key]
            if isinstance(source, BuiltImage):
                continue
            if source.entry.kind in (BundleKind.AGENT, BundleKind.ENV):
                agent = source.entry.kind is BundleKind.AGENT
                accept = A2AAgent.accept_toml if agent else self._env_accept(source.entry)
                try:
                    if accept is not None:
                        self.accepted[key] = accept(source.config, AuthoringContext(self.resolved.bundle, source.entry))
                except BundleError as e:
                    self.problems.extend(e.problems)
                continue
            if source.entry.kind is not BundleKind.ARTIFACT:
                continue
            cls = registry.get(canonical_type(source.entry.type))
            walk = folder_walk(cls)
            if walk is None:
                continue
            try:
                walk(self.resolved.bundle, source.entry)
            except BundleError as e:
                self.problems.extend(e.problems)
            try:
                AuthoringContext(self.resolved.bundle, source.entry).accept(source.config, **cls.toml_keys)
            except BundleError as e:
                self.problems.extend(e.problems)

    def _env_accept(self, entry: BundleEntry) -> Any:
        """How ``entry``'s env type checks its env.toml, when it's an env written from one and checks it at all."""
        cls = get_env_registry().get(entry.type) if entry.kind is BundleKind.ENV else None
        return getattr(cls, "accept_toml", None) if env_writer(cls) else None

    def _check_env_types(self, closure: set[_Key]) -> None:
        """An env keeps its type across versions, so an env folder whose id is another type of env in the store
        is refused here, not by the store once earlier entities are written."""
        store = get_config().get_document_store()
        for key in sorted(closure, key=self.rank.__getitem__):
            source = self.nodes[key]
            if isinstance(source, BuiltImage) or source.entry.kind is not BundleKind.ENV:
                continue
            latest = Sort.by("version", descending=True)
            stored = store.find_one(ENVS_COLLECTION, Filter.of(id=source.entry.id), sort=latest)
            if stored is not None and stored.get("type") not in (None, source.entry.type):
                self.problems.append(f"{self._path(source.entry)}: {source.entry.id!r} is "
                                     f"{with_article(stored.get('type'))} env in the store, and an env keeps its type "
                                     "across versions; rename the folder")

    def _check_multi_names(self, closure: set[_Key], store_latest: dict[tuple[EntityKind, str], int]) -> None:
        """Each env of a multi runs as containers its environment_name names: an MCP server as one by that name, a
        website as its backend's and frontend's. No two of a multi's envs may name one container, and a provider
        other than the core's, which gives a multi one env card, can't tell an MCP server and a website of one name
        apart."""
        for key in sorted(closure, key=self.rank.__getitem__):
            source = self.nodes[key]
            if isinstance(source, BuiltImage) or source.entry.kind is not BundleKind.ENV:
                continue
            if not issubclass(get_env_registry().get(source.entry.type, Env), MultiEnv):
                continue
            where = self._path(source.entry)
            taken: dict[str, str] = {}  # each container a child runs as, by the field naming the child
            named: dict[str, set[str]] = {}  # the environment_names of each list's envs
            for ref in source.references:
                name = self._environment_name(ref, store_latest)
                if name is None:
                    continue
                group = ref.where.partition("[")[0]
                named.setdefault(group, set()).add(name)
                containers = [name] if group == "mcp_server_envs" else [
                    f"{name}-{AGENT_ENV_WEBSITE_BACKEND_SUFFIX}", f"{name}-{AGENT_ENV_WEBSITE_FRONTEND_SUFFIX}"]
                clash = next((container for container in containers if container in taken), None)
                if clash is not None:
                    self.problems.append(f"{where}: {ref.where}: environment_name {name!r} names the container "
                                         f"{clash!r}, as {taken[clash]}'s does; each env of a multi needs its own")
                else:
                    taken.update(dict.fromkeys(containers, ref.where))
            shared = named.get("mcp_server_envs", set()) & named.get("website_envs", set())
            provider_type = source.config.get("env_provider_type", "gateway")
            if shared and not _deploys_in_sandboxes(provider_type):
                self.problems.append(f"{where}: env_provider_type {provider_type!r} gives a multi one env card, which "
                                     "can't tell an MCP server and a website apart by name, and both are named "
                                     f"{', '.join(map(repr, sorted(shared)))}; rename one")

    def _environment_name(self, ref: Reference, store_latest: dict[tuple[EntityKind, str], int]) -> str | None:
        if ref.local is not None:
            return self.accepted.get(_key(ref.local), {}).get("environment_name")
        try:
            env = Env.get(ref.id, ref.version if ref.version is not None else store_latest.get((ref.kind, ref.id)))
        except (NotFoundError, ValueError, KeyError, TypeError):
            return None  # reported by the store ref check
        return getattr(env, "environment_name", None)

    def _build_tasks(self, closure: set[_Key]) -> None:
        """Build each task's steps the way a write will, so a bad step fails before anything is written."""
        registry = get_task_step_registry()
        for key in sorted(closure, key=self.rank.__getitem__):
            source = self.nodes[key]
            if not (isinstance(source, ResolvedEntry) and source.entry.kind is BundleKind.TASK):
                continue
            steps, failed = [], False
            for step in source.config:
                try:
                    steps.append(build_step(step))
                except Exception as e:
                    failed = True
                    self.problems.append(f"{self._path(source.entry)}: step {step.get('id')!r}: "
                                         f"{registry[step['type']].type} can't read it ({type(e).__name__}: {e})")
            if not failed:
                try:
                    Task(id=source.entry.id, version=None, steps=steps)
                except ValueError as e:
                    self.problems.append(f"{self._path(source.entry)}: {e}")

    # Order

    def _ordered(self, closure: set[_Key]) -> tuple[Write, ...]:
        waiting = {key: len(self.edges[key]) for key in closure}
        dependents: dict[_Key, list[_Key]] = {}
        for key in closure:
            for need in self.edges[key]:
                dependents.setdefault(need, []).append(key)
        ready = [(self.rank[key], key) for key, count in waiting.items() if count == 0]
        heapq.heapify(ready)
        writes = []
        while ready:
            _, key = heapq.heappop(ready)
            source = self.nodes[key]
            kind = BundleKind.ARTIFACT if isinstance(source, BuiltImage) else source.entry.kind
            writes.append(Write(kind, key[1], source, tuple(self.edges[key])))
            for dependent in dependents.get(key, ()):
                waiting[dependent] -= 1
                if waiting[dependent] == 0:
                    heapq.heappush(ready, (self.rank[dependent], dependent))
        return tuple(writes)

    def _path(self, entry: BundleEntry) -> str:
        return relative(self.resolved.bundle.root, entry.path)


def unpinned_store_refs(plan: Plan, write: Write) -> dict[tuple[EntityKind, str], int]:
    """The version the plan read for each store entity ``write`` names without a version, by (kind, id). Its writer
    pins each to that version and the ledger hashes them, so a version the store gains in between isn't written,
    and one it gains later rewrites ``write``."""
    if isinstance(write.source, BuiltImage):
        return {}
    return {(ref.kind, ref.id): plan.store_latest[ref.kind, ref.id]
            for ref in write.source.references if ref.local is None and ref.version is None}


def _deploys_in_sandboxes(provider_type: str) -> bool:
    """Whether the provider ``provider_type`` names is one of the core's, which deploy each env in a sandbox of its
    own; a type no installed provider has is accepted here, since the env's toml check refuses it."""
    try:
        return issubclass(_env_provider_class(provider_type), _SandboxEnvironmentProvider)
    except ValueError:
        return True


def env_writer(cls: type | None) -> bool:
    """Whether an env of type ``cls`` is written from its env.toml, by a ``from_toml`` that reads only the toml and
    what it names: its own, or one it inherits from a core env type."""
    return cls is not None and next(c for c in cls.__mro__ if "from_toml" in vars(c)) in _ENV_FROM_TOMLS


def folder_walk(cls: type | None) -> Any:
    """The listing a type's write runs: a subclass keeping its base's ``from_toml`` lists the same way,
    one that overrides it may read its folder differently, so it gets none."""
    for base, walk in _FOLDER_WALKS.items():
        if cls is not None and issubclass(cls, base):
            return walk if keeps_base_from_toml(cls, base) else None
    return None


def keeps_base_from_toml(cls: type, base: type) -> bool:
    """Whether ``cls`` is built from its TOML by ``base``'s ``from_toml`` rather than one of its own."""
    return next(c for c in cls.__mro__ if "from_toml" in vars(c)) is base


def _key(entity: BundleEntry | BuiltImage) -> _Key:
    kind = BundleKind.ARTIFACT if isinstance(entity, BuiltImage) else entity.kind
    return (kind.store, entity.id)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _components(nodes: dict[_Key, Any], edges: dict[_Key, list[_Key]], rank: dict[_Key, int]) -> list[list[_Key]]:
    """The strongly connected components that hold a loop (Tarjan's algorithm, without recursion)."""
    index: dict[_Key, int] = {}
    low: dict[_Key, int] = {}
    stack: list[_Key] = []
    on_stack: set[_Key] = set()
    loops = []
    for root in sorted(nodes, key=rank.__getitem__):
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(edges[root]))]
        while work:
            node, pending = work[-1]
            for need in pending:
                if need not in index:
                    index[need] = low[need] = len(index)
                    stack.append(need)
                    on_stack.add(need)
                    work.append((need, iter(edges[need])))
                    break
                if need in on_stack:
                    low[node] = min(low[node], index[need])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1 or node in edges[node]:
                        loops.append(component)
    return loops


def _loop(start: _Key, component: set[_Key], edges: dict[_Key, list[_Key]]) -> list[_Key]:
    """A shortest path from ``start`` back to itself inside its component."""
    previous: dict[_Key, _Key | None] = {start: None}
    queue = [start]
    for node in queue:
        if start in edges[node]:
            break
        for need in edges[node]:
            if need in component and need not in previous:
                previous[need] = node
                queue.append(need)
    path = [node]
    while previous[path[-1]] is not None:
        path.append(previous[path[-1]])
    return [*reversed(path), start]
