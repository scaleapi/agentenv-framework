"""Plan what a run of a resolved bundle writes, in order, and check it before anything is written.

Planning selects the tasks and evals to run, collects the entities and built images they reach, and
orders them so each is written after what it references. Everything that can be known to fail
before a write is reported together: an id two writers claim, a reference loop, a store id that
doesn't exist or has the wrong type, and a task whose steps don't build. It reads stores but never
writes to one.

A reference by id to an env, agent or artifact an installed bundle defines, and a step's fallback to the configured
default agent when an installed bundle defines it, pulls that entity into the plan with what it references in its
bundle, so the run checks and writes it as it does its own. Such an id has no fixed version, and one the installed
bundle doesn't define is refused rather than read from the store. A reference to an agent by an alias an installed
bundle gives it (its ``aliases.toml``) does the same, unless the store has an agent of that id, which it names instead;
the task is written naming the agent by its id.
"""

from __future__ import annotations

import copy
import heapq
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent.store import get_a2a_agent_store
from agent_env.artifact import Artifact, CliArtifact, EnvironmentArtifact, SeedsEnvironments
from agent_env.artifact.artifact import FolderChild
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.registry import canonical_type, get_artifact_registry
from agent_env.config import get_config
from agent_env.entity.refs import EntityKind, RefRole, default_agent_refs, ref_sites
from agent_env.env.env import Env
from agent_env.env.envs._deployment import provider_refusal
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.app_env import ComputerAppEnv
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.published_images import published_bundle_image
from agent_env.env.registry import get_env_registry
from agent_env.env.store import ENVS_COLLECTION, get_env_store
from agent_env.plugins._bundles import (AGENT_ALIASES, ARTIFACT_ALIASES, ENV_ALIASES, InstalledAliases,
                                        installed_aliases)
from agent_env.providers.env_providers.constants import (
    AGENT_ENV_WEBSITE_BACKEND_SUFFIX,
    AGENT_ENV_WEBSITE_FRONTEND_SUFFIX,
)
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_providers.env_provider import _env_provider_class
from agent_env.store import Filter, Sort
from agent_env.store.base import NotFoundError
from agent_env.store.ids import LOCAL_PREFIX
from agent_env.store.routing import namespace_routing_enabled
from agent_env.task.task import Task
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

from ._fs import fold, relative, with_article
from .authoring import AuthoringContext, entry_files, staged_build_context
from .parse import NAMED_BY, Bundle, BundleEntry, BundleError, BundleKind, parse_bundle
from .resolve import (BuiltImage, BundleImage, Reference, ResolvedBundle, ResolvedEntry, build_step, entity_class,
                      resolve_bundle)

if TYPE_CHECKING:
    from .installed import InstalledBundle

_Key = tuple[str, str]  # (store, id)
# What lists the installed bundles, installed_bundles, which imports this module.
Installed = Callable[[], "tuple[InstalledBundle, ...]"]
# The kinds a reference pulls in from an installed bundle.
_PULLED = (EntityKind.ENV, EntityKind.AGENT, EntityKind.ARTIFACT)


def _stored_agent(id: str, version: int | None) -> A2AAgent:
    """The store's agent of ``id`` itself: a plan weighs an installed alias against the store on its own."""
    return get_a2a_agent_store().get(id, version)


def _stored_env(id: str, version: int | None) -> Env:
    """The store's env of ``id`` itself, as ``_stored_agent`` is the store's agent."""
    return get_env_store().get(id, version)


STORE_GETTERS = {EntityKind.ENV: _stored_env, EntityKind.AGENT: _stored_agent, EntityKind.ARTIFACT: Artifact.get,
                 EntityKind.TASK: Task.get}

# The type a field naming an environment artifact takes: what an artifact referencing one holds.
_ENVIRONMENT = EnvironmentArtifact.model_fields["type"].default
# The folder hooks the plan asks an artifact type, each with the class declaring it: every artifact type answers
# written_besides, and a type opts into seeded_environments by declaring SeedsEnvironments.
_FOLDER_HOOKS = {"written_besides": Artifact, "seeded_environments": SeedsEnvironments}
# The env types whose from_toml reads only the toml and what it names, so a type built by one of them is written from
# its env.toml and the ledger can list what it's made from.
_ENV_FROM_TOMLS = (Env, MCPServerEnv, WebsiteEnv, MultiEnv, ComputerDriverEnv, ComputerEnv, ComputerAppEnv)
# The env types whose from_toml also reads every file in the env's folder, which the ledger hashes as an artifact's.
_ENV_FROM_FOLDERS = (ComputerDriverEnv, ComputerAppEnv)

_NOTHING_TO_RUN = "this bundle has no tasks or evals to run"


@dataclass(frozen=True)
class Publication:
    """The image published for the sources a bundle builds an agent's image from, which the run registers by its tag
    and digest, with those sources as its build context."""

    ref: str  # its repository and source tag
    digest: str
    sources: str  # the source digest of the build context it was found for

    @property
    def pinned(self) -> str:
        return f"{self.ref}@{self.digest}"

    def unwritten(self, image_id: str) -> DockerImageArtifact:
        """The image as the run will store it as ``image_id``, but for its build context, which only the write uploads:
        what a deploy's providers are checked against before it's written."""
        return DockerImageArtifact(id=image_id, description="", image_name=self.ref, image_digest=self.digest,
                                   source_digest=self.sources)


# The aliases.toml table of each kind a task's reference may name by an alias.
_ALIAS_TABLES = {EntityKind.AGENT: AGENT_ALIASES, EntityKind.ENV: ENV_ALIASES, EntityKind.ARTIFACT: ARTIFACT_ALIASES}

@dataclass(frozen=True)
class AliasUse:
    """An agent, env or artifact alias the run names, and what it names: the one an installed bundle gives the alias
    or, when the store has one of that id, that one, which shadows it."""

    alias: str
    target_id: str  # the agent, env or artifact the installed bundle gives the alias
    declared_by: str  # "<dist>'s <name> bundle"
    store_version: int | None = None  # the store's of the alias's id, when it shadows the alias
    kind: EntityKind = EntityKind.AGENT

    @property
    def agent_id(self) -> str:
        """The id of the agent the alias names in this run."""
        return self.alias if self.store_version is not None else self.target_id


@dataclass(frozen=True)
class AgentOverride:
    """The agent a run deploys for the solver in place of the one each task names: at each step that deploys or
    installs an agent a step prompts, not as its user simulator (``solver_steps``), by task id."""

    name: str  # as the run was given it
    agent_id: str
    steps: Mapping[str, tuple[str, ...]]
    version: int | None = None  # the store's version the plan read; None for an agent the run writes


@dataclass(frozen=True)
class Write:
    """An entity or built image a run writes."""

    kind: BundleKind  # a built image is an ARTIFACT; kind.store keys uniqueness
    id: str
    source: ResolvedEntry | BundleImage
    needs: tuple[_Key, ...]  # (kind.store, id) of the earlier writes it references
    requires: tuple[str, ...] | None = None  # a task's plugin packages, which a registration records (Task.requires)


@dataclass(frozen=True)
class Plan:
    bundle: ResolvedBundle
    writes: tuple[Write, ...]  # everything the selection reaches, each after what it references
    tasks: tuple[ResolvedEntry, ...]  # the tasks selected to run on their own
    evals: tuple[ResolvedEntry, ...]
    store_refs: tuple[Reference, ...]  # the store ids the writes reference, checked to exist
    store_latest: dict[tuple[EntityKind, str], int]  # the version each unpinned store ref named when planned
    origins: Mapping[BundleEntry, Bundle] = field(default_factory=dict)  # each entry from another bundle, with it
    # Each agent image the bundle builds from a shipped agent's sources, by id: the image published for them, which the
    # run registers in place of its build context alone.
    published: Mapping[str, Publication] = field(default_factory=dict)
    # Each alias the run names, by the kind it names and then by alias.
    alias_uses: Mapping[EntityKind, Mapping[str, AliasUse]] = field(default_factory=dict)
    agent: AgentOverride | None = None  # the agent --agent names for the solver

    def bundle_of(self, entry: BundleEntry) -> Bundle:
        """The bundle ``entry`` comes from: the installed one for an entry pulled in from one, else the run's."""
        return self.origins.get(entry, self.bundle.bundle)

    def path(self, entry: BundleEntry) -> str:
        """``entry``'s path, as messages name it: in the run's bundle, or after the name of the bundle it comes from
        (``agentenv-framework/agents:agents/claude-code``)."""
        return _shown_path(self.bundle_of(entry), entry, pulled=entry in self.origins)


def plan_bundle(resolved: ResolvedBundle, *, tasks: Sequence[str] = (), evals: Sequence[str] = (),
                installed: Installed | None = None, agent: str | None = None, default_agents: bool = True) -> Plan:
    """Plan a run of ``tasks`` and ``evals``, each a name or id in this bundle. With neither, the
    run is every eval or, in a bundle without evals, every task. ``installed`` lists the installed bundles whose
    entities a reference pulls in (``installed_bundles``); without it, an id under one is a store id. ``agent`` names
    the agent the solver runs as, as a task's reference would, leaving the tasks as written (``Plan.agent``). Without
    ``default_agents``, a step that leaves its agent unset pulls in no configured default, as for a task written to
    run wherever another's default applies. Raises BundleError listing every problem found."""
    return _Planner(resolved, installed, default_agents).plan(tasks, evals, agent)


def plan_entities(resolved: ResolvedBundle, ids: Sequence[str], *, installed: Installed | None = None) -> Plan:
    """Plan writing the envs, agents and artifacts ``ids`` names in this bundle, with what they reference, and
    running nothing: what a run writes for one a task of another bundle names. Raises BundleError listing every
    problem found."""
    return _Planner(resolved, installed).plan_entities(ids)


def check_bundle(resolved: ResolvedBundle, *, installed: Installed | None = None) -> None:
    """The checks ``plan_bundle`` makes without reading a store, over every task and eval in the bundle. Raises
    BundleError listing every problem found."""
    _Planner(resolved, installed).check()


@dataclass(frozen=True)
class _Pullable:
    """An installed bundle, resolved: its entities by (kind, id), as a reference names one, each with its resolved
    entry or, for an image folder, its image."""

    resolved: ResolvedBundle
    by_id: dict[tuple[EntityKind, str], BundleEntry]
    by_entry: dict[BundleEntry, ResolvedEntry | BundleImage]


class _Planner:
    def __init__(self, resolved: ResolvedBundle, installed: Installed | None = None, default_agents: bool = True):
        self.resolved = resolved
        self.default_agents = default_agents
        self.problems: list[str] = []
        self.nodes: dict[_Key, ResolvedEntry | BundleImage] = {}
        self.rank: dict[_Key, int] = {}
        self.edges: dict[_Key, list[_Key]] = {}
        self.named = {fold(entry.entry.name): entry.entry for entry in resolved.entries}
        self.identified = {entry.entry.id: entry.entry for entry in resolved.entries}
        self.outputs = {(output.kind.value, output.id): (entry.entry, output)
                        for entry in resolved.entries for output in entry.outputs}
        self.accepted: dict[_Key, dict] = {}  # each toml a type checks, as it accepted it
        self.answers: dict[tuple[_Key, str], tuple | BundleError | None] = {}  # each artifact type's folder hooks
        self.steps: dict[_Key, list[TaskStep]] = {}  # each task's steps that build, as a write will build them
        self.origins: dict[BundleEntry, Bundle] = {}  # each entry from another bundle, with it
        self.installed = installed
        self.others: tuple[InstalledBundle, ...] | None = None  # the installed bundles but this one, once listed
        self.pullable: dict[str, _Pullable | str] = {}  # each installed bundle read, by id root, or why it can't be
        self.unpulled: dict[tuple[EntityKind, str, int | None], str] = {}  # why an installed bundle's id isn't pulled
        # By kind: its installed aliases, once a plan reads them; each of them the plan names, by alias; and the store's
        # latest version of each alias's id, as read.
        self.alias_tables: dict[EntityKind, InstalledAliases] = {}
        self.alias_uses: dict[EntityKind, dict[str, AliasUse]] = {kind: {} for kind in _ALIAS_TABLES}
        self.alias_stored: dict[EntityKind, dict[str, int | None]] = {kind: {} for kind in _ALIAS_TABLES}
        self.pulling: set[_Key] = set()
        self.own = _by_id(resolved.bundle)  # what a pulled entity's id names in this bundle
        images: dict[BundleEntry, list[BundleImage]] = {}
        for image in resolved.images:
            images.setdefault(image.entry, []).append(image)
        entries = {entry.entry: entry for entry in resolved.entries}
        self.entries: list[ResolvedEntry] = []  # the bundle's own, each reference to another bundle's entity local
        for bundle_entry in resolved.bundle.entries:
            for image in images.get(bundle_entry, ()):
                self._add(_key(image), image, [])
            if (entry := entries.get(bundle_entry)) is None:  # an image folder, which is its image
                continue
            entry = self._localized(entry)
            self.entries.append(entry)
            self._add(_key(entry.entry), entry, _needs(entry))

    def _add(self, key: _Key, source: ResolvedEntry | BundleImage, needs: list[_Key]) -> None:
        """Record a write in bundle order, a built image just before the entry that builds it."""
        self.nodes[key], self.rank[key], self.edges[key] = source, len(self.rank), needs

    # Entities from installed bundles

    def _localized(self, entry: ResolvedEntry) -> ResolvedEntry:
        """``entry`` with each reference to an installed bundle's entity pointing at it, pulled into the plan, as a
        reference to the bundle's own entity points at that; for a task, with a reference for each step that deploys
        the configured default agent when this bundle or an installed one defines it."""
        entry = self._pointed(entry)
        references = [ref if ref.local is not None else self._pulled(ref) for ref in entry.references]
        if entry.entry.kind is BundleKind.TASK and self.default_agents:
            references += self._default_agents(entry)
        if references == list(entry.references):
            return entry
        return replace(entry, references=tuple(references))

    def _default_agents(self, task: ResolvedEntry, agent_id: str | None = None) -> list[Reference]:
        """A reference to the configured default agent, or ``agent_id``, for each step of ``task`` that leaves an agent
        field unset, when this bundle or an installed one defines that agent. One in the store is preflight's to
        check."""
        registry = get_task_step_registry()
        found = []
        for step in task.config:
            cls = registry.get(step.get("type")) if isinstance(step.get("type"), str) else None
            for field_ref in default_agent_refs(cls.entity_refs or () if cls is not None else (), step):
                version = step.get(field_ref.version_field) if field_ref.version_field else None
                ref = Reference(EntityKind.AGENT, agent_id or get_config().get_default_a2a_agent_id(), version, None,
                                f"step {step.get('id')!r}: {field_ref.path} (the default agent)", None)
                pulled = self._pulled(ref)
                if pulled.local is not None or (ref.kind, ref.id, ref.version) in self.unpulled:
                    found.append(pulled)
        return found

    # Agents named by an installed bundle's alias

    def _resolve_aliases(self, *, runs: bool = True) -> None:
        """Point each entry's reference to an agent, env or artifact by an alias an installed bundle gives it, and each
        task's to the configured default agent when it's one, at what it names, pulled into the plan, unless the store
        has one of the alias's id, which the reference keeps naming. A field naming the alias is rewritten to the id. A
        reference that can't be pointed at the alias's entity is left with the problem the store ref check reports;
        ``alias_uses`` holds the aliases the plan names. Without ``runs``, as when entities are written alone, tasks
        and evals are left as they are, since none of them runs."""
        if self.installed is None:
            return
        installed = self.installed()
        self.alias_tables = {kind: installed_aliases(installed, table) for kind, table in _ALIAS_TABLES.items()}
        own = {_key(entry.entry) for entry in self.entries}
        for key, node in list(self.nodes.items()):  # those pulled in before the aliases were read
            if key not in own and isinstance(node, ResolvedEntry) and (pointed := self._pointed(node)) is not node:
                self.nodes[key], self.edges[key] = pointed, _needs(pointed)
        default = _UNREAD
        for index, entry in enumerate(self.entries):
            if not runs and entry.entry.kind in (BundleKind.TASK, BundleKind.EVAL):
                continue
            pointed = self._pointed(entry)
            references = list(pointed.references)
            if entry.entry.kind is BundleKind.TASK and self.default_agents and _leaves_an_agent_unset(entry):
                if default is _UNREAD:
                    default = self._alias_use(EntityKind.AGENT, get_config().get_default_a2a_agent_id(), None)
                    if isinstance(default, str):
                        self.problems.append(f"[agents] default_a2a_agent_id: {default}")
                if isinstance(default, AliasUse) and default.store_version is None:
                    references += self._default_agents(entry, default.target_id)
            if references != list(entry.references):
                self._replace(index, replace(pointed, references=tuple(references)))

    def _pointed(self, entry: ResolvedEntry) -> ResolvedEntry:
        """``entry`` with each store reference naming an installed alias pointed at what the alias names, pulled in, and
        each field naming the alias naming its id; none is pointed until ``_resolve_aliases`` reads the aliases."""
        references, aliased = [], {}
        for ref in entry.references:
            if ref.local is None and (use := self._alias_of(ref)) is not None:
                aliased[ref.kind, ref.id] = use.target_id
                ref = self._pulled(replace(ref, id=use.target_id))
            references.append(ref)
        if not aliased:
            return entry
        return replace(entry, config=_renamed(entry, aliased), references=tuple(references))

    def _alias_of(self, ref: Reference) -> AliasUse | None:
        """The installed alias ``ref``, a store reference, names, to point it at: None when it names none, or the store's
        entity of its id shadows it. Why an alias can't be named so is kept for the store ref check to report."""
        if ref.kind not in self.alias_tables:
            return None
        use = self._alias_use(ref.kind, ref.id, ref.version)
        if isinstance(use, str):
            self.unpulled[ref.kind, ref.id, ref.version] = use
            return None
        return use if use is not None and use.store_version is None else None

    def _agent_override(self, name: str) -> AgentOverride | None:
        """The agent ``name`` names, as a task's reference to it would: this bundle's agent, an installed bundle's
        pulled in, an alias's, or the store's, read once. Each task with solver steps gets a reference to it, so it's
        written and checked with what the task needs, and its config is left as written. Records a problem and gives
        None when ``name`` names no agent, or no task has a step for the solver."""
        where = f"--agent {name!r}"
        agents = [entry.entry for entry in self.resolved.entries if entry.entry.kind is BundleKind.AGENT]
        own = next((agent for agent in agents if _nfc(name) in (agent.name, agent.id)), None)
        near = next((agent for agent in agents if fold(name) in (fold(agent.name), fold(agent.id))), None)
        if own is None and near is not None:
            self.problems.append(f"{where}: isn't in this bundle, but {near.name!r} is; names are case-sensitive")
            return None
        if own is not None:
            ref = Reference(EntityKind.AGENT, own.id, None, own, where, None)
        else:
            ref = Reference(EntityKind.AGENT, name, None, None, where, None)
            use = self._alias_use(EntityKind.AGENT, name, None) if EntityKind.AGENT in self.alias_tables else None
            if isinstance(use, str):
                self.problems.append(f"{where}: {use}")
                return None
            ref = self._pulled(replace(ref, id=use.target_id) if use is not None and use.store_version is None else ref)
            if (problem := self.unpulled.get((ref.kind, ref.id, None)) or self._unstored(ref)) is not None:
                self.problems.append(f"{where}: {problem}")
                return None
        steps = {}
        for index, entry in enumerate(self.entries):
            if entry.entry.kind is BundleKind.TASK and (solver := solver_steps(entry.config)):
                steps[entry.entry.id] = solver
                self._replace(index, replace(entry, references=(*entry.references, ref)))
        if not steps:
            self.problems.append(f"{where}: no step in this bundle's tasks prompts an agent, so none deploys a solver")
            return None
        return AgentOverride(name, ref.local.id if ref.local is not None else ref.id, steps)

    def _unstored(self, ref: Reference) -> str | None:
        """Why the store has no agent ``ref`` names, a store ref, or None when it has one or ``ref`` is local."""
        if ref.local is not None:
            return None
        try:
            STORE_GETTERS[EntityKind.AGENT](ref.id, None)
        except NotFoundError:
            hint = table.hint(ref.id) if (table := self.alias_tables.get(EntityKind.AGENT)) is not None else ""
            return f"there is no agent {ref.id!r} in this bundle, an installed bundle or the store{hint}"
        except (ValueError, KeyError, TypeError) as e:
            return f"agent {ref.id!r} can't be read ({type(e).__name__}: {e})"
        return None

    def _alias_use(self, kind: EntityKind, id: str, version: int | None) -> AliasUse | str | None:
        """What ``id``, naming one of ``kind``, names when it's an installed alias, the store's of that id read first
        (its latest version, or None, cached in ``alias_stored``), recorded in ``alias_uses``; why it can't be named
        so; or None when it's no alias, or names the store's as any store reference does: pinned, or with the alias
        refused."""
        if id.startswith(LOCAL_PREFIX):
            return None
        table, uses, stored = self.alias_tables[kind], self.alias_uses[kind], self.alias_stored[kind]
        refusal, named = table.refusal(id), table.target(id)
        if refusal is None and named is None:
            return None
        if id not in stored:
            try:
                stored[id] = STORE_GETTERS[kind](id, None).version
            except NotFoundError:
                stored[id] = None
            except (ValueError, KeyError, TypeError):  # one the store has but can't read, which the store ref check says
                return None
        if stored[id] is not None and (refusal is not None or version is not None):
            return None
        if refusal is not None:
            return refusal
        if version is not None:
            return (f"{id!r} is an alias of {named.target_id}, which comes with {named.declared_by}, so it has no "
                    "fixed version; drop the pin")
        return uses.setdefault(id, AliasUse(id, named.target_id, named.declared_by, stored[id], kind))

    def _replace(self, index: int, entry: ResolvedEntry) -> None:
        """Put ``entry`` in place of the bundle's own entry at ``index``, with what it now references."""
        key = _key(entry.entry)
        self.entries[index] = self.nodes[key] = entry
        self.edges[key] = _needs(entry)

    def _pulled(self, ref: Reference) -> Reference:
        """``ref``, a store reference, pointing at what its id names: this bundle's entity, for a reference a pulled
        entity or a default makes, or an installed bundle's, pulled into the plan. One naming neither stays a store
        reference, as does one that can't be pulled, whose problem the store ref check reports."""
        if ref.kind not in _PULLED:
            return ref
        if (own := self.own.get((ref.kind, ref.id))) is not None:
            target, problem = own, _misnamed(own, ref, "this bundle's", "is defined in this bundle")
        elif ref.id.startswith(LOCAL_PREFIX) and (installed := self._installed_for(ref.id)) is not None:
            target, problem = self._target(installed, ref)
        else:
            return ref
        if problem is not None:
            self.unpulled[ref.kind, ref.id, ref.version] = problem
            return ref
        return replace(ref, local=target)

    def _installed_for(self, entity_id: str) -> InstalledBundle | None:
        """The installed bundle, other than this one, whose ids ``entity_id`` is under."""
        if self.installed is None:
            return None
        if self.others is None:
            self.others = tuple(bundle for bundle in self.installed() if bundle.id_root != self.resolved.bundle.id_root)
        return next((bundle for bundle in self.others if entity_id.startswith(f"{bundle.id_root}/")), None)

    def _target(self, installed: InstalledBundle, ref: Reference) -> tuple[BundleEntry | None, str | None]:
        """What ``ref`` names in ``installed``, pulled into the plan, or why it names nothing there."""
        name = installed.qualified
        pullable = self._pullable(installed)
        if isinstance(pullable, str):
            return None, f"{ref.id!r} comes with the installed bundle {name}, which can't be read: {pullable}"
        target = pullable.by_id.get((ref.kind, ref.id))
        if target is None:
            other = next((found for (kind, id), found in pullable.by_id.items() if id == ref.id), None)
            if other is not None:
                return None, (f"{ref.id!r} is the installed bundle {name}'s {other.kind.store}, but this field takes "
                              f"{with_article(ref.kind.value)}")
            return None, f"there is no {ref.kind.value} {ref.id!r} in the installed bundle {name}"
        if problem := _misnamed(target, ref, f"the installed bundle {name}'s", f"comes with the installed bundle {name}"):
            return None, problem
        self._pull(pullable, target)
        return target, None

    def _pullable(self, installed: InstalledBundle) -> _Pullable | str:
        """``installed`` parsed and resolved, once, or the problem that stops it."""
        if installed.id_root not in self.pullable:
            if installed.problem or installed.root is None:
                self.pullable[installed.id_root] = installed.problem or "it has no folder"
            else:
                try:
                    resolved = resolve_bundle(parse_bundle(installed.root, id_root=installed.id_root))
                except BundleError as e:
                    self.pullable[installed.id_root] = e.summary
                else:
                    sources: dict[BundleEntry, ResolvedEntry | BundleImage] = {
                        image.entry: image for image in resolved.images if image.entry.id == image.id}
                    sources.update({entry.entry: entry for entry in resolved.entries})
                    self.pullable[installed.id_root] = _Pullable(resolved, _by_id(resolved.bundle), sources)
        return self.pullable[installed.id_root]

    def _pull(self, pullable: _Pullable, target: BundleEntry | BundleImage) -> None:
        """Add ``target``, an installed bundle's entity or image, to the plan as the bundle's own are added, with what
        it references in its bundle and, through its store references, in other installed bundles."""
        key = _key(target)
        if key in self.nodes or key in self.pulling:
            return
        source = target if isinstance(target, BundleImage) else pullable.by_entry[target]
        self.origins[source.entry] = pullable.resolved.bundle
        if isinstance(source, BundleImage):  # an image key's, or an image folder, which is its image
            self._add(key, source, [])
            return
        self.pulling.add(key)
        for ref in source.references:
            if ref.local is not None:
                self._pull(pullable, ref.local)
        source = self._localized(source)
        self.pulling.discard(key)
        self._add(key, source, _needs(source))

    def plan(self, tasks: Sequence[str], evals: Sequence[str], agent: str | None = None) -> Plan:
        self._resolve_aliases()
        override = self._agent_override(agent) if agent is not None else None
        self._check_writers()
        self._check_loops()
        selected_tasks, selected_evals = self._selection(tasks, evals)
        return self._planned(selected_tasks + selected_evals, selected_tasks, selected_evals, override)

    def plan_entities(self, ids: Sequence[str]) -> Plan:
        self._resolve_aliases(runs=False)
        self._check_writers()
        self._check_loops()
        return self._planned(self._select_entities(ids), [], [], None)

    def _planned(self, roots: list[ResolvedEntry], selected_tasks: list[ResolvedEntry],
                 selected_evals: list[ResolvedEntry], override: AgentOverride | None) -> Plan:
        closure = self._closure(roots)
        store_refs, store_latest = self._check_store_refs(closure)
        if override is not None:
            override = replace(override, version=store_latest.get((EntityKind.AGENT, override.agent_id)))
        self._build_tasks(closure)
        self._check_files(closure)
        self._check_cli_loads(closure)
        self._check_env_types(closure)
        self._check_multi_names(closure, store_latest)
        self._check_environment_names(closure, store_latest)
        self._check_loaded_environments(closure, store_latest)
        if self.problems:
            raise BundleError(self.problems)
        writes = self._ordered(closure)
        return Plan(self.resolved, writes, tuple(selected_tasks), tuple(selected_evals), store_refs, store_latest,
                    dict(self.origins), self._published(writes),
                    {kind: dict(uses) for kind, uses in self.alias_uses.items()}, override)

    def check(self) -> None:
        """The checks a run makes before it reads a store. A bundle with nothing to run passes only when its
        aliases.toml gives what it holds aliases: other bundles pull from it, as from a library."""
        self._check_writers()
        self._check_loops()
        runs = any(entry.entry.kind in (BundleKind.TASK, BundleKind.EVAL) for entry in self.entries)
        if not runs and not any(self.resolved.bundle.aliases.values()):
            self.problems.append(_NOTHING_TO_RUN)
        everything = set(self.nodes)
        self._check_store_refs(everything, stores=False)
        self._build_tasks(everything)
        self._check_files(everything)
        self._check_cli_loads(everything)
        self._check_environment_names(everything, None)
        self._check_loaded_environments(everything, None)
        if self.problems:
            raise BundleError(self.problems)

    # Checks over the whole bundle

    def _check_writers(self) -> None:
        """Every id is written by one entity, built image or task output, per store."""
        writers: dict[_Key, str] = {}
        claims: list[tuple[_Key, str, str]] = []
        for key, source in self.nodes.items():
            if isinstance(source, BundleImage):
                how = "built" if isinstance(source, BuiltImage) else "referenced"
                claims.append((key, self._path(source.entry), f"the image {how} for {self._path(source.entry)}"))
            else:
                claims.append((key, self._path(source.entry), self._path(source.entry)))
                for id, folder in self._written_beside(source):
                    claims.append(((BundleKind.ARTIFACT.store, id), folder, f"the artifact written from {folder}"))
        for entry in self.entries:
            for output in entry.outputs:
                claims.append(((output.kind.value, output.id), f"{self._path(entry.entry)}: step {output.step_id!r}",
                               f"step {output.step_id!r} of {self._path(entry.entry)}"))
        for key, where, writer in claims:
            if key in writers:
                self.problems.append(f"{where}: {key[1]!r} is also written by {writers[key]}; give each its own id")
            else:
                writers[key] = writer

    def _written_beside(self, source: ResolvedEntry) -> list[tuple[str, str]]:
        """The ids an artifact writes besides its own, each with the folder it's written from. A folder that can't
        be written names none; the file check reports why."""
        children, where = self._children(source.entry), self._path(source.entry)
        return [(child.id, f"{where}/{child.path}") for child in children or ()]

    def _children(self, entry: BundleEntry) -> tuple[FolderChild, ...] | None:
        """What the artifact ``entry`` writes besides itself (``written_besides``): None for one its type doesn't
        write from a folder, or whose folder the file check refuses."""
        children = self._answer(entry, "written_besides")
        return None if isinstance(children, BundleError) else children

    def _folder_environments(self, entry: BundleEntry) -> tuple[tuple[str, str], ...] | None:
        """``(path in the folder, environment_name)`` of each environment the artifact ``entry`` seeds, as a type
        declaring SeedsEnvironments answers ``seeded_environments`` (the path ``""`` for the folder itself), and none
        for another type: None for one its type doesn't write from a folder, or whose folder the file check
        refuses."""
        held = self._answer(entry, "seeded_environments")
        return None if isinstance(held, BundleError) else held

    def _answer(self, entry: BundleEntry, hook: str) -> tuple | BundleError | None:
        """What the artifact ``entry``'s type answers to ``hook``, one of ``_FOLDER_HOOKS``, for its folder, read
        once: () for a type not declaring it, None for one its type doesn't write from a folder, or the BundleError the
        hook raised."""
        if entry.kind not in (BundleKind.ARTIFACT, BundleKind.SKILL):
            return None
        key = (_key(entry), hook)
        if key not in self.answers:
            cls = get_artifact_registry().get(canonical_type(entry.type))
            try:
                if not artifact_writer(cls):
                    self.answers[key] = None
                elif not issubclass(cls, _FOLDER_HOOKS[hook]):
                    self.answers[key] = ()
                else:
                    self.answers[key] = getattr(cls, hook)(self._ctx(entry))
            except BundleError as e:
                self.answers[key] = e
        return self.answers[key]

    def _check_loops(self) -> None:
        for component in _components(self.nodes, self.edges, self.rank):
            start = min(component, key=self.rank.__getitem__)
            loop = " -> ".join(self._path(self.nodes[key].entry) for key in _loop(start, set(component), self.edges))
            self.problems.append(f"{self._path(self.nodes[start].entry)}: references form a loop: {loop}")

    # Selection

    def _selection(self, tasks: Sequence[str], evals: Sequence[str]) -> tuple[list[ResolvedEntry], list[ResolvedEntry]]:
        if tasks or evals:
            return self._select(BundleKind.TASK, tasks), self._select(BundleKind.EVAL, evals)
        every_task = [entry for entry in self.entries if entry.entry.kind is BundleKind.TASK]
        every_eval = [entry for entry in self.entries if entry.entry.kind is BundleKind.EVAL]
        if every_eval:
            return [], every_eval
        if not every_task:
            self.problems.append(_NOTHING_TO_RUN)
        return every_task, []

    def _select(self, kind: BundleKind, wanted: Sequence[str]) -> list[ResolvedEntry]:
        candidates = [entry for entry in self.entries if entry.entry.kind is kind]
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

    def _select_entities(self, ids: Sequence[str]) -> list[ResolvedEntry]:
        entities = {_nfc(entry.entry.id): entry for entry in self.entries
                    if entry.entry.kind not in (BundleKind.TASK, BundleKind.EVAL)}
        for entity_id in ids:
            if _nfc(entity_id) not in entities:
                self.problems.append(f"this bundle has no env, agent or artifact {entity_id!r}")
        return [entities[key] for key in dict.fromkeys(map(_nfc, ids)) if key in entities]

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
            if isinstance(source, BundleImage):
                continue
            for ref in source.references:
                if ref.local is not None:
                    continue
                problem = self.unpulled.get((ref.kind, ref.id, ref.version)) or self._bundle_ref_problem(ref)
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
                read[lookup] = STORE_GETTERS[ref.kind](ref.id, ref.version)
            except NotFoundError:
                read[lookup] = None
            except (ValueError, KeyError, TypeError) as e:
                read[lookup] = f"can't be read ({type(e).__name__}: {e})"
        found = read[lookup]
        if found is None:
            at = f" version {ref.version}" if ref.version is not None else ""
            other = self.named.get(fold(ref.id))
            if other is not None:
                hint = f"; this bundle's {other.kind.store} {other.name!r} has that name"
            else:
                hint = table.hint(ref.id) if (table := self.alias_tables.get(ref.kind)) is not None else ""
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
            if isinstance(source, BundleImage):
                continue
            if source.entry.kind in (BundleKind.AGENT, BundleKind.ENV):
                agent = source.entry.kind is BundleKind.AGENT
                accept = A2AAgent.accept_toml if agent else self._env_accept(source.entry)
                try:
                    if accept is not None:
                        self.accepted[key] = accept(source.config, self._ctx(source.entry))
                except BundleError as e:
                    self.problems.extend(self._named(source.entry, e.problems))
                continue
            if source.entry.kind not in (BundleKind.ARTIFACT, BundleKind.SKILL):
                continue
            cls = registry.get(canonical_type(source.entry.type))
            if not artifact_writer(cls):
                continue
            try:
                folder_files(self._bundle_of(source.entry), source.entry)
            except BundleError as e:
                self.problems.extend(self._named(source.entry, e.problems))
            ctx = self._ctx(source.entry)
            accept = getattr(cls, "accept_toml", None)
            try:
                if accept is None:
                    ctx.accept(source.config, **cls.toml_keys)
                else:
                    self.accepted[key] = accept(source.config, ctx)
            except BundleError as e:
                self.problems.extend(self._named(source.entry, e.problems))
            for hook in _FOLDER_HOOKS:
                if isinstance(refused := self._answer(source.entry, hook), BundleError):
                    # What only the hook refuses, less what the type's accept_toml already reported.
                    named = self._named(source.entry, refused.problems)
                    self.problems.extend([problem for problem in named if problem not in self.problems])

    def _check_cli_loads(self, closure: set[_Key]) -> None:
        """Each load of one of the bundle's CLIs names what installing it needs, an agent and an env whose gateway it
        reads, and no two of a task's loads install one command into one agent, where the second would replace the
        first."""
        registry = get_artifact_registry()
        clis = {source.entry.id: source.entry for source in self.nodes.values()
                if isinstance(source, ResolvedEntry) and source.entry.kind is BundleKind.ARTIFACT
                and issubclass(registry.get(canonical_type(source.entry.type)) or object, CliArtifact)}
        for key in sorted(closure, key=self.rank.__getitem__):
            installed: dict[tuple[str, str], tuple[Any, str]] = {}
            for step in self.steps.get(key, ()):
                if not isinstance(step, LoadArtifactTaskStep):
                    continue
                loaded = [clis[ref["id"]] for ref in step.artifacts if ref.get("id") in clis]
                if not loaded:
                    continue
                where = f"{self._path(self.nodes[key].entry)}: step {step.id!r}"
                if missing := [name for name in ("agent_name", "env_id") if not getattr(step, name)]:
                    names, it = " and ".join(repr(cli.name) for cli in loaded), "them" if len(loaded) > 1 else "it"
                    self.problems.append(f"{where}: loading the CLI{'s' * (it == 'them')} {names} installs {it} into "
                                         f"an agent and points {it} at an env's gateway, so the step needs "
                                         f"{' and '.join(missing)}")
                    continue
                for cli in loaded:
                    command = self.accepted.get(_key(cli), {}).get("command_name")
                    if command is None:
                        continue  # its artifact.toml's problem is reported by the file check
                    if (step.agent_name, command) in installed:
                        first, name = installed[step.agent_name, command]
                        self.problems.append(f"{where}: installs the CLI {cli.name!r} as {command!r} in agent "
                                             f"{step.agent_name!r}, where step {first!r} already installed {name!r} "
                                             "under that command; give one another command_name")
                    else:
                        installed[step.agent_name, command] = (step.id, cli.name)

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
            if isinstance(source, BundleImage) or source.entry.kind is not BundleKind.ENV:
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
            if isinstance(source, BundleImage) or source.entry.kind is not BundleKind.ENV:
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
            provider_type = source.config.get("env_provider_type", EnvironmentGatewayProvider.type)
            try:
                provider = _env_provider_class(provider_type)
            except ValueError:
                continue  # the env's toml check refuses a type no installed provider has
            if provider_refusal(provider, MultiEnv) is not None:
                continue  # and one that deploys no multi at all
            reason = provider_refusal(provider, MultiEnv, mcp_names=named.get("mcp_server_envs", ()),
                                      website_names=named.get("website_envs", ()))
            if reason is not None:
                self.problems.append(f"{where}: env_provider_type {provider_type!r} {reason}; rename one")

    def _check_environment_names(self, closure: set[_Key],
                                 store_latest: dict[tuple[EntityKind, str], int] | None) -> None:
        """The environments one artifact holds need names of their own: those its type names
        (``seeded_environments``), and the environment artifacts it references, named by their tomls or, from the
        store, at the version the plan read. Without ``store_latest`` (a check that reads no store), the store's
        environments aren't counted."""
        for key in sorted(closure, key=self.rank.__getitem__):
            source = self.nodes[key]
            if isinstance(source, BundleImage) or (environments := self._folder_environments(source.entry)) is None:
                continue
            where = self._path(source.entry)
            held = [*((_within(where, path), name) for path, name in environments),
                    *((f"{where}: {ref.where}", name) for ref, name in self._referenced(source, store_latest))]
            taken: dict[str, str] = {}  # who has each name
            for who, name in held:
                if name in taken:
                    self.problems.append(f"{who}: environment_name {name!r} is also {taken[name]}'s; "
                                         f"{with_article(source.entry.type)}'s environments need names of their own")
                else:
                    taken[name] = who

    def _referenced(self, source: ResolvedEntry,
                    store_latest: dict[tuple[EntityKind, str], int] | None) -> list[tuple[Reference, str]]:
        """Each environment artifact ``source`` references, with its environment_name when that's known."""
        return [(ref, name) for ref in source.references
                if ref.kind is EntityKind.ARTIFACT and ref.artifact_type is not None
                and canonical_type(ref.artifact_type) == _ENVIRONMENT
                and (name := self._environment_name(ref, store_latest)) is not None]

    def _check_loaded_environments(self, closure: set[_Key],
                                   store_latest: dict[tuple[EntityKind, str], int] | None) -> None:
        """Each environment the bundle writes seeds a server of the env a task loads it into: one named after no
        server is a typo or a stray, which a multi would skip and an MCP server or website would fail on. A store
        environment or universe isn't checked, since a server loading a shared universe takes its own environment
        from it. Without ``store_latest`` (a check that reads no store), a store env's servers aren't known."""
        registry = get_task_step_registry()
        for key in sorted(closure, key=self.rank.__getitem__):
            source = self.nodes[key]
            if not (isinstance(source, ResolvedEntry) and source.entry.kind is BundleKind.TASK):
                continue
            loads: dict[str, list[Reference]] = {}  # each load_artifact step's references, by the step's label
            for step in source.config:
                cls = registry.get(step["type"]) if isinstance(step.get("type"), str) else None
                if cls is not None and issubclass(cls, LoadArtifactTaskStep):
                    loads[f"step {step.get('id')!r}"] = []
            for ref in source.references:
                label = ref.where.partition(": ")[0]
                if label in loads:
                    loads[label].append(ref)
            for label, refs in loads.items():
                env = next((ref for ref in refs if ref.kind is EntityKind.ENV), None)
                servers = self._server_names(env, store_latest) if env is not None else None
                if servers is None:
                    continue
                into = self._path(env.local) if env.local is not None else f"env {env.id!r}"
                for ref in refs:
                    for what, name in self._seeded(ref, store_latest):
                        if name not in servers:
                            listed = ", ".join(repr(server) for server in sorted(servers))
                            self.problems.append(
                                f"{self._path(source.entry)}: {label}: {what} seeds an environment named {name!r}, "
                                f"and {into} has no server by that name ({listed}); name it after the server it "
                                "seeds, or leave it out")

    def _server_names(self, ref: Reference, store_latest: dict[tuple[EntityKind, str], int] | None) -> set[str] | None:
        """The environment_names of the servers the env ``ref`` names runs: an MCP server's or a website's own, each
        of a multi's envs', or a computer's own and each of its apps'. None when they aren't all known."""
        if isinstance(ref.local, BundleEntry):
            cls = get_env_registry().get(ref.local.type)
            children = self.nodes[_key(ref.local)].references
            if cls is not None and issubclass(cls, MultiEnv):
                names = {self._environment_name(child, store_latest) for child in children
                         if child.kind is EntityKind.ENV}
            else:
                names = {self.accepted.get(_key(ref.local), {}).get("environment_name")}
            if cls is not None and issubclass(cls, ComputerEnv):
                names |= {self._environment_name(child, store_latest) for child in children
                          if child.kind is EntityKind.ENV and child.artifact_type == ComputerAppEnv.type}
            return None if None in names else names
        if ref.local is not None or store_latest is None:
            return None
        try:
            env = Env.get(ref.id, ref.version if ref.version is not None else store_latest.get((ref.kind, ref.id)))
        except (NotFoundError, ValueError, KeyError, TypeError):
            return None  # reported by the store ref check
        if isinstance(env, MultiEnv):
            return {child.environment_name for child in [*env.mcp_server_envs, *env.website_envs]}
        if isinstance(env, ComputerEnv):
            return {env.environment_name, *(app.environment_name for app in env.composed_apps())}
        name = getattr(env, "environment_name", None)
        return {name} if name else None

    def _seeded(self, ref: Reference, store_latest: dict[tuple[EntityKind, str], int] | None) -> list[tuple[str, str]]:
        """Each environment the bundle's own artifact ``ref`` names seeds, as (where it's named, its
        environment_name): the ones its type names (``seeded_environments``) and the environment artifacts it
        references. Nothing for a store artifact, or one whose folder the file check refuses."""
        entry = ref.local
        if not isinstance(entry, BundleEntry) or (environments := self._folder_environments(entry)) is None:
            return []
        where = self._path(entry)
        return [*((_within(where, path), name) for path, name in environments),
                *((f"{where}: {named.where}", name) for named, name in self._referenced(self.nodes[_key(entry)],
                                                                                       store_latest))]

    def _environment_name(self, ref: Reference, store_latest: dict[tuple[EntityKind, str], int] | None) -> str | None:
        """The environment_name of the env or environment artifact ``ref`` names: a bundle env's from its accepted
        toml, a bundle artifact's as its type names it (``seeded_environments``), a store one's at the version the
        plan read, or None when that's unknown."""
        if isinstance(ref.local, BundleEntry) and ref.kind is EntityKind.ARTIFACT:
            return next((name for path, name in self._folder_environments(ref.local) or () if not path), None)
        if ref.local is not None:
            return self.accepted.get(_key(ref.local), {}).get("environment_name")
        if store_latest is None:
            return None
        try:
            entity = STORE_GETTERS[ref.kind](ref.id, ref.version if ref.version is not None
                                             else store_latest.get((ref.kind, ref.id)))
        except (NotFoundError, ValueError, KeyError, TypeError):
            return None  # reported by the store ref check
        return getattr(entity, "environment_name", None)

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
            self.steps[key] = steps
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
            kind = BundleKind.ARTIFACT if isinstance(source, BundleImage) else source.entry.kind
            writes.append(Write(kind, key[1], source, tuple(self.edges[key])))
            for dependent in dependents.get(key, ()):
                waiting[dependent] -= 1
                if waiting[dependent] == 0:
                    heapq.heappush(ready, (self.rank[dependent], dependent))
        return tuple(writes)

    def _published(self, writes: Sequence[Write]) -> dict[str, Publication]:
        """The published image of each image in ``writes`` built from the sources of a shipped bundle image, an agent's,
        an example env's or a base computer's, by id."""
        published = {}
        for write in writes:
            image = write.source
            if not isinstance(image, BuiltImage):
                continue
            with staged_build_context(self._bundle_of(image.entry), image.entry, image.dockerfile) as staged:
                try:
                    sources, _ = staged.identity(write.id)
                except ValueError:  # a context its write refuses, which says why
                    continue
            if (found := published_bundle_image(sources)) is not None:
                published[write.id] = Publication(*found, sources)
        return published

    def _bundle_of(self, entry: BundleEntry) -> Bundle:
        return self.origins.get(entry, self.resolved.bundle)

    def _ctx(self, entry: BundleEntry) -> AuthoringContext:
        return AuthoringContext(self._bundle_of(entry), entry)

    def _path(self, entry: BundleEntry) -> str:
        return _shown_path(self._bundle_of(entry), entry, pulled=entry in self.origins)

    def _named(self, entry: BundleEntry, problems: Sequence[str]) -> list[str]:
        """``problems``, which name paths in ``entry``'s bundle, as the plan names them: after the name of the bundle
        an entry pulled in comes from."""
        if entry not in self.origins:
            return list(problems)
        return [f"{_bundle_name(self.origins[entry])}:{problem}" for problem in problems]


def unpinned_store_refs(plan: Plan, write: Write) -> dict[tuple[EntityKind, str], int]:
    """The version the plan read for each store entity ``write`` names without a version, by (kind, id). Its writer
    pins each to that version and the ledger hashes them, so a version the store gains in between isn't written,
    and one it gains later rewrites ``write``."""
    if isinstance(write.source, BundleImage):
        return {}
    return {(ref.kind, ref.id): plan.store_latest[ref.kind, ref.id]
            for ref in write.source.references if ref.local is None and ref.version is None}


def env_writer(cls: type | None) -> bool:
    """Whether an env of type ``cls`` is written from its env.toml, by a ``from_toml`` that reads only the toml and
    what it names: its own, or one it inherits from a core env type."""
    return cls is not None and next(c for c in cls.__mro__ if "from_toml" in vars(c)) in _ENV_FROM_TOMLS


def env_folder_writer(cls: type | None) -> bool:
    """Whether an env of type ``cls`` is written from every file in its folder, as well as its env.toml."""
    return cls is not None and next(c for c in cls.__mro__ if "from_toml" in vars(c)) in _ENV_FROM_FOLDERS


def artifact_writer(cls: type | None) -> bool:
    """Whether an artifact of type ``cls`` is written from its bundle folder: by a ``from_toml`` of its own or one it
    inherits from a type that has one, not the base's, which writes a document and no folder."""
    return cls is not None and next(c for c in cls.__mro__ if "from_toml" in vars(c)) is not Artifact


def folder_files(bundle: Bundle, entry: BundleEntry) -> Mapping[str, Path]:
    """What the write of the artifact ``entry`` reads from its folder, which the ledger hashes: every file in it, none
    at all included; a type whose write needs a file says so in its ``accept_toml``. Raises BundleError."""
    return entry_files(bundle, entry, empty_ok=True)


_UNREAD: Any = object()  # the configured default agent, before a task needs it


def solver_steps(steps: Any) -> tuple[str, ...]:
    """The ids of the steps in ``steps``, a task's config, that deploy or install an agent a ``prompt_agent`` step
    prompts, and not as its user simulator: the solver's."""
    prompts = [step for step in steps if isinstance(step, dict) and step.get("type") == PromptAgentTaskStep.type]
    prompted = {step.get("agent_name") or TaskStep.DEFAULT_AGENT_NAME for step in prompts}
    prompted -= {step.get("user_agent_name") or PromptAgentTaskStep.DEFAULT_USER_AGENT_NAME for step in prompts}
    return tuple(step["id"] for step in steps if isinstance(step, dict) and isinstance(step.get("id"), str)
                 and step.get("type") in (DeployAgentTaskStep.type, InstallAgentTaskStep.type)
                 and (step.get("agent_name") or TaskStep.DEFAULT_AGENT_NAME) in prompted)


def _leaves_an_agent_unset(task: ResolvedEntry) -> bool:
    """Whether a step of ``task`` leaves an agent field unset, so deploys the configured default agent."""
    registry = get_task_step_registry()
    for step in task.config:
        cls = registry.get(step.get("type")) if isinstance(step.get("type"), str) else None
        if any(default_agent_refs(cls.entity_refs or () if cls is not None else (), step)):
            return True
    return False


def _renamed(entry: ResolvedEntry, ids: Mapping[tuple[EntityKind, str], str]) -> Any:
    """A copy of ``entry``'s config with each field naming an alias in ``ids``, by kind, naming what it names instead:
    a task's steps by their types' refs, any other entry's toml by its class's ``toml_refs``."""
    if not ids:
        return entry.config
    config = copy.deepcopy(entry.config)
    if entry.entry.kind is BundleKind.TASK:
        registry = get_task_step_registry()
        sites = [site for step in config
                 for site in ref_sites(getattr(registry.get(step.get("type")) if isinstance(step.get("type"), str)
                                               else None, "entity_refs", None) or (), step)]
    else:
        sites = list(ref_sites(getattr(entity_class(entry.entry), "toml_refs", ()), config, inline_pins=True))
    for site in sites:
        if site.ref.role is RefRole.INPUT and (site.ref.kind, site.value) in ids:
            site.rewrite(ids[site.ref.kind, site.value])
    return config


def _needs(entry: ResolvedEntry) -> list[_Key]:
    """The writes ``entry`` references, each once."""
    return list(dict.fromkeys(_key(ref.local) for ref in entry.references if ref.local is not None))


def _by_id(bundle: Bundle) -> dict[tuple[EntityKind, str], BundleEntry]:
    """``bundle``'s envs, agents and artifacts by (kind, id), image folders included, as a reference names one."""
    return {(kind, entry.id): entry for entry in bundle.entries for kind in _PULLED if entry.kind in NAMED_BY[kind]}


def _misnamed(target: BundleEntry, ref: Reference, whose: str, comes: str) -> str | None:
    """Why ``ref`` can't name ``target``, an entity a run writes: a pin, or a type its field doesn't take."""
    if ref.version is not None:
        return f"{ref.id!r} {comes}, so it has no fixed version; drop the pin"
    if ref.artifact_type and canonical_type(target.type) != canonical_type(ref.artifact_type):
        return f"{ref.id!r} is {whose} {target.type}, but this field takes {ref.artifact_type}"
    return None


def _shown_path(bundle: Bundle, entry: BundleEntry, *, pulled: bool) -> str:
    where = relative(bundle.root, entry.path)
    return f"{_bundle_name(bundle)}:{where}" if pulled else where


def _bundle_name(bundle: Bundle) -> str:
    """An installed bundle's name as messages give it, ``<dist>/<bundle>``: its id root, less the namespace."""
    return bundle.id_root.removeprefix(LOCAL_PREFIX)


def _within(where: str, path: str) -> str:
    """Where ``path`` in the folder at ``where`` is, ``""`` being the folder itself."""
    return f"{where}/{path}" if path else where


def _key(entity: BundleEntry | BundleImage) -> _Key:
    kind = BundleKind.ARTIFACT if isinstance(entity, BundleImage) else entity.kind
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
