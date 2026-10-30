"""Register a bundle folder into the configured stores under registry ids, so whatever runs a store's tasks runs what the
folder ran.

A registration plans every task and eval in the folder, or those selected, as a run plans them, but without the
configured default agent a step that leaves its agent unset falls back to: the registered task leaves it unset, so
whatever runs it takes its own default. Each ``@local`` id the plan writes then takes a registry id (``registry_ids``),
and each task and eval is written naming the version of each entity it references, so a registered version never
changes meaning. Each image the bundle builds is built here for linux/amd64 and pushed to the configured image store
under a tag of its sources, with its build context beside it; one published for its sources is registered as published.

Nothing is written when any problem is found before the first write: an id the rules can't give, two writes to one store
given one id, an ``@local`` id left in what would be written, a store reference the store doesn't hold, or, for a store
that isn't this machine's, a credential written out or a URL that only reaches this machine. A dry run checks the same
and writes, builds and pushes nothing. Each write goes through a ledger kept in the target store, so registering again
writes only what changed. Two registrations racing for one id can't both write a version of it: the loser fails, and
running it again reuses what the winner wrote.
"""

from __future__ import annotations

import re
import subprocess
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agent_env.config import get_config
from agent_env.entity.refs import RefRole, ref_sites
from agent_env.plugins._bundles import AGENT_ALIASES
from agent_env.providers import build_sandbox_provider
from agent_env.store.document_store import DocumentStore, LocalSqliteDocumentStore
from agent_env.store.ids import LOCAL_PREFIX, is_local_id
from agent_env.store.image_store.oci_registry_credentials import is_loopback_host
from agent_env.store.routing import configured_store, namespace_routing
from agent_env.task_step.registry import get_task_step_registry

from ._secrets import credentials_in
from .installed import installed_bundles
from .ledger import Ledger
from .materialize import Materialization, Materialized, materialize
from .parse import Bundle, BundleEntry, BundleError, BundleKind, parse_bundle
from .plan import AliasUse, Plan, Write, plan_bundle, plan_entities
from .requires import Declaring, Providers, declare, declaring_for, detected, entry_uses
from .run import _label
from .resolve import BundleImage, Reference, ResolvedBundle, ResolvedEntry, resolve_bundle

REGISTRATIONS = "bundle_registrations"
# A registry id: what a hub route, an object key and an image repository all take.
_REGISTRY_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_NOT_IN_ID = re.compile(r"[^a-z0-9._-]+")
_URL = re.compile(r"\b(?:file:///?[^\s\"']*|[a-z][a-z0-9+.-]*://[^\s/\"']+)", re.IGNORECASE)
_GIT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class Registration:
    """What registering a bundle wrote, or in a dry run would write, under which registry ids."""

    plan: Plan  # under the registry ids
    materialization: Materialization
    ids: Mapping[str, str]  # each ``@local`` id the plan writes or its tasks' outputs take, and its registry id
    floating: tuple[str, ...]  # each reference written without a version, so whatever runs it reads its latest
    bundle: str  # the bundle's identity, which its ledger rows name
    shared: bool  # the target is a store other than this machine's own
    requires: tuple[str, ...]  # what to say of the plugin packages it uses that its bundle.toml doesn't require


def register_bundle(
    root: Path | str,
    *,
    tasks: Sequence[str] = (),
    evals: Sequence[str] = (),
    prefix: str | None = None,
    dry_run: bool = False,
    declaring: Declaring | None = None,
    on_progress: Callable[[str], None] | None = None,
    sandbox: str | None = None,
) -> Registration:
    """Register the bundle at ``root``: ``tasks`` and ``evals`` by name, or with neither every task and eval it has,
    or every entity when it has none, with what they reference. ``prefix`` opens each entry's registry id, the folder's
    name and a dash unless given. An id the store holds already takes a new version, whoever wrote it. ``on_progress``
    is given a line for each write, as it's checked or written, and before each image is built. A folder with no
    bundle.toml that uses plugin packages is given one, or told what to declare, as ``declaring`` says: by default as a
    run of the folder would (``declaring_for``). ``sandbox`` is the provider its computer envs compose on, by default
    the env's.

    It turns namespace routing on while it runs. Raises BundleError listing every problem found before anything is
    written; a write that fails stops it, and every earlier one stays, in the ledger, for the next run to reuse."""
    provider = build_sandbox_provider(sandbox) if sandbox else None
    with namespace_routing():
        bundle = parse_bundle(Path(root))
        resolved = resolve_bundle(bundle)
        plan = _plan(resolved, tasks, evals)
        ids, problems = registry_ids(plan, _prefix(bundle, prefix))
        if problems:
            raise BundleError(problems)
        registered = with_task_requires(translated(plan, ids))
        target = configured_store(get_config().get_document_store())
        shared = shared_store(target)
        identity, commit, dirty = bundle_identity(bundle.root)
        ledger = RegistrationLedger(target, registered, identity, commit, dirty)
        problems = list(_unregistrable(registered, shared))
        if problems:
            raise BundleError(problems)
        requires = tuple(declare(resolved, declaring or declaring_for(bundle.root, installed=False, dry_run=dry_run)))
        say = on_progress or (lambda line: None)
        materialization = materialize(
            registered, dry_run=dry_run, ledger=ledger,
            on_wait=lambda: say("waiting for another agent-env run to finish writing this bundle's ids"),
            on_build=lambda write: say(f"  {_label(registered, write)}: building with docker and pushing"),
            on_write=lambda done: say(f"  {registered_line(registered, done)}"), sandbox_provider=provider,
            on_progress=lambda write, line: say(f"  {_label(registered, write)}: {line}"))
    return Registration(registered, materialization, ids, tuple(_floating(registered)), identity, shared, requires)


def shared_store(store: DocumentStore) -> bool:
    """Whether ``store`` is other than this machine's own, so others read what's written to it."""
    return not isinstance(store, LocalSqliteDocumentStore)


def registered_line(plan: Plan, done: Materialized) -> str:
    """A write, as a registration names it: where it's written from, the registry id and version, and why."""
    why = "unchanged" if done.reused else "; ".join(done.reasons)
    return f"{_label(plan, done.write)} -> {done.write.kind.store} {done.write.id} v{done.version} ({why})"


def _plan(resolved: ResolvedBundle, tasks: Sequence[str], evals: Sequence[str]) -> Plan:
    """The plan of ``tasks`` and ``evals``, or of every task and eval, or of every entity of a bundle with none."""
    entries = resolved.bundle.entries
    if not tasks and not evals:
        tasks = [entry.name for entry in entries if entry.kind is BundleKind.TASK]
        evals = [entry.name for entry in entries if entry.kind is BundleKind.EVAL]
        if not tasks and not evals:
            return plan_entities(resolved, [entry.id for entry in entries], installed=installed_bundles)
    return plan_bundle(resolved, tasks=tasks, evals=evals, installed=installed_bundles, default_agents=False)


def _prefix(bundle: Bundle, prefix: str | None) -> str:
    """``prefix``, or the folder's name as an id and a dash."""
    if prefix is not None:
        if prefix and not _REGISTRY_ID.fullmatch(prefix):
            raise BundleError([f"--prefix {prefix!r}: an id opens with a lowercase letter or digit and holds only "
                               "lowercase letters, digits and . _ -"])
        return prefix
    name = _slug(bundle.name)
    if not name:
        raise BundleError([f"{bundle.name!r}: the folder's name gives no id; pass --prefix"])
    return f"{name}-"


def registry_ids(plan: Plan, prefix: str) -> tuple[dict[str, str], list[str]]:
    """The registry id each ``@local`` id the plan writes takes, and each problem giving them.

    The bundle's own entry is ``<prefix><name>``, its name lowercased with each run of what an id can't hold made
    ``-``. An installed bundle's agent pulled in by a reference takes the first alias its bundle gives it, which the
    reference named it by; any other entity of an installed bundle has no registry id, and an env's alias can't be
    one, since no store env may start on an installed env alias's id. An id derived from another (``<id>__env_image``) keeps its suffix after the
    other's registry id, and a task's output is ``<task id>-<name>``."""
    ids: dict[str, str] = {}
    problems: list[str] = []
    for write in plan.writes:
        entry = write.source.entry
        if entry.id in ids or (isinstance(write.source, BundleImage) and write.id != entry.id):
            continue
        if entry in plan.origins:
            alias = _alias(plan.origins[entry], entry)
            if alias is None:
                problems.append(f"{plan.path(entry)}: comes with an installed bundle, and only an agent with an alias "
                                "is registered from one; name an entity the store holds instead")
                continue
            ids[entry.id] = alias
        elif name := _slug(entry.name):
            ids[entry.id] = f"{prefix}{name}"
        else:
            problems.append(f"{plan.path(entry)}: its name gives no id; rename it")
    for write in plan.writes:
        if write.id not in ids and (derived := _derived(write.id, ids)) is not None:
            ids[write.id] = derived
        source = write.source
        if isinstance(source, ResolvedEntry) and source.entry.id in ids:
            for output in source.outputs:
                name = _slug(output.id.removeprefix(f"{source.entry.id}/").rsplit("/", 1)[-1])
                ids[output.id] = f"{ids[source.entry.id]}-{name}"
    stores: dict[str, set[str]] = {}
    for write in plan.writes:
        stores.setdefault(write.id, set()).add(write.kind.store)
        if isinstance(write.source, ResolvedEntry):
            for output in write.source.outputs:
                stores.setdefault(output.id, set()).add(BundleKind.ARTIFACT.store)
    by_target: dict[tuple[str, str], list[str]] = {}
    for local, registry_id in ids.items():
        for store in stores.get(local, ()):
            by_target.setdefault((store, registry_id), []).append(local)
        if not _REGISTRY_ID.fullmatch(registry_id):
            problems.append(f"{registry_id!r}: isn't an id a registry takes; rename what it's made from")
    for (_, registry_id), locals_ in by_target.items():
        if len(locals_) > 1:
            problems.append(f"{', '.join(sorted(locals_))} would all be registered as {registry_id!r}; rename one")
    return ids, problems


def _alias(bundle: Bundle, entry: BundleEntry) -> str | None:
    aliases = bundle.aliases.get(AGENT_ALIASES, {}).get(entry.name, ()) if entry.kind is BundleKind.AGENT else ()
    return aliases[0] if aliases else None


def _derived(entity_id: str, ids: Mapping[str, str]) -> str | None:
    """The registry id of ``entity_id`` when it's derived from an id in ``ids``: that id's, with the same suffix."""
    base = max((known for known in ids if entity_id.startswith(f"{known}__")), key=len, default=None)
    return None if base is None else ids[base] + entity_id[len(base):]


def _slug(name: str) -> str:
    """``name`` lowercased, in ASCII, with each run of what an id can't hold made ``-``, and no ``-``, ``.`` or ``_``
    at its ends."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return _NOT_IN_ID.sub("-", ascii_name.replace("__", "-")).strip("-._")


# The fields of a Plan that hold no ``@local`` id, so a translated plan keeps them: the store's latest versions, by store
# id, and the agent ``--agent`` names, which a registration plans without.
_KEPT = frozenset({"store_latest", "agent"})


def translated(plan: Plan, ids: Mapping[str, str]) -> Plan:
    """``plan`` writing each entity under its registry id, and naming it so wherever the plan's documents do."""
    return replace(plan, **_translations(plan, ids))


def _translations(plan: Plan, ids: Mapping[str, str]) -> dict[str, Any]:
    """Each field of ``plan`` but those it keeps (``_KEPT``), with each ``@local`` id in it its registry id."""

    def entity(entry: BundleEntry) -> BundleEntry:
        if entry not in entries:
            entries[entry] = replace(entry, id=ids.get(entry.id, entry.id))
        return entries[entry]

    def image(source: BundleImage) -> BundleImage:
        return replace(source, id=ids.get(source.id, source.id), entry=entity(source.entry))

    def reference(ref: Reference) -> Reference:
        local = ref.local
        if isinstance(local, BundleImage):
            local = image(local)
        elif isinstance(local, BundleEntry):
            local = entity(local)
        return replace(ref, id=ids.get(ref.id, ref.id), local=local)

    def resolved(source: ResolvedEntry) -> ResolvedEntry:
        return ResolvedEntry(entity(source.entry), _renamed(source.config, ids),
                             tuple(reference(ref) for ref in source.references),
                             tuple(replace(output, id=ids.get(output.id, output.id)) for output in source.outputs))

    def source_of(source: ResolvedEntry | BundleImage) -> ResolvedEntry | BundleImage:
        return image(source) if isinstance(source, BundleImage) else resolved(source)

    entries: dict[BundleEntry, BundleEntry] = {}
    writes = tuple(Write(write.kind, ids.get(write.id, write.id), source_of(write.source),
                         tuple((store, ids.get(id, id)) for store, id in write.needs)) for write in plan.writes)
    sources = {write.source.entry: write.source for write in writes if isinstance(write.source, ResolvedEntry)}
    own = plan.bundle.bundle
    bundle = replace(own, entries=tuple(entity(entry) for entry in own.entries))
    resolved_bundle = ResolvedBundle(bundle, tuple(sources.get(entity(entry.entry)) or resolved(entry)
                                                   for entry in plan.bundle.entries),
                                     tuple(image(source) for source in plan.bundle.images))
    return {
        "bundle": resolved_bundle, "writes": writes,
        "tasks": tuple(sources[entity(task.entry)] for task in plan.tasks),
        "evals": tuple(sources[entity(item.entry)] for item in plan.evals),
        "store_refs": tuple(reference(ref) for ref in plan.store_refs),
        "origins": {entity(entry): origin for entry, origin in plan.origins.items()},
        "published": {ids.get(id, id): publication for id, publication in plan.published.items()},
        "alias_uses": {kind: {alias: _aliased(use, ids) for alias, use in uses.items()}
                       for kind, uses in plan.alias_uses.items()},
    }


def _aliased(use: AliasUse, ids: Mapping[str, str]) -> AliasUse:
    return replace(use, target_id=ids.get(use.target_id, use.target_id))


def _renamed(value: Any, ids: Mapping[str, str]) -> Any:
    """A copy of ``value`` with each string that's an ``@local`` id in ``ids``, or derived from one, its registry id."""
    if isinstance(value, str):
        return ids.get(value) or _derived(value, ids) or value if is_local_id(value) else value
    if isinstance(value, dict):
        return {key: _renamed(item, ids) for key, item in value.items()}
    if isinstance(value, list):
        return [_renamed(item, ids) for item in value]
    return value


class RegistrationLedger(Ledger):
    """A registration's rows, in the target store: each names the bundle that wrote it, by its identity, and the commit
    and state of the folder it was written from. A task or eval is hashed with the versions it names, each write lands
    on exactly the version its check found next, and each image the bundle builds is built here and pushed."""

    collection = REGISTRATIONS
    pins_references = True
    strict = True
    pushes_images = True

    def __init__(self, store: DocumentStore, plan: Plan, identity: str, commit: str | None, dirty: bool | None):
        super().__init__(store, plan)
        self.identity, self.commit, self.dirty = identity, commit, dirty

    def _bundle(self, write: Write) -> str:
        """The bundle's identity, or for an entity an installed bundle comes with, that bundle's, which every bundle
        registering it shares."""
        origin = self._plan.origins.get(write.source.entry)
        return f"installed:{origin.id_root.removeprefix(LOCAL_PREFIX)}" if origin is not None else self.identity

    def row_fields(self) -> dict[str, Any]:
        return {"commit": self.commit, "dirty": self.dirty, "agent_env": package_version("agentenv-framework")}


def bundle_identity(root: Path) -> tuple[str, str | None, bool | None]:
    """Who the bundle at ``root`` is, which its ledger rows name: its git remote, without credentials, and its path in
    the repo, or else ``folder:`` and its path on this machine; with the commit checked out and whether the folder
    holds uncommitted changes, both None outside git."""
    remote = _git(root, "remote", "get-url", "origin")
    path_in_repo = _git(root, "rev-parse", "--show-prefix")
    if remote is None or path_in_repo is None:
        return f"folder:{root}", None, None
    status = _git(root, "status", "--porcelain", "--", ".")
    return (f"{_remote_name(remote)}:{path_in_repo.rstrip('/')}", _git(root, "rev-parse", "HEAD"),
            None if status is None else bool(status))


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                timeout=_GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _remote_name(remote: str) -> str:
    """A git remote as ``host/path``, without its scheme, credentials, port or ``.git``."""
    if "://" in remote:
        parts = urlsplit(remote)
        host, path = parts.hostname or "", parts.path
    else:  # scp-like, user@host:path
        host, _, path = remote.rpartition("@")[2].partition(":")
    return f"{host}/{path.strip('/').removesuffix('.git')}"


def _unregistrable(plan: Plan, shared: bool) -> Iterator[str]:
    """A problem for each ``@local`` id left in what the plan writes and, for a ``shared`` store, each credential
    written out and each URL that reaches only this machine."""
    for write in plan.writes:
        if not isinstance(write.source, ResolvedEntry):
            continue
        where = plan.path(write.source.entry)
        for path, text in _strings(write.source.config):
            if is_local_id(text):
                yield f"{where}: {path or 'its value'} names {text!r}, which only this machine's stores hold"
            elif shared and (url := _local_url(text)) is not None:
                yield f"{where}: {path or 'its value'} names {url}, which only this machine reaches"
        if shared:
            for path, why in credentials_in(write.source.config):
                yield f"{where}: {path}: {why}; leave it out, for whatever runs the task to give it"


def _strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def _local_url(text: str) -> str | None:
    """The first URL in ``text`` that reaches only this machine: a file, or a loopback host's."""
    for match in _URL.finditer(text):
        url = match.group(0)
        if url.lower().startswith("file:") or is_loopback_host(urlsplit(url).hostname or ""):
            return url
    return None


def with_task_requires(plan: Plan) -> Plan:
    """``plan`` with each task it writes requiring the plugin packages behind its own step types and the types of the
    bundle entities it references (``Write.requires``): as bundle.toml writes a package when it requires it, else at
    least the version installed here. A task using only what core ships requires nothing."""
    providers = Providers()
    writes = {(write.kind.store, write.id): write for write in plan.writes}

    def reached(task: Write) -> Iterator[Write]:
        seen, queue = set(), [task]
        while queue:
            write = queue.pop()
            if (key := (write.kind.store, write.id)) in seen:
                continue
            seen.add(key)
            yield write
            queue.extend(writes[need] for need in write.needs if need in writes)

    def requires(task: Write) -> tuple[str, ...] | None:
        found = [use for write in reached(task) if isinstance(write.source, ResolvedEntry)
                 for use in entry_uses(plan.bundle_of(write.source.entry).root, write.source, providers)]
        return tuple(detected(found, plan.bundle.bundle.requires or ())) or None

    return replace(plan, writes=tuple(replace(write, requires=requires(write)) if write.kind is BundleKind.TASK
                                      else write for write in plan.writes))


def _floating(plan: Plan) -> Iterator[str]:
    """Each entity a registered task names in a field that takes no version, so whatever runs it reads its latest."""
    registry = get_task_step_registry()
    for write in plan.writes:
        if write.kind is not BundleKind.TASK:
            continue
        for step in write.source.config:
            for site in ref_sites(registry[step["type"]].entity_refs or (), step):
                if site.ref.role is RefRole.INPUT and site.version_key is None and isinstance(site.value, str):
                    yield (f"{plan.path(write.source.entry)}: step {step.get('id')!r}: {site.path}: "
                           f"{site.ref.kind.value} {site.value}")


def describe_target() -> list[str]:
    """The configured stores a registration writes to, as lines: their kinds, and where each is."""
    config = get_config()
    documents = configured_store(config.get_document_store())
    objects = configured_store(config.get_object_store())
    images = configured_store(config.get_image_store())
    where = (str(documents.path) if isinstance(documents, LocalSqliteDocumentStore)
             else getattr(getattr(documents, "database", None), "name", None))
    return [f"documents: {type(documents).__name__}" + (f" ({where})" if where else ""),
            f"objects: {type(objects).__name__} ({objects.object_url('')})",
            f"images: {type(images).__name__}" + (f" ({images.registry_host})" if hasattr(images, "registry_host")
                                                   else "")]

