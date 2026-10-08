"""Centralized configuration for agent-env.

Provides a singleton Config that manages store backends and model routing.
Configure once at startup, then access via get_config().

Backends resolve from ``.agentenv/config.toml`` (or ``AGENT_ENV_CONFIG``); with no
config, every store defaults to its local backend — no external coordinates are
built in:

- Document store (AGENT_ENV_DOCUMENT_STORE): "local" (or unset): stdlib SQLite at
  document_store/documents.db under the per-user state root (``paths.state_root``).
  MongoDB has no built-in coordinates; configure it via a [stores.document] table.
- Object store (AGENT_ENV_OBJECT_STORE): "local" (or unset): filesystem under
  object_store/ in the per-user state root. A hosted backend (S3, Cloud Storage) has no
  built-in coordinates; configure it via a [stores.object] table.
- Image store (AGENT_ENV_IMAGE_STORE): "local" (or unset): an OCI registry at
  localhost:5000. A hosted registry (ECR) has no built-in coordinates; configure it via a
  [stores.image] table.
- Secret store (AGENT_ENV_SECRET_STORE): "local" (or unset): process env vars /
  optional local file. A hosted secret manager (AWS Secrets Manager, Google Cloud Secret
  Manager) has no built-in coordinates; configure it via a [stores.secret] table.

AGENT_ENV_FIXTURE_PREFIX (default ""): prepends <prefix>/ to the keys of artifact
objects, image builds, env and agent snapshots, changelogs, default agent and judge
trajectories, verifier outputs and the A2A validator's fixtures, so a fresh control-plane
backend can write without colliding with existing objects in a shared store. Empty in prod.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import yaml

from agent_env.config import loader as config_loader
from agent_env.config import model as model_config
from agent_env.config.errors import ConfigError, EmptySecretBundleError
from agent_env.config import snapshot as config_snapshot
from agent_env.config.paths import state_root
from agent_env.config.provenance import (
    KIND_DEFAULT,
    KIND_ENV,
    KIND_FILE,
    KIND_INSTALLED,
    Layer,
    Traced,
)

if TYPE_CHECKING:
    from pymongo.database import Database

    from agent_env.store.document_store import DocumentStore, LocalSqliteDocumentStore
    from agent_env.store.image_store import ImageStore
    from agent_env.store.object_store import ObjectStore
    from agent_env.store.secret_store import SecretStore

logger = logging.getLogger(__name__)

_STORE_BACKEND_MONGO = "mongo"
_STORE_BACKEND_LOCAL = "local"

_RUNNER_BACKEND_LOCAL = "local"

_DEFAULT_LOCAL_REGISTRY_HOST = "localhost:5000"


def _default_sandbox_mode() -> str:
    from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM
    return SANDBOX_MODE_VM

_HUMAN_A2A_CONFIG_SECTION = "conversations"
_HUMAN_A2A_CONFIG_KEY = "default_human_a2a_url"
_AGENTS_CONFIG_SECTION = "agents"
_AGENTS_CONFIG_KEY = "default_a2a_agent_id"
_DEFAULT_A2A_AGENT_ID = "a2a-default"


def _validated_human_a2a_url(value: object, source: str) -> str:
    # Fail here rather than as an AttributeError or bogus URL inside a task step.
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{source} must be a non-empty string URL, got {value!r}")
    return value


def _validated_a2a_agent_id(value: object, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{source} must be a non-empty string A2A agent id, got {value!r}")
    return value


def validated_agents_section(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``[agents]`` table, checked as the reader checks it; ``config show`` runs this too."""
    unknown = set(section) - {_AGENTS_CONFIG_KEY}
    if unknown:
        raise ConfigError(
            f"[{_AGENTS_CONFIG_SECTION}] has unknown keys {sorted(unknown)}; "
            f"allowed: ['{_AGENTS_CONFIG_KEY}']"
        )
    if _AGENTS_CONFIG_KEY in section:
        _validated_a2a_agent_id(
            section[_AGENTS_CONFIG_KEY], f"[{_AGENTS_CONFIG_SECTION}] {_AGENTS_CONFIG_KEY}"
        )
    return section

@dataclass(frozen=True)
class ResolvedKey:
    """A single key resolved as ``env > configure() field > config.toml > built-in default``.

    Declared once in ``RESOLVED_KEYS`` below and read by both the getter and
    ``agent_env.config.describe``, so a report cannot name a layer the getter would not have
    taken — the contract ``AliasedSection`` already carries for whole sections, and the one
    whose absence let a report claim the file while ``configure()`` was winning.

    The declaration owns *precedence and coordinates* only. Which layers get validated, and
    how, stays with the getter: those rules differ per key and are none of a report's
    business.
    """

    toml_path: tuple[str, ...]
    env_var: Optional[str] = None
    field: Optional[str] = None
    default: Any = None
    default_label: str = "built-in default"

    @property
    def name(self) -> str:
        return ".".join(self.toml_path)

    @property
    def file_label(self) -> str:
        """The spelling the getters' errors use: ``[agents] default_a2a_agent_id``."""
        return f"[{self.toml_path[0]}] {'.'.join(self.toml_path[1:])}"


RESOLVED_KEYS: tuple[ResolvedKey, ...] = (
    ResolvedKey(("model", "api_key"), env_var="LITELLM_API_KEY"),
    ResolvedKey(("model", "base_url"), env_var="LITELLM_BASE_URL"),
    ResolvedKey(("conversations", "default_human_a2a_url"),
                env_var="AGENT_ENV_HUMAN_A2A_URL", field="default_human_a2a_url"),
    ResolvedKey(("agents", "default_a2a_agent_id"), field="default_a2a_agent_id",
                default=_DEFAULT_A2A_AGENT_ID, default_label="built-in default in config.runtime"),
)
_KEY_BY_NAME = {key.name: key for key in RESOLVED_KEYS}


@dataclass(frozen=True)
class AliasedSection:
    """A section resolved as ``default < config.toml < env var``, whose value may be written
    as a name that ``alias`` expands into an impl table.

    Declared once in ``ALIASED_SECTIONS`` below and read by both the resolvers and
    ``agent_env.config.describe``, so a report cannot name coordinates the resolver would
    not have used.
    """

    name: str
    env_var: str
    toml_path: tuple[str, ...]
    default: str
    alias: Callable[["Config", str], dict]


class PluginFailures:
    """What a Config's registry builds skipped, by group then name, with the reason
    (``agent_env.plugins.load_failures``)."""

    def __init__(self) -> None:
        self._groups: dict[str, dict[str, str]] = {}

    def replace(self, group: str, failures: dict[str, str]) -> None:
        """Record ``group``'s skipped plugins; a rebuild of the group replaces its entry."""
        self._groups[group] = failures

    def reason(self, group: str, name: str) -> Optional[str]:
        return self._groups.get(group, {}).get(name)

    def snapshot(self) -> dict[str, dict[str, str]]:
        # Copied before reading: another thread may be recording a group it is building.
        groups = dict(self._groups)
        return {group: dict(failures) for group, failures in groups.items() if failures}


# Identity, not field equality: a Config owns a document, so two are interchangeable only if
# they are the same object. Field equality called two on different config files equal.
@dataclass(eq=False)
class Config:
    """Configuration for storage connections."""

    fixture_prefix: str = field(
        default_factory=lambda: os.getenv("AGENT_ENV_FIXTURE_PREFIX", "")
    )
    agent_sandbox_mode: str = field(
        default_factory=lambda: os.getenv("AGENT_SANDBOX_MODE") or _default_sandbox_mode()
    )
    default_gateway_env_id: str = "default"
    default_service_db_env_id: str = "default-db"
    default_website_browser_env_id: str = "website-browser"
    default_a2a_agent_id: Optional[str] = None
    modal_default_region: str = field(
        default_factory=lambda: os.getenv("AGENT_ENV_MODAL_REGION", "us-east-1")
    )
    default_human_a2a_url: Optional[str] = None

    _secret: Optional[Mapping[str, Any]] = field(default=None, repr=False, compare=False)
    _snapshot: Optional[config_snapshot.Snapshot] = field(default=None, repr=False, compare=False)
    _snapshot_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    _document_store: Optional["DocumentStore"] = field(default=None, repr=False, compare=False)
    _document_store_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    _image_store: Optional[ImageStore] = field(default=None, repr=False, compare=False)
    _image_store_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    _object_store: Optional[ObjectStore] = field(default=None, repr=False, compare=False)
    _object_store_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    _local_stores: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _routed_stores: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _separate_stores: dict[str, tuple[Any, Any]] = field(default_factory=dict, repr=False, compare=False)
    _routing_lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)
    _secret_store: Optional[SecretStore] = field(default=None, repr=False, compare=False)
    _secret_store_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    _model_cfg: Optional[model_config.ModelConfig] = field(default=None, repr=False, compare=False)
    _runner: Optional[Any] = field(default=None, repr=False, compare=False)
    _runner_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    _registries: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _plugin_failures: PluginFailures = field(default_factory=PluginFailures, repr=False, compare=False)

    def _registry(self, name: str, build: Callable[..., Any]) -> Any:
        """One registry, memoized against this Config's document.

        Unlocked unlike the stores: ``build`` runs ``load_impl``, and racing builds read the
        same pinned document anyway.
        """
        cached = self._registries.get(name)
        if cached is None:
            # Pinned before ``build`` imports any plugin: a plugin package that points
            # AGENT_ENV_CONFIG elsewhere on import must not change which config this one reads.
            self._document()
            state = _registries_building.__dict__
            building = state.setdefault("names", set())
            if name in building:
                # Re-entered while this thread is still building that registry (a plugin asked
                # for it while being imported), so it may lack that plugin: returned, not kept,
                # and its failures unrecorded, since the outer build records the whole group.
                state["nested"] = state.get("nested", 0) + 1
                try:
                    return build(self)
                finally:
                    state["nested"] -= 1
            building.add(name)
            # A registry first built inside a re-entered one is kept, so it records its own failures.
            outer, state["nested"] = state.get("nested", 0), 0
            try:
                cached = self._registries[name] = build(self)
            finally:
                building.discard(name)
                state["nested"] = outer
        return cached

    # Imported per method: circular at module scope, and they drag in the model classes.
    def env_registry(self) -> dict[str, Any]:
        from agent_env.env import registry

        return self._registry("envs", registry._build_registry)

    def artifact_registry(self) -> dict[str, Any]:
        from agent_env.artifact import registry

        return self._registry("artifacts", registry._build_registry)

    def artifact_type_aliases(self) -> dict[str, str]:
        from agent_env.artifact import registry

        return self._registry("artifact_type_aliases", registry._load_type_aliases)

    def task_step_registry(self) -> dict[str, Any]:
        from agent_env.task_step import registry

        return self._registry("task_steps", registry._build_registry)

    def sandbox_registry(self) -> dict[str, dict]:
        from agent_env.providers.sandbox_providers import sandbox_provider

        return self._registry("sandbox", sandbox_provider._build_registry)

    def state_registry(self) -> dict[str, dict]:
        from agent_env.providers.env_state import env_state_provider

        return self._registry("state", env_state_provider._build_registry)

    def env_provider_registry(self) -> dict[str, Any]:
        from agent_env.providers.env_providers import env_provider

        return self._registry("env_providers", env_provider._build_registry)

    def _get_secret(self) -> Mapping[str, Any]:
        """Combined secret mapping from the configured secret store — the whole-bundle reader
        for callers that index individual keys, sharing ``get_secret_store()`` with ``secret:``
        interpolation (so they can't diverge). Needs a bundle-backed store."""
        if self._secret is None:
            store = self.get_secret_store()
            load = getattr(store, "_load", None)
            if not callable(load):
                raise ConfigError(
                    f"{type(store).__name__} exposes no combined secret mapping; "
                    "_get_secret() needs a bundle-backed secret store, one whose _load() returns every secret."
                )
            bundle = load()
            if not bundle:
                raise EmptySecretBundleError(
                    "Secret bundle is empty: configure a [stores.secret] table (or point "
                    "AGENT_ENV_CONFIG at a config.toml with one), or populate the local "
                    "secret store's file/values."
                )
            self._secret = bundle
        return self._secret

    def get_litellm_api_key(self) -> str:
        return self._model_config().resolve_api_key(secret_resolver=self._resolve_secret)

    def get_modal_credentials(self) -> tuple[str, str]:
        token_id = os.getenv("MODAL_TOKEN_ID")
        token_secret = os.getenv("MODAL_TOKEN_SECRET")
        if token_id and token_secret:
            return token_id, token_secret

        modal_toml = Path.home() / ".modal.toml"
        if modal_toml.exists():
            import tomllib
            for profile in tomllib.loads(modal_toml.read_text()).values():
                if isinstance(profile, dict) and profile.get("active") and profile.get("token_id") and profile.get("token_secret"):
                    return profile["token_id"], profile["token_secret"]

        secret = self._get_secret()
        return secret["modal_token_id"], secret["modal_token_secret"]

    def get_litellm_base_url(self) -> str:
        base = os.getenv("LITELLM_BASE_URL") or self._model_config().base_url
        if not base:
            raise ConfigError(
                "No model endpoint configured: set [model] base_url in "
                ".agentenv/config.toml or the LITELLM_BASE_URL env var."
            )
        return base

    def _model_config(self) -> model_config.ModelConfig:
        """The parsed ``[model]`` config, cached."""
        if self._model_cfg is None:
            self._model_cfg = model_config.ModelConfig.from_section(
                self.section("model"), secret_resolver=self._resolve_secret
            )
        return self._model_cfg

    def get_default_human_a2a_url(self) -> str:
        """The base URL a human-A2A (HITL) step parks its conversation on.

        First match wins: ``AGENT_ENV_HUMAN_A2A_URL`` (absolute — local dev repoints
        it at the local hub), an explicit ``configure(default_human_a2a_url=...)``,
        then config.toml ``[conversations] default_human_a2a_url``; unconfigured
        resolution raises, so call this only where the URL is consumed. Returns the
        *base* URL; ``deploy_human_agent`` appends ``/instance/{instance_id}``.
        """
        key = _KEY_BY_NAME[f"{_HUMAN_A2A_CONFIG_SECTION}.{_HUMAN_A2A_CONFIG_KEY}"]
        for layer in self.key_candidates(key):
            if layer.kind == KIND_ENV:
                if layer.raw:
                    return layer.raw              # deliberately unvalidated, as before
                continue
            if layer.kind == KIND_INSTALLED:
                return _validated_human_a2a_url(layer.raw, key.field)
            # Only here, where the file is what we are about to take, does its shape matter.
            # Checking it earlier lets a malformed [conversations] beat an override that was
            # never going to read the file at all.
            self.section(_HUMAN_A2A_CONFIG_SECTION)
            resolved = config_loader.interpolate(layer.raw, secret_resolver=self._resolve_secret)
            return _validated_human_a2a_url(resolved, key.file_label)
        raise ConfigError(f"No human-A2A URL configured: set [{_HUMAN_A2A_CONFIG_SECTION}] {_HUMAN_A2A_CONFIG_KEY} or AGENT_ENV_HUMAN_A2A_URL")

    def get_default_a2a_agent_id(self) -> str:
        """The agent a ``deploy_agent`` step without an id deploys:
        ``configure(default_a2a_agent_id=...)`` > ``[agents] default_a2a_agent_id`` > ``a2a-default``."""
        key = _KEY_BY_NAME[f"{_AGENTS_CONFIG_SECTION}.{_AGENTS_CONFIG_KEY}"]
        for layer in self.key_candidates(key):
            if layer.kind == KIND_INSTALLED:
                return _validated_a2a_agent_id(layer.raw, key.field)
            # The file is in play now — or the built-in default beneath it, which the old
            # code also validated the section before returning. Not before: a broken table
            # must not beat configure().
            validated_agents_section(self.section(_AGENTS_CONFIG_SECTION))
            if layer.kind == KIND_DEFAULT:
                return layer.raw
            resolved = config_loader.interpolate(layer.raw, secret_resolver=self._resolve_secret)
            return _validated_a2a_agent_id(resolved, key.file_label)
        return _DEFAULT_A2A_AGENT_ID

    def get_default_model(self) -> str | None:
        return self._model_config().default

    def get_model_for_role(self, role: str) -> str | None:
        return self._model_config().model_for_role(role)

    def get_model_params(self, overrides: dict | None = None) -> dict:
        params = dict(self._model_config().params)
        if overrides:
            params.update(config_loader.interpolate(overrides, secret_resolver=self._resolve_secret))
        return params

    def resolve_model_call(
        self,
        model: str,
        *,
        default_api_key: str | None = None,
        base_override: str | None = None,
        model_overrides: dict[str, dict[str, str]] | None = None,
    ) -> model_config.ModelCallConfig:
        return self._model_config().resolve_call(
            model,
            default_api_key=default_api_key,
            base_override=base_override,
            model_overrides=model_overrides,
            secret_resolver=self._resolve_secret,
        )

    @property
    def db(self) -> "Database":
        """Raw pymongo Database shared with the configured Mongo document store, for callers
        that need collections the DocumentStore API does not model."""
        from agent_env.store.document_store import MongoDocumentStore

        store = self._configured_document_store()
        if not isinstance(store, MongoDocumentStore):
            raise ConfigError(
                "Config.db needs a Mongo document store, but the configured store is "
                f"{type(store).__name__}: configure a [stores.document] table with a "
                "MongoDocumentStore impl (and unset AGENT_ENV_DOCUMENT_STORE, which "
                "overrides it)."
            )
        return store.database

    def get_document_store(self) -> "DocumentStore":
        """The control-plane document store (cached): an explicitly-set backend, else the one from
        ``AGENT_ENV_DOCUMENT_STORE`` / config.toml (``local`` → SQLite default). With namespace
        routing on, ``@local`` documents and ``@local`` runs' records go to the ``@local`` namespace's
        own local store (``agent_env.store.routing``)."""
        from agent_env.store import routing

        configured = self._configured_document_store()
        if not routing.namespace_routing_enabled():
            return configured
        return self._routed("document", configured)

    def _configured_document_store(self) -> "DocumentStore":
        if self._document_store is None:
            with self._document_store_lock:
                if self._document_store is None:
                    self._document_store = self._build_default_document_store()
        return self._document_store

    def set_document_store(self, store: "DocumentStore") -> None:
        """Install an explicit backend; get_document_store() then returns it verbatim."""
        self._document_store = store

    def _build_default_document_store(self) -> DocumentStore:
        from agent_env.store.document_store import DocumentStore

        section = self._resolve_document_section()
        return config_loader.build_store(section, DocumentStore, secret_resolver=self._resolve_secret)

    def trace_section(self, name: str) -> Traced:
        """A declared section's value plus every layer that could have supplied it — winner
        first, later layer wins: default < config.toml < env var.

        A string (an env value or a bare toml value) expands through the section's alias; a
        table is used as-is. Reporting reads the same layers, so both the precedence rule and
        the section's coordinates are stated exactly once.
        """
        section = _BY_NAME[name]
        candidates = self.section_candidates(name)
        if candidates[0].error is not None:
            raise ConfigError(candidates[0].error)
        raw = candidates[0].raw
        return Traced(
            value=section.alias(self, raw) if isinstance(raw, str) else raw,
            winner=candidates[0],
            shadowed=tuple(candidates[1:]),
        )

    def key_candidates(self, key: "ResolvedKey") -> list[Layer]:
        """Every layer that could supply ``key``, highest priority first.

        Separate from the getters so the report reads the same order they take, rather than
        a second copy of it that can drift.
        """
        candidates: list[Layer] = []
        if key.env_var is not None:
            override = os.getenv(key.env_var)
            if override is not None:
                candidates.append(Layer(KIND_ENV, f"${key.env_var}", override))
        if key.field is not None:
            installed = getattr(self, key.field, None)
            if installed is not None:
                candidates.append(Layer(KIND_INSTALLED, f"configure({key.field}=...)", installed))
        from_file = self.file_candidate(key.toml_path)
        if from_file is not None:
            candidates.append(from_file)
        if key.default is not None:
            candidates.append(Layer(KIND_DEFAULT, key.default_label, key.default))
        return candidates

    def section_candidates(self, name: str) -> list[Layer]:
        """Every layer that could supply a declared section, highest priority first.

        Separate from resolution so a report can still name the layer at fault when turning
        that layer's value into a section is what failed.
        """
        section = _BY_NAME[name]
        candidates: list[Layer] = []
        override = os.getenv(section.env_var)
        if override is not None:
            candidates.append(Layer(KIND_ENV, f"${section.env_var}", override))
        from_file = self.file_candidate(section.toml_path)
        if from_file is not None:
            candidates.append(from_file)
        candidates.append(Layer(KIND_DEFAULT, "built-in default", section.default))
        return candidates

    def file_candidate(self, toml_path: tuple[str, ...]) -> Optional[Layer]:
        """The file's contribution to this section, or a candidate carrying why it could not
        be read. Never raises: a layer that something outranks must not be able to break the
        layer that outranks it — `trace_section` raises only if this one wins.
        """
        label = f"[{'.'.join(toml_path)}]"
        try:
            node: Any = self.config_file()
        except (ConfigError, OSError) as e:
            return Layer(KIND_FILE, label, None, error=str(e))
        walked: list[str] = []
        for part in toml_path:
            if not isinstance(node, dict):
                return Layer(KIND_FILE, label, None, error=(
                    f"config.toml [{'.'.join(walked)}] must be a table, got {type(node).__name__}"
                ))
            walked.append(part)
            node = node.get(part)
            if node is None:
                return None
        return Layer(KIND_FILE, label, node)

    def _resolve_document_section(self) -> dict:
        return self.trace_section("document").value

    def _document_alias(self, name: str) -> dict:
        if name == _STORE_BACKEND_MONGO:
            raise ConfigError(
                "The 'mongo' document backend has no built-in coordinates: configure a "
                "[stores.document] table (and unset AGENT_ENV_DOCUMENT_STORE, which "
                "overrides it)."
            )
        if name == _STORE_BACKEND_LOCAL:
            return {
                "impl": "agent_env.store.document_store:LocalSqliteDocumentStore",
                "config": {"path": self._local_document_store_path()},
            }
        raise ConfigError(
            f"Unknown AGENT_ENV_DOCUMENT_STORE={name!r} (expected 'local', or a [stores.document] table for mongo)"
        )

    def _local_document_store_path(self) -> str:
        return str(state_root() / "document_store" / "documents.db")

    def local_namespace_document_store(self) -> "LocalSqliteDocumentStore":
        """The ``@local`` namespace's own document store, whether or not routing is on. Records about
        ``@local`` entities, like the bundle ledger, live beside them here and never reach a configured
        store. Building it creates nothing; its first write creates the file."""
        return self._local_store("document")

    def _local_namespace_document_store_path(self) -> str:
        return str(state_root() / "document_store" / "local.db")

    def _document(self) -> config_snapshot.Snapshot:
        """The document this Config reads, resolved on first use and kept.

        Lazy, not in ``__init__``: ``config show`` builds a Config to *report* a broken
        ``AGENT_ENV_CONFIG``, and a raising constructor would take that report away.
        """
        if self._snapshot is None:
            with self._snapshot_lock:
                if self._snapshot is None:
                    self._snapshot = config_snapshot.resolve()
        return self._snapshot

    def section(self, *names: str) -> Mapping[str, Any]:
        """The config table at ``names``: ``{}`` when absent, an error when mis-shaped."""
        return self._document().section(*names)

    def config_path(self) -> Optional[Path]:
        """The config.toml this Config resolves against. The process resolves one document,
        so every section and every local-store path agree about which file they came from."""
        return self._document().path

    def config_file(self) -> dict:
        """The parsed config.toml (``{}`` when absent) — this Config's one resolution.

        Raises if that resolution failed to parse. The path stays available through
        ``config_path`` either way, which is what lets a report name a file it cannot read.
        """
        return dict(self._document().section())

    def get_object_store(self) -> ObjectStore:
        """The object (blob) store (cached): an explicitly-set backend, else the one from
        ``AGENT_ENV_OBJECT_STORE`` / config.toml (``local`` → filesystem default). While an ``@local``
        task runs under namespace routing, its view of it keeps writes local."""
        from agent_env.store import routing

        configured = self._configured_object_store()
        if not routing.in_local_run():
            return configured
        return self._routed("object", configured)

    def get_object_store_for(self, entity_id: str) -> ObjectStore:
        """The object store ``entity_id``'s objects are written to: under namespace routing, the
        per-user local one for an ``@local`` id; otherwise the configured one."""
        from agent_env.store import routing
        from agent_env.store.ids import is_local_id

        configured, local = self._namespace_stores("object")
        if local is None:
            return configured
        if is_local_id(entity_id):
            return local
        routing.refuse_in_local_run(f"writing objects for {entity_id!r}")
        return configured

    def get_object_store_at(self, object_url: str) -> ObjectStore:
        """The object store ``object_url`` addresses: under namespace routing a url the local store
        owns is the local store's, unless the configured store owns it too."""
        from agent_env.store import routing

        configured, local = self._namespace_stores("object")
        return configured if local is None else routing.object_store_holding(object_url, configured, local)

    def get_object_store_to_write(self, object_url: str, entity_id: str) -> ObjectStore:
        """``get_object_store_at`` for writing, or registering, ``entity_id``'s object at ``object_url``."""
        self.check_object_url(entity_id, object_url)
        return self.get_object_store_at(object_url)

    def check_object_url(self, entity_id: str, object_url: str) -> None:
        """Refuse recording ``object_url`` as ``entity_id``'s object across namespaces: under namespace
        routing an ``@local`` entity's objects live in the local store, a bare one's in the configured
        store, and an ``@local`` task run writes only locally."""
        from agent_env.store import routing
        from agent_env.store.ids import is_local_id

        configured, local = self._namespace_stores("object")
        if local is None:
            return
        store = routing.object_store_holding(object_url, configured, local)
        if is_local_id(entity_id) and store is not local:
            raise ValueError(f"{entity_id!r} is an @local id, so its objects go to the local object store, not {object_url!r}")
        if not is_local_id(entity_id) and store is local:
            raise ValueError(f"{entity_id!r} is not an @local id, so its objects go to the configured object store, not {object_url!r}")
        if store is configured:
            routing.refuse_in_local_run(f"a write to {object_url!r}")

    def check_local_run_write(self, entity_id: str) -> None:
        """Refuse writing bare-id ``entity_id`` while an ``@local`` task runs where a configured store
        isn't the per-user one: the write would land in a shared store."""
        from agent_env.store import routing
        from agent_env.store.ids import is_local_id

        if not routing.in_local_run() or is_local_id(entity_id):
            return
        if any(self._namespace_stores(kind)[1] is not None for kind in ("document", "object", "image")):
            routing.refuse_in_local_run(f"writing {entity_id!r}")

    def _configured_object_store(self) -> ObjectStore:
        if self._object_store is None:
            with self._object_store_lock:
                if self._object_store is None:
                    self._object_store = self._build_default_object_store()
        return self._object_store

    def set_object_store(self, store: ObjectStore) -> None:
        """Install an explicit backend; get_object_store() then returns it verbatim."""
        self._object_store = store

    def _build_default_object_store(self) -> ObjectStore:
        from agent_env.store.object_store import ObjectStore

        section = self._resolve_object_section()
        return config_loader.build_store(section, ObjectStore, secret_resolver=self._resolve_secret)

    def _resolve_object_section(self) -> dict:
        return self.trace_section("object").value

    def _object_alias(self, name: str) -> dict:
        if name == _STORE_BACKEND_LOCAL:
            return {
                "impl": "agent_env.store.object_store:LocalFilesystemObjectStore",
                "config": {"root": self._local_object_store_path()},
            }
        raise ConfigError(
            f"Unknown AGENT_ENV_OBJECT_STORE={name!r}: expected 'local', or a [stores.object] table for a "
            "hosted backend (and unset AGENT_ENV_OBJECT_STORE, which overrides it)"
        )

    def _local_object_store_path(self) -> str:
        return str(state_root() / "object_store")

    def get_image_store(self) -> ImageStore:
        """The image (registry) store (cached): an explicitly-set backend, else the one from
        ``AGENT_ENV_IMAGE_STORE`` / config.toml (``local`` → local registry default). While an
        ``@local`` task runs under namespace routing, its view of it keeps pushes local."""
        from agent_env.store import routing

        configured = self._configured_image_store()
        if not routing.in_local_run():
            return configured
        return self._routed("image", configured)

    def get_image_store_for(self, entity_id: str) -> ImageStore:
        """The image store ``entity_id``'s image is pushed to: under namespace routing, the local
        registry for an ``@local`` id; otherwise the configured one."""
        from agent_env.store import routing
        from agent_env.store.ids import is_local_id

        configured, local = self._namespace_stores("image")
        if local is None:
            return configured
        if is_local_id(entity_id):
            return local
        routing.refuse_in_local_run(f"pushing an image for {entity_id!r}")
        return configured

    def get_image_store_at(self, ref: str) -> ImageStore:
        """The image store serving ``ref``: under namespace routing, the local registry for a ref on
        its host; otherwise the configured one."""
        from agent_env.store import routing

        configured, local = self._namespace_stores("image")
        return configured if local is None else routing.image_store_holding(ref, configured, local)

    def _configured_image_store(self) -> ImageStore:
        if self._image_store is None:
            with self._image_store_lock:
                if self._image_store is None:
                    self._image_store = self._build_default_image_store()
        return self._image_store

    def set_image_store(self, store: ImageStore) -> None:
        """Install an explicit backend; get_image_store() then returns it verbatim."""
        self._image_store = store

    def _build_default_image_store(self) -> ImageStore:
        from agent_env.store.image_store import ImageStore

        section = self._resolve_image_section()
        return config_loader.build_store(section, ImageStore, secret_resolver=self._resolve_secret)

    def _resolve_image_section(self) -> dict:
        return self.trace_section("image").value

    def _image_alias(self, name: str) -> dict:
        if name == _STORE_BACKEND_LOCAL:
            return {
                "impl": "agent_env.store.image_store:LocalRegistryImageStore",
                "config": {"registry_host": _DEFAULT_LOCAL_REGISTRY_HOST},
            }
        raise ConfigError(
            f"Unknown AGENT_ENV_IMAGE_STORE={name!r}: expected 'local', or a [stores.image] table for a "
            "hosted registry (and unset AGENT_ENV_IMAGE_STORE, which overrides it)"
        )

    def _local_store(self, kind: str) -> Any:
        """The store of ``kind`` the ``@local`` namespace lives in. Documents get a file of their own
        beside the default local store's, so neither ever holds the other's documents; objects and
        images use the per-user object store and local registry, whose urls and repositories already
        say whose they are."""
        from agent_env.store.document_store import DocumentStore
        from agent_env.store.image_store import ImageStore
        from agent_env.store.object_store import ObjectStore

        with self._routing_lock:
            if kind not in self._local_stores:
                if kind == "document":
                    section, abc = {
                        "impl": "agent_env.store.routing:LocalNamespaceDocumentStore",
                        "config": {"path": self._local_namespace_document_store_path()},
                    }, DocumentStore
                else:
                    alias, abc = {"object": (self._object_alias, ObjectStore), "image": (self._image_alias, ImageStore)}[kind]
                    section = alias(_STORE_BACKEND_LOCAL)
                self._local_stores[kind] = config_loader.build_store(section, abc, secret_resolver=self._resolve_secret)
            return self._local_stores[kind]

    def _namespace_stores(self, kind: str) -> tuple[Any, Any]:
        """The configured store of ``kind`` and, under namespace routing, the ``@local`` namespace's
        store when it is a separate one (else None)."""
        from agent_env.store import routing

        configured = {"document": self._configured_document_store, "object": self._configured_object_store,
                      "image": self._configured_image_store}[kind]()
        local = self._separate_local_store(kind, configured) if routing.namespace_routing_enabled() else None
        return configured, local

    def _separate_local_store(self, kind: str, configured: Any) -> Any:
        """The ``@local`` namespace's store of ``kind``, or None when ``configured`` already is it.
        Decided once per configured store: every store access asks."""
        cached = self._separate_stores.get(kind)
        if cached is not None and cached[0] is configured:
            return cached[1]
        local = self._compare_local_store(kind, configured)
        self._separate_stores[kind] = (configured, local)
        return local

    def _compare_local_store(self, kind: str, configured: Any) -> Any:
        from agent_env.store.document_store import LocalSqliteDocumentStore
        from agent_env.store.image_store import LocalRegistryImageStore
        from agent_env.store.image_store.oci_registry_credentials import normalize_registry_host
        from agent_env.store.object_store import LocalFilesystemObjectStore

        local = self._local_store(kind)
        if kind == "document":
            same = isinstance(configured, LocalSqliteDocumentStore) and configured.path.resolve() == local.path.resolve()
        elif kind == "object":
            same = isinstance(configured, LocalFilesystemObjectStore) and configured.root.resolve() == local.root.resolve()
        else:
            same = isinstance(configured, LocalRegistryImageStore) and (
                normalize_registry_host(configured.registry_host) == normalize_registry_host(local.registry_host)
            )
        return None if same else local

    def _routed(self, kind: str, configured: Any) -> Any:
        """``configured`` wrapped by namespace routing, or ``configured`` itself when it already is
        the ``@local`` namespace's store."""
        from agent_env.store import routing

        local = self._separate_local_store(kind, configured)
        if local is None:
            if kind == "document":
                raise ConfigError(
                    f"{self._local_namespace_document_store_path()} is kept for @local documents; "
                    "point [stores.document] somewhere else"
                )
            return configured
        with self._routing_lock:
            routed = self._routed_stores.get(kind)
            if routed is None or routed.configured is not configured:
                if kind == "document":
                    routed = routing.RoutingDocumentStore(configured, local, local.path)
                elif kind == "object":
                    routed = routing.LocalRunObjectStore(configured, local)
                else:
                    routed = routing.LocalRunImageStore(configured, local)
                self._routed_stores[kind] = routed
            return routed

    def get_runner(self):
        """The configured Runner (cached): an explicitly-set one, else ``[runner]`` /
        ``AGENT_ENV_RUNNER`` (``local`` default -> the in-process asyncio runner)."""
        if self._runner is None:
            with self._runner_lock:
                if self._runner is None:
                    self._runner = self._build_default_runner()
        return self._runner

    def set_runner(self, runner) -> None:
        """Install an explicit runner; get_runner() then returns it verbatim."""
        self._runner = runner

    def _build_default_runner(self):
        from agent_env.config import loader as _loader
        from agent_env.runner.runner import Runner

        section = self._resolve_runner_section()
        return _loader.build_store(section, Runner, secret_resolver=self._resolve_secret)

    def _resolve_runner_section(self) -> dict:
        return self.trace_section("runner").value

    def _runner_alias(self, name: str) -> dict:
        if name == _RUNNER_BACKEND_LOCAL:
            return {"impl": "agent_env.runner.local_runner:LocalRunner", "config": {}}
        raise ConfigError(f"Unknown AGENT_ENV_RUNNER={name!r} (expected 'local', or a [runner] table)")

    def get_secret_store(self) -> SecretStore:
        """The secret store (cached) that ``secret:`` references resolve through: an explicitly-set
        backend, else ``AGENT_ENV_SECRET_STORE`` / config.toml (``local`` default → env/file)."""
        if self._secret_store is None:
            with self._secret_store_lock:
                if self._secret_store is None:
                    self._secret_store = self._build_default_secret_store()
        return self._secret_store

    def _resolve_secret(self, name: str) -> Optional[str]:
        """Bound so the secret store is built only when a ``secret:`` reference is actually met."""
        return self.get_secret_store().get(name)

    def set_secret_store(self, store: SecretStore) -> None:
        """Install an explicit backend; get_secret_store() then returns it verbatim."""
        self._secret_store = store

    def _build_default_secret_store(self) -> SecretStore:
        from agent_env.store.secret_store import SecretStore

        section = self._resolve_secret_section()
        return config_loader.build_store(section, SecretStore)

    def _resolve_secret_section(self) -> dict:
        return self.trace_section("secret").value

    def _secret_alias(self, name: str) -> dict:
        if name == _STORE_BACKEND_LOCAL:
            return {"impl": "agent_env.store.secret_store:LocalSecretStore", "config": {}}
        raise ConfigError(
            f"Unknown AGENT_ENV_SECRET_STORE={name!r}: expected 'local', or a [stores.secret] table for a "
            "hosted backend (and unset AGENT_ENV_SECRET_STORE, which overrides it)"
        )

    def get_artifact_key_prefix(self) -> str:
        """Key prefix for every object key core builds: artifacts, image builds, env and agent
        snapshots, changelogs, default trajectories, verifier outputs and validator fixtures (set
        via AGENT_ENV_FIXTURE_PREFIX; empty in prod)."""
        return f"{self.fixture_prefix}/" if self.fixture_prefix else ""


# One row per aliased section, carrying every coordinate that section has. The resolvers and
# the report both read this, so neither can drift from the other. The alias is held as a
# function object rather than a method name: a rename then fails at import, not at use.
ALIASED_SECTIONS = (
    AliasedSection("document", "AGENT_ENV_DOCUMENT_STORE", ("stores", "document"),
                   _STORE_BACKEND_LOCAL, Config._document_alias),
    AliasedSection("object", "AGENT_ENV_OBJECT_STORE", ("stores", "object"),
                   _STORE_BACKEND_LOCAL, Config._object_alias),
    AliasedSection("image", "AGENT_ENV_IMAGE_STORE", ("stores", "image"),
                   _STORE_BACKEND_LOCAL, Config._image_alias),
    AliasedSection("secret", "AGENT_ENV_SECRET_STORE", ("stores", "secret"),
                   _STORE_BACKEND_LOCAL, Config._secret_alias),
    AliasedSection("runner", "AGENT_ENV_RUNNER", ("runner",),
                   _RUNNER_BACKEND_LOCAL, Config._runner_alias),
)
_BY_NAME = {section.name: section for section in ALIASED_SECTIONS}

_config: Optional[Config] = None
_config_lock = threading.Lock()
# Registry names this thread is building, on any Config; see Config._registry.
_registries_building = threading.local()


def in_nested_registry_build() -> bool:
    """Whether this thread is inside a re-entered, unmemoized registry build."""
    return _registries_building.__dict__.get("nested", 0) > 0


def configure(
    *,
    default_human_a2a_url: Optional[str] = None,
    default_a2a_agent_id: Optional[str] = None,
    document_store: Optional[DocumentStore] = None,
    image_store: Optional[ImageStore] = None,
    object_store: Optional[ObjectStore] = None,
    secret_store: Optional[SecretStore] = None,
) -> None:
    """Rebuild the process-wide config from the config file ``AGENT_ENV_CONFIG`` resolves to.

    Pass ``document_store`` / ``image_store`` / ``object_store`` / ``secret_store`` to install explicit backends on the new config.

    ``default_human_a2a_url`` overrides config.toml ``[conversations]`` but not
    ``AGENT_ENV_HUMAN_A2A_URL``. ``default_a2a_agent_id`` overrides config.toml ``[agents]``.

    Select the stage *before* calling this, by pointing ``AGENT_ENV_CONFIG`` at a per-stage
    config file (a plugin can do this for you, for instance from a CLI root option it registers).
    """
    global _config
    overrides: dict[str, Any] = {}
    if default_human_a2a_url is not None:
        overrides["default_human_a2a_url"] = default_human_a2a_url
    if default_a2a_agent_id is not None:
        overrides["default_a2a_agent_id"] = default_a2a_agent_id
    config_path = config_loader.discover_config_path()
    _config = Config(**overrides)
    if document_store is not None:
        _config.set_document_store(document_store)
    if image_store is not None:
        _config.set_image_store(image_store)
    if object_store is not None:
        _config.set_object_store(object_store)
    if secret_store is not None:
        _config.set_secret_store(secret_store)
    logger.info("Store config resolved: %s", config_path or "(no config file)")


def set_document_store(store: DocumentStore) -> None:
    """Override the process-wide control-plane document store backend."""
    get_config().set_document_store(store)


def set_image_store(store: ImageStore) -> None:
    """Override the process-wide image (registry) store backend."""
    get_config().set_image_store(store)


def set_object_store(store: ObjectStore) -> None:
    """Override the process-wide object (blob) store backend."""
    get_config().set_object_store(store)


def set_secret_store(store: SecretStore) -> None:
    """Override the process-wide secret store backend."""
    get_config().set_secret_store(store)


def set_runner(runner) -> None:
    """Install an explicit Runner on the singleton config."""
    get_config().set_runner(runner)


def get_runner():
    """The configured Runner (see ``[runner]`` in .agentenv/config.toml)."""
    return get_config().get_runner()


def get_config() -> Config:
    """The process-wide Config, built on first use.

    Locked: the Config owns a document now, so a losing racer's — and anything built from
    it — would be discarded while its caller kept using it.
    """
    global _config
    cached = _config
    if cached is not None:
        return cached
    with _config_lock:
        if _config is None:
            _config = Config()
        return _config


def reset_config() -> None:
    """Drop the process-wide config, so the next read resolves the file again.

    The registries and stores the Config holds go with it, and a store singleton resolves
    its backend per call, so it follows. Only an explicitly installed object persists.
    """
    global _config
    _config = None
