"""The EnvStateProvider abstracts WHERE an environment's data/state lives.

A provider's lifecycle splits into three separable stages so backends can vary (or skip) each
independently:

  1. **Allocate** (``acquire``) — reserve a private, credentialed store for one run, keyed on run
     identity (ttl, name_hint, …), NOT on the services that will use it.
  2. **Prepare** (``prepare``) — idempotently make the store ready for a specific set of services 
     (assumes a per-service schemas/structure). 
  3. **Load** (the ``load_environment_universe_artifact`` path) — populate rows.

Separating these keeps the per-run deploy path honest: the service set is assembled dynamically
by the gateway, and the load stage may run or be skipped (e.g. a pre-materialized base). Coupling
the stages is appropriate only for deliberate out-of-band materialization, where the full service
set + data are known up front.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from agent_env.plugins import _registration

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from agent_env.config.runtime import Config

logger = logging.getLogger(__name__)

# Tag for the built-in state store. Defined here (not as the provider's ``type``) so it can be
# referenced without importing a provider and causing an import cycle. Every other backend is
# registered by a plugin or from config.toml and names itself.
LOCAL_POSTGRES_STATE_TYPE = "local_postgres"

# Default lifetime of an acquired store, stamped onto the instance's expires_at.
DEFAULT_STATE_TTL_SECONDS = 10800

# Format for an EnvStateInstance's created_at/expires_at strings. Lives here (with the model) as the
# single source of truth; the store imports it to stamp timestamps, is_expired parses with it.
TS_FMT = "%Y-%m-%d %H:%M UTC"

@dataclass
class StateContext:
    """Base class to provide context to the state provider on how to acquire a connection to
    the store.
    """


@dataclass
class EnvStateInstance:
    """Persisted record of one secured per-run store (collection ``env_state_instances``).

    The output of ``acquire`` and the unit ``teardown`` releases. It is an *addressable
    state object*, NOT owned by a single run/env — deploys *reference* it (via
    ``DeployedEnv.env_state_instance_ids``), many-to-many over time.

    ``_db_url_base`` is the live connection URL (may embed a per-run secret); it is used to
    wire ``DATABASE_URL`` for THIS deploy only and is NEVER persisted.
    Reattaching a state instance re-mints credentials.
    """

    state_type: str
    instance_id: str = ""
    metadata: dict = field(default_factory=dict)
    created_at_utc: str | None = None
    expires_at_utc: str | None = None
    _db_url_base: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "state_type": self.state_type,
            "metadata": self.metadata,
            "created_at_utc": self.created_at_utc,
            "expires_at_utc": self.expires_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvStateInstance":
        return cls(
            instance_id=data["instance_id"],
            state_type=data["state_type"],
            metadata=data.get("metadata") or {},
            created_at_utc=data.get("created_at_utc"),
            expires_at_utc=data.get("expires_at_utc"),
        )

    def is_expired(self) -> bool:
        from datetime import datetime, timezone

        if not self.expires_at_utc:
            return False
        try:
            dt = datetime.strptime(self.expires_at_utc, TS_FMT).replace(tzinfo=timezone.utc)
        except ValueError:
            return False
        return datetime.now(timezone.utc) >= dt


class EnvStateProvider(ABC):
    """Secures a per-run instance of an env's state and persists the record to reach it."""

    # The ``state_type`` tag this impl handles; set by each subclass.
    type: str

    # Whether this backend supports the servicedb-image snapshot fast-path (capture a loaded
    # universe as a store image, then restore it to skip the slow per-service load).
    supports_restore_from_snapshot: bool = False

    @classmethod
    def from_config(cls, **config) -> "EnvStateProvider":
        """Construct from a resolved config table; backends that need custom wiring override this."""
        return cls(**config)

    def deploy_state_context(
        self, *, ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS, name_hint: str | None = None
    ) -> StateContext | None:
        """This backend's ``acquire`` context for a deploy, built from run identity — or ``None`` to
        allocate nothing upstream of the gateway. Raises rather than defaulting to ``None`` so a
        backend that forgets it fails loud instead of silently getting a local store instead."""
        raise NotImplementedError(f"env state type not wired into the deploy path yet: {type(self).__name__}")

    @abstractmethod
    async def acquire(self, ctx: StateContext) -> EnvStateInstance:
        """Stage 1 (allocate): secure a private store for this run and persist its
        ``EnvStateInstance`` record."""
        ...

    async def prepare(
        self,
        environment_names: list[str],
        *,
        instance: EnvStateInstance | None = None,
    ) -> None:
        """Stage 2 (prepare): idempotently make the acquired store ready for ``environment_names`` — the
        per-service structure they'll use."""
        return None

    async def attach(
        self, instance: EnvStateInstance, *, ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS, name_hint: str | None = None
    ) -> EnvStateInstance:
        """Attach a deploy to an EXISTING env state instance"""
        if instance.is_expired():
            raise ValueError(
                f"env state instance {instance.instance_id!r} is expired "
                f"(expires_at_utc={instance.expires_at_utc!r}); provision a fresh one."
            )
        return await self._attach(instance, ttl_seconds=ttl_seconds, name_hint=name_hint)

    async def _attach(
        self, instance: EnvStateInstance, *, ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS, name_hint: str | None = None
    ) -> EnvStateInstance:
        return instance

    async def teardown(self, instance: EnvStateInstance) -> None:
        """Release whatever ``acquire`` allocated and retire the record."""
        from agent_env.providers.state.store import retire_env_state_instance
        try:
            await self._teardown(instance)
        finally:
            retire_env_state_instance(instance.instance_id)

    @abstractmethod
    async def _teardown(self, instance: EnvStateInstance) -> None:
        ...

class DatabaseStateProvider(EnvStateProvider):
    """Intermediate base for DB-backed state providers."""

    def base_url(self, instance: EnvStateInstance) -> str:
        """Unscoped connection URL"""
        return instance._db_url_base

    @abstractmethod
    def url_for_environment(self, environment_name: str, *, instance: EnvStateInstance) -> str:
        """Per-service ``DATABASE_URL``, scoped to the service's schema."""
        ...

    @abstractmethod
    def env_state_docker_service_name(self) -> str:
        """Compose service name of this backend's readiness service — clients ``depends_on`` it, the
        changelog shim execs into it, and ``render_healthcheck_service`` names its container from it."""
        ...

    @abstractmethod
    def render_healthcheck_service(
        self,
        environment_names: list[str],
        *,
        instance: EnvStateInstance | None = None,
    ) -> list[str]:
        """Compose lines for the ONE readiness service — named ``env_state_docker_service_name``,
        carrying a ``healthcheck`` — that clients ``depends_on``. Every DB backend must render one."""
        ...

    def render_sidecar_containers(
        self,
        environment_names: list[str],
        *,
        host_port: Optional[Callable[[int], int]] = None,
    ) -> list[str]:
        """Extra co-deployed compose services beyond the readiness one (e.g. pgweb / db-mcp browse
        UIs). Optional: a backend may have none, in which case this returns an empty list."""
        return []

    def render_compose_containers(
        self,
        environment_names: list[str],
        *,
        instance: EnvStateInstance | None = None,
        host_port: Optional[Callable[[int], int]] = None,
    ) -> list[str]:
        """All co-deployed compose services for this backend — the mandatory readiness service plus
        any sidecars — spliced into the gateway's docker-compose doc.

        ``host_port`` maps a container port to the host port to publish it on; defaults
        to identity.
        """
        return [
            *self.render_healthcheck_service(environment_names, instance=instance),
            *self.render_sidecar_containers(environment_names, host_port=host_port),
        ]

    def docker_service_dependency(self) -> list[str]:
        """The ``depends_on`` fragment gating a client on the readiness service. Non-optional —
        every DB backend renders a readiness service, so there's always an edge to add."""
        return [
            f"      {self.env_state_docker_service_name()}:",
            "        condition: service_healthy",
        ]

    async def install_changelog_triggers(
        self,
        environment_name: str,
        *,
        instance: EnvStateInstance | None = None,
        store_exec: "Callable[[list[str]], Awaitable[tuple[int, str, str]]] | None" = None,
    ) -> None:
        """Install per-schema changelog triggers. 

        ``instance`` may be ``None``: the changelog shim's reattach/snapshot fallback has no
        acquired instance, so a backend that needs one must guard for it here.
        """
        return None

    # --- Shared Postgres init SQL (schemas + changelog) ---------------------------------

    @staticmethod
    def get_init_script(environment_names: list[str]) -> str:
        """PostgreSQL init script: creates one schema per MCP service + the shared changelog objects."""
        lines = [f'CREATE SCHEMA IF NOT EXISTS "{name}";' for name in environment_names]
        lines.append(DatabaseStateProvider._CHANGELOG_SQL)
        return "\n".join(lines)

    # The changelog audit objects — the `_install_changelog_triggers` function this defines is
    # what ``install_changelog_triggers`` calls per schema.
    _CHANGELOG_SQL = """
-- Sequence lives off `public` so it's invisible to BaseService.sync_sequences,
-- which otherwise rewinds it under concurrent /api/reset and breaks
-- _changelog_pkey. No OWNED BY: cross-schema link isn't allowed and we don't
-- drop _changelog anyway.
CREATE SCHEMA IF NOT EXISTS agent_env_internal;
CREATE SEQUENCE IF NOT EXISTS agent_env_internal._changelog_id_seq;
CREATE TABLE IF NOT EXISTS public._changelog (
    id BIGINT PRIMARY KEY DEFAULT nextval('agent_env_internal._changelog_id_seq'),
    "timestamp" TIMESTAMPTZ NOT NULL DEFAULT now(),
    schema_name TEXT NOT NULL,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('INSERT','UPDATE','DELETE')),
    row_id TEXT,
    summary TEXT NOT NULL,
    changed_fields JSONB
);

CREATE OR REPLACE FUNCTION public._changelog_trigger_fn()
RETURNS TRIGGER AS $$
DECLARE
    _row_id TEXT;
    _summary TEXT;
    _changed JSONB;
    _old_json JSONB;
    _new_json JSONB;
    _key TEXT;
    _old_val TEXT;
    _new_val TEXT;
    _changes TEXT[];
    _pk_cols TEXT[];
    _rec JSONB;
BEGIN
    -- Serialize concurrent writers to the shared public._changelog table.
    -- Re-entrant within a transaction (a multi-thousand-row bulk load takes
    -- it once on the first trigger fire; subsequent fires in the same xact
    -- are no-ops). Across transactions, the second waits until the first
    -- commits — eliminates deadlocks and serial-PK collisions during
    -- concurrent /load-universe resets without serializing user-edit
    -- traffic beyond the changelog write itself.
    PERFORM pg_advisory_xact_lock(hashtext('public._changelog'));

    -- Capture the changed row's identity from the table's ACTUAL primary key
    -- (not a hardcoded 'id' column — universe tables use varied PKs, e.g.
    -- record_id). Single-column PKs are stored as the plain value; composite
    -- PKs as an unambiguous JSON object (so values containing a delimiter can't
    -- collide); PK-less tables fall back to an 'id' column if present, else NULL.
    IF TG_OP = 'DELETE' THEN _rec := row_to_json(OLD)::jsonb;
    ELSE _rec := row_to_json(NEW)::jsonb; END IF;

    SELECT array_agg(a.attname ORDER BY k.ord)
    INTO _pk_cols
    FROM pg_index i
    JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord) ON true
    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
    WHERE i.indrelid = TG_RELID AND i.indisprimary;

    IF _pk_cols IS NULL THEN
        _row_id := _rec ->> 'id';
    ELSIF array_length(_pk_cols, 1) = 1 THEN
        _row_id := _rec ->> _pk_cols[1];
    ELSE
        SELECT jsonb_object_agg(col, _rec -> col)::text
        INTO _row_id
        FROM unnest(_pk_cols) AS col;
    END IF;

    IF TG_OP = 'INSERT' THEN
        _changed := row_to_json(NEW)::jsonb;
        _summary := 'Inserted: ' || left(_changed::text, 200);

    ELSIF TG_OP = 'DELETE' THEN
        _changed := row_to_json(OLD)::jsonb;
        _summary := 'Deleted: ' || left(_changed::text, 200);

    ELSIF TG_OP = 'UPDATE' THEN
        _old_json := row_to_json(OLD)::jsonb;
        _new_json := row_to_json(NEW)::jsonb;
        _changed := '{}'::jsonb;
        _changes := ARRAY[]::TEXT[];

        FOR _key IN SELECT jsonb_object_keys(_new_json)
        LOOP
            IF _old_json -> _key IS DISTINCT FROM _new_json -> _key THEN
                _changed := _changed || jsonb_build_object(
                    _key, jsonb_build_object('old', _old_json -> _key, 'new', _new_json -> _key)
                );
                _old_val := left(_old_json ->> _key, 50);
                _new_val := left(_new_json ->> _key, 50);
                _changes := array_append(
                    _changes,
                    _key || ': ' || COALESCE(quote_literal(_old_val), 'NULL')
                         || ' -> ' || COALESCE(quote_literal(_new_val), 'NULL')
                );
            END IF;
        END LOOP;

        _summary := array_to_string(_changes, ', ');
    END IF;

    INSERT INTO public._changelog (schema_name, table_name, operation, row_id, summary, changed_fields)
    VALUES (TG_TABLE_SCHEMA, TG_TABLE_NAME, TG_OP, _row_id, _summary, _changed);

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION public._install_changelog_triggers(_schema_name TEXT)
RETURNS void AS $$
DECLARE
    _tbl RECORD;
BEGIN
    DELETE FROM public._changelog WHERE schema_name = _schema_name;

    FOR _tbl IN
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE schemaname = _schema_name
          AND tablename != '_changelog'
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS _changelog_trigger ON %I.%I',
            _tbl.schemaname, _tbl.tablename
        );
        EXECUTE format(
            'CREATE TRIGGER _changelog_trigger '
            'AFTER INSERT OR UPDATE OR DELETE ON %I.%I '
            'FOR EACH ROW EXECUTE FUNCTION public._changelog_trigger_fn()',
            _tbl.schemaname, _tbl.tablename
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;
"""


# Declared beside the function that reads them so the config report cannot describe a
# group this module has since changed. `AGENT_ENV_RDS_HOST` selects the group.
RDS_ADMIN_ENV_VARS = (
    "AGENT_ENV_RDS_HOST", "AGENT_ENV_RDS_PORT", "AGENT_ENV_RDS_DBNAME",
    "AGENT_ENV_RDS_USERNAME", "AGENT_ENV_RDS_PASSWORD", "AGENT_ENV_RDS_SSLMODE",
    "AGENT_ENV_RDS_AUTH", "AGENT_ENV_RDS_REGION",
)


def admin_config_from_env() -> dict | None:
    """An explicit admin connection from ``AGENT_ENV_RDS_*``, or ``None`` when unset — the local /
    integration escape hatch that skips a backend's own credential lookup. ``AGENT_ENV_RDS_HOST``
    alone selects it, and it is then all-or-nothing: no per-key merge with the backend's own."""
    host = os.getenv("AGENT_ENV_RDS_HOST")
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.getenv("AGENT_ENV_RDS_PORT", "5432")),
        "dbname": os.getenv("AGENT_ENV_RDS_DBNAME", "postgres"),
        "username": os.getenv("AGENT_ENV_RDS_USERNAME", "agentenv"),
        "password": os.getenv("AGENT_ENV_RDS_PASSWORD", ""),
        "sslmode": os.getenv("AGENT_ENV_RDS_SSLMODE", "require"),
        "auth": os.getenv("AGENT_ENV_RDS_AUTH", "password"),
        "region": os.getenv("AGENT_ENV_RDS_REGION"),
    }


# --- Provider registry: built-ins + config.toml [state.providers] (mirrors [sandbox.providers]) ---

_BUILTIN_STATE_PROVIDERS: dict[str, str] = {
    LOCAL_POSTGRES_STATE_TYPE: "agent_env.providers.state.local_postgres:LocalPostgresStateProvider",
}



def _state_config(source: Config | None = None) -> dict:
    from agent_env.config import runtime
    

    return (source or runtime.get_config()).section("state")


def _build_registry(source: Config | None = None) -> dict[str, dict]:
    """Built-ins, then ``agent_env.state_providers`` plugins, then config.toml
    ``[state.providers]`` declared in ``source``."""
    registry: dict[str, dict] = {name: {"impl": impl} for name, impl in _BUILTIN_STATE_PROVIDERS.items()}
    from_plugins = _registration.merge(
        registry, _registration.STATE_PROVIDERS, _validate_plugin, source=source, entry=lambda cls: {"impl": cls}
    )
    _merge_config_toml_state_providers(registry, source=source, from_plugins=from_plugins)
    return registry


def _validate_plugin(name: str, loaded: Any) -> type[EnvStateProvider]:
    cls = _registration.require_subclass(loaded, EnvStateProvider)
    if getattr(cls, "type", None) != name:
        raise TypeError(
            f"{cls.__qualname__} has type {getattr(cls, 'type', None)!r}; it must equal the entry-point "
            "name, the identity a store is reattached and torn down by"
        )
    return cls


def _get_state_registry() -> dict[str, dict]:
    from agent_env.config import runtime

    return runtime.get_config().state_registry()


def _merge_config_toml_state_providers(
    registry: dict[str, dict],
    *,
    source: Config | None = None,
    from_plugins: _registration.Registrations | None = None,
) -> None:
    """Register ``[state.providers]`` entries (a ``module:Class`` string or a table with ``impl`` +
    optional ``config``) under their name. A table for a PLUGIN's name may carry ``config`` only,
    or an ``impl`` that replaces the plugin's with a warning. Collisions with a built-in and bad
    impls fail loud."""
    from agent_env.config import ConfigError, load_impl

    providers = _state_config(source).get("providers", {})
    if not isinstance(providers, dict):
        raise ConfigError(f"config.toml [state.providers] must be a table, got {type(providers).__name__}")

    from_plugins = from_plugins if from_plugins is not None else _registration.Registrations.empty()
    for name, entry in providers.items():
        plugin = from_plugins.plugin(name)
        if name in registry and plugin is None:
            raise ConfigError(f"config.toml state provider {name!r} collides with a built-in backend")
        if not isinstance(entry, (str, dict)):
            raise ConfigError(f"config.toml state provider {name!r} must be a 'module:Class' string or a "
                              f"table with 'impl', got {type(entry).__name__}")
        if isinstance(entry, dict) and "impl" not in entry:
            if plugin is None and from_plugins.failed(name):
                logger.warning("[state.providers.%s] configures a plugin that failed to load; skipped", name)
                continue
            if plugin is not None:
                unknown = sorted(set(entry) - {"config"})
                if unknown:
                    raise ConfigError(f"[state.providers.{name}] configures plugin {plugin} and accepts "
                                      f"only a 'config' table; got {unknown}")
                config = entry.get("config", {})
                if not isinstance(config, dict):
                    raise ConfigError(f"[state.providers.{name}].config must be a table, got {type(config).__name__}")
                registry[name] = {**registry[name], "config": dict(config)}
                continue
        section = {"impl": entry} if isinstance(entry, str) else dict(entry)
        impl = section.get("impl")
        if not impl:
            raise ConfigError(f"config.toml state provider {name!r} is missing an 'impl' dotted path")
        cls = load_impl(impl, EnvStateProvider)
        state_type = getattr(cls, "type", None)
        if state_type != name:
            raise ConfigError(f"config.toml state provider {name!r} declares impl {impl!r} with type "
                              f"{state_type!r}; they must match — the name is the identity a store is "
                              "reattached and torn down by")
        if stray := set(section) - {"impl", "config"}:
            logger.warning("config.toml state provider %r: ignoring %s next to 'impl' — constructor kwargs "
                           "belong under [state.providers.%s.config]", name, ", ".join(sorted(stray)), name)
        from_plugins.release(name, f"[state.providers.{name}]", cls)
        registry[name] = section


def build_state_provider(env_state_type: str) -> EnvStateProvider:
    """Map an ``env_state_type`` tag to a provider impl through the open registry — built-ins plus
    config.toml ``[state.providers]`` (mirrors ``build_sandbox_provider``). The name is the impl's own
    ``type``: the identity persisted on the instance and rebuilt from on reattach/teardown."""
    from agent_env.config import get_config, interpolate, load_impl

    registry = _get_state_registry()
    if env_state_type not in registry:
        note = _registration.failure_note(_registration.STATE_PROVIDERS, env_state_type)
        raise ValueError(f"Unknown env state type: {env_state_type!r}{note} (expected one of {sorted(registry)})")
    section = registry[env_state_type]
    cls = load_impl(section["impl"], EnvStateProvider)
    config = interpolate(
        section.get("config", {}), secret_resolver=lambda key: get_config().get_secret_store().get(key)
    )
    return cls.from_config(**config)


async def acquire_state_for_deploy(
    *,
    env_state_type: str | None = None,
    ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS,
    name_hint: str | None = None,
    env_state_instance_id: str | None = None,
) -> "EnvStateInstance | None":
    """Reserve the run's env state instance for a deploy, or ``None`` when the store is stood up later
    by the gateway. Called upstream in ``Env.deploy`` (before the gateway/sandbox exists).

    - no ``env_state_instance_id`` → create a fresh store via ``env_state_type``.
    - ``env_state_instance_id`` given → attach to that existing instance; its own type selects the
      provider (``env_state_type``, if also passed, must match).
    """
    # Attach path: provider is derived from the named instance, not from env_state_type.
    if env_state_instance_id:
        from agent_env.providers.state.store import get_env_state_instance_store

        instance = get_env_state_instance_store().get(env_state_instance_id)  # raises NotFoundError
        # Reconcile the caller's requested type against the instance's own (the non-expired guard is
        # enforced by attach).
        if env_state_type and instance.state_type != env_state_type:
            raise ValueError(
                f"env_state_instance {env_state_instance_id!r} is a {instance.state_type!r} store, but "
                f"env_state_type={env_state_type!r} was requested — omit env_state_type when attaching, "
                "or pass an instance of the requested type."
            )
        provider = build_state_provider(instance.state_type)
        return await provider.attach(instance, ttl_seconds=ttl_seconds, name_hint=name_hint)

    # Create-fresh
    provider = build_state_provider(env_state_type or LOCAL_POSTGRES_STATE_TYPE)
    ctx = provider.deploy_state_context(ttl_seconds=ttl_seconds, name_hint=name_hint)
    if ctx is None:
        return None
    return await provider.acquire(ctx)
