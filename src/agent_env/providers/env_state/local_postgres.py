"""LocalPostgresStateProvider — today's shared per-service Postgres as an EnvStateProvider.

The environment provider ( compute layer ) stands up the ``servicedb`` container (from this provider's
declared ``store_spec``) and hands its address back on the acquire context. ``acquire`` does NOT
create the container and does NOT load data — it builds the connection URL, persists an
``EnvStateInstance``, and returns it. ``teardown`` only marks the record retired (sets
``expires_at_utc`` to now): the container itself is reaped with the sandbox.

All the Postgres store knowledge lives here — connection URLs, the schema + changelog init
SQL, and the container healthcheck. ``ServiceDBEnv`` keeps only the docker image artifacts
(its ``ServiceDBConfig``); this provider owns how the store is set up and run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from agent_env.config import get_config
from agent_env.env.envs.service_db import (
    DATABASE_SERVICE_NAME,
    DB_MCP_CONTAINER_PORT,
    DB_MCP_PORT,
    DB_MCP_SERVICE_NAME,
    DB_NAME,
    DB_PASSWORD,
    DB_USER,
    DB_WEB_PORT,
    PGWEB_SERVICE_NAME,
    SERVICE_DB_PORT,
)
from agent_env.providers.sandbox_providers.sandbox import port_bindings
from agent_env.providers.env_state.env_state_provider import (
    DatabaseStateProvider,
    DEFAULT_STATE_TTL_SECONDS,
    EnvStateInstance,
    StateContext,
    LOCAL_POSTGRES_STATE_TYPE,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_env.artifact import DockerImageArtifact
    from agent_env.env.envs.service_db import ServiceDBConfig, ServiceDBEnv

logger = logging.getLogger(__name__)

# Where the schema-init SQL is mounted inside the Postgres container — the image's
# entrypoint runs everything here on first boot. A Postgres-image convention, so it's
# owned by this backend rather than hardcoded in the compute layer.
INIT_SCRIPT_MOUNT_PATH = "/docker-entrypoint-initdb.d/init-schemas.sql"

_STOCK_POSTGRES_IMAGE = "public.ecr.aws/docker/library/postgres:16-alpine"


@dataclass
class ContainerSpec:
    """Declarative spec for a store/sidecar container the compute layer provisions (Modal
    path). The gateway's ``create_container`` loop consumes these and adds its own sandbox
    knobs (i6pn, attribution)
    """

    name: str
    image: str
    env: dict[str, str]
    port: int
    command: list[str] | None = None


@dataclass
class LocalPostgresStoreSpec:
    """The servicedb container the compute layer stands up for the local-Postgres backend."""

    env: dict[str, str]
    healthcheck: dict
    init_sql: str
    port: int


@dataclass
class LocalPostgresStateContext(StateContext):
    """``acquire`` input for the local-Postgres backend.

    environment_names: the services this shared store must host, one schema each.
    host: the servicedb address, supplied by the compute layer after it stands the
        container up — e.g. ``"servicedb"`` (compose) or ``"[fdaa::…]"`` (i6pn container).
    ttl_seconds: the deploy TTL, used to stamp the record's ``expires_at_utc``.
    """

    environment_names: list[str]
    host: str
    ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS


class LocalPostgresStateProvider(DatabaseStateProvider):
    type = LOCAL_POSTGRES_STATE_TYPE

    # The co-deployed servicedb is snapshottable as a container image
    supports_restore_from_snapshot = True

    def __init__(self, service_db_config: "ServiceDBConfig | None" = None):
        """``service_db_config`` (the servicedb image config) is static per provider, so it's held
        here rather than threaded through each render call. Pass it to inject baked image tags
        (export/snapshot); ``None`` resolves the deploy's default ServiceDBEnv lazily."""
        super().__init__()
        self._service_db_config = service_db_config
        self._service_db_env: "ServiceDBEnv | None" = None

    def deploy_state_context(
        self, *, ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS, name_hint: str | None = None
    ) -> None:
        """None: its context needs the compose host, so the gateway acquires this store, not the deploy."""
        return None

    def _default_service_db_env(self) -> "ServiceDBEnv":
        """The deploy's default ServiceDBEnv, resolved once + cached. Source of both the image
        config (names, for compose) and the images to load (artifacts, for the VM). Also caches
        ``service_db_config`` off the same resolve unless one was injected at construction."""
        if self._service_db_env is None:
            from agent_env.env.env import Env
            from agent_env.config import get_config

            self._service_db_env = Env.get(get_config().default_service_db_env_id)
            if self._service_db_config is None:
                self._service_db_config = self._service_db_env.to_config()
        return self._service_db_env

    @property
    def service_db_config(self) -> "ServiceDBConfig":
        """The injected config, else the deploy's default ServiceDBEnv config (resolved once)."""
        if self._service_db_config is None:
            self._default_service_db_env()  # resolves the env and caches _service_db_config
        return self._service_db_config

    def store_images_to_load(self) -> list["DockerImageArtifact"]:
        """The servicedb image set (servicedb + pgweb + db-mcp) the VM preloads via S3 tar — the
        artifacts (with ``tar_gz_s3_url``), which the image-name config can't provide. Reuses the
        cached default env, so no extra ``Env.get`` beyond ``service_db_config``'s."""
        env = self._default_service_db_env()
        return [
            env.db_docker_image_artifact,
            env.db_web_docker_image_artifact,
            env.db_mcp_docker_image_artifact,
        ]

    @classmethod
    def default_instance(cls) -> EnvStateInstance:
        """Static (unpersisted) instance for the co-deployed local store, used when reattaching an
        env that has no recorded state instance (ie. envs deployed before EnvStateInstance was created, or the 
        EnvStateInstance is expired). """
        return EnvStateInstance(
            state_type=cls.type,
            metadata={"host": DATABASE_SERVICE_NAME},
            _db_url_base=cls._pg_url(DATABASE_SERVICE_NAME),  # static compose-network base
        )

    def store_spec(self, environment_names: list[str]) -> LocalPostgresStoreSpec:
        return LocalPostgresStoreSpec(
            env={
                "POSTGRES_USER": DB_USER,
                "POSTGRES_PASSWORD": DB_PASSWORD,
                "POSTGRES_DB": DB_NAME,
            },
            healthcheck={
                "test": ["CMD-SHELL", f"pg_isready -U {DB_USER}"],
                "interval": "5s",
                "timeout": "5s",
                "retries": 5,
            },
            init_sql=self.get_init_script(environment_names),
            port=SERVICE_DB_PORT,
        )

    async def acquire(self, ctx: StateContext) -> EnvStateInstance:
        if not isinstance(ctx, LocalPostgresStateContext):
            raise TypeError(
                "LocalPostgresStateProvider.acquire requires a LocalPostgresStateContext, "
                f"got {type(ctx).__name__}."
            )
        if not ctx.host:
            raise ValueError(
                "LocalPostgresStateProvider.acquire requires ctx.host — the servicedb "
                "address, supplied by the compute layer after standup."
            )
        from agent_env.providers.env_state.store import register_env_state_instance

        db_url_base = self._pg_url(ctx.host)  # base URL, no search_path
        instance = EnvStateInstance(
            state_type=self.type,
            metadata={"host": ctx.host},
            _db_url_base=db_url_base,
        )
        return register_env_state_instance(instance, ctx.ttl_seconds)

    async def _teardown(self, instance: EnvStateInstance) -> None:
        # The servicedb container is reaped with the sandbox/VM — nothing to release here
        return None

    # --- URL construction (single source) ----------------------------------------------

    @staticmethod
    def _pg_url(
        host: str,
        *,
        port: int = SERVICE_DB_PORT,
        scheme: str = "postgresql",
        sslmode: str | None = None,
    ) -> str:
        """Build the local postgres (servicedb) connection URL from the store creds."""
        url = f"{scheme}://{DB_USER}:{DB_PASSWORD}@{host}:{port}/{DB_NAME}"
        return f"{url}?sslmode={sslmode}" if sslmode else url

    # --- DatabaseStateProvider seam (Postgres/compose specifics live here) -------------

    def base_url(self, instance: EnvStateInstance) -> str:
        return instance._db_url_base

    def url_for_environment(self, environment_name: str, *, instance: EnvStateInstance) -> str:
        base = self.base_url(instance)
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}options=-c%20search_path%3D%22{environment_name}%22,public"

    def env_state_docker_service_name(self) -> str:
        return DATABASE_SERVICE_NAME

    def render_healthcheck_service(
        self,
        environment_names: list[str],
        *,
        instance: EnvStateInstance | None = None,
    ) -> list[str]:
        """The readiness service for local Postgres IS the servicedb container itself (its own
        ``pg_isready`` healthcheck). ``environment_names`` are the MCP services hosted here (one schema
        each); ``instance`` is unused (the container is described statically)."""
        cfg = self.service_db_config
        spec = self.store_spec(environment_names)
        hc = spec.healthcheck
        return [
            f"  {self.env_state_docker_service_name()}:",
            f"    image: {cfg.db_image}",
            "    environment:",
            *[f"      - {key}={value}" for key, value in spec.env.items()],
            "    healthcheck:",
            f"      test: {hc['test']}",
            f"      interval: {hc['interval']}",
            f"      timeout: {hc['timeout']}",
            f"      retries: {hc['retries']}",
            "    volumes:",
            f"      - ./init-schemas.sql:{INIT_SCRIPT_MOUNT_PATH}",
            "    networks:",
            "      - env-network",
            "",
        ]

    def render_sidecar_containers(
        self,
        environment_names: list[str],
        *,
        host_port: Optional[Callable[[int], int]] = None,
        host_ips: tuple[str, ...] = (),
    ) -> list[str]:
        """The pgweb + db-mcp direct-SQL browse UIs (VM/compose path), each gated on the
        servicedb's health. Beyond the mandatory readiness service."""
        # Opened from the host only, so kept off the bridge address containers reach the host through.
        host_ips = host_ips[:1]
        publish = host_port or (lambda port: port)
        cfg = self.service_db_config
        lines: list[str] = []
        if cfg.db_web_image:
            lines.extend([
                f"  {PGWEB_SERVICE_NAME}:",
                f"    image: {cfg.db_web_image}",
                "    restart: unless-stopped",
                "    depends_on:",
                f"      {DATABASE_SERVICE_NAME}:",
                "        condition: service_healthy",
                "    environment:",
                f"      - PGWEB_DATABASE_URL={self._pg_url(DATABASE_SERVICE_NAME, scheme='postgres', sslmode='disable')}",
                # Lock to the single connection + hide Connect/Disconnect (see
                # sidecar_specs for why: Disconnect tore down pgweb's shared
                # session and 500'd the changelog export on Save & Next).
                "      - PGWEB_LOCK_SESSION=true",
                "    ports:",
                *(f'      - "{spec}"' for spec in port_bindings(host_ips, publish(DB_WEB_PORT), DB_WEB_PORT)),
                "    networks:",
                "      - env-network",
                "",
            ])
        if cfg.db_mcp_image:
            lines.extend([
                f"  {DB_MCP_SERVICE_NAME}:",
                f"    image: {cfg.db_mcp_image}",
                "    restart: unless-stopped",
                "    depends_on:",
                f"      {DATABASE_SERVICE_NAME}:",
                "        condition: service_healthy",
                "    environment:",
                f"      - DATABASE_URI={self._pg_url(DATABASE_SERVICE_NAME, scheme='postgresql')}",
                "    command: --transport=streamable-http --streamable-http-host=0.0.0.0 --access-mode=unrestricted",
                "    ports:",
                *(f'      - "{spec}"' for spec in port_bindings(host_ips, publish(DB_MCP_PORT), DB_MCP_CONTAINER_PORT)),
                "    networks:",
                "      - env-network",
                "",
            ])
        return lines

    def rendered_sidecar_service_names(self) -> list[str]:
        """The sidecar services ``render_sidecar_containers`` actually emits (same per-image gating),
        so the gateway can check/wire what was rendered without reading ``ServiceDBConfig`` itself."""
        cfg = self.service_db_config
        names: list[str] = []
        if cfg.db_web_image:
            names.append(PGWEB_SERVICE_NAME)
        if cfg.db_mcp_image:
            names.append(DB_MCP_SERVICE_NAME)
        return names

    @staticmethod
    def _in_image_store(image: str | None) -> bool:
        """Whether the image store serving ``image`` owns it: the container path pulls with that
        store's credentials, and has none for any other image."""
        return bool(image) and get_config().get_image_store_at(image).owns(image)

    def container_store_image(self) -> str:
        """servicedb image for the Modal container path: the configured image when the image store
        owns it, else stock postgres (postgres needs no customization — schema init runs via
        ``psql -f``)."""
        db_image = self.service_db_config.db_image
        if self._in_image_store(db_image):
            return db_image
        if db_image != _STOCK_POSTGRES_IMAGE:
            logger.warning(f"servicedb image {db_image} is not in the image store; running {_STOCK_POSTGRES_IMAGE}")
        return _STOCK_POSTGRES_IMAGE

    def sidecar_specs(self, *, instance: EnvStateInstance | None = None) -> list[ContainerSpec]:
        """pgweb + db-mcp container specs (Modal/container path), pointed at the stood-up
        servicedb. Only sidecar images the image store owns are provisionable in container mode;
        any other (or unset) pgweb/db-mcp is skipped + logged."""
        if instance is None:
            raise ValueError(
                "LocalPostgresStateProvider.sidecar_specs requires an instance — the acquired "
                "store whose host the sidecars connect to."
            )
        cfg = self.service_db_config
        db_host = instance.metadata["host"]
        specs: list[ContainerSpec] = []
        if self._in_image_store(cfg.db_web_image):
            pgweb_url = self._pg_url(db_host, scheme="postgres", sslmode="disable")
            specs.append(ContainerSpec(
                name=PGWEB_SERVICE_NAME, image=cfg.db_web_image, port=DB_WEB_PORT,
                # PGWEB_LOCK_SESSION locks pgweb to the single PGWEB_DATABASE_URL
                # connection and hides its Connect/Disconnect UI. Contributors were
                # able to click Disconnect, which tore down pgweb's single shared
                # session — breaking both the Raw DB view AND the hub's changelog
                # export (it queries the same pgweb `/api/query`), so Save & Next
                # 500'd. Locking the session removes the button and the failure.
                env={"PGWEB_DATABASE_URL": pgweb_url, "PGWEB_LOCK_SESSION": "true"},
            ))
        elif cfg.db_web_image:
            logger.warning(f"pgweb image {cfg.db_web_image} is not in the image store; skipping pgweb provisioning")
        if self._in_image_store(cfg.db_mcp_image):
            db_mcp_uri = self._pg_url(db_host, scheme="postgresql")
            specs.append(ContainerSpec(
                name=DB_MCP_SERVICE_NAME, image=cfg.db_mcp_image, port=DB_MCP_CONTAINER_PORT,
                env={"DATABASE_URI": db_mcp_uri},
                command=["--transport=streamable-http", "--streamable-http-host=0.0.0.0", "--access-mode=unrestricted"],
            ))
        elif cfg.db_mcp_image:
            logger.warning(f"db-mcp image {cfg.db_mcp_image} is not in the image store; skipping db-mcp provisioning")
        return specs

    async def install_changelog_triggers(
        self,
        environment_name: str,
        *,
        instance: EnvStateInstance | None = None,
        store_exec: "Callable[[list[str]], Awaitable[tuple[int, str, str]]] | None" = None,
    ) -> None:
        """Run ``SELECT _install_changelog_triggers('<schema>')`` via ``psql`` inside the store
        sandbox (the local store is unreachable from the deploy host, so the gateway injects
        ``store_exec`` — a 'run this argv against the store' capability)
        """
        if store_exec is None:
            raise ValueError("LocalPostgresStateProvider.install_changelog_triggers requires store_exec")
        argv = [
            "psql", "-U", DB_USER, "-d", DB_NAME,
            "-c", f"SELECT _install_changelog_triggers('{environment_name}')",
        ]
        exit_code, _out, err = await store_exec(argv)
        if exit_code != 0:
            raise RuntimeError(f"Changelog trigger install failed for '{environment_name}': {err[-500:]}")
