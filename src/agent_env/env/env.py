from __future__ import annotations

import logging
import posixpath
import shlex
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from agentenv_protocol import client as protocol_v1

from agent_env.entity_refs import EntityRef
from agent_env.utils.paths import validate_relative_filename

from .gateway import GatewayMode
from .gateway.constants import WELL_KNOWN_PATH
from .store import get_env_store

logger = logging.getLogger(__name__)

# Default timeout for one invoke() call; above the class because a default argument needs it.
_EXTENSION_CALL_TIMEOUT_S = 30

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext
    from .store import EnvQuery


class EnvCapabilityUnsupported(RuntimeError):
    """A deployed env's card does not offer a capability method its caller needs."""

    def __init__(self, capability: str, method: str, env_id: str, instance_id: Optional[str] = None,
                 environment_name: Optional[str] = None) -> None:
        self.capability = capability
        self.method = method
        self.env_id = env_id
        self.instance_id = instance_id
        self.environment_name = environment_name
        child = f" child env '{environment_name}'" if environment_name else ""
        super().__init__(f"env '{env_id}'{child} does not offer '{method}' on {capability}.")


class EnvNeedsGateway(RuntimeError):
    """A caller reaches an env through its gateway, and the env was deployed without one."""

    def __init__(self, what: str, env_id: Optional[str] = None) -> None:
        self.what = what
        self.env_id = env_id
        env = f"env '{env_id}'" if env_id else "this env"
        super().__init__(f"{what} needs a gateway; {env} was deployed without one")


class EnvNeedsSandbox(RuntimeError):
    """A caller acts on an env's sandbox, and the env runs outside agent-env's sandboxes."""

    def __init__(self, what: str, env_id: str) -> None:
        self.what = what
        self.env_id = env_id
        super().__init__(f"{what} needs a sandbox; env '{env_id}' runs outside agent-env's sandboxes")


@dataclass(kw_only=True)
class DeployedEnv:
    """A deployed env: the env, its card, and the registry's bookkeeping. Each topology's subclass adds what it created."""

    env_id: str
    env_version: int
    # The provider that made the record, so the class it loads as (see from_dict); capabilities still come from the card.
    env_provider_type: Optional[str] = None
    environment_card_url: Optional[str] = None
    # The env card as last read, and when (ISO 8601, UTC); written at deploy from the readiness probe.
    environment_card: Optional[dict] = None
    environment_card_read_at_utc: Optional[str] = None
    # From the card whenever the record carries it (__post_init__); stored because readers of the JSON use them.
    mcp_url: Optional[str] = None
    mcp_server_name: Optional[str] = None
    # Set by the registry and the deploy step, never by a provider.
    metadata: Optional[dict] = None
    instance_id: Optional[str] = None
    created_at_utc: Optional[str] = None
    expires_at_utc: Optional[str] = None

    def __post_init__(self) -> None:
        # Only older records lack their card: they keep the stored values.
        if self.environment_card and self.environment_card_url:
            self.mcp_url = _mcp_url(self.environment_url, self.environment_card)
            self.mcp_server_name = self.environment_card.get("name")

    @classmethod
    def from_dict(cls, data: dict) -> DeployedEnv:
        record_class = _record_class(data)
        return record_class(**record_class._fields_from(data))

    @property
    def environment_url(self) -> Optional[str]:
        """The env's address: its card URL without the well-known path; card paths join onto it."""
        return _environment_url(self.environment_card_url) if self.environment_card_url else None

    def supports(self, uri: str, method: str) -> bool:
        """Whether the stored env card offers `method` on the extension at `uri`; no network."""
        return protocol_v1.find_extension_method(self.environment_card or {}, uri, method) is not None

    def require(self, uri: str, method: str) -> None:
        if not self.supports(uri, method):
            raise EnvCapabilityUnsupported(uri, method, self.env_id, self.instance_id)

    def get_child_env_card(self, environment_name: str) -> Optional[dict]:
        """The stored card's child env named `environment_name`; a leaf card is its own only child env."""
        card = self.environment_card or {}
        if not isinstance(card.get("children_environments"), list):
            return card if card.get("name") == environment_name else None
        return protocol_v1.find_child(card, environment_name)

    async def invoke(self, uri: str, method: str, params: Optional[dict] = None, *,
                     environment_name: Optional[str] = None, timeout: int = _EXTENSION_CALL_TIMEOUT_S) -> Any:
        """Call `method` on the extension at `uri` as the env card, or a child env's card, advertises it."""
        card = self.environment_card if environment_name is None else self.get_child_env_card(environment_name)
        if card is None or not self.environment_url or protocol_v1.find_extension_method(card, uri, method) is None:
            raise EnvCapabilityUnsupported(uri, method, self.env_id, self.instance_id, environment_name)
        return await protocol_v1.invoke_extension(self.environment_url, card, uri, params, timeout, method=method)

    @classmethod
    def _fields_from(cls, data: dict) -> dict:
        return dict(
            env_id=data["env_id"],
            env_version=data["env_version"],
            env_provider_type=data.get("env_provider_type"),
            environment_card_url=data.get("environment_card_url"),
            environment_card=data.get("environment_card"),
            environment_card_read_at_utc=data.get("environment_card_read_at_utc"),
            mcp_url=data.get("mcp_url"),
            mcp_server_name=data.get("mcp_server_name"),
            metadata=data.get("metadata"),
            instance_id=data.get("instance_id"),
            created_at_utc=data.get("created_at_utc"),
            expires_at_utc=data.get("expires_at_utc"),
        )


@dataclass(kw_only=True)
class DeployedSandboxEnv(DeployedEnv):
    """A deployed env running in our sandboxes: what restore, teardown and the reapers act on."""

    sandbox_id: str
    sandbox_type: Optional[str] = None
    sandbox_ids: dict[str, str | dict[str, str]] = field(default_factory=dict)

    @classmethod
    def _fields_from(cls, data: dict) -> dict:
        return {
            **super()._fields_from(data),
            "sandbox_id": data.get("sandbox_id"),
            "sandbox_type": data.get("sandbox_type"),
            "sandbox_ids": data.get("sandbox_ids") or {},
        }


@dataclass(kw_only=True)
class DeployedGatewayEnv(DeployedSandboxEnv):
    """A deployed env fronted by a gateway: the gateway's URLs and mode, and the state store its deploy holds."""

    env_provider_type: Optional[str] = "gateway"
    gateway_url: str
    gateway_mode: str = GatewayMode.PERFORMANCE.value
    db_web_url: Optional[str] = None
    db_mcp_url: Optional[str] = None
    website_frontend_urls: Optional[dict[str, str]] = None
    vnc_url: Optional[str] = None
    # Ids of the EnvStateInstance record(s) this deploy's state was provisioned into
    # (one today — one backend per deploy). The forward pointer deploy -> state store.
    env_state_instance_ids: list[str] = field(default_factory=list)

    @classmethod
    def _fields_from(cls, data: dict) -> dict:
        return {
            **super()._fields_from(data),
            "env_provider_type": data.get("env_provider_type") or "gateway",
            "gateway_url": data.get("gateway_url"),
            "gateway_mode": data.get("gateway_mode", GatewayMode.PERFORMANCE.value),
            "db_web_url": data.get("db_web_url"),
            "db_mcp_url": data.get("db_mcp_url"),
            "website_frontend_urls": data.get("website_frontend_urls"),
            "vnc_url": data.get("vnc_url"),
            "env_state_instance_ids": data.get("env_state_instance_ids") or [],
        }


@dataclass
class LoadEnvironmentUniverseArtifactResult:
    metadata_filepaths: dict[str, str] = field(default_factory=dict)
    # Which path the load actually took. A snapshot restore swaps a pre-ingested servicedb
    # image; a miss re-ingests every service over HTTP under a 600s per-service cap. The
    # two are comparable in wall-clock on a well-sized VM (measured ~4min vs ~3.5min on a
    # 13-service / ~8.7GB universe) -- what differs is the risk: only the re-ingest can
    # blow a timeout, and only it needs the CPU. Reported so callers can tell the two
    # apart, since otherwise an identical step silently costs minutes or fails.
    restored_from_snapshot: bool = False
    snapshot_db_image_artifact_id: str | None = None
    # Set only on the re-ingest path: whether a reusable snapshot was baked afterwards, so
    # the *next* load can restore instead. None means no bake was attempted.
    snapshot_baked: bool | None = None
    snapshot_bake_error: str | None = None


@dataclass
class LoadFileArtifactUniverseResult:
    """Records what was staged onto the env sandbox by load_file_artifact_universe."""

    destination_path: str
    files: dict[str, str] = field(default_factory=dict)  # filename -> absolute dest path


class Env(ABC):
    """Base class for all envs."""

    description: ClassVar[str]
    type: ClassVar[str] = "env"
    toml_refs: ClassVar[tuple[EntityRef, ...]] = ()
    env_provider_types: ClassVar[tuple[str, ...]] = ()

    def __init__(self, id: str, version: Optional[int], metadata: Optional[dict[str, Any]] = None):
        self.id: str = id
        self.version: Optional[int] = version
        self.metadata: dict[str, Any] = metadata or {}
        self._instance_id: Optional[str] = None

    @classmethod
    def get(cls, id: str, version: Optional[int] = None) -> "Env":
        from .store import get_env_store
        return get_env_store().get(id, version)

    @classmethod
    async def from_instance_id(cls, instance_id: str) -> "Env":
        from .store import get_env_instance_store
        deployed = get_env_instance_store().get(instance_id)
        env = cls.get(deployed.env_id, deployed.env_version)
        return await type(env).from_deployed_env(deployed)

    @classmethod
    def put(cls, **kwargs: Any) -> Self:
        from .store import get_env_store
        kwargs.setdefault("version", None)
        instance = cls(**kwargs)
        return get_env_store().put_document(instance)

    @classmethod
    def query(cls) -> "EnvQuery":
        from .store import EnvQuery, get_env_store
        return EnvQuery(get_env_store())

    def update_metadata(self, new_metadata: dict[str, Any]) -> None:
        """Replace this env's metadata in the database using compare-and-swap.

        The current in-memory metadata is used as the expected old value.
        On success, self.metadata is updated to new_metadata.

        Raises:
            ValueError: If the env has not been saved (version is None).
            ConcurrentModificationError: If metadata was modified since this env was loaded.
        """
        from .store import get_env_store
        if self.version is None:
            raise ValueError("Cannot update metadata on an unsaved env (version is None)")
        self.metadata = get_env_store().update_metadata(self.id, self.version, self.metadata, new_metadata)

    def merge_metadata(self, updates: dict[str, Any], retries: int = 3) -> None:
        """Merge ``updates`` into this env's metadata, re-reading and retrying on a
        lost CAS. Use instead of ``update_metadata`` when writing one key of a doc
        another writer may be updating concurrently.
        """
        from agent_env.store.base import ConcurrentModificationError
        from .store import get_env_store
        if self.version is None:
            raise ValueError("Cannot update metadata on an unsaved env (version is None)")
        metadata = self.metadata
        for attempt in range(retries + 1):
            try:
                self.metadata = get_env_store().update_metadata(
                    self.id, self.version, metadata, {**metadata, **updates}
                )
                return
            except ConcurrentModificationError:
                if attempt == retries:
                    raise
                metadata = Env.get(self.id, self.version).metadata

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "version": self.version, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Env":
        raise NotImplementedError(f"{cls.__name__} must implement from_dict")

    @classmethod
    def from_toml(cls, data: dict[str, Any], ctx: AuthoringContext) -> "Env":
        """Write the env authored as ``data`` (its toml, with the keys ``toml_refs`` declares resolved
        to ids) under ``ctx.id`` and return it. The default reads ``data`` as ``from_dict`` reads a
        stored document; a type whose toml differs overrides this."""
        return get_env_store().put_document(cls.from_dict({**data, "id": ctx.id, "version": None}))

    async def deploy(self, **kwargs) -> DeployedEnv:
        raise NotImplementedError(f"{type(self).__name__} must implement deploy()")

    @classmethod
    async def from_deployed_env(cls, deployed: DeployedEnv) -> Self:
        """The env a running deployment serves, for its caller to drive. A type that can be
        deployed on its own overrides this; one deployed only inside another, or never, need not.
        An override takes ``deployed: DeployedEnv`` and narrows it with ``isinstance``.
        """
        raise NotImplementedError(f"{cls.__name__} cannot be rebuilt from a deployment")

    async def reset(self, deployed: "DeployedEnv") -> None:
        """Return an already-deployed env to a clean between-tasks state.

        Base raises so the ``reset_env`` step treats it as unsupported; envs that
        can reset in place override this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support reset()")

    async def load_file_artifact_universe(
        self,
        file_artifact_universe: "Any",
        destination_path: Optional[str] = None,
    ) -> "LoadFileArtifactUniverseResult":
        """Stage every FileArtifact in `file_artifact_universe` onto this env's sandbox VM.

        Files land at ``<destination>/<filename>``. Nested relative paths
        (e.g. ``subdir/file.txt``) are supported — parent dirs are created
        as needed. The sandbox fetches each file directly from the object
        store through a signed URL where the store can sign one, so the bytes
        don't round-trip through this process.

        Default implementation uses ``self._sandbox`` (set by ``deploy()`` /
        ``from_deployed_env()`` on most env subclasses). Subclasses that
        manage several sandboxes, or none, should override.
        """
        sandbox = getattr(self, "_sandbox", None)
        if sandbox is None:
            raise RuntimeError(
                f"{type(self).__name__} has no _sandbox attribute; either call "
                "deploy() / from_deployed_env() first or override "
                "load_file_artifact_universe in this env class"
            )

        destination = (destination_path or "/tmp/file_artifacts").rstrip("/") or "/"

        file_artifacts = file_artifact_universe.get_file_artifacts()
        if not file_artifacts:
            logger.warning(
                f"FileArtifactUniverse '{file_artifact_universe.id}' "
                f"v{file_artifact_universe.version} has no files; nothing to load"
            )
            return LoadFileArtifactUniverseResult(destination_path=destination, files={})

        dirs_to_make = {destination}
        loaded: dict[str, str] = {}
        for filename in file_artifacts:
            validate_relative_filename(filename)
            dest_path = posixpath.join(destination, filename)
            parent = posixpath.dirname(dest_path)
            if parent:
                dirs_to_make.add(parent)
            loaded[filename] = dest_path

        mkdir_cmd = " && ".join(f"mkdir -p {shlex.quote(d)}" for d in sorted(dirs_to_make))
        await sandbox.exec_script(mkdir_cmd)

        total = len(file_artifacts)
        logger.info(
            f"Loading FileArtifactUniverse '{file_artifact_universe.id}' "
            f"v{file_artifact_universe.version} ({total} file(s)) at {destination}"
        )
        for idx, (filename, file_artifact) in enumerate(file_artifacts.items(), 1):
            dest_path = loaded[filename]
            logger.info(f"  [{idx}/{total}] {file_artifact.object_url} -> {dest_path}")
            await sandbox.load_object_file(file_artifact.object_url, dest_path)

        logger.info(
            f"Loaded FileArtifactUniverse '{file_artifact_universe.id}' "
            f"v{file_artifact_universe.version}: {total} file(s) at {destination}"
        )
        return LoadFileArtifactUniverseResult(destination_path=destination, files=loaded)


def gateway_url_of(deployed: Optional[DeployedEnv]) -> Optional[str]:
    """The gateway's URL, or None for a record without a gateway."""
    return deployed.gateway_url if isinstance(deployed, DeployedGatewayEnv) else None


def require_gateway_url(deployed: DeployedEnv, what: str) -> str:
    """The gateway's URL, or EnvNeedsGateway naming `what` for a record without a gateway."""
    if url := gateway_url_of(deployed):
        return url
    raise EnvNeedsGateway(what, deployed.env_id)


def require_sandbox(deployed: DeployedEnv, what: str) -> DeployedSandboxEnv:
    """The record of an env in one of agent-env's sandboxes, or EnvNeedsSandbox naming `what` for one outside them."""
    if isinstance(deployed, DeployedSandboxEnv):
        return deployed
    raise EnvNeedsSandbox(what, deployed.env_id)


def _environment_url(card_url: str) -> str:
    """The env's address: its card URL without the well-known path, where the protocol puts the card."""
    return card_url.removesuffix(WELL_KNOWN_PATH).rstrip("/")


def _mcp_url(url: str, card: dict) -> str:
    """The MCP endpoint the card declares, joined onto the env's URL with exactly one slash.

    Plain joining, not URL resolution: resolving "/mcp" would drop a path prefix such as a sandbox
    proxy's /sandbox/<id>, and resolving a relative path would replace the URL's last segment.
    """
    return f"{url.rstrip('/')}/{protocol_v1.mcp_path(card).lstrip('/')}"


def _record_class(data: dict) -> type[DeployedEnv]:
    """A stored record's class: its provider type's, else its shape's, so a record from before the type (or of one unknown here) keeps every field."""
    from agent_env.providers.env_providers.env_provider import record_class_for  # providers import this module

    provider_type = data.get("env_provider_type")
    record_class = record_class_for(provider_type)
    if record_class is not None:
        return record_class
    if any(data.get(name) for name in _GATEWAY_ONLY_FIELDS):
        return DeployedGatewayEnv
    # Every record from before the type ran in a sandbox; only a newer kernel's type can mean none.
    return DeployedSandboxEnv if data.get("sandbox_id") or not provider_type else DeployedEnv


# A record carrying any of these is a gateway's. Not gateway_mode: an older kernel writes its default into any record it round-trips.
_GATEWAY_ONLY_FIELDS = ("gateway_url", "db_web_url", "db_mcp_url", "website_frontend_urls", "vnc_url", "env_state_instance_ids")
