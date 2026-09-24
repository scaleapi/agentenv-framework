from __future__ import annotations

import logging
import posixpath
import shlex
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from agent_env.utils.paths import validate_relative_filename

from .gateway import GatewayMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .store import EnvQuery


@dataclass
class DeployedEnv:
    """Result of deploying an environment."""

    env_id: str
    env_version: int
    gateway_url: Optional[str]
    mcp_url: str
    db_web_url: Optional[str]
    sandbox_id: str
    db_mcp_url: Optional[str] = None
    environment_card_url: Optional[str] = None
    website_frontend_urls: Optional[dict[str, str]] = None
    sandbox_type: Optional[str] = None
    vnc_url: Optional[str] = None
    metadata: Optional[dict] = None
    instance_id: Optional[str] = None
    created_at_utc: Optional[str] = None
    expires_at_utc: Optional[str] = None
    gateway_mode: str = GatewayMode.PERFORMANCE.value
    sandbox_ids: dict[str, str | dict[str, str]] = field(default_factory=dict)
    # Ids of the EnvStateInstance record(s) this deploy's state was provisioned into
    # (one today — one backend per deploy). The forward pointer deploy -> state store.
    env_state_instance_ids: list[str] = field(default_factory=list)
    mcp_server_name: Optional[str] = None
    # The env card as last read, and when (ISO 8601, UTC); written at deploy from the readiness probe.
    environment_card: Optional[dict] = None
    environment_card_read_at_utc: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> DeployedEnv:
        return cls(
            env_id=data["env_id"],
            env_version=data["env_version"],
            gateway_url=data.get("gateway_url"),
            mcp_url=data["mcp_url"],
            db_web_url=data.get("db_web_url"),
            sandbox_id=data["sandbox_id"],
            sandbox_type=data.get("sandbox_type"),
            db_mcp_url=data.get("db_mcp_url"),
            environment_card_url=data.get("environment_card_url"),
            website_frontend_urls=data.get("website_frontend_urls"),
            vnc_url=data.get("vnc_url"),
            metadata=data.get("metadata"),
            instance_id=data.get("instance_id"),
            created_at_utc=data.get("created_at_utc"),
            expires_at_utc=data.get("expires_at_utc"),
            gateway_mode=data.get("gateway_mode", GatewayMode.PERFORMANCE.value),
            sandbox_ids=data.get("sandbox_ids") or {},
            env_state_instance_ids=data.get("env_state_instance_ids") or [],
            mcp_server_name=data.get("mcp_server_name"),
            environment_card=data.get("environment_card"),
            environment_card_read_at_utc=data.get("environment_card_read_at_utc"),
        )


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
    def put(cls, **kwargs) -> "Env":
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

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "version": self.version, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> "Env":
        raise NotImplementedError(f"{cls.__name__} must implement from_dict")

    async def deploy(self, **kwargs) -> DeployedEnv:
        raise NotImplementedError(f"{type(self).__name__} must implement deploy()")

    async def reset(self, deployed: "DeployedEnv") -> None:
        """Return an already-deployed env to a clean between-tasks state.

        Base raises so the ``reset_env`` step treats it as unsupported; envs that
        can reset in place (e.g. iOS CUA → home screen) override this.
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
        as needed. Each file is fetched directly from S3 by the sandbox via
        a presigned URL, so we never round-trip the bytes through this
        process.

        Default implementation uses ``self._sandbox`` (set by ``deploy()`` /
        ``from_deployed_env()`` on most env subclasses). Subclasses that
        manage multiple sandboxes (e.g. a desktop-VM env) or none should override.
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
            await sandbox.load_s3_file(file_artifact.object_url, dest_path)

        logger.info(
            f"Loaded FileArtifactUniverse '{file_artifact_universe.id}' "
            f"v{file_artifact_universe.version}: {total} file(s) at {destination}"
        )
        return LoadFileArtifactUniverseResult(destination_path=destination, files=loaded)

