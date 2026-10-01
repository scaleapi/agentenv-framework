"""Env snapshot store for caching pre-loaded servicedb images."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER
from agent_env.config import get_config
from agent_env.store.document_store import AbsentOrNull, Filter, Sort
from agent_env.store.ids import derive_id, fs_safe, is_local_id, key_segment
from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

ENV_SNAPSHOTS_COLLECTION = "env_snapshots"

ProgressCallback = Callable[[str, str, int], None]


def compute_env_fingerprint(env) -> str:
    """A stable digest of the service images a MultiEnv is built from.

    Snapshots carry pre-ingested PGDATA, and the schema in that PGDATA is produced by
    the *service images* — not by the env id. So the env id is the wrong cache key in
    both directions: re-registering the same images under a new id needlessly orphans a
    valid snapshot, while rebuilding a service under the same id would silently serve a
    snapshot whose schema no longer matches.

    Keying on the images fixes both. Sorted by environment name so dict/list ordering
    can't change the digest, and truncated because this is a cache key, not a security
    claim.
    """
    parts: list[str] = []
    for mcp_env in getattr(env, "mcp_server_envs", None) or []:
        image = getattr(getattr(mcp_env, "docker_image_artifact", None), "image_name", "")
        parts.append(f"mcp:{mcp_env.environment_name}={image}")
    for site_env in getattr(env, "website_envs", None) or []:
        backend = getattr(getattr(site_env, "backend_docker_image_artifact", None), "image_name", "")
        frontend = getattr(getattr(site_env, "frontend_docker_image_artifact", None), "image_name", "")
        parts.append(f"web:{site_env.environment_name}={backend}+{frontend}")
    digest = hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()
    return digest[:32]


@dataclass
class EnvSnapshot:
    env_id: str
    environment_universe_id: str
    environment_universe_version: int
    instance_id: str
    is_clean: bool
    db_image_artifact_id: str
    db_image_artifact_version: int
    created_at_utc: datetime | None = None
    # Absent on rows written before image-fingerprint keying; those match on env_id only.
    env_fingerprint: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> EnvSnapshot:
        return cls(
            env_id=data["env_id"],
            environment_universe_id=data["service_universe_id"],
            environment_universe_version=data["service_universe_version"],
            instance_id=data["instance_id"],
            is_clean=data["is_clean"],
            db_image_artifact_id=data["db_image_artifact_id"],
            db_image_artifact_version=data["db_image_artifact_version"],
            created_at_utc=data.get("created_at_utc"),
            env_fingerprint=data.get("env_fingerprint"),
        )

    @classmethod
    async def create(cls, instance_id: str, on_progress: ProgressCallback | None = None) -> EnvSnapshot:
        """Create a snapshot of a deployed MultiEnv's servicedb.

        Reconnects to the sandbox, checks changelog status, exports the
        PostgreSQL data directory, builds a Docker image with the data
        pre-loaded, stores it as a DockerImageArtifact, and persists an
        EnvSnapshot record.
        """
        from agent_env.artifact import DockerImageArtifact
        from agent_env.env.env import DeployedSandboxEnv, Env
        from agent_env.env.envs.multi_env import MultiEnv, _gateway_state_provider
        from agent_env.env.store import get_env_instance_store

        log = on_progress or (lambda step, msg, pct: None)

        log("validate", "Fetching env instance...", 5)
        deployed_env = get_env_instance_store().get(instance_id)
        env = Env.get(deployed_env.env_id, deployed_env.env_version)
        if not isinstance(env, MultiEnv):
            raise ValueError(
                f"Env '{deployed_env.env_id}' is type '{env.type}', "
                "only MultiEnv is supported"
            )

        env_id = deployed_env.env_id
        without_our_gateway = (f"Snapshot capture needs our gateway and its local Postgres store; env '{env_id}' was deployed by "
                               f"env_provider_type '{deployed_env.env_provider_type}'")
        if not isinstance(deployed_env, DeployedSandboxEnv):
            raise NotImplementedError(without_our_gateway)

        environment_universe = get_env_instance_store().get_environment_universe(instance_id)
        if environment_universe is None:
            raise ValueError(f"Instance '{instance_id}' has no service_universe loaded")
        universe_id = environment_universe["id"]
        universe_version = environment_universe["version"]
        if is_local_id(env_id) != is_local_id(universe_id):
            raise ValueError(
                f"env {env_id!r} and universe {universe_id!r} are in different namespaces; a snapshot is recorded "
                "under both, so both have to be @local ids or neither"
            )
        log("reconnect", f"Reconnecting to sandbox {deployed_env.sandbox_id}...", 10)
        multi_env = await MultiEnv.from_deployed_env(deployed_env)
        if multi_env._sandbox is None:
            raise NotImplementedError(without_our_gateway)
        if multi_env._sandbox.mode == SANDBOX_MODE_CONTAINER:
            raise NotImplementedError("Snapshot creation not supported on container mode")

        # Snapshot capture bakes the local servicedb container into an image (docker cp PGDATA →
        # image), so it's only meaningful for a backend that can also restore from that image
        state_provider = _gateway_state_provider(multi_env)
        if state_provider is None or not state_provider.supports_restore_from_snapshot:
            raise NotImplementedError(
                "Snapshot capture is only supported for the local Postgres backend "
                f"(this env's state backend does not support it: {type(state_provider).__name__ if state_provider else None}). "
                "Remote-backed deploys use the normal per-service load."
            )
        # Fail on an unsignable store before any command runs on the sandbox; the upload presigns a fresh url.
        await asyncio.to_thread(_presigned_put_url, *_tarball_destination(env_id, universe_id))

        log("check_changelog", "Checking changelog...", 20)
        is_clean = await _check_changelog_empty(multi_env._sandbox)
        logger.info(f"Changelog empty: {is_clean} ({'clean' if is_clean else 'dirty'} snapshot)")
        log("check_changelog", f"Changelog: {'clean' if is_clean else 'dirty'}", 30)

        log("snapshot_db", "Exporting servicedb data...", 40)
        tar_gz_s3_url = await _snapshot_servicedb(multi_env._sandbox, env_id, universe_id, log)

        log("store_artifact", "Storing Docker image artifact...", 70)
        logger.info("Storing snapshot as DockerImageArtifact...")
        image_tag = _image_tag(env_id, universe_id)
        artifact = DockerImageArtifact.put_tar(
            id=derive_id(env_id, "env-snapshot"),
            description=(
                f"Snapshot of servicedb for env={env_id} "
                f"universe={universe_id} v{universe_version}"
            ),
            image_name=image_tag,
            tar_gz_s3_url=tar_gz_s3_url,
        )
        logger.info(f"Artifact: id={artifact.id} version={artifact.version}")
        log("store_artifact", f"Artifact: id={artifact.id} version={artifact.version}", 80)

        log("store_snapshot", "Persisting snapshot record...", 90)
        snapshot = get_env_snapshot_store().put(
            env_id=env_id,
            environment_universe_id=universe_id,
            environment_universe_version=universe_version,
            instance_id=instance_id,
            is_clean=is_clean,
            db_image_artifact_id=artifact.id,
            db_image_artifact_version=artifact.version,
            # Recorded from the env whose images produced this PGDATA, so a later
            # image-identical env (any id) can match it — and a rebuilt one cannot.
            env_fingerprint=compute_env_fingerprint(env),
        )
        log("done", "Snapshot complete", 100)
        return snapshot


async def _check_changelog_empty(sandbox) -> bool:
    """Return True if the _changelog table is empty (no modifications since data load)."""
    from agent_env.env.envs.service_db import DB_NAME, DB_USER
    from agent_env.env.envs.service_db import DATABASE_SERVICE_NAME
    from agent_env.providers.env_providers.constants import DOCKER_COMPOSE_PATH

    cmd = (
        f"docker compose -f {DOCKER_COMPOSE_PATH} exec -T {DATABASE_SERVICE_NAME} "
        f'psql -U {DB_USER} -d {DB_NAME} -tAc "SELECT count(*) FROM public._changelog"'
    )
    output = await sandbox.exec_script(cmd)
    count = int(output.strip())
    return count == 0


async def _snapshot_servicedb(sandbox, env_id: str, environment_universe_id: str, log: ProgressCallback) -> str:
    """Export pgdata from sandbox servicedb, build Docker image on sandbox, upload to S3.

    Returns the S3 URL of the uploaded tar.gz.
    """
    from agent_env.env.envs.service_db import DATABASE_SERVICE_NAME
    from agent_env.providers.env_providers.constants import DOCKER_COMPOSE_PATH

    container_id_output = await sandbox.exec_script(
        f"docker compose -f {DOCKER_COMPOSE_PATH} ps -a -q {DATABASE_SERVICE_NAME}"
    )
    container_id = container_id_output.strip()
    if not container_id:
        raise RuntimeError("servicedb container not found on sandbox")

    log("snapshot_db", "Copying pgdata from container...", 45)
    logger.info("Exporting pgdata from servicedb container...")
    await sandbox.exec_script(f"docker cp {container_id}:/var/lib/postgresql/data /tmp/pgdata")

    image_tag = _image_tag(env_id, environment_universe_id)
    log("snapshot_db", f"Building Docker image on sandbox: {image_tag}...", 50)
    logger.info(f"Building Docker image on sandbox: {image_tag}...")
    await sandbox.exec_script(
        "mkdir -p /tmp/snapshot-build && "
        "mv /tmp/pgdata /tmp/snapshot-build/pgdata && "
        "cat > /tmp/snapshot-build/Dockerfile << 'DEOF'\n"
        "FROM public.ecr.aws/docker/library/postgres:16-alpine\n"
        "COPY pgdata /var/lib/postgresql/data\n"
        "RUN chown -R postgres:postgres /var/lib/postgresql/data\n"
        "DEOF"
    )
    await sandbox.exec_script(
        f"docker build --platform linux/amd64 -t {image_tag} /tmp/snapshot-build"
    )

    log("snapshot_db", "Saving Docker image...", 55)
    logger.info("Saving Docker image to tar.gz on sandbox...")
    await sandbox.exec_script(f"docker save {image_tag} | gzip > /tmp/snapshot-image.tar.gz")

    object_store, object_url = _tarball_destination(env_id, environment_universe_id)
    put_url = await asyncio.to_thread(_presigned_put_url, object_store, object_url)
    log("snapshot_db", "Uploading Docker image...", 60)
    logger.info(f"Uploading Docker image to {object_url}...")
    await sandbox.exec_script(
        f'curl -fsSL -X PUT --upload-file /tmp/snapshot-image.tar.gz "{put_url}"'
    )

    await sandbox.exec_script(f"rm -rf /tmp/snapshot-build /tmp/snapshot-image.tar.gz && docker rmi {image_tag}")

    return object_url


def _image_tag(env_id: str, universe_id: str) -> str:
    """The snapshot image's tag, a docker reference the sandbox builds with."""
    return f"env-snapshot-{fs_safe(env_id)}-{fs_safe(universe_id)}"


def _tarball_destination(env_id: str, universe_id: str) -> tuple[ObjectStore, str]:
    """The object store the snapshot image's tarball is uploaded to, and its url there."""
    config = get_config()
    store = config.get_object_store_for(env_id)
    key = f"env-snapshots/{key_segment(env_id)}/{key_segment(universe_id)}/{_image_tag(env_id, universe_id)}.tar.gz"
    return store, store.object_url(f"{config.get_artifact_key_prefix()}{key}")


def _presigned_put_url(object_store: ObjectStore, object_url: str) -> str:
    put_url = object_store.signed_put_url(object_url)
    if put_url is None:
        raise RuntimeError(
            f"{type(object_store).__name__} can't presign uploads; env snapshots need a signable object store or local execution."
        )
    return put_url


class EnvSnapshotStore:
    def __init__(self) -> None:
        self._indexed = None

    @property
    def _doc_store(self):
        # Resolved per call: a cached store outlives reset_config(), so a process that
        # re-pointed would read the new config and write the old backend.
        store = get_config().get_document_store()
        if self._indexed is not store:
            store.ensure_index(
                ENV_SNAPSHOTS_COLLECTION,
                ["env_id", "service_universe_id", "instance_id"],
                unique=True,
            )
            # The get_clean lookup shape — hit on every universe load, so it should
            # not table-scan as snapshots accumulate.
            store.ensure_index(
                ENV_SNAPSHOTS_COLLECTION,
                ["env_fingerprint", "service_universe_id", "service_universe_version", "is_clean"],
            )
            self._indexed = store
        return store

    def put(
        self,
        env_id: str,
        environment_universe_id: str,
        environment_universe_version: int,
        instance_id: str,
        is_clean: bool,
        db_image_artifact_id: str,
        db_image_artifact_version: int,
        env_fingerprint: str | None = None,
    ) -> EnvSnapshot:
        doc = {
            "env_id": env_id,
            "service_universe_id": environment_universe_id,
            "service_universe_version": environment_universe_version,
            "instance_id": instance_id,
            "is_clean": is_clean,
            "db_image_artifact_id": db_image_artifact_id,
            "db_image_artifact_version": db_image_artifact_version,
            "created_at_utc": datetime.now(timezone.utc),
            "env_fingerprint": env_fingerprint,
        }
        self._doc_store.replace(
            ENV_SNAPSHOTS_COLLECTION,
            # `Filter.of` keywords are stored document keys, not Python names: this one keeps
            # the on-disk spelling while the parameter follows the rename.
            Filter.of(env_id=env_id, service_universe_id=environment_universe_id, instance_id=instance_id),
            doc,
            upsert=True,
        )
        return EnvSnapshot.from_dict(doc)

    def get(self, env_id: str, environment_universe_id: str, instance_id: str) -> EnvSnapshot | None:
        doc = self._doc_store.find_one(
            ENV_SNAPSHOTS_COLLECTION,
            Filter.of(env_id=env_id, service_universe_id=environment_universe_id, instance_id=instance_id),
        )
        if not doc:
            return None
        return EnvSnapshot.from_dict(doc)

    def get_clean(
        self,
        env_id: str,
        environment_universe_id: str,
        environment_universe_version: int | None = None,
        env_fingerprint: str | None = None,
    ) -> EnvSnapshot | None:
        """Newest clean snapshot that can be restored into this env, or None.

        With an ``env_fingerprint``, an image-identical env matches regardless of its
        id — so re-registering the same services under a new env id reuses the existing
        snapshot instead of paying for a fresh full load. The env_id lookup is kept as a
        fallback for rows written before fingerprinting, which carry no digest to match.

        Note the fingerprint match is deliberately NOT widened to a "close enough" env
        id comparison: an env whose service images differ has a different schema, and
        restoring a stale snapshot into it would corrupt the env rather than slow it down.
        """
        base = {"service_universe_id": environment_universe_id, "is_clean": True}
        if environment_universe_version is not None:
            base["service_universe_version"] = environment_universe_version

        candidates: list[Filter] = []
        # Fingerprint first: it is the correct key, and it lets a renamed env hit.
        if env_fingerprint:
            candidates.append(Filter.of(**base, env_fingerprint=env_fingerprint))
        # Legacy rows predate fingerprinting, so the exact env id is the only provenance
        # they carry. Restricted to rows with NO fingerprint on purpose: a row that has
        # one and did not match above was built from different service images, and
        # matching it on env_id alone would restore a stale schema into this env.
        candidates.append(
            Filter.of(**base, env_id=env_id).where("env_fingerprint", AbsentOrNull())
        )

        for query in candidates:
            doc = self._doc_store.find_one(
                ENV_SNAPSHOTS_COLLECTION,
                query,
                sort=Sort.by("created_at_utc", descending=True),
            )
            if doc:
                return EnvSnapshot.from_dict(doc)
        return None


_env_snapshot_store: Optional[EnvSnapshotStore] = None


def get_env_snapshot_store() -> EnvSnapshotStore:
    global _env_snapshot_store
    if _env_snapshot_store is None:
        _env_snapshot_store = EnvSnapshotStore()
    return _env_snapshot_store


def set_env_snapshot_store(store: EnvSnapshotStore) -> None:
    global _env_snapshot_store
    _env_snapshot_store = store


def reset_env_snapshot_store() -> None:
    global _env_snapshot_store
    _env_snapshot_store = None
