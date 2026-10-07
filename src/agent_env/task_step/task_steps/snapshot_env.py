"""Snapshot a deployed env's live service state into a new EnvironmentUniverseArtifact."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar, Optional
from urllib.parse import urlparse

import httpx

from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI, FilePart, uploaded_object_path
from agentenv_protocol.client import find_extension
from agentenv_protocol.transfers import WriteNamespaceGrant

from agent_env.a2a_agent.object_transfer import ObjectLimits, namespace_grant
from agent_env.env.gateway.constants import EXT_TRAJECTORY_URI
from agent_env.store import get_config
from agent_env.store.base import GrantUnavailableError
from agent_env.store.ids import derive_id, is_local_id, validate_local_id
from agent_env.store.object_store import MIN_GRANT_LIFETIME_SECONDS, ObjectStore, S3ObjectStore
from agent_env.store.object_store.object_store import issues_grants_to
from agent_env.store.routing import in_local_run
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef, RefRole
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_TIMEOUT_SECONDS = 600
DEFAULT_EXPORT_CONCURRENCY = 3
_CONNECT_TIMEOUT_SECONDS = 30

_TRAJECTORY_CAPTURE_BUDGET_SECONDS = 300

# Must match the S3 credentials extension URI the server-side credentials mixin advertises.
_S3_CREDENTIALS_EXTENSION_URI = "urn:agentenv:add-s3-credentials/v1"

# What a service may upload through its snapshot grant: one bundle, as large as one upload to the
# store can be up to this.
ENV_SNAPSHOT_LIMITS = ObjectLimits(max_objects=1, max_object_bytes=50 * 1024**3, max_total_bytes=50 * 1024**3)
_SNAPSHOT_KEY_PREFIX = "agentenv-snapshots"


@dataclass(frozen=True)
class _SnapshotUpload:
    """Where one service may upload its snapshot bundle, and the grant it uploads with."""

    store: ObjectStore
    grant: WriteNamespaceGrant

    def object_url(self, path: str) -> str:
        return self.store.object_url(f"{self.grant.root_path}/{path}")


def _snapshot_upload(card: dict, timeout_seconds: float, sandbox_type: Optional[str]) -> Optional[_SnapshotUpload]:
    """A grant for a service advertising the data-objects form to upload its bundle with, under a namespace
    of its own; None when it does not advertise the form, or the object store issues no grant that reaches
    the service's sandbox, of ``sandbox_type`` (None: unknown) and lasts the export."""
    if find_extension(card, DATA_OBJECTS_EXTENSION_URI) is None:
        return None
    config = get_config()
    store = config.get_object_store()
    if not issues_grants_to(store, sandbox_type):
        return None
    cap = store.max_single_upload_bytes or ENV_SNAPSHOT_LIMITS.max_object_bytes
    limits = ObjectLimits(
        max_objects=ENV_SNAPSHOT_LIMITS.max_objects,
        max_object_bytes=min(ENV_SNAPSHOT_LIMITS.max_object_bytes, cap),
        max_total_bytes=min(ENV_SNAPSHOT_LIMITS.max_total_bytes, cap),
    )
    namespace_url = store.object_url(f"{config.get_artifact_key_prefix()}{_SNAPSHOT_KEY_PREFIX}/{uuid.uuid4().hex}")
    try:
        grant = namespace_grant(
            store, namespace_url, limits=limits, expires_in=max(int(timeout_seconds), MIN_GRANT_LIFETIME_SECONDS)
        )
    except GrantUnavailableError as exc:
        logger.info("snapshot_env: the object store issues no snapshot upload grant (%s)", exc)
        return None
    return _SnapshotUpload(store, grant)


async def _push_s3_credentials(base_url: str, card: dict, timeout_seconds: float) -> None:
    """Best-effort: push the object store's shared AWS creds + bucket to a service
    advertising add_s3_credentials, if the store shares any. Never raises (get_data falls back).
    An @local run's snapshots land in the local stores, so it hands a service no credentials to
    upload one to S3 with; the service returns it instead."""
    from agentenv_protocol import client as protocol_v1

    if protocol_v1.find_extension(card, _S3_CREDENTIALS_EXTENSION_URI) is None or in_local_run():
        return
    try:
        store = get_config().get_object_store()
        if not isinstance(store, S3ObjectStore):
            return
        env = await asyncio.to_thread(store.shared_credentials_env)
        if "AWS_ACCESS_KEY_ID" not in env:
            return
        params = {
            "aws_access_key_id": env["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": env["AWS_SECRET_ACCESS_KEY"],
            "aws_session_token": env.get("AWS_SESSION_TOKEN"),
            "region_name": env.get("AWS_DEFAULT_REGION"),
            "bucket": store.bucket,
        }
        await protocol_v1.invoke_extension(
            base_url,
            card,
            _S3_CREDENTIALS_EXTENSION_URI,
            params,
            timeout=int(timeout_seconds),
        )
        logger.info("snapshot_env: pushed S3 creds to %s", base_url)
    except Exception:
        logger.warning(
            "snapshot_env: failed to push S3 creds to %s (get_data will fall back)",
            base_url,
            exc_info=True,
        )


@dataclass
class EnvSnapshotResult:
    environment_universe_artifact_id: str
    environment_universe_artifact_version: int
    environments_snapshotted: list[str]
    total: int


class SnapshotEnvTaskStep(TaskStep):
    """Snapshot a deployed env's current service state into a new
    ``EnvironmentUniverseArtifact``.

    Picks the env from ``context.deployed_envs`` by ``env_instance_id``, else
    ``env_id``, else the context's single env; ``env_step_id`` narrows that to
    one ``deploy_env`` step's deployment when an env is deployed more than once.
    ``gateway_url``, ``snapshot_id`` and ``original_universe_artifact_id`` are
    optional overrides; universe metadata carries over from that id, else the
    loaded universe. The generated snapshot id is stable across
    retries of a run, so retries bump versions instead of minting
    duplicate universes.

    Output: ``context.metadata["env_snapshotted_universes"][self.id]`` =
    ``{"id", "version"}``. Fails the step if any service fails to export.
    """

    type: ClassVar[str] = "snapshot_env"
    entity_refs = (
        EntityRef.env("env_id"),
        EntityRef.artifact("snapshot_id", role=RefRole.OUTPUT, artifact_type="environment_universe"),
        EntityRef.artifact("original_universe_artifact_id", artifact_type="environment_universe"),
    )
    DEFAULT_EXPORT_TIMEOUT_SECONDS: ClassVar[int] = DEFAULT_EXPORT_TIMEOUT_SECONDS

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: Optional[str] = None,
        env_instance_id: Optional[str] = None,
        env_step_id: Optional[str] = None,
        gateway_url: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        original_universe_artifact_id: Optional[str] = None,
        export_timeout_seconds: int = DEFAULT_EXPORT_TIMEOUT_SECONDS,
        include_env_trajectory: bool = False,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.env_instance_id = env_instance_id
        self.env_step_id = env_step_id
        self.gateway_url = gateway_url
        self.snapshot_id = snapshot_id
        self.original_universe_artifact_id = original_universe_artifact_id
        self.export_timeout_seconds = export_timeout_seconds
        self.include_env_trajectory = include_env_trajectory

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["env_instance_id"] = self.env_instance_id
        base["env_step_id"] = self.env_step_id
        base["gateway_url"] = self.gateway_url
        base["snapshot_id"] = self.snapshot_id
        base["original_universe_artifact_id"] = self.original_universe_artifact_id
        base["export_timeout_seconds"] = self.export_timeout_seconds
        base["include_env_trajectory"] = self.include_env_trajectory
        return base

    @classmethod
    def from_dict(cls, data: dict) -> SnapshotEnvTaskStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data.get("env_id"),
            env_instance_id=data.get("env_instance_id"),
            env_step_id=data.get("env_step_id"),
            gateway_url=data.get("gateway_url"),
            snapshot_id=data.get("snapshot_id"),
            original_universe_artifact_id=data.get("original_universe_artifact_id"),
            export_timeout_seconds=data.get(
                "export_timeout_seconds", cls.DEFAULT_EXPORT_TIMEOUT_SECONDS
            ),
            include_env_trajectory=data.get("include_env_trajectory", False),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env import Env, gateway_url_of

        snapshot_id = self._derive_snapshot_id(context)
        # Refused before any export: past here a write that fails only skips its service.
        if is_local_id(snapshot_id):
            validate_local_id(snapshot_id)
        deployed = self._resolve_deployed_env(context)
        env_id = self.env_id or (deployed.env_id if deployed else None)
        if not env_id:
            raise RuntimeError("snapshot_env could not resolve an env_id")
        gateway_url = self.gateway_url or gateway_url_of(deployed)
        if not gateway_url:
            raise RuntimeError(
                f"Deployed env '{env_id}' has no gateway_url and no override was provided"
            )

        if self.include_env_trajectory:
            entry: dict = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "capture_step_id": self.id}
            try:
                await asyncio.wait_for(self._capture_env_trajectory(entry, context, env_id, deployed), timeout=_TRAJECTORY_CAPTURE_BUDGET_SECONDS)
            except Exception as e:
                detail = f"{type(e).__name__}: {str(e)[:200]}"
                logger.warning(f"snapshot_env: env trajectory capture aborted (continuing): {detail}")
                entry["error"] = detail
            context.metadata.setdefault("env_trajectory", {})[env_id] = entry

        # Pin the env version the sandbox actually runs, so service enumeration
        # can't drift if the env was re-registered since this sandbox deployed.
        env_version = deployed.env_version if deployed else None
        env = await asyncio.to_thread(Env.get, env_id, env_version)

        result = await self.snapshot_env_state(
            env=env,
            gateway_url=gateway_url,
            snapshot_id=snapshot_id,
            deployed=deployed,
            original_universe_artifact_id=self.original_universe_artifact_id,
            export_timeout_seconds=self.export_timeout_seconds,
        )

        logger.info(
            f"snapshot_env: {env_id} -> {result.environment_universe_artifact_id} "
            f"v{result.environment_universe_artifact_version} "
            f"({len(result.environments_snapshotted)}/{result.total} services)"
        )
        # Identifiers only — the context is heartbeated into task_instances.
        context.metadata.setdefault("env_snapshotted_universes", {})[self.id] = {
            "id": result.environment_universe_artifact_id,
            "version": result.environment_universe_artifact_version,
        }
        return context

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _deploy_step_of(deployed_env) -> Optional[str]:
        return (deployed_env.metadata or {}).get("deploy_step_id")

    def _select_by_step_id(self, candidates: list) -> list:
        """The candidates ``env_step_id`` names; all of them when it isn't set."""
        if self.env_step_id is None:
            return candidates
        # A filter, not a hint: an unmatched id is an error, never a substitute.
        # Empty candidates are left to the callers' own not-deployed handling.
        named = [d for d in candidates if self._deploy_step_of(d) == self.env_step_id]
        if candidates and not named:
            raise RuntimeError(
                f"snapshot_env '{self.id}': env_step_id '{self.env_step_id}' names no deployment of "
                f"{self._env_label} (deployed by: {[self._deploy_step_of(d) for d in candidates]})"
            )
        return named

    @property
    def _env_label(self) -> str:
        return f"env '{self.env_id}'" if self.env_id else "any deployed env"

    def _resolve_deployed_env(self, context: TaskStepContext):
        """The DeployedEnv this step targets, or None in pure-override mode."""
        if self.env_instance_id:
            deployed = next(
                (d for d in context.deployed_envs if d.instance_id == self.env_instance_id),
                None,
            )
            if deployed is None:
                raise RuntimeError(
                    f"No deployed env with instance_id '{self.env_instance_id}' in context "
                    f"(have: {[d.instance_id for d in context.deployed_envs]})"
                )
            return deployed
        if self.env_id:
            matches = self._select_by_step_id(
                [d for d in context.deployed_envs if d.env_id == self.env_id]
            )
            if len(matches) > 1:
                raise RuntimeError(
                    f"snapshot_env '{self.id}': {len(matches)} deployments of env '{self.env_id}' "
                    f"(deployed by: {[self._deploy_step_of(d) for d in matches]}). Set env_step_id "
                    "to the deploy_env step this snapshot should target."
                )
            deployed = matches[0] if matches else None
            if deployed is None and self.gateway_url:
                return None  # pure-override mode: env_id + gateway_url suffice
            if deployed is None:
                raise RuntimeError(
                    f"No deployed env with env_id '{self.env_id}' in context "
                    f"(have: {[d.env_id for d in context.deployed_envs]}) and no "
                    "gateway_url override was provided"
                )
            return deployed
        # `env_step_id` alone also names one of k, so it answers this ambiguity.
        candidates = self._select_by_step_id(context.deployed_envs)
        if len(candidates) == 1:
            return candidates[0]
        raise RuntimeError(
            "snapshot_env needs env_id, env_instance_id or env_step_id when the context has "
            f"{len(context.deployed_envs)} deployed envs "
            f"(have: {[d.env_id for d in context.deployed_envs]})"
        )

    async def _capture_env_trajectory(self, entry: dict, context: TaskStepContext, env_id: str, deployed) -> None:
        """Stream the trajectory to a temp file, upload it verbatim (no envelope), and fill ``entry``."""
        method, url = _trajectory_route(deployed, self.gateway_url)
        store = get_config().get_object_store()
        event_count = 0
        timeout = httpx.Timeout(_TRAJECTORY_CAPTURE_BUDGET_SECONDS, connect=_CONNECT_TIMEOUT_SECONDS)
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(method, url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        tmp.write(chunk)
                        event_count += chunk.count(b"\n")
            tmp.close()
            instance_key = context.instance_id or f"adhoc-{uuid.uuid4().hex[:12]}"
            key = (
                f"{get_config().get_artifact_key_prefix()}env_trajectory/"
                f"instance_id={instance_key}/{env_id}-{uuid.uuid4().hex[:8]}.jsonl"
            )
            object_url = await asyncio.to_thread(store.put_file, key, tmp.name, "application/jsonl")
        finally:
            tmp.close()
            os.unlink(tmp.name)
        entry["object_url"] = object_url
        entry["event_count"] = event_count

    def _derive_snapshot_id(self, context: TaskStepContext) -> str:
        if self.snapshot_id:
            return self.snapshot_id
        instance_id = context.instance_id or context.metadata.get("instance_id")
        if instance_id:
            # Stable per run instance: activity retries re-put the same ids.
            return derive_id(instance_id, f"snapshot-{self.id}")
        generated = derive_id(f"adhoc-{uuid.uuid4().hex[:12]}", f"snapshot-{self.id}")
        logger.warning(
            f"snapshot_env: no instance_id in context; using random snapshot_id "
            f"{generated} (retries will not be idempotent)"
        )
        return generated

    @staticmethod
    def _resolve_loaded_universe_ref(deployed) -> Optional[tuple[str, Optional[int]]]:
        """(id, version) of the universe this instance recorded loading, or None (best-effort)."""
        instance_id = getattr(deployed, "instance_id", None) if deployed else None
        if not instance_id:
            return None
        try:
            from agent_env.env.store import get_env_instance_store

            su = get_env_instance_store().get_environment_universe(instance_id)
        except Exception:
            logger.warning("snapshot_env: could not read loaded universe for %s", instance_id, exc_info=True)
            return None
        return (su["id"], su.get("version")) if su and su.get("id") else None

    @staticmethod
    def _enumerate_environments(env) -> list[str]:
        """environment names for a multi env or a single MCP server env."""
        if getattr(env, "mcp_server_envs", None):
            return [e.environment_name for e in env.mcp_server_envs]
        if getattr(env, "environment_name", None):
            return [env.environment_name]
        raise RuntimeError(
            f"Env '{env.id}' v{env.version} has no MCP services to snapshot "
            "(expected mcp_server_envs or environment_name)"
        )

    @staticmethod
    async def _export_environment_to_file(
        gateway_url: str, environment_name: str, tmp_path: str, timeout_seconds: float, deployed=None
    ) -> str:
        """Export one service; return the suffix written (".json"/".zip") to
        ``tmp_path``, or the object url a FilePart names when the service uploaded the
        bundle itself (nothing written to ``tmp_path`` then — export_one registers it).

        v1 ``get_data``, handed a namespace grant when the service takes the data-objects
        form: an upload through it → its object url; ``DataPart`` → json; ``FilePart`` →
        object url (no download) / base64 ``bytes`` / streamed relative-or-http ``uri`` →
        ".zip"; else legacy ``GET /export-state`` streamed to disk with a first-byte JSON
        guard. Errors are caught by the caller (fails just this service)."""
        from agent_env.env import legacy_protocol
        from agentenv_protocol import client as protocol_v1

        base_url = await legacy_protocol.v1_base_url(deployed, gateway_url, environment_name, mcp=True)
        if base_url is not None:
            card_base, card = await legacy_protocol.child_env_card(deployed, gateway_url, environment_name)
            upload = await asyncio.to_thread(
                _snapshot_upload, card or {}, timeout_seconds, getattr(deployed, "sandbox_type", None)
            )
            # Push S3 creds (no-op unless the service advertises the extension), with a grant too: a
            # bundle the grant cannot hold can still be uploaded with them.
            await _push_s3_credentials(card_base, card or {}, timeout_seconds)
            grant = {} if upload is None else {"write_namespace": upload.grant}
            resp = await protocol_v1.get_data(base_url, timeout=int(timeout_seconds), **grant)
            part = resp.parts[0] if resp.parts else None
            if upload is not None and (path := uploaded_object_path(part)) is not None:
                return upload.object_url(path)
            if isinstance(part, FilePart):
                raw = getattr(part.file, "bytes", None)
                if raw is not None:
                    with open(tmp_path, "wb") as f:
                        f.write(base64.b64decode(raw))
                else:
                    uri = getattr(part.file, "uri", None)
                    if not uri:
                        # ValueError, not TypeError, so export_one catches it and
                        # fails only this service — not the whole snapshot.
                        raise ValueError(
                            f"FilePart for '{environment_name}' has neither bytes nor uri"
                        )
                    # Any url but http(s) is an object the service uploaded itself — hand
                    # it back as-is; export_one registers it directly (no download). Its
                    # last segment carries the shape (e.g. gdrive.zip).
                    scheme = urlparse(uri).scheme
                    if scheme and scheme not in ("http", "https"):
                        return uri
                    # A relative uri (e.g. "export-snapshot") means "stream it from
                    # my service endpoint" — resolve against the service base_url so
                    # multi-GB bundles never ride inline in the JSON-RPC response.
                    if not scheme:
                        uri = f"{base_url.rstrip('/')}/{uri.lstrip('/')}"
                    async with httpx.AsyncClient() as client:
                        async with client.stream(
                            "GET",
                            uri,
                            timeout=httpx.Timeout(
                                timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS
                            ),
                        ) as resp_file:
                            resp_file.raise_for_status()
                            with open(tmp_path, "wb") as f:
                                async for chunk in resp_file.aiter_bytes():
                                    f.write(chunk)
                return os.path.splitext(getattr(part.file, "name", "") or "")[1] or ".zip"
            state = part.data if part is not None else {}
            with open(tmp_path, "w") as f:
                json.dump(state, f, default=str)
            return ".json"

        legacy_url = f"{legacy_protocol.environment_base_url(gateway_url, environment_name, mcp=True)}/export-state"
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "GET",
                legacy_url,
                timeout=httpx.Timeout(timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS),
            ) as resp:
                resp.raise_for_status()
                seen_first_bytes = False
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        if not seen_first_bytes and chunk:
                            lead = chunk.lstrip()[:1]
                            if lead not in (b"{", b"["):
                                raise ValueError(
                                    f"export-state returned non-JSON body (starts with {lead!r})"
                                )
                            seen_first_bytes = True
                        f.write(chunk)
        return ".json"

    @classmethod
    async def snapshot_env_state(
        cls,
        *,
        env,
        gateway_url: str,
        snapshot_id: str,
        deployed=None,
        original_universe_artifact_id: Optional[str] = None,
        export_timeout_seconds: int = DEFAULT_EXPORT_TIMEOUT_SECONDS,
        concurrency: int = DEFAULT_EXPORT_CONCURRENCY,
    ) -> EnvSnapshotResult:
        """Export all services via the gateway into a new EnvironmentUniverseArtifact
        ``snapshot_id``. All-or-nothing; the raised error carries public
        per-service summaries (verbose details go to the log).

        A classmethod so callers that have no step instance can reuse it."""
        from agent_env.artifact import FileArtifact, EnvironmentArtifact, EnvironmentUniverseArtifact
        from agent_env.env.env import gateway_url_of

        services = cls._enumerate_environments(env)
        gateway = gateway_url.rstrip("/")
        # `gateway` identifies which deployment was exported: env= alone doesn't when one env_id has several.
        logger.info(
            f"snapshot_env: env={env.id} v{env.version} snapshot_id={snapshot_id} "
            f"gateway={gateway} services={services}"
        )

        # Carry the source universe's metadata into the snapshot: explicit
        # original id (fail-fast), else the loaded universe (best-effort).
        snapshot_metadata = None
        source_id = original_universe_artifact_id
        explicit = source_id is not None
        source_version: Optional[int] = None
        if not explicit:
            source_id, source_version = cls._resolve_loaded_universe_ref(deployed) or (None, None)
        if source_id:
            try:
                source = await asyncio.to_thread(
                    EnvironmentUniverseArtifact.get, source_id, version=source_version
                )
                snapshot_metadata = await asyncio.to_thread(source.get_metadata) or None
                if not snapshot_metadata:
                    logger.warning("snapshot_env: source universe %s has no metadata to carry", source_id)
            except Exception:
                if explicit:
                    raise  # caller named this id — surface a bad one
                logger.warning("snapshot_env: failed to carry metadata from %s", source_id, exc_info=True)

        # The stored card describes the deployment's own gateway; an override pointing elsewhere keeps the live probe.
        card_record = deployed if deployed is not None and (gateway_url_of(deployed) or "").rstrip("/") == gateway else None
        sem = asyncio.Semaphore(concurrency)

        async def export_one(environment_name: str):
            """Returns EnvironmentArtifact on success, (public_err, log_err) on failure."""
            async with sem:
                # Suffix (.json/.zip) is reported by _export_environment_to_file.
                fd, tmp_path = tempfile.mkstemp(prefix=f"{environment_name}-")
                os.close(fd)
                cleanup_paths = [tmp_path]
                artifact_path = tmp_path
                try:
                    try:
                        result = await cls._export_environment_to_file(
                            gateway, environment_name, tmp_path, export_timeout_seconds,
                            deployed=card_record,
                        )
                    except httpx.HTTPStatusError as e:
                        body_snippet = ""
                        try:
                            body_snippet = (await e.response.aread())[:500].decode(errors="replace")
                        except Exception:
                            pass
                        public = f"export-state returned HTTP {e.response.status_code}"
                        return public, f"{public} from {environment_name}: {body_snippet!r}"
                    except httpx.RequestError as e:
                        public = f"export-state request failed ({type(e).__name__})"
                        return public, f"{public}: {e}"
                    except ValueError as e:
                        return str(e), str(e)

                    # Service uploaded the bundle itself — register a FileArtifact
                    # pointing at it directly (no download). Derive filename/content_type
                    # from the url's last segment (the extension drives load-time parsing),
                    # like FileArtifact.put.
                    if urlparse(result).scheme:
                        from agent_env.artifact.store import get_artifact_store

                        object_url = result
                        filename = object_url.rsplit("/", 1)[-1] or f"{environment_name}.zip"
                        content_type = mimetypes.guess_type(filename)[0] or "application/zip"

                        def _register_uploaded_bundle():
                            if get_config().get_object_store().get_object_metadata_at(object_url) is None:
                                raise FileNotFoundError(f"snapshot bundle not found at {object_url}")
                            store = get_artifact_store()
                            fa_id = f"{snapshot_id}-{environment_name}-file"
                            get_config().check_object_url(fa_id, object_url)
                            fa = FileArtifact(
                                id=fa_id,
                                version=store.next_version(fa_id),
                                description=f"State snapshot of {environment_name} from env {env.id}",
                                filename=filename,
                                content_type=content_type,
                                object_url=object_url,
                            )
                            fa = store.put_document(fa)
                            return EnvironmentArtifact.put(
                                id=f"{snapshot_id}-{environment_name}",
                                environment_name=environment_name,
                                file_artifact=fa,
                            )

                        try:
                            return await asyncio.to_thread(_register_uploaded_bundle)
                        except Exception as e:
                            public = f"failed to register the uploaded bundle ({type(e).__name__})"
                            return public, f"{public} for {environment_name}: {e}"

                    # Otherwise `result` is the written suffix (.json/.zip).
                    # FileArtifact.filename = basename(path); a .zip name routes the
                    # load side to reset_data() (lossless).
                    if result and not tmp_path.endswith(result):
                        artifact_path = tmp_path + result
                        os.rename(tmp_path, artifact_path)
                        cleanup_paths.append(artifact_path)

                    try:
                        file_artifact = await asyncio.to_thread(
                            FileArtifact.put,
                            id=f"{snapshot_id}-{environment_name}-file",
                            description=f"State snapshot of {environment_name} from env {env.id}",
                            file_path=artifact_path,
                        )
                        return await asyncio.to_thread(
                            EnvironmentArtifact.put,
                            id=f"{snapshot_id}-{environment_name}",
                            environment_name=environment_name,
                            file_artifact=file_artifact,
                        )
                    except Exception as e:
                        public = f"failed to create artifacts ({type(e).__name__})"
                        return public, f"{public}: {e}"
                finally:
                    for path in cleanup_paths:
                        try:
                            os.unlink(path)
                        except OSError:
                            pass

        results = await asyncio.gather(*(export_one(name) for name in services))

        environment_artifacts = []
        skipped: dict[str, tuple[str, str]] = {}
        for name, result in zip(services, results):
            if isinstance(result, tuple):
                skipped[name] = result
            else:
                environment_artifacts.append(result)

        if skipped:
            details = "; ".join(f"{name}: {log}" for name, (_, log) in skipped.items())
            logger.error(
                f"snapshot_env: skipped {len(skipped)}/{len(services)} services for "
                f"{env.id} (snapshot_id={snapshot_id}): {details}"
            )
            public_summary = "; ".join(
                f"{name}: {public}" for name, (public, _) in skipped.items()
            )
            raise RuntimeError(
                f"{len(skipped)}/{len(services)} services were not exported "
                f"(missing={sorted(skipped)}): {public_summary}"
            )

        universe = await asyncio.to_thread(
            EnvironmentUniverseArtifact.put,
            id=snapshot_id,
            environment_artifacts=environment_artifacts,
            metadata=snapshot_metadata,
        )
        logger.info(
            f"snapshot_env: created {universe.id} v{universe.version} "
            f"({len(environment_artifacts)} services)"
        )
        return EnvSnapshotResult(
            environment_universe_artifact_id=universe.id,
            environment_universe_artifact_version=universe.version,
            environments_snapshotted=[sa.environment_name for sa in environment_artifacts],
            total=len(services),
        )


def _trajectory_route(deployed, override_url: Optional[str]) -> tuple[str, str]:
    """The verb and URL of the env trajectory: an override keeps `{override}/trajectory`, else the stored card names them."""
    if override_url:
        return "GET", f"{override_url.rstrip('/')}/trajectory"
    from agentenv_protocol import client as protocol_v1

    deployed.require(EXT_TRAJECTORY_URI, "get")
    op = protocol_v1.find_extension_method(deployed.environment_card, EXT_TRAJECTORY_URI, "get")
    return op["method"], f"{deployed.environment_url}{op['endpoint']}"
