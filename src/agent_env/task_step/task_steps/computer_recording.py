"""Record a computer env's display. ``start_computer_recording`` starts a recording on a deployed env whose card, or
whose child env's card (``environment_name``), declares computer-recording/v1, with a grant to upload its segments
under a namespace of its own;
``end_computer_recording`` ends it and registers its segments and manifest as one FileArtifactUniverse. Teardown
ends whatever a run left open (``end_open_recordings``)."""
from __future__ import annotations

import json
import logging
import uuid
from typing import ClassVar, Optional

from agentenv_protocol.computer.card import COMPUTER_RECORDING
from agentenv_protocol.computer.manifest import ManifestSegment, RecordingManifest
from agentenv_protocol.client import find_extension

from agent_env.a2a_agent.object_transfer import ObjectLimits, namespace_grant
from agent_env.artifact import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.store import get_artifact_store
from agent_env.config import get_config
from agent_env.entity.refs import EntityRef
from agent_env.store.base import GrantUnavailableError
from agent_env.store.ids import derive_id, is_local_id, validate_local_id
from agent_env.store.object_store.object_store import issues_grants_to
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

#: ``context.metadata[RECORDINGS_KEY][name]``: each recording a run started, open or ended.
RECORDINGS_KEY = "computer_recordings"
_KEY_PREFIX = "agentenv-recordings"
#: How much longer than the env's ``max_duration_s`` a recording's grant lasts.
GRANT_MARGIN_S = 600
_MAX_SEGMENT_BYTES = 5 << 30
#: The segments a grant takes: a recording also rolls over early, on a size change or after a gap.
MAX_SEGMENTS = 10_000
MAX_RECORDING_BYTES = 64 << 30


class StartComputerRecordingTaskStep(TaskStep):
    """Start recording a deployed env's display, or its child env ``environment_name``'s, under ``name``. Best effort
    by default."""

    type: ClassVar[str] = "start_computer_recording"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, name: str, fps: Optional[int] = None,
                 max_edge: Optional[int] = None, cursor: bool = True, environment_name: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = False):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not env_id or not name:
            raise ValueError("start_computer_recording needs an env_id and a name for the recording")
        self.env_id, self.name, self.fps, self.max_edge, self.cursor = env_id, name, fps, max_edge, cursor
        self.environment_name = environment_name

    def to_dict(self) -> dict:
        data = {**super().to_dict(), "env_id": self.env_id, "name": self.name, "fps": self.fps,
                "max_edge": self.max_edge, "cursor": self.cursor}
        if self.environment_name is not None:  # left out otherwise, so a step without one serializes as before
            data["environment_name"] = self.environment_name
        return data

    @classmethod
    def from_dict(cls, data: dict) -> StartComputerRecordingTaskStep:
        base = cls._base_from_dict(data)
        base["fail_task_on_error"] = data.get("fail_task_on_error", False)
        return cls(**base, env_id=data["env_id"], name=data["name"], fps=data.get("fps"),
                   max_edge=data.get("max_edge"), cursor=data.get("cursor", True),
                   environment_name=data.get("environment_name"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        recordings = context.metadata.setdefault(RECORDINGS_KEY, {})
        if (recordings.get(self.name) or {}).get("state") == "open":
            error = RuntimeError(f"a recording named {self.name!r} is already open in this run; give this one "
                                 "another name")
            logger.warning("start_computer_recording: %s", error)
            if self.fail_task_on_error:
                raise error
            return context
        # A retried step joins its first attempt's recording, so it keeps that attempt's namespace too.
        request_id = f"{context.instance_id}:{self.id}" if context.instance_id else uuid.uuid4().hex
        try:
            deployed = _deployed(context, self.env_id)
            namespace = _namespace(request_id)
            grant = _grant(deployed, namespace, self.environment_name)
            started = await deployed.invoke(COMPUTER_RECORDING, "start", {
                "request_id": request_id, "fps": self.fps, "max_edge": self.max_edge, "cursor": self.cursor,
                "upload": grant}, environment_name=self.environment_name)
        except Exception as e:
            recordings[self.name] = {"env_id": self.env_id, "state": "failed", "error": f"{type(e).__name__}: {e}"[:500]}
            logger.warning("start_computer_recording %s on %s: %s", self.name, self.env_id, e)
            if self.fail_task_on_error:
                raise
            return context
        recordings[self.name] = {
            "env_id": self.env_id, "recording_id": started["recording_id"], "started_at": started["started_at"],
            "namespace": namespace, "state": "open"}
        if self.environment_name is not None:
            recordings[self.name]["environment_name"] = self.environment_name
        logger.info("start_computer_recording %s on %s: %s", self.name, self.env_id, started["recording_id"])
        return context


class EndComputerRecordingTaskStep(TaskStep):
    """End the recording ``name`` and register it. Best effort by default."""

    type: ClassVar[str] = "end_computer_recording"
    entity_refs = ()

    def __init__(self, id: str, version: Optional[int], name: str,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = False):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not name:
            raise ValueError("end_computer_recording needs the name its start_computer_recording gave")
        self.name = name

    def to_dict(self) -> dict:
        return {**super().to_dict(), "name": self.name}

    @classmethod
    def from_dict(cls, data: dict) -> EndComputerRecordingTaskStep:
        base = cls._base_from_dict(data)
        base["fail_task_on_error"] = data.get("fail_task_on_error", False)
        return cls(**base, name=data["name"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        entry = (context.metadata.get(RECORDINGS_KEY) or {}).get(self.name)
        if entry is None:
            error = RuntimeError(f"no recording named {self.name!r} was started in this run")
            logger.warning("end_computer_recording: %s", error)
            if self.fail_task_on_error:
                raise error
            return context
        await end_recording(context, self.name, "end")
        if self.fail_task_on_error and entry["state"] != "ended":
            raise RuntimeError(f"recording {self.name!r} didn't end: {entry.get('error')}")
        return context


async def end_open_recordings(context: TaskStepContext, env_ids: Optional[list[str]] = None) -> None:
    """End and register every recording still open in ``context`` (of ``env_ids`` only, when given), before its
    computer is torn down."""
    for name, entry in list((context.metadata.get(RECORDINGS_KEY) or {}).items()):
        if entry.get("state") == "open" and (env_ids is None or entry["env_id"] in env_ids):
            try:
                await end_recording(context, name, "teardown")
            except Exception as e:  # what tears the computer down goes on regardless
                entry["state"], entry["error"] = "failed", f"{type(e).__name__}: {e}"[:500]
                logger.warning("ending recording %s: %s", name, e)


async def end_recording(context: TaskStepContext, name: str, reason: str) -> None:
    """End the open recording ``name`` with a fresh grant and register it. When the driver no longer knows it, as
    after a restart, what its namespace holds is registered instead. Never raises: the entry says what happened."""
    entry = context.metadata[RECORDINGS_KEY][name]
    if entry.get("state") != "open":
        return
    try:
        deployed, environment_name = _deployed(context, entry["env_id"]), entry.get("environment_name")
        manifest = RecordingManifest.model_validate(await deployed.invoke(COMPUTER_RECORDING, "end", {
            "recording_id": entry["recording_id"], "upload": _grant(deployed, entry["namespace"], environment_name),
            "reason": reason}, environment_name=environment_name))
    except Exception as e:
        logger.warning("end_computer_recording %s: %s; registering what its namespace holds", name, e)
        entry["error"] = f"{type(e).__name__}: {e}"[:500]
        try:
            manifest = _from_namespace(entry)
        except Exception as listing:
            logger.warning("end_computer_recording %s: listing its namespace failed: %s", name, listing)
            entry["state"], entry["error"] = "failed", f"{entry['error']}; then {type(listing).__name__}: {listing}"[:500]
            return
    try:
        entry["artifact"] = _register(context, name, entry["namespace"], manifest)
    except Exception as e:
        logger.warning("end_computer_recording %s: registering it failed: %s", name, e)
        entry["state"], entry["error"] = "failed", f"{type(e).__name__}: {e}"[:500]
        return
    uploaded = [s for s in manifest.segments if s.uploaded]
    entry.update(state="ended", ended_by=manifest.ended_by, segments=len(uploaded),
                 missing_segments=len(manifest.segments) - len(uploaded))
    logger.info("end_computer_recording %s: %d segments, ended by %s", name, len(uploaded), manifest.ended_by)


def _deployed(context: TaskStepContext, env_id: str):
    deployed = next((d for d in context.deployed_envs if d.env_id == env_id), None)
    if deployed is None:
        raise RuntimeError(f"env {env_id!r} isn't deployed in this run")
    return deployed


def _namespace(request_id: str) -> str:
    """Where a recording's segments go: named by its start's request id, so a retried start uploads alongside."""
    config = get_config()
    key = uuid.uuid5(uuid.NAMESPACE_URL, request_id).hex
    return config.get_object_store().object_url(f"{config.get_artifact_key_prefix()}{_KEY_PREFIX}/{key}")


def _grant(deployed, namespace: str, environment_name: Optional[str] = None) -> dict:
    """A grant that uploads under ``namespace``, lasting the recorded env's ``max_duration_s`` and a margin, or the
    longest the store issues."""
    card = deployed.environment_card if environment_name is None else deployed.get_child_env_card(environment_name)
    params = (find_extension(card or {}, COMPUTER_RECORDING) or {}).get("params") or {}
    store = get_config().get_object_store()
    sandbox_type = getattr(deployed, "sandbox_type", None)
    if not issues_grants_to(store, sandbox_type):
        raise RuntimeError(f"the object store issues no upload grant that reaches env {deployed.env_id}'s sandbox "
                           f"({sandbox_type or 'unknown'}), so its recording couldn't upload")
    duration = params["max_duration_s"]
    cap = min(store.max_single_upload_bytes or _MAX_SEGMENT_BYTES, _MAX_SEGMENT_BYTES)
    limits = ObjectLimits(max_objects=MAX_SEGMENTS, max_object_bytes=cap, max_total_bytes=MAX_RECORDING_BYTES)
    try:
        grant = namespace_grant(store, namespace, limits=limits, expires_in=duration + GRANT_MARGIN_S)
    except GrantUnavailableError:
        grant = namespace_grant(store, namespace, limits=limits, expires_in=store.grant_lifetime_seconds)
    return grant.model_dump(mode="json")


def _from_namespace(entry: dict) -> RecordingManifest:
    """A manifest of what the recording's namespace holds, for a recording its driver can't end."""
    store = get_config().get_object_store()
    root = entry["namespace"].rstrip("/") + "/"
    paths = sorted(url[len(root):] for url in store.list_at(root) if url.endswith(".mp4"))
    return RecordingManifest(recording_id=entry.get("recording_id"), ended_by="error", error=entry.get("error"),
                             segments=[ManifestSegment(index=i, path=path, uploaded=True)
                                       for i, path in enumerate(paths)])


def _register(context: TaskStepContext, name: str, namespace: str, manifest: RecordingManifest) -> dict:
    """The uploaded segments and the manifest as FileArtifacts, pointing at the objects, in one universe."""
    universe_id = derive_id(context.instance_id or "run", f"recording-{name}")
    if is_local_id(universe_id):
        validate_local_id(universe_id)
    objects, documents = get_config().get_object_store(), get_artifact_store()
    root = namespace.rstrip("/")
    manifest_url = objects.put(objects.get_object_key(f"{root}/manifest.json"),
                               json.dumps(manifest.wire(), indent=1).encode(), content_type="application/json",
                               allow_overwrite=True)
    files = {"manifest.json": (manifest_url, "application/json")}
    for segment in manifest.segments:
        if segment.uploaded:
            files[segment.path.replace("/", "-")] = (f"{root}/{segment.path}", "video/mp4")
    stored = {filename: documents.put_document(FileArtifact(
        id=derive_id(universe_id, filename), filename=filename, content_type=content_type, object_url=url,
        description=f"Recording {name!r} of task instance {context.instance_id}"))
        for filename, (url, content_type) in files.items()}
    universe = FileArtifactUniverse.put(id=universe_id, file_artifacts=stored)
    return {"id": universe.id, "version": universe.version}
