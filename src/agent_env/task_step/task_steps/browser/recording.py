"""Record a website browser. ``start_browser_recording`` starts browser-recording/v1 on the browser in a deployed env,
with frames by default, under a grant to upload its frames, log and manifest in a namespace of its own;
``end_browser_recording`` ends it and registers them as one FileArtifactUniverse, and fails when its log didn't upload
or, with frames, when a frame is missing: an infrastructure failure. Teardown ends what a run left open, with fresh
grants, before any env closes (``end_open_browser_recordings``), and registers from its namespace one it couldn't end
(``register_open_browser_recordings``)."""
from __future__ import annotations

import json
import logging
import uuid
from typing import ClassVar, Optional

from agentenv_protocol.client import find_extension
from agentenv_protocol.website_browser import BROWSER_RECORDING, BrowserRecordingManifest, RecordedFrame
from agentenv_protocol.website_browser.recording import END_S, LOG, MANIFEST, MAX_FRAMES, START_S, frame_of

from agent_env.artifact import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.store import get_artifact_store
from agent_env.config import get_config
from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedEnv, EnvCapabilityUnsupported
from agent_env.store.ids import derive_id, is_local_id, validate_local_id
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.browser.base import BrowserStep, deployed_env, find_browser
from agent_env.task_step.task_steps.recordings import (
    ERROR_CHARS,
    end_open,
    recording_grant,
    recording_namespace,
    refuse_no_network,
    register_open,
    why,
)

logger = logging.getLogger(__name__)

#: ``context.metadata[RECORDINGS_KEY][name]``: each browser recording a run started, open or ended.
RECORDINGS_KEY = "browser_recordings"
_KEY_PREFIX = "agentenv-browser-recordings"
#: A grant's limits: each frame or the log at most this big, and the whole recording at most this.
MAX_OBJECT_BYTES = 256 << 20
MAX_RECORDING_BYTES = 16 << 30
#: What a recording uploads besides its frames, which its grant has room for too.
BESIDE_FRAMES = (LOG, MANIFEST)


class StartBrowserRecordingTaskStep(BrowserStep):
    """Start recording the website browser in the env ``env_id`` (the one ``browser`` names, else its only one) under
    ``name``: with ``frames``, a frame of the tab an agent last acted in at the start, after each burst of its actions
    and at the end. A start that fails fails the task unless ``fail_task_on_error`` is off."""

    type: ClassVar[str] = "start_browser_recording"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, name: str, frames: bool = True,
                 browser: Optional[str] = None, depends_on: Optional[list[TaskStepDependency]] = None,
                 fail_task_on_error: bool = True):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not name:
            raise ValueError("start_browser_recording needs a name for the recording")
        self.name, self.frames = name, frames

    def to_dict(self) -> dict:
        return {**super().to_dict(), "name": self.name, "frames": self.frames}

    @classmethod
    def from_dict(cls, data: dict) -> StartBrowserRecordingTaskStep:
        return cls(**cls._browser_from_dict(data), name=data["name"], frames=data.get("frames", True))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        recordings = context.metadata.setdefault(RECORDINGS_KEY, {})
        if (recordings.get(self.name) or {}).get("state") == "open":
            raise RuntimeError(f"a browser recording named {self.name!r} is already open in this run; give this one "
                               "another name")
        # A retried step joins its first attempt's recording, so it keeps that attempt's namespace too.
        request_id = f"{context.instance_id}:{self.id}" if context.instance_id else uuid.uuid4().hex
        try:
            record, browser, card = self._find(context)
            refuse_no_network(record)
            namespace = recording_namespace(request_id, _KEY_PREFIX)
            started = await record.invoke(BROWSER_RECORDING, "start", {
                "request_id": request_id, "frames": self.frames, "upload": _grant(record, card, namespace)},
                environment_name=browser, timeout=START_S)
        except Exception as e:
            recordings[self.name] = {"env_id": self.env_id, "browser": self.browser, "frames": self.frames,
                                     "state": "failed", "error": why(e)}
            logger.warning("start_browser_recording %s on %s: %s", self.name, self.env_id, e)
            if self.fail_task_on_error:
                raise
            return context
        recordings[self.name] = {
            "env_id": self.env_id, "browser": browser, "frames": self.frames, "recording_id": started["recording_id"],
            "started_at": started["started_at"], "namespace": namespace, "state": "open"}
        logger.info("start_browser_recording %s on %s: %s", self.name, self.env_id, started["recording_id"])
        return context


class EndBrowserRecordingTaskStep(TaskStep):
    """End the browser recording ``name`` and register it; fail, unless ``fail_task_on_error`` is off, when it didn't
    end, its log didn't upload, or, with frames, a frame is missing."""

    type: ClassVar[str] = "end_browser_recording"
    entity_refs = ()

    def __init__(self, id: str, version: Optional[int], name: str,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not name:
            raise ValueError("end_browser_recording needs the name its start_browser_recording gave")
        self.name = name

    def to_dict(self) -> dict:
        return {**super().to_dict(), "name": self.name}

    @classmethod
    def from_dict(cls, data: dict) -> EndBrowserRecordingTaskStep:
        return cls(**cls._base_from_dict(data), name=data["name"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        entry = (context.metadata.get(RECORDINGS_KEY) or {}).get(self.name)
        if entry is None:
            raise RuntimeError(f"no browser recording named {self.name!r} was started in this run")
        await end_browser_recording(context, self.name, "end")
        if self.fail_task_on_error and (problem := incomplete(entry)):
            raise RuntimeError(f"browser recording {self.name!r} is incomplete, an infrastructure failure: {problem}")
        return context


def incomplete(entry: dict) -> Optional[str]:
    """What's missing from the recording ``entry`` records: that it didn't end, its log or entries of it, or its
    frames; None for a complete one."""
    if entry.get("state") != "ended":
        return f"it didn't end: {entry.get('error')}"
    if not entry.get("log_uploaded"):
        return f"its log didn't upload: {entry.get('error')}"
    if entry.get("log_missing"):
        return f"its log is missing {entry['log_missing']} of the entries between its start and its end"
    if entry.get("frames") and entry.get("missing_frames"):
        return f"{entry['missing_frames']} of its {entry['frames_taken'] + entry['missing_frames']} frames are missing"
    return None


async def end_open_browser_recordings(context: TaskStepContext, env_ids: Optional[list[str]] = None) -> None:
    """End and register every browser recording still open in ``context`` (of ``env_ids`` only, when given), each with
    a fresh grant, before its env is torn down; ``end_browser_recording`` says what happened to each."""
    await end_open(context, RECORDINGS_KEY, env_ids, end_browser_recording)


def register_open_browser_recordings(context: TaskStepContext, env_ids: Optional[list[str]] = None) -> None:
    """Register every browser recording still open in ``context`` (of ``env_ids`` only, when given) from what its
    namespace holds, once its env has closed. Never raises: each entry says what happened."""
    register_open(context, RECORDINGS_KEY, env_ids, _from_namespace, _registered, "browser recording")


async def end_browser_recording(context: TaskStepContext, name: str, reason: str) -> None:
    """End the open browser recording ``name`` with a fresh grant and register it; when the browser no longer knows
    it, as after its server restarted, what its namespace holds is registered instead. Never raises: the entry says
    what happened."""
    entry = context.metadata[RECORDINGS_KEY][name]
    if entry.get("state") != "open":
        return
    try:
        record, browser, card = find_browser(deployed_env(context, entry["env_id"]), entry["browser"])
        manifest = BrowserRecordingManifest.model_validate(await record.invoke(BROWSER_RECORDING, "end", {
            "recording_id": entry["recording_id"], "upload": _grant(record, card, entry["namespace"], "end"),
            "reason": reason}, environment_name=browser, timeout=END_S))
    except Exception as e:
        logger.warning("end_browser_recording %s: %s; registering what its namespace holds", name, e)
        entry["error"] = why(e)
        try:
            manifest = _from_namespace(entry)
        except Exception as listing:
            logger.warning("end_browser_recording %s: listing its namespace failed: %s", name, listing)
            entry["state"], entry["error"] = "failed", f"{entry['error']}; then {why(listing)}"[:ERROR_CHARS]
            return
    _registered(context, name, entry, manifest)


def _grant(record: DeployedEnv, card: dict, namespace: str, method: str = "start") -> dict:
    """A grant for a browser recording's frames, log and manifest; EnvCapabilityUnsupported for a browser whose card
    declares no recording."""
    extension = find_extension(card or {}, BROWSER_RECORDING)
    if extension is None:
        raise EnvCapabilityUnsupported(BROWSER_RECORDING, method, record.env_id, getattr(record, "instance_id", None),
                                       card.get("name"))
    params = extension.get("params") or {}
    return recording_grant(record, params, namespace, params.get("max_frames", MAX_FRAMES) + len(BESIDE_FRAMES),
                           MAX_OBJECT_BYTES, MAX_RECORDING_BYTES)


def _registered(context: TaskStepContext, name: str, entry: dict, manifest: BrowserRecordingManifest) -> None:
    """Register the recording ``name`` as ``manifest`` says it went, and its entry as ended, or as failed."""
    try:
        entry["artifact"] = _register(context, name, entry["namespace"], manifest)
    except Exception as e:
        logger.warning("browser recording %s: registering it failed: %s", name, e)
        entry["state"], entry["error"] = "failed", why(e)
        return
    taken = [frame for frame in manifest.frames if frame.uploaded]
    entry.update(state="ended", ended_by=manifest.ended_by, frames_taken=len(taken),
                 missing_frames=manifest.missing_frames, log_uploaded=manifest.log_uploaded,
                 log_missing=len(manifest.log_missing))
    if manifest.error:
        entry["error"] = manifest.error
    logger.info("browser recording %s: %d frames, %d missing, ended by %s", name, len(taken), manifest.missing_frames,
                manifest.ended_by)


def _from_namespace(entry: dict) -> BrowserRecordingManifest:
    """What the recording's namespace holds, for one its browser didn't end for the run: the manifest the browser
    uploaded when it ended it, or else its frames there, each by the index its file is named by, its kind and when it
    was taken unknown, and its log if that uploaded."""
    store = get_config().get_object_store()
    root = f"{entry['namespace'].rstrip('/')}/{entry.get('recording_id')}/"
    urls = store.list_at(root)
    if (uploaded := f"{root}{MANIFEST}") in urls:
        return BrowserRecordingManifest.model_validate(json.loads(store.get(uploaded)))
    frames = sorted((found, url[len(root):]) for url in urls if (found := frame_of(url[len(root):])))
    logged = f"{root}{LOG}" in urls
    return BrowserRecordingManifest(
        recording_id=entry.get("recording_id") or "", started_at=entry.get("started_at") or "", ended_by="error",
        error=entry.get("error"), frames_on=bool(entry.get("frames")), log_uploaded=logged, log=LOG if logged else None,
        frames=[RecordedFrame(index=index, path=path, content_type=content_type, uploaded=True)
                for (index, content_type), path in frames])


def _register(context: TaskStepContext, name: str, namespace: str, manifest: BrowserRecordingManifest) -> dict:
    """The manifest, the log and the uploaded frames as FileArtifacts, pointing at the objects, in one universe."""
    universe_id = derive_id(context.instance_id or "run", f"browser-recording-{name}")
    if is_local_id(universe_id):
        validate_local_id(universe_id)
    objects, documents = get_config().get_object_store(), get_artifact_store()
    root = f"{namespace.rstrip('/')}/{manifest.recording_id}"
    manifest_url = objects.put(objects.get_object_key(f"{root}/{MANIFEST}"),
                               json.dumps(manifest.model_dump(mode="json"), indent=1).encode(),
                               content_type="application/json", allow_overwrite=True)
    files = {MANIFEST: (manifest_url, "application/json")}
    if manifest.log_uploaded and manifest.log:
        files[manifest.log] = (f"{root}/{manifest.log}", "application/x-ndjson")
    for frame in manifest.frames:
        if frame.uploaded and frame.path:
            files[frame.path.replace("/", "-")] = (f"{root}/{frame.path}", frame.content_type)
    stored = {filename: documents.put_document(FileArtifact(
        id=derive_id(universe_id, filename), filename=filename, content_type=content_type, object_url=url,
        description=f"Browser recording {name!r} of task instance {context.instance_id}"))
        for filename, (url, content_type) in files.items()}
    universe = FileArtifactUniverse.put(id=universe_id, file_artifacts=stored)
    return {"id": universe.id, "version": universe.version}
