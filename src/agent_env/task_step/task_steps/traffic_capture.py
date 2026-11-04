"""Collect a website env's traffic captures. A website or multi env deployed with ``capture_traffic`` records every
packet its websites' containers send and receive as rotating pcap files, which its gateway's card serves as traffic/v1
behind the deploy's control token. ``collect_traffic_capture`` stores each as a FileArtifact and registers them as one
FileArtifactUniverse, recorded in ``context.metadata[TRAFFIC_KEY][env_id]``. Teardown collects any capturing env a run
didn't, before the env closes (``collect_uncollected_captures``), so a capture isn't lost with its VM."""
from __future__ import annotations

import functools
import logging
import os
import tempfile
from collections.abc import Sequence
from typing import ClassVar, Optional

from agentenv_protocol.traffic import (
    PCAP_MEDIA_TYPE,
    TRAFFIC_EXTENSION_URI,
    TRAFFIC_LIST_OP,
    TrafficCapture,
    fetch_traffic_capture,
    list_traffic_captures,
    serves_traffic,
)

from agent_env.artifact import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.store import get_artifact_store
from agent_env.config import get_config
from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedEnv, EnvCapabilityUnsupported
from agent_env.store.ids import derive_id, is_local_id, validate_local_id
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.thread_work import finish_on_thread

logger = logging.getLogger(__name__)

#: ``context.metadata[TRAFFIC_KEY][env_id]``: each env whose captures a run collected, or tried to.
TRAFFIC_KEY = "traffic_captures"
_ARTIFACT_TYPE = "traffic_captures"
_ERROR_CHARS = 500


class CollectTrafficCaptureTaskStep(TaskStep):
    """Store the traffic captures of the deployed env ``env_id`` as pcap FileArtifacts. Best effort by default: a
    failure is recorded and logged, and fails the task only with ``fail_task_on_error``."""

    type: ClassVar[str] = "collect_traffic_capture"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = False):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not env_id:
            raise ValueError("collect_traffic_capture needs the env_id of an env deployed with capture_traffic")
        self.env_id = env_id

    def to_dict(self) -> dict:
        return {**super().to_dict(), "env_id": self.env_id}

    @classmethod
    def from_dict(cls, data: dict) -> CollectTrafficCaptureTaskStep:
        base = cls._base_from_dict(data)
        base["fail_task_on_error"] = data.get("fail_task_on_error", False)
        return cls(**base, env_id=data["env_id"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            error = RuntimeError(f"env {self.env_id!r} isn't deployed in this run")
            context.metadata.setdefault(TRAFFIC_KEY, {})[self.env_id] = _failed(error)
            logger.warning("collect_traffic_capture: %s", error)
            if self.fail_task_on_error:
                raise error
            return context
        entry = await collect_captures(context, deployed)
        if self.fail_task_on_error and entry["state"] != "collected":
            raise RuntimeError(f"env {self.env_id!r}'s traffic captures weren't collected: {entry.get('error')}")
        return context


def captures_traffic(deployed: DeployedEnv) -> bool:
    """Whether the stored card of ``deployed`` serves traffic/v1, as a deploy with ``capture_traffic`` does; no
    network."""
    return isinstance(deployed, DeployedEnv) and serves_traffic(deployed.environment_card)


async def collect_uncollected_captures(context: TaskStepContext, envs: Sequence[DeployedEnv]) -> None:
    """Collect the captures of each of ``envs`` that serves traffic/v1 and whose captures the run hasn't collected, a
    collection that failed tried again, as teardown does before the envs close; ``collect_captures`` records what
    happened to each."""
    entries = context.metadata.get(TRAFFIC_KEY) or {}
    for deployed in envs:
        if captures_traffic(deployed) and (entries.get(deployed.env_id) or {}).get("state") != "collected":
            await collect_captures(context, deployed)


async def collect_captures(context: TaskStepContext, deployed: DeployedEnv) -> dict:
    """Store each capture ``deployed`` serves as a pcap FileArtifact, register them as one FileArtifactUniverse, and
    record it in ``context.metadata[TRAFFIC_KEY]``. Each file is named for its container and its place among that
    container's files, oldest first, so they read in the order their packets were captured. Never raises: the entry,
    which it also returns, says what happened."""
    entries = context.metadata.setdefault(TRAFFIC_KEY, {})
    try:
        if not captures_traffic(deployed):
            raise EnvCapabilityUnsupported(TRAFFIC_EXTENSION_URI, TRAFFIC_LIST_OP, deployed.env_id, deployed.instance_id)
        listing = await list_traffic_captures(deployed.environment_url, deployed.environment_card,
                                              headers=deployed.control_headers)
        entries[deployed.env_id] = await _store(context, deployed, listing.captures)
    except Exception as e:
        logger.warning("collect_traffic_capture %s: %s", deployed.env_id, e)
        entries[deployed.env_id] = _failed(e)
    return entries[deployed.env_id]


async def _store(context: TaskStepContext, deployed: DeployedEnv, captures: list[TrafficCapture]) -> dict:
    universe_id = derive_id(context.instance_id or "run", f"traffic-{deployed.env_id}")
    if is_local_id(universe_id):
        validate_local_id(universe_id)
    get_config().check_local_run_write(universe_id)
    store = get_artifact_store()
    prefix = store.attempt_prefix(_ARTIFACT_TYPE, universe_id)
    files, total, seen = {}, 0, {}
    for capture in captures:
        container = capture.container
        seen[container] = seen.get(container, -1) + 1
        filename = f"{container}-{seen[container]:02d}.pcap"
        handle, path = tempfile.mkstemp(suffix=".pcap")
        os.close(handle)
        try:
            total += await fetch_traffic_capture(deployed.environment_url, deployed.environment_card, capture.name,
                                                 path, headers=deployed.control_headers)
        except BaseException:
            _remove(path)
            raise
        url = await finish_on_thread(functools.partial(_upload, universe_id, prefix + filename, path),
                                     f"Uploading traffic capture {filename} of env {deployed.env_id!r}",
                                     if_never_run=functools.partial(_remove, path))
        files[filename] = store.put_document(FileArtifact(
            id=derive_id(universe_id, filename), filename=filename, content_type=PCAP_MEDIA_TYPE, object_url=url,
            description=f"Traffic capture of {container} in env {deployed.env_id!r}, task instance {context.instance_id}"))
    artifact = None
    if files:
        universe = FileArtifactUniverse.put(id=universe_id, file_artifacts=files)
        artifact = {"id": universe.id, "version": universe.version}
    logger.info("collect_traffic_capture %s: %d capture files, %d bytes", deployed.env_id, len(files), total)
    return {"state": "collected", "artifact": artifact, "files": sorted(files), "bytes": total,
            "containers": sorted(seen)}


def _upload(universe_id: str, object_url: str, path: str) -> str:
    """Upload the capture at ``path`` to ``object_url``, write-once, and remove it: the upload owns the file, so a
    cancelled collection never removes it from under the upload."""
    try:
        return get_config().get_object_store_to_write(object_url, universe_id).put_file_at(object_url, path,
                                                                                            PCAP_MEDIA_TYPE)
    finally:
        _remove(path)


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _failed(error: BaseException) -> dict:
    return {"state": "failed", "error": f"{type(error).__name__}: {error}"[:_ERROR_CHARS]}
