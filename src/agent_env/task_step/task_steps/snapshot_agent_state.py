"""Snapshot a deployed A2A agent's conversation state to a FileArtifactUniverse."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef, RefRole
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.snapshot_utils import agent_state_capture

logger = logging.getLogger(__name__)


class SnapshotAgentStateTaskStep(TaskStep):
    """Capture a deployed agent's conversation state and persist it as a
    ``FileArtifactUniverse``.

    Looks up the deployed agent (by ``agent_name``) and the source ``context_id``
    via a previously-recorded ``PromptResponse`` (by ``prompt_id``). Calls the
    agent's ``/ext/snapshot`` extension to write the conversation transcript to
    the object store, then wraps that prefix as a ``FileArtifactUniverse`` via
    ``put_existing`` (no download/re-upload). Appends an entry recording
    ``{id, version, bundle_object_url, source_agent_name, source_context_id}`` to
    ``context.metadata['agent_snapshots']`` (a list of such entries).

    When ``env_id`` is set, the step ALSO reads each MCP service's current
    state from the deployed env (its ``data/get`` JSON, else its
    ``/export-state``), uploads each as JSON alongside the workspace tarball, and
    writes a JSON-stringified ``{service: object_url}`` (unsigned object-store
    references) to ``context.metadata['snapshot_json_url']``. A downstream
    verifier re-signs each value before handing the map to its test script, so
    the script can fetch each service's state and assert against it. Without this, only the
    agent's *workspace files* are visible to the verifier — not per-MCP-service
    DB state.
    """

    type: ClassVar[str] = "snapshot_agent_state"
    entity_refs = (
        EntityRef.artifact("artifact_id", role=RefRole.OUTPUT, artifact_type="file_artifact_universe"),
        EntityRef.env("env_id"),
        EntityRef.artifact("universe_artifact_id", artifact_type="environment_universe"),
    )
    DEFAULT_TIMEOUT_SECONDS: ClassVar[int] = 120

    def __init__(
        self,
        id: str,
        version: Optional[int],
        artifact_id: str,
        prompt_id: str,
        agent_name: Optional[str] = None,
        env_id: Optional[str] = None,
        universe_artifact_id: Optional[str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.artifact_id = artifact_id
        self.prompt_id = prompt_id
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        # Set both `env_id` (which deployed env to query for /export-state) AND
        # `universe_artifact_id` (to enumerate the env's service names) to
        # enable per-MCP-service capture. Leaving either unset captures only
        # the agent's workspace (legacy behaviour).
        self.env_id = env_id
        self.universe_artifact_id = universe_artifact_id
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["artifact_id"] = self.artifact_id
        base["prompt_id"] = self.prompt_id
        base["agent_name"] = self.agent_name
        base["env_id"] = self.env_id
        base["universe_artifact_id"] = self.universe_artifact_id
        base["timeout_seconds"] = self.timeout_seconds
        return base

    @classmethod
    def from_dict(cls, data: dict) -> SnapshotAgentStateTaskStep:
        return cls(
            **cls._base_from_dict(data),
            artifact_id=data["artifact_id"],
            prompt_id=data["prompt_id"],
            agent_name=data.get("agent_name"),
            env_id=data.get("env_id"),
            universe_artifact_id=data.get("universe_artifact_id"),
            timeout_seconds=data.get("timeout_seconds", cls.DEFAULT_TIMEOUT_SECONDS),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
        if deployed is None:
            raise RuntimeError(f"No deployed agent named '{self.agent_name}' found in context")
        a2a_url = deployed.a2a_url or deployed.api_url
        if not a2a_url:
            raise RuntimeError(f"Deployed agent '{self.agent_name}' has no a2a_url")

        prompt = next((p for p in context.prompt_responses if p.prompt_id == self.prompt_id), None)
        if prompt is None:
            raise RuntimeError(f"No PromptResponse with prompt_id '{self.prompt_id}' in context")
        a2a_context_id = prompt.a2a_context_id
        if not a2a_context_id:
            raise RuntimeError(
                f"PromptResponse for '{self.prompt_id}' has no a2a_context_id "
                "(prompt step did not capture one — make sure the agent is using a recent gateway)"
            )

        workspace = await agent_state_capture.capture_workspace(
            a2a_url=a2a_url,
            a2a_card=deployed.a2a_card or {},
            agent_name=self.agent_name,
            a2a_context_id=a2a_context_id,
            artifact_id=self.artifact_id,
            timeout_seconds=self.timeout_seconds,
            sandbox_type=deployed.sandbox_type,
        )

        snapshots = context.metadata.setdefault("agent_snapshots", [])
        snapshots.append({
            "id": workspace.universe_id,
            "version": workspace.universe_version,
            "bundle_object_url": workspace.bundle_object_url,
            "source_agent_name": self.agent_name,
            "source_context_id": a2a_context_id,
        })

        # Per-MCP-service state dump. Only runs when env_id +
        # universe_artifact_id are both configured. Mirrors the export-state
        # pattern in `multienv_validator/verify_universe_roundtrip.py:_export_all`.
        # Result lands in `context.metadata['snapshot_json_url']` as a
        # JSON-stringified `{service: object_url}` (unsigned object-store references).
        # A downstream verifier re-signs each value and hands the map to its test
        # script, which accepts either a single URL or this map.
        if self.env_id and self.universe_artifact_id:
            await self._capture_universe_state(context, workspace.capture_prefix)

        return context

    async def _capture_universe_state(
        self, context: TaskStepContext, capture_prefix: str
    ) -> None:
        """Read each service's state as JSON, upload it beside the workspace capture, and publish the
        object URLs."""
        from agent_env.artifact import EnvironmentUniverseArtifact
        from agent_env.config import get_config

        deployed_env = next(
            (d for d in context.deployed_envs if d.env_id == self.env_id), None
        )
        if deployed_env is None:
            logger.warning(
                "snapshot_agent_state: env_id=%s not found in deployed_envs (%s); "
                "skipping per-service capture",
                self.env_id,
                [d.env_id for d in context.deployed_envs],
            )
            return

        try:
            universe = EnvironmentUniverseArtifact.get(self.universe_artifact_id)
            environment_names = [sa.environment_name for sa in universe.get_environment_artifacts()]
        except Exception as exc:
            logger.warning(
                "snapshot_agent_state: could not enumerate services from universe %s: %s",
                self.universe_artifact_id, exc,
            )
            return

        # Reuse the workspace capture's prefix, in the store that holds it — services land in a
        # `services/` subdir alongside the workspace tarball.
        store = get_config().get_object_store_at(capture_prefix)
        key_prefix = store.get_object_key(capture_prefix).rstrip("/") + "/"

        from agent_env.env import legacy_protocol
        from agent_env.env.env import gateway_url_of
        urls: dict[str, str] = {}
        for name in environment_names:
            try:
                state = await legacy_protocol.service_state(deployed_env, gateway_url_of(deployed_env), name)
                urls[name] = await asyncio.to_thread(
                    store.put, f"{key_prefix}services/{name}.json", json.dumps(state).encode(),
                    content_type="application/json", allow_overwrite=True,
                )
            except Exception as exc:
                logger.warning(
                    "snapshot_agent_state: capturing %s state failed: %s — skipping",
                    name, exc,
                )
                continue

        if urls:
            context.metadata["snapshot_json_url"] = json.dumps(urls)
            logger.info(
                "snapshot_agent_state: captured %d service states → snapshot_json_url",
                len(urls),
            )
