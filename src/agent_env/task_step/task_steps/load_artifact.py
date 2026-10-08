"""Load artifact into a deployed environment task step."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import posixpath
import shlex
import uuid
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional
from urllib.parse import unquote, urlparse

from agent_env.providers.sandbox_providers.sandbox_provider import reachable_url
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.utils.paths import validate_relative_filename

logger = logging.getLogger(__name__)

# Where producer steps record what they made, keyed by their own step id. Resolving
# against these lets config name an artifact whose id is minted mid-run, without this
# step knowing which kind of step produced it. A step writes exactly one of these, so
# the order arbitrates a malformed context rather than ranking real producers.
_STEP_ARTIFACT_SOURCES: tuple[tuple[str, Callable[[dict], Optional[dict]]], ...] = (
    ("collected_artifacts", lambda entry: entry.get("file_artifact_universe")),
    ("env_snapshotted_universes", lambda entry: entry),
)

# A `urls` entry: a bare URL, saved under the name its path ends in, or
# {"url": ..., "filename": ...}, saved under that filename verbatim.
UrlEntry = str | dict[str, str]
_URL_ENTRY_KEYS = frozenset({"url", "filename"})


def _named_download(step_id: str, entry: dict) -> tuple[str, str]:
    """The ``(url, filename)`` of an object entry, rejecting anything but a plain file name for ``filename``."""
    unknown = sorted(set(entry) - _URL_ENTRY_KEYS)
    if unknown:
        raise ValueError(
            f"load_artifact '{step_id}': urls entry {entry!r} has unknown key(s) {unknown}; "
            f"an object entry takes only 'url' and 'filename'"
        )
    url, filename = entry.get("url"), entry.get("filename")
    if not isinstance(url, str) or not url:
        raise ValueError(f"load_artifact '{step_id}': urls entry {entry!r} needs a non-empty string 'url'")
    if not isinstance(filename, str) or not filename:
        raise ValueError(
            f"load_artifact '{step_id}': urls entry {entry!r} needs a non-empty string 'filename' "
            f"(a bare URL string is saved under the name its path ends in)"
        )
    try:
        validate_relative_filename(filename)
    except ValueError as e:
        raise ValueError(f"load_artifact '{step_id}': urls entry {entry!r}: {e}") from e
    if "/" in filename or "\\" in filename or filename == ".":
        raise ValueError(
            f"load_artifact '{step_id}': urls entry {entry!r}: filename must be a plain file name, not a path"
        )
    return url, filename


def _url_downloads(step_id: str, urls: list[UrlEntry]) -> list[tuple[str, str]]:
    """Each ``urls`` entry as ``(url, filename)``, the name it is saved under in the destination.

    A bare URL takes its last path segment, a repeat suffixed ``-1``, ``-2``, ...; an explicit
    filename is never renamed, so one given twice, or matching a bare URL's name, raises.
    """
    explicit: dict[str, str] = {}
    for entry in urls:
        if isinstance(entry, dict):
            url, filename = _named_download(step_id, entry)
            if filename in explicit:
                raise ValueError(
                    f"load_artifact '{step_id}': filename {filename!r} is given to more than one urls entry "
                    f"({explicit[filename]!r} and {url!r})"
                )
            explicit[filename] = url
        elif not isinstance(entry, str):
            raise ValueError(
                f"load_artifact '{step_id}': urls entry {entry!r} must be a URL string or an object "
                f"with 'url' and 'filename'"
            )

    downloads: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for entry in urls:
        if isinstance(entry, dict):
            downloads.append((entry["url"], entry["filename"]))
            continue
        base = unquote(os.path.basename(urlparse(entry).path)) or "downloaded"
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count == 0:
            filename = base
        else:
            stem, ext = os.path.splitext(base)
            filename = f"{stem}-{count}{ext}"
        if filename in explicit:
            raise ValueError(
                f"load_artifact '{step_id}': {entry!r} would be saved as {filename!r}, the filename given to "
                f"{explicit[filename]!r}; give it a 'filename' of its own"
            )
        downloads.append((entry, filename))
    return downloads


@dataclass
class _LoadInputs:
    """What a load_artifact step loads and where, after per-run overrides."""

    artifacts: list[dict]
    downloads: list[tuple[str, str]]
    destination_path: Optional[str]


class LoadArtifactTaskStep(TaskStep):
    type: ClassVar[str] = "load_artifact"
    entity_refs = (
        EntityRef.env("env_id"),
        EntityRef.artifact("artifact_id", version_field="artifact_version"),
        EntityRef.artifact("artifacts[].id", version_field="version"),
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: Optional[str] = None,
        env_step_id: Optional[str] = None,
        artifacts: Optional[list[dict]] = None,
        artifact_id: Optional[str] = None,
        artifact_version: Optional[int] = None,
        urls: Optional[list[UrlEntry]] = None,
        collected_artifacts_step_id: Optional[str] = None,
        artifact_from_step_id: Optional[str] = None,
        destination_path: Optional[str] = None,
        agent_name: Optional[str] = None,
        sandbox_name: Optional[str] = None,
        container_name: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        snapshot_after_load: Optional[bool] = None,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if artifacts is not None and artifact_id is not None:
            raise ValueError("Pass either `artifacts=[...]` or `artifact_id=...`, not both")
        if artifacts is None and artifact_id is not None:
            artifacts = [{"id": artifact_id, "version": artifact_version}]
        if collected_artifacts_step_id and (artifacts or urls):
            raise ValueError(
                "`collected_artifacts_step_id` is mutually exclusive with `artifacts`/`artifact_id`/`urls`"
            )
        if artifact_from_step_id and (artifacts or urls or collected_artifacts_step_id):
            raise ValueError(
                "`artifact_from_step_id` is mutually exclusive with "
                "`artifacts`/`artifact_id`/`urls`/`collected_artifacts_step_id`"
            )
        if not (artifacts or urls or collected_artifacts_step_id or artifact_from_step_id):
            raise ValueError(
                "Must pass at least one of `artifacts=[...]`, `artifact_id=...`, `urls=[...]`, "
                "`artifact_from_step_id=...`, or `collected_artifacts_step_id=...`"
            )
        if agent_name is not None and container_name is not None:
            raise ValueError("Pass either `agent_name` or `container_name`, not both")
        if env_id is None and agent_name is None and container_name is None and sandbox_name is None:
            raise ValueError(
                "Must pass at least one of `env_id`, `agent_name`, `sandbox_name`, or "
                "`container_name`+`sandbox_name`"
            )
        if container_name is not None and sandbox_name is None:
            raise ValueError("`container_name` requires `sandbox_name`")
        if urls and container_name is None and agent_name is None and sandbox_name is None:
            raise ValueError(
                "`urls` requires `agent_name`, `sandbox_name`, or `container_name`+`sandbox_name`"
            )
        if collected_artifacts_step_id and agent_name is None and container_name is None and sandbox_name is None:
            raise ValueError(
                "`collected_artifacts_step_id` requires `agent_name`, `sandbox_name`, or "
                "`container_name`+`sandbox_name`"
            )
        self.env_id = env_id
        self.env_step_id = env_step_id
        self.artifacts = artifacts or []
        self.urls = list(urls or [])
        # Stored urls fail here, at construction; a per-run `urls` override is checked in `_resolve_inputs`.
        _url_downloads(self.id, self.urls)
        # Superseded by the producer-agnostic `artifact_from_step_id`, but kept
        # readable/writable: it is persisted on existing task-step documents.
        self.collected_artifacts_step_id = collected_artifacts_step_id
        self.artifact_from_step_id = artifact_from_step_id
        self.destination_path = destination_path
        self.agent_name = agent_name
        self.sandbox_name = sandbox_name
        self.container_name = container_name
        # None = defer to the AGENT_ENV_SNAPSHOT_AFTER_LOAD flag on the worker, so a task
        # doc doesn't hard-code a cost decision an operator may want to flip.
        self.snapshot_after_load = snapshot_after_load

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["env_step_id"] = self.env_step_id
        base["artifacts"] = self.artifacts
        base["urls"] = self.urls
        base["collected_artifacts_step_id"] = self.collected_artifacts_step_id
        base["artifact_from_step_id"] = self.artifact_from_step_id
        base["destination_path"] = self.destination_path
        base["agent_name"] = self.agent_name
        base["sandbox_name"] = self.sandbox_name
        base["container_name"] = self.container_name
        base["snapshot_after_load"] = self.snapshot_after_load
        return base

    @classmethod
    def from_dict(cls, data: dict) -> LoadArtifactTaskStep:
        if "artifacts" in data:
            artifacts = data["artifacts"]
        elif "artifact_id" in data:
            artifacts = [{"id": data["artifact_id"], "version": data.get("artifact_version")}]
        else:
            artifacts = None
        return cls(
            **cls._base_from_dict(data),
            env_id=data.get("env_id"),
            env_step_id=data.get("env_step_id"),
            artifacts=artifacts,
            urls=data.get("urls"),
            collected_artifacts_step_id=data.get("collected_artifacts_step_id"),
            artifact_from_step_id=data.get("artifact_from_step_id"),
            destination_path=data.get("destination_path"),
            agent_name=data.get("agent_name"),
            sandbox_name=data.get("sandbox_name"),
            container_name=data.get("container_name"),
            snapshot_after_load=data.get("snapshot_after_load"),
        )

    @staticmethod
    def _record_universe_load(context: TaskStepContext, artifact, load_result) -> None:
        """Record which load path ran, so a slow run is explicable after the fact.

        "This took 40 minutes" and "this took 90 seconds" are the same step with the same
        inputs; the only difference is whether a snapshot was hit. Without this the
        distinction exists solely in worker logs, which are gone by the time anyone asks.
        """
        if load_result is None:
            return
        entry = {
            "id": artifact.id,
            "version": artifact.version,
            "restored_from_snapshot": getattr(load_result, "restored_from_snapshot", False),
        }
        for key in ("snapshot_db_image_artifact_id", "snapshot_baked", "snapshot_bake_error"):
            value = getattr(load_result, key, None)
            if value is not None:
                entry[key] = value
        context.metadata.setdefault("loaded_environment_universes", []).append(entry)

    @staticmethod
    def _resolve_step_artifact(context: TaskStepContext, step_id: str) -> Optional[dict]:
        """The ``{id, version}`` a prior step recorded under ``step_id``.

        ``None`` when that step ran but produced nothing; raises when nothing was
        recorded at all, since a wrong id or a producer that never ran is
        misconfiguration rather than an empty result.
        """
        for key, extract in _STEP_ARTIFACT_SOURCES:
            entry = (context.metadata.get(key) or {}).get(step_id)
            if entry is None:
                continue
            ref = extract(entry)
            return {"id": ref["id"], "version": ref.get("version")} if ref else None
        raise RuntimeError(
            f"No artifact recorded by step '{step_id}'. Ensure a step that produces "
            f"one (e.g. `collect_artifacts`, `snapshot_env`) ran with that id before "
            f"this load step."
        )

    @staticmethod
    def _deploy_step_of(deployed_env) -> Optional[str]:
        return (deployed_env.metadata or {}).get("deploy_step_id")

    def _deployed_env(self, context: TaskStepContext):
        """The deployment of `env_id` this step loads into, or None if it isn't deployed."""
        matches = [d for d in context.deployed_envs if d.env_id == self.env_id]
        if self.env_step_id is not None:
            # A filter, not a hint: an unmatched id is an error, never a substitute.
            named = [d for d in matches if self._deploy_step_of(d) == self.env_step_id]
            if matches and not named:
                raise RuntimeError(
                    f"load_artifact '{self.id}': env_step_id '{self.env_step_id}' names no deployment of "
                    f"env '{self.env_id}' (deployed by: {[self._deploy_step_of(d) for d in matches]})"
                )
            matches = named
        if len(matches) > 1:
            raise RuntimeError(
                f"load_artifact '{self.id}': {len(matches)} deployments of env '{self.env_id}' "
                f"(deployed by: {[self._deploy_step_of(d) for d in matches]}). Set env_step_id "
                "to the deploy_env step this load should target."
            )
        return matches[0] if matches else None

    def _resolve_inputs(self, context: TaskStepContext) -> _LoadInputs:
        """Resolve ``(artifacts, downloads, destination_path)``, honoring per-run
        ``step_overrides``. An override wins over the stored value; an explicit
        artifact override also wins over ``artifact_from_step_id`` /
        ``collected_artifacts_step_id`` wiring. A ``urls`` override is named and
        validated here, before anything is loaded.
        Topology (env_id/agent_name/container_name) is not overridable here.
        """
        overrides = self.step_param_overrides(context)

        if "artifacts" in overrides:
            resolved_artifacts = list(overrides.get("artifacts") or [])
        elif "artifact_id" in overrides:
            if not overrides["artifact_id"]:
                raise ValueError(
                    f"load '{self.id}': step_overrides 'artifact_id' must be a non-empty string, "
                    f"got {overrides['artifact_id']!r}"
                )
            resolved_artifacts = [
                {"id": overrides["artifact_id"], "version": overrides.get("artifact_version")}
            ]
        else:
            resolved_artifacts = list(self.artifacts)
            step_id = self.artifact_from_step_id or self.collected_artifacts_step_id
            if step_id:
                ref = self._resolve_step_artifact(context, step_id)
                # Produced nothing → skip the load and let a downstream verifier
                # score the empty result, rather than failing the task here.
                if ref is None:
                    logger.warning("step %r produced no artifacts — skipping load", step_id)
                    resolved_artifacts = []
                else:
                    resolved_artifacts = [ref]

        # `or []` so a `{"urls": null}` override cleanly clears rather than crashing on list(None).
        urls = list(overrides["urls"] or []) if "urls" in overrides else list(self.urls)
        downloads = _url_downloads(self.id, urls)
        destination_path = overrides.get("destination_path", self.destination_path)

        if overrides:
            logger.info("load '%s': applying step_overrides for %s", self.id, sorted(overrides))
        return _LoadInputs(artifacts=resolved_artifacts, downloads=downloads, destination_path=destination_path)

    async def _load_url_onto_vm(self, sandbox, url: str, destination_path: str) -> None:
        """Host counterpart of ``sandbox.load_object_file``; ``write_file_from_url`` targets a container instead."""
        from agent_env.providers.sandbox_providers.sandbox import CURL_RETRY_FLAGS

        parent = posixpath.dirname(destination_path)
        if parent:
            await sandbox.exec_script(f"mkdir -p {shlex.quote(parent)}")
        await sandbox.exec_script(
            f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(url)} -o {shlex.quote(destination_path)}"
        )

    async def _load_url_into_container(self, sandbox, container: str, url: str, destination_path: str) -> None:
        """Download ``url`` onto the VM host, then copy it into ``container`` at ``destination_path``:
        ``write_file_from_url`` reaches only the sandbox's own agent container."""
        vm_temp = f"/tmp/_load_url_{uuid.uuid4().hex[:8]}"
        try:
            await self._load_url_onto_vm(sandbox, url, vm_temp)
            await sandbox.exec_script(
                f"docker exec -u 0 {shlex.quote(container)} mkdir -p {shlex.quote(posixpath.dirname(destination_path))}"
            )
            await sandbox.docker_cp(vm_temp, f"{container}:{destination_path}", remove_source=True)
        except Exception:
            with contextlib.suppress(Exception):  # best effort: the load's own error is the one to report
                await sandbox.exec_script(f"rm -f {shlex.quote(vm_temp)}")
            raise

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent
        from agent_env.a2a_agent.store import get_a2a_agent_instance_store
        from agent_env.artifact import (
            Artifact,
            CliArtifact,
            FileArtifact,
            FileArtifactUniverse,
            EnvironmentArtifact,
            EnvironmentUniverseArtifact,
        )
        from agent_env.env.env import Env, require_gateway_url
        from agent_env.providers.sandbox_providers.sandbox import VmSandbox
        from agent_env.providers.sandbox_providers.sandbox_provider import (
            SANDBOX_MODE_VM,
            build_sandbox_provider,
            get_agent_sandbox_provider,
            get_sandbox_provider,
        )

        deployed = None
        if self.env_id is not None:
            deployed = self._deployed_env(context)
            if deployed is None:
                raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")

        inputs = self._resolve_inputs(context)
        resolved_artifacts = inputs.artifacts
        downloads = inputs.downloads
        destination_path = inputs.destination_path

        # With container_name it hosts that container; alone it is the target itself.
        target_sandbox = None
        if self.sandbox_name is not None:
            if self.container_name is not None:
                containers = context.metadata.get("deployed_docker_containers", [])
                container_entry = next(
                    (
                        c for c in containers
                        if c.get("container_name") == self.container_name
                        and c.get("sandbox_name") == self.sandbox_name
                    ),
                    None,
                )
                if container_entry is None:
                    raise RuntimeError(
                        f"Container '{self.container_name}' not found on sandbox '{self.sandbox_name}' in "
                        f"context.metadata['deployed_docker_containers']"
                    )
            ds = next((s for s in context.deployed_sandboxes if s.sandbox_name == self.sandbox_name), None)
            if ds is None:
                raise RuntimeError(f"Sandbox '{self.sandbox_name}' not found in context.deployed_sandboxes")
            provider = (
                build_sandbox_provider(ds.sandbox_type) if ds.sandbox_type else get_sandbox_provider()
            )
            target_sandbox = await provider.get_sandbox(ds.sandbox_id)
        # A VM-mode sandbox with no container_name is loaded onto its host; a container-mode one, into its container.
        onto_vm_host = (
            self.sandbox_name is not None and self.container_name is None and ds.sandbox_mode == SANDBOX_MODE_VM
        )

        env = None
        for ref in resolved_artifacts:
            artifact = Artifact.get(ref["id"], ref.get("version"))
            # Split on the target, not the type: a service universe aimed at an
            # agent/container is a grading load (stage the frozen exports as files),
            # aimed at an env it still means restore into the live MCP services.
            stage_as_files = isinstance(artifact, (FileArtifactUniverse, FileArtifact)) or (
                isinstance(artifact, EnvironmentUniverseArtifact)
                and deployed is None
                and (
                    self.agent_name is not None
                    or self.container_name is not None
                    or self.sandbox_name is not None
                )
            )
            if stage_as_files:
                if self.sandbox_name is not None:
                    destination = (destination_path or "/loaded").rstrip("/") or "/"
                    if self.container_name is not None:
                        files = await _load_universe_into_container(
                            target_sandbox, target_sandbox.scoped_name(self.container_name), artifact, destination,
                        )
                    elif onto_vm_host:
                        files = await _load_universe_onto_vm(target_sandbox, artifact, destination)
                    else:
                        files = await _load_universe_into_sandbox_container(target_sandbox, artifact, destination)
                    context.metadata.setdefault("loaded_file_artifact_universes", []).append({
                        "id": artifact.id,
                        "version": artifact.version,
                        "artifact_type": artifact.type,
                        "env_id": None,
                        "agent_name": None,
                        "sandbox_name": self.sandbox_name,
                        "container_name": self.container_name,
                        "destination_path": destination,
                        "files": files,
                    })
                    continue
                if (self.env_id is None) == (self.agent_name is None):
                    raise RuntimeError(
                        f"{artifact.type} '{artifact.id}' requires exactly one of `env_id` or `agent_name`"
                    )
                if self.env_id is not None:
                    # Only a FileArtifactUniverse or a FileArtifact reaches here — `stage_as_files`
                    # excludes a service universe whenever an env is targeted.
                    if env is None:
                        env_obj = Env.get(deployed.env_id, deployed.env_version)
                        env = await type(env_obj).from_deployed_env(deployed)
                    result = await env.load_file_artifact_universe(artifact, destination_path=destination_path)
                    context.metadata.setdefault("loaded_file_artifact_universes", []).append({
                        "id": artifact.id,
                        "version": artifact.version,
                        "artifact_type": artifact.type,
                        "env_id": deployed.env_id,
                        "agent_name": None,
                        "destination_path": result.destination_path,
                        "files": result.files,
                    })
                else:
                    agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
                    if agent is None or not agent.instance_id:
                        raise RuntimeError(f"Agent '{self.agent_name}' not found in context.deployed_agents (or missing instance_id)")
                    deployed_agent = get_a2a_agent_instance_store().get(agent.instance_id)
                    destination = (destination_path or "/tmp/file_artifacts").rstrip("/") or "/"
                    files = await A2AAgent.load_file_artifact_universe(deployed_agent, artifact, destination)
                    context.metadata.setdefault("loaded_file_artifact_universes", []).append({
                        "id": artifact.id,
                        "version": artifact.version,
                        "artifact_type": artifact.type,
                        "env_id": None,
                        "agent_name": self.agent_name,
                        "destination_path": destination,
                        "files": files,
                    })
            elif isinstance(artifact, (EnvironmentArtifact, EnvironmentUniverseArtifact)):
                if (
                    deployed is None
                    and isinstance(artifact, EnvironmentArtifact)
                    and (
                        self.agent_name is not None
                        or self.container_name is not None
                        or self.sandbox_name is not None
                    )
                ):
                    # Stage the service payload (filesystem-style data.json with
                    # `files[]`, or a `data.json` + `root/` doc-carrier zip) as a
                    # plain file tree inside the agent/container — no MCP server
                    # involved.
                    if self.sandbox_name is not None:
                        sandbox = target_sandbox
                        # None targets the VM host itself, or a container sandbox directly.
                        container = sandbox.scoped_name(self.container_name) if self.container_name else None
                        if container is None and not onto_vm_host and isinstance(sandbox, VmSandbox):
                            container = sandbox.container_name
                    else:
                        agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
                        if agent is None or not agent.sandbox_id:
                            raise RuntimeError(
                                f"Agent '{self.agent_name}' not found in context.deployed_agents (or missing sandbox_id)"
                            )
                        provider = (
                            build_sandbox_provider(agent.sandbox_type)
                            if agent.sandbox_type
                            else get_agent_sandbox_provider()
                        )
                        sandbox = await provider.get_sandbox(agent.sandbox_id)
                        container = sandbox.container_name if isinstance(sandbox, VmSandbox) else None
                    destination = (destination_path or "/app/files").rstrip("/") or "/"
                    files = await _stage_environment_payload_into_container(
                        sandbox, container, artifact, destination,
                    )
                    context.metadata.setdefault("loaded_environment_artifacts", []).append({
                        "id": artifact.id,
                        "version": artifact.version,
                        "service_name": artifact.environment_name,
                        "environment_name": artifact.environment_name,
                        "agent_name": self.agent_name,
                        "sandbox_name": self.sandbox_name,
                        "container_name": self.container_name,
                        "destination_path": destination,
                        "file_count": len(files),
                        # Cap the recorded list — TaskStepContext is persisted and
                        # can be shipped to an external runner; a huge corpus must
                        # not bloat the payload.
                        "files": files[:_ENVIRONMENT_PAYLOAD_MAX_RECORDED_FILES],
                    })
                    continue
                if deployed is None:
                    raise RuntimeError(f"`env_id` is required when loading a {artifact.type}")
                if env is None:
                    env_obj = Env.get(deployed.env_id, deployed.env_version)
                    env = await type(env_obj).from_deployed_env(deployed)
                if isinstance(artifact, EnvironmentArtifact):
                    await env.load_environment_artifact(artifact)
                else:
                    from agent_env.env.envs.multi_env import MultiEnv

                    # Only MultiEnv has a servicedb to snapshot; the other env types
                    # don't accept the kwarg, so don't invent one for them.
                    if isinstance(env, MultiEnv):
                        load_result = await env.load_environment_universe_artifact(
                            artifact, snapshot_after_load=self.snapshot_after_load
                        )
                    else:
                        load_result = await env.load_environment_universe_artifact(artifact)
                    self._record_universe_load(context, artifact, load_result)
            elif isinstance(artifact, CliArtifact):
                if not self.agent_name:
                    raise RuntimeError("agent_name is required when loading a CliArtifact")
                if deployed is None:
                    raise RuntimeError("env_id is required when loading a CliArtifact (for gateway_url)")
                agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
                if agent is None or not agent.instance_id:
                    raise RuntimeError(f"Agent '{self.agent_name}' not found in context.deployed_agents (or missing instance_id)")
                deployed_agent = get_a2a_agent_instance_store().get(agent.instance_id)
                gateway_url = reachable_url(require_gateway_url(deployed, "Installing a CliArtifact"),
                                            from_sandbox_type=deployed.sandbox_type, to_sandbox_type=agent.sandbox_type)
                install_path = await A2AAgent.install_cli(deployed_agent, artifact, gateway_url)
                agent_clis = context.metadata.setdefault("installed_clis", {}).setdefault(self.agent_name, {})
                agent_clis[artifact.id] = {
                    "install_path": install_path,
                    "command_name": artifact.command_name,
                }
            else:
                raise ValueError(f"Unsupported artifact type for loading: '{artifact.type}'")

        if downloads:
            if self.sandbox_name is not None:
                sandbox = target_sandbox
            else:
                agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
                if agent is None or not agent.sandbox_id:
                    raise RuntimeError(f"Agent '{self.agent_name}' not found in context.deployed_agents (or missing sandbox_id)")
                provider = (
                    build_sandbox_provider(agent.sandbox_type)
                    if agent.sandbox_type
                    else get_agent_sandbox_provider()
                )
                sandbox = await provider.get_sandbox(agent.sandbox_id)
            destination = (destination_path or "/tmp/file_artifacts").rstrip("/") or "/"
            semaphore = asyncio.Semaphore(8)

            if onto_vm_host:
                target_desc = f"VM sandbox '{self.sandbox_name}'"
            elif self.container_name is not None:
                target_desc = f"container '{self.container_name}'"
            else:
                target_desc = "agent"

            async def _load_one(url: str, filename: str) -> None:
                async with semaphore:
                    dest = f"{destination}/{filename}"
                    if onto_vm_host:
                        await self._load_url_onto_vm(sandbox, url, dest)
                    elif self.container_name is not None:
                        await self._load_url_into_container(sandbox, sandbox.scoped_name(self.container_name), url, dest)
                    else:
                        await sandbox.write_file_from_url(url, dest)
                    logger.info(f"Loaded URL into {target_desc}: {url} -> {dest}")

            await asyncio.gather(*(_load_one(u, f) for u, f in downloads))
            context.metadata.setdefault("loaded_urls", []).append({
                "step_id": self.id,
                "agent_name": None if self.sandbox_name is not None else self.agent_name,
                "sandbox_name": self.sandbox_name,
                "destination_path": destination,
                "files": [filename for _, filename in downloads],
            })

        return context


# Cap on how many staged paths get recorded in TaskStepContext metadata
# (persisted, and shipped to external runners) — `file_count` always carries
# the real total.
_ENVIRONMENT_PAYLOAD_MAX_RECORDED_FILES = 200

# Expands a service payload into a plain file tree using only the Python
# stdlib, so it can run on the sandbox VM or inside the agent container
# without any agent-env install. Mirrors the synthetic filesystem server's
# load contract (BaseService._reset_from_zip_bundle / load_from_json):
#   • zip bundle: `root/` members are real files (streamed out first, zip-slip
#     guarded); an optional top-level `data.json` provides `files[]` entries
#     that are skipped when the path already exists (root/ takes precedence).
#   • bare data.json: `files[]` entries with `path`, `content`, and optional
#     `encoding: "base64"` for binary content.
# argv: [<payload path>, <destination dir>]; prints one staged relative path
# per line.
_ENVIRONMENT_PAYLOAD_EXPAND_SCRIPT = """\
import base64, json, os, shutil, sys, zipfile

payload, dest = sys.argv[1], sys.argv[2]
staged = []


def _target(rel):
    rel = rel.replace("\\\\", "/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        raise SystemExit("unsafe or empty path in payload: %r" % rel)
    path = os.path.join(dest, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path, "/".join(parts)


os.makedirs(dest, exist_ok=True)
entries = []
if zipfile.is_zipfile(payload):
    with zipfile.ZipFile(payload) as zf:
        data_json = None
        # root/ members land first so data.json entries defer to real files —
        # same precedence as the filesystem server's zip-bundle reset.
        for info in zf.infolist():
            if info.is_dir():
                continue
            if info.filename == "data.json":
                data_json = info
                continue
            if info.filename.startswith("root/") and info.filename[len("root/"):]:
                path, rel = _target(info.filename[len("root/"):])
                with zf.open(info) as src, open(path, "wb") as out:
                    shutil.copyfileobj(src, out)
                staged.append(rel)
        if data_json is not None:
            with zf.open(data_json) as f:
                entries = json.load(f).get("files", []) or []
else:
    with open(payload, "r", encoding="utf-8") as f:
        entries = json.load(f).get("files", []) or []

for entry in entries:
    rel = entry.get("path")
    if not rel:
        continue
    path, rel_norm = _target(rel)
    if os.path.exists(path):
        continue
    content = entry.get("content", "")
    if entry.get("encoding") == "base64":
        with open(path, "wb") as f:
            f.write(base64.b64decode(content))
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    staged.append(rel_norm)

for rel in staged:
    print(rel)
"""


async def _stage_environment_payload_into_container(
    sandbox, container_name: Optional[str], environment_artifact, destination: str
) -> list[str]:
    """Expand an EnvironmentArtifact's payload into a file tree at ``destination`` — in the container, or on the VM host when ``container_name`` is None."""
    from agent_env.providers.sandbox_providers.sandbox import VmSandbox

    file_artifact = environment_artifact.get_file_artifact()
    token = uuid.uuid4().hex[:8]
    safe_name = file_artifact.filename.replace("/", "_")

    if isinstance(sandbox, VmSandbox):
        vm_payload = f"/tmp/_svc_{token}_{safe_name}"
        vm_stage = f"/tmp/_svc_stage_{token}"
        await sandbox.load_object_file(file_artifact.object_url, vm_payload)
        try:
            # exec_script raises on a non-zero exit, but only with the raw
            # script output — re-raise with the artifact context so a failed
            # expansion (corrupt payload, zip-slip refusal) is diagnosable,
            # matching the container-mode branch below.
            try:
                out = await sandbox.exec_script(
                    f"python3 - {shlex.quote(vm_payload)} {shlex.quote(vm_stage)} <<'AGENTENV_EXPAND_PY'\n"
                    f"{_ENVIRONMENT_PAYLOAD_EXPAND_SCRIPT}\nAGENTENV_EXPAND_PY"
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Service payload expansion failed for "
                    f"'{environment_artifact.id}' v{environment_artifact.version}: {e}"
                ) from e
            if container_name is None:
                # Targeting the VM host: the tree is already on it, so it only moves.
                await sandbox.exec_script(
                    f"mkdir -p {shlex.quote(destination)} && "
                    f"cp -a {shlex.quote(vm_stage)}/. {shlex.quote(destination)}/"
                )
            else:
                await sandbox.exec_script(
                    f"docker exec -u 0 {shlex.quote(container_name)} mkdir -p {shlex.quote(destination)}"
                )
                await sandbox.docker_cp(f"{vm_stage}/.", f"{container_name}:{destination}")
        finally:
            try:
                await sandbox.exec_script(f"rm -rf {shlex.quote(vm_payload)} {shlex.quote(vm_stage)}")
            except Exception as e:
                logger.warning(f"Best-effort cleanup of service payload staging failed (ignored): {e}")
    else:
        # Container-mode sandbox: the sandbox IS the agent container — expand
        # in place at the destination.
        payload_tmp = f"/tmp/_svc_{token}_{safe_name}"
        await sandbox.write_file_from_object(file_artifact.object_url, payload_tmp)
        try:
            exit_code, out, err = await sandbox.exec_with_output(
                "python3", "-c", _ENVIRONMENT_PAYLOAD_EXPAND_SCRIPT, payload_tmp, destination
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"Service payload expansion failed (exit {exit_code}) for "
                    f"'{environment_artifact.id}' v{environment_artifact.version}: {err[-1500:]}"
                )
        finally:
            try:
                await sandbox.exec("rm", "-rf", payload_tmp)
            except Exception as e:
                logger.warning(f"Best-effort cleanup of service payload staging failed (ignored): {e}")

    files = [line.strip() for line in out.splitlines() if line.strip()]
    if not files:
        logger.warning(
            f"EnvironmentArtifact '{environment_artifact.id}' v{environment_artifact.version} "
            f"({file_artifact.filename}) produced no files — payload has no root/ tree "
            f"and no files[] entries; is this a filesystem-style component?"
        )
    else:
        logger.info(
            f"Staged {len(files)} file(s) from EnvironmentArtifact '{environment_artifact.id}' "
            f"v{environment_artifact.version} into "
            f"{'the VM host' if container_name is None else repr(container_name)} at {destination}"
        )
    return files


async def _load_universe_onto_vm(sandbox, universe, destination: str) -> list[str]:
    """Pull each file in `universe` from the object store straight onto the VM host — no temp file, no copy inward."""
    file_artifacts = universe.get_file_artifacts()
    if not file_artifacts:
        logger.warning(
            f"FileArtifactUniverse '{universe.id}' v{universe.version} has no files; nothing to load"
        )
        return []

    await sandbox.exec_script(f"mkdir -p {shlex.quote(destination)}")
    loaded: list[str] = []
    total = len(file_artifacts)
    logger.info(
        f"Loading FileArtifactUniverse '{universe.id}' v{universe.version} ({total} file(s)) "
        f"onto the VM host at {destination}"
    )
    for idx, (filename, fa) in enumerate(file_artifacts.items(), 1):
        validate_relative_filename(filename)
        dest_path = posixpath.join(destination, filename)
        parent = posixpath.dirname(dest_path)
        logger.info(f"  [{idx}/{total}] {fa.object_url} -> {dest_path}")
        if parent and parent != destination:
            await sandbox.exec_script(f"mkdir -p {shlex.quote(parent)}")
        await sandbox.load_object_file(fa.object_url, dest_path)
        loaded.append(filename)
    return loaded


async def _load_universe_into_sandbox_container(sandbox, universe, destination: str) -> list[str]:
    """Stage each file in `universe` into a container-mode sandbox, as an agent's container is loaded."""
    from agent_env.providers.sandbox_providers.sandbox import stage_files_into_container

    file_artifacts = universe.get_file_artifacts()
    if not file_artifacts:
        logger.warning(
            f"FileArtifactUniverse '{universe.id}' v{universe.version} has no files; nothing to load"
        )
        return []
    logger.info(
        f"Loading FileArtifactUniverse '{universe.id}' v{universe.version} ({len(file_artifacts)} file(s)) "
        f"into the container sandbox at {destination}"
    )
    return list(await stage_files_into_container(sandbox, file_artifacts, destination))


async def _load_universe_into_container(sandbox, container_name: str, universe, destination: str) -> list[str]:
    """For each file in `universe`: pull from the object store to a VM temp path, then `docker cp` into `container_name` at `destination/<rel_path>`."""
    file_artifacts = universe.get_file_artifacts()
    if not file_artifacts:
        logger.warning(
            f"FileArtifactUniverse '{universe.id}' v{universe.version} has no files; nothing to load"
        )
        return []

    await sandbox.exec_script(
        f"docker exec {shlex.quote(container_name)} mkdir -p {shlex.quote(destination)}"
    )
    loaded: list[str] = []
    total = len(file_artifacts)
    logger.info(
        f"Loading FileArtifactUniverse '{universe.id}' v{universe.version} ({total} file(s)) "
        f"into container '{container_name}' at {destination}"
    )
    for idx, (filename, fa) in enumerate(file_artifacts.items(), 1):
        validate_relative_filename(filename)
        dest_path = posixpath.join(destination, filename)
        parent = posixpath.dirname(dest_path)
        vm_temp = f"/tmp/_load_{uuid.uuid4().hex[:8]}_{filename.replace('/', '_')}"
        logger.info(f"  [{idx}/{total}] {fa.object_url} -> {container_name}:{dest_path}")
        await sandbox.load_object_file(fa.object_url, vm_temp)
        if parent and parent != destination:
            await sandbox.exec_script(
                f"docker exec {shlex.quote(container_name)} mkdir -p {shlex.quote(parent)}"
            )
        await sandbox.docker_cp(vm_temp, f"{container_name}:{dest_path}", remove_source=True)
        loaded.append(filename)
    return loaded
