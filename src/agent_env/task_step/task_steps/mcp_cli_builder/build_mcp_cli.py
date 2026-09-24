"""Build a CLI from a deployed server's manifest and live MCP tools."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Optional

import httpx

from agentenv_protocol import GET_INTERFACES_EXTENSION_URI, INTERFACE_MANIFEST_PATH
from agentenv_protocol import client as protocol_v1

from agent_env.artifact import CliArtifact
from agent_env.env import Env
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef, RefRole
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.mcp_cli_builder.codegen import generate_cli_script

logger = logging.getLogger(__name__)


async def _resolve_manifest_endpoint(gateway_url: str, environment_name: str) -> str | None:
    """The server's advertised manifest index path, gateway-rewritten and reachable as-is.

    Returns None on every failure rather than raising: until the fleet is rebuilt,
    having nothing to advertise is the normal case, so these are debug-level.
    """
    try:
        card = await protocol_v1.get_card(gateway_url)
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("env card for %r unavailable (%s); constructing the manifest path", environment_name, e)
        return None

    child = protocol_v1.find_child(card, environment_name)
    if child is None:
        logger.debug("env card has no child named %r; constructing the manifest path", environment_name)
        return None
    return protocol_v1.extension_params(child, GET_INTERFACES_EXTENSION_URI).get("endpoint")


async def fetch_interface_manifest(
    gateway_url: str,
    environment_name: str,
) -> dict[str, Any] | None:
    """Fetch a deployed server's CLI manifest, or None when none is usable.

    The index path comes off the card when the server advertises it, and is
    constructed the old way when it does not.

    Contract: *degrade the unreachable, guard the misconfigured*. A missing,
    unreachable, or non-manifest response (404, other non-2xx incl. a transient
    502, transport error, non-JSON, or JSON without a ``service`` field) returns
    None so the caller still builds the legacy flat CLI — building a CLI must not
    depend on a manifest being served. Only a server that actually served a
    manifest object for the wrong ``service`` raises: that is an opted-in server
    that is misconfigured, a real bug worth failing on.
    """
    endpoint = await _resolve_manifest_endpoint(gateway_url, environment_name)
    if not endpoint:
        # Fallback until every image advertises the extension; the server owns
        # this path and the gateway owns the /svc/mcp-{name} prefix, not the renderer.
        endpoint = f"/svc/mcp-{environment_name}{INTERFACE_MANIFEST_PATH}"
    try:
        manifest = await protocol_v1.get_interface_manifest(
            gateway_url, "cli", endpoint=endpoint
        )
    except httpx.HTTPStatusError as e:
        # 404 = opted out; any other error status (405, 5xx, a transient 502
        # while the server is still booting) is treated the same: no usable
        # manifest, fall back to the flat CLI rather than fail the build.
        if e.response.status_code != 404:
            logger.warning(
                "Interface Manifest endpoint for %r returned HTTP %s; building flat CLI",
                environment_name, e.response.status_code,
            )
        return None
    except httpx.HTTPError as e:
        logger.warning(
            "Interface Manifest fetch for %r failed (%s); building flat CLI",
            environment_name, e,
        )
        return None
    except ValueError:  # includes json.JSONDecodeError
        logger.warning(
            "Interface Manifest for %r was not JSON; building flat CLI", environment_name
        )
        return None
    if not isinstance(manifest, dict):
        logger.warning(
            "Interface Manifest for %r was not a JSON object; building flat CLI",
            environment_name,
        )
        return None

    served_service = manifest.get("service")
    if served_service is None:
        # Not actually a manifest (e.g. an unrelated 200 body); degrade quietly.
        logger.warning(
            "Interface Manifest endpoint for %r returned JSON without a 'service' "
            "field; building flat CLI", environment_name,
        )
        return None
    if served_service != environment_name:
        raise ValueError(
            f"Interface Manifest service {served_service!r} does not "
            f"match deployed service {environment_name!r}"
        )
    # manifest_version is checked in codegen, alongside the interface and shape.
    return manifest


class BuildMcpCliTaskStep(TaskStep):
    type: ClassVar[str] = "build_mcp_cli"
    entity_refs = (
        EntityRef.env("env_id"),
        EntityRef.artifact("cli_artifact_id", role=RefRole.OUTPUT, artifact_type="cli"),
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        command_name: str,
        cli_artifact_id: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.command_name = command_name
        self.cli_artifact_id = cli_artifact_id or f"cli-{env_id}"

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["command_name"] = self.command_name
        base["cli_artifact_id"] = self.cli_artifact_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> BuildMcpCliTaskStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            command_name=data["command_name"],
            cli_artifact_id=data.get("cli_artifact_id"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")

        env = Env.get(self.env_id, deployed.env_version)
        environment_name = getattr(env, "environment_name", None)
        if not environment_name:
            raise RuntimeError(f"Env '{self.env_id}' has no environment_name")
        manifest = await fetch_interface_manifest(deployed.gateway_url, environment_name)
        script_source = generate_cli_script(
            self.command_name,
            interface_manifest=manifest,
            environment_name=environment_name,
        )

        with tempfile.TemporaryDirectory() as tmp_str:
            cli_dir = Path(tmp_str)
            bin_dir = cli_dir / "bin"
            bin_dir.mkdir()
            entrypoint_path = bin_dir / self.command_name
            entrypoint_path.write_text(script_source)
            os.chmod(entrypoint_path, 0o755)

            artifact = CliArtifact.put(
                id=self.cli_artifact_id,
                command_name=self.command_name,
                entrypoint=f"bin/{self.command_name}",
                cli_dir=cli_dir,
                env_id=self.env_id,
                env_version=deployed.env_version,
            )

        context.metadata["cli_artifact"] = {"id": artifact.id, "version": artifact.version, "type": artifact.type}
        logger.info(f"Created CliArtifact: id={artifact.id} version={artifact.version} command_name={artifact.command_name}")
        return context
