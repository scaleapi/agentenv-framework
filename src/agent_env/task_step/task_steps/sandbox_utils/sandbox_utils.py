"""Sandbox diagnostics shared by the steps that drive an A2A agent."""

from __future__ import annotations

import logging
from typing import Optional

from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_sandbox_provider

logger = logging.getLogger(__name__)

# Room for the agent's own account of a failure (its protocol ``error_message``).
_AGENT_ERROR_TEXT_LIMIT = 2000
# Containers tailed when none is the agent's own, e.g. an agent installed into an env's containers.
_FALLBACK_CONTAINER_LIMIT = 3
# Core names the agent containers it starts ``agent-api``, ``agent-<id>`` or ``a2a-agent-*``.
_AGENT_CONTAINER_MARKER = "agent"
# Per stream, so a long stdout cannot push the stderr error out of the window.
_STREAM_TAIL_CHARS = 500


def agent_error_text(error_message: Optional[str], response_text: Optional[str]) -> str:
    """The agent's own failure text, bounded: its ``error_message``, else its response."""
    return (error_message or response_text or "")[:_AGENT_ERROR_TEXT_LIMIT]


async def fetch_container_logs(agent, tail: int = 500) -> Optional[str]:
    """Tail the docker logs of the agent's container, so a failed A2A task stays diagnosable
    after its sandbox is reaped.

    The agent's own container is the one named ``sandbox.container_name``. When none is, the
    containers are tailed up to ``_FALLBACK_CONTAINER_LIMIT``, agent-named ones first. Best
    effort: None on any failure, including a sandbox with no docker, so it never masks the
    task's own failure.
    """
    sandbox_id = getattr(agent, "sandbox_id", None)
    if not sandbox_id:
        return None
    try:
        sandbox_type = getattr(agent, "sandbox_type", None)
        provider = (
            build_sandbox_provider(sandbox_type) if sandbox_type else get_sandbox_provider()
        )
        sandbox = await provider.get_sandbox(sandbox_id)
        chunks: list[str] = []
        try:
            _, ps_out, _ = await sandbox.exec_with_output(
                "sudo", "docker", "ps", "-a", "--format", "{{.ID}}|{{.Names}}",
            )
            rows = [
                (cid.strip(), name.strip())
                for cid, _, name in (line.partition("|") for line in (ps_out or "").splitlines())
                if cid.strip()
            ]
            own = [cid for cid, name in rows if name == sandbox.container_name]
            by_agent_name = sorted(rows, key=lambda row: _AGENT_CONTAINER_MARKER not in row[1])
            cids = own or [cid for cid, _ in by_agent_name][:_FALLBACK_CONTAINER_LIMIT]
            for cid in cids:
                _, out, err = await sandbox.exec_with_output(
                    "sudo", "docker", "logs", "--tail", str(tail), cid,
                )
                parts = []
                out_tail = (out or "").strip()[-_STREAM_TAIL_CHARS:]
                err_tail = (err or "").strip()[-_STREAM_TAIL_CHARS:]
                if out_tail:
                    parts.append(f"[stdout]\n{out_tail}")
                if err_tail:
                    parts.append(f"[stderr]\n{err_tail}")
                if parts:
                    chunks.append(f"[container {cid}]\n" + "\n".join(parts))
        except Exception as e:
            logger.debug("docker logs unavailable on sandbox %s: %s", sandbox_id, e)
        return "\n\n".join(chunks) or None
    except Exception as e:
        logger.warning(
            "Could not fetch agent container logs from sandbox %s: %s",
            sandbox_id, e, exc_info=True,
        )
        return None


async def find_agent_container(sandbox) -> str:
    """The agent container running on a VM sandbox: the sandbox's own ``container_name``, else the
    first ``a2a-agent-*``. A sandbox that owns its container never takes another: its Docker host
    is shared, so any other agent container there is another run's."""
    exit_code, stdout, stderr = await sandbox.exec_with_output("sudo", "docker", "ps", "--format", "{{.Names}}")
    if exit_code != 0:
        raise RuntimeError(f"Failed to list containers: {stderr[:300]}")
    running = [n.strip() for n in stdout.splitlines() if n.strip()]
    if sandbox.container_name in running:
        return sandbox.container_name
    if getattr(sandbox, "owns_container", False):
        raise RuntimeError(
            f"Agent container {sandbox.container_name!r} is not running. Running containers: {running}."
        )
    fallback = [n for n in running if n.startswith("a2a-agent-")]
    if fallback:
        if len(fallback) > 1:
            logger.warning(f"Multiple a2a-agent-* containers found; using first: {fallback}")
        return fallback[0]
    raise RuntimeError(
        f"No agent container found on the VM (looked for {sandbox.container_name!r} or 'a2a-agent-*'). "
        f"Running containers: {running}. Has deploy_agent been run in this task?"
    )
