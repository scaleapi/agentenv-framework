"""Sandbox diagnostics shared by the steps that drive an A2A agent."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# On-disk agent logs to tail when the agent is NOT a docker container. The
# openclaw sidecar runs as the sandbox's *main process*, so `docker ps` lists
# nothing and the container path below comes back empty — which is how a run
# dies with "no container logs were retrievable" while the real cause (e.g. a
# LiteLLM 429 the harness swallowed into an opaque `cli_error`) sits in plain
# text in this file, inside a sandbox that gets reaped minutes later.
_AGENT_LOG_PATHS = (
    "/tmp/openclaw-gateway.log",
    "/tmp/agent.log",
)

# Markers for the lines worth lifting out of a log tail. A raw 500-line tail is
# mostly boot noise; the terminal error is what belongs in ``instance.error``
# and the hub UI, so prefer matching lines and fall back to the tail only when
# none match (an unrecognized failure shape must not silently yield nothing).
_AGENT_LOG_ERROR_MARKERS = (
    "isError=true",
    "rawError=",
    "failover decision",
    "rate limit",
    "Traceback",
    "ERROR",
)


def _salient_log_lines(text: str, max_lines: int = 8, max_chars: int = 2000) -> str:
    """Pick the error-bearing lines out of a log tail, newest last."""
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    hits = [ln for ln in lines if any(m.lower() in ln.lower() for m in _AGENT_LOG_ERROR_MARKERS)]
    picked = (hits or lines)[-max_lines:]
    return "\n".join(picked)[-max_chars:]


async def fetch_container_logs(agent, tail: int = 500) -> Optional[str]:
    """Best-effort dump of the agent's logs from its sandbox.

    The A2A protocol surfaces agent-side crashes as an opaque ``error_type``
    (e.g. ``cli_error``) with no message. The actual error — an empty LLM
    response, a missing model alias, a provider rate limit, a crashed harness —
    lives only in the agent's stdout/stderr. Pull a tail of it so failures are
    diagnosable from the worker logs and the hub UI instead of requiring a
    manual ``docker logs`` inside a sandbox that is usually gone by the time
    anyone looks.

    Two sources, because agents come in two shapes: ``docker logs`` for
    containerized agents, and ``_AGENT_LOG_PATHS`` for agents that ARE the
    sandbox's main process (openclaw). Both are attempted — a containerized
    agent may also write a log file, and neither source is authoritative.

    Returns None on any failure — this is diagnostics, never load-bearing,
    so it must not mask the underlying task failure.
    """
    sandbox_id = getattr(agent, "sandbox_id", None)
    if not sandbox_id:
        return None
    try:
        from agent_env.providers.sandbox_provider import (
            build_sandbox_provider,
            get_sandbox_provider,
        )

        sandbox_type = getattr(agent, "sandbox_type", None)
        provider = (
            build_sandbox_provider(sandbox_type) if sandbox_type else get_sandbox_provider()
        )
        sandbox = await provider.get_sandbox(sandbox_id)
        chunks: list[str] = []

        # Source 1: containerized agents. Scoped in its own try so a sandbox
        # with no docker at all (the openclaw shape) still reaches source 2
        # instead of aborting the whole fetch on the first `docker ps`.
        try:
            _, ps_out, _ = await sandbox.exec_with_output(
                "sudo", "docker", "ps", "-a", "--format", "{{.ID}}|{{.Names}}|{{.Image}}",
            )
            rows = [l for l in (ps_out or "").splitlines() if l.strip()]
            # Prefer the agent container; fall back to every container.
            preferred = [
                r.split("|", 1)[0].strip()
                for r in rows
                if "agent" in r.lower() or "openclaw" in r.lower()
            ]
            cids = preferred or [r.split("|", 1)[0].strip() for r in rows]
            for cid in cids[:3]:
                _, out, err = await sandbox.exec_with_output(
                    "sudo", "docker", "logs", "--tail", str(tail), cid,
                )
                # Last 500 chars of EACH stream (1000 total/container): stderr's
                # tail carries the actual error (e.g. "Agent completed with no
                # response"), stdout's tail the recent request/boot context.
                # Per-stream slicing guarantees a long stdout can't push the
                # stderr error out of the window.
                parts = []
                out_tail = (out or "").strip()[-500:]
                err_tail = (err or "").strip()[-500:]
                if out_tail:
                    parts.append(f"[stdout]\n{out_tail}")
                if err_tail:
                    parts.append(f"[stderr]\n{err_tail}")
                if parts:
                    chunks.append(f"[container {cid}]\n" + "\n".join(parts))
        except Exception as e:
            logger.debug("docker log source unavailable on sandbox %s: %s", sandbox_id, e)

        # Source 2: on-disk agent logs, for agents that are the sandbox's main
        # process. `sudo` fallback because the file is root-owned on some
        # images; both arms are quieted so a missing path costs one exec and
        # contributes nothing rather than raising.
        for path in _AGENT_LOG_PATHS:
            try:
                _, out, _ = await sandbox.exec_with_output(
                    "sh", "-c",
                    f"tail -n {int(tail)} {path} 2>/dev/null "
                    f"|| sudo tail -n {int(tail)} {path} 2>/dev/null",
                )
            except Exception as e:
                logger.debug("could not tail %s on sandbox %s: %s", path, sandbox_id, e)
                continue
            salient = _salient_log_lines(out or "")
            if salient:
                chunks.append(f"[{path}]\n{salient}")

        return "\n\n".join(chunks) or None
    except Exception as e:
        logger.warning(
            "Could not fetch agent container logs from sandbox %s: %s",
            sandbox_id, e, exc_info=True,
        )
        return None
