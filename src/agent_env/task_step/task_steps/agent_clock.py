"""Agents on the env's clock (FD-5916): an agent whose image runs the clock follower serves clock/v1's ``sync_time``
(``agentenv_protocol.a2a_agent.clock``), and follows the gateway's clock the way a computer does, so every program it
runs reads the env's time. ``follow`` declares it at the gateway as a follower with a read token of its own, then calls
its ``sync_time`` with the follower's config: the gateway's watch URL as the agent's sandbox reaches it, the name the
gateway knows it by and that token. From then on a clock write that waits for followers (``wait_followers_s``) waits
for the agent too. ``sync_env_clock`` has every agent registered with the env follow once it has armed the clock, and
``deploy_agent`` an agent it deploys while the clock is armed. An agent has one follower, so it follows one env's
clock: the first a run has it follow; another env's is reported and skipped (``follows_another_env``). When
``teardown_sandboxes`` takes the agent down, ``retire_followers`` has each gateway it followed stop waiting for it."""
from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING, Optional

import httpx

from agent_env.env.env import EnvCapabilityUnsupported, gateway_url_of
from agent_env.env.gateway.constants import CLOCK_WATCH_PATH, EXT_CLOCK_URI
from agent_env.providers.sandbox_providers.local_docker.provider import host_url_for
from agent_env.providers.sandbox_providers.sandbox_provider import reachable_url, sandbox_request_headers_for_url
from agent_env.task_step.context import StepContextError

if TYPE_CHECKING:
    from agent_env.env.env import DeployedEnv
    from agent_env.task_step.context import DeployedAgent, TaskStepContext

logger = logging.getLogger(__name__)

#: What an agent's ``sync_time`` must take to run a follower of its own.
FOLLOWER_FIELDS = ("watch_url", "name", "read_token")
#: Bytes of randomness in an agent follower's read token.
READ_TOKEN_BYTES = 32
#: How much of a failing agent's answer an error quotes.
ERROR_TEXT_CHARS = 600
#: How long ``deploy_agent`` gives a late agent's declaration and ``sync_time``, which waits for its follower to connect.
LATE_FOLLOW_TIMEOUT_S = 60
#: How long a teardown gives each gateway to retire an agent's follower.
RETIRE_TIMEOUT_S = 10


def follower_name(agent: DeployedAgent) -> str:
    """The name the gateway knows ``agent``'s follower by, apart from the computers' names."""
    return f"agent-{agent.agent_name}"


def _sync_time(agent: DeployedAgent) -> tuple[Optional[dict], Optional[str]]:
    from agent_env.a2a_agent import A2AAgent
    return A2AAgent.operation(A2AAgent.find_extension(agent.a2a_card or {}, EXT_CLOCK_URI), "sync_time")


def _fields(method: dict) -> set[str]:
    request = method.get("request") or {}
    return {*request.get("required", ()), *request.get("optional", ()), *(request.get("properties") or {})}


def watch_url_for(agent: DeployedAgent, env: DeployedEnv) -> str:
    """The gateway's watch URL as ``agent``'s sandbox reaches it, as its MCP URL is (``deploy_agent``)."""
    url = gateway_url_of(env).rstrip("/") + CLOCK_WATCH_PATH
    if not getattr(env, "sandbox_id", None):
        return host_url_for(url, agent.sandbox_type)
    return reachable_url(url, from_sandbox_type=env.sandbox_type, to_sandbox_type=agent.sandbox_type)


async def follow(agent: DeployedAgent, env: DeployedEnv, *, env_get_time_url: Optional[str],
                 timezone: Optional[str], timeout: float) -> dict:
    """Have ``agent`` follow ``env``'s gateway clock. ``{"synced": False, "reason": ...}`` for an agent that can't:
    its card serves no ``sync_time`` (``not_advertised``) or one without the follower's fields
    (``no_follower_fields``), or the gateway takes no followers after it starts (``gateway_takes_no_followers``); raises
    for one that serves it and fails, or can't take the task's zone, as for a server."""
    method, path = _sync_time(agent)
    if method is None:
        return {"synced": False, "reason": "not_advertised"}
    fields = _fields(method)
    if not set(FOLLOWER_FIELDS) <= fields:
        return {"synced": False, "reason": "no_follower_fields"}
    if timezone is not None and "timezone" not in fields:
        raise StepContextError(f"agent '{agent.agent_name}' serves {EXT_CLOCK_URI} sync_time without a timezone; "
                               f"rebuild its image or drop timezone={timezone!r} from the task")
    name, token = follower_name(agent), secrets.token_urlsafe(READ_TOKEN_BYTES)
    try:
        await env.invoke(EXT_CLOCK_URI, "declare_follower", {"name": name, "read_token": token}, timeout=timeout)
    except EnvCapabilityUnsupported:
        return {"synced": False, "reason": "gateway_takes_no_followers"}
    body = {"env_get_time_url": env_get_time_url or "", "watch_url": watch_url_for(agent, env), "name": name,
            "read_token": token, **({"timezone": timezone} if timezone is not None else {})}
    url = (agent.a2a_url or agent.api_url).rstrip("/") + path
    async with httpx.AsyncClient() as client:
        answer = await client.post(url, json=body, timeout=timeout, headers=sandbox_request_headers_for_url(url))
    if answer.status_code >= 400:
        raise RuntimeError(f"agent '{agent.agent_name}' doesn't follow the clock of env '{env.env_id}' "
                           f"(HTTP {answer.status_code}): {answer.text[:ERROR_TEXT_CHARS]}")
    return {"synced": True, "follower": name, "result": answer.json()}


def followed_env(context: TaskStepContext, agent: DeployedAgent) -> Optional[str]:
    """The env whose clock ``agent`` follows in this run, if any."""
    return next((c["env_id"] for c in context.metadata.get("clock_configurations") or []
                 for a in c.get("agents") or () if a.get("agent") == agent.agent_name and a.get("synced")), None)


async def follow_in(context: TaskStepContext, agent: DeployedAgent, env: DeployedEnv, *,
                    env_get_time_url: Optional[str], timezone: Optional[str], timeout: float) -> dict:
    """``follow``, unless ``agent`` already follows another env's clock in this run, whose follower a second one
    would take over while that gateway still waits for it: reported and skipped (``follows_another_env``)."""
    followed = followed_env(context, agent)
    if followed is not None and followed != env.env_id:
        return {"synced": False, "reason": "follows_another_env", "follows": followed}
    return await follow(agent, env, env_get_time_url=env_get_time_url, timezone=timezone, timeout=timeout)


def summary(outcome: dict) -> dict:
    """What a log says of an outcome: never the agent's whole answer."""
    return {key: outcome[key] for key in ("synced", "reason", "follower", "follows") if key in outcome}


def _armed(context: TaskStepContext, env_id: str) -> Optional[dict]:
    """The configuration of the last ``sync_env_clock`` that armed ``env_id``'s clock in this run, if any."""
    return next((c for c in reversed(context.metadata.get("clock_configurations") or [])
                 if c.get("env_id") == env_id and "env_get_time_url" in c), None)


async def follow_armed_clocks(context: TaskStepContext, agent: DeployedAgent, *,
                              timeout: float = LATE_FOLLOW_TIMEOUT_S) -> list[dict]:
    """Have an agent ``deploy_agent`` just deployed follow the clock of each env registered with it that this run has
    armed, as ``sync_env_clock`` has every agent then: one that can't is reported and skipped, or fails the step under
    that step's ``require_agent_clock``. Each outcome is recorded with that step's, under its ``agents``."""
    outcomes = []
    for env_id in agent.env_ids:
        if (armed := _armed(context, env_id)) is None:
            continue
        env = next((d for d in context.deployed_envs if d.env_id == env_id), None)
        if env is None or gateway_url_of(env) is None:
            continue
        outcome = await follow_in(context, agent, env, env_get_time_url=armed.get("env_get_time_url"),
                                  timezone=armed.get("timezone"), timeout=timeout)
        if not outcome["synced"] and armed.get("require_agent_clock", False):
            raise StepContextError(f"agent '{agent.agent_name}' can't follow the clock of env '{env_id}': "
                                   f"{outcome['reason']}")
        logger.info(f"agent '{agent.agent_name}' on env {env_id}'s clock: {summary(outcome)}")
        armed.setdefault("agents", []).append({"agent": agent.agent_name, **outcome})
        outcomes.append({"env_id": env_id, "agent": agent.agent_name, **outcome})
    return outcomes


async def retire_followers(context: TaskStepContext, agent_names, *, timeout: float = RETIRE_TIMEOUT_S) -> list[str]:
    """Have each gateway whose clock one of ``agent_names`` follows in this run stop waiting for it, as its agent is
    torn down, so later writes that wait for followers don't wait for it or answer 504 naming it. Best-effort, as a
    teardown is: a gateway that's gone or takes no retirement is logged. The followers retired."""
    names, retired = set(agent_names), []
    for configuration in context.metadata.get("clock_configurations") or []:
        env = next((d for d in context.deployed_envs if d.env_id == configuration.get("env_id")), None)
        for outcome in configuration.get("agents") or ():
            if env is None or outcome.get("agent") not in names or not outcome.get("synced"):
                continue
            try:
                await env.invoke(EXT_CLOCK_URI, "retire_follower", {"name": outcome["follower"]}, timeout=timeout)
            except Exception as e:  # noqa: BLE001 - best-effort, as the rest of a teardown
                logger.warning(f"env {env.env_id}'s gateway didn't retire follower {outcome['follower']!r}: "
                               f"{type(e).__name__}: {e}"[:300])
                continue
            outcome["synced"], outcome["retired"] = False, True
            retired.append(outcome["follower"])
    return retired
