"""Agents on the env's clock (FD-5916): an agent whose image runs the clock follower serves clock/v1's ``sync_time``
(``agentenv_protocol.a2a_agent.clock``), and follows the gateway's clock the way a computer does, so every program it
runs reads the env's time. ``follow`` declares it at the gateway as a follower with a read token of its own, then calls
its ``sync_time`` with the follower's config: the gateway's watch URL as the agent's sandbox reaches it, the name the
gateway knows it by and that token. From then on a clock write that waits for followers (``wait_followers_s``) waits
for the agent too. ``sync_env_clock`` has every agent registered with the env follow once it has armed the clock, and
``deploy_agent`` an agent it deploys while the clock is armed. An agent has one follower, so it follows one env's
clock: the first env it's registered with (``clock_env``), whatever order steps arm clocks in, even at once; another
env's is reported and skipped (``follows_another_env``). When ``teardown_sandboxes`` takes the agent's sandbox down,
``retire_followers`` has the gateway it follows stop waiting for it. Which
env each agent follows is kept under ``metadata[FOLLOWING_KEY]``, one key per agent, so steps that run at once and
each save the context never overwrite another's."""
from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING, Optional

import httpx

from agent_env.a2a_agent.a2a_agent import A2AAgent
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
#: ``metadata`` key of the env each agent follows: ``{slot: {"agent", "env_id", "follower"}}``, one slot per agent.
FOLLOWING_KEY = "agent_clock_following"


def follower_name(agent: DeployedAgent) -> str:
    """The name the gateway knows ``agent``'s follower by, apart from the computers' names."""
    return f"agent-{agent.agent_name}"


def _sync_time(agent: DeployedAgent) -> tuple[Optional[dict], Optional[str]]:
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
    for one that serves it and fails, having retired the follower it declared, or can't take the task's zone, as for
    a server."""
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
    try:
        async with httpx.AsyncClient() as client:
            answer = await client.post(url, json=body, timeout=timeout,
                                       headers={**sandbox_request_headers_for_url(url), **agent.control_headers})
    except httpx.HTTPError as e:
        why = f"({type(e).__name__}): {e}"
    else:
        if answer.status_code < 400:
            return {"synced": True, "follower": name, "result": answer.json()}
        why = f"(HTTP {answer.status_code}): {answer.text}"
    # Declared and not following: the gateway would wait for it on every later write, and nothing records it to retire.
    await _retire(env, name, timeout=RETIRE_TIMEOUT_S)
    raise RuntimeError(f"agent '{agent.agent_name}' doesn't follow the clock of env '{env.env_id}' "
                       f"{why[:ERROR_TEXT_CHARS]}")


async def _retire(env: DeployedEnv, follower: str, *, timeout: float) -> bool:
    """Have ``env``'s gateway stop waiting for ``follower``; whether it did. Best-effort: a gateway that's gone or
    takes no retirement is logged."""
    try:
        await env.invoke(EXT_CLOCK_URI, "retire_follower", {"name": follower}, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - best-effort: a teardown or a failed sync goes on either way
        logger.warning(f"env {env.env_id}'s gateway didn't retire follower {follower!r}: {type(e).__name__}: {e}"[:300])
        return False
    return True


def _slot(agent_name: str) -> str:
    """``agent_name`` as a key of its own when the context is saved, where a ``.`` or a leading ``$`` would make one
    save the whole map: percent-encoded, ``%`` first, so two names never share a key."""
    return agent_name.replace("%", "%25").replace(".", "%2E").replace("$", "%24")


def followed_env(context: TaskStepContext, agent: DeployedAgent) -> Optional[str]:
    """The env whose clock ``agent`` follows in this run, if any."""
    return ((context.metadata.get(FOLLOWING_KEY) or {}).get(_slot(agent.agent_name)) or {}).get("env_id")


def torn_down(context: TaskStepContext, agent: DeployedAgent) -> bool:
    """Whether ``teardown_sandboxes`` took ``agent``'s sandbox down: its record stays in the context, its server is gone."""
    from agent_env.task.teardown import TORN_DOWN_KEY

    return bool(agent.sandbox_id) and agent.sandbox_id in (context.metadata.get(TORN_DOWN_KEY) or ())


def clock_env(agent: DeployedAgent) -> Optional[str]:
    """The one env whose clock ``agent`` follows: the first it's registered with. Fixed by the agent alone, so two
    steps arming two of its envs' clocks at once never both take its one follower."""
    return next(iter(agent.env_ids or ()), None)


async def follow_in(context: TaskStepContext, agent: DeployedAgent, env: DeployedEnv, *,
                    env_get_time_url: Optional[str], timezone: Optional[str], timeout: float) -> dict:
    """``follow`` when ``env`` is the agent's ``clock_env``, recording it; another env's clock, which would take over
    its follower while that gateway still waits for it, is reported and skipped (``follows_another_env``)."""
    if env.env_id != clock_env(agent):
        return {"synced": False, "reason": "follows_another_env", "follows": clock_env(agent)}
    outcome = await follow(agent, env, env_get_time_url=env_get_time_url, timezone=timezone, timeout=timeout)
    if outcome["synced"]:
        context.metadata.setdefault(FOLLOWING_KEY, {})[_slot(agent.agent_name)] = {
            "agent": agent.agent_name, "env_id": env.env_id, "follower": outcome["follower"]}
    return outcome


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
    that step's ``require_agent_clock``."""
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
        outcomes.append({"env_id": env_id, "agent": agent.agent_name, **outcome})
    return outcomes


async def retire_followers(context: TaskStepContext, agent_names, *, timeout: float = RETIRE_TIMEOUT_S) -> list[str]:
    """Have the gateway whose clock each of ``agent_names`` follows in this run stop waiting for it, as its agent is
    torn down, so later writes that wait for followers don't wait for it or answer 504 naming it. Best-effort, as a
    teardown is: a gateway that's gone or takes no retirement is logged and its agent left recorded. The followers
    retired."""
    following, retired = context.metadata.get(FOLLOWING_KEY) or {}, []
    for name in dict.fromkeys(agent_names):
        if (record := following.get(_slot(name))) is None:
            continue
        env = next((d for d in context.deployed_envs if d.env_id == record["env_id"]), None)
        if env is None:
            continue
        if not await _retire(env, record["follower"], timeout=timeout):
            continue
        del following[_slot(name)]
        retired.append(record["follower"])
    return retired
