"""Task step that hands a deployed agent a CDP address to an env's website browser, through the agent's cdp-config/v1,
and hides that browser's MCP tools from the agent. A browser in one of a multi env's computers is reached through the
multi env's gateway, as its agents reach the rest of it. A reset ends the address, so a task connects the agent again
after reset_env, or with a retry_config that reruns this step."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx
from agentenv_protocol.client import find_extension_method, invoke_extension
from agentenv_protocol.tool_access import DISABLE, DISABLE_TOOL_EXTENSION_URI, WILDCARD
from agentenv_protocol.tool_context import DEFAULT_ROLE
from agentenv_protocol.website_browser import BROWSER

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
from agent_env.providers.sandbox_providers.local_docker.provider import host_url_for
from agent_env.providers.sandbox_providers.sandbox_provider import reachable_url
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.browser.base import BrowserStep, deployed_env

logger = logging.getLogger(__name__)

#: Where the step records each connection it makes, one per agent and name: the role its tools were hidden from.
CONNECTIONS_KEY = "browser_connections"
#: The name the address is added to the agent's cdp-config under when the step names none.
DEFAULT_NAME = "chromium"
#: How long the browser may take to answer cdp, which waits for the browser to be ready.
CDP_TIMEOUT_S = 180
#: How long the agent may take to answer cdp-config's add.
ADD_TIMEOUT_S = 60


class BrowserConnectCdpTaskStep(BrowserStep):
    """Hand the agent ``agent_name`` a CDP address to the website browser ``browser`` in the env ``env_id``, the
    env's only one when unnamed: added to its cdp-config/v1 as ``name``, with every tool the env serves hidden from
    its role unless ``keep_tools``. The address is the browser's own for that agent, as its sandbox reaches it; for a
    browser in one of a multi env's computers, the multi env's gateway's, which relays it to the browser. With a
    ``context``, a lease's label granted to the agent, the address is scoped to that lease, which the agent sees as
    the browser's default context, and ends with it."""

    type: ClassVar[str] = "browser_connect_cdp"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        agent_name: Optional[str] = None,
        browser: Optional[str] = None,
        name: str = DEFAULT_NAME,
        keep_tools: bool = False,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        context: Optional[str] = None,
    ):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         context=context)
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.name = name
        self.keep_tools = keep_tools

    def config_problems(self) -> list[str]:
        problems = super().config_problems()
        if not isinstance(self.name, str) or not self.name:
            problems.append(f"step {self.id!r}: name must be a non-empty string")
        if not isinstance(self.keep_tools, bool):
            problems.append(f"step {self.id!r}: keep_tools must be true or false")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "agent_name": self.agent_name, "name": self.name, "keep_tools": self.keep_tools}

    @classmethod
    def from_dict(cls, data: dict) -> BrowserConnectCdpTaskStep:
        return cls(**cls._in_context_from_dict(data), agent_name=data.get("agent_name"),
                   name=data.get("name") or DEFAULT_NAME, keep_tools=data.get("keep_tools", False))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed, browser, card = self._find(context)
        env = deployed_env(context, self.env_id)  # what the agent reaches: a multi env's gateway, or the browser's env
        agent = _deployed_agent(context, self.agent_name)
        cdp_config = A2AAgent.find_extension(agent.a2a_card or {}, A2AAgent.EXT_CDP_CONFIG)
        if cdp_config is None or not agent.a2a_url:
            raise RuntimeError(f"agent {agent.agent_name!r} can't take a CDP address: its card has no "
                               f"{A2AAgent.EXT_CDP_CONFIG}")
        if self.context_label is not None and not _takes(card, "context"):
            raise RuntimeError(f"the website browser {browser or deployed.env_id!r} scopes no CDP address to a lease: "
                               "rebuild it on a newer agentenv-framework-protocol")
        base, theirs = _env_urls(env, agent)
        asked = {"user": agent.agent_name, **({"origin": theirs} if _takes(card, "origin") else {}),
                 **({"context": self.context_label} if self.context_label is not None else {})}
        address = (await self._call(context, "cdp", asked, CDP_TIMEOUT_S, in_context=False)
                   if deployed.env_id == self.env_id else await _cdp_at_gateway(env, deployed, card, asked))
        http_url = _rebased(address["http_url"], base, theirs)
        ws_url = _rebased(address["ws_url"], _ws(base), _ws(theirs))
        async with httpx.AsyncClient(timeout=ADD_TIMEOUT_S) as client:
            added = await client.post(agent.a2a_url + cdp_config["params"]["endpoint"], json={
                "name": self.name, "http_url": http_url, "ws_url": ws_url,
                "metadata": {"browser": browser or deployed.environment_card.get("name")}})
        added.raise_for_status()
        role = agent.role or DEFAULT_ROLE
        hidden = None if self.keep_tools else WILDCARD
        if hidden:
            await env.invoke(DISABLE_TOOL_EXTENSION_URI, DISABLE, {"role": role, "tools": hidden})
        logger.info(f"agent {agent.agent_name!r} has {self.name!r}, a CDP address to {browser or 'the env'} in env "
                    f"{env.env_id!r}" + (f", scoped to the lease {self.context_label!r}" if self.context_label else "")
                    + (f"; the env's tools are hidden from role {role!r}" if hidden else ""))
        # The agent's cdp-config now holds this address under its name, so it replaces any connection recorded so.
        records = context.metadata.setdefault(CONNECTIONS_KEY, [])
        records[:] = [record for record in records
                      if (record["agent_name"], record["name"]) != (self.agent_name, self.name)] + [{
            "step_id": self.id, "env_id": self.env_id, "agent_name": self.agent_name, "browser": browser,
            "name": self.name, "keep_tools": self.keep_tools, "role": role, "tools_hidden": hidden,
            **({"context": self.context_label} if self.context_label is not None else {})}]
        return context


async def _cdp_at_gateway(gateway: DeployedEnv, computer: DeployedEnv, card: dict, params: dict) -> dict:
    """The browser's ``cdp`` answer, the browser ``card`` names in ``computer``, one of the multi env ``gateway``'s
    computers: asked through the gateway's ``/svc`` route to that computer, with the computer's control token, so the
    gateway names its own address in it, its CDP relay's, as it does for any client reaching it there, keeping the
    https or wss of the origin in ``params`` the browser minted it under."""
    key = next((name for name, record in (getattr(gateway, "child_envs", None) or {}).items()
                if record.get("sandbox_id") == computer.sandbox_id), None)
    if key is None:
        raise RuntimeError(f"env {gateway.env_id!r} keeps no record of the computer its website browser is in")
    if find_extension_method(card, BROWSER, "cdp") is None:
        raise RuntimeError(f"the website browser in env {gateway.env_id!r} serves no browser/v1 cdp: rebuild it on a "
                           "newer agentenv-framework-protocol")
    base = gateway.environment_url.rstrip("/")
    address = await invoke_extension(f"{base}/svc/{key}", card, BROWSER, params, CDP_TIMEOUT_S, method="cdp",
                                     headers=computer.control_headers)
    relayed = (f"{base}/browser/{key}/", f"{_ws(base)}/browser/{key}/")
    http_url, ws_url = str(address.get("http_url", "")), str(address.get("ws_url", ""))
    if http_url.startswith(relayed[0]) and ws_url.startswith(relayed[1]):
        return address
    if "origin" not in params and http_url.split("://", 1)[-1].startswith(relayed[0].split("://", 1)[1]):
        raise RuntimeError(f"the website browser in env {gateway.env_id!r}'s computer {key!r} takes no origin, so "
                           f"behind a tunnel that ends TLS its CDP address doesn't say {base.split('://')[0]} as the "
                           "gateway is reached: compose it again on agentenv-framework 0.9.1505 or later")
    raise RuntimeError(f"env {gateway.env_id!r}'s gateway didn't put the CDP address of the website browser in its "
                       f"computer {key!r} at its relay, as it does for a computer with one website browser; one "
                       "with several it doesn't tell apart yet")


def _deployed_agent(context: TaskStepContext, agent_name: str) -> DeployedAgent:
    agent = next((a for a in context.deployed_agents if a.agent_name == agent_name), None)
    if agent is None:
        raise RuntimeError(f"agent {agent_name!r} isn't in context.deployed_agents")
    return agent


def _env_urls(deployed: DeployedEnv, agent: DeployedAgent) -> tuple[str, str]:
    """The env's URL, and the env's URL as the agent's sandbox reaches it, as deploy_agent gives it MCP URLs: the
    origin the browser mints the agent's address under, since behind a tunnel that ends TLS the browser can't tell
    how it was reached."""
    base = deployed.environment_url.rstrip("/")
    if isinstance(deployed, DeployedSandboxEnv):
        theirs = reachable_url(base, from_sandbox_type=deployed.sandbox_type, to_sandbox_type=agent.sandbox_type)
    else:
        theirs = host_url_for(base, agent.sandbox_type)
    return base, theirs.rstrip("/")


def _takes(card: dict, field: str) -> bool:
    """Whether the browser's cdp takes ``field``, an origin or a context: one composed before it did refuses any it
    doesn't know."""
    method = find_extension_method(card, BROWSER, "cdp") or {}
    return field in method.get("request", {}).get("properties", {})


def _rebased(url: str, base: str, theirs: str) -> str:
    """``url`` moved under ``theirs`` when it's under the env's URL as the step reaches it, as a gateway's CDP answers
    are and a browser's that takes no origin; else kept."""
    return theirs + url[len(base):] if url.startswith(base + "/") else url


def _ws(url: str) -> str:
    return "ws" + url.removeprefix("http") if url.startswith("http") else url
