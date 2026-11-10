"""Task step that hands a deployed agent a CDP address to an env's website browser, through the agent's cdp-config/v1,
and hides that browser's MCP tools from the agent. A reset ends the address, so a task connects the agent again after
reset_env, or with a retry_config that reruns this step."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx
from agentenv_protocol.tool_access import DISABLE, DISABLE_TOOL_EXTENSION_URI, WILDCARD
from agentenv_protocol.tool_context import DEFAULT_ROLE

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
from agent_env.providers.sandbox_providers.local_docker.provider import host_url_for
from agent_env.providers.sandbox_providers.sandbox_provider import reachable_url
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.browser.base import BrowserStep

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
    its role unless ``keep_tools``. The address is the browser's own for that agent, as its sandbox reaches it."""

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
    ):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
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
        return cls(**cls._browser_from_dict(data), agent_name=data.get("agent_name"),
                   name=data.get("name") or DEFAULT_NAME, keep_tools=data.get("keep_tools", False))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed, browser, _ = self._find(context)
        if deployed.env_id != self.env_id:
            raise RuntimeError(f"the website browser {browser!r} is in a computer of the multi env {self.env_id!r}, whose "
                               "agents reach it through the multi env's gateway; browser_connect_cdp doesn't hand out "
                               "that address yet")
        agent = _deployed_agent(context, self.agent_name)
        cdp_config = A2AAgent.find_extension(agent.a2a_card or {}, A2AAgent.EXT_CDP_CONFIG)
        if cdp_config is None or not agent.a2a_url:
            raise RuntimeError(f"agent {agent.agent_name!r} can't take a CDP address: its card has no "
                               f"{A2AAgent.EXT_CDP_CONFIG}")
        address = await self._call(context, "cdp", {"user": agent.agent_name}, CDP_TIMEOUT_S)
        http_url, ws_url = _as_agent_reaches(address, deployed, agent)
        async with httpx.AsyncClient(timeout=ADD_TIMEOUT_S) as client:
            added = await client.post(agent.a2a_url + cdp_config["params"]["endpoint"], json={
                "name": self.name, "http_url": http_url, "ws_url": ws_url,
                "metadata": {"browser": browser or deployed.environment_card.get("name")}})
        added.raise_for_status()
        role = agent.role or DEFAULT_ROLE
        hidden = None if self.keep_tools else WILDCARD
        if hidden:
            await deployed.invoke(DISABLE_TOOL_EXTENSION_URI, DISABLE, {"role": role, "tools": hidden})
        logger.info(f"agent {agent.agent_name!r} has {self.name!r}, a CDP address to {browser or 'the env'} in env "
                    f"{deployed.env_id!r}" + (f"; the env's tools are hidden from role {role!r}" if hidden else ""))
        # The agent's cdp-config now holds this address under its name, so it replaces any connection recorded so.
        records = context.metadata.setdefault(CONNECTIONS_KEY, [])
        records[:] = [record for record in records
                      if (record["agent_name"], record["name"]) != (self.agent_name, self.name)] + [{
            "step_id": self.id, "env_id": self.env_id, "agent_name": self.agent_name, "browser": browser,
            "name": self.name, "keep_tools": self.keep_tools, "role": role, "tools_hidden": hidden}]
        return context


def _deployed_agent(context: TaskStepContext, agent_name: str) -> DeployedAgent:
    agent = next((a for a in context.deployed_agents if a.agent_name == agent_name), None)
    if agent is None:
        raise RuntimeError(f"agent {agent_name!r} isn't in context.deployed_agents")
    return agent


def _as_agent_reaches(address: dict, deployed: DeployedEnv, agent: DeployedAgent) -> tuple[str, str]:
    """``address``'s URLs as the agent's sandbox reaches them: under the env's URL as the agent reaches that, as
    deploy_agent gives it MCP URLs. A URL anywhere else, at a gateway's public address say, is kept."""
    base = deployed.environment_url.rstrip("/")
    if isinstance(deployed, DeployedSandboxEnv):
        theirs = reachable_url(base, from_sandbox_type=deployed.sandbox_type, to_sandbox_type=agent.sandbox_type)
    else:
        theirs = host_url_for(base, agent.sandbox_type)
    theirs = theirs.rstrip("/")
    return _rebased(address["http_url"], base, theirs), _rebased(address["ws_url"], _ws(base), _ws(theirs))


def _rebased(url: str, base: str, theirs: str) -> str:
    return theirs + url[len(base):] if url.startswith(base + "/") else url


def _ws(url: str) -> str:
    return "ws" + url.removeprefix("http") if url.startswith("http") else url
