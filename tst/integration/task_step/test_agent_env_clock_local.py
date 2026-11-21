"""An agent image on a task env's clock (FD-5916), end to end on local Docker: the agent deployed with the env serves
clock/v1 sync_time; sync_env_clock declares it a follower at the real gateway and hands it its config; from then on
every program the agent's user runs, its CLI included, reads the env's time while the A2A server stays on real time; a
clock write that waits for followers waits for it; and once teardown_sandboxes takes it down, writes stop waiting for
it. An agent whose CLI speaks a protocol the stub endpoint serves also runs one real turn.

Runs over images built beforehand, from this branch: AGENTENV_E2E_AGENT_IMAGES names one per agent, as
``claude-code=<tag>,codex=<tag>``; an agent it doesn't name is skipped as a missing capability."""

import asyncio
import json
import re
import subprocess
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from agentenv_protocol.client import ExtensionCallError

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.a2a_agent.clock import FOLLOWING_KEY
from agent_env.artifact import DockerImageArtifact
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.providers.sandbox_providers.local_docker.provider import (
    _host_ips,
    host_url_for,
)
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
from agent_env.task_step.task_steps.teardown_sandboxes import TeardownSandboxesTaskStep
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs
from tst.integration.task_step.test_prompt_agent_live_trajectory_local import (
    local_registry as local_registry,
)
from tst.integration.task_step.test_register_env_triggers_local import _deploy_items
from tst.util.capabilities import agent_clock_image, skip_without_agent_clock_image
from tst.util.fake_anthropic import FakeAnthropic, tool_then_answer

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

VIRTUAL = "2031-03-04T05:06:07Z"
LATER = "2031-03-05T10:00:00Z"
PAUSED = "2031-03-05T12:00:00Z"


def _in_agent(container: str, *command: str, real_time: bool = False) -> str:
    env = ["-e", "AGENTENV_REAL_TIME=1"] if real_time else []
    return subprocess.run(["docker", "exec", "-u", "agent", *env, container, *command], capture_output=True,
                          text=True, check=True, timeout=30).stdout.strip()


def _routes_through_proxy(container: str, bridge: str) -> bool:
    """Whether the CLI the agent starts now goes out through the egress proxy, as the image's bridge decides with the
    ``clock_armed`` and ``egress_env`` every clocked bridge imports from ``agent_egress`` beside it."""
    # The A2A server's environment has the proxy (root.conf); this shell's doesn't, so it's passed as the server has it.
    return _in_agent(container, "python3", "-c", f"import {bridge} as b, agent_egress as e; print('HTTPS_PROXY' in "
                     "b.egress_env({e.EGRESS_PROXY_ENV: 'http://127.0.0.1:3128'}, armed=b.clock_armed()))",
                     real_time=True) == "True"


async def _claude_turn(context: TaskStepContext, endpoint: FakeAnthropic) -> None:
    """One turn of the real Claude Code CLI: the date it gives the model and a Bash `date` the model runs are the env's."""
    endpoint.respond = tool_then_answer("Bash", {"command": "date -u +%FT%TZ"}, prefix="the shell said: ")
    await PromptAgentTaskStep(id="ask", version=None, prompt="What's the date? Run date.", agent_name="solver",
                              max_conversation_turns=1).execute(context)
    bodies = [json.dumps(call.body) for call in endpoint.messages()]
    dated = {m for body in bodies for m in re.findall(r"Today's date is ([0-9]{4}-[0-9]{2}-[0-9]{2})", body)}
    print("the CLI's date line:", dated, "Bash date:", endpoint.tool_results())
    assert dated == {VIRTUAL[:10]}, dated
    assert any(VIRTUAL in result for result in endpoint.tool_results()), endpoint.tool_results()


@dataclass(frozen=True)
class Agent:
    """What the test needs of an agent's image: its bridge module, a command its CLI answers without a model (which
    must return under the preloaded library), and a turn against the stub endpoint, for a CLI that speaks its protocol."""

    bridge: str
    version: tuple[str, ...]
    shows: str
    turn: Callable[[TaskStepContext, FakeAnthropic], Awaitable[None]] | None = None


AGENTS = {
    "claude-code": Agent(bridge="claude_bridge", version=("claude", "--version"), shows="(Claude Code)",
                         turn=_claude_turn),
    "opencode": Agent(bridge="opencode_bridge", version=("opencode", "--version"), shows="1.17.18"),
}


def _container_of(sandbox_id: str, *, required: bool = True) -> str | None:
    """The container behind ``sandbox_id``, stopped or running; None, when not ``required``, for one that's gone."""
    names = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True,
                           check=True).stdout.split()
    found = [n for n in names if sandbox_id in n]
    if not required and len(found) != 1:
        return None
    [name] = found
    return name


async def _remove_agents(context: TaskStepContext) -> None:
    """Remove the agent containers teardown didn't take down; one already gone, or failing to go, is left."""
    gone = set(context.metadata.get("torn_down_sandbox_ids") or [])
    for agent in context.deployed_agents:
        name = agent.sandbox_id and agent.sandbox_id not in gone and _container_of(agent.sandbox_id, required=False)
        if name:
            await asyncio.to_thread(subprocess.run, ["docker", "rm", "-f", name], capture_output=True)


async def _followers(deployed) -> dict:
    return (await deployed.invoke(EXT_CLOCK_URI, "state", {}, timeout=10)).get("followers") or {}


@pytest.mark.parametrize("name", [pytest.param(name, marks=skip_without_agent_clock_image(name)) for name in AGENTS])
@pytest.mark.asyncio
async def test_an_agent_image_follows_the_envs_clock_until_it_is_torn_down(local_registry, name):
    with FakeAnthropic(host=_host_ips()[-1]) as endpoint:  # where this machine's containers reach it
        await _follows_until_torn_down(name, AGENTS[name], agent_clock_image(name), endpoint)


async def _follows_until_torn_down(name: str, spec: Agent, tag: str, endpoint: FakeAnthropic) -> None:
    _seed_bootstrap_envs()
    env, deployed, _ = await _deploy_items("agentclock")
    uid = uuid.uuid4().hex[:8]
    image = DockerImageArtifact.put(id=f"{name}-clock-img-{uid}", description="e2e", image_name=tag)
    agent = A2AAgent.put(id=f"{name}-clock-{uid}", docker_image_artifact=image,
                         default_env_vars={"LITELLM_API_KEY": "fake-key",
                                           # as the agent's container reaches it: on macOS, loopback is the host's
                                           "LITELLM_BASE_URL": host_url_for(endpoint.url, "local")})
    context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[], metadata={})
    try:
        await DeployAgentTaskStep(id="deploy-solver", version=None, env_ids=[env.id], a2a_agent_id=agent.id,
                                  a2a_agent_version=agent.version, agent_name="solver",
                                  sandbox_type="local").execute(context)
        container = _container_of(context.deployed_agents[-1].sandbox_id)
        assert _in_agent(container, "date", "-u", "+%Y") == str(datetime.now(timezone.utc).year)  # not armed: real
        assert not _routes_through_proxy(container, spec.bridge)  # and the CLI goes out directly, as before
        # The CLI starts under the preloaded library: without its init guard, Claude Code's first clock read deadlocked.
        assert spec.shows in _in_agent(container, "timeout", "30", *spec.version)

        await SyncEnvClockTaskStep(id="clock", version=None, env_id=env.id, virtual_time=VIRTUAL,
                                   virtual_seconds_per_real_second=0, require_agent_clock=True,
                                   timeout_seconds=60).execute(context)

        [outcome] = context.metadata["clock_configurations"][-1]["agents"]
        print("sync outcome:", {k: v for k, v in outcome.items() if k != "result"}, outcome.get("result"))
        assert (outcome["agent"], outcome["synced"], outcome["follower"]) == ("solver", True, "agent-solver")
        assert context.metadata[FOLLOWING_KEY]["solver"]["env_id"] == env.id
        assert "agent-solver" in await _followers(deployed)

        # Every program the agent's user runs reads the env's time, frozen at rate 0; the marker opts out.
        shown = {"date": _in_agent(container, "date", "-u", "+%FT%TZ"),
                 "python": _in_agent(container, "python3", "-c", "import datetime; print(datetime.datetime.now("
                                                              "datetime.timezone.utc).isoformat())"),
                 "real": _in_agent(container, "date", "-u", "+%FT%TZ", real_time=True)}
        print("agent reads:", shown)
        assert shown["date"] == VIRTUAL and shown["python"].startswith("2031-03-04T05:06:07")
        assert shown["real"].startswith(str(datetime.now(timezone.utc).year))
        assert _routes_through_proxy(container, spec.bridge)  # armed: live sites' TLS goes through the proxy
        assert spec.shows in _in_agent(container, "timeout", "30", *spec.version)  # the CLI starts on the env's clock
        if spec.turn is not None:
            await spec.turn(context, endpoint)

        # A write that waits for followers waits for the agent, which then shows it at once.
        started = time.monotonic()
        await deployed.invoke(EXT_CLOCK_URI, "advance", {"virtual_time": LATER, "wait_followers_s": 20}, timeout=30)
        print(f"advance acknowledged by the agent in {time.monotonic() - started:.2f}s")
        assert _in_agent(container, "date", "-u", "+%FT%TZ") == LATER
        assert (await _followers(deployed))["agent-solver"]["version"] is not None

        # It is really waited for: frozen, the agent can't acknowledge, and the write answers 504 naming it; thawed,
        # its follower catches up on its own.
        subprocess.run(["docker", "pause", container], check=True, capture_output=True)
        try:
            started = time.monotonic()
            with pytest.raises(ExtensionCallError) as frozen:
                await deployed.invoke(EXT_CLOCK_URI, "advance", {"virtual_time": PAUSED, "wait_followers_s": 3},
                                      timeout=30)
            print(f"advance while the agent was frozen: {time.monotonic() - started:.2f}s, "
                  f"{frozen.value.response.status_code} {frozen.value.response.text[:200]}")
        finally:
            subprocess.run(["docker", "unpause", container], check=True, capture_output=True)
        assert frozen.value.response.status_code == 504 and frozen.value.response.json()["stale"] == ["agent-solver"]
        deadline = time.monotonic() + 30
        while _in_agent(container, "date", "-u", "+%FT%TZ") != PAUSED:
            assert time.monotonic() < deadline, "the agent's follower didn't catch up after it was thawed"
            await asyncio.sleep(0.5)

        await TeardownSandboxesTaskStep(id="down", version=None, agent_names=["solver"]).execute(context)
        assert "agent-solver" not in await _followers(deployed)
        assert "solver" not in (context.metadata.get(FOLLOWING_KEY) or {})

        # Torn down, the agent is no longer waited for: a write answers at once, not after 20 s or with a 504.
        started = time.monotonic()
        await deployed.invoke(EXT_CLOCK_URI, "advance", {"virtual_time": "2031-03-06T00:00:00Z",
                                                         "wait_followers_s": 20}, timeout=30)
        waited = time.monotonic() - started
        print(f"advance after teardown answered in {waited:.2f}s")
        assert waited < 5
    finally:
        try:
            await _remove_agents(context)
        finally:
            await env.close()
