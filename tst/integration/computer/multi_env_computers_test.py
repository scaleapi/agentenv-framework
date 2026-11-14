"""The desktop example's computer as a child env of a multi env beside an MCP server, on the local backend.

The deploy_env step deploys each computer on a sandbox of its own and the gateway in front of them all. Through the
gateway's one /mcp an agent sees the server's tools and each computer's under its name, takes screenshots by those names,
and a role rule narrows them as it would a server's; /svc/<computer>/ reaches the computer. A later process reattaches the
computers from the multi env's record, and the run's teardown leaves nothing up. Two computers keep serving through a
computer's restart and the gateway's, a failed redeploy leaves the earlier deployment serving, and a computer that fails
mid-deploy, or an entry the multi env refuses, leaves nothing up. Needs Docker and the local registry on localhost:5000."""

import asyncio
import json
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import httpx
import pytest
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.tool_access import (DISABLE, DISABLE_TOOL_EXTENSION_URI, ENABLE, ENABLE_TOOL_EXTENSION_URI,
                                           WILDCARD)
from agentenv_protocol.tool_context import DEFAULT_ROLE
from agentenv_protocol.types import WELL_KNOWN_PATH
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent_env.artifact import DockerImageArtifact, EnvironmentArtifact, EnvironmentUniverseArtifact, FileArtifact
from agent_env.config import get_config, website_browser_env_id
from agent_env.env import Env
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.computer.env import COMPUTER_TOOLS, ComputerEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.playwright_mcp import PLAYWRIGHT_MCP_ENVIRONMENT_NAME
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.legacy_protocol import card_servers
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep
from tst.util import local_sandboxes
from tst.integration.computer.conftest import CONTAINER, multi_env
from tst.integration.env.envs.server_env_local_test import _items_artifact
from tst.integration.store.local_parallel_deploy_test import _docker

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

UNPREFIXED = re.compile(r"^computer_")


def _image(fixture: str, tag: str) -> DockerImageArtifact:
    """The image of the ``tst/data`` server ``fixture``, built with this checkout's agentenv_protocol."""
    repo = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as build_dir:
        context = Path(build_dir)
        shutil.copytree(repo / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol", context / "agentenv_protocol",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("server.py", "Dockerfile"):
            shutil.copy(repo / "tst" / "data" / fixture / name, context / name)
        built = _docker("build", "-t", tag, str(context))
        assert built.returncode == 0, built.stderr[-2000:]
    return DockerImageArtifact.put(id=tag, description="multi env sites e2e", image_name=tag)


@pytest.fixture(scope="module")
def shop(items):
    """A website whose backend is the small agentenv website, named shop.test, under a local id as ``items`` is. A gateway
    in front of websites also runs the website browser's MCP server; a small one stands in for it, since only the
    routing of the site is under test here."""
    uid = uuid.uuid4().hex[:8]
    with namespace_routing():
        MCPServerEnv.put(id=website_browser_env_id(get_config()), environment_name=PLAYWRIGHT_MCP_ENVIRONMENT_NAME,
                         docker_image_artifact=_image("cardless_mcp", f"office-browser-{uid}"))
        return WebsiteEnv.put(id=f"{items.id.rsplit('/', 1)[0]}/shop", environment_name="shop", domains=["shop.test"],
                              backend_docker_image_artifact=_image("agentenv_website", f"office-shop-{uid}"))


LOAD_TIMEOUT_S = 20  # a site's answer inside the computer
RUN_TIMEOUT_S = LOAD_TIMEOUT_S + 20  # the command that loads it, which must outlast the load


async def _reaches(computer, url: str) -> str:
    """The HTTP status ``url`` answers inside the computer, through its site proxy, over HTTPS, trusting the deploy's CA
    from the system's store."""
    load = f"import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout={LOAD_TIMEOUT_S}).status)"
    answer = await inside.run(computer, ["python3", "-c", load, url], timeout_s=RUN_TIMEOUT_S)
    return answer["stdout"].strip() or answer.get("stderr", "")[-300:]


async def _tools(deployed) -> tuple[set[str], bool]:
    """The tools the gateway lists to the default role, and whether a screenshot by the desktop's name answered."""
    async with streamable_http_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            shot = await session.call_tool("desktop_computer_screenshot", {}) if "desktop_computer_screenshot" in names else None
    return names, shot is not None and not shot.isError and any(part.type == "image" for part in shot.content)


async def _answers(deployed, tools: list[str], *, until: bool = True, tries: int = 40) -> dict[str, bool]:
    """Whether each of ``tools`` answers through the gateway, called again every 3 s until all answer, or with
    ``until=False``, once."""
    answered: dict[str, bool] = {}
    for _ in range(tries if until else 1):
        try:
            async with streamable_http_client(deployed.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    answered = {tool: not (await session.call_tool(tool, {})).isError for tool in tools}
        except Exception:  # noqa: BLE001 - the gateway restarting
            answered = dict.fromkeys(tools, False)
        if all(answered.values()) or not until:
            return answered
        await asyncio.sleep(3)
    return answered


def _universe(uid: str, files: dict[str, bytes]) -> EnvironmentUniverseArtifact:
    """The items server's two items, and the desktop's ``files`` by their paths under its user's home."""
    desktop = EnvironmentArtifact.put(id=f"office-desktop-{uid}", environment_name="desktop", file_artifacts={
        path: FileArtifact.put_bytes(id=f"office-desktop-{uid}-{index}", description="multi env computers e2e",
                                     filename=path.rsplit("/", 1)[-1], content=content, content_type="text/plain")
        for index, (path, content) in enumerate(files.items())})
    return EnvironmentUniverseArtifact.put(id=f"office-world-{uid}", environment_artifacts=[_items_artifact(uid), desktop])


def _snapshotted(context: TaskStepContext, step: str) -> dict[str, dict[str, bytes]]:
    """Each environment the snapshot ``step`` kept, with its files by their paths."""
    kept = context.metadata["env_snapshotted_universes"][step]
    universe = EnvironmentUniverseArtifact.get(kept["id"], version=kept["version"])
    return {environment.environment_name: {path: file.load() for path, file in environment.get_file_artifacts().items()}
            for environment in universe.get_environment_artifacts()}


async def _items(deployed) -> str:
    async with streamable_http_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return " ".join(part.text for part in (await session.call_tool("list_items", {})).content if part.type == "text")


def _containers(sandbox_id: str) -> list[str]:
    return _docker("ps", "-aq", "--filter", f"name=agent-{sandbox_id}").stdout.split()


@pytest.fixture
def made(monkeypatch) -> list[str]:
    return local_sandboxes.recorded_sandbox_ids(monkeypatch)


def _containers_made(made: list[str]) -> set[str]:
    return local_sandboxes.container_names_carrying(made)


def _gateway(deployed, made: list[str]) -> str:
    return next(name for name in _containers_made(made)
                if name.endswith("-gateway-1") and deployed.sandbox_id.removeprefix("local-") in name)


@pytest.mark.asyncio
async def test_a_computer_in_a_multi_env_is_served_through_its_gateway_under_its_name_and_torn_down_with_it(desktop, items,
                                                                                                         shop):
    with namespace_routing():
        office = multi_env(items, Env.get(desktop.id), websites=[shop])
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"desktop": {**CONTAINER, "env_params": {"form": "container", "site_routing": True}}})
        assert step.preflight() == []
        context = TaskStepContext()
        try:
            [deployed] = (await step.execute(context)).deployed_envs
            computer = deployed.child_envs["desktop"]
            listed, shot = await _tools(deployed)
            stated = {child.environment_name: child.name for child in protocol_v1.children(deployed.environment_card)}
            site = deployed.child_env("shop", mcp=False)
            servers = {(server.environment_name, server.mcp)
                       for server in card_servers(deployed.environment_card, deployed.environment_url)}
            async with httpx.AsyncClient(timeout=30) as client:
                card = (await client.get(f"{deployed.gateway_url}/svc/desktop{WELL_KNOWN_PATH}")).json()
            # Load, snapshot and reset reach the computer through the multi env, beside its MCP server.
            uid, own = uuid.uuid4().hex[:8], DeployedEnv.from_dict(computer)
            routed = await _reaches(own, "https://shop.test/api/health")
            world = _universe(uid, {"Documents/note.txt": b"from the universe"})
            await LoadArtifactTaskStep(id="seed", version=None, env_id=office.id, artifact_id=world.id).execute(context)
            note = (await inside.run(own, ["cat", "Documents/note.txt"], timeout_s=30))["stdout"]
            listed_items = await _items(deployed)
            await SnapshotEnvTaskStep(id="snap", version=None, env_id=office.id,
                                      snapshot_id=f"office-snap-{uid}").execute(context)
            snapshot = _snapshotted(context, "snap")
            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            after_reset = (await inside.run(own, ["sh", "-c", "test -e Documents/note.txt && echo kept || echo gone"],
                                            timeout_s=30))["stdout"].strip()
            reset_answers = await _answers(deployed, ["desktop_computer_screenshot"])
            routed_after_reset = await _reaches(own, "https://shop.test/api/health")
            emptied_items = await _items(deployed)
            # A reset empties the server too, so the next attempt loads again, every service and the computer.
            await LoadArtifactTaskStep(id="reseed", version=None, env_id=office.id, artifact_id=world.id).execute(context)
            reloaded = ((await inside.run(own, ["cat", "Documents/note.txt"], timeout_s=30))["stdout"],
                        await _items(deployed))

            await deployed.invoke(DISABLE_TOOL_EXTENSION_URI, DISABLE, {"role": DEFAULT_ROLE, "tools": [WILDCARD]})
            await deployed.invoke(ENABLE_TOOL_EXTENSION_URI, ENABLE, {"role": DEFAULT_ROLE,
                                                                      "tools": ["desktop_computer_screenshot"]})
            narrowed, narrowed_shot = await _tools(deployed)
            reattached = await Env.from_instance_id(deployed.instance_id)
            up = _containers(computer["sandbox_id"])

        finally:
            report = await teardown_run(context)

    assert not report.failed
    assert deployed.get_child_env_card("desktop") is not None and card["name"] == "desktop"
    assert "list_items" in listed and "desktop_computer_screenshot" in listed and shot
    assert {name for name in listed if "computer_" in name} <= {f"desktop_{tool}" for tool in COMPUTER_TOOLS}
    assert not any(UNPREFIXED.match(name) for name in listed)
    assert narrowed == {"desktop_computer_screenshot"} and narrowed_shot
    assert deployed.sandbox_ids["local"] == {"desktop": computer["sandbox_id"]} and up
    assert [env._deployed.sandbox_id for env in reattached.computer_envs] == [computer["sandbox_id"]]
    assert _containers(computer["sandbox_id"]) == []
    assert json.dumps(deployed.child_envs)  # the record stores as JSON
    assert note == "from the universe" and "snap-x" in listed_items
    assert {"items", "desktop"} <= set(snapshot) and snapshot["desktop"]["Documents/note.txt"] == b"from the universe"
    # The gateway states each child's environment_name, which lookups, servers and snapshots go by: the shop
    # website's card is named webitems.
    assert {name: stated.get(name) for name in ("items", "shop", "desktop")} == {
        "items": "items", "shop": "webitems", "desktop": "desktop"}, stated
    assert site is not None and site.url.startswith("/svc/shop/"), site
    assert {("items", True), ("shop", False)} <= servers and not any(name == "webitems" for name, _ in servers), servers
    assert "shop" in snapshot and "webitems" not in snapshot, sorted(snapshot)
    assert after_reset == "gone" and all(reset_answers.values()), reset_answers
    assert computer["metadata"]["computer"]["sites"]["names"]["routes"] == {"shop": ["shop.test"]}
    assert routed == routed_after_reset == "200"
    assert "snap-x" not in emptied_items, emptied_items
    assert reloaded[0] == "from the universe" and "snap-x" in reloaded[1], reloaded


@pytest.mark.asyncio
async def test_two_computers_keep_serving_through_restarts_and_a_failed_redeploy_and_close_ends_them(desktop, items, made,
                                                                                                     monkeypatch):
    """One deployment, to keep the tier's time: two computers side by side, the desktop's container killed and started
    again, the gateway restarted, then a redeploy of the same env object whose gateway fails, which leaves this
    deployment as it was; close() ends it all."""
    every = ["list_items", "desktop_computer_screenshot", "laptop_computer_screenshot"]
    with namespace_routing():
        laptop = ComputerEnv.put(id=f"{desktop.id}-laptop", computer=desktop.computer, driver=desktop.driver,
                                 environment_name="laptop")
        env = Env.get(multi_env(items, Env.get(desktop.id), laptop).id)
        try:
            deployed = await env.deploy(sandbox_type="local", ttl_seconds=900,
                                        child_envs={"desktop": CONTAINER, "laptop": CONTAINER})
            desktop_box = f"agent-{deployed.child_envs['desktop']['sandbox_id']}"
            both = await _answers(deployed, every)

            _docker("kill", desktop_box)
            killed = await _answers(deployed, every, until=False)
            _docker("start", desktop_box)
            started = await _answers(deployed, every)
            _docker("restart", _gateway(deployed, made))
            restarted = await _answers(deployed, every)

            deployed_before = _containers_made(made)
            real = EnvironmentGatewayProvider._compose_up

            async def no_gateway(self, sandbox, args=""):
                if not args:
                    raise RuntimeError("the gateway didn't start")
                return await real(self, sandbox, args)

            with monkeypatch.context() as patch:
                patch.setattr(EnvironmentGatewayProvider, "_compose_up", no_gateway)
                with pytest.raises(RuntimeError, match="the gateway didn't start"):
                    await env.deploy(sandbox_type="local", ttl_seconds=900,
                                     child_envs={"desktop": CONTAINER, "laptop": CONTAINER})
            after_failure = _containers_made(made)
            still = await _answers(deployed, every)
            kept = [computer._deployed.sandbox_id for computer in env.computer_envs]
        finally:
            await env.close()

    assert all(both.values()), both
    assert killed == {"list_items": True, "desktop_computer_screenshot": False, "laptop_computer_screenshot": True}
    assert all(started.values()) and all(restarted.values()), (started, restarted)
    assert after_failure == deployed_before and all(still.values()), (after_failure - deployed_before, still)
    assert kept == [deployed.child_envs[name]["sandbox_id"] for name in ("desktop", "laptop")]
    assert _containers_made(made) == set()


@pytest.mark.asyncio
async def test_a_computer_that_fails_mid_deploy_fails_the_deploy_and_leaves_nothing_up(desktop, items, made,
                                                                                        monkeypatch):
    async def refuse(self, deployed):
        raise RuntimeError("the computer's tools couldn't be applied")

    monkeypatch.setattr(ComputerEnv, "_give_tools", refuse)
    with namespace_routing():
        office = multi_env(items, Env.get(desktop.id))
        context = TaskStepContext()
        with pytest.raises(RuntimeError, match="the computer's tools couldn't be applied"):
            await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=600,
                                    child_envs={"desktop": CONTAINER}).execute(context)

    assert context.deployed_envs == [] and _containers_made(made) == set()


@pytest.mark.asyncio
async def test_a_computer_entry_the_multi_env_refuses_makes_no_sandbox(desktop, items, monkeypatch):
    made = []
    for name in ("create_sandbox", "create_vm", "create_container"):
        monkeypatch.setattr(LocalSandboxProvider, name, lambda *args, _name=name, **kwargs: made.append(_name))
    with namespace_routing():
        office = multi_env(items, Env.get(desktop.id))
        refused = [
            ({"desktop": {**CONTAINER, "env_params": {"site_routing": True}}},
             "child_envs.desktop: env_params.site_routing routes the multi env's websites, and it has none"),
            ({"desktop": {"tools": "standard"}},
             "child_envs.desktop: an entry is a table of sandbox_type, cpu, memory_mb, disk_size_gb, env_params, not tools"),
        ]
        for child_envs, problem in refused:
            step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=600,
                                     child_envs=child_envs)
            assert step.preflight() == [f"deploy_env 'office': {problem}"]
            with pytest.raises(ValueError, match=re.escape(problem)):
                await step.execute(TaskStepContext())

        monkeypatch.setattr(LocalSandboxProvider, "REQUEST_HEADERS", {"*.example": {"X-Token": "secret"}})
        [problem] = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                      ttl_seconds=600).preflight()

    assert "takes credentials on each request to a sandbox's URL" in problem
    assert "set child_envs.desktop.sandbox_type to a sandbox type whose URLs take none" in problem
    assert made == []
