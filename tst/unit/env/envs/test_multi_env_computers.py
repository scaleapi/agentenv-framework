"""A multi env's computers: each deploys on a sandbox of its own, as it would alone, before the gateway starts, which then
fronts it by its name; its tools are served under that name, its record is kept in the multi env's, and closing the multi
env closes it. Its website browser, which ``browser_envs`` names or the run's: an MCP server env runs among the
gateway's MCP servers, and a browser computer as one more computer, routing the synthetic sites, its network held to the
gateway."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.cli.env.multi import multi
from agent_env.config import get_config
from agent_env.config.runtime import WEBSITE_BROWSER_OVERRIDE, run_env_ids
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, DeployedSandboxEnv
from agent_env.env.envs import _deployment, _multi_computers
from agent_env.env.envs.computer import COMPUTER_ENV_TYPE
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv, WebsiteBrowser
from agent_env.env.gateway import ExternalChild
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.env.store import EnvInstanceStore
from agent_env.providers.env_providers import env_gateway_provider
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider, MCPServerConfig
from agent_env.providers.env_providers.env_provider import EnvironmentProvider
from agent_env.providers.env_state import LocalPostgresStateProvider
from agent_env.task.teardown import env_sandbox_ids
from agent_env.task_step.task_steps.browser.base import stored_browsers
from tst.util.sandbox_providers import mock_provider

GATEWAY = "https://gw.example"


#: A computer's card whose driver serves clock/v1's sync_time: it follows the gateway's clock.
CLOCK_CARD = {"capabilities": {"extensions": [
    {"uri": "urn:agentenv:clock/v1", "params": {"methods": {"sync_time": {"method": "POST", "endpoint": "/clock/sync"}}}}]}}


def _record(name: str, *, clock: bool = False) -> DeployedSandboxEnv:
    return DeployedSandboxEnv(env_id=f"{name}-computer", env_version=1, env_provider_type="computer", sandbox_id=f"sb-{name}",
                              sandbox_type="modal_vm", environment_card_url=f"https://{name}.example{WELL_KNOWN_PATH}",
                              mcp_url=f"https://{name}.example/mcp", control_token=f"token-{name}",
                              environment_card={"name": name, **CLOCK_CARD} if clock else None)


class FakeComputer:
    """A computer env as a multi env uses one: refused, deployed and closed, each call recorded in ``log``."""

    type = COMPUTER_ENV_TYPE

    def __init__(self, name: str, *, fails: Exception | None = None, refusal: str | None = None, log: list | None = None,
                 apps: tuple[str, ...] = (), resets: Exception | None = None, routes: Exception | None = None,
                 loads: Exception | None = None, clock: bool = False):
        self.id, self.version, self.environment_name = f"{name}-computer", 1, name
        self.fails, self.refusal, self.log = fails, refusal, [] if log is None else log
        self.deploys, self.refusals, self._deployed, self.closes = [], [], None, 0
        self.apps, self.reset_fails, self.route_fails, self.load_fails = apps, resets, routes, loads
        self.clock = clock

    def composed_apps(self):
        return [SimpleNamespace(environment_name=app, declaration=SimpleNamespace(browser=False)) for app in self.apps]

    def app_defaults(self):
        return {app: {} for app in self.apps}

    def holds(self, environment_name):
        return environment_name in (self.environment_name, *self.apps)

    async def load_environments(self, environments):
        inspect.signature(ComputerEnv.load_environments).bind(self, environments)
        await asyncio.sleep(0.01)
        self.log.append(("loaded", self.environment_name, [environment.environment_name for environment in environments]))
        if self.load_fails:
            raise self.load_fails

    async def load_environment_artifact(self, environment_artifact):
        inspect.signature(ComputerEnv.load_environment_artifact).bind(self, environment_artifact)
        self.log.append(("loaded", self.environment_name, [environment_artifact.environment_name]))

    async def route_sites(self, sites):
        inspect.signature(ComputerEnv.route_sites).bind(self, sites)
        self.log.append(("routed", self.environment_name, [site["name"] for site in sites]))
        if self.route_fails:
            raise self.route_fails

    async def follow_clock(self, gateway_url):
        inspect.signature(ComputerEnv.follow_clock).bind(self, gateway_url)
        self.log.append(("follows", self.environment_name, gateway_url))

    async def reset(self, deployed):
        inspect.signature(ComputerEnv.reset).bind(self, deployed)
        await asyncio.sleep(0.01)
        self.log.append(("reset", self.environment_name, deployed.sandbox_id))
        if self.reset_fails:
            raise self.reset_fails

    def deploy_refusal(self, **options):
        inspect.signature(ComputerEnv.deploy_refusal).bind(self, **options)  # only what the real one takes
        self.refusals.append(options)
        return self.refusal

    async def deploy(self, **options):
        inspect.signature(ComputerEnv.deploy).bind(self, **options)  # only what the real one takes
        self.deploys.append(options)
        await asyncio.sleep(0.01)
        if self.fails:
            raise self.fails
        self._deployed = _record(self.environment_name, clock=self.clock)
        self.log.append(("deployed", self.environment_name))
        return self._deployed

    async def close(self):
        self.closes += 1
        self.log.append(("close", self.environment_name))

    @classmethod
    async def from_deployed_env(cls, deployed):
        computer = cls(deployed.env_id.removesuffix("-computer"))
        computer._deployed = deployed
        return computer


def _multi(*computers, mcp=(), websites=(), **kwargs) -> MultiEnv:
    return MultiEnv(id="office", version=1, mcp_server_envs=[MagicMock(environment_name=name) for name in mcp],
                    website_envs=[MagicMock(id=f"{name}-site", environment_name=name, domains=[]) for name in websites],
                    computer_envs=list(computers), **kwargs)


@pytest.fixture
def providers(monkeypatch):
    """Every sandbox provider a check or a deploy builds: one that takes no credentials on its URLs."""
    made = {}

    def build(sandbox_type=None):
        return made.setdefault(sandbox_type, mock_provider(REQUEST_HEADERS={}))

    monkeypatch.setattr(_multi_computers, "build_sandbox_provider", build)
    monkeypatch.setattr(_multi_computers, "get_env_sandbox_provider", build)
    monkeypatch.setattr("agent_env.providers.build_sandbox_provider", build)
    monkeypatch.setattr("agent_env.providers.get_env_sandbox_provider", build)
    return made


# --- the field ---

def test_computer_envs_round_trip_and_are_written_only_when_set():
    assert "computer_envs" not in _multi(mcp=["slack"]).to_dict()
    doc = _multi(FakeComputer("desktop"), mcp=["slack"]).to_dict()
    assert doc["computer_envs"] == [{"id": "desktop-computer", "type": COMPUTER_ENV_TYPE, "version": 1}]
    with patch("agent_env.env.env.Env.get", side_effect=lambda id, version=None: FakeComputer(id.removesuffix("-computer"))):
        assert [env.environment_name for env in MultiEnv.from_dict(doc).computer_envs] == ["desktop"]


@pytest.mark.parametrize("name, problem", [
    ("desktop-number-1", "computer 'desktop-number-1' serves its tools under its name and '_': tool prefix "
                         "'desktop-number-1_' is 17 characters, and a tool prefix is at most 16"),
    ("my desk", "computer 'my desk' serves its tools under its name and '_': tool prefix 'my desk_' holds characters a "
                "tool's name can't"),
])
def test_a_computer_whose_name_cant_make_a_tool_prefix_is_refused(name, problem):
    with pytest.raises(ValueError, match="^multi env 'office': (.*)$") as raised:
        _multi(FakeComputer(name))
    assert str(raised.value) == f"multi env 'office': {problem}"


def test_a_computer_name_of_fifteen_characters_is_taken():
    assert _multi(FakeComputer("desktop-number1")).computer_envs[0].environment_name == "desktop-number1"


@pytest.mark.parametrize("computers, children, problems", [
    (["desktop", "desktop"], {}, ["'desktop' names 2 of its child envs, and the gateway knows a computer by its name alone"]),
    (["slack"], {"mcp_servers": ["slack"]},
     ["'slack' names 2 of its child envs, and the gateway knows a computer by its name alone"]),
    (["shop"], {"websites": ["shop"]}, ["'shop' names 2 of its child envs, and the gateway knows a computer by its name alone"]),
    (["mcp-slack"], {"mcp_servers": ["slack"]},
     ["'mcp-slack' is the gateway's route to MCP server 'slack', and the gateway knows a computer by its name alone"]),
    (["mcp-slack"], {"websites": ["slack"]}, []),
    (["desktop"], {"mcp_servers": ["shop"], "websites": ["shop"]}, []),  # an MCP server and a website may share a name
], ids=["two-computers", "a-server", "a-website", "a-servers-route", "a-websites-isnt", "server-and-website"])
def test_a_computer_needs_a_name_no_other_child_env_or_route_of_the_gateways_has(computers, children, problems):
    assert _multi_computers.name_problems(computers, **children) == problems


# --- child_envs: a computer's entry is what a deploy_env step takes to deploy it alone ---

def test_each_computer_is_checked_as_it_would_deploy_alone_on_its_entrys_sandbox_or_the_multi_envs(providers):
    desktop, laptop = FakeComputer("desktop"), FakeComputer("laptop", refusal="computer env 'laptop-computer': no arch")
    env = _multi(desktop, laptop, mcp=["slack"])

    refusal = env.deploy_refusal(sandbox_type="modal", child_envs={
        "desktop": {"sandbox_type": "modal_vm", "cpu": 4, "env_params": {"form": "disk"}}})

    assert desktop.refusals == [{"sandbox_type": "modal_vm", "cpu": 4, "env_params": {"form": "disk"}, "sites_later": False}]
    assert laptop.refusals == [{"sandbox_type": "modal", "sites_later": False}]
    assert refusal == "computer env 'laptop-computer': no arch"
    laptop.refusal = None
    assert env.child_envs_refusal({"laptop": {"memory_mb": 8192}}) is None
    desktop.refusal = "computer env 'desktop-computer': no arch"
    assert env.child_envs_refusal({"desktop": {"cpu": 2}}) == "child_envs.desktop: computer env 'desktop-computer': no arch"


def test_saving_a_multi_env_leaves_how_each_computer_fits_a_sandbox_to_a_deploy_as_for_a_computer_saved_alone(providers):
    desktop = FakeComputer("desktop", refusal="computer env 'desktop-computer': no arch")
    assert _multi(desktop).save_refusal() is None and desktop.refusals == []


@pytest.mark.parametrize("websites, child_envs, problem", [
    ((), {"desktop": {"env_params": {"site_routing": True}}},
     "child_envs.desktop: env_params.site_routing routes the multi env's websites, and it has none"),
    (("shop",), {"desktop": {"env_params": {"site_routing": ["shop-site", "blog-site"]}}},
     "child_envs.desktop: env_params.site_routing names 'blog-site', which isn't one of the multi env's websites ('shop-site')"),
    ((), {"desktop": {"tools": "standard"}}, "child_envs.desktop: an entry is a table of sandbox_type, cpu, memory_mb, "
                                             "disk_size_gb, env_params, not tools"),
    ((), {"slack": {"cpu": 4}}, "child_envs.slack: slack runs on the multi env's own sandbox, which the step's own fields set, "
                                "so it takes no entry"),
], ids=["site-routing-without-websites", "site-routing-naming-another", "unknown-key", "mcp-server"])
def test_a_computer_entry_whose_sites_the_multi_env_lacks_or_that_holds_what_a_step_doesnt_take_is_refused(
        providers, websites, child_envs, problem):
    desktop = FakeComputer("desktop")
    assert _multi(desktop, mcp=["slack"], websites=websites).child_envs_refusal(child_envs) == problem
    assert desktop.refusals == ([] if "desktop" in child_envs else [{"sandbox_type": None, "sites_later": False}])


def test_env_params_that_arent_a_table_are_left_to_the_computers_own_check_to_refuse(providers):
    desktop = FakeComputer("desktop", refusal="computer env 'desktop-computer': env_params is a table")
    refusal = _multi(desktop, websites=["shop"]).child_envs_refusal({"desktop": {"env_params": "invalid"}})
    assert refusal == "child_envs.desktop: computer env 'desktop-computer': env_params is a table"
    assert desktop.refusals == [{"sandbox_type": None, "env_params": "invalid", "sites_later": False}]


def test_a_site_routing_list_of_other_than_ids_is_left_to_the_computers_own_check_to_refuse(providers):
    desktop = FakeComputer("desktop", refusal="computer env 'desktop-computer': site_routing is a list of names")
    env_params = {"site_routing": [["shop-site"]]}
    refusal = _multi(desktop, websites=["shop"]).child_envs_refusal({"desktop": {"env_params": env_params}})
    assert refusal == "child_envs.desktop: computer env 'desktop-computer': site_routing is a list of names"
    assert desktop.refusals == [{"sandbox_type": None, "env_params": env_params, "sites_later": True}]


@pytest.mark.parametrize("env_params, later", [({"site_routing": True}, True), ({"site_routing": ["shop-site"]}, True),
                                                ({"site_routing": True, "sites": [{"name": "shop"}]}, False)],
                         ids=["all", "named", "its-own-sites"])
def test_a_computer_routing_the_multi_envs_websites_is_checked_as_routing_them_later(providers, env_params, later):
    desktop = FakeComputer("desktop")
    assert _multi(desktop, websites=["shop"]).child_envs_refusal({"desktop": {"env_params": env_params}}) is None
    assert desktop.refusals == [{"sandbox_type": None, "env_params": env_params, "sites_later": later}]


def test_a_computer_on_a_sandbox_provider_whose_urls_take_credentials_is_refused(providers):
    providers["scale"] = mock_provider(REQUEST_HEADERS={"*.example": {"x-token": "env:TOKEN"}})
    refusal = _multi(FakeComputer("desktop")).child_envs_refusal({"desktop": {"sandbox_type": "scale"}})
    assert refusal == ("child_envs.desktop: MagicMock takes credentials on each request to a sandbox's URL, which the multi "
                       "env's gateway doesn't send to a child env; set child_envs.desktop.sandbox_type to a sandbox type "
                       "whose URLs take none")


def test_only_a_provider_whose_deploy_takes_children_deploys_computers():
    class Plugin(EnvironmentProvider):
        type = "acme"

        async def deploy(self, env, sandbox_provider, **options):
            raise NotImplementedError

    assert _deployment.provider_refusal(Plugin, MultiEnv, computer_names=["desktop", "laptop"]) == (
        "deploys no computers, so not its 'desktop', 'laptop'")
    assert _deployment.provider_refusal(Plugin, MultiEnv) is None
    assert _deployment.provider_refusal(EnvironmentGatewayProvider, MultiEnv, computer_names=["desktop"]) is None


# --- deploy: every computer before the gateway, then recorded in the multi env's record ---

@pytest.fixture
def gateway(monkeypatch, local_stores):
    """The gateway provider's deploy, as the multi env drives it: it starts the env's children, given the gateway's address,
    as many times as ``tries`` says, and returns a record of its VM. Each close is logged, as each computer's is."""
    made = SimpleNamespace(tries=1, started=[], options=[], log=[], fails=None)

    async def deploy(self, env, sandbox_provider, *, children=None, **options):
        made.options.append({**options, "children": children})
        for _ in range(made.tries):
            made.started.append(await children(GATEWAY) if children else None)
        if made.fails:
            raise made.fails
        return DeployedGatewayEnv(env_id=env.id, env_version=env.version, gateway_url=GATEWAY, sandbox_id="gw",
                                  sandbox_type="modal_vm", sandbox_ids={"gateway_server": "gw"},
                                  environment_card_url=f"{GATEWAY}{WELL_KNOWN_PATH}", mcp_url=f"{GATEWAY}/mcp")

    async def close(self):
        made.log.append(("close", "gateway"))

    monkeypatch.setattr(EnvironmentGatewayProvider, "deploy", deploy)
    monkeypatch.setattr(EnvironmentGatewayProvider, "close", close)
    return made


@pytest.mark.asyncio
async def test_deploy_hands_the_gateway_each_computer_deployed_under_its_name_and_records_it(gateway, providers):
    desktop, laptop = FakeComputer("desktop"), FakeComputer("laptop")
    attribution = {"project_id": "6a6d2e30b2710a6cf93a6430"}

    deployed = await _multi(desktop, laptop, mcp=["slack"]).deploy(
        ttl_seconds=600, sandbox_type="modal_vm", attribution=attribution,
        child_envs={"desktop": {"sandbox_type": "modal", "cpu": 4, "env_params": {"form": "container"}}})

    assert desktop.deploys == [{"ttl_seconds": 600, "attribution": attribution, "tool_prefix": "desktop_", "reach": GATEWAY,
                                "sites_later": False, "sandbox_type": "modal", "cpu": 4, "env_params": {"form": "container"}}]
    assert laptop.deploys == [{"ttl_seconds": 600, "attribution": attribution, "tool_prefix": "laptop_", "reach": GATEWAY,
                               "sites_later": False, "sandbox_type": "modal_vm"}]
    assert gateway.started == [[ExternalChild("desktop", "https://desktop.example/mcp", "https://desktop.example"),
                                ExternalChild("laptop", "https://laptop.example/mcp", "https://laptop.example")]]
    assert deployed.child_envs == {name: dataclasses.asdict(_record(name)) for name in ("desktop", "laptop")}
    assert DeployedEnv.from_dict(dataclasses.asdict(deployed)).child_envs == deployed.child_envs
    assert deployed.sandbox_ids == {"gateway_server": "gw", "modal_vm": {"desktop": "sb-desktop", "laptop": "sb-laptop"}}
    assert sorted(env_sandbox_ids(DeployedEnv.from_dict(dataclasses.asdict(deployed)))) == [
        ("gw", "modal_vm"), ("gw", "modal_vm"), ("sb-desktop", "modal_vm"), ("sb-laptop", "modal_vm")]


def test_a_computer_whose_card_follows_a_clock_hands_the_gateway_its_control_token_as_its_read_token():
    from agent_env.env.envs._multi_computers import _clock_read_token

    assert (_clock_read_token(_record("desktop", clock=True)), _clock_read_token(_record("laptop"))) == (
        "token-desktop", None)


@pytest.mark.asyncio
async def test_once_the_gateway_is_up_each_computer_whose_card_follows_the_clock_follows_the_gateways(gateway, providers):
    log = gateway.log
    desktop, laptop = FakeComputer("desktop", log=log, clock=True), FakeComputer("laptop", log=log)

    deployed = await _multi(desktop, laptop).deploy()

    assert [entry for entry in log if entry[0] == "follows"] == [("follows", "desktop", GATEWAY)]
    assert set(deployed.child_envs) == {"desktop", "laptop"}


@pytest.mark.asyncio
async def test_a_multi_env_without_computers_gives_the_gateway_no_children(gateway, providers):
    await _multi(mcp=["slack"]).deploy()
    assert gateway.options[0]["children"] is None and gateway.started == [None]


@pytest.mark.asyncio
async def test_a_computer_that_fails_to_deploy_fails_the_deploy_once_every_computer_has_ended_and_closes_each(gateway, providers):
    log = gateway.log
    desktop, laptop = FakeComputer("desktop", fails=RuntimeError("no VM"), log=log), FakeComputer("laptop", log=log)

    with pytest.raises(RuntimeError, match="^no VM$"):
        await _multi(desktop, laptop).deploy()

    assert log[0] == ("deployed", "laptop")
    assert sorted(log[1:3]) == [("close", "desktop"), ("close", "laptop")] and log[3:] == [("close", "gateway")]


@pytest.mark.asyncio
async def test_a_gateway_deploy_tried_again_closes_the_computers_the_earlier_try_made_first(gateway, providers):
    gateway.tries = 2
    desktop = FakeComputer("desktop", log=gateway.log)

    await _multi(desktop).deploy()

    assert gateway.log == [("deployed", "desktop"), ("close", "desktop"), ("deployed", "desktop")]
    assert len(desktop.deploys) == 2


@pytest.mark.asyncio
async def test_a_redeploy_deploys_computers_of_its_own_so_a_failure_leaves_the_earlier_ones_and_close_ends_both(
        gateway, providers):
    env = _multi(FakeComputer("desktop"))
    await env.deploy()
    [earlier] = env.computer_envs
    fresh = []

    def get(id, version=None):
        fresh.append(FakeComputer(id.removesuffix("-computer")))
        return fresh[-1]

    gateway.fails = RuntimeError("no gateway")
    with patch("agent_env.env.env.Env.get", side_effect=get), pytest.raises(RuntimeError, match="^no gateway$"):
        await env.deploy()
    assert env.computer_envs == [earlier] and earlier.closes == 0 and [computer.closes for computer in fresh] == [1]

    gateway.fails = None
    with patch("agent_env.env.env.Env.get", side_effect=get):
        await env.deploy()
    assert env.computer_envs == [fresh[1]] and earlier.closes == 0

    await env.close()
    assert earlier.closes == 1 and [computer.closes for computer in fresh] == [1, 1]


@pytest.mark.asyncio
async def test_close_closes_each_computer_then_the_gateway_and_one_that_fails_to_close_is_logged(gateway, providers, caplog):
    desktop, laptop = FakeComputer("desktop", log=gateway.log), FakeComputer("laptop", log=gateway.log)
    env = _multi(desktop, laptop)
    await env.deploy()
    gateway.log.clear()
    laptop.close = AsyncMock(side_effect=RuntimeError("gone"))

    await env.close()

    assert gateway.log == [("close", "desktop"), ("close", "gateway")]
    assert "failed to close computer 'laptop': RuntimeError('gone')" in caplog.text


@pytest.mark.asyncio
async def test_from_deployed_env_reattaches_each_computer_from_the_record_its_own_deploy_wrote(gateway, providers, monkeypatch):
    deployed = await _multi(FakeComputer("desktop"), FakeComputer("laptop")).deploy()
    stored = DeployedEnv.from_dict(dataclasses.asdict(deployed))
    monkeypatch.setattr(EnvironmentGatewayProvider, "_reattach", AsyncMock())

    with patch("agent_env.env.env.Env.get", return_value=_multi(FakeComputer("desktop"), FakeComputer("laptop"))):
        env = await MultiEnv.from_deployed_env(stored)

    assert [(computer.environment_name, computer._deployed) for computer in env.computer_envs] == [
        ("desktop", _record("desktop")), ("laptop", _record("laptop"))]


# --- the CLI ---

def test_cli_put_takes_computers_and_refuses_another_type():
    computer = FakeComputer("desktop")
    with patch("agent_env.cli.env.multi.Env.get", return_value=computer), \
         patch("agent_env.cli.env.multi.detect_base_metadata", return_value={}), \
         patch("agent_env.cli.env.multi.MultiEnv.put", return_value=MagicMock(id="office", version=1)) as put:
        result = CliRunner().invoke(multi, ["put", "--id", "office", "--computer", "desktop-computer"])
    assert result.exit_code == 0, result.output
    assert put.call_args.kwargs["computer_envs"] == [computer]

    with patch("agent_env.cli.env.multi.Env.get", return_value=MagicMock(type="mcp_server", id="slack", version=1)):
        result = CliRunner().invoke(multi, ["put", "--id", "office", "--computer", "slack"])
    assert result.exit_code == 1 and "Error: slack is type 'mcp_server', expected 'computer'" in result.output


def test_cli_put_names_a_computer_name_the_multi_env_refuses():
    with patch("agent_env.cli.env.multi.Env.get", return_value=FakeComputer("desktop-number-one")), \
         patch("agent_env.cli.env.multi.detect_base_metadata", return_value={}):
        result = CliRunner().invoke(multi, ["put", "--id", "office", "--computer", "desktop-number-one-computer"])
    assert result.exit_code == 1 and "is 19 characters, and a tool prefix is at most 16" in result.output


# --- load and reset through the multi env ---

def _artifact(name: str) -> SimpleNamespace:
    return SimpleNamespace(environment_name=name)


@pytest.mark.parametrize("computers, others, problems", [
    ({"desktop": ["mail", "notes"]}, ["slack", "desktop"], []),
    ({"desktop": ["slack"]}, ["slack", "desktop"], [
        "app 'slack' of computer 'desktop' has the name of another of its child envs, and loads and snapshots find an "
        "environment by its name"]),
    ({"desktop": ["mail"], "laptop": ["mail"]}, ["desktop", "laptop"], [
        "app 'mail' of computer 'desktop' has the name of another of its child envs, and loads and snapshots find an "
        "environment by its name",
        "app 'mail' of computer 'laptop' has the name of another of its child envs, and loads and snapshots find an "
        "environment by its name"]),
    ({"desktop": ["laptop"]}, ["desktop", "laptop"], [
        "app 'laptop' of computer 'desktop' has the name of another of its child envs, and loads and snapshots find an "
        "environment by its name"]),
], ids=["own", "a-server", "two-computers", "another-computer"])
def test_a_computers_apps_names_are_no_other_child_envs(computers, others, problems):
    assert _multi_computers.app_name_problems(computers, others) == problems


def test_a_multi_env_refuses_a_computer_whose_app_another_child_env_names():
    with pytest.raises(ValueError, match="app 'slack' of computer 'desktop' has the name of another of its child envs"):
        _multi(FakeComputer("desktop", apps=("slack",)), mcp=["slack"])


@pytest.mark.asyncio
async def test_an_environment_artifact_named_for_a_computer_or_one_of_its_apps_loads_into_it():
    log = []
    env = _multi(FakeComputer("desktop", apps=("notes",), log=log), FakeComputer("laptop", log=log), mcp=["slack"])
    env._gateway_url = GATEWAY
    [slack] = env.mcp_server_envs
    slack.load_environment_artifact = AsyncMock()

    for name in ("notes", "desktop", "laptop", "slack"):
        await env.load_environment_artifact(_artifact(name))

    assert log == [("loaded", "desktop", ["notes"]), ("loaded", "desktop", ["desktop"]), ("loaded", "laptop", ["laptop"])]
    slack.load_environment_artifact.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_computer_loads_its_own_environment_before_its_apps():
    computer = SimpleNamespace(computer_name="desktop", _load=AsyncMock())
    await ComputerEnv.load_environments(computer, [_artifact("notes"), _artifact("desktop"), _artifact("mail")])
    assert [environment.environment_name for environment in computer._load.await_args.args[0]] == [
        "desktop", "notes", "mail"]


async def _load_universe(env: MultiEnv, log: list, *, restored: bool) -> MagicMock:
    """Load a universe of five environments into ``env``, restored from a clean snapshot or re-ingested, each
    service's load or the restore in ``log``; the universe, whose reads of its environments a test counts."""
    env._instance_id, env._sandbox = "inst", MagicMock(mode="vm", exec_script=AsyncMock(return_value="8\n"))
    env._env_provider = EnvironmentGatewayProvider()
    env._env_provider._state_provider = LocalPostgresStateProvider()
    env._load_from_snapshot = AsyncMock(side_effect=lambda snapshot: log.append(("restored",)))
    env.load_environment_artifact = AsyncMock(side_effect=lambda artifact: log.append(("service", artifact.environment_name)))
    universe = MagicMock(id="uni", version=1, **{"get_metadata.return_value": {}})
    universe.get_environment_artifacts.return_value = [_artifact(name) for name in ("notes", "a", "desktop", "laptop", "b")]
    snapshots = MagicMock(**{"get_clean.return_value": MagicMock() if restored else None})
    with patch("agent_env.env.store.get_env_instance_store", return_value=MagicMock(**{"get_loaded_environments.return_value": []})), \
         patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=snapshots), \
         patch("agent_env.env.snapshot_store.compute_env_fingerprint", return_value="fp"), \
         patch("agent_env.env.snapshot_store._check_changelog_empty", AsyncMock(return_value=True)), \
         patch("agent_env.env.store.update_env_instance_environment_universe", MagicMock()):
        await env.load_environment_universe_artifact(universe, snapshot_after_load=False)
    return universe


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True], ids=["re-ingested", "restored"])
async def test_a_universe_loads_each_computers_environments_after_the_services_even_after_a_restore(restored):
    log = []
    env = _multi(FakeComputer("desktop", apps=("notes",), log=log), FakeComputer("laptop", log=log))

    universe = await _load_universe(env, log, restored=restored)

    services = [("restored",)] if restored else [("service", "a"), ("service", "b")]
    assert log[:len(services)] == services
    assert sorted(log[len(services):]) == [("loaded", "desktop", ["notes", "desktop"]), ("loaded", "laptop", ["laptop"])]
    assert universe.get_environment_artifacts.call_count == 1  # the services and the computers load the same read


@pytest.mark.asyncio
@pytest.mark.parametrize("restored, reads", [(False, 1), (True, 0)], ids=["re-ingested", "restored"])
async def test_a_multi_env_without_computers_reads_the_universes_environments_only_to_re_ingest_them(restored, reads):
    log = []

    universe = await _load_universe(_multi(mcp=["a", "b"]), log, restored=restored)

    assert log == ([("restored",)] if restored else [("service", name) for name in ("notes", "a", "desktop", "laptop", "b")])
    assert universe.get_environment_artifacts.call_count == reads  # a restore needs none of them, as before computers


@pytest.mark.asyncio
async def test_each_computer_that_fails_to_load_is_named_once_every_computer_has_loaded():
    log = []
    env = _multi(FakeComputer("desktop", log=log, loads=RuntimeError("disk full")),
                 FakeComputer("laptop", log=log, loads=ValueError("bad path")), FakeComputer("kiosk", log=log))

    with pytest.raises(RuntimeError) as raised:
        await env._computers.load([_artifact(name) for name in ("desktop", "laptop", "kiosk")])

    assert str(raised.value) == ("multi env 'office': computer 'desktop' failed to load: RuntimeError('disk full'); "
                                 "computer 'laptop' failed to load: ValueError('bad path')")
    assert sorted(name for _, name, _ in log) == ["desktop", "kiosk", "laptop"]


@pytest.fixture
def server_resets(monkeypatch):
    """Each MCP server's and website's reset, as (name, mcp) in ``log``, attached to the multi env's record: False, as
    for one serving no reset, for a name in ``unserved``, and raising for one in ``failing``; the instance whose loads
    are forgotten in ``forgotten``, with how many servers had reset by then."""
    resets = SimpleNamespace(log=[], unserved=set(), failing={}, forgotten=[], attached=[])

    async def reset(env, *, mcp):
        inspect.signature(_deployment.reset_server_data).bind(env, mcp=mcp)
        await asyncio.sleep(0.01)
        resets.log.append((env.environment_name, mcp, env._deployed.instance_id))
        if env.environment_name in resets.failing:
            raise resets.failing[env.environment_name]
        return env.environment_name not in resets.unserved

    async def attached(cls, deployed):
        resets.attached.append(deployed.instance_id)
        env = _multi(mcp=["slack"], websites=["shop"])
        for child in [*env.mcp_server_envs, *env.website_envs]:
            child._deployed = deployed
        return env

    monkeypatch.setattr("agent_env.env.envs.multi_env.reset_server_data", reset)
    monkeypatch.setattr(MultiEnv, "_attached", classmethod(attached))
    monkeypatch.setattr(EnvInstanceStore, "clear_loaded_environments",
                        lambda store, instance_id: resets.forgotten.append((instance_id, len(resets.log))))
    return resets


async def _stored(*computers) -> DeployedEnv:
    """The record of a multi env with ``computers``, as reset_env reads it back."""
    deployed = await _multi(*computers, mcp=["slack"], websites=["shop"]).deploy()
    return DeployedEnv.from_dict({**dataclasses.asdict(deployed), "instance_id": "inst-1"})


@pytest.mark.asyncio
async def test_reset_resets_every_child_env_at_once_each_as_it_resets_alone_and_forgets_what_loads_put_in(
        gateway, providers, server_resets):
    stored, log = await _stored(FakeComputer("desktop"), FakeComputer("laptop")), []

    await _multi(FakeComputer("desktop", log=log), FakeComputer("laptop", log=log), mcp=["slack"],
                 websites=["shop"]).reset(stored)

    assert sorted(log) == [("reset", "desktop", "sb-desktop"), ("reset", "laptop", "sb-laptop")]
    assert sorted(server_resets.log) == [("shop", False, "inst-1"), ("slack", True, "inst-1")]
    assert server_resets.attached == ["inst-1"] and server_resets.forgotten == [("inst-1", 0)]  # before any reset


@pytest.mark.asyncio
async def test_a_multi_env_of_computers_alone_resets_them_without_reattaching_its_gateway(gateway, providers,
                                                                                          server_resets):
    deployed, log = await _multi(FakeComputer("desktop")).deploy(), []

    await _multi(FakeComputer("desktop", log=log)).reset(DeployedEnv.from_dict(dataclasses.asdict(deployed)))

    assert log == [("reset", "desktop", "sb-desktop")] and server_resets.attached == server_resets.log == []


@pytest.mark.asyncio
async def test_a_server_serving_no_reset_stays_as_it_is_and_the_rest_reset(gateway, providers, server_resets, caplog):
    stored, log = await _stored(FakeComputer("desktop")), []
    server_resets.unserved.add("shop")

    with caplog.at_level("INFO"):
        await _multi(FakeComputer("desktop", log=log), mcp=["slack"], websites=["shop"]).reset(stored)

    assert log == [("reset", "desktop", "sb-desktop")] and len(server_resets.log) == 2
    assert "multi env 'office': website 'shop' serves no reset and takes no load, so it stays as it is" in caplog.text


@pytest.mark.asyncio
async def test_each_child_that_fails_to_reset_is_named_once_every_child_env_has_reset(gateway, providers, server_resets):
    stored, log = await _stored(FakeComputer("desktop"), FakeComputer("laptop")), []
    server_resets.failing["slack"] = RuntimeError("data/reset timed out")
    env = _multi(FakeComputer("desktop", log=log, resets=RuntimeError("no reboot")), FakeComputer("laptop", log=log),
                 mcp=["slack"], websites=["shop"])

    with pytest.raises(RuntimeError) as raised:
        await env.reset(stored)

    assert str(raised.value) == ("multi env 'office': computer 'desktop' failed to reset: RuntimeError('no reboot'); "
                                 "MCP server 'slack' failed to reset: RuntimeError('data/reset timed out')")
    assert sorted(log) == [("reset", "desktop", "sb-desktop"), ("reset", "laptop", "sb-laptop")]
    assert len(server_resets.log) == 2 and server_resets.forgotten == [("inst-1", 0)]


@pytest.mark.asyncio
async def test_a_reset_that_cannot_forget_what_loads_put_in_empties_nothing(gateway, providers, server_resets,
                                                                           monkeypatch):
    stored, log = await _stored(FakeComputer("desktop")), []
    monkeypatch.setattr(EnvInstanceStore, "clear_loaded_environments", MagicMock(side_effect=OSError("store down")))

    with pytest.raises(OSError, match="store down"):
        await _multi(FakeComputer("desktop", log=log), mcp=["slack"], websites=["shop"]).reset(stored)

    assert log == server_resets.log == []  # a resumed load would skip a service a reset had emptied


# --- routing the multi env's websites ---

@pytest.mark.asyncio
async def test_a_computer_routing_sites_boots_held_and_routes_the_multi_envs_websites_once_the_gateway_is_up(
        gateway, providers, monkeypatch):
    log = gateway.log
    found = [{"name": "shop", "url": f"{GATEWAY}/website/shop/"}, {"name": "blog", "url": f"{GATEWAY}/website/blog/"}]
    read = AsyncMock(side_effect=lambda records, services=None: log.append(("sites read", records[0].env_id, services))
                     or [site for site in found if services is None or site["name"] in services])
    ready = AsyncMock()
    monkeypatch.setattr(_multi_computers.sites, "deployed_sites", read)
    monkeypatch.setattr(_multi_computers.sites, "require_ready", ready)
    desktop, laptop, kiosk = (FakeComputer(name, log=log) for name in ("desktop", "laptop", "kiosk"))

    deployed = await _multi(desktop, laptop, kiosk, websites=["shop", "blog"]).deploy(child_envs={
        "desktop": {"env_params": {"site_routing": True}}, "laptop": {"env_params": {"site_routing": ["blog-site"]}}})

    assert [deploy["sites_later"] for deploy in (desktop.deploys + laptop.deploys + kiosk.deploys)] == [True, True, False]
    assert log[3:5] == [("sites read", "office", None), ("sites read", "office", ["blog"])]  # by environment_name
    assert [r.env_id for r in ready.await_args.args[0]] == ["office"]
    assert sorted(log[5:]) == [("routed", "desktop", ["shop", "blog"]), ("routed", "laptop", ["blog"])]
    assert set(deployed.child_envs) == {"desktop", "laptop", "kiosk"}


@pytest.mark.asyncio
async def test_a_routing_computers_start_tabs_are_checked_against_the_names_its_sites_cards_declare_before_it_routes(
        gateway, providers, monkeypatch):
    monkeypatch.setattr(_multi_computers.sites, "deployed_sites", AsyncMock(return_value=[
        {"name": "shop", "url": "u", "domains": ["shop.test"]}, {"name": "blog", "url": "u"}]))
    monkeypatch.setattr(_multi_computers.sites, "require_ready", AsyncMock())

    def office(urls):
        desktop = FakeComputer("desktop", log=gateway.log, apps=("browser",))
        desktop.composed_apps = lambda: [SimpleNamespace(environment_name="browser",
                                                         declaration=SimpleNamespace(browser=True))]
        websites = [MagicMock(id=f"{name}-site", environment_name=name, domains=[], live=False, allow_hosts=[])
                    for name in ("shop", "blog")]  # names the save can't know: their cards declare them
        return MultiEnv(id="office", version=1, mcp_server_envs=[], website_envs=websites, computer_envs=[desktop]), {
            "desktop": {"env_params": {"site_routing": True, "app_settings": {"browser": {"start_urls": urls}}}}}

    env, child_envs = office(["https://shop.test/", "http://blog.localhost/"])
    await env.deploy(child_envs=child_envs)
    routed = gateway.log[-1]
    env, child_envs = office(["https://example.com/"])
    with pytest.raises(ValueError, match=re.escape("multi env 'office': computer 'desktop': browser's start tab "
                                                   "https://example.com/ names example.com, which none of the multi")):
        await env.deploy(child_envs=child_envs)

    assert routed == ("routed", "desktop", ["shop", "blog"])
    assert gateway.log[-2:] == [("close", "desktop"), ("close", "gateway")]


@pytest.mark.asyncio
async def test_a_computer_that_fails_to_route_its_sites_fails_the_deploy_and_closes_everything(gateway, providers,
                                                                                               monkeypatch):
    monkeypatch.setattr(_multi_computers.sites, "deployed_sites", AsyncMock(return_value=[{"name": "shop", "url": "u"}]))
    monkeypatch.setattr(_multi_computers.sites, "require_ready", AsyncMock())
    desktop = FakeComputer("desktop", log=gateway.log, routes=RuntimeError("shop.test didn't load"))

    with pytest.raises(RuntimeError, match=re.escape(
            "multi env 'office': computer 'desktop' failed to route its sites: RuntimeError(\"shop.test didn't load\")")):
        await _multi(desktop, websites=["shop"]).deploy(child_envs={"desktop": {"env_params": {"site_routing": True}}})

    assert gateway.log[-2:] == [("close", "desktop"), ("close", "gateway")]


class FakeBrowser(FakeComputer, ComputerEnv):
    """A browser computer as a multi env uses one: chromium deployed alone, its one app itself."""

    def __init__(self, name: str = "chromium", *, browser: bool = True, **kwargs):
        super().__init__(name, **kwargs)
        self.declaration = SimpleNamespace(browser=browser)

    def composed_apps(self):
        return [self]

    def app_defaults(self):
        return {self.environment_name: {}}

    def serves_website_browser(self):
        return self.declaration.browser


def _image(id: str) -> DockerImageArtifact:
    return DockerImageArtifact(id=id, version=1, description=id, image_name=f"registry.example/{id}:v1",
                               tar_gz_s3_url=f"s3://bucket/{id}.tar.gz")


def _playwright() -> MCPServerEnv:
    return MCPServerEnv("playwright-mcp", 1, docker_image_artifact=_image("playwright-image"),
                        environment_name="playwright-mcp")


def _site(name: str, live: bool = False) -> MagicMock:
    return MagicMock(id=f"{name}-site", environment_name=name, domains=[f"{name}.test"], live=live,
                     allow_hosts=[f"api.{name}.test"] if live else [])


def _browsing(*computers, browser_envs=None, websites=("shop",), live=()) -> MultiEnv:
    return MultiEnv(id="office", version=1, mcp_server_envs=[],
                    website_envs=[*(_site(name) for name in websites), *(_site(name, live=True) for name in live)],
                    computer_envs=list(computers), browser_envs=browser_envs)


@pytest.fixture
def routes(monkeypatch):
    """The sites the gateway's record names, by environment_name, and their readiness, as a deploy reads them."""
    monkeypatch.setattr(_multi_computers.sites, "deployed_sites",
                        AsyncMock(side_effect=lambda records, services=None: [{"name": "shop", "domains": ["shop.test"],
                                                                               "url": f"{GATEWAY}/website/shop/"}]))
    monkeypatch.setattr(_multi_computers.sites, "require_ready", AsyncMock())


# --- its website browser: the field ---

def test_browser_envs_is_written_only_when_set_and_none_and_empty_read_back_as_they_were():
    chromium = FakeBrowser()
    assert "browser_envs" not in _browsing().to_dict()
    assert _browsing(browser_envs=[]).to_dict()["browser_envs"] == []
    doc = _browsing(browser_envs=[chromium]).to_dict()
    assert doc["browser_envs"] == [{"id": "chromium-computer", "type": "computer", "version": 1}]

    with patch("agent_env.env.env.Env.get", side_effect=lambda id, version=None: _site("shop") if id == "shop-site"
               else FakeBrowser()):
        assert MultiEnv.from_dict(_browsing().to_dict()).browser_envs is None
        assert MultiEnv.from_dict(_browsing(browser_envs=[]).to_dict()).browser_envs == []
        [read] = MultiEnv.from_dict(doc).browser_envs
    assert read.id == "chromium-computer"


def test_a_browser_computer_is_kept_out_of_computer_envs():
    office = _browsing(FakeComputer("desktop"), browser_envs=[FakeBrowser()])
    assert [env.environment_name for env in office.computer_envs] == ["desktop"]
    assert office.to_dict()["computer_envs"] == [{"id": "desktop-computer", "type": "computer", "version": 1}]


@pytest.mark.parametrize("browsers, problem", [
    (lambda: [FakeBrowser(), _playwright()], "browser_envs names 2 envs, and a multi env runs one website browser"),
    (lambda: [FakeBrowser("desktop", browser=False)],
     "browser_envs[0]: computer env 'desktop-computer' has no website browser among its apps"),
    (lambda: [_site("blog")], "browser_envs[0]: 'blog-site' is a"),
], ids=["two", "a-computer-without-a-browser", "not-a-browser-env"])
def test_browser_envs_takes_one_website_browser_env(browsers, problem):
    with pytest.raises(ValueError, match=f"^multi env 'office': {re.escape(problem)}"):
        _browsing(browser_envs=browsers())


@pytest.mark.parametrize("browser", [FakeBrowser, _playwright], ids=["browser-computer", "mcp-server"])
def test_browser_envs_takes_a_browser_computer_or_an_mcp_server_serving_one(browser):
    assert _browsing(browser_envs=[browser()]).website_browser().env.id in ("chromium-computer", "playwright-mcp")


def test_a_browser_computer_needs_a_name_no_other_computer_has():
    with pytest.raises(ValueError, match="'chromium' names 2 of its child envs"):
        _browsing(FakeComputer("chromium"), browser_envs=[FakeBrowser()])


def test_an_mcp_server_browser_needs_a_name_no_computer_or_other_mcp_server_has():
    with pytest.raises(ValueError, match="'playwright-mcp' names 2 of its child envs"):
        _browsing(FakeComputer("playwright-mcp"), browser_envs=[_playwright()])
    with pytest.raises(ValueError, match="website browser 'playwright-mcp' has the name of another of its MCP servers"):
        MultiEnv(id="office", version=1, mcp_server_envs=[_playwright()], browser_envs=[_playwright()])


def test_the_runs_mcp_server_browser_is_refused_a_name_a_computer_has(providers):
    with patch("agent_env.env.env.Env.get", return_value=_playwright()):
        assert "'playwright-mcp' names 2 of its child envs" in _browsing(
            FakeComputer("playwright-mcp")).child_envs_refusal(None)


# --- its website browser: which one ---

def test_the_browser_is_the_one_named_none_for_an_empty_list_and_else_the_runs_when_there_are_websites(local_stores):
    chromium, playwright = FakeBrowser(), _playwright()
    assert _browsing(browser_envs=[chromium]).website_browser() == WebsiteBrowser(computer=chromium)
    assert _browsing(browser_envs=[playwright]).website_browser() == WebsiteBrowser(server=playwright)
    assert _browsing(browser_envs=[]).website_browser() == WebsiteBrowser()
    assert _browsing(websites=()).website_browser() == WebsiteBrowser()
    with patch("agent_env.env.env.Env.get", return_value=chromium) as get, \
            run_env_ids({WEBSITE_BROWSER_OVERRIDE: "chromium-computer"}):
        assert _browsing().website_browser() == WebsiteBrowser(computer=chromium)
    get.assert_called_with("chromium-computer")


def test_without_a_run_override_a_multi_env_with_websites_gets_the_configs_default_browser(local_stores):
    chromium, config = FakeBrowser(), get_config()
    config.default_website_browser_env_id = "chromium-computer"
    try:
        with patch("agent_env.env.env.Env.get", return_value=chromium) as get:
            assert _browsing().website_browser() == WebsiteBrowser(computer=chromium)
            assert _browsing(browser_envs=[]).website_browser() == WebsiteBrowser()
        get.assert_called_once_with("chromium-computer")
    finally:
        config.default_website_browser_env_id = None


def test_a_runs_browser_the_store_doesnt_hold_yet_is_left_for_the_gateway_to_look_up(local_stores):
    office = _browsing()
    assert office.website_browser() == WebsiteBrowser(runs=True)
    assert env_gateway_provider._gateway_topology(office).website_browser is env_gateway_provider._THE_RUNS


@pytest.mark.parametrize("browser_envs, runs, expected", [
    (None, _playwright, "playwright-mcp"),
    (None, FakeBrowser, None),
    ([], _playwright, None),
    ("chromium", _playwright, None),
    ("playwright", FakeBrowser, "playwright-mcp"),
], ids=["runs-mcp-server", "runs-computer", "none", "named-computer", "named-mcp-server"])
def test_the_gateway_runs_an_mcp_server_browser_and_leaves_a_browser_computer_to_its_own_sandbox(browser_envs, runs,
                                                                                                 expected):
    named = {"chromium": [FakeBrowser()], "playwright": [_playwright()]}[browser_envs] \
        if isinstance(browser_envs, str) else browser_envs
    office = _browsing(browser_envs=named)
    with patch("agent_env.env.env.Env.get", return_value=runs()), \
            patch.object(env_gateway_provider, "website_config", return_value=MagicMock()):
        browser = env_gateway_provider._gateway_topology(office).website_browser
    assert getattr(browser, "id", None) == expected


@pytest.mark.parametrize("browser_envs, runs, by_name", [
    (None, _playwright, False),
    (None, FakeBrowser, True),
    ([], _playwright, False),
    ("chromium", _playwright, True),
    ("playwright", FakeBrowser, False),
], ids=["runs-mcp-server", "runs-computer", "none", "named-computer", "named-mcp-server"])
def test_the_gateway_lists_the_sites_by_name_only_for_a_browser_computer(browser_envs, runs, by_name):
    named = {"chromium": [FakeBrowser()], "playwright": [_playwright()]}[browser_envs] \
        if isinstance(browser_envs, str) else browser_envs
    office = _browsing(browser_envs=named)
    with patch("agent_env.env.env.Env.get", return_value=runs()), \
            patch.object(env_gateway_provider, "website_config", return_value=MagicMock()) as config:
        env_gateway_provider._gateway_topology(office)
    assert config.call_args_list and {call.kwargs["listed_by_name"] for call in config.call_args_list} == {by_name}


def test_a_lone_website_with_a_browser_computer_as_the_runs_browser_is_refused_naming_a_multi_env():
    with patch("agent_env.env.env.Env.get", return_value=FakeBrowser()), \
            pytest.raises(ValueError, match="'chromium-computer' is a computer env, which runs on a sandbox of its own "
                                            "beside a multi env's websites"):
        env_gateway_provider.run_website_browser()


def test_a_browser_mcp_server_joins_the_gateways_mcp_servers_on_the_sites_network():
    servers, images = env_gateway_provider._with_browser([], [], _playwright())
    assert servers == [MCPServerConfig(image="registry.example/playwright-image:v1", environment_name="playwright-mcp",
                                       networks=(env_gateway_provider._SITES_NETWORK,))]
    assert [image.id for image in images] == ["playwright-image"]


@pytest.mark.asyncio
async def test_a_named_browser_mcp_server_runs_in_container_mode_too():
    gp, provider = EnvironmentGatewayProvider(), MagicMock(links=None)
    with patch.object(gp, "in_vm", return_value=False), patch.object(gp, "sandbox_problem", return_value=None), \
            patch.object(gp, "_deploy_via_containers", AsyncMock()) as containers:
        await gp._deploy_gateway(provider, state_instance=None, mcp_servers=[],
                                 mcp_server_images=[], website_browser=_playwright())
    assert [server.environment_name for server in containers.await_args.args[1]] == ["playwright-mcp"]


# --- its website browser: a browser computer's deploy ---

@pytest.mark.asyncio
async def test_a_browser_computer_deploys_as_a_computer_child_routing_the_sites_held_to_the_gateway(gateway, providers,
                                                                                                     routes):
    chromium, desktop = FakeBrowser(), FakeComputer("desktop")

    deployed = await _browsing(desktop, browser_envs=[chromium], live=("github",)).deploy(
        sandbox_type="modal_vm", child_envs={"chromium": {"sandbox_type": "modal", "cpu": 2, "memory_mb": 4096}})

    assert chromium.deploys == [{"ttl_seconds": 10800, "attribution": None, "tool_prefix": "chromium_",
                                 "reach": GATEWAY, "sites_later": True,
                                 "live_hosts": ("github.test", "api.github.test"),
                                 "sandbox_type": "modal", "cpu": 2, "memory_mb": 4096,
                                 "env_params": {"site_routing": True}}]
    assert desktop.deploys[0]["sites_later"] is False
    assert gateway.started == [[ExternalChild("desktop", "https://desktop.example/mcp", "https://desktop.example"),
                                ExternalChild("chromium", "https://chromium.example/mcp", "https://chromium.example")]]
    assert set(deployed.child_envs) == {"desktop", "chromium"}
    assert chromium.log[-1] == ("routed", "chromium", ["shop"])


@pytest.mark.asyncio
async def test_the_runs_browser_computer_takes_the_same_path(gateway, providers, routes):
    chromium = FakeBrowser()
    with patch("agent_env.env.env.Env.get", return_value=chromium), \
            run_env_ids({WEBSITE_BROWSER_OVERRIDE: "chromium-computer"}):
        deployed = await _browsing().deploy()
    assert chromium.deploys[0]["tool_prefix"] == "chromium_" and set(deployed.child_envs) == {"chromium"}


@pytest.mark.asyncio
async def test_a_browser_entry_saying_whether_it_routes_the_sites_is_kept(gateway, providers):
    chromium = FakeBrowser()
    await _browsing(browser_envs=[chromium]).deploy(child_envs={"chromium": {"env_params": {"site_routing": False}}})
    assert chromium.deploys[0]["env_params"] == {"site_routing": False} and chromium.deploys[0]["sites_later"] is False


@pytest.mark.asyncio
async def test_a_browser_beside_live_sites_alone_routes_nothing(gateway, providers):
    chromium = FakeBrowser()
    await _browsing(browser_envs=[chromium], websites=(), live=("github",)).deploy()
    assert "env_params" not in chromium.deploys[0] and chromium.deploys[0]["live_hosts"] == ("github.test",
                                                                                            "api.github.test")


@pytest.mark.asyncio
async def test_no_browser_deploys_no_child_and_gives_the_gateway_none(gateway, providers):
    office = _browsing(browser_envs=[])
    await office.deploy()
    assert gateway.options[0]["children"] is None
    assert env_gateway_provider._gateway_topology(_browsing(browser_envs=[], websites=())).website_browser is None


def test_a_browser_computers_entry_is_checked_as_any_computers_and_an_mcp_server_browsers_is_refused(providers):
    office = _browsing(browser_envs=[FakeBrowser(refusal="no native build")])
    assert office.child_envs_refusal({"chromium": {"cpu": 2}}) == "child_envs.chromium: no native build"
    assert office.child_envs_refusal({"chromium": {"ttl_seconds": 60}}).startswith("child_envs.chromium: an entry is")
    assert _browsing(browser_envs=[_playwright()]).child_envs_refusal({"playwright-mcp": {"cpu": 2}}).startswith(
        "child_envs.playwright-mcp: env 'office' has no child env named 'playwright-mcp'")


def test_the_runs_browser_computer_is_refused_a_name_a_computer_has(providers):
    with patch("agent_env.env.env.Env.get", return_value=FakeBrowser()), \
            run_env_ids({WEBSITE_BROWSER_OVERRIDE: "chromium-computer"}):
        assert "'chromium' names 2 of its child envs" in _browsing(FakeComputer("chromium")).child_envs_refusal(None)


# --- its website browser: after the deploy ---

@pytest.mark.asyncio
async def test_the_browser_is_reattached_reset_and_closed_from_its_own_record(gateway, providers, routes, monkeypatch):
    deployed = await _browsing(FakeComputer("desktop"), browser_envs=[FakeBrowser()]).deploy()
    stored = DeployedEnv.from_dict({**dataclasses.asdict(deployed), "instance_id": "inst-1"})
    monkeypatch.setattr(EnvironmentGatewayProvider, "_reattach", AsyncMock())
    stored_office = _browsing(FakeComputer("desktop"), browser_envs=[FakeBrowser()])

    with patch("agent_env.env.env.Env.get", side_effect=lambda id, version=None: stored_office if id == "office"
               else FakeBrowser()), \
            patch.object(FakeBrowser, "from_deployed_env",
                         classmethod(lambda cls, record: _reattached(cls, record)), create=True):
        env = await MultiEnv.from_deployed_env(stored)
    assert env._computers.browser._deployed == _record("chromium")

    log = []
    env._computers.browser.log = log
    monkeypatch.setattr(MultiEnv, "_attached", classmethod(AsyncMock(return_value=_browsing(websites=()))))
    await env.reset(stored)
    assert ("reset", "chromium", "sb-chromium") in log
    await env.close()
    assert ("close", "chromium") in log


@pytest.mark.asyncio
async def test_a_reset_of_the_multi_env_as_stored_resets_the_browser_its_record_holds(gateway, providers, routes,
                                                                                    monkeypatch):
    deployed = await _browsing(FakeComputer("desktop"), browser_envs=[FakeBrowser()]).deploy()
    stored = DeployedEnv.from_dict({**dataclasses.asdict(deployed), "instance_id": "inst-1"})
    log = []
    monkeypatch.setattr(MultiEnv, "_attached", classmethod(AsyncMock(return_value=_browsing(websites=()))))

    with patch("agent_env.env.env.Env.get", return_value=FakeBrowser(log=log)):
        await _browsing(FakeComputer("desktop", log=log), browser_envs=[FakeBrowser()]).reset(stored)

    assert sorted(log) == [("reset", "chromium", "sb-chromium"), ("reset", "desktop", "sb-desktop")]


async def _reattached(cls, record):
    browser = cls()
    browser._deployed = record
    return browser


@pytest.mark.asyncio
async def test_a_failed_redeploy_closes_only_its_own_browser_and_leaves_the_earlier_one_for_close(gateway, providers,
                                                                                                routes):
    earlier = FakeBrowser()
    office = _browsing(browser_envs=[earlier])
    await office.deploy()
    fresh = []

    def get(id, version=None):
        fresh.append(FakeBrowser())
        return fresh[-1]

    gateway.fails = RuntimeError("no gateway")
    with patch("agent_env.env.env.Env.get", side_effect=get), pytest.raises(RuntimeError, match="^no gateway$"):
        await office.deploy()
    assert office._computers.browser is earlier and earlier.closes == 0 and [b.closes for b in fresh] == [1]

    gateway.fails = None
    with patch("agent_env.env.env.Env.get", side_effect=get):
        await office.deploy()
    assert office._computers.browser is fresh[1] and earlier.closes == 0

    await office.close()
    assert earlier.closes == 1 and [b.closes for b in fresh] == [1, 1]


@pytest.mark.asyncio
async def test_a_redeploy_keeps_the_earlier_browser_for_close(gateway, providers, routes):
    chromium = FakeBrowser()
    office = _browsing(browser_envs=[chromium])
    await office.deploy()
    with patch("agent_env.env.env.Env.get", return_value=FakeBrowser(log=chromium.log)):
        await office.deploy()
    assert office._computers.browser is not chromium
    await office.close()
    assert chromium.log.count(("close", "chromium")) == 2  # the earlier deployment's browser and the later one's


# --- its website browser: elsewhere ---

def test_a_named_browser_computer_counts_among_a_multi_envs_browsers():
    assert stored_browsers(_browsing(FakeComputer("desktop"), browser_envs=[FakeBrowser()])) == ["chromium"]
    assert stored_browsers(_browsing(browser_envs=[_playwright()])) == []


def test_the_runs_browser_computer_counts_among_a_multi_envs_browsers_when_it_names_none():
    with patch("agent_env.env.env.Env.get", return_value=FakeBrowser()), \
            run_env_ids({WEBSITE_BROWSER_OVERRIDE: "chromium-computer"}):
        assert stored_browsers(_browsing()) == ["chromium"]
    with patch("agent_env.env.env.Env.get", return_value=_playwright()):
        assert stored_browsers(_browsing()) == []


def test_cli_put_takes_a_browser_or_none():
    chromium = FakeBrowser()
    slack = MCPServerEnv("slack", 1, docker_image_artifact=_image("slack-image"), environment_name="slack")
    for args, expected in ((["--browser", "chromium-computer"], [chromium]), (["--no-browser"], []), ([], None)):
        with patch("agent_env.cli.env.multi.Env.get",
                   side_effect=lambda id, version=None: chromium if id == "chromium-computer" else slack), \
                patch("agent_env.cli.env.multi.detect_base_metadata", return_value={}), \
                patch("agent_env.cli.env.multi.MultiEnv.put", return_value=MagicMock(id="office", version=1)) as put:
            result = CliRunner().invoke(multi, ["put", "--id", "office", "--mcp-server", "slack", *args])
        assert result.exit_code == 0, result.output
        assert put.call_args.kwargs["browser_envs"] == expected

    result = CliRunner().invoke(multi, ["put", "--id", "office", "--mcp-server", "slack", "--browser", "b",
                                        "--no-browser"])
    assert result.exit_code == 1 and "--browser and --no-browser can't both be given" in result.output
