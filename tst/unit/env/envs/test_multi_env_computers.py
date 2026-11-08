"""A multi env's computers: each deploys on a sandbox of its own, as it would alone, before the gateway starts, which then
fronts it by its name; its tools are served under that name, its record is kept in the multi env's, and closing the multi
env closes it."""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.cli.env.multi import multi
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, DeployedSandboxEnv
from agent_env.env.envs import _deployment
from agent_env.env.envs.computer import COMPUTER_ENV_TYPE
from agent_env.env.envs.multi_env import COMPUTERS_KEY, MultiEnv, computer_name_problems
from agent_env.env.gateway import ExternalChild
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_providers.env_provider import EnvironmentProvider
from agent_env.task.teardown import env_sandbox_ids
from tst.util.sandbox_providers import mock_provider

GATEWAY = "https://gw.example"


def _record(name: str) -> DeployedSandboxEnv:
    return DeployedSandboxEnv(env_id=f"{name}-computer", env_version=1, env_provider_type="computer", sandbox_id=f"sb-{name}",
                              sandbox_type="modal_vm", environment_card_url=f"https://{name}.example{WELL_KNOWN_PATH}",
                              mcp_url=f"https://{name}.example/mcp", control_token=f"token-{name}")


class FakeComputer:
    """A computer env as a multi env uses one: refused, deployed and closed, each call recorded in ``log``."""

    type = COMPUTER_ENV_TYPE

    def __init__(self, name: str, *, fails: Exception | None = None, refusal: str | None = None, log: list | None = None):
        self.id, self.version, self.environment_name = f"{name}-computer", 1, name
        self.fails, self.refusal, self.log = fails, refusal, [] if log is None else log
        self.deploys, self.refusals, self._deployed, self.closes = [], [], None, 0

    def deploy_refusal(self, **options):
        self.refusals.append(options)
        return self.refusal

    async def deploy(self, **options):
        self.deploys.append(options)
        await asyncio.sleep(0.01)
        if self.fails:
            raise self.fails
        self._deployed = _record(self.environment_name)
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
                    website_envs=[MagicMock(environment_name=name, domains=[]) for name in websites],
                    computer_envs=list(computers), **kwargs)


@pytest.fixture
def providers(monkeypatch):
    """Every sandbox provider a check or a deploy builds: one that takes no credentials on its URLs."""
    made = {}

    def build(sandbox_type=None):
        return made.setdefault(sandbox_type, mock_provider(REQUEST_HEADERS={}))

    from agent_env.env.envs import multi_env
    monkeypatch.setattr(multi_env, "build_sandbox_provider", build)
    monkeypatch.setattr(multi_env, "get_env_sandbox_provider", build)
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
    assert computer_name_problems(computers, **children) == problems


# --- child_envs: a computer's entry is what a deploy_env step takes to deploy it alone ---

def test_each_computer_is_checked_as_it_would_deploy_alone_on_its_entrys_sandbox_or_the_multi_envs(providers):
    desktop, laptop = FakeComputer("desktop"), FakeComputer("laptop", refusal="computer env 'laptop-computer': no arch")
    env = _multi(desktop, laptop, mcp=["slack"])

    refusal = env.deploy_refusal(sandbox_type="modal", child_envs={
        "desktop": {"sandbox_type": "modal_vm", "cpu": 4, "env_params": {"form": "disk"}}})

    assert desktop.refusals == [{"sandbox_type": "modal_vm", "cpu": 4, "env_params": {"form": "disk"}}]
    assert laptop.refusals == [{"sandbox_type": "modal"}]
    assert refusal == "computer env 'laptop-computer': no arch"
    laptop.refusal = None
    assert env.child_envs_refusal({"laptop": {"memory_mb": 8192}}) is None
    desktop.refusal = "computer env 'desktop-computer': no arch"
    assert env.child_envs_refusal({"desktop": {"cpu": 2}}) == "child_envs.desktop: computer env 'desktop-computer': no arch"


def test_saving_a_multi_env_leaves_how_each_computer_fits_a_sandbox_to_a_deploy_as_for_a_computer_saved_alone(providers):
    desktop = FakeComputer("desktop", refusal="computer env 'desktop-computer': no arch")
    assert _multi(desktop).save_refusal() is None and desktop.refusals == []


@pytest.mark.parametrize("child_envs, problem", [
    ({"desktop": {"env_params": {"site_routing": True}}}, "child_envs.desktop: env_params.site_routing: a computer in a multi env "
                                                          "doesn't route sites yet; deploy it beside the multi env to route them"),
    ({"desktop": {"env_params": {"sites": [{"name": "shop"}]}}}, "child_envs.desktop: env_params.sites: a computer in a multi "
                                                                 "env doesn't route sites yet; deploy it beside the multi env to "
                                                                 "route them"),
    ({"desktop": {"tools": "standard"}}, "child_envs.desktop: an entry is a table of sandbox_type, cpu, memory_mb, disk_size_gb, "
                                         "env_params, not tools"),
    ({"slack": {"cpu": 4}}, "child_envs.slack: slack runs on the multi env's own sandbox, which the step's own fields set, so "
                            "it takes no entry"),
], ids=["site-routing", "sites", "unknown-key", "mcp-server"])
def test_a_computer_entry_that_routes_sites_or_holds_what_a_step_doesnt_take_is_refused(providers, child_envs, problem):
    desktop = FakeComputer("desktop")
    assert _multi(desktop, mcp=["slack"]).child_envs_refusal(child_envs) == problem
    assert desktop.refusals == ([] if "desktop" in child_envs else [{"sandbox_type": None}])  # a refused entry deploys nothing


def test_a_computer_on_a_sandbox_provider_whose_urls_take_credentials_is_refused(providers):
    providers["scale"] = mock_provider(REQUEST_HEADERS={"*.example": {"x-token": "env:TOKEN"}})
    refusal = _multi(FakeComputer("desktop")).child_envs_refusal({"desktop": {"sandbox_type": "scale"}})
    assert refusal == ("child_envs.desktop: MagicMock takes credentials on each request to a sandbox's URL, which the multi "
                       "env's gateway doesn't send to a child env")


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
                                "sandbox_type": "modal", "cpu": 4, "env_params": {"form": "container"}}]
    assert laptop.deploys == [{"ttl_seconds": 600, "attribution": attribution, "tool_prefix": "laptop_", "reach": GATEWAY,
                               "sandbox_type": "modal_vm"}]
    assert gateway.started == [[ExternalChild("desktop", "https://desktop.example/mcp", "https://desktop.example"),
                                ExternalChild("laptop", "https://laptop.example/mcp", "https://laptop.example")]]
    assert deployed.metadata[COMPUTERS_KEY] == {name: dataclasses.asdict(_record(name)) for name in ("desktop", "laptop")}
    assert deployed.sandbox_ids == {"gateway_server": "gw", "modal_vm": {"desktop": "sb-desktop", "laptop": "sb-laptop"}}
    assert sorted(env_sandbox_ids(DeployedEnv.from_dict(dataclasses.asdict(deployed)))) == [
        ("gw", "modal_vm"), ("gw", "modal_vm"), ("sb-desktop", "modal_vm"), ("sb-laptop", "modal_vm")]


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
