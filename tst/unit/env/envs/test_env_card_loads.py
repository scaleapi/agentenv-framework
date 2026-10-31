"""Loads find each child env's data plane in the stored env card.

A child env the card lists is loaded through its v1 data plane at the card's own `url`, with no live
card read. One the card doesn't list, a server that serves no card, takes the legacy REST path, which takes one file
and refuses an environment of several before sending anything; a website stages several in a folder of the load's own,
which it removes after, and sends each with its path. Without
a stored card (no deployed record, or a record with none), the live probe decides. Every env keeps the
record it deploys or reattaches to, and MultiEnv hands it to every child env except a name an MCP
server and a website share.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, DeployedSandboxEnv
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider

_GW = "https://gw.example"
# The card's address differs from gateway_url here, so a load that reads the card is told apart from one that builds /svc/... itself.
_CARD = "https://sandbox.example/sandbox/sb-1-18765"


@pytest.fixture
def sent(monkeypatch) -> list[httpx.Request]:
    """Every request a load sends; JSON-RPC calls get an empty result, anything else an empty 200."""
    requests, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/agentenv"):
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return requests


@pytest.mark.asyncio
async def test_a_child_env_on_the_card_loads_through_its_v1_data_plane_with_no_card_read(sent):
    await _mcp(_record(await _card("mcp-email"))).load_environment_artifact(_artifact("email"))

    assert [(r.method, str(r.url)) for r in sent] == [("POST", f"{_CARD}/svc/mcp-email/agentenv")] * 2
    assert [json.loads(r.content)["method"] for r in sent] == ["data/reset", "data/add"]
    assert json.loads(sent[1].content)["params"]["parts"][0]["file"]["uri"] == "file:///data/email.json"


@pytest.mark.asyncio
async def test_a_child_env_missing_from_the_card_takes_the_legacy_path(sent):
    # email serves no card, so the gateway leaves it out of the composed card.
    await _mcp(_record(await _card("mcp-slack"))).load_environment_artifact(_artifact("email"))

    [request] = sent
    assert (request.method, str(request.url)) == ("POST", f"{_GW}/svc/mcp-email/api/reset")
    assert json.loads(request.content) == {"mock_data_path": "/data/email.json"}


@pytest.mark.asyncio
async def test_the_legacy_path_refuses_an_environment_of_several_files_before_sending_anything(sent):
    several = SimpleNamespace(environment_name="email", id="mail-seed", file_artifact_refs={"a.json": None, "b.json": None})

    with pytest.raises(RuntimeError, match="'email' takes one file through its pre-v1 load, and environment artifact "
                                           "'mail-seed' holds 2"):
        await _mcp(_record(await _card("mcp-slack"))).load_environment_artifact(several)
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("has_record", [False, True], ids=["no-record", "no-stored-card"])
async def test_an_env_without_a_stored_card_keeps_the_live_probe(sent, has_record):
    await _mcp(_record(None) if has_record else None).load_environment_artifact(_artifact("email"))

    assert [(r.method, str(r.url)) for r in sent] == [
        ("GET", f"{_GW}/svc/mcp-email{WELL_KNOWN_PATH}"),
        ("POST", f"{_GW}/svc/mcp-email/agentenv"),
        ("POST", f"{_GW}/svc/mcp-email/agentenv"),
    ]


@pytest.mark.asyncio
async def test_a_website_child_env_loads_at_its_own_card_url(sent):
    web = _website()
    web._sandbox, web._gateway_url, web._deployed = MagicMock(), _GW, _record(await _card("shop"))
    web._copy_artifact_into_container = AsyncMock(return_value="/tmp/data/shop.json")
    web._env_provider = MagicMock(spec=EnvironmentGatewayProvider, install_changelog_triggers=AsyncMock())

    await web.load_environment_artifact(_artifact("shop"))

    assert [(r.method, str(r.url)) for r in sent] == [("POST", f"{_CARD}/svc/shop/agentenv")] * 2


@pytest.mark.asyncio
async def test_a_website_stages_several_files_in_a_folder_of_the_loads_own_sends_each_with_its_path_and_removes_it(sent):
    web = _website()
    web._sandbox, web._gateway_url, web._deployed = MagicMock(exec_script=AsyncMock()), _GW, _record(await _card("shop"))
    web._copy_artifact_into_container = AsyncMock(side_effect=lambda file, path: f"/tmp/data/{path}")
    web._env_provider = MagicMock(spec=EnvironmentGatewayProvider, install_changelog_triggers=AsyncMock())
    files = {path: SimpleNamespace(content_type="application/json", filename=path.rsplit("/", 1)[-1])
             for path in ("orders/1.json", "cart.json")}

    await web.load_environment_artifact(SimpleNamespace(environment_name="shop", file_artifact_refs=dict.fromkeys(files),
                                                        get_file_artifacts=lambda: files))

    staged = [c.args[1] for c in web._copy_artifact_into_container.await_args_list]
    folder = staged[0].removesuffix("/orders/1.json")
    assert folder.startswith(".agentenv-load-") and staged == [f"{folder}/{path}" for path in files]
    parts = json.loads(sent[1].content)["params"]["parts"]
    assert [(p["file"]["uri"], p["metadata"][DATA_OBJECTS_EXTENSION_URI]["path"]) for p in parts] == [
        (f"file:///tmp/data/{folder}/{path}", path) for path in files]
    assert web._sandbox.exec_script.await_args.args[0].endswith(f"rm -rf /tmp/data/{folder}")


@pytest.mark.asyncio
@pytest.mark.parametrize("make, module", [(lambda: _mcp_env(), "mcp_server"), (lambda: _website(), "website")], ids=["mcp", "website"])
async def test_a_standalone_env_keeps_the_record_it_deploys(make, module):
    env = make()
    provider = MagicMock(
        spec=EnvironmentGatewayProvider, sandbox=SimpleNamespace(sandbox_id="sb-1", type="vm"),
        deploy=AsyncMock(return_value=DeployedGatewayEnv(env_id="e", env_version=1, gateway_url=_GW, mcp_url=f"{_GW}/mcp", db_web_url=None, sandbox_id="sb-1")),
        _environment_sandboxes={},
    )
    provider.environment_sandbox.return_value = None
    setattr(env, "_env_provider" if module == "mcp_server" else "_env_provider", provider)
    with patch("agent_env.config.get_config", MagicMock()), patch("agent_env.env.env.Env.get", MagicMock()), \
         patch("agent_env.providers.get_env_sandbox_provider", MagicMock()), \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", AsyncMock(return_value=None)), \
         patch("agent_env.env.envs._deployment.register_env_instance", side_effect=lambda deployed, ttl: deployed):
        deployed = await env.deploy()

    assert env._deployed is deployed and env._sandbox is provider.sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("make", [lambda: _website(), lambda: _multi()], ids=["website", "multi"])
async def test_a_second_deploy_with_a_bad_attribution_leaves_the_first_deployment_alone(make):
    env, first, second = make(), MagicMock(close=AsyncMock()), MagicMock(deploy=AsyncMock(), close=AsyncMock())
    env._env_provider, env._sandbox, env._deployed = first, MagicMock(terminate=AsyncMock()), MagicMock(env_id=env.id)
    with patch("agent_env.providers.build_env_provider", return_value=second) as build, pytest.raises(ValueError):
        await env.deploy(attribution="not-a-mapping")
    assert build.called and env._env_provider is first
    assert (first.close.called, env._sandbox.terminate.called, second.deploy.called, second.close.called) == (False, False, False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls, make", [(MCPServerEnv, lambda: _mcp_env()), (WebsiteEnv, lambda: _website())], ids=["mcp", "website"])
async def test_a_standalone_env_keeps_the_record_it_reattaches_to(cls, make):
    record = _record(None)
    provider = MagicMock(get_sandbox=AsyncMock(return_value=MagicMock()))
    with patch("agent_env.env.env.Env.get", return_value=make()), \
         patch("agent_env.providers.get_env_sandbox_provider", return_value=provider):
        env = await cls.from_deployed_env(record)

    assert env._deployed is record and env._sandbox is provider.get_sandbox.return_value
    provider.get_sandbox.assert_awaited_once_with(record.sandbox_id)


@pytest.mark.asyncio
async def test_multi_env_deploy_hands_its_record_to_every_child_env():
    env = _multi()
    provider = MagicMock(
        spec=EnvironmentGatewayProvider, sandbox=SimpleNamespace(sandbox_id="sb-1", type="vm"),
        _environment_sandboxes={}, _db_sandbox=None, _pgweb_sandbox=None, _db_mcp_sandbox=None,
    )
    container = MagicMock()
    provider.environment_sandbox.side_effect = lambda name: container if name == "email" else None  # container mode: the server's own
    provider.deploy = AsyncMock(return_value=DeployedGatewayEnv(env_id="multi", env_version=1, gateway_url=_GW, mcp_url=f"{_GW}/mcp",
                                                                db_web_url=None, sandbox_id="sb-1"))

    with patch("agent_env.providers.build_env_provider", return_value=provider), \
         patch("agent_env.providers.get_env_sandbox_provider", MagicMock()), \
         patch("agent_env.env.envs._deployment.register_env_instance", side_effect=lambda deployed, ttl: deployed):
        deployed = await env.deploy()

    assert [child._deployed for child in [*env.mcp_server_envs, *env.website_envs]] == [deployed, deployed]
    [mcp], [web] = env.mcp_server_envs, env.website_envs
    assert (mcp._env_provider, web._env_provider, mcp._gateway_url, web._gateway_url) == (provider, provider, _GW, _GW)
    assert (mcp._sandbox, web._sandbox) == (container, provider.sandbox)


@pytest.mark.asyncio
async def test_multi_env_reattach_hands_its_record_to_every_child_env_and_loads_at_the_card_url(sent):
    deployed = _record(await _card("mcp-email", "shop"))
    provider = MagicMock(get_sandbox=AsyncMock(return_value=MagicMock(mode="vm")))
    with patch("agent_env.env.env.Env.get", return_value=_multi()), \
         patch("agent_env.providers.get_env_sandbox_provider", return_value=provider):
        env = await MultiEnv.from_deployed_env(deployed)

    assert [child._deployed for child in [*env.mcp_server_envs, *env.website_envs]] == [deployed, deployed]
    assert all(child._env_provider is env._env_provider for child in [*env.mcp_server_envs, *env.website_envs])
    [mcp] = env.mcp_server_envs
    mcp._copy_artifact_into_container = AsyncMock(return_value="/data/email.json")
    mcp._staged_artifact_size = AsyncMock(return_value=None)
    env._env_provider.install_changelog_triggers = AsyncMock()

    await env.load_environment_artifact(_artifact("email"))

    assert [(r.method, str(r.url)) for r in sent] == [("POST", f"{_CARD}/svc/mcp-email/agentenv")] * 2
    env._env_provider.install_changelog_triggers.assert_awaited_once_with("email")



@pytest.mark.asyncio
async def test_multi_env_reattach_of_a_record_without_a_gateway_has_no_gateway_url():
    deployed = DeployedSandboxEnv(env_id="multi", env_version=1, sandbox_id="sb-1", environment_card_url=f"{_CARD}{WELL_KNOWN_PATH}",
                                  environment_card=await _card("mcp-email", "shop"))
    provider = MagicMock(get_sandbox=AsyncMock(return_value=MagicMock(mode="vm")))
    with patch("agent_env.env.env.Env.get", return_value=_multi()), \
         patch("agent_env.providers.get_env_sandbox_provider", return_value=provider):
        env = await MultiEnv.from_deployed_env(deployed)
    assert env._gateway_url is None

@pytest.mark.asyncio
async def test_multi_env_withholds_the_record_from_a_name_an_mcp_server_and_a_website_share():
    # A card-less slack MCP server and a carded slack website would both be looked up as "slack".
    email, slack, slack_web = _mcp_env(), _mcp_env("slack"), _website("slack")
    deployed = _record(await _card("mcp-email", "slack"))
    provider = MagicMock(get_sandbox=AsyncMock(return_value=MagicMock(mode="vm")))
    with patch("agent_env.env.env.Env.get", return_value=MultiEnv(id="multi", version=1, mcp_server_envs=[email, slack], website_envs=[slack_web])), \
         patch("agent_env.providers.get_env_sandbox_provider", return_value=provider):
        await MultiEnv.from_deployed_env(deployed)

    assert [email._deployed, slack._deployed, slack_web._deployed] == [deployed, None, None]


async def _card(*keys: str) -> dict:
    """The real gateway's composed card, with a v1 backing server behind each key."""
    gw = Gateway(host="127.0.0.1", port=0, server_name="env1234", internal_mcp_servers=[],
                 rest_proxy_urls={key: f"http://{key}:18765" for key in keys})

    async def fetch(client, key, base_url):
        return gw._rewrite_child_card(key, {"name": key.removeprefix("mcp-"), "url": "/agentenv"})

    gw._fetch_child_card = fetch
    return json.loads((await gw._serve_env_card()).body)


def _record(card: dict | None) -> DeployedEnv:
    return DeployedGatewayEnv(
        env_id="multi", env_version=1, gateway_url=_GW, mcp_url=f"{_GW}/mcp", db_web_url=None,
        sandbox_id="sb-1", environment_card_url=f"{_CARD}{WELL_KNOWN_PATH}", environment_card=card,
    )


def _artifact(name: str) -> SimpleNamespace:
    file = SimpleNamespace(content_type="application/json", filename=f"{name}.json")
    return SimpleNamespace(environment_name=name, file_artifact_refs=None, get_file_artifact=lambda: file,
                           get_file_artifacts=lambda: {file.filename: file})


def _mcp_env(name: str = "email") -> MCPServerEnv:
    return MCPServerEnv(id=f"mcp-{name}", version=1, docker_image_artifact=SimpleNamespace(image_name=f"{name}:1"), environment_name=name)


def _mcp(deployed: DeployedEnv | None) -> MCPServerEnv:
    env = _mcp_env()
    env._sandbox, env._gateway_url, env._deployed = MagicMock(), _GW, deployed
    env._copy_artifact_into_container = AsyncMock(return_value="/data/email.json")
    env._staged_artifact_size = AsyncMock(return_value=None)
    env._env_provider = MagicMock(spec=EnvironmentGatewayProvider, install_changelog_triggers=AsyncMock())
    return env


def _website(name: str = "shop") -> WebsiteEnv:
    return WebsiteEnv(id=f"web-{name}", version=1, backend_docker_image_artifact=SimpleNamespace(image_name="be:1"),
                      frontend_docker_image_artifact=SimpleNamespace(image_name="fe:1"), environment_name=name)


def _multi() -> MultiEnv:
    return MultiEnv(id="multi", version=1, mcp_server_envs=[_mcp_env()], website_envs=[_website()])
