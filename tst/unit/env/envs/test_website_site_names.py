"""A website env stores the names its env.toml gives it, a multi env refuses two of its sites claiming one name, and
a computer's site table takes the names a site stores before those its card declares."""

import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env import Env
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.envs import MultiEnv, WebsiteEnv
from agent_env.env.envs.computer.utils import sites
from agent_env.env.envs.computer.utils.params import Site


def _image(id):
    return get_artifact_store().put_document(DockerImageArtifact(
        id=id, description="", image_name=f"registry.example/{id}@sha256:" + "a" * 64))


def _site(name, *domains):
    return WebsiteEnv.put(id=name, server_docker_image_artifact=_image(f"{name}-server"), environment_name=name,
                          domains=list(domains))


def _card_child(name, *domains):
    extension = {"uri": "urn:agentenv:website/v1", "params": {"domains": list(domains), "browse_url": f"/website/{name}/"}}
    return {"name": name, "url": f"/svc/{name}/agentenv", "capabilities": {"extensions": [extension]}}


def test_a_site_stores_its_names_and_one_that_cant_be_a_name_is_refused(local_stores):
    _site("slack", "slack.com", "*.slack.com")

    assert Env.get("slack").domains == ["slack.com", "*.slack.com"]
    assert "domains" not in _site("notes").to_dict()
    with pytest.raises(ValueError) as refused:
        WebsiteEnv(id="bad", version=None, server_docker_image_artifact=_image("bad-server"), environment_name="bad",
                   domains=["Slack.com"])
    assert str(refused.value) == "website env 'bad': domains: 'Slack.com' must be lowercase, with no trailing dot"


def test_a_multi_env_whose_sites_claim_one_name_is_refused(local_stores):
    slack, notes = _site("slack", "slack.com", "*.slack.com"), _site("notes")

    with pytest.raises(sites.SiteConflict) as refused:
        MultiEnv.put(id="suite", mcp_server_envs=[], website_envs=[slack, notes, _site("chat", "acme.slack.com")])
    assert str(refused.value) == ("multi env 'suite': its websites claim names twice: 'acme.slack.com' (chat at "
                                  "website_envs[2]) overlaps '*.slack.com' (slack at website_envs[0])")
    assert MultiEnv.put(id="suite", mcp_server_envs=[],
                        website_envs=[slack, notes, _site("forum", "forum.example")]).version == 1


def test_two_versions_of_one_website_env_claiming_one_name_are_refused(local_stores):
    shop = _site("shop", "shop.test")
    renamed = WebsiteEnv.put(id="shop", server_docker_image_artifact=_image("shop-server"), environment_name="chat",
                             domains=["shop.test"])

    with pytest.raises(sites.SiteConflict, match=r"'shop.test' \(chat at website_envs\[1\]\) overlaps 'shop.test' "
                                                 r"\(shop at website_envs\[0\]\)"):
        MultiEnv.put(id="suite", mcp_server_envs=[], website_envs=[shop, renamed])


@pytest.mark.asyncio
async def test_a_computers_site_table_takes_the_names_a_site_stores_before_its_cards(local_stores, monkeypatch):
    async def get_card(url, timeout=10, verify=True):
        raise AssertionError(f"read {url} live")

    monkeypatch.setattr(protocol_v1, "get_card", get_card)
    suite = MultiEnv.put(id="suite", mcp_server_envs=[],
                         website_envs=[_site("shop", "shop.test"), _site("notes"), _site("forum", "forum.test")])
    card = {"name": "suite", "children_environments": [_card_child("shop", "shop.example"), _card_child("notes")]}
    record = DeployedGatewayEnv(env_id="suite", env_version=suite.version, sandbox_id="s",
                                gateway_url="https://gw.example", environment_card=card,
                                website_frontend_urls={"forum": "https://gw.example/website/forum/"})

    found = await sites.deployed_sites([record])

    table = sites.site_table([Site(**site) for site in found])
    assert {route.site: route.domains for route in table.routes} == {
        "shop": ("shop.test",), "notes": ("notes.localhost",), "forum": ("forum.test",)}
