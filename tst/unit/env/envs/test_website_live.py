"""Live website envs: a website env with ``live = true`` is a real site, such as github.com, whose server serves none of
its pages. The env stores ``live``, ``allow_hosts`` and ``settings`` as it stores ``domains``, read from its server's
``@website(...)`` when they aren't given. A gateway gives it no site route and hands its server its settings, each
reference resolved only then. A computer routing a multi env's sites leaves it out of its site table, and one whose
network is an allowlist may reach its names, a synthetic site's claim winning. A computer's start tabs must name a
site of the run."""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from agentenv_protocol.website import WEBSITE_EXTENSION_URI, WEBSITE_SETTINGS_ENV

from agent_env.artifact import Artifact
from agent_env.env.env import DeployedGatewayEnv, Env
from agent_env.env.envs._multi_computers import start_tab_problems
from agent_env.env.envs.computer.utils import sites
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.service_db import ServiceDBConfig
from agent_env.env.envs.website import WebsiteEnv, declared_live
from agent_env.providers.env_providers.constants import GATEWAY_SERVICE_NAME
from agent_env.providers.env_providers.env_computer_provider import _reaching
from agent_env.providers.env_providers.env_gateway_provider import (EnvironmentGatewayProvider, WebsiteConfig,
                                                                     _frontend_urls, _redact_compose_secrets,
                                                                     website_config)
from agent_env.providers.env_state import LocalPostgresStateProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.utils import card_naming
from agent_env.utils.card_naming import WebsiteSource

SOURCE = '''from pydantic import BaseModel, SecretStr
from agentenv_protocol.website import WebsiteEnvironment, website


@website(domains=["github.com"], live=True, allow_hosts=["*.githubassets.com"])
class GitHub(WebsiteEnvironment):
    class Settings(BaseModel):
        token: SecretStr
        org: str = "octo"
'''


def _live(**overrides) -> WebsiteEnv:
    fields = dict(id="github-site", version=1, server_docker_image_artifact=MagicMock(image_name="github-server"),
                  environment_name="github", domains=["github.com"], live=True,
                  allow_hosts=["*.githubassets.com", "api.github.com"], settings={"token": "env:GH_TOKEN", "org": "octo"},
                  secret_settings=["token"])
    return WebsiteEnv(**{**fields, **overrides})


def _synthetic(name: str = "shop", domains: tuple[str, ...] = ("shop.test",)) -> WebsiteEnv:
    return WebsiteEnv(id=f"{name}-site", version=1, server_docker_image_artifact=MagicMock(image_name=f"{name}-server"),
                      environment_name=name, domains=list(domains))


def test_a_live_website_env_keeps_what_it_is_in_its_record_and_its_env_toml(monkeypatch):
    env = _live()
    data = env.to_dict()
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: env.server_docker_image_artifact))
    monkeypatch.setattr(WebsiteEnv, "provider_toml", lambda self, out, provider_type: {})

    back = WebsiteEnv.from_dict(data)
    toml = env.to_toml(MagicMock())

    assert (back.live, back.allow_hosts, back.settings, back.secret_settings) == (
        True, ["*.githubassets.com", "api.github.com"], {"token": "env:GH_TOKEN", "org": "octo"}, ["token"])
    assert {key: toml[key] for key in ("live", "allow_hosts", "settings")} == {
        "live": True, "allow_hosts": ["*.githubassets.com", "api.github.com"],
        "settings": {"token": "env:GH_TOKEN", "org": "octo"}}
    assert "secret_settings" not in toml  # read from the site's source
    assert not {"live", "allow_hosts", "settings", "secret_settings"} & _synthetic().to_dict().keys()


@pytest.mark.parametrize(("overrides", "problem"), [
    ({"domains": []}, "a live website env names the domains the real site is at"),
    ({"live": False}, "allow_hosts are the other hosts a live site's pages reach; set live = true"),
    ({"allow_hosts": ["https://github.com"]}, "allow_hosts:"),
    ({"settings": {"token": "ghp_plain"}}, "settings.token is a secret, so it's a secret:KEY or env:NAME reference"),
    ({"settings": {"token": "env:GH_TOKEN?ghp_plain"}}, "settings.token is a secret, so it's a secret:KEY or env:NAME "
                                                        "reference with no ?default"),
    ({"settings": {"token": "secret:"}}, "settings.token is a secret"),
    ({"settings": {"token": "env:T", "orgs": ["a"]}}, "settings.orgs is a string, number or boolean"),
    ({"settings": {"token": "env:T", "1org": "a"}}, "settings: '1org' isn't a setting's name"),
], ids=["no-domains", "allow-hosts-not-live", "bad-allow-host", "plain-secret", "fallback-secret", "nameless-secret",
        "list-setting", "bad-name"])
def test_a_website_env_refuses_what_a_live_site_cant_be(overrides, problem):
    with pytest.raises(ValueError, match=re.escape(problem)):
        _live(**overrides)


def test_put_reads_live_its_names_allow_hosts_and_secret_settings_from_the_servers_source(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "site.py").write_text(SOURCE)

    source = card_naming.website_source_from_source(str(tmp_path / "Dockerfile"), None)

    assert source == WebsiteSource(live=True, domains=["github.com"], allow_hosts=["*.githubassets.com"],
                                   secret_settings=["token"])
    assert declared_live(source) == ({"live": True, "allow_hosts": ["*.githubassets.com"], "secret_settings": ["token"],
                                      "domains": ["github.com"]}, [])
    _, problems = declared_live(source, live=False, allow_hosts=["api.github.com"])
    assert problems == ["live is False here, but the site's @website(...) declares live=True",
                        "allow_hosts is ['api.github.com'] here, but the site's @website(...) declares "
                        "allow_hosts=['*.githubassets.com']"]


def test_a_synthetic_sites_names_stay_as_given_and_a_source_with_two_sites_or_no_literals_declares_nothing(tmp_path):
    assert declared_live(WebsiteSource(live=False, domains=["shop.test"])) == (
        {"live": False, "allow_hosts": [], "secret_settings": []}, [])
    other = SOURCE.replace('live=True, allow_hosts=["*.githubassets.com"]', "live=True").replace("GitHub", "Gist")
    (tmp_path / "a.py").write_text(SOURCE)
    (tmp_path / "b.py").write_text(other)
    assert card_naming.website_source_in_files({"a.py": tmp_path / "a.py", "b.py": tmp_path / "b.py"}, "Dockerfile") is None
    (tmp_path / "c.py").write_text("@website(domains=DOMAINS, live=LIVE)\nclass Site(WebsiteEnvironment):\n    pass\n")
    assert card_naming.website_source_in_files({"c.py": tmp_path / "c.py"}, "Dockerfile") == WebsiteSource(
        allow_hosts=[])
    (tmp_path / "d.py").write_text("@website(**OPTIONS)\nclass Site(WebsiteEnvironment):\n    pass\n")
    assert card_naming.website_source_in_files({"d.py": tmp_path / "d.py"}, "Dockerfile") == WebsiteSource()


def test_a_secret_setting_annotated_as_a_string_is_a_secret_too(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "site.py").write_text(SOURCE.replace("        token: SecretStr\n", '        token: "SecretStr"\n'
                                                     '        key: Optional["SecretStr"] = None\n'
                                                     '        note: "str" = ""\n'))

    source = card_naming.website_source_from_source(str(tmp_path / "Dockerfile"), None)

    assert source.secret_settings == ["token", "key"]
    with pytest.raises(ValueError, match="settings.token is a secret"):
        _live(settings={"token": "ghp_plain"}, secret_settings=source.secret_settings)


def test_a_source_leaving_live_and_allow_hosts_out_declares_their_defaults_so_a_flag_cant_contradict_it(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "site.py").write_text('@website(domains=["shop.test"])\nclass Shop(WebsiteEnvironment):\n    pass\n')

    source = card_naming.website_source_from_source(str(tmp_path / "Dockerfile"), None)
    _, problems = declared_live(source, live=True, allow_hosts=["api.shop.test"])

    assert source == WebsiteSource(live=False, domains=["shop.test"], allow_hosts=[])
    assert problems == ["live is True here, but the site's @website(...) declares live=False",
                        "allow_hosts is ['api.shop.test'] here, but the site's @website(...) declares allow_hosts=[]"]


def test_an_env_toml_takes_what_the_source_declares_and_refuses_what_it_contradicts():
    def refuse(problems):
        raise ValueError(problems)

    ctx = SimpleNamespace(config_problem=lambda problem: f"env.toml: {problem}", refuse=refuse,
                          accept_env=lambda data, cls, named_by: dict(data),
                          website_source=lambda image_key: WebsiteSource(live=True, domains=["github.com"],
                                                                         allow_hosts=["api.github.com"],
                                                                         secret_settings=["token"]))
    data = {"backend_image": "server", "environment_name": "github", "settings": {"token": "secret:gh"}}

    assert WebsiteEnv.accept_toml(data, ctx) == {**data, "live": True, "domains": ["github.com"],
                                                 "allow_hosts": ["api.github.com"], "secret_settings": ["token"]}
    with pytest.raises(ValueError, match="live is False here"):
        WebsiteEnv.accept_toml({**data, "live": False}, ctx)
    with pytest.raises(ValueError, match="settings.token is a secret"):
        WebsiteEnv.accept_toml({**data, "settings": {"token": "ghp_plain"}}, ctx)


def test_a_live_site_gets_no_site_route_and_its_settings_resolved_only_in_its_container(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "ghp_re$$olved${HOME}")

    compose = EnvironmentGatewayProvider().create_docker_compose(
        mcp_servers=[], gateway_image="agent-gateway",
        website_configs=[website_config(_live()),
                         WebsiteConfig(backend_image="shop-server", frontend_image=None, environment_name="shop")],
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance())

    services = yaml.safe_load(compose)["services"]
    gateway = dict(item.split("=", 1) for item in services[GATEWAY_SERVICE_NAME]["environment"])
    assert gateway["WEBSITE_URLS"] == "shop=http://shop-website-backend:8000"
    assert "github=http://github-website-backend:8000" in gateway["REST_PROXY_URLS"].split(",")
    handed = dict(item.split("=", 1) for item in services["github-website-backend"]["environment"])
    # Compose reads $$ as a $ and interpolates no other.
    assert json.loads(handed[WEBSITE_SETTINGS_ENV].replace("$$", "$")) == {"token": "ghp_re$$olved${HOME}", "org": "octo"}
    assert "$" not in handed[WEBSITE_SETTINGS_ENV].replace("$$", "")
    assert WEBSITE_SETTINGS_ENV not in dict(item.split("=", 1) for item in services["shop-website-backend"]["environment"])
    assert "olved" not in _redact_compose_secrets(compose)
    assert (_frontend_urls("https://gw", [website_config(_live())]),
            _frontend_urls("https://gw", [website_config(_live()), website_config(_synthetic())])) == (
        None, {"shop": "https://gw/website/shop/"})


@pytest.mark.asyncio
async def test_a_computer_routes_a_multi_envs_synthetic_sites_and_leaves_its_live_one_to_the_real_site(monkeypatch):
    record = DeployedGatewayEnv(
        env_id="office", env_version=1, sandbox_id="gw", sandbox_type="local", gateway_url="http://gw",
        environment_card={"name": "office", "children_environments": [
            {"name": "shop", "environment_name": "shop", "url": "/svc/shop/agentenv", "capabilities": {"extensions": [
                {"uri": WEBSITE_EXTENSION_URI, "params": {"domains": ["shop.test"], "browse_url": "/website/shop/"}}]}},
            {"name": "github", "environment_name": "github", "url": "/svc/github/agentenv", "capabilities": {
                "extensions": [{"uri": WEBSITE_EXTENSION_URI, "params": {
                    "domains": ["github.com"], "live": True, "allow_hosts": ["api.github.com"]}}]}}]})
    monkeypatch.setattr(Env, "get", classmethod(lambda cls, id, version=None: SimpleNamespace(
        website_envs=[_synthetic(), _live()])))

    found = await sites.deployed_sites([record])

    assert [(site["name"], site["domains"]) for site in found] == [("shop", ["shop.test"])]


def test_a_multi_env_lets_its_computers_reach_its_live_sites_names_a_synthetic_sites_claim_winning(caplog):
    office = MultiEnv(id="office", version=1, mcp_server_envs=[],
                      website_envs=[_synthetic(name="api", domains=("api.github.com",)), _live()])

    with caplog.at_level(logging.WARNING):
        hosts = office.live_hosts()

    assert hosts == ["github.com", "*.githubassets.com"]
    assert "names 'api.github.com', which the synthetic site at 'api.github.com' claims" in caplog.text
    gist = MultiEnv(id="office", version=1, mcp_server_envs=[],
                    website_envs=[_synthetic(name="gist", domains=("gist.github.com",)), _live(allow_hosts=["*.github.com"])])
    assert gist.live_hosts() == ["github.com", "*.github.com"]  # the wider name stays, gist.github.com routed within it
    assert office.live_websites() == [{"env_id": "github-site", "environment_name": "github",
                                       "domains": ["github.com"], "allow_hosts": ["*.githubassets.com", "api.github.com"]}]


def test_a_live_and_a_synthetic_site_may_claim_one_name_and_two_live_ones_may_not():
    MultiEnv(id="office", version=1, mcp_server_envs=[], website_envs=[_synthetic(domains=("github.com",)), _live()])
    with pytest.raises(sites.SiteConflict, match="'github.com'"):
        MultiEnv(id="office", version=1, mcp_server_envs=[],
                 website_envs=[_live(), _live(id="gist-site", environment_name="gist", allow_hosts=[])])


def test_a_computer_routing_sites_opens_start_tabs_only_at_names_a_website_of_the_run_claims():
    chromium = SimpleNamespace(environment_name="chromium", declaration=SimpleNamespace(browser=True))
    computer = SimpleNamespace(composed_apps=lambda: [chromium],
                               app_defaults=lambda: {"chromium": {"start_urls": ["https://shop.test/"]}})
    office = MultiEnv(id="office", version=1, mcp_server_envs=[], website_envs=[_synthetic(), _live()])

    def entry(urls=None, routing=True):
        settings = {"app_settings": {"chromium": {"start_urls": urls}}} if urls is not None else {}
        return {"env_params": {**({"site_routing": routing} if routing else {}), **settings}}

    names = office._start_tab_names(entry())
    assert names == ["shop.test", "github.com", "*.githubassets.com", "api.github.com"]
    assert office._start_tab_names(entry(routing=["another-site"])) == names[1:]
    assert start_tab_problems(computer, entry(), names) == []
    assert start_tab_problems(computer, entry(["https://github.com/", "https://x.githubassets.com/", "about:blank"]),
                              names) == []
    [problem] = start_tab_problems(computer, entry(["https://example.com/"]), names)
    assert "chromium's start tab https://example.com/ names example.com, which none of the multi env's" in problem
    assert start_tab_problems(computer, entry(["https://example.com/"], routing=False), names) == []
    # A routed site storing no names takes those its card declares at deploy, which checks the tabs then.
    carded = MultiEnv(id="office", version=1, mcp_server_envs=[], website_envs=[_synthetic(domains=()), _live()])
    assert carded._start_tab_names(entry()) is None
    assert carded._start_tab_names(entry(routing=["github-site"])) == ["github.com", "*.githubassets.com",
                                                                        "api.github.com"]


def test_a_computers_allowlist_takes_the_live_hosts_and_one_that_allows_everything_stays_as_it_is():
    provider = SimpleNamespace(url_from_sandbox=lambda url: url)
    allowlist = NetworkPolicy(mode="allowlist", allow_hosts=("pypi.org",))

    assert _reaching(allowlist, provider, "https://gw.example", ("github.com", "*.githubassets.com")).allow_hosts == (
        "pypi.org", "gw.example", "github.com", "*.githubassets.com")
    assert _reaching(NetworkPolicy(), provider, "https://gw.example", ("github.com",)) == NetworkPolicy()
    assert _reaching(None, provider, "", ("github.com",)) is None
