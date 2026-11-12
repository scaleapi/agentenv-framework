"""Websites in the computer: the names a deployment's sites claim and which wins, the CA and certificates made for a
deploy, how the proxy is installed through the sandbox's root channel, and the sites a run deployed."""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from cryptography import x509
from cryptography.x509.verification import PolicyBuilder, Store, VerificationError

from agent_env.env.envs.computer.utils import sites
from agent_env.env.envs.computer.utils.params import Site
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, VmSandbox


def _site(name, *domains):
    return Site(name=name, url=f"https://gw.example/website/{name}/", domains=list(domains))


def test_a_synthetic_sites_names_are_routed_and_a_site_declaring_none_is_its_name_on_localhost():
    table = sites.site_table([_site("notes", "notes.test"), _site("slack", "slack.com", "*.slack.com"),
                              _site("bare")])

    assert [(r.site, r.domains, r.tls_names) for r in table.routes] == [
        ("notes", ("notes.test",), ("notes.test",)),
        ("slack", ("slack.com", "*.slack.com"), ("slack.com", "*.slack.com")),
        ("bare", ("bare.localhost",), ())]
    assert table.unrouted == {} and table.shadowed == []


def test_a_sites_name_wins_over_the_allowlist():
    table = sites.site_table([_site("gh", "github.com")], allow_hosts=["github.com", "*.wikipedia.org"])

    assert [(r.site, r.domains) for r in table.routes] == [("gh", ("github.com",))]
    assert table.unrouted == {"*.wikipedia.org": "allowlist"}
    assert table.to_dict()["shadowed"] == [{"name": "github.com", "rank": "allowlist", "owner": "network", "by": "gh"}]


def test_an_allowlisted_wildcard_a_site_takes_one_name_of_stays_for_the_rest():
    table = sites.site_table([_site("app", "app.slack.com")], allow_hosts=["*.slack.com"])

    assert table.unrouted == {"*.slack.com": "allowlist"} and len(table.shadowed) == 1


def test_a_site_named_with_capitals_or_underscores_declaring_none_is_reached_by_a_dns_name_on_localhost():
    table = sites.site_table([_site("my_site"), _site("Notes"), _site("-edge_")])

    assert [(r.site, r.domains) for r in table.routes] == [
        ("my_site", ("my-site.localhost",)), ("Notes", ("notes.localhost",)), ("-edge_", ("edge.localhost",))]


def test_two_sites_given_one_fallback_name_are_refused():
    with pytest.raises(sites.SiteConflict, match=r"'my-site[.]localhost' \(my-site\) overlaps 'my-site[.]localhost' "
                                                 r"\(my_site\)"):
        sites.site_table([_site("my_site"), _site("my-site")])


@pytest.mark.parametrize("claimed, conflict", [
    ([_site("a", "notes.test"), _site("b", "notes.test")], "'notes.test' (b) overlaps 'notes.test' (a)"),
    ([_site("a", "*.slack.com"), _site("b", "eu.slack.com")], "'eu.slack.com' (b) overlaps '*.slack.com' (a)"),
], ids=["synthetic", "wildcard"])
def test_two_sites_claiming_a_name_are_refused(claimed, conflict):
    with pytest.raises(sites.SiteConflict, match=conflict.replace("*", r"\*").replace(".", "[.]").replace("(", r"\(")
                       .replace(")", r"\)")):
        sites.site_table(claimed)


def test_a_wildcard_and_a_name_one_label_deeper_or_its_apex_dont_overlap():
    assert not sites.overlap("*.slack.com", "slack.com") and not sites.overlap("*.slack.com", "a.b.slack.com")
    assert sites.overlap("*.slack.com", "eu.slack.com") and sites.overlap("eu.slack.com", "*.slack.com")


@pytest.mark.parametrize("name, problem", [
    ("Notes.test", "must be lowercase"), ("notes.test.", "no trailing dot"), ("10.0.0.1", "an address"),
    ("*.test", "a wildcard over a single label"), ("a.*.test", "isn't a DNS name"),
    ("no_underscore.test", "isn't a DNS"),
])
def test_a_name_that_cant_be_a_sites_is_refused(name, problem):
    with pytest.raises(ValueError, match=problem):
        sites.site_table([_site("a", name)])


def test_two_sites_of_one_name_are_refused():
    with pytest.raises(ValueError, match="two sites are named 'a'"):
        sites.site_table([_site("a", "a.test"), _site("a", "b.test")])


def _verify(ca_pem, cert_pem, name, now):
    ca = x509.load_pem_x509_certificate(ca_pem.encode())
    verifier = PolicyBuilder().store(Store([ca])).time(now).build_server_verifier(x509.DNSName(name))
    return verifier.verify(x509.load_pem_x509_certificate(cert_pem.encode()), [])


def test_a_deploys_ca_signs_each_sites_names_and_is_constrained_to_them():
    now = datetime.now(UTC).replace(microsecond=0)
    table = sites.site_table([_site("notes", "notes.test"), _site("slack", "slack.com", "*.slack.com"),
                              _site("bare")])

    issued = sites.issue(table.routes, valid_for=timedelta(hours=3), label="desktop", now=now)

    assert set(issued.leaves) == {"notes", "slack"}
    ca = x509.load_pem_x509_certificate(issued.ca_pem.encode())
    constraints = ca.extensions.get_extension_for_class(x509.NameConstraints).value
    assert [n.value for n in constraints.permitted_subtrees] == ["notes.test", "slack.com"]
    assert ca.not_valid_after_utc == now + timedelta(hours=3) + sites.VALID_MARGIN
    for site, name in (("notes", "notes.test"), ("slack", "slack.com"), ("slack", "eu.slack.com")):
        _verify(issued.ca_pem, issued.leaves[site][0], name, now)
    with pytest.raises(VerificationError):
        _verify(issued.ca_pem, issued.leaves["notes"][0], "slack.com", now)
    assert "PRIVATE KEY" in issued.leaves["notes"][1] and "PRIVATE KEY" not in issued.ca_pem


def test_a_name_longer_than_a_common_name_takes_is_signed_by_its_alternative_names_alone():
    now = datetime.now(UTC).replace(microsecond=0)
    name = f"{'a' * 63}.test"
    issued = sites.issue(sites.site_table([_site("long", name)]).routes, valid_for=timedelta(hours=1), label="x",
                         now=now)

    leaf = x509.load_pem_x509_certificate(issued.leaves["long"][0].encode())
    assert len(leaf.subject) == 0 and leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).critical
    _verify(issued.ca_pem, issued.leaves["long"][0], name, now)


def test_no_ca_is_made_when_no_name_is_served_over_https():
    assert sites.issue(sites.site_table([_site("bare")]).routes, valid_for=timedelta(hours=1), label="x") == \
        sites.Issued(None, {})


class FakeVm(VmSandbox):
    """A VM sandbox whose container's root channel is recorded, the proxy's install answering ``report``."""

    type = "fake"
    sandbox_id = "vm-1"
    tunnel_urls = {}
    vnc_url = None

    def __init__(self, report):
        self.report, self.scripts, self.written, self.events, self.staged = report, [], {}, [], {}

    @property
    def container_name(self):
        return "computer-1"

    def url_from_sandbox(self, url):
        return url.replace("https://gw.example", "http://host.docker.internal:8080")

    async def terminate(self):
        pass

    async def write_file_from_text(self, content, destination_path, container=None):
        self.written[destination_path] = content
        self.events.append(("write", destination_path))

    @contextlib.asynccontextmanager
    async def secret_dir(self):
        yield "/tmp/secret-1"
        self.events.append(("removed", "/tmp/secret-1"))

    async def write_secret_file(self, data, vm_path):
        self.staged[vm_path] = data.decode()
        self.events.append(("secret", vm_path))

    async def docker_cp(self, source, destination, *, remove_source=False):
        container, path = destination.split(":", 1)
        self.written[path] = self.staged.pop(source) if remove_source else self.staged[source]
        self.events.append(("cp", source, destination))

    async def exec_script(self, script, *, max_retries=0):
        self.scripts.append(script)
        self.events.append(script)
        return "noise\n" + json.dumps(self.report) + "\n"


_REPORT = {"trust": {"system": "/usr/local/share/ca-certificates/agentenv-sites.crt", "nss": []},
           "names": ["notes.test"], "checks": {"notes.test": {"url": "https://notes.test/", "status": 200}}}


@pytest.mark.asyncio
async def test_route_installs_the_proxy_as_root_with_each_sites_route_as_the_computer_reaches_it():
    sandbox = FakeVm(_REPORT)

    routed = await sites.route(sandbox, [_site("notes", "notes.test")], user="user", network=None,
                               valid_for=timedelta(hours=1), label="desktop", os_type="linux", display="x11")

    assert set(sandbox.written) == {"/etc/agentenv/sites/site_proxy.py", "/etc/agentenv/sites/config.json"}
    config = json.loads(sandbox.written["/etc/agentenv/sites/config.json"])
    (site,) = config["sites"]
    assert (config["user"], site["upstream"], site["domains"]) == (
        "user", "http://host.docker.internal:8080/website/notes/", ["notes.test"])
    assert site["key_pem"] and config["ca_pem"] == routed["ca"]
    assert sandbox.scripts == ["docker exec -u 0 computer-1 mkdir -p /etc/agentenv/sites",
                               "docker exec -u 0 computer-1 chmod 0700 /etc/agentenv/sites",
                               "docker exec -u 0 computer-1 mkdir -p /etc/agentenv/sites",
                               "docker exec -u 0 computer-1 chmod 0600 /etc/agentenv/sites/config.json",
                               "docker exec -u 0 computer-1 python3 /etc/agentenv/sites/site_proxy.py install "
                               "/etc/agentenv/sites/config.json"]
    assert "upstream_headers" not in site and "proxy_port" not in config
    assert routed["checks"] == _REPORT["checks"] and routed["names"]["routes"] == {"notes": ["notes.test"]}
    assert "PRIVATE KEY" not in json.dumps(routed)


@pytest.mark.asyncio
async def test_route_gives_a_browser_alone_the_forward_port_and_a_computer_with_a_display_none():
    alone, desktop = FakeVm(_REPORT), FakeVm(_REPORT)

    for sandbox, display in ((alone, "none"), (desktop, "x11")):
        await sites.route(sandbox, [_site("notes", "notes.test")], user="user", network=None,
                          valid_for=timedelta(hours=1), label="desktop", os_type="linux", display=display)

    configs = [json.loads(sandbox.written["/etc/agentenv/sites/config.json"]) for sandbox in (alone, desktop)]
    assert [config.get("proxy_port") for config in configs] == [3129, None]


@pytest.mark.asyncio
async def test_a_browser_alones_forward_port_takes_the_live_sites_hosts_straight_to_them_with_no_site_to_route():
    routed, alone, desktop = FakeVm(_REPORT), FakeVm(_REPORT), FakeVm(_REPORT)
    live = ("github.com", "*.githubassets.com")

    await sites.route(routed, [_site("notes", "notes.test")], user="user", network=None, valid_for=timedelta(hours=1),
                      label="browser", os_type="linux", display="none", live_hosts=live)
    for sandbox, display in ((alone, "none"), (desktop, "x11")):
        await sites.route(sandbox, [], user="user", network=None, valid_for=timedelta(hours=1), label="browser",
                          os_type="linux", display=display, live_hosts=live)

    beside = json.loads(routed.written["/etc/agentenv/sites/config.json"])
    only = json.loads(alone.written["/etc/agentenv/sites/config.json"])
    assert (beside["proxy_port"], beside["direct"]) == (3129, list(live))
    assert (only["sites"], only["ca_pem"], only["proxy_port"], only["direct"]) == ([], None, 3129, list(live))
    assert desktop.written == {} and desktop.scripts == []


@pytest.mark.asyncio
async def test_the_config_holding_the_keys_is_written_over_stdin_into_a_private_staging_folder_never_as_text():
    sandbox = FakeVm(_REPORT)

    await sites.route(sandbox, [_site("notes", "notes.test")], user="user", network=None,
                      valid_for=timedelta(hours=1), label="desktop", os_type="linux", display="x11")

    assert ("write", "/etc/agentenv/sites/config.json") not in sandbox.events
    secret = sandbox.events.index(("secret", "/tmp/secret-1/config.json"))
    copied = sandbox.events.index(("cp", "/tmp/secret-1/config.json", "computer-1:/etc/agentenv/sites/config.json"))
    assert secret < copied < sandbox.events.index(("removed", "/tmp/secret-1")) and sandbox.staged == {}
    assert not any("PRIVATE KEY" in script for script in sandbox.scripts)


@pytest.mark.asyncio
async def test_the_headers_the_gateways_provider_asks_for_go_in_the_private_config_alone(monkeypatch):
    from agent_env.providers.sandbox_providers import sandbox_provider

    sandbox = FakeVm(_REPORT)
    monkeypatch.setattr(sandbox_provider, "sandbox_request_headers_for_url", lambda url: {
        "CF-Access-Client-Secret": "s3cret"} if url.startswith("http://host.docker.internal:8080/") else {})

    routed = await sites.route(sandbox, [_site("notes", "notes.test")], user="user", network=None,
                               valid_for=timedelta(hours=1), label="desktop", os_type="linux", display="x11")

    (site,) = json.loads(sandbox.written["/etc/agentenv/sites/config.json"])["sites"]
    assert site["upstream_headers"] == {"CF-Access-Client-Secret": "s3cret"}
    private = sandbox.events.index("docker exec -u 0 computer-1 chmod 0600 /etc/agentenv/sites/config.json")
    copied = ("cp", "/tmp/secret-1/config.json", "computer-1:/etc/agentenv/sites/config.json")
    assert sandbox.events.index(copied) < private < len(sandbox.events) - 1
    assert "s3cret" not in json.dumps(routed)


@pytest.mark.asyncio
async def test_the_sites_keys_land_in_a_folder_root_alone_can_enter():
    sandbox = FakeVm(_REPORT)

    await sites.route(sandbox, [_site("notes", "notes.test")], user="user", network=None,
                      valid_for=timedelta(hours=1), label="desktop", os_type="linux", display="x11")

    private = sandbox.events.index("docker exec -u 0 computer-1 chmod 0700 /etc/agentenv/sites")
    assert private < sandbox.events.index(("cp", "/tmp/secret-1/config.json", "computer-1:/etc/agentenv/sites/config.json"))
    assert "key_pem" in sandbox.written["/etc/agentenv/sites/config.json"]


@pytest.mark.asyncio
async def test_a_name_that_doesnt_load_in_the_computer_fails_the_route():
    report = {**_REPORT, "checks": {"notes.test": {"url": "https://notes.test/", "error": "certificate verify failed"}}}

    with pytest.raises(RuntimeError, match="https://notes.test/ didn't load in the computer: certificate verify"):
        await sites.route(FakeVm(report), [_site("notes", "notes.test")], user="user", network=None,
                          valid_for=timedelta(hours=1), label="desktop", os_type="linux", display="x11")


@pytest.mark.asyncio
async def test_a_network_that_keeps_the_computer_from_the_gateway_is_refused_before_anything_is_installed():
    sandbox = FakeVm(_REPORT)

    with pytest.raises(RuntimeError, match="allows only api.example, so it can't reach the gateway at "
                                           "host.docker.internal"):
        await sites.route(sandbox, [_site("notes", "notes.test")], user="user",
                          network={"mode": "allowlist", "allow_hosts": ["api.example"]}, valid_for=timedelta(hours=1),
                          label="desktop", os_type="linux", display="x11")

    assert sandbox.scripts == [] and sandbox.written == {}


@pytest.mark.asyncio
async def test_a_gateway_the_sandboxs_applied_network_allows_by_its_address_range_is_reached():
    sandbox = FakeVm(_REPORT)
    sandbox.url_from_sandbox = lambda url: url.replace("https://gw.example", "http://10.0.0.7:8080")
    sandbox.network_policy = NetworkPolicy(mode="allowlist", allow_hosts=("api.example",),
                                           allow_cidrs=("10.0.0.0/24",))

    routed = await sites.route(sandbox, [_site("notes", "notes.test")], user="user",
                               network={"mode": "allowlist", "allow_hosts": ["api.example"]},
                               valid_for=timedelta(hours=1), label="desktop", os_type="linux", display="x11")

    assert routed["checks"] == _REPORT["checks"]
    routes = [sites.Route("notes", "http://10.0.0.7:8080/website/notes/", ("notes.test",))]
    assert sites.reachable_problem(NetworkPolicy(mode="allowlist", allow_cidrs=("10.1.0.0/24",)), routes) == (
        "the computer's network allows only 10.1.0.0/24, so it can't reach the gateway at 10.0.0.7 its sites are on")
    assert sites.reachable_problem(NetworkPolicy(mode="allowlist", allow_hosts=("*.internal", "10.0.0.7")),
                                   routes) is None


@pytest.mark.asyncio
async def test_a_sandbox_that_is_the_container_is_reached_as_root_through_its_own_exec():
    calls = []

    async def exec_with_output(*argv):
        calls.append(argv)
        return 0, {"python3": json.dumps(_REPORT), "id": "0\n"}.get(argv[0], ""), ""

    async def write_file_from_text(content, path):
        calls.append(("write", path))

    sandbox = SimpleNamespace(url_from_sandbox=lambda url: url, exec_with_output=exec_with_output,
                              write_file_from_text=write_file_from_text)

    await sites.route(sandbox, [_site("notes", "notes.test")], user="user", network=None,
                      valid_for=timedelta(hours=1), label="desktop", os_type="linux", display="x11")

    assert calls == [("id", "-u"), ("mkdir", "-p", "/etc/agentenv/sites"), ("chmod", "0700", "/etc/agentenv/sites"),
                     ("mkdir", "-p", "/etc/agentenv/sites"), ("write", "/etc/agentenv/sites/site_proxy.py"),
                     ("mkdir", "-p", "/etc/agentenv/sites"), ("write", "/etc/agentenv/sites/config.json"),
                     ("chmod", "0600", "/etc/agentenv/sites/config.json"),
                     ("python3", "/etc/agentenv/sites/site_proxy.py", "install", "/etc/agentenv/sites/config.json")]


@pytest.mark.asyncio
async def test_a_container_sandbox_that_runs_the_computer_as_its_user_is_refused_before_anything_is_written():
    calls = []

    async def exec_with_output(*argv):
        calls.append(argv)
        return 0, "1000\n", ""

    async def write_file_from_text(content, path):
        calls.append(("write", path))

    sandbox = SimpleNamespace(type="coreweave", url_from_sandbox=lambda url: url, exec_with_output=exec_with_output,
                              write_file_from_text=write_file_from_text)

    with pytest.raises(RuntimeError, match="the coreweave sandbox runs the computer's commands as uid 1000"):
        await sites.route(sandbox, [_site("notes", "notes.test")], user="user", network=None,
                          valid_for=timedelta(hours=1), label="desktop", os_type="linux", display="x11")

    assert calls == [("id", "-u")]


@pytest.mark.asyncio
@pytest.mark.parametrize("domains", [("gw.sites.test",), ("*.sites.test",)])
async def test_a_site_naming_the_gateway_its_routes_are_on_is_refused_before_anything_is_installed(domains):
    sandbox = FakeVm(_REPORT)
    sandbox.url_from_sandbox = lambda url: url.replace("https://gw.example", "http://gw.sites.test:8080")

    with pytest.raises(RuntimeError, match=r"the gateway at gw.sites.test \(named by notes's .*sites.test\) would"):
        await sites.route(sandbox, [_site("notes", *domains)], user="user", network=None,
                          valid_for=timedelta(hours=1), label="desktop", os_type="linux", display="x11")

    assert sandbox.scripts == [] and sandbox.written == {}


@pytest.mark.asyncio
async def test_the_sites_a_run_deployed_are_read_from_each_gateways_records_and_their_cards(monkeypatch):
    from agentenv_protocol import client as protocol_v1

    cards = {"https://gw.example/svc/notes": {"capabilities": {"extensions": [
                 {"uri": "urn:agentenv:website/v1", "params": {"domains": ["notes.test"]}}]}},
             "https://gw.example/svc/gh": {"capabilities": {"extensions": [
                 {"uri": "urn:agentenv:website/v1", "params": {"domains": ["github.com"]}}]}}}

    async def get_card(url, timeout=10, verify=True):
        if url not in cards:
            raise _status_error(url, 404)
        return cards[url]

    monkeypatch.setattr(protocol_v1, "get_card", get_card)
    from agent_env.env.env import DeployedGatewayEnv
    gateway = DeployedGatewayEnv(env_id="sites", env_version=1, sandbox_id="s", gateway_url="https://gw.example",
                                 website_frontend_urls={name: f"https://gw.example/website/{name}/"
                                                        for name in ("notes", "gh", "legacy")})
    computer = SimpleNamespace(env_id="desktop")

    found = await sites.deployed_sites([gateway, computer])

    assert found == [
        {"name": "notes", "url": "https://gw.example/website/notes/", "domains": ["notes.test"]},
        {"name": "gh", "url": "https://gw.example/website/gh/", "domains": ["github.com"]},
        {"name": "legacy", "url": "https://gw.example/website/legacy/", "domains": []}]
    assert await sites.deployed_sites([gateway, computer], only=["desktop"]) == []
    with pytest.raises(ValueError, match="site_routing names other, which this run hasn't deployed"):
        await sites.deployed_sites([gateway], only=["other"])


@pytest.mark.asyncio
async def test_two_envs_each_serving_a_site_of_one_name_are_routed_apart_and_one_domain_twice_is_still_refused():
    from agent_env.env.env import DeployedGatewayEnv

    def gateway(env_id, *domains):
        extension = {"uri": "urn:agentenv:website/v1",
                     "params": {"domains": list(domains), "browse_url": "/website/notes/"}}
        card = {"name": env_id, "children_environments": [
            {"name": "notes", "url": "/svc/notes/agentenv", "capabilities": {"extensions": [extension]}}]}
        return DeployedGatewayEnv(env_id=env_id, env_version=1, sandbox_id=env_id,
                                  gateway_url=f"https://{env_id}.example", environment_card=card)

    found = await sites.deployed_sites([gateway("team.a", "notes.test"), gateway("team-b", "notes.example")])

    assert [(s["name"], s["url"], s["domains"]) for s in found] == [
        ("team-a-notes", "https://team.a.example/website/notes/", ["notes.test"]),
        ("team-b-notes", "https://team-b.example/website/notes/", ["notes.example"])]
    table = sites.site_table([Site(**s) for s in found])
    assert {route.site: route.domains for route in table.routes} == {"team-a-notes": ("notes.test",),
                                                                      "team-b-notes": ("notes.example",)}
    assert set(sites.issue(table.routes, valid_for=timedelta(hours=1), label="x").leaves) == {"team-a-notes",
                                                                                              "team-b-notes"}
    clash = await sites.deployed_sites([gateway("team-a", "notes.test"), gateway("team-b", "notes.test")])
    with pytest.raises(sites.SiteConflict, match="'notes.test' \\(team-b-notes\\) overlaps 'notes.test' \\(team-a-notes\\)"):
        sites.site_table([Site(**s) for s in clash])


@pytest.mark.asyncio
async def test_the_sites_a_run_deployed_are_its_stored_cards_website_children_at_the_routes_the_card_names(
        monkeypatch):
    from agentenv_protocol import client as protocol_v1

    from agent_env.env.env import DeployedGatewayEnv

    def website(name, browse=None, **params):
        extension = {"uri": "urn:agentenv:website/v1",
                     "params": {**params, **({"browse_url": browse} if browse else {})}}
        return {"name": name, "url": f"/svc/{name}/agentenv", "capabilities": {"extensions": [extension]}}

    card = {"name": "env", "children_environments": [
        website("notes", "/website/notes/", domains=["notes.test"]),
        {"name": "mcp-notes", "url": "/svc/mcp-notes/agentenv", "capabilities": {"extensions": []}},
        website("gh", domains=["github.com"]),
        website("orphan", domains=["orphan.test"])]}
    reads = []

    async def get_card(url, timeout=10, verify=True):
        reads.append(url)
        raise _status_error(url, 404)

    monkeypatch.setattr(protocol_v1, "get_card", get_card)
    gateway = DeployedGatewayEnv(env_id="sites", env_version=1, sandbox_id="s", gateway_url="https://gw.example",
                                 environment_card=card,
                                 website_frontend_urls={name: f"https://gw.example/website/{name}/"
                                                        for name in ("gh", "legacy")})

    found = await sites.deployed_sites([gateway])

    assert found == [
        {"name": "notes", "url": "https://gw.example/website/notes/", "domains": ["notes.test"]},
        {"name": "gh", "url": "https://gw.example/website/gh/", "domains": ["github.com"]},
        {"name": "legacy", "url": "https://gw.example/website/legacy/", "domains": []}]
    assert reads == ["https://gw.example/svc/legacy"]


@pytest.mark.asyncio
async def test_a_card_child_named_apart_from_its_service_is_matched_to_its_frontend_by_its_route(monkeypatch):
    from agentenv_protocol import client as protocol_v1

    from agent_env.env.env import DeployedGatewayEnv

    async def get_card(url, timeout=10, verify=True):
        raise AssertionError(f"read {url} live")

    monkeypatch.setattr(protocol_v1, "get_card", get_card)
    card = {"name": "env", "children_environments": [{"name": "Notes", "url": "/svc/notes/agentenv", "capabilities": {
        "extensions": [{"uri": "urn:agentenv:website/v1", "params": {"domains": ["notes.test"]}}]}}]}
    gateway = DeployedGatewayEnv(env_id="sites", env_version=1, sandbox_id="s", gateway_url="https://gw.example",
                                 environment_card=card,
                                 website_frontend_urls={"notes": "https://gw.example/website/notes/"})

    assert await sites.deployed_sites([gateway]) == [
        {"name": "Notes", "url": "https://gw.example/website/notes/", "domains": ["notes.test"]}]
    # Selected by the name it's served as, its environment_name, not its card's.
    assert await sites.deployed_sites([gateway], services=["notes"]) == await sites.deployed_sites([gateway])
    assert await sites.deployed_sites([gateway], services=["Notes"]) == []


@pytest.mark.asyncio
async def test_two_websites_of_one_env_whose_cards_share_a_name_are_each_named_as_theyre_served(monkeypatch):
    from agentenv_protocol import client as protocol_v1

    from agent_env.env.env import DeployedGatewayEnv

    async def get_card(url, timeout=10, verify=True):
        raise AssertionError(f"read {url} live")

    monkeypatch.setattr(protocol_v1, "get_card", get_card)

    def child(served: str, domain: str) -> dict:
        return {"name": "Site", "url": f"/svc/{served}/agentenv", "capabilities": {
            "extensions": [{"uri": "urn:agentenv:website/v1", "params": {"domains": [domain]}}]}}

    card = {"name": "env", "children_environments": [child("shop", "shop.test"), child("blog", "blog.test")]}
    gateway = DeployedGatewayEnv(env_id="sites", env_version=1, sandbox_id="s", gateway_url="https://gw.example",
                                 environment_card=card,
                                 website_frontend_urls={name: f"https://gw.example/website/{name}/" for name in ("shop", "blog")})

    found = await sites.deployed_sites([gateway])

    assert found == [{"name": "shop", "url": "https://gw.example/website/shop/", "domains": ["shop.test"]},
                     {"name": "blog", "url": "https://gw.example/website/blog/", "domains": ["blog.test"]}]
    assert len(sites.site_table([Site(**site) for site in found]).routes) == 2  # names of their own: both route
    assert await sites.deployed_sites([gateway], services=["blog"]) == [
        {"name": "Site", "url": "https://gw.example/website/blog/", "domains": ["blog.test"]}]  # alone, its card's name


@pytest.mark.asyncio
async def test_a_website_is_selected_by_the_environment_name_its_gateway_states(monkeypatch):
    from agentenv_protocol import client as protocol_v1

    from agent_env.env.env import DeployedGatewayEnv

    async def get_card(url, timeout=10, verify=True):
        raise AssertionError(f"read {url} live")

    monkeypatch.setattr(protocol_v1, "get_card", get_card)
    card = {"name": "env", "children_environments": [{
        "name": "Notes", "url": "/sites/notes-v2/agentenv", "environment_name": "notes", "capabilities": {
            "extensions": [{"uri": "urn:agentenv:website/v1", "params": {"domains": ["notes.test"]}}]}}]}
    gateway = DeployedGatewayEnv(env_id="sites", env_version=1, sandbox_id="s", gateway_url="https://gw.example",
                                 environment_card=card,
                                 website_frontend_urls={"notes": "https://gw.example/website/notes/"})

    found = [{"name": "Notes", "url": "https://gw.example/website/notes/", "domains": ["notes.test"]}]
    assert await sites.deployed_sites([gateway], services=["notes"]) == found  # not the path's notes-v2
    assert await sites.deployed_sites([gateway], services=["notes-v2"]) == []


@pytest.mark.asyncio
async def test_a_card_that_fails_to_load_is_retried_then_fails_rather_than_dropping_the_sites_names(monkeypatch):
    from agentenv_protocol import client as protocol_v1

    from agent_env.env.env import DeployedGatewayEnv

    card = {"capabilities": {"extensions": [{"uri": "urn:agentenv:website/v1", "params": {"domains": ["notes.test"]}}]}}
    answers = [httpx.ReadTimeout("slow"), _status_error("https://gw.example/svc/notes", 503), card]

    async def get_card(url, timeout=10, verify=True):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(protocol_v1, "get_card", get_card)
    monkeypatch.setattr(sites, "CARD_RETRY_S", 0)
    gateway = DeployedGatewayEnv(env_id="sites", env_version=1, sandbox_id="s", gateway_url="https://gw.example",
                                 website_frontend_urls={"notes": "https://gw.example/website/notes/"})

    (found,) = await sites.deployed_sites([gateway])

    assert found["domains"] == ["notes.test"] and answers == []
    answers.extend([_status_error("https://gw.example/svc/notes", 500)] * sites.CARD_ATTEMPTS)
    with pytest.raises(RuntimeError, match="the card of the site notes at https://gw.example/svc/notes couldn't be "
                                           "read, so its names are unknown: HTTPStatusError"):
        await sites.deployed_sites([gateway])


def _status_error(url, code):
    request = httpx.Request("GET", f"{url}/.well-known/agent-env.json")
    return httpx.HTTPStatusError(f"{code}", request=request, response=httpx.Response(code, request=request))


class _Website:
    """A deployed env serving a website, whose ``readiness`` answers each of ``answers`` in turn, then the last again: a
    readiness, an error to raise, or None for a record with no address."""

    def __init__(self, env_id, *answers, card=None):
        self.env_id, self.answers, self.asked = env_id, list(answers), 0
        self.environment_card = card
        self.website_frontend_urls = None if card is not None else {"site": f"http://gw/website/{env_id}/"}

    async def readiness(self):
        from agent_env.env.env import EnvCapabilityUnsupported
        from agentenv_protocol.website import WebsiteReadiness

        answer = self.answers[min(self.asked, len(self.answers) - 1)]
        self.asked += 1
        if answer is None:
            raise EnvCapabilityUnsupported("urn:agentenv:website/v1", "ready", self.env_id, None)
        if isinstance(answer, Exception):
            raise answer
        return WebsiteReadiness(ready=not answer, problems=answer)


@pytest.mark.asyncio
async def test_the_websites_a_computer_routes_are_waited_on_till_ready_one_never_ready_named_and_others_not_asked():
    late = _Website("late", ["notes: warming up"], [])
    shop = _Website("shop", [])
    other = _Website("other", ["never asked"])
    unaddressed = _Website("unaddressed", None)

    on_its_card = _Website("on-its-card", [], card={"name": "kit", "children_environments": [{
        "name": "notes", "capabilities": {"extensions": [{"uri": "urn:agentenv:website/v1", "params": {}}]}}]})
    mcp_only = _Website("mcp-only", httpx.ConnectError("never asked"), card={"name": "mcp", "children_environments": [
        {"name": "tools", "capabilities": {"extensions": []}}]})

    await sites.require_ready([late, shop, other, unaddressed], ["late", "shop", "unaddressed"], timeout_s=5,
                              poll_s=0.01)
    await sites.require_ready([on_its_card, mcp_only], timeout_s=0.05, poll_s=0.01)
    broken = _Website("broken", ["shop: its database is down"])
    unreachable = _Website("unreachable", httpx.ConnectError("refused"))
    with pytest.raises(RuntimeError) as never:
        await sites.require_ready([broken, unreachable, shop], timeout_s=0.05, poll_s=0.01)

    assert (late.asked, shop.asked, other.asked, unaddressed.asked) == (2, 2, 0, 1)
    assert (on_its_card.asked, mcp_only.asked) == (1, 0)
    assert str(never.value).startswith("site routing: the run's websites weren't ready after 0.05 s: ")
    assert str(never.value).endswith("broken (shop: its database is down); unreachable (ConnectError: refused)")


def test_routing_names_each_site_that_didnt_load_runtime_that_didnt_trust_the_ca_and_jdk_store_that_didnt_take_it():
    report = {"checks": {"notes.test": {"url": "https://notes.test/", "status": 200},
                         "chat.test": {"url": "https://chat.test/", "error": "502 Bad Gateway: no route"}},
              "runtimes": {"node": {"url": "https://notes.test/", "status": 200},
                           "curl": {"url": "https://notes.test/", "error": "curl: (60) SSL certificate problem"}},
              "trust": {"java": {"/usr/lib/jvm/a/lib/security/cacerts": {"trusted": True},
                                 "/opt/b/lib/security/cacerts": {"error": "keystore password was incorrect"}}}}

    assert sites.routing_problems(report) == [
        "https://chat.test/ didn't load in the computer: 502 Bad Gateway: no route",
        "curl in the computer didn't trust the deploy's CA loading https://notes.test/: curl: (60) SSL certificate "
        "problem",
        "the JDK store /opt/b/lib/security/cacerts didn't take the deploy's CA: keystore password was incorrect"]
    assert sites.routing_problems({"checks": {}, "trust": {}}) == []
    forward = {"proxy": "http://127.0.0.1:3129", "url": "https://chat.test/", "error": "[Errno 111] Connection refused"}
    assert sites.routing_problems({"checks": {}, "trust": {}, "forward_port": forward}) == [
        "the browser's forward port http://127.0.0.1:3129 didn't load https://chat.test/: "
        "[Errno 111] Connection refused"]
    loaded = {"proxy": "http://127.0.0.1:3129", "url": "https://chat.test/", "status": 200}
    assert sites.routing_problems({"checks": {}, "trust": {}, "forward_port": loaded}) == []


def test_the_trust_a_routing_linux_computer_boots_with_is_the_local_providers_so_the_two_always_agree():
    from agent_env.providers.sandbox_providers.local_docker.provider import LOCAL_TRUST_ENV

    assert sites.TRUST_ENVIRONMENTS == {"linux": LOCAL_TRUST_ENV} and LOCAL_TRUST_ENV == {
        "SSL_CERT_FILE": "/etc/agentenv/ca-bundle.pem", "REQUESTS_CA_BUNDLE": "/etc/agentenv/ca-bundle.pem",
        "NODE_EXTRA_CA_CERTS": "/etc/agentenv/ca.pem"}
    assert sites.trust_files("linux") == {"ca": "/etc/agentenv/ca.pem", "bundle": "/etc/agentenv/ca-bundle.pem"}


def test_a_browser_alone_routing_sites_is_named_the_forward_port_and_a_computer_with_a_display_is_named_none():
    assert sites.browser_environment("none") == {"AGENTENV_SITE_PROXY": "http://127.0.0.1:3129"}
    assert sites.browser_environment("x11") == {}


def test_a_computer_boots_with_the_trust_variables_only_when_a_site_it_routes_is_served_over_https():
    local = [Site(name="notes", url="http://gw/website/notes/", domains=["notes.localhost"]),
             Site(name="bare", url="http://gw/website/bare/")]
    secure = [*local, Site(name="shop", url="http://gw/website/shop/", domains=["shop.test"])]

    assert sites.trust_environment("linux", local) == {}
    assert sites.trust_environment("linux", secure) == sites.TRUST_ENVIRONMENTS["linux"]
