"""The egress rule every agent on the env's clock goes out by (utils/faketime_clock/agent_egress.py), imported by its
own name with clockpage beside it, as each clocked agent's image copies them beside its bridge: on an armed clock, every
host but loopback, the deploy's NO_PROXY hosts and plain-HTTP hosts goes through the proxy; unarmed, nothing changes."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import agent_env

sys.path.insert(0, str(Path(agent_env.__file__).parent / "utils" / "faketime_clock"))
agent_egress = importlib.import_module("agent_egress")
clockpage = importlib.import_module("clockpage")

PROXY = "http://127.0.0.1:3128"
LOOPBACK = ["localhost", "127.0.0.1", "::1", "host.docker.internal"]


def _direct(environ: dict[str, str] | None = None, urls: list[str] | None = None) -> list[str]:
    env = agent_egress.egress_env({agent_egress.EGRESS_PROXY_ENV: PROXY, **(environ or {})}, urls or [], armed=True)
    assert env["NO_PROXY"] == env["no_proxy"]
    return env["NO_PROXY"].split(",")


def test_on_an_armed_clock_everything_goes_through_the_proxy_but_loopback():
    env = agent_egress.egress_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
                                   "all_proxy")} == {PROXY}
    assert env["NODE_USE_ENV_PROXY"] == "1"
    assert _direct() == LOOPBACK


@pytest.mark.parametrize("url, direct", [
    ("http://gateway:8080/mcp", True),
    ("http://10.0.0.5:4000", True),
    ("HTTP://Gateway.Internal/mcp", True),
    ("https://abc-123.w.modal.host/mcp", False),
    ("https://10.0.0.5:4000", False),
    ("https://[fd00::5]:4000", False),
    ("https://litellm.example.com", False),
])
def test_a_plain_http_host_goes_direct_and_a_tls_one_through_the_proxy(url, direct):
    assert (urlsplit(url).hostname in _direct(urls=[url])) is direct


def test_the_hosts_the_deploy_names_in_no_proxy_stay_direct():
    hosts = _direct({"NO_PROXY": "registry.internal, .corp", "no_proxy": "mirror.internal"})

    assert hosts == [*LOOPBACK, "registry.internal", ".corp", "mirror.internal"]


def test_empty_and_duplicate_hosts_are_dropped():
    hosts = _direct({"NO_PROXY": "gateway,, localhost,", "no_proxy": " gateway "},
                    ["http://gateway/mcp", "", "http:///mcp", "http://gateway:9000", "http://127.0.0.1:8000/peers/mcp"])

    assert hosts == [*LOOPBACK, "gateway"]


@pytest.mark.parametrize("environ, armed", [
    pytest.param({agent_egress.EGRESS_PROXY_ENV: PROXY}, False, id="unarmed"),
    pytest.param({}, True, id="no-proxy-in-the-image"),
    pytest.param({agent_egress.EGRESS_PROXY_ENV: ""}, True, id="empty-proxy"),
])
def test_unarmed_or_without_the_proxy_nothing_changes(environ, armed):
    assert agent_egress.egress_env({**environ, "NO_PROXY": "registry.internal"}, ["http://gateway/mcp"],
                                   armed=armed) == {}


def _page(armed: int, seq: int = 2) -> bytes:
    return clockpage.encode(seq, 1_900_000_000 * 10**9 if armed else 0, 10**18 if armed else 0, 1.0, armed, 3, 0, 0)


def _torn(page: bytes) -> bytes:
    """``page`` with a field changed after its check was written."""
    return page[:16] + bytes([page[16] ^ 1]) + page[17:]


@pytest.mark.parametrize("page, armed", [
    pytest.param(_page(1), True, id="armed"),
    pytest.param(_page(0), False, id="unarmed"),
    pytest.param(_page(1, seq=3), True, id="its-writer-died-mid-write"),
    pytest.param(_torn(_page(1)), False, id="torn"),
    pytest.param(_page(1)[:40], False, id="short"),
    pytest.param(b"\0" * 80, False, id="no-magic"),
    pytest.param(bytes(range(80)), False, id="garbage"),
    pytest.param(None, False, id="no-page"),
])
def test_the_clock_is_armed_when_the_page_libfaketime_reads_says_so(tmp_path, page, armed):
    path = tmp_path / "agentenv-clock"
    if page is not None:
        path.write_bytes(page)
    assert agent_egress.clock_armed(str(path)) is armed


def test_it_reads_the_page_clockpage_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(clockpage, "PAGE", str(tmp_path / "agentenv-clock"))
    monkeypatch.setattr(clockpage, "FILE", str(tmp_path / "clock"))

    clockpage.write(1, 1_900_000_000.0, 1.0, time.time())
    assert agent_egress.clock_armed()
    clockpage.write(0, 0.0, 1.0, time.time())
    assert not agent_egress.clock_armed()
