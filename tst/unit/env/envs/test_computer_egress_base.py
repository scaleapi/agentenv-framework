"""The ubuntu-24.04 base's egress proxy agrees with the rest of the base and with site routing: its certificates span
the window routed sites' do, root's supervisord runs it as its own user, init boots it before that supervisord starts,
every session program trusts its CA through agentenv's trust files as the local provider names them, the session's
proxy is the address it listens on, a login shell gets the session's settings, and the clock pages its Go reader is
tested on are the bytes the follower's writer writes."""
from __future__ import annotations

import configparser
import importlib.util
import json
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import agent_env
from agent_env.env.envs.computer.utils import site_proxy, sites
from agent_env.providers.sandbox_providers.local_docker.provider import LOCAL_TRUST_ENV

BASE = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04"
EGRESS = BASE / "egress"


def _go_date(name: str) -> datetime:
    match = re.search(rf"{name}\s*=\s*time\.Date\(([^)]*)\)", (EGRESS / "main.go").read_text())
    year, month, day, hour, minute, second, _ns, _zone = (part.strip() for part in match.group(1).split(","))
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=UTC)


def _session_environment() -> dict[str, str]:
    conf = configparser.ConfigParser(interpolation=None)
    conf.read_string((BASE / "supervisord.conf").read_text())
    lexer = shlex.shlex(conf["supervisord"]["environment"], posix=True)
    lexer.whitespace += ","
    lexer.wordchars += "/.:+-()|=_"
    pairs = [token.split("=", 1) for token in lexer]
    return {key: value for key, value in pairs}


def _profile() -> dict[str, str]:
    out = {}
    for line in (EGRESS / "agentenv-egress.sh").read_text().splitlines():
        if line.startswith("export "):
            for token in shlex.split(line.removeprefix("export ")):
                key, _, value = token.partition("=")
                out[key] = value
    return out


def test_the_egress_ca_spans_the_window_routed_sites_certificates_do():
    assert (_go_date("CAFrom"), _go_date("CAUntil")) == (sites.UNIVERSE_FROM, sites.UNIVERSE_UNTIL)


def test_roots_supervisord_runs_the_proxy_as_its_own_user_and_init_boots_it_first():
    root = configparser.ConfigParser(interpolation=None)
    root.read_string((BASE / "root.conf").read_text())
    init = (BASE / "start-desktop").read_text()

    assert (root["program:egress"]["command"], root["program:egress"]["user"]) == (
        "/usr/local/lib/agentenv-egress/egress serve", "agentenv-egress")
    assert init.index("/usr/local/lib/agentenv-egress/egress boot") < init.index("exec /usr/bin/supervisord")


def test_a_login_shell_gets_the_proxy_settings_every_session_program_gets():
    session, profile = _session_environment(), _profile()

    assert {key: session[key] for key in profile} == profile
    assert session["HTTPS_PROXY"] == "http://127.0.0.1:3128" and session["NODE_USE_ENV_PROXY"] == "1"
    assert "-Dhttps.proxyPort=3128" in session["JAVA_TOOL_OPTIONS"]
    assert session["NO_PROXY"].split(",") == session["JAVA_TOOL_OPTIONS"].split("nonProxyHosts=")[1].split("|")
    assert {key: session[key] for key in LOCAL_TRUST_ENV} == LOCAL_TRUST_ENV


def test_the_session_goes_out_through_the_address_the_proxy_listens_on():
    listen = re.search(r'Listen\s*=\s*"([^"]+)"', (EGRESS / "main.go").read_text()).group(1)

    assert urlsplit(_session_environment()["HTTPS_PROXY"]).netloc == listen


def test_the_pages_the_proxy_is_tested_on_are_the_ones_the_followers_writer_writes():
    spec = importlib.util.spec_from_file_location("clockpage_written", BASE / "clock" / "clockpage.py")
    page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(page)
    golden = {name: bytes.fromhex((EGRESS / "testdata" / f"page-{name}.hex").read_text().strip())
              for name in ("armed", "unarmed")}

    assert golden == {
        "armed": page.encode(4, 1577836800_000000000, 1794000000_123456789, 2.0, 1, 7, 5_000_000_000, 5_000_000_000),
        "unarmed": page.encode(2, 0, 0, 1.0, 0, 3, 0, 0)}


def test_every_chromium_build_goes_out_through_the_proxy_by_a_policy_that_bypasses_what_the_session_does():
    policy = json.loads((EGRESS / "chromium-policy.json").read_text())
    session = _session_environment()
    loop = re.search(r"for build in (.+?); do", (BASE / "Dockerfile").read_text()).group(1).split()

    assert policy["ProxySettings"] == {
        "ProxyMode": "fixed_servers", "ProxyServer": session["HTTPS_PROXY"],
        "ProxyBypassList": ";".join(f"[{host}]" if ":" in host else host for host in session["NO_PROXY"].split(","))}
    assert policy["WebRtcIPHandling"] == "disable_non_proxied_udp"
    assert {str(path.parents[2]) for path in site_proxy.Paths().chromium_policies} == set(loop)


def test_x_has_no_screensaver_or_dpms_whose_timeouts_run_on_real_time():
    xvfb = shlex.split(next(line for line in (BASE / "start-desktop").read_text().splitlines() if "exec Xvfb" in line))

    assert xvfb[xvfb.index("-s") + 1] == "0" and "-dpms" in xvfb
