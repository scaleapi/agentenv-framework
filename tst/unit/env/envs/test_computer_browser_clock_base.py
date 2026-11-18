"""The ubuntu-24.04-browser base on the universe clock: through its image_kits.toml it takes the clock library and the
egress proxy from utils, as ubuntu-24.04 does, and ubuntu-24.04's root start and root's supervisord, so a change
to any lands in both; it starts as root,
its init the root start; the egress proxy listens where the variable its site proxies read names, which the base sets
to an address the stand-in site proxy doesn't hold; the stand-in keeps real time; and X has no screensaver or DPMS,
whose timeouts would run on the universe's clock."""
from __future__ import annotations

import configparser
import importlib.util
import re
import shlex
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import agent_env
from agent_env.env.envs.computer.utils import site_proxy
from tst.util.build_context import sent

ARTIFACTS = Path(agent_env.__file__).parent / "computers" / "artifacts"
DESKTOP, BROWSER = ARTIFACTS / "ubuntu-24.04", ARTIFACTS / "ubuntu-24.04-browser"
SOURCES = Path(agent_env.__file__).parent


def _stand_in():
    spec = importlib.util.spec_from_file_location("stand_in_site_proxy", BROWSER / "site-proxy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env() -> dict[str, str]:
    """The browser base's image environment, as its Dockerfile's ENV sets it."""
    line = re.search(r"^ENV (LANG=.+?)$(?:\n {4}(.+?)$)?", (BROWSER / "Dockerfile").read_text(), re.MULTILINE)
    words = shlex.split(" ".join(part.rstrip(" \\") for part in line.groups() if part))
    return dict(word.split("=", 1) for word in words)


def _conf(name: str) -> configparser.ConfigParser:
    conf = configparser.ConfigParser(interpolation=None)
    conf.read_string(sent(BROWSER, name).read_text())
    return conf


@pytest.mark.parametrize(("base", "part", "shared"), [
    (DESKTOP, "clock", SOURCES / "utils" / "faketime_clock"), (DESKTOP, "egress", SOURCES / "utils" / "egress_proxy"),
    (BROWSER, "clock", SOURCES / "utils" / "faketime_clock"), (BROWSER, "egress", SOURCES / "utils" / "egress_proxy"),
    (BROWSER, "root", DESKTOP / "root"), (BROWSER, "root.conf", DESKTOP / "root.conf")])
def test_what_the_bases_share_is_taken_from_its_one_copy(base, part, shared):
    taken = tomllib.loads((base / "image_kits.toml").read_text())

    assert not (base / part).exists(), f"{base.name}/{part} is a copy: take it in {base.name}/image_kits.toml instead"
    assert (SOURCES / taken[part]).resolve() == shared.resolve()


def test_it_starts_as_root_its_init_the_root_start_whose_supervisord_runs_the_session_as_the_session_user():
    artifact = tomllib.loads((BROWSER / "artifact.toml").read_text())
    session = _conf("root.conf")["program:session"]
    init = (BROWSER / "start-computer").read_text()

    assert artifact["forms"]["container"]["start_as"] == "root"
    assert "    init)\n        exec /usr/local/lib/agentenv-root/init\n" in init
    assert (session["command"], session["user"]) == (
        "/usr/bin/supervisord -c /etc/supervisor/supervisord.conf", artifact["session"]["user"])


def test_the_egress_proxy_listens_where_the_variable_its_site_proxies_read_names():
    named = re.search(r'ProxyEnv\s*=\s*"([^"]+)"', sent(BROWSER, "egress/main.go").read_text()).group(1)

    assert named == site_proxy.EGRESS_PROXY_ENV and named in _env()
    assert shlex.split(_conf("root.conf")["program:egress"]["command"]) == [
        "/usr/local/lib/agentenv-egress/egress", "serve"]


def test_the_stand_in_holds_the_site_proxy_address_the_base_names_and_the_egress_proxy_another():
    stand_in, env = _stand_in(), _env()

    assert urlsplit(env["AGENTENV_SITE_PROXY"]).port == stand_in.PORT
    assert urlsplit(env[site_proxy.EGRESS_PROXY_ENV]).port not in (stand_in.PORT, 3128)
    assert stand_in.EGRESS_PROXY_ENV == site_proxy.EGRESS_PROXY_ENV


def test_the_stand_in_keeps_real_time_for_its_network_log():
    assert _conf("supervisord.conf")["program:site-proxy"]["environment"] == 'AGENTENV_REAL_TIME="1"'


def test_x_has_no_screensaver_or_dpms_whose_timeouts_run_on_the_universes_clock():
    xvfb = shlex.split(next(line for line in (BROWSER / "start-computer").read_text().splitlines()
                            if "exec Xvfb" in line))

    assert xvfb[xvfb.index("-s") + 1] == "0" and "-dpms" in xvfb
    assert _conf("supervisord.conf")["program:xvfb"]["command"] == "/usr/local/bin/start-computer xvfb"
