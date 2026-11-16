"""The ubuntu-24.04-browser base on the universe clock: its copies of the clock library and the egress proxy are
ubuntu-24.04's, byte for byte, so a change to either lands in both; it starts as root; root's supervisord runs the
egress proxy on the address the base names to its site proxies, one the stand-in site proxy doesn't hold, and init
boots it first; both site proxies read the egress proxy from the variable the base sets; the stand-in keeps real time;
and X has no screensaver or DPMS, whose timeouts would run on the universe's clock."""
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

ARTIFACTS = Path(agent_env.__file__).parent / "computers" / "artifacts"
DESKTOP, BROWSER = ARTIFACTS / "ubuntu-24.04", ARTIFACTS / "ubuntu-24.04-browser"


def _files(folder: Path) -> dict[str, bytes]:
    """The folder's files by path, but Python's caches, which importing a module there leaves."""
    return {str(path.relative_to(folder)): path.read_bytes() for path in sorted(folder.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


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
    conf.read_string((BROWSER / name).read_text())
    return conf


@pytest.mark.parametrize("part", ["clock", "egress"])
def test_the_browser_bases_copy_is_the_desktops_byte_for_byte(part):
    desktop, browser = _files(DESKTOP / part), _files(BROWSER / part)

    assert sorted(browser) == sorted(desktop), f"{part}/ differs in its files: copy ubuntu-24.04/{part}/ over it"
    assert [name for name in desktop if browser[name] != desktop[name]] == [], (
        f"{part}/ differs from ubuntu-24.04's: copy ubuntu-24.04/{part}/ over it")


def test_it_starts_as_root_whose_supervisord_runs_the_session_as_the_session_user():
    artifact = tomllib.loads((BROWSER / "artifact.toml").read_text())
    session = _conf("root.conf")["program:session"]

    assert artifact["forms"]["container"]["start_as"] == "root"
    assert (session["command"], session["user"]) == (
        "/usr/bin/supervisord -c /etc/supervisor/supervisord.conf", artifact["session"]["user"])


def test_the_egress_proxy_listens_where_the_base_names_it_and_init_boots_it_first():
    egress = _conf("root.conf")["program:egress"]
    command = shlex.split(egress["command"])
    init = (BROWSER / "start-computer").read_text()

    assert (command[:2], egress["user"]) == (["/usr/local/lib/agentenv-egress/egress", "serve"], "agentenv-egress")
    assert command[2:] == ["--listen", urlsplit(_env()[site_proxy.EGRESS_PROXY_ENV]).netloc]
    assert init.index("/usr/local/lib/agentenv-egress/egress boot") < init.index("exec /usr/bin/supervisord")


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
