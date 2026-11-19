"""How an agent on the env's clock goes out, imported by every clocked agent's bridge, whose image copies it and
``clockpage`` beside it: the agent's CLI checks certificates against the env's date, which can be before their issue, so
every TLS host goes through the image's egress proxy, on real time, and only hosts with nothing to check go direct."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit

import clockpage

EGRESS_PROXY_ENV = "AGENTENV_EGRESS_PROXY"
_DIRECT_HOSTS = ("localhost", "127.0.0.1", "::1", "host.docker.internal")
_READ_TRIES, _RETRY_S = 50, 0.001


def clock_armed(page_path: str | None = None) -> bool:
    """Whether the clock page libfaketime reads, which keeps the last clock while the follower reconnects, holds an
    armed clock, so a CLI started now reads the env's time. A page mid-write or torn through every retry is taken at its
    armed flag, as the proxy relays untouched when unarmed; one that isn't a page or isn't there reads as unarmed."""
    try:
        with open(page_path or clockpage.PAGE, "rb") as page:
            for _ in range(_READ_TRIES):
                page.seek(0)
                raw = page.read(clockpage.LAYOUT.size)
                if len(raw) < clockpage.LAYOUT.size:
                    return False
                magic, version, seq, universe, anchor, rate, armed, _, generation, checked, mono_u, mono_a = \
                    clockpage.LAYOUT.unpack(raw)
                if (magic, version) != (clockpage.MAGIC, clockpage.VERSION):
                    return False
                if seq % 2 == 0 and checked == clockpage.check(seq, universe, anchor, rate, armed, generation, mono_u,
                                                               mono_a):
                    return bool(armed)
                time.sleep(_RETRY_S)
    except OSError:
        return False
    return bool(armed)


def egress_env(environ: Mapping[str, str], urls: Iterable[str] = (), *, armed: bool) -> dict[str, str]:
    """The proxy variables for the CLI and what it starts where the image runs the proxy and the clock is ``armed``,
    else none: every host goes through the proxy but loopback, which it refuses, the hosts the deploy's NO_PROXY names,
    and the hosts of plain-HTTP ``urls``, whose connections carry no certificate."""
    proxy = environ.get(EGRESS_PROXY_ENV)
    if not proxy or not armed:
        return {}
    configured = [host.strip() for name in ("NO_PROXY", "no_proxy") for host in (environ.get(name) or "").split(",")]
    plain = [parts.hostname for parts in map(urlsplit, urls) if parts.scheme == "http"]
    hosts = ",".join(dict.fromkeys(filter(None, [*_DIRECT_HOSTS, *configured, *plain])))
    return {**{name: proxy for base in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") for name in (base, base.lower())},
            "NO_PROXY": hosts, "no_proxy": hosts, "NODE_USE_ENV_PROXY": "1"}
