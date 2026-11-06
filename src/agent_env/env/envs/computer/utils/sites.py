"""Websites in the computer: the browser in a computer opens a site by the names it declares (``https://notes.test/``)
and reaches it on the env gateway, at the site's route (``<gateway>/website/notes/``), with TLS it trusts, while every
other host follows the computer's network.

A deploy with ``env_params.site_routing`` is handed the run's deployed websites (``deployed_sites``), as ``sites``. Each
name is claimed once (``site_table``): a site's names are routed and the network's allowlist is left to the network,
a site's name winning over the allowlist; two sites claiming a name are refused. Once the computer is booted, a CA made for the deploy signs a certificate for each
site's names (``issue``), and the site proxy (``site_proxy.py``) is installed through the sandbox's root channel and
started (``route``), which a reset does again on its fresh computer. In the computer they live in
``/etc/agentenv/sites``, readable by root alone but the CA, ``ca.pem``, at a fixed path for an app with a certificate
list of its own: the folder is root's alone (0700) before the config and its keys are written into it, until the
install has made the keys and the config private; the CA and certificates are valid for the deploy's ttl and a day more, from a few minutes back for a
clock that's behind."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import shlex
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable, Optional
from urllib.parse import urlsplit

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from agent_env.env.envs.computer.utils.site_proxy import is_local, matches
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, VmSandbox

if TYPE_CHECKING:
    from agent_env.env.env import DeployedEnv
    from agent_env.env.envs.computer.utils.params import Site
    from agent_env.providers.sandbox_providers.sandbox import Sandbox

FOLDER = PurePosixPath("/etc/agentenv/sites")
PROXY = FOLDER / "site_proxy.py"
CONFIG = FOLDER / "config.json"
CA_PATH = FOLDER / "ca.pem"
LOG = FOLDER / "site-proxy.log"
SYNTHETIC, ALLOWLIST = "synthetic", "allowlist"
RANKS = (SYNTHETIC, ALLOWLIST)
VALID_MARGIN = timedelta(days=1)
CLOCK_SKEW = timedelta(minutes=5)
CARD_ATTEMPTS = 3
CARD_RETRY_S = 1.0
_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
_PROXY_SOURCE = Path(__file__).with_name("site_proxy.py")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Claim:
    """A name and who claims it: a site, or the network's allowlist, at its rank."""

    name: str
    rank: str
    owner: str


@dataclass(frozen=True)
class Route:
    """A synthetic site as the computer serves it: its names, and its route on the gateway as the computer reaches
    it."""

    site: str
    url: str
    domains: tuple[str, ...]

    @property
    def tls_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.domains if not is_local(name))


@dataclass
class SiteTable:
    """Every name of the deployment: the routes, the names left to the network, and each claim a stronger one
    shadows."""

    routes: list[Route] = field(default_factory=list)
    unrouted: dict[str, str] = field(default_factory=dict)
    shadowed: list[tuple[Claim, Claim]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"routes": {route.site: list(route.domains) for route in self.routes}, "unrouted": self.unrouted,
                "shadowed": [{"name": weak.name, "rank": weak.rank, "owner": weak.owner, "by": strong.owner}
                             for weak, strong in self.shadowed]}


class SiteConflict(ValueError):
    """Two sites of one kind claim a name."""


def name_problem(name: str) -> Optional[str]:
    """Why ``name`` can't be a site's name, or None: a DNS name of letters, digits and hyphens, a wildcard only as its
    whole first label, and no address."""
    if name != name.lower() or name.endswith("."):
        return f"{name!r} must be lowercase, with no trailing dot"
    try:
        ipaddress.ip_address(name)
        return f"{name!r} is an address, not a name"
    except ValueError:
        pass
    labels = name.split(".")
    if labels[0] == "*":
        labels = labels[1:]
        if len(labels) < 2:
            return f"{name!r} is a wildcard over a single label"
    if len(name) > 253 or not labels or not all(_LABEL.match(label) for label in labels):
        return f"{name!r} isn't a DNS name (letters, digits and hyphens, a '*' only as its whole first label)"
    return None


def overlap(a: str, b: str) -> bool:
    """Whether a host could be named by both ``a`` and ``b``, a wildcard covering one label."""
    if a == b:
        return True
    for wild, other in ((a, b), (b, a)):
        if wild.startswith("*.") and not other.startswith("*.") and other.partition(".")[2] == wild[2:]:
            return True
    return False


def covers(strong: str, weak: str) -> bool:
    """Whether every host ``weak`` names is one ``strong`` names."""
    return strong == weak or (strong.startswith("*.") and not weak.startswith("*.")
                              and weak.partition(".")[2] == strong[2:])


def fallback_name(site: str) -> str:
    """The name a synthetic site declaring none is reached by: its name on ``.localhost``, lowercase, each ``_`` a
    ``-``, and no ``-`` at either end of the label (``My_Site`` is ``my-site.localhost``). Two sites it gives one name
    are refused as any two claiming it."""
    return f"{site.lower().replace('_', '-')[:63].strip('-')}.localhost"


def claims(sites: Iterable[Site], allow_hosts: Iterable[str] = ()) -> list[Claim]:
    """Each name claimed: a site's, its own or ``fallback_name`` when it declares none, and the network's allowlist."""
    found = [Claim(name, SYNTHETIC, site.name) for site in sites
             for name in dict.fromkeys(site.domains or [fallback_name(site.name)])]
    return found + [Claim(host.lower(), ALLOWLIST, "network") for host in dict.fromkeys(allow_hosts)]


def site_table(sites: list[Site], allow_hosts: Iterable[str] = ()) -> SiteTable:
    """The deployment's names: a synthetic site's routed, the rest left to the network but those a stronger claim
    covers, a stronger claim shadowing each weaker one that overlaps it. Raises SiteConflict naming every name two sites
    of one kind claim, and ValueError for a name that can't be one or a site named twice."""
    every = claims(sites, allow_hosts)
    names = [site.name for site in sites]
    invalid = [f"{claim.owner}: {problem}" for claim in every if claim.rank != ALLOWLIST
               and (problem := name_problem(claim.name))]
    invalid += [f"two sites are named {name!r}" for name in dict.fromkeys(names) if names.count(name) > 1]
    invalid += [f"{name!r} isn't a site's name (letters, digits, '-' and '_')" for name in names
                if not re.fullmatch(r"[A-Za-z0-9_-]+", name)]
    if invalid:
        raise ValueError("; ".join(invalid))
    conflicts, kept, table = [], [], SiteTable()
    for rank in RANKS:
        for claim in (c for c in every if c.rank == rank):
            same = [k for k in kept if k.rank == rank and k.owner != claim.owner and overlap(k.name, claim.name)]
            conflicts += [f"{claim.name!r} ({claim.owner}) overlaps {k.name!r} ({k.owner})" for k in same]
            stronger = [k for k in kept if k.rank != rank and overlap(k.name, claim.name)]
            if stronger := sorted(stronger, key=lambda k: not covers(k.name, claim.name))[:1]:
                table.shadowed.append((claim, stronger[0]))
            kept.append(claim)
    if conflicts:
        raise SiteConflict("names claimed twice: " + "; ".join(conflicts))
    by_site = {site.name: site for site in sites}
    for owner in dict.fromkeys(c.owner for c in kept if c.rank == SYNTHETIC):
        table.routes.append(Route(owner, by_site[owner].url,
                                  tuple(c.name for c in kept if c.owner == owner and c.rank == SYNTHETIC)))
    covered = {weak for weak, strong in table.shadowed if covers(strong.name, weak.name)}
    table.unrouted = {c.name: c.rank for c in kept if c.rank != SYNTHETIC and c not in covered}
    return table


def table_problems(sites: list[Site], allow_hosts: Iterable[str] = ()) -> list[str]:
    try:
        site_table(sites, allow_hosts)
    except ValueError as e:
        return [f"env_params.sites: {e}"]
    return []


@dataclass(frozen=True)
class Issued:
    """The deploy's CA certificate, and each site's certificate and key, all PEM; no CA when no name is served over
    HTTPS. The CA's key signs them and is gone: nothing more can be issued under it."""

    ca_pem: Optional[str]
    leaves: dict[str, tuple[str, str]]


def issue(routes: list[Route], *, valid_for: timedelta, label: str, now: Optional[datetime] = None) -> Issued:
    """A CA name-constrained to the routes' names, so it vouches for nothing else however widely it's trusted, and a
    certificate signed by it for each route's names served over HTTPS, all valid ``valid_for`` and a margin more."""
    tls = [route for route in routes if route.tls_names]
    if not tls:
        return Issued(None, {})
    now = now or datetime.now(UTC)
    start, end = now - CLOCK_SKEW, now + valid_for + VALID_MARGIN
    ca_key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "agentenv"),
                         x509.NameAttribute(NameOID.COMMON_NAME, f"agentenv sites {label}"[:64])])
    permitted = list(dict.fromkeys(name.removeprefix("*.") for route in tls for name in route.tls_names))
    ca = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start).not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(_key_usage(key_cert_sign=True, crl_sign=True), critical=True)
        .add_extension(x509.NameConstraints(permitted_subtrees=[x509.DNSName(n) for n in permitted],
                                            excluded_subtrees=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    leaves = {}
    for route in tls:
        key = ec.generate_private_key(ec.SECP256R1())
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([]))
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(start).not_valid_after(end)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(n) for n in route.tls_names]), critical=True)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(_key_usage(digital_signature=True), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        leaves[route.site] = (cert.public_bytes(serialization.Encoding.PEM).decode(), key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode())
    return Issued(ca.public_bytes(serialization.Encoding.PEM).decode(), leaves)


def _key_usage(**granted: bool) -> x509.KeyUsage:
    usages = ("digital_signature", "content_commitment", "key_encipherment", "data_encipherment", "key_agreement",
              "key_cert_sign", "crl_sign", "encipher_only", "decipher_only")
    return x509.KeyUsage(**{usage: granted.get(usage, False) for usage in usages})


def proxy_config(table: SiteTable, issued: Issued, *, user: Optional[str],
                 upstream_headers: Optional[dict[str, dict[str, str]]] = None) -> dict:
    """What ``site_proxy.py install`` reads: the session user, the CA, and each route with its certificate and key, and
    the headers its gateway's sandbox provider asks requests to it to carry (``upstream_headers``, by site), which the
    proxy sends upstream alone."""
    sites = []
    for route in table.routes:
        site = {"name": route.site, "upstream": route.url, "domains": list(route.domains)}
        if route.site in issued.leaves:
            site["cert_pem"], site["key_pem"] = issued.leaves[route.site]
        if (upstream_headers or {}).get(route.site):
            site["upstream_headers"] = dict(upstream_headers[route.site])
        sites.append(site)
    return {"user": user, "ca_pem": issued.ca_pem, "sites": sites, "log": str(LOG)}


def reachable_problem(policy: Optional[NetworkPolicy], routes: list[Route]) -> Optional[str]:
    """Why ``policy``, the computer's network as its sandbox applies it, keeps it from a route's gateway, or None: a
    gateway is reachable by its name in the allowed hosts, exactly or under a ``*.domain`` entry, or by an address in an
    allowed range."""
    if policy is None or not policy.restricts_egress:
        return None
    hosts = sorted({urlsplit(route.url).hostname or "" for route in routes})
    blocked = [host for host in hosts if not _allows(policy, host)]
    if blocked:
        allowed = [*policy.allow_hosts, *policy.allow_cidrs]
        return (f"the computer's network allows only {', '.join(allowed) or 'nothing'}, so it can't reach "
                f"the gateway at {', '.join(blocked)} its sites are on")
    return None


def gateway_problem(routes: list[Route]) -> Optional[str]:
    """Why the computer couldn't reach a route's gateway once its sites' names are routed, or None: a gateway named by a
    route's name, exactly or under its wildcard, would resolve to the proxy itself."""
    hosts = {urlsplit(route.url).hostname or "" for route in routes}
    hidden = sorted({f"{host} (named by {route.site}'s {name})" for host in hosts for route in routes
                     for name in route.domains if matches(name, host)})
    if hidden:
        return f"the gateway at {', '.join(hidden)} would resolve to the computer's own site proxy"
    return None


def _allows(policy: NetworkPolicy, host: str) -> bool:
    host = host.lower().rstrip(".")
    if any(host == allowed or (allowed.startswith("*.") and host.endswith(allowed[1:]))
           for allowed in (entry.lower().rstrip(".") for entry in policy.allow_hosts)):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    for entry in (*policy.allow_cidrs, *policy.allow_hosts):
        try:
            if address in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


class RootChannel:
    """Root inside a booted computer, through its sandbox: on a VM sandbox ``docker exec -u 0`` into its container,
    elsewhere the sandbox's own exec, which must run as root, as a container sandbox running the image as its own user
    (``USER user``) doesn't: that's refused before anything is written. The driver's computer-exec/v1 runs only as the
    session user."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.checked = False

    async def private_folder(self, path: PurePosixPath) -> None:
        """Make ``path`` a folder root alone can enter, an existing one too, so what's written into it next is never
        the session user's to read, whatever mode a file lands with."""
        await self.run(["mkdir", "-p", str(path)])
        await self.run(["chmod", "0700", str(path)])

    async def write(self, path: PurePosixPath, text: str) -> None:
        if isinstance(self.sandbox, VmSandbox):
            await self.sandbox.write_file_from_text(text, str(path))
            return
        await self.run(["mkdir", "-p", str(path.parent)])
        await self.sandbox.write_file_from_text(text, str(path))

    async def write_secret(self, path: PurePosixPath, text: str) -> None:
        """Write ``text``, which holds keys or credentials, to ``path``. On a VM it's staged on the host in a folder
        its owner alone can open (``secret_dir``), written over the exec's stdin (``write_secret_file``), so it's
        never on a command line, and moved into the container, its staged copy removed. Other sandboxes write files
        through their provider's API, as ``write`` does."""
        if not isinstance(self.sandbox, VmSandbox):
            await self.write(path, text)
            return
        await self.run(["mkdir", "-p", str(path.parent)])
        async with self.sandbox.secret_dir() as secrets:
            staged = f"{secrets}/{path.name}"
            await self.sandbox.write_secret_file(text.encode(), staged)
            await self.sandbox.docker_cp(staged, f"{self.sandbox.container_name}:{path}", remove_source=True)

    async def run(self, argv: list[str]) -> str:
        if isinstance(self.sandbox, VmSandbox):
            return await self.sandbox.exec_script(f"docker exec -u 0 {shlex.quote(self.sandbox.container_name)} "
                                                  f"{shlex.join(argv)}")
        if not self.checked:
            uid = (await self._exec(["id", "-u"])).strip()
            if uid != "0":
                raise RuntimeError(f"site routing installs as root, and the {getattr(self.sandbox, 'type', 'given')} "
                                   f"sandbox runs the computer's commands as uid {uid}; deploy it on a VM sandbox")
            self.checked = True
        return await self._exec(argv)

    async def _exec(self, argv: list[str]) -> str:
        code, stdout, stderr = await self.sandbox.exec_with_output(*argv)
        if code != 0:
            raise RuntimeError(f"{shlex.join(argv)} failed (exit {code}):\nstdout: {stdout[-1500:]}\n"
                               f"stderr: {stderr[-1500:]}")
        return stdout


async def route(sandbox: Sandbox, sites: list[Site], *, user: Optional[str], network: Optional[dict],
                valid_for: timedelta, label: str) -> dict:
    """Have the computer on ``sandbox`` reach each of ``sites`` by its names: the table made, the CA and certificates
    issued, the proxy installed and started as root, and each name checked as ``user``; what the deploy record keeps of
    it. Raises naming a name that didn't load."""
    requested = NetworkPolicy.from_dict(network) if network else None
    table = site_table(sites, requested.allow_hosts if requested else ())
    table.routes = [Route(r.site, sandbox.url_from_sandbox(r.url), r.domains) for r in table.routes]
    applied = getattr(sandbox, "network_policy", None) or requested
    if (problem := reachable_problem(applied, table.routes) or gateway_problem(table.routes)) is not None:
        raise RuntimeError(f"site routing: {problem}")
    if not table.routes:
        return {"names": table.to_dict()}
    from agent_env.providers.sandbox_providers.sandbox_provider import sandbox_request_headers_for_url

    issued = issue(table.routes, valid_for=valid_for, label=label)
    upstream_headers = {r.site: sandbox_request_headers_for_url(r.url) for r in table.routes}
    channel = RootChannel(sandbox)
    await channel.private_folder(FOLDER)
    await channel.write(PROXY, _PROXY_SOURCE.read_text())
    await channel.write_secret(CONFIG, json.dumps(proxy_config(table, issued, user=user,
                                                               upstream_headers=upstream_headers)))
    await channel.run(["chmod", "0600", str(CONFIG)])
    output = await channel.run(["python3", str(PROXY), "install", str(CONFIG)])
    report = json.loads(output.strip().splitlines()[-1])
    failed = {name: result for name, result in report["checks"].items() if "error" in result}
    if failed:
        raise RuntimeError("site routing: " + "; ".join(f"{result.get('url', name)} didn't load in the computer: "
                                                        f"{result['error']}" for name, result in failed.items()))
    return {"names": table.to_dict(), "ca": issued.ca_pem, "ca_path": str(CA_PATH), "trust": report["trust"],
            "checks": report["checks"]}


async def deployed_sites(deployed_envs: list[DeployedEnv], only: Optional[list[str]] = None) -> list[dict]:
    """The websites a run has deployed, as ``env_params.sites`` names them: each ``website/v1`` child env of a deployed
    env's stored card, with the route its composer serves it to a browser at (its params' ``browse_url``, joined onto
    the env's address), and the names its params declare. A record whose card names no ``browse_url``, as
    one a gateway made before it did, takes the route from its ``website_frontend_urls``. A site those name that the
    stored card doesn't hold, every site of a record without a card or one whose card the gateway couldn't read when it
    composed its own, is read live at ``/svc/<name>``, and has no names when it serves no card (404 or no JSON).
    Reading one fails otherwise, after ``CARD_ATTEMPTS`` tries, rather than route the site by names it didn't declare.
    ``only`` keeps the websites of the envs it names, each of which must be deployed. Sites of different envs that
    share a name are each named for their env (``_env_qualified``), so they're routed apart, while two that claim one
    domain are still refused."""
    from agentenv_protocol import client as protocol_v1
    from agentenv_protocol.website import WEBSITE_BROWSE_URL_PARAM, WEBSITE_EXTENSION_URI

    from agent_env.env.env import gateway_url_of

    records = list(deployed_envs)
    if only is not None:
        missing = [env_id for env_id in only if env_id not in {d.env_id for d in deployed_envs}]
        if missing:
            raise ValueError(f"site_routing names {', '.join(missing)}, which this run hasn't deployed before the "
                             "computer")
        records = [d for d in records if d.env_id in only]
    found: list[tuple[str, dict]] = []
    for record in records:
        address = gateway_url_of(record) or getattr(record, "environment_url", None)
        frontends = getattr(record, "website_frontend_urls", None) or {}
        named = set()
        for child in protocol_v1.children(getattr(record, "environment_card", None) or {}):
            if child.find_extension(WEBSITE_EXTENSION_URI) is None or not child.name:
                continue
            params = protocol_v1.extension_params(child.card, WEBSITE_EXTENSION_URI)
            browse = params.get(WEBSITE_BROWSE_URL_PARAM)
            service = _served_as(child)
            url = (_joined(address, browse) if isinstance(browse, str) and browse and address
                   else frontends.get(service))
            if url is None:
                logger.warning("site routing: the website %s of env %s names no route a browser reaches it at, so it "
                               "isn't routed", child.name, record.env_id)
                continue
            named.add(service)
            found.append((record.env_id, _found(child.name, url, params)))
        for name, url in frontends.items():
            if name not in named:
                card = await _site_card(protocol_v1, f"{gateway_url_of(record)}/svc/{name}", name)
                found.append((record.env_id, _found(name, url, protocol_v1.extension_params(card, WEBSITE_EXTENSION_URI))))
    return _env_qualified(found)


def _env_qualified(found: list[tuple[str, dict]]) -> list[dict]:
    """Each site found, a name that sites of different envs share made ``<env_id>-<name>`` for each of them, letters,
    digits, ``-`` and ``_`` kept, so every site of the run has a name of its own."""
    envs: dict[str, set[str]] = {}
    for env_id, site in found:
        envs.setdefault(site["name"], set()).add(env_id)
    return [{**site, "name": re.sub(r"[^A-Za-z0-9_-]", "-", f"{env_id}-{site['name']}")}
            if len(envs[site["name"]]) > 1 else site for env_id, site in found]


def _served_as(child) -> str:
    """The gateway service a child env is served as: the ``<name>`` of the ``/svc/<name>/`` route its url is at, which
    ``website_frontend_urls`` keys it by, else its card's name."""
    parts = urlsplit(child.url or "").path.strip("/").split("/")
    return parts[1] if len(parts) > 1 and parts[0] == "svc" and parts[1] else child.name


def _found(name: str, url: str, params: dict) -> dict:
    return {"name": name, "url": url, "domains": list(params.get("domains") or [])}


def _joined(address: str, url: str) -> str:
    """``url`` resolved against ``address``: a full URL stands alone, and a path joins onto it."""
    return url if "://" in url else f"{address.rstrip('/')}/{url.lstrip('/')}"


async def _site_card(protocol_v1, url: str, name: str) -> dict:
    """The card of the site ``name`` at ``url``, or none for a site that serves none."""
    for attempt in range(1, CARD_ATTEMPTS + 1):
        try:
            return await protocol_v1.get_card(url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {}
            error: Exception = e
        except ValueError:
            return {}
        except (httpx.HTTPError, OSError) as e:
            error = e
        if attempt < CARD_ATTEMPTS:
            await asyncio.sleep(CARD_RETRY_S)
    raise RuntimeError(f"site routing: the card of the site {name} at {url} couldn't be read, so its names are "
                       f"unknown: {type(error).__name__}: {error}")
