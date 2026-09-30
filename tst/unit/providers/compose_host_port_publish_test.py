"""Every host port a generated compose publishes must go through the sandbox port resolver.

LocalSandbox shares the host network namespace, so two concurrent deployments contend for
any fixed host port. Each deployment is therefore given its own host port and the compose
publishes ``resolved_host:container``, on the sandbox's host IPs. Container ports are untouched --
they are the env's internal contract, and callers read the host side back out of ``tunnel_urls``.

Coverage is two layers. The source guard holds for every renderer in the tree, including
env types added later. The rendered-document checks then exercise the config axes that
change which services appear at all.
"""

import re
from pathlib import Path

import pytest

from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
from agent_env.providers.env_providers import EnvironmentGatewayProvider, MCPServerConfig, SidecarConfig
from agent_env.env.envs.service_db import DB_MCP_CONTAINER_PORT, DB_MCP_PORT, DB_WEB_PORT
from agent_env.providers.env_state.local_postgres import LocalPostgresStateProvider

SRC = Path(__file__).resolve().parents[3] / "src"

# Above any port a compose can legitimately declare, so a resolved host port always lands
# above it and a hard-coded one always below.
OFFSET = 40000


# --- source guard: exhaustive over the tree ---------------------------------

# A compose port mapping, ``[ip:]host:container``, double-quoted, single-quoted or bare. Excludes
# volume mounts and image refs (which contain "/" or "://") and docstring illustrations written as
# <placeholder>.
_PORT_PAIR = re.compile(r'-\s+["\']?(?:([^"\':/\s]+):)?([^"\':/\s]+):([^"\':/\s]+)')

# A compose port entry rendered from ``port_bindings(host_ips, host_port, container_port)``.
_PUBLISH_SITE = re.compile(r'-\s+"\{(\w+)\}"\' for \1 in port_bindings\([^,]+, (.+), [^,]+\)')


def _is_port_expression(text: str) -> bool:
    return text.isdigit() or (text.startswith("{") and text.endswith("}"))


def _source_lines():
    for path in sorted(SRC.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            yield path, lineno, line


def _publish_sites():
    """Every compose port entry written anywhere under src/, with its host-port expression."""
    for path, lineno, line in _source_lines():
        for _, host in _PUBLISH_SITE.findall(line):
            yield path, lineno, line.strip(), host


def _bypasses_resolver(host: str) -> bool:
    """A host port is resolved only if it is computed, not written as a constant."""
    return "publish(" not in host and "host_port(" not in host


def test_no_renderer_publishes_a_fixed_host_port():
    """Scanning the tree, rather than listing known renderers, is what makes this hold for
    an env type added later. A sidecar was already written ``host:container``, so it read
    as correct while the host half was still a constant."""
    sites = list(_publish_sites())
    assert sites, "found no port publications -- the scan or the compose templates changed"

    violations = [
        f"{path.relative_to(SRC)}:{lineno}: {line}"
        for path, lineno, line, host in sites
        if _bypasses_resolver(host)
    ]
    assert not violations, "host ports published without the resolver:\n" + "\n".join(violations)


def test_no_renderer_writes_a_port_mapping_by_hand():
    """A mapping written out skips ``port_bindings``, so on local it would publish on every interface."""
    by_hand = [
        f"{path.relative_to(SRC)}:{lineno}: {line.strip()}"
        for path, lineno, line in _source_lines()
        for _, host, container in _PORT_PAIR.findall(line)
        if _is_port_expression(host) and _is_port_expression(container)
    ]
    assert not by_hand, "port mappings written without port_bindings:\n" + "\n".join(by_hand)


def test_the_mapping_guard_reads_every_quoting():
    """Compose accepts a mapping in any of the three forms, so a hand-written one can't slip past as another."""
    for entry in ('- "8080:8080"', "- '8080:8080'", "- 8080:8080"):
        assert _PORT_PAIR.findall(entry) == [("", "8080", "8080")]
    # With an IP in front, the IP is its own field and the host port is still the second.
    assert _PORT_PAIR.findall('- "127.0.0.1:18768:8000"') == [("127.0.0.1", "18768", "8000")]


def test_source_guard_flags_a_fixed_host_port():
    """The guard above only means something if a constant actually trips it."""
    assert _bypasses_resolver("{DB_WEB_PORT}")
    assert _bypasses_resolver("18768")
    assert not _bypasses_resolver("{publish(DB_WEB_PORT)}")
    assert not _bypasses_resolver("{sandbox.host_port(8080)}")


# --- rendered documents -----------------------------------------------------


def _published_pairs(compose: str) -> list[tuple[int, int]]:
    """Every numeric ``host:container`` pair in a rendered compose."""
    pairs = []
    for line in compose.splitlines():
        for _, host, container in _PORT_PAIR.findall(line.strip()):
            if host.isdigit() and container.isdigit():
                pairs.append((int(host), int(container)))
    return pairs


def assert_all_resolved(compose: str) -> list[tuple[int, int]]:
    pairs = _published_pairs(compose)
    unresolved = [(h, c) for h, c in pairs if h < OFFSET]
    assert not unresolved, f"published without the resolver: {unresolved}\n{compose}"
    return pairs


def test_published_pairs_reads_a_hard_coded_line():
    assert _published_pairs('      - "18768:8000"') == [(18768, 8000)]


class _Cfg:
    """Only the image fields are consulted for the port lines; each sidecar is gated on
    its image being set."""

    db_web_image = "pgweb:test"
    db_mcp_image = "db-mcp:test"
    db_image = "postgres:test"


class _StateInstance:
    _db_url_base = "postgresql://stub:stub@servicedb:5432/stub"


@pytest.fixture
def stub_postgres(monkeypatch):
    """``service_db_config`` is a read-only property that resolves the default ServiceDBEnv
    from the store, so it is patched on the class rather than per-instance."""
    monkeypatch.setattr(
        LocalPostgresStateProvider, "service_db_config", property(lambda self: _Cfg()),
    )
    monkeypatch.setattr(
        LocalPostgresStateProvider, "_pg_url", lambda self, *a, **k: "postgres://stub",
    )
    return LocalPostgresStateProvider.__new__(LocalPostgresStateProvider)


def _sidecar(name: str, host_port: int, container_port: int = 8000) -> SidecarConfig:
    class _Artifact:
        image_name = f"{name}:test"

    return SidecarConfig(
        environment_name=name,
        image_artifact=_Artifact(),
        container_port=container_port,
        host_port=host_port,
    )


# The axes that change which services a gateway compose renders: how many MCP servers are
# attached, and how many sidecars each contributes a published port.
@pytest.mark.parametrize(
    "environments",
    [
        pytest.param([], id="no-mcp-servers"),
        pytest.param(["slack"], id="one-mcp-server"),
        pytest.param(["slack", "email", "calendar"], id="many-mcp-servers"),
    ],
)
@pytest.mark.parametrize(
    "sidecars",
    [
        pytest.param([], id="no-sidecars"),
        pytest.param([("relay", 18768)], id="one-sidecar"),
        pytest.param(
            [("relay", 18768), ("exporter", 18769)], id="many-sidecars",
        ),
    ],
)
def test_gateway_compose_resolves_every_host_port(stub_postgres, environments, sidecars):
    """The gateway line and each sidecar line, across every combination of attached
    services -- rendered with the real Postgres state provider, so pgweb and the DB MCP
    sidecar are in the document too."""
    compose = EnvironmentGatewayProvider.create_docker_compose(
        EnvironmentGatewayProvider.__new__(EnvironmentGatewayProvider),
        mcp_servers=[
            MCPServerConfig(image=f"{name}:test", environment_name=name) for name in environments
        ],
        gateway_image="gateway:test",
        sidecars=[_sidecar(name, port) for name, port in sidecars],
        state_provider=stub_postgres,
        state_instance=_StateInstance(),
        host_port=lambda port: port + OFFSET,
    )

    pairs = assert_all_resolved(compose)
    # The gateway, pgweb and the DB MCP sidecar are always present; each extra sidecar adds
    # one more publication.
    assert len(pairs) == 3 + len(sidecars)


def test_gateway_compose_defaults_to_identity(stub_postgres):
    """No resolver means same-port publishing -- the per-VM behaviour, unchanged. Backends
    that give each deployment its own namespace must keep working untouched."""
    compose = EnvironmentGatewayProvider.create_docker_compose(
        EnvironmentGatewayProvider.__new__(EnvironmentGatewayProvider),
        mcp_servers=[MCPServerConfig(image="slack:test", environment_name="slack")],
        gateway_image="gateway:test",
        sidecars=[_sidecar("relay", 18768)],
        state_provider=stub_postgres,
        state_instance=_StateInstance(),
    )

    assert f'"{DB_WEB_PORT}:{DB_WEB_PORT}"' in compose
    assert f'"{DB_MCP_PORT}:{DB_MCP_CONTAINER_PORT}"' in compose
    assert '"18768:8000"' in compose


def test_postgres_sidecar_fragment_resolves_host_ports(stub_postgres):
    """The state provider renders its own fragment, so it is checked directly as well as
    through the full compose above."""
    lines = "\n".join(
        stub_postgres.render_sidecar_containers(["slack"], host_port=lambda p: p + OFFSET),
    )

    assert f'"{DB_WEB_PORT + OFFSET}:{DB_WEB_PORT}"' in lines
    assert f'"{DB_MCP_PORT + OFFSET}:{DB_MCP_CONTAINER_PORT}"' in lines
    # Container side untouched: it is the env's internal contract.
    assert f'"{DB_WEB_PORT}:{DB_WEB_PORT}"' not in lines


def test_postgres_sidecar_fragment_defaults_to_identity(stub_postgres):
    lines = "\n".join(stub_postgres.render_sidecar_containers(["slack"]))

    assert f'"{DB_WEB_PORT}:{DB_WEB_PORT}"' in lines
    assert f'"{DB_MCP_PORT}:{DB_MCP_CONTAINER_PORT}"' in lines


@pytest.mark.parametrize("host_ips", [("127.0.0.1",), ("127.0.0.1", "172.17.0.1")], ids=["loopback", "loopback-and-bridge"])
def test_gateway_compose_publishes_only_on_the_host_ips(stub_postgres, host_ips):
    """Each port once per host IP, and none on every interface."""
    compose = EnvironmentGatewayProvider.create_docker_compose(
        EnvironmentGatewayProvider.__new__(EnvironmentGatewayProvider),
        mcp_servers=[MCPServerConfig(image="slack:test", environment_name="slack")],
        gateway_image="gateway:test",
        sidecars=[_sidecar("relay", 18768)],
        state_provider=stub_postgres,
        state_instance=_StateInstance(),
        host_port=lambda port: port + OFFSET,
        host_ips=host_ips,
    )

    ports = [(AGENT_ENV_GATEWAY_MCP_PORT, AGENT_ENV_GATEWAY_MCP_PORT), (DB_WEB_PORT, DB_WEB_PORT),
             (DB_MCP_PORT, DB_MCP_CONTAINER_PORT), (18768, 8000)]
    assert sorted(_PORT_PAIR.findall(compose)) == sorted(
        (ip, str(host + OFFSET), str(container)) for ip in host_ips for host, container in ports
    )
