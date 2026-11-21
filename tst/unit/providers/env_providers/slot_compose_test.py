"""A slot's compose file fits its walls, and the default stack's is the file it has always been."""

from __future__ import annotations

import copy
import re
from ipaddress import ip_address
from pathlib import Path

import pytest
import yaml

from agent_env.env.gateway.constants import DOOR_ROUTES_ENV, DOOR_TOKEN_ENV
from agent_env.providers.env_providers.slot_compose import (
    DOOR_COMMAND,
    DOOR_SERVICE_NAME,
    slot_compose,
    slot_compose_problems,
    slot_resolv_conf,
)
from agent_env.providers.env_providers.stack_layout import DEFAULT_STACK_LAYOUT, StackLayout
from agent_env.providers.sandbox_providers.shared_box_walls import slot_bridges, slot_networks
from tst.unit.providers.env_providers.compose_fixture import render

GOLDEN = Path(__file__).parent / "golden" / "default_stack_compose.yml"
SLOT = StackLayout.for_slot(3)
TOKEN = "door-token-value"
# A URL's host, as a service's settings name the service they reach: http://gateway:18765, postgres://u:p@servicedb:5432.
_URL_HOST = re.compile(r"[a-z]+://(?:[^@/\s,]*@)?([A-Za-z0-9.-]+):\d+")


def test_the_default_stack_renders_byte_for_byte_as_before():
    assert render() == GOLDEN.read_text()
    assert render(stack_layout=DEFAULT_STACK_LAYOUT, door_token=None) == GOLDEN.read_text()


@pytest.fixture(scope="module")
def slot() -> dict:
    return yaml.safe_load(render(stack_layout=SLOT, door_token=TOKEN))


def _networks(service: dict) -> dict[str, str]:
    return {network: config["ipv4_address"] for network, config in service["networks"].items()}


def _extra_hosts(service: dict) -> dict[str, str]:
    return dict(entry.split(":", 1) for entry in service.get("extra_hosts") or [])


def _names_used(name: str, service: dict) -> set[str]:
    """The hosts a service's settings reach other services by: URL hosts, the gateway's bare internal server names,
    the door's upstreams."""
    env = dict(entry.split("=", 1) for entry in service.get("environment") or [])
    used = {host for value in env.values() for host in _URL_HOST.findall(value)}
    used |= {entry for entry in env.get("INTERNAL_MCP_SERVERS", "").split(",") if entry and "=" not in entry}
    used |= {route.split("=", 1)[1].rsplit(":", 1)[0] for route in env.get(DOOR_ROUTES_ENV, "").split(",") if route}
    return {host for host in used if "." not in host and host != name}


def test_every_service_runs_under_gvisor_with_the_slots_resolvers(slot):
    for service in slot["services"].values():
        assert service["runtime"] == "runsc"
        assert f"{SLOT.app_dir}/resolv.conf:/etc/resolv.conf:ro" in service["volumes"]


def test_the_slots_networks_have_its_subnets_and_bridges_and_no_ipv6(slot):
    subnets, bridges = slot_networks(3), slot_bridges(3)
    assert slot["networks"] == {
        "env-network": {"driver": "bridge", "enable_ipv6": False,
                        "driver_opts": {"com.docker.network.bridge.name": bridges.env},
                        "ipam": {"config": [{"subnet": str(subnets.env)}]}},
        "website-network": {"driver": "bridge", "enable_ipv6": False,
                            "driver_opts": {"com.docker.network.bridge.name": bridges.website},
                            "ipam": {"config": [{"subnet": str(subnets.website)}]}},
    }


def test_every_service_has_its_own_fixed_address_in_the_slots_subnets(slot):
    subnet = {"env-network": slot_networks(3).env, "website-network": slot_networks(3).website}
    seen = [address for service in slot["services"].values() for address in _networks(service).values()]
    assert len(seen) == len(set(seen))
    for service in slot["services"].values():
        for network, address in _networks(service).items():
            assert ip_address(address) in subnet[network] and ip_address(address) != subnet[network][1]


def test_extra_hosts_name_every_service_a_service_reaches_and_only_on_a_network_they_share(slot):
    services = slot["services"]
    for name, service in services.items():
        hosts = _extra_hosts(service)
        missing = _names_used(name, service) - set(hosts)
        assert not missing, f"{name} can't resolve {missing}"
        for peer, address in hosts.items():
            if peer == "host.docker.internal":
                continue
            shared = set(_networks(service)) & set(_networks(services[peer]))
            assert address in {_networks(services[peer])[network] for network in shared}, (name, peer)
    assert set(_extra_hosts(services["browser"])) == {"gateway", "shop-website-frontend", "blog-website-backend"}
    assert "servicedb" not in _extra_hosts(services["browser"])


def test_only_the_door_publishes_and_it_takes_every_port_the_default_stack_published(slot):
    default = yaml.safe_load(GOLDEN.read_text())["services"]
    published = {name: service["ports"] for name, service in default.items() if "ports" in service}
    assert [name for name, service in slot["services"].items() if "ports" in service] == [DOOR_SERVICE_NAME]
    door = slot["services"][DOOR_SERVICE_NAME]
    expected_ports, expected_routes = [], []
    for name, specs in published.items():
        for spec in specs:
            ip, host_port, container_port = spec.split(":")
            expected_ports.append(f"{ip}:{host_port}:{host_port}")
            expected_routes.append(f"{host_port}={name}:{container_port}")
    assert door["ports"] == expected_ports
    assert door["command"] == DOOR_COMMAND
    assert door["image"] == "agent-gateway"
    assert door["environment"] == [f"{DOOR_ROUTES_ENV}={','.join(expected_routes)}", f"{DOOR_TOKEN_ENV}={TOKEN}"]
    assert list(door["networks"]) == ["env-network"]


def test_the_slots_compose_passes_its_own_wall_check(slot):
    assert slot_compose_problems(slot, SLOT) == []


@pytest.mark.parametrize(("change", "problem"), [
    (lambda s: s["gateway"].update(privileged=True), "gateway sets privileged"),
    (lambda s: s["gateway"].update(cap_add=["NET_ADMIN"]), "gateway sets cap_add"),
    (lambda s: s["gateway"].update(devices=["/dev/kvm"]), "gateway sets devices"),
    (lambda s: s["gateway"].update(network_mode="host"), "gateway sets network_mode"),
    (lambda s: s["gateway"].update(pid="host"), "gateway sets pid"),
    (lambda s: s["gateway"].update(ipc="host"), "gateway sets ipc"),
    (lambda s: s["gateway"].update(userns_mode="host"), "gateway sets userns_mode"),
    (lambda s: s["gateway"].update(security_opt=["seccomp=unconfined"]), "gateway sets security_opt"),
    (lambda s: s["gateway"].update(runtime="runc"), "gateway runs under 'runc'"),
    (lambda s: s["gateway"].update(ports=["18765:18765"]), "gateway publishes ports"),
    (lambda s: s["gateway"]["volumes"].append("/var/run/docker.sock:/var/run/docker.sock"), "docker socket"),
    (lambda s: s["gateway"]["volumes"].append({"type": "bind", "source": "/run/docker.sock", "target": "/s"}), "docker socket"),
    (lambda s: s["gateway"]["volumes"].append("/etc:/host-etc:ro"), "bind-mounts /etc"),
    (lambda s: s["gateway"]["volumes"].append("../s4:/other"), "bind-mounts ../s4"),
    (lambda s: s["gateway"]["volumes"].append("~/x:/x"), "bind-mounts ~/x"),
    (lambda s: s["gateway"]["volumes"].append({"type": "bind", "source": "/app", "target": "/a"}), "bind-mounts /app"),
])
def test_the_wall_check_refuses_whatever_would_reach_past_the_slot(slot, change, problem):
    broken = copy.deepcopy(slot)
    change(broken["services"])
    assert any(problem in found for found in slot_compose_problems(broken, SLOT))


def test_named_volumes_and_binds_inside_the_slots_dir_pass(slot):
    fine = copy.deepcopy(slot)
    fine["services"]["gateway"]["volumes"] += ["logs:/var/log", "./data:/data", f"{SLOT.app_dir}/x:/x"]
    assert slot_compose_problems(fine, SLOT) == []


def test_a_slot_compose_needs_a_slot_layout_and_a_token():
    with pytest.raises(ValueError, match="for_slot"):
        slot_compose(render(), DEFAULT_STACK_LAYOUT, door_image="gw", door_token=TOKEN)
    with pytest.raises(ValueError, match="needs a token"):
        render(stack_layout=SLOT, door_token=None)


def test_a_slot_mounts_the_boxs_ipv4_resolvers():
    assert slot_resolv_conf("search x\nnameserver 172.21.0.1\nnameserver fd00::1\n") == "nameserver 172.21.0.1\n"
    with pytest.raises(ValueError):
        slot_resolv_conf("nameserver fd00::1\n")
