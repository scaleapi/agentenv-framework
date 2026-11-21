"""A slot's compose file: the default stack's, refitted to the walls of a shared box.

Every service runs under gVisor (``runsc``) at a fixed address on its slot's networks, which have fixed subnets and
bridge names (``shared_box_walls``). Docker's embedded DNS doesn't answer under gVisor, so each service resolves the
services it shares a network with from ``extra_hosts``, exactly the names Docker's DNS would give it, and the internet
through the box's resolvers in a mounted resolv.conf. No service publishes a port but the door, which takes over every
port the stack would have published and lets in only requests carrying the slot's token (``gateway.door``).
"""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from ipaddress import IPv4Network

import yaml

from agent_env.env.gateway.constants import DOOR_ROUTES_ENV, DOOR_TOKEN_ENV
from agent_env.providers.env_providers.constants import ENV_NETWORK, SITES_NETWORK
from agent_env.providers.env_providers.stack_layout import StackLayout
from agent_env.providers.sandbox_providers.shared_box_walls import slot_bridges, slot_networks, upstream_resolvers

DOOR_SERVICE_NAME = "agentenv-door"
DOOR_COMMAND = ["python", "-m", "gateway.door"]
SLOT_RUNTIME = "runsc"
_BRIDGE_NAME_OPTION = "com.docker.network.bridge.name"
# The only keys a slot's compose may use: the ones the default stack's generator and slot_compose write. Every other
# key is refused rather than listed, as many reach past the walls: host devices, capabilities or namespaces, a looser
# profile, a build context, files read from the box (env_file, include, extends, configs, secrets).
_ALLOWED_TOP_LEVEL_KEYS = frozenset({"services", "networks", "volumes"})
_ALLOWED_SERVICE_KEYS = frozenset({"image", "command", "depends_on", "environment", "extra_hosts", "healthcheck",
                                   "networks", "ports", "restart", "shm_size", "volumes", "runtime"})
_MOUNT_TYPES = frozenset({"bind", "volume", "tmpfs"})


def resolv_conf_path(layout: StackLayout) -> str:
    return f"{layout.app_dir}/resolv.conf"


def slot_resolv_conf(box_resolv_conf: str) -> str:
    """The resolv.conf a slot's services mount: the box's IPv4 resolvers, the only ones the walls let them reach."""
    return "".join(f"nameserver {resolver}\n" for resolver in upstream_resolvers(box_resolv_conf))


def slot_compose(compose: str, layout: StackLayout, *, door_image: str, door_token: str) -> str:
    """``compose``, a default-stack compose file, refitted for ``layout``'s slot (see the module doc), with a door run
    from ``door_image`` that takes ``door_token``. Raises if the result breaks a wall (``slot_compose_problems``)."""
    index = layout.slot_index
    if index is None:
        raise ValueError(f"only a layout made by StackLayout.for_slot has a slot's walls; got {layout}")
    if not door_token:
        raise ValueError("a slot's door needs a token")
    networks = _slot_network_plan(index)
    doc = yaml.safe_load(compose)
    services: dict[str, dict] = doc["services"]
    if DOOR_SERVICE_NAME in services:
        raise ValueError(f"a service is already named {DOOR_SERVICE_NAME!r}, the slot's door")
    for name, service in services.items():
        if unknown := [n for n in _service_networks(service) if n not in networks]:
            raise ValueError(f"service {name!r} is on {unknown}, which a slot has no subnet for")
    services[DOOR_SERVICE_NAME] = _door(services, door_image, door_token)

    addresses = _addresses(services, networks)
    resolv_conf_mount = f"{resolv_conf_path(layout)}:/etc/resolv.conf:ro"
    for name, service in services.items():
        service["runtime"] = SLOT_RUNTIME
        service["networks"] = {network: {"ipv4_address": address} for network, address in addresses[name].items()}
        peers: dict[str, str] = {}
        for network in addresses[name]:
            for peer, theirs in addresses.items():
                if peer != name and network in theirs:
                    peers.setdefault(peer, theirs[network])
        service["extra_hosts"] = [*(service.get("extra_hosts") or []), *(f"{peer}:{ip}" for peer, ip in peers.items())]
        service["volumes"] = [*(service.get("volumes") or []), resolv_conf_mount]
    used = {network for service_addresses in addresses.values() for network in service_addresses}
    doc["networks"] = {network: config for network, config in _slot_networks(index).items() if network in used}
    if problems := slot_compose_problems(doc, layout):
        raise ValueError(f"slot {index}'s compose file breaks its walls: {'; '.join(problems)}")
    return yaml.safe_dump(doc, sort_keys=False, width=float("inf"))


def slot_compose_problems(doc: Mapping, layout: StackLayout) -> list[str]:
    """Why a slot's parsed compose file would let a container past its walls: a key outside the allowed ones, a named
    volume with any settings (a driver's options can bind any host path), a network other than the slot's own as
    ``slot_compose`` writes it, as one on another bridge is outside the walls, a service on none of them, which Compose
    would put on such a network, a runtime other than gVisor, a published port on any service but the door, the docker
    socket, a bind mount from outside ``layout.app_dir``, or a ``$`` in a mount's source. Empty when there is none."""
    problems = [f"the file sets {key}" for key in doc if key not in _ALLOWED_TOP_LEVEL_KEYS]
    problems += [f"volume {name} sets {sorted(config)}" for name, config in (doc.get("volumes") or {}).items() if config]
    own = _slot_networks(layout.slot_index) if layout.slot_index is not None else {}
    networks = doc.get("networks") or {}
    problems += [f"network {name} isn't one of the slot's own" for name, config in networks.items()
                 if own.get(name) != config]
    for name, service in (doc.get("services") or {}).items():
        problems += [f"{name} sets {key}" for key in service if key not in _ALLOWED_SERVICE_KEYS]
        if not (on := _service_networks(service)) or not networks.keys() >= set(on):
            problems.append(f"{name} is on {on or 'no network'}, not only the slot's own")
        if service.get("runtime") != SLOT_RUNTIME:
            problems.append(f"{name} runs under {service.get('runtime')!r}, not {SLOT_RUNTIME}")
        if service.get("ports") and name != DOOR_SERVICE_NAME:
            problems.append(f"{name} publishes ports; only the door may")
        for volume in service.get("volumes") or []:
            if problem := _volume_problem(volume, layout.app_dir):
                problems.append(f"{name} {problem}")
    return problems


def _slot_network_plan(index: int) -> dict[str, tuple[IPv4Network, str]]:
    """Each network a stack may use, with slot ``index``'s subnet and bridge name for it."""
    subnets, bridges = slot_networks(index), slot_bridges(index)
    return {ENV_NETWORK: (subnets.env, bridges.env), SITES_NETWORK: (subnets.website, bridges.website)}


def _slot_networks(index: int) -> dict[str, dict]:
    """The top-level ``networks`` entry of each network slot ``index`` may use: a bridge with its fixed name and
    subnet, and no IPv6."""
    return {network: {
        "driver": "bridge",
        "enable_ipv6": False,
        "driver_opts": {_BRIDGE_NAME_OPTION: bridge},
        "ipam": {"config": [{"subnet": str(subnet)}]},
    } for network, (subnet, bridge) in _slot_network_plan(index).items()}


def _service_networks(service: Mapping) -> list[str]:
    networks = service.get("networks") or []
    return list(networks) if isinstance(networks, (list, Mapping)) else []


def _door(services: dict[str, dict], image: str, token: str) -> dict:
    """The door service: every port the stack's services publish, taken off them, on the same host port, forwarded
    to the service's container port. Its own container listens on the host port's number."""
    routes, ports = [], []
    for name, service in services.items():
        for spec in service.pop("ports", None) or []:
            *host_ip, host_port, container_port = str(spec).split(":")
            if ENV_NETWORK not in _service_networks(service):
                raise ValueError(f"service {name!r} publishes {spec} but isn't on {ENV_NETWORK}, where the door is")
            routes.append(f"{host_port}={name}:{container_port}")
            ports.append(":".join([*host_ip, host_port, host_port]))
    return {
        "image": image,
        "command": list(DOOR_COMMAND),
        "environment": [f"{DOOR_ROUTES_ENV}={','.join(routes)}", f"{DOOR_TOKEN_ENV}={token}"],
        "ports": ports,
        "restart": "unless-stopped",
        "networks": [ENV_NETWORK],
    }


def _addresses(services: Mapping[str, Mapping], networks: Mapping[str, tuple[IPv4Network, str]]) -> dict[str, dict[str, str]]:
    """Each service's address on each of its networks, handed out in service order from the subnet's second host;
    the first is the bridge's own."""
    free = {network: iter(list(subnet.hosts())[1:]) for network, (subnet, _) in networks.items()}
    addresses: dict[str, dict[str, str]] = {}
    for name, service in services.items():
        try:
            addresses[name] = {network: str(next(free[network])) for network in _service_networks(service)}
        except StopIteration:
            raise ValueError(f"slot networks have no address left for service {name!r}") from None
    return addresses


def _volume_problem(volume: object, app_dir: str) -> str | None:
    """Why a service's volume would reach past its slot: a ``$`` Compose would fill in, the docker socket, a mount
    type other than bind, volume or tmpfs, or a bind from outside ``app_dir``. An anonymous volume (a container path
    alone) and a named volume pass."""
    if isinstance(volume, Mapping):
        source = str(volume.get("source") or "")
        kind = volume.get("type")
        if kind not in _MOUNT_TYPES:
            return f"mounts a {kind!r}, not one of {sorted(_MOUNT_TYPES)}"
        bind = kind == "bind"
    else:
        source, separator, _ = str(volume).partition(":")
        if not separator:
            source = ""
        bind = source.startswith(("/", ".", "~"))
    if "$" in source:
        return f"mounts {source}, which Compose would fill in from the box"
    if "docker.sock" in source:
        return f"mounts the docker socket ({source})"
    if not bind:
        return None
    if source.startswith("~"):
        return f"bind-mounts {source}, outside {app_dir}"
    resolved = posixpath.normpath(posixpath.join(app_dir, source))
    if resolved != app_dir and not resolved.startswith(f"{app_dir}/"):
        return f"bind-mounts {source}, outside {app_dir}"
    return None
