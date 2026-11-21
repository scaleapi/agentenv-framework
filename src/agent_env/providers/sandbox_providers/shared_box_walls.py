"""The walls between the slots of a shared box, the networks each slot owns and the firewall that keeps them apart.

Slot ``n`` of a box owns the /23 ``10.250.2n.0/23``: ``10.250.2n.0/24`` for its env's network and ``10.250.2n+1.0/24``
for its website's. The firewall is one root shell script for the whole box, every slot at once, applied once dockerd is
up and its ``DOCKER-USER`` chain exists. It only drops: what it lets through still meets Docker's own rules, so it never
opens a path Docker would close. Forwarded traffic from a slot may reach its own /23, the box's upstream resolvers on
port 53, and anything outside the private ranges; traffic from a slot to the box itself is dropped.
"""

from __future__ import annotations

import ipaddress
from ipaddress import IPv4Address, IPv4Network
from typing import NamedTuple, Sequence

from agent_env.providers.sandbox_providers.sandbox import Sandbox

SLOT_COUNT = 46
SLOT_RANGE = IPv4Network("10.250.0.0/16")
# Every slot network's Linux bridge is named with this prefix, so the firewall can match slot traffic by the interface
# it arrived on, which a container can't spoof as it can a source address.
SLOT_BRIDGE_PREFIX = "aes"
_PRIVATE_RANGES = tuple(IPv4Network(n) for n in
                        ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "100.64.0.0/10"))
FORWARD_CHAIN, INPUT_CHAIN = "AGENTENV-WALLS-FWD", "AGENTENV-WALLS-IN"
_ESTABLISHED = "-m conntrack --ctstate ESTABLISHED,RELATED"


class SlotNetworks(NamedTuple):
    slot: IPv4Network
    env: IPv4Network
    website: IPv4Network


def slot_networks(index: int) -> SlotNetworks:
    """The /23 slot ``index`` owns, and its env's and website's /24 in it."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < SLOT_COUNT:
        raise ValueError(f"slot index {index!r} is not an int from 0 to {SLOT_COUNT - 1}")
    env = IPv4Network(f"10.250.{2 * index}.0/24")
    website = IPv4Network(f"10.250.{2 * index + 1}.0/24")
    return SlotNetworks(env.supernet(), env, website)


class SlotBridges(NamedTuple):
    env: str
    website: str


def slot_bridges(index: int) -> SlotBridges:
    """The Linux bridge names of slot ``index``'s env and website networks, at most 9 characters (Linux allows 15)."""
    slot_networks(index)
    return SlotBridges(f"{SLOT_BRIDGE_PREFIX}{index}env", f"{SLOT_BRIDGE_PREFIX}{index}web")


def upstream_resolvers(resolv_conf: str) -> tuple[IPv4Address, ...]:
    """The IPv4 nameservers of a resolv.conf, in order. IPv6 ones are skipped, as slots have no IPv6; a resolver in
    the slot range, or none at all, raises."""
    resolvers = []
    for line in resolv_conf.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0] != "nameserver":
            continue
        address = ipaddress.ip_address(fields[1].split("%")[0])
        if address.version != 4:
            continue
        if address in SLOT_RANGE:
            raise ValueError(f"nameserver {address} is in the slot range {SLOT_RANGE}")
        if address not in resolvers:
            resolvers.append(address)
    if not resolvers:
        raise ValueError(f"no IPv4 nameserver in resolv.conf: {resolv_conf!r}")
    return tuple(resolvers)


def walls_firewall_script(resolvers: Sequence[IPv4Address]) -> str:
    """A root shell script that sets the box's firewall to the walls for every slot, the same rules however often it
    runs. While it refills the chains, a guard at the top of ``DOCKER-USER`` and ``INPUT`` drops all slot traffic, so
    a run that fails part way leaves the walls closed, not open."""
    if not resolvers:
        raise ValueError("the walls need at least one upstream resolver")
    slots = str(SLOT_RANGE)
    forward = [f"{_ESTABLISHED} -j RETURN"]
    forward += [f"-s {n.slot} -d {n.slot} -j RETURN" for n in map(slot_networks, range(SLOT_COUNT))]
    forward.append(f"-s {slots} -d {slots} -j DROP")
    forward += [f"-s {slots} -d {r}/32 -p {proto} --dport 53 -j RETURN" for r in resolvers for proto in ("udp", "tcp")]
    forward += [f"-s {slots} -d {n} -j DROP" for n in _PRIVATE_RANGES]
    incoming = [f"{_ESTABLISHED} -j RETURN", f"-s {slots} -j DROP"]
    guard = f"-s {slots} -j DROP"
    lines = ["set -eu"]
    for parent in ("DOCKER-USER", "INPUT"):
        lines.append(f"iptables -C {parent} {guard} 2>/dev/null || iptables -I {parent} 1 {guard}")
    for chain, parent, rules in ((FORWARD_CHAIN, "DOCKER-USER", forward), (INPUT_CHAIN, "INPUT", incoming)):
        lines.append(f"iptables -N {chain} 2>/dev/null || iptables -F {chain}")
        lines += [f"iptables -A {chain} {rule}" for rule in rules]
        lines.append(f"iptables -C {parent} -j {chain} 2>/dev/null || iptables -I {parent} 1 -j {chain}")
    for parent in ("DOCKER-USER", "INPUT"):
        lines.append(f"iptables -D {parent} {guard}")
    return "\n".join(lines) + "\n"


async def apply_walls_firewall(sandbox: Sandbox) -> None:
    """Set the walls on ``sandbox``, a shared box whose dockerd is up, against the resolvers its resolv.conf names."""
    code, resolv_conf, stderr = await sandbox.exec_with_output("cat", "/etc/resolv.conf")
    if code:
        raise RuntimeError(f"couldn't read /etc/resolv.conf on {sandbox.sandbox_id} (exit {code}): {stderr.strip()}")
    script = walls_firewall_script(upstream_resolvers(resolv_conf))
    await sandbox.exec_script(script, max_retries=1)
