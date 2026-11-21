"""The walls between the slots of a shared box, the networks each slot owns and the firewall that keeps them apart.

Slot ``n`` of a box owns the /23 ``10.250.2n.0/23``: ``10.250.2n.0/24`` for its env's network and ``10.250.2n+1.0/24``
for its website's, on the bridges ``aes<n>env`` and ``aes<n>web``. The firewall is one root shell script for the whole
box, every slot at once, applied once dockerd is up and its ``DOCKER-USER`` chain exists. It matches slot traffic by
the bridge it came in on, which a container can't spoof as it can a source address. It only drops: what it lets
through still meets Docker's own rules, so it never opens a path Docker would close. Forwarded traffic from a slot may
reach its own two bridges, the box's upstream resolvers on port 53, and anything outside the private ranges; traffic
from a slot to the box itself, and any IPv6 from a slot, is dropped.
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
# Where each walls chain is jumped to. The box's dockerd isn't set up for IPv6, so it may make no ip6tables
# DOCKER-USER; the IPv6 walls hook into FORWARD itself.
_HOOKS = (("iptables", "DOCKER-USER", FORWARD_CHAIN), ("iptables", "INPUT", INPUT_CHAIN),
          ("ip6tables", "FORWARD", FORWARD_CHAIN), ("ip6tables", "INPUT", INPUT_CHAIN))


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


def _restore_input(chains: dict[str, list[str]]) -> str:
    """``iptables-restore --noflush`` input that empties ``chains``, creating any that are missing, and fills them with
    their rules, in one transaction: it changes every chain or, if any line fails, none."""
    lines = ["*filter", *(f":{chain} - [0:0]" for chain in chains)]
    for chain, rules in chains.items():
        lines.append(f"-F {chain}")
        lines += [f"-A {chain} {rule}" for rule in rules]
    lines.append("COMMIT")
    return "\n".join(lines) + "\n"


def walls_firewall_script(resolvers: Sequence[IPv4Address]) -> str:
    """A root shell script that sets the box's firewall to the walls for every slot, the same rules however often it
    runs. Each of iptables and ip6tables refills the walls' chains in one transaction, so a run that fails leaves the
    walls as they were, then each chain is hooked in once. A box without ip6tables fails the script."""
    if not resolvers:
        raise ValueError("the walls need at least one upstream resolver")
    from_slot = f"-i {SLOT_BRIDGE_PREFIX}+"
    forward = [f"{_ESTABLISHED} -j RETURN"]
    for bridges in map(slot_bridges, range(SLOT_COUNT)):
        forward += [f"-i {a} -o {b} -j RETURN" for a in bridges for b in bridges]
    forward.append(f"{from_slot} -o {SLOT_BRIDGE_PREFIX}+ -j DROP")
    forward += [f"{from_slot} -d {r}/32 -p {proto} --dport 53 -j RETURN" for r in resolvers for proto in ("udp", "tcp")]
    forward += [f"{from_slot} -d {n} -j DROP" for n in _PRIVATE_RANGES]
    tables = {
        "iptables": {FORWARD_CHAIN: forward, INPUT_CHAIN: [f"{_ESTABLISHED} -j RETURN", f"{from_slot} -j DROP"]},
        "ip6tables": {FORWARD_CHAIN: [f"{from_slot} -j DROP"], INPUT_CHAIN: [f"{from_slot} -j DROP"]},
    }
    lines = ["set -eu"]
    for tool, chains in tables.items():
        lines += [f"{tool}-restore -w --noflush <<'WALLS'", _restore_input(chains) + "WALLS"]
    for tool, parent, chain in _HOOKS:
        lines.append(f"{tool} -w -C {parent} -j {chain} 2>/dev/null || {tool} -w -I {parent} 1 -j {chain}")
    return "\n".join(lines) + "\n"


async def apply_walls_firewall(sandbox: Sandbox) -> None:
    """Set the walls on ``sandbox``, a shared box whose dockerd is up, against the resolvers its resolv.conf names."""
    code, resolv_conf, stderr = await sandbox.exec_with_output("cat", "/etc/resolv.conf")
    if code:
        raise RuntimeError(f"couldn't read /etc/resolv.conf on {sandbox.sandbox_id} (exit {code}): {stderr.strip()}")
    script = walls_firewall_script(upstream_resolvers(resolv_conf))
    await sandbox.exec_script(script, max_retries=1)
