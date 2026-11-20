"""The walls between a shared box's slots: the networks each slot owns, the resolvers read from the box, and the
firewall script, run here against a small model of iptables that decides which packets the walls let through."""

from __future__ import annotations

import itertools
import shlex
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network, ip_address, ip_network
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.providers.sandbox_providers.shared_box_walls import (
    SLOT_COUNT,
    SLOT_RANGE,
    apply_walls_firewall,
    slot_networks,
    upstream_resolvers,
    walls_firewall_script,
)

MODAL_RESOLV_CONF = "nameserver 172.21.0.1\nnameserver 1.1.1.1\nnameserver 8.8.8.8\n"
RESOLVERS = upstream_resolvers(MODAL_RESOLV_CONF)


class IptablesFailed(Exception):
    pass


class FakeIptables:
    """The filter table as the script leaves it: chains of rules, run line by line as ``bash -e`` would. ``fail_on``
    makes the one command holding that text fail, as a broken iptables would."""

    def __init__(self, *, docker_user: bool = True, fail_on: str | None = None):
        self.chains: dict[str, list[tuple[str, ...]]] = {"INPUT": [], "FORWARD": []}
        if docker_user:
            self.chains["DOCKER-USER"] = [("-j", "RETURN")]
            self.chains["FORWARD"].append(("-j", "DOCKER-USER"))
        self.fail_on = fail_on

    def run(self, script: str) -> None:
        for line in script.splitlines():
            if line == "set -eu":
                continue
            if not any(self._command(part.replace("2>/dev/null", "")) for part in line.split(" || ")):
                raise IptablesFailed(line)

    def _command(self, command: str) -> bool:
        argv = shlex.split(command)
        assert argv[0] == "iptables"
        if self.fail_on and self.fail_on in command:
            return False
        op, chain, rest = argv[1], argv[2], tuple(argv[3:])
        if op == "-N":
            if chain in self.chains:
                return False
            self.chains[chain] = []
            return True
        if chain not in self.chains:
            return False
        rules = self.chains[chain]
        if op == "-F":
            rules.clear()
        elif op == "-A":
            rules.append(rest)
        elif op == "-I":
            rules.insert(int(rest[0]) - 1, rest[1:])
        elif op == "-C":
            return rest in rules
        elif op == "-D":
            if rest not in rules:
                return False
            rules.remove(rest)
        else:
            raise AssertionError(f"unexpected iptables {op}")
        return True

    def verdict(self, packet: Packet, chain: str) -> str:
        """``ACCEPT`` or ``DROP`` for ``packet`` entering ``chain``, whose policy, like every built-in chain's on the
        box, accepts."""
        return "DROP" if self._walk(packet, chain) == "DROP" else "ACCEPT"

    def _walk(self, packet: Packet, chain: str) -> str:
        for rule in self.chains[chain]:
            options = dict(zip(rule[::2], rule[1::2]))
            if not _matches(packet, rule, options):
                continue
            target = options["-j"]
            if target in ("DROP", "RETURN"):
                return target
            if self._walk(packet, target) == "DROP":
                return "DROP"
        return "RETURN"


def _matches(packet: Packet, rule: tuple[str, ...], options: dict[str, str]) -> bool:
    if "--ctstate" in rule and not packet.established:
        return False
    if "-s" in options and packet.src not in ip_network(options["-s"]):
        return False
    if "-d" in options and packet.dst not in ip_network(options["-d"]):
        return False
    if "-p" in options and packet.proto != options["-p"]:
        return False
    return "--dport" not in options or packet.dport == int(options["--dport"])


@dataclass(frozen=True)
class Packet:
    src: IPv4Address
    dst: IPv4Address
    proto: str = "tcp"
    dport: int = 443
    established: bool = False


def packet(src: str, dst: str, **kw) -> Packet:
    return Packet(ip_address(src), ip_address(dst), **kw)


@pytest.fixture
def walled() -> FakeIptables:
    table = FakeIptables()
    table.run(walls_firewall_script(RESOLVERS))
    return table


def host(network: IPv4Network, n: int = 2) -> str:
    return str(network[n])


def test_every_slot_owns_one_23_in_the_slot_range_split_into_its_env_and_website_24s_and_no_two_overlap():
    slots = [slot_networks(i) for i in range(SLOT_COUNT)]

    for i, s in enumerate(slots):
        assert s.slot.prefixlen == 23 and s.slot.subnet_of(SLOT_RANGE)
        assert list(s.slot.subnets(new_prefix=24)) == [s.env, s.website]
        assert s.env == IPv4Network(f"10.250.{2 * i}.0/24")
    assert not any(a.slot.overlaps(b.slot) for a, b in itertools.combinations(slots, 2))


@pytest.mark.parametrize("index", [-1, SLOT_COUNT, True, "1", 1.0])
def test_a_slot_index_outside_the_slots_or_not_an_int_is_refused(index):
    with pytest.raises(ValueError):
        slot_networks(index)


def test_the_resolvers_are_the_ipv4_nameservers_of_resolv_conf_in_order():
    conf = "# by Modal\nsearch modal.internal\nnameserver 172.21.0.1\noptions ndots:1\nnameserver fe80::1%eth0\n" \
           "nameserver 1.1.1.1\nnameserver 172.21.0.1\n"

    assert upstream_resolvers(conf) == (IPv4Address("172.21.0.1"), IPv4Address("1.1.1.1"))
    assert RESOLVERS == tuple(map(IPv4Address, ("172.21.0.1", "1.1.1.1", "8.8.8.8")))


@pytest.mark.parametrize("conf", ["", "search x\n", "nameserver ::1\n", "nameserver 10.250.3.1\n", "nameserver x\n"],
                         ids=["empty", "no nameserver", "only ipv6", "in the slot range", "not an address"])
def test_a_resolv_conf_with_no_usable_resolver_is_refused(conf):
    with pytest.raises(ValueError):
        upstream_resolvers(conf)


def test_a_slot_reaches_both_halves_of_its_own_23_and_no_other_slot(walled):
    for i in range(SLOT_COUNT):
        own = slot_networks(i)
        assert walled.verdict(packet(host(own.env), host(own.website)), "FORWARD") == "ACCEPT"
        assert walled.verdict(packet(host(own.website), host(own.env, 3)), "FORWARD") == "ACCEPT"
        for j in (j for j in range(SLOT_COUNT) if j != i):
            other = slot_networks(j)
            assert walled.verdict(packet(host(own.env), host(other.env)), "FORWARD") == "DROP"
            assert walled.verdict(packet(host(own.website), host(other.website)), "FORWARD") == "DROP"


@pytest.mark.parametrize("resolver", [str(r) for r in RESOLVERS])
def test_a_slot_reaches_each_resolver_on_port_53_only(walled, resolver):
    slot = host(slot_networks(7).env)

    for proto in ("udp", "tcp"):
        assert walled.verdict(packet(slot, resolver, proto=proto, dport=53), "FORWARD") == "ACCEPT"
    if ip_address(resolver).is_private:
        assert walled.verdict(packet(slot, resolver, dport=80), "FORWARD") == "DROP"


@pytest.mark.parametrize("dst", ["10.0.0.5", "10.251.0.2", "172.16.4.4", "172.21.0.2", "192.168.1.1",
                                 "169.254.169.254", "100.64.0.1", "100.127.255.254"])
def test_a_slot_reaches_no_private_link_local_or_carrier_address(walled, dst):
    assert walled.verdict(packet(host(slot_networks(0).env), dst, proto="udp", dport=53), "FORWARD") == "DROP"


def test_a_slot_reaches_the_internet_and_replies_to_what_reached_it(walled):
    slot = host(slot_networks(45).website)

    assert walled.verdict(packet(slot, "8.8.4.4"), "FORWARD") == "ACCEPT"
    assert walled.verdict(packet(slot, "172.20.0.9", established=True), "FORWARD") == "ACCEPT"


def test_a_slot_reaches_nothing_on_the_box_but_replies_to_the_box(walled):
    slot = host(slot_networks(3).env)

    assert walled.verdict(packet(slot, "10.250.6.1"), "INPUT") == "DROP"
    assert walled.verdict(packet(slot, "172.21.0.9", dport=22), "INPUT") == "DROP"
    assert walled.verdict(packet(slot, "10.250.6.1", established=True), "INPUT") == "ACCEPT"


def test_traffic_not_from_a_slot_is_left_to_docker(walled):
    assert walled.verdict(packet("10.251.0.2", "172.21.0.9", dport=80), "FORWARD") == "ACCEPT"
    assert walled.verdict(packet("172.21.0.9", host(slot_networks(0).env)), "FORWARD") == "ACCEPT"
    assert walled.verdict(packet("10.251.0.2", "10.251.0.1"), "INPUT") == "ACCEPT"


def test_running_the_script_again_leaves_the_same_rules_and_no_guard(walled):
    before = {chain: list(rules) for chain, rules in walled.chains.items()}

    walled.run(walls_firewall_script(RESOLVERS))

    assert walled.chains == before
    assert walled.chains["DOCKER-USER"][0] == ("-j", "AGENTENV-WALLS-FWD")
    assert walled.chains["INPUT"] == [("-j", "AGENTENV-WALLS-IN")]


def test_a_run_that_fails_part_way_leaves_every_slot_walled_off():
    table = FakeIptables()
    table.run(walls_firewall_script(RESOLVERS))
    table.fail_on = "-A AGENTENV-WALLS-FWD -s 10.250.0.0/16 -d 10.0.0.0/8"

    with pytest.raises(IptablesFailed):
        table.run(walls_firewall_script(RESOLVERS))

    slot = host(slot_networks(0).env)
    assert table.verdict(packet(slot, host(slot_networks(0).website)), "FORWARD") == "DROP"
    assert table.verdict(packet(slot, "8.8.4.4"), "FORWARD") == "DROP"
    assert table.verdict(packet(slot, "10.250.0.1"), "INPUT") == "DROP"


def test_the_script_fails_before_dockerd_has_made_its_docker_user_chain():
    with pytest.raises(IptablesFailed):
        FakeIptables(docker_user=False).run(walls_firewall_script(RESOLVERS))


def test_the_script_needs_a_resolver():
    with pytest.raises(ValueError):
        walls_firewall_script(())


@pytest.mark.asyncio
async def test_applying_the_walls_reads_the_boxs_resolvers_and_runs_the_script_for_them():
    box = MagicMock(sandbox_id="sb-1")
    box.exec_with_output = AsyncMock(return_value=(0, MODAL_RESOLV_CONF, ""))
    box.exec_script = AsyncMock(return_value="")

    await apply_walls_firewall(box)

    box.exec_with_output.assert_awaited_once_with("cat", "/etc/resolv.conf")
    assert box.exec_script.await_args.args == (walls_firewall_script(RESOLVERS),)


@pytest.mark.asyncio
async def test_applying_the_walls_raises_and_runs_nothing_when_resolv_conf_cant_be_read():
    box = MagicMock(sandbox_id="sb-1")
    box.exec_with_output = AsyncMock(return_value=(1, "", "No such file"))
    box.exec_script = AsyncMock()

    with pytest.raises(RuntimeError, match="resolv.conf"):
        await apply_walls_firewall(box)

    box.exec_script.assert_not_called()
