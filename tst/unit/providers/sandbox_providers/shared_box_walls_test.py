"""The walls between a shared box's slots: the networks each slot owns, the resolvers read from the box, and the
firewall script, run here against a small model of iptables and ip6tables that decides which packets the walls let
through, and once under real bash."""

from __future__ import annotations

import copy
import itertools
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network, ip_address, ip_network
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.providers.sandbox_providers.shared_box_walls import (
    SLOT_BRIDGE_PREFIX,
    SLOT_COUNT,
    SLOT_RANGE,
    apply_walls_firewall,
    slot_bridges,
    slot_networks,
    upstream_resolvers,
    walls_firewall_script,
)

MODAL_RESOLV_CONF = "nameserver 172.21.0.1\nnameserver 1.1.1.1\nnameserver 8.8.8.8\n"
RESOLVERS = upstream_resolvers(MODAL_RESOLV_CONF)
TOOLS = {"iptables": 4, "ip6tables": 6}


class IptablesFailed(Exception):
    pass


class FakeIptables:
    """The filter tables of iptables and ip6tables as the script leaves them, the script run line by line as
    ``bash -e`` would, each restore all or nothing. ``fail_on`` makes the one command or restore line holding that text
    fail, and ``missing`` names tools the box lacks. ``calls`` is every command run, in order."""

    def __init__(self, *, docker_user: bool = True, fail_on: str | None = None, missing: tuple[str, ...] = ()):
        self.tables: dict[int, dict[str, list[tuple[str, ...]]]] = {
            family: {"INPUT": [], "FORWARD": []} for family in TOOLS.values()}
        if docker_user:
            self.tables[4]["DOCKER-USER"] = [("-j", "RETURN")]
            self.tables[4]["FORWARD"].append(("-j", "DOCKER-USER"))
        self.fail_on, self.missing = fail_on, missing
        self.calls: list[str] = []

    @property
    def chains(self) -> dict[str, list[tuple[str, ...]]]:
        return self.tables[4]

    def run(self, script: str) -> None:
        lines = iter(script.splitlines())
        for line in lines:
            if line == "set -eu":
                continue
            if "<<'WALLS'" in line:
                body = list(itertools.takewhile(lambda body_line: body_line != "WALLS", lines))
                ok = self._restore(line.split("<<")[0].strip(), body)
            else:
                ok = any(self._command(part.replace("2>/dev/null", "")) for part in line.split(" || "))
            if not ok:
                raise IptablesFailed(line)

    def _tool(self, command: str) -> tuple[list[str], dict[str, list[tuple[str, ...]]]] | None:
        self.calls.append(command.strip())
        argv = shlex.split(command)
        tool = argv[0].removesuffix("-restore")
        assert tool in TOOLS and "-w" in argv
        if argv[0] in self.missing or (self.fail_on and self.fail_on in command):
            return None
        return [a for a in argv[1:] if a != "-w"], self.tables[TOOLS[tool]]

    def _restore(self, command: str, body: list[str]) -> bool:
        found = self._tool(command)
        if found is None:
            return False
        args, table = found
        assert args == ["--noflush"] and body[0] == "*filter" and body[-1] == "COMMIT"
        staged = copy.deepcopy(table)
        for line in body[1:-1]:
            if self.fail_on and self.fail_on in line:
                return False
            if line.startswith(":"):
                staged.setdefault(line[1:].split()[0], []).clear()
                continue
            op, chain, *rule = shlex.split(line)
            if chain not in staged:
                return False
            if op == "-F":
                staged[chain].clear()
            elif op == "-A":
                staged[chain].append(tuple(rule))
            else:
                raise AssertionError(f"unexpected restore line {line!r}")
        table.clear()
        table.update(staged)
        return True

    def _command(self, command: str) -> bool:
        found = self._tool(command)
        if found is None:
            return False
        (op, chain, *rest), table = found
        if chain not in table:
            return False
        rules, rest = table[chain], tuple(rest)
        if op == "-I":
            rules.insert(int(rest[0]) - 1, rest[1:])
        elif op == "-C":
            return rest in rules
        else:
            raise AssertionError(f"unexpected iptables {op}")
        return True

    def verdict(self, packet: Packet, chain: str) -> str:
        """``ACCEPT`` or ``DROP`` for ``packet`` entering ``chain``, whose policy, like every built-in chain's on the
        box, accepts."""
        return "DROP" if self._walk(packet, chain) == "DROP" else "ACCEPT"

    def _walk(self, packet: Packet, chain: str) -> str:
        for rule in self.tables[packet.dst.version][chain]:
            options = dict(zip(rule[::2], rule[1::2]))
            if not _matches(packet, rule, options):
                continue
            target = options["-j"]
            if target in ("DROP", "RETURN"):
                return target
            if self._walk(packet, target) == "DROP":
                return "DROP"
        return "RETURN"


def _interface(pattern: str, name: str | None) -> bool:
    return name is not None and (name.startswith(pattern[:-1]) if pattern.endswith("+") else name == pattern)


def _matches(packet: Packet, rule: tuple[str, ...], options: dict[str, str]) -> bool:
    if "--ctstate" in rule and not packet.established:
        return False
    if "-i" in options and not _interface(options["-i"], packet.in_iface):
        return False
    if "-o" in options and not _interface(options["-o"], packet.out_iface):
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
    in_iface: str
    out_iface: str | None = None
    proto: str = "tcp"
    dport: int = 443
    established: bool = False


def packet(src: str, dst: str, in_iface: str = "eth0", out_iface: str | None = None, **kw) -> Packet:
    return Packet(ip_address(src), ip_address(dst), in_iface, out_iface, **kw)


def from_slot(index: int, dst: str, *, half: str = "env", out_iface: str | None = "eth0", src: str | None = None,
              **kw) -> Packet:
    """A packet that came in on slot ``index``'s ``half`` bridge, from an address of that half unless ``src`` spoofs
    another."""
    network = getattr(slot_networks(index), half)
    return packet(src or host(network), dst, getattr(slot_bridges(index), half), out_iface, **kw)


def to_slot(index: int, half: str = "env") -> dict[str, str]:
    """The destination and out interface of a packet to slot ``index``'s ``half``."""
    return {"dst": host(getattr(slot_networks(index), half), 3), "out_iface": getattr(slot_bridges(index), half)}


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


def test_a_slot_reaches_both_its_bridges_and_no_other_slots(walled):
    for i in range(SLOT_COUNT):
        for half, other_half in itertools.product(("env", "website"), repeat=2):
            assert walled.verdict(from_slot(i, half=half, **to_slot(i, other_half)), "FORWARD") == "ACCEPT"
        for j in (j for j in range(SLOT_COUNT) if j != i):
            assert walled.verdict(from_slot(i, **to_slot(j)), "FORWARD") == "DROP"
            assert walled.verdict(from_slot(i, half="website", **to_slot(j, "website")), "FORWARD") == "DROP"


def test_a_slot_that_spoofs_another_slots_address_is_still_walled_in_by_its_bridge(walled):
    spoofed = host(slot_networks(5).env)

    assert walled.verdict(from_slot(3, src=spoofed, **to_slot(5)), "FORWARD") == "DROP"
    assert walled.verdict(from_slot(3, src=spoofed, **to_slot(5, "website")), "FORWARD") == "DROP"
    assert walled.verdict(from_slot(3, "10.0.0.5", src="8.8.8.8"), "FORWARD") == "DROP"
    assert walled.verdict(from_slot(3, "10.250.6.1", src="172.21.0.9", out_iface=None), "INPUT") == "DROP"


@pytest.mark.parametrize("resolver", [str(r) for r in RESOLVERS])
def test_a_slot_reaches_each_resolver_on_port_53_only(walled, resolver):
    for proto in ("udp", "tcp"):
        assert walled.verdict(from_slot(7, resolver, proto=proto, dport=53), "FORWARD") == "ACCEPT"
    if ip_address(resolver).is_private:
        assert walled.verdict(from_slot(7, resolver, dport=80), "FORWARD") == "DROP"


@pytest.mark.parametrize("dst", ["10.0.0.5", "10.251.0.2", "172.16.4.4", "172.21.0.2", "192.168.1.1",
                                 "169.254.169.254", "100.64.0.1", "100.127.255.254"])
def test_a_slot_reaches_no_private_link_local_or_carrier_address(walled, dst):
    assert walled.verdict(from_slot(0, dst, proto="udp", dport=53), "FORWARD") == "DROP"


def test_a_slot_reaches_the_internet_and_replies_to_what_reached_it(walled):
    assert walled.verdict(from_slot(45, "8.8.4.4", half="website"), "FORWARD") == "ACCEPT"
    assert walled.verdict(from_slot(45, "172.20.0.9", half="website", established=True), "FORWARD") == "ACCEPT"
    assert walled.verdict(from_slot(45, half="website", established=True, **to_slot(2)), "FORWARD") == "ACCEPT"


def test_a_slot_reaches_nothing_on_the_box_but_replies_to_the_box(walled):
    assert walled.verdict(from_slot(3, "10.250.6.1", out_iface=None), "INPUT") == "DROP"
    assert walled.verdict(from_slot(3, "172.21.0.9", out_iface=None, dport=22), "INPUT") == "DROP"
    assert walled.verdict(from_slot(3, "10.250.6.1", out_iface=None, established=True), "INPUT") == "ACCEPT"


def test_no_ipv6_from_a_slot_is_forwarded_or_reaches_the_box_even_as_a_reply(walled):
    for established in (False, True):
        for dst, out_iface in (("2001:4860:4860::8888", "eth0"), ("fd00::2", "aes3web"), ("fe80::1", "aes3env")):
            assert walled.verdict(packet("fd00::9", dst, "aes3env", out_iface, established=established),
                                  "FORWARD") == "DROP"
        assert walled.verdict(packet("fd00::9", "fe80::1", "aes3env", established=established), "INPUT") == "DROP"


def test_traffic_not_from_a_slot_is_left_to_docker(walled):
    assert walled.verdict(packet("10.251.0.2", "172.21.0.9", "br-1a2b", "eth0", dport=80), "FORWARD") == "ACCEPT"
    assert walled.verdict(packet("172.21.0.9", in_iface="eth0", **to_slot(0)), "FORWARD") == "ACCEPT"
    assert walled.verdict(packet("10.251.0.2", "10.251.0.1", "br-1a2b"), "INPUT") == "ACCEPT"
    assert walled.verdict(packet("2001:db8::2", "2001:db8::1", "eth0"), "INPUT") == "ACCEPT"


def test_each_restore_is_one_transaction_that_empties_and_fills_both_walls_chains():
    lines = walls_firewall_script(RESOLVERS).splitlines()
    restores = [i for i, line in enumerate(lines) if "-restore" in line]

    assert [lines[i] for i in restores] == ["iptables-restore -w --noflush <<'WALLS'",
                                            "ip6tables-restore -w --noflush <<'WALLS'"]
    for i in restores:
        body = lines[i + 1:lines.index("WALLS", i)]
        assert body[0] == "*filter" and body[-1] == "COMMIT" and body.count("COMMIT") == 1
        for chain in ("AGENTENV-WALLS-FWD", "AGENTENV-WALLS-IN"):
            assert f":{chain} - [0:0]" in body
            assert body.index(f"-F {chain}") < min(n for n, line in enumerate(body) if line.startswith(f"-A {chain} "))


def test_running_the_script_again_leaves_the_same_rules(walled):
    before = copy.deepcopy(walled.tables)

    walled.run(walls_firewall_script(RESOLVERS))

    assert walled.tables == before
    assert walled.chains["DOCKER-USER"][0] == ("-j", "AGENTENV-WALLS-FWD")
    assert walled.chains["INPUT"] == [("-j", "AGENTENV-WALLS-IN")]
    assert walled.tables[6]["FORWARD"] == [("-j", "AGENTENV-WALLS-FWD")]
    assert walled.tables[6]["INPUT"] == [("-j", "AGENTENV-WALLS-IN")]


def test_a_run_over_older_walls_replaces_their_rules():
    table = FakeIptables()
    table.run(walls_firewall_script(RESOLVERS))
    table.chains["AGENTENV-WALLS-FWD"].insert(0, ("-s", "10.250.0.0/16", "-j", "RETURN"))

    table.run(walls_firewall_script(RESOLVERS))

    assert table.verdict(from_slot(0, **to_slot(1)), "FORWARD") == "DROP"
    assert table.chains["DOCKER-USER"].count(("-j", "AGENTENV-WALLS-FWD")) == 1


@pytest.mark.parametrize("fail_on, family", [("-A AGENTENV-WALLS-FWD -i aes+ -d 10.0.0.0/8", 4),
                                             ("-A AGENTENV-WALLS-IN -i aes+ -j DROP", 4), ("ip6tables-restore", 6)])
def test_a_restore_that_fails_leaves_its_walls_as_they_were_and_nothing_runs_after(fail_on, family):
    table = FakeIptables()
    table.run(walls_firewall_script(RESOLVERS))
    for chains in table.tables.values():
        chains["AGENTENV-WALLS-FWD"].append(("-j", "DROP"))
    before = copy.deepcopy(table.tables[family])
    table.calls.clear()
    table.fail_on = fail_on

    with pytest.raises(IptablesFailed):
        table.run(walls_firewall_script(RESOLVERS))

    assert table.tables[family] == before
    assert "-restore" in table.calls[-1] and not any(" -C " in call or " -I " in call for call in table.calls)


def test_a_box_without_ip6tables_fails_the_script_before_any_wall_is_hooked_in():
    table = FakeIptables(missing=("ip6tables-restore",))

    with pytest.raises(IptablesFailed, match="ip6tables-restore"):
        table.run(walls_firewall_script(RESOLVERS))

    assert table.chains["DOCKER-USER"] == [("-j", "RETURN")] and table.chains["INPUT"] == []


def test_the_script_fails_before_dockerd_has_made_its_docker_user_chain():
    with pytest.raises(IptablesFailed):
        FakeIptables(docker_user=False).run(walls_firewall_script(RESOLVERS))


# A stand-in for one iptables tool: it logs its command line and any restore input it is fed, says no rule exists to
# ``-C``, and fails any command matching the case pattern FAIL.
_STUB = """#!BASH
call="${0##*/} $*"
printf '%s\\n' "$call" >> LOG
case "$call" in *-restore*) while IFS= read -r line; do printf '%s\\n' "$line" >> LOG; done ;; esac
case "$call" in FAIL) exit 1 ;; *" -C "*) exit 1 ;; esac
"""


def _run_under_bash(tmp_path, tools: tuple[str, ...], fail: str = "__never__") -> tuple[int, list[str]]:
    """Run the script under bash with only ``tools``, as stubs, on its PATH: its exit code and the stubs' log."""
    bash, log = shutil.which("bash"), tmp_path / "log"
    (tmp_path / "bin").mkdir()
    for tool in tools:
        stub = tmp_path / "bin" / tool
        stub.write_text(_STUB.replace("BASH", bash).replace("LOG", str(log)).replace("FAIL", fail))
        stub.chmod(0o755)
    done = subprocess.run([bash, "-c", walls_firewall_script(RESOLVERS)], env={"PATH": str(tmp_path / "bin")},
                          capture_output=True, text=True)
    return done.returncode, log.read_text().splitlines() if log.exists() else []


_ALL_TOOLS = ("iptables", "iptables-restore", "ip6tables", "ip6tables-restore")


def test_under_bash_each_restore_reads_its_whole_input_then_each_wall_is_hooked_in(tmp_path):
    code, log = _run_under_bash(tmp_path, _ALL_TOOLS)

    script = walls_firewall_script(RESOLVERS).splitlines()
    assert code == 0
    assert log[:log.index("ip6tables-restore -w --noflush")] == \
        ["iptables-restore -w --noflush", *script[2:script.index("WALLS")]]
    assert [line for line in log if " -I " in line] == [
        "iptables -w -I DOCKER-USER 1 -j AGENTENV-WALLS-FWD", "iptables -w -I INPUT 1 -j AGENTENV-WALLS-IN",
        "ip6tables -w -I FORWARD 1 -j AGENTENV-WALLS-FWD", "ip6tables -w -I INPUT 1 -j AGENTENV-WALLS-IN"]


@pytest.mark.parametrize("tools, fail", [
    pytest.param(_ALL_TOOLS, "iptables-restore*", id="iptables-restore fails"),
    pytest.param(_ALL_TOOLS, "ip6tables-restore*", id="ip6tables-restore fails"),
    pytest.param(("iptables", "iptables-restore", "ip6tables"), "__never__", id="no ip6tables-restore"),
])
def test_under_bash_a_failed_or_missing_restore_fails_the_script_before_any_hook(tmp_path, tools, fail):
    code, log = _run_under_bash(tmp_path, tools, fail)

    assert code != 0
    assert not any(" -C " in line or " -I " in line for line in log)


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
async def test_applying_the_walls_raises_when_a_restore_fails_and_keeps_the_walls_as_they_were():
    table = FakeIptables()
    table.run(walls_firewall_script(RESOLVERS))
    before = copy.deepcopy(table.tables)
    table.fail_on = "ip6tables-restore"
    box = MagicMock(sandbox_id="sb-1")
    box.exec_with_output = AsyncMock(return_value=(0, MODAL_RESOLV_CONF, ""))
    box.exec_script = AsyncMock(side_effect=lambda script, **_: table.run(script))

    with pytest.raises(IptablesFailed):
        await apply_walls_firewall(box)

    assert table.tables == before


@pytest.mark.asyncio
async def test_applying_the_walls_raises_and_runs_nothing_when_resolv_conf_cant_be_read():
    box = MagicMock(sandbox_id="sb-1")
    box.exec_with_output = AsyncMock(return_value=(1, "", "No such file"))
    box.exec_script = AsyncMock()

    with pytest.raises(RuntimeError, match="resolv.conf"):
        await apply_walls_firewall(box)

    box.exec_script.assert_not_called()


def test_every_slot_network_has_its_own_bridge_name_that_linux_accepts():
    names = [name for index in range(SLOT_COUNT) for name in slot_bridges(index)]
    assert len(set(names)) == len(names) == 2 * SLOT_COUNT
    assert all(name.startswith(SLOT_BRIDGE_PREFIX) and len(name) <= 15 for name in names)
    assert slot_bridges(7) == ("aes7env", "aes7web")
    with pytest.raises(ValueError):
        slot_bridges(SLOT_COUNT)
