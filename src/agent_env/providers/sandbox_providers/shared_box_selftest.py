"""The probes of a new shared box's walls, run once dockerd is up and the walls are applied, before any slot is.

The probes run in order from one table, each a small function whose answer is a short string that passes only if it is
one of the probe's expected answers. They check Docker's settings, the firewall's rules, then bring up two probe slots,
0 and 1, from compose files the real ``slot_compose`` renders, and check from slot 0 that nothing outside it is
reachable but its own services, the box's resolvers and the internet, that the door of slot 1 takes only its token,
and that both slots tear down. Every container is the pinned Python image the gateway is built on; the door's is that
image with the door's three dependencies, pinned, and the gateway package's sources from this checkout, built on the
box, so the probe runs this checkout's ``door.py`` with the command ``slot_compose`` gives it.

Any probe that answers otherwise, raises or runs out of time ends the run: the box's walls are not verified.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import shlex
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent_env.env import gateway as gateway_package
from agent_env.env.gateway.constants import SLOT_TOKEN_HEADER
from agent_env.providers.env_providers.constants import ENV_NETWORK
from agent_env.providers.env_providers.env_gateway_provider import tear_down_slot
from agent_env.providers.env_providers.slot_compose import (
    DOOR_SERVICE_NAME,
    SLOT_RUNTIME,
    resolv_conf_path,
    slot_compose,
    slot_resolv_conf,
)
from agent_env.providers.env_providers.stack_layout import StackLayout
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.providers.sandbox_providers.shared_box_walls import PRIVATE_RANGES, upstream_resolvers, walls_check_script

logger = logging.getLogger(__name__)

WALLS_PROBE_EVENT = "agent_env.shared_box_walls_probe"
#: Every probe's answer is the last line its script prints; this line before it names the probe, so a log of the box's
#: execs, or a test's fake box, can tell which probe ran.
PROBE_MARKER = "# walls probe: "
#: The whole run's limit, in seconds, past which the probe running is cut off and the walls are not verified.
TOTAL_BUDGET_S = 300.0

#: The gateway image's base, by digest: every probe container runs it.
PROBE_IMAGE = ("public.ecr.aws/docker/library/python:3.11-slim"
               "@sha256:c20888b6acdd1e63e1c433a185bf3ad162c0288fe484616ce062e0d28add2900")
_DOOR_DEPENDENCIES = ("httpx==0.28.1", "uvicorn==0.49.0", "websockets==17.2")
_DOOR_SOURCES = ("__init__.py", "constants.py", "door.py")
DOOR_IMAGE = "agentenv-walls-probe-door:local"
_PROBE_SLOTS = (0, 1)
_PROBER, _PEER = "probe", "web"
_PEER_PORT = 8080
#: Slot 0's scratch dir, which its teardown removes.
_DOOR_CONTEXT = StackLayout.for_slot(0).scratch_path("door")
_PUBLIC_HOST = "www.google.com"
_CONNECTIVITY_URL = f"https://{_PUBLIC_HOST}/generate_204"
_PUBLIC_IPV6 = ("[2606:4700:4700::1111]:443", "[2001:4860:4860::8888]:443")
_BLOCKED = frozenset({"blocked"})


class WallsNotVerified(RuntimeError):
    """A probe of a shared box's walls answered otherwise than expected, raised or ran out of time. ``results`` holds
    every probe that ran, the failed one last."""

    def __init__(self, sandbox_id: str, results: list[ProbeResult]):
        self.results = results
        failed = results[-1]
        super().__init__(f"the walls of shared box {sandbox_id} failed the probe {failed.name}: expected "
                         f"{' or '.join(sorted(failed.expected))}, got {failed.observed!r}")


@dataclass(frozen=True)
class ProbeResult:
    name: str
    expected: frozenset[str]
    observed: str
    seconds: float

    @property
    def passed(self) -> bool:
        return self.observed in self.expected


@dataclass(frozen=True)
class BoxFacts:
    """What the probes need to know of the box: its own IPv4 address, its kernel's release and its resolv.conf."""

    address: str
    kernel: str
    resolv_conf: str


@dataclass(frozen=True)
class ProbeSlot:
    layout: StackLayout
    compose: str
    door_token: str
    peer_address: str
    door_address: str

    @property
    def door_port(self) -> int:
        return self.layout.host_port(_PEER_PORT)

    def exec_in_prober(self, *argv: str) -> str:
        return shlex.join([*self.layout.compose_argv, "exec", "-T", _PROBER, *argv])


@dataclass(frozen=True)
class ProbePlan:
    facts: BoxFacts
    slots: tuple[ProbeSlot, ProbeSlot]
    no_new_privileges: bool
    open_egress: bool

    @property
    def inside(self) -> ProbeSlot:
        return self.slots[0]

    @property
    def other(self) -> ProbeSlot:
        return self.slots[1]


@dataclass(frozen=True)
class Probe:
    name: str
    expected: frozenset[str]
    timeout_s: float
    run: Callable[[VmSandbox, ProbePlan], Awaitable[str]]
    #: Run only when the box's egress is open: under an allowlist, the internet and its names aren't the walls' to give.
    needs_open_egress: bool = False


async def run_walls_probes(box: VmSandbox, *, dockerd_config: Mapping) -> list[ProbeResult]:
    """Every probe's result, in order, once all have passed on ``box``, a new shared box whose dockerd is up, run
    from ``dockerd_config``, and whose walls are applied; raises WallsNotVerified at the first that doesn't. It leaves
    no probe slot behind when they pass; when one fails, the caller ends the box."""
    deadline = time.monotonic() + TOTAL_BUDGET_S
    results: list[ProbeResult] = []

    async def attempt(name: str, expected: frozenset[str], timeout_s: float, run: Callable[[], Awaitable[str]]) -> None:
        started = time.monotonic()
        try:
            observed = await asyncio.wait_for(run(), timeout=max(0.0, min(timeout_s, deadline - started)))
        except asyncio.TimeoutError:
            observed = f"no answer within {min(timeout_s, deadline - started):.0f}s"
        except Exception as e:
            observed = f"raised {type(e).__name__}: {e}"
        result = ProbeResult(name, expected, observed, time.monotonic() - started)
        results.append(result)
        _log(box, result)
        if not result.passed:
            raise WallsNotVerified(box.sandbox_id, results)

    plan: ProbePlan | None = None

    async def read_plan() -> str:
        nonlocal plan
        plan = await _plan(box, dockerd_config)
        return "read"

    await attempt("box_facts", frozenset({"read"}), 30, read_plan)
    for probe in PROBES:
        if probe.needs_open_egress and not plan.open_egress:
            logger.info(f"walls probe {probe.name} not run on {box.sandbox_id}: its egress is allowlisted")
            continue
        await attempt(probe.name, probe.expected, probe.timeout_s, lambda probe=probe: probe.run(box, plan))
    return results


def _log(box: VmSandbox, result: ProbeResult) -> None:
    logger.info(f"walls probe {result.name} on {box.sandbox_id}: {'passed' if result.passed else 'FAILED'}, got "
                f"{result.observed!r} in {result.seconds:.1f}s",
                extra={"event": WALLS_PROBE_EVENT, "sandbox_id": box.sandbox_id, "probe": result.name,
                       "passed": result.passed, "expected": sorted(result.expected), "observed": result.observed,
                       "seconds": round(result.seconds, 2)})


_FACTS = """
import json, platform, socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("1.1.1.1", 53))
print(json.dumps({"address": s.getsockname()[0], "kernel": platform.release(),
                  "resolv_conf": open("/etc/resolv.conf").read()}))
"""


async def _plan(box: VmSandbox, dockerd_config: Mapping) -> ProbePlan:
    answer = await _answer(box, "box_facts", shlex.join(["python3", "-c", _FACTS]))
    try:
        facts = BoxFacts(**json.loads(answer))
    except (ValueError, TypeError):
        raise RuntimeError(f"the box's address, kernel and resolv.conf weren't read: {answer}") from None
    policy = box.network_policy
    return ProbePlan(facts, tuple(_probe_slot(StackLayout.for_slot(i)) for i in _PROBE_SLOTS),
                     no_new_privileges=bool(dockerd_config.get("no-new-privileges")),
                     open_egress=policy is None or not policy.restricts_egress)


def _probe_slot(layout: StackLayout) -> ProbeSlot:
    """A slot of a prober, which runs nothing, and a peer, a web server whose port the door takes over."""
    stack = {"services": {
        _PROBER: {"image": PROBE_IMAGE, "command": ["sleep", "infinity"], "networks": [ENV_NETWORK]},
        _PEER: {"image": PROBE_IMAGE, "command": ["python", "-m", "http.server", "--bind", "0.0.0.0", str(_PEER_PORT)],
                "networks": [ENV_NETWORK], "ports": [f"{layout.host_port(_PEER_PORT)}:{_PEER_PORT}"]},
    }}
    token = secrets.token_urlsafe(32)
    compose = slot_compose(yaml.safe_dump(stack), layout, door_image=DOOR_IMAGE, door_token=token)
    services = yaml.safe_load(compose)["services"]
    return ProbeSlot(layout, compose, token, services[_PEER]["networks"][ENV_NETWORK]["ipv4_address"],
                     services[DOOR_SERVICE_NAME]["networks"][ENV_NETWORK]["ipv4_address"])


async def _answer(box: VmSandbox, name: str, script: str) -> str:
    """The last line ``script``, run as probe ``name``, printed, or how it failed."""
    code, stdout, stderr = await box.exec_with_output("bash", "-c", f"{PROBE_MARKER}{name}\n{script}")
    if code:
        return f"exit {code}: {(stderr.strip() or stdout.strip())[-300:]}"
    lines = stdout.strip().splitlines()
    return lines[-1].strip() if lines else ""


_DOCKER_INFO = """
import json, sys
info = json.load(sys.stdin)
options = info.get("SecurityOptions") or []
problems = []
if info.get("DefaultRuntime") != sys.argv[1]:
    problems.append("the default runtime is " + str(info.get("DefaultRuntime")))
if not any(option.startswith("name=userns") for option in options):
    problems.append("no userns in " + str(options))
if sys.argv[2] == "1" and "name=no-new-privileges" not in options:
    problems.append("no no-new-privileges in " + str(options))
print("; ".join(problems) or "ok")
"""


async def _docker_runs_gvisor_in_a_user_namespace(box: VmSandbox, plan: ProbePlan) -> str:
    check = shlex.join(["python3", "-c", _DOCKER_INFO, SLOT_RUNTIME, "1" if plan.no_new_privileges else "0"])
    return await _answer(box, "docker_info", f"set -o pipefail\ndocker info --format '{{{{json .}}}}' | {check}")


async def _firewall_holds_the_walls(box: VmSandbox, plan: ProbePlan) -> str:
    return await _answer(box, "firewall", walls_check_script(upstream_resolvers(plan.facts.resolv_conf)))


async def _probe_slots_come_up(box: VmSandbox, plan: ProbePlan) -> str:
    files = {f"{_DOOR_CONTEXT}/Dockerfile": _door_dockerfile().encode()}
    files |= {f"{_DOOR_CONTEXT}/gateway/{name}": (Path(gateway_package.__file__).parent / name).read_bytes()
              for name in _DOOR_SOURCES}
    for slot in plan.slots:
        files[slot.layout.compose_file] = slot.compose.encode()
        files[resolv_conf_path(slot.layout)] = slot_resolv_conf(plan.facts.resolv_conf).encode()
    for path, data in files.items():
        await box.write_host_file(data, path)
    up = [f"{slot.layout.compose_command} up -d --quiet-pull" for slot in plan.slots]
    return await _answer(box, "probe_slots_up", "\n".join(
        ["set -eu", f"docker build -q -t {DOOR_IMAGE} {_DOOR_CONTEXT} >/dev/null", *up, "echo up"]))


def _door_dockerfile() -> str:
    return "\n".join([f"FROM {PROBE_IMAGE}", f"RUN pip install --no-cache-dir {' '.join(_DOOR_DEPENDENCIES)}",
                      "COPY gateway /app/gateway", "ENV PYTHONPATH=/app", ""])


async def _slots_run_under_gvisor(box: VmSandbox, plan: ProbePlan) -> str:
    listed = [f"ids=\"$ids $(docker ps -q --filter label=com.docker.compose.project={slot.layout.compose_project})\""
              for slot in plan.slots]
    return await _answer(box, "slots_under_gvisor", "\n".join([
        "set -eu", 'ids=""', *listed,
        '[ -n "$(echo $ids)" ] || { echo "no probe container is running"; exit 0; }',
        "others=$(docker inspect -f '{{.Name}} {{.HostConfig.Runtime}}' $ids | grep -v ' " + SLOT_RUNTIME + "$' || true)",
        f"kernel=$({plan.inside.exec_in_prober('uname', '-r')})",
        'if [ -n "$others" ]; then echo "not under ' + SLOT_RUNTIME + ': $others"',
        f'elif [ "$kernel" = {shlex.quote(plan.facts.kernel)} ]; then echo "a slot runs the box\'s kernel"',
        "else echo gvisor; fi",
    ]))


_GET = """
import sys, time, urllib.request
deadline = time.monotonic() + float(sys.argv[2])
while True:
    try:
        print(urllib.request.urlopen(sys.argv[1], timeout=5).status)
        break
    except Exception as e:
        if time.monotonic() > deadline:
            print(type(e).__name__ + ": " + str(e))
            break
        time.sleep(1)
"""


def _peer_by_name(index: int) -> Callable[[VmSandbox, ProbePlan], Awaitable[str]]:
    async def run(box: VmSandbox, plan: ProbePlan) -> str:
        slot = plan.slots[index]
        url = f"http://{_PEER}:{_PEER_PORT}/"
        return await _answer(box, f"slot_{index}_peer_by_name", slot.exec_in_prober("python", "-c", _GET, url, "30"))
    return run


_REACH = """
import socket, sys
reached = []
for target in sys.argv[1:]:
    host, _, port = target.rpartition(":")
    host = host.strip("[]")
    try:
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as s:
            s.settimeout(3)
            s.connect((host, int(port)))
        reached.append(target + " open")
    except ConnectionRefusedError:
        reached.append(target + " refused")
    except OSError:
        pass
print("reached " + ", ".join(reached) if reached else "blocked")
"""


def _private_range_targets(plan: ProbePlan) -> list[str]:
    """Private addresses a slot must not reach, including a private resolver on a port other than DNS. A public
    resolver such as 1.1.1.1 is internet, which a slot may reach, so it isn't one."""
    private_resolvers = [r for r in upstream_resolvers(plan.facts.resolv_conf) if any(r in n for n in PRIVATE_RANGES)]
    return ["10.0.0.1:80", "169.254.169.254:80", "192.168.0.1:80", "100.64.0.1:80",
            *(f"{r}:80" for r in private_resolvers)]


def _blocked_from_inside(name: str, targets: Callable[[ProbePlan], list[str]]) -> Probe:
    """A probe that slot 0's prober reaches none of ``targets``: a connection that opens or is refused got there."""
    async def run(box: VmSandbox, plan: ProbePlan) -> str:
        return await _answer(box, name, plan.inside.exec_in_prober("python", "-c", _REACH, *targets(plan)))
    return Probe(name, _BLOCKED, 45, run)


_DNS = """
import socket, sys
have = [line.split()[1] for line in open("/etc/resolv.conf") if line.split()[:1] == ["nameserver"]]
if have != sys.argv[2:]:
    print("resolv.conf names " + " ".join(have))
else:
    try:
        socket.getaddrinfo(sys.argv[1], 443, socket.AF_INET)
        print("answers")
    except OSError as e:
        print(type(e).__name__ + ": " + str(e))
"""


async def _dns_answers_through_the_mounted_resolv_conf(box: VmSandbox, plan: ProbePlan) -> str:
    resolvers = [str(r) for r in upstream_resolvers(plan.facts.resolv_conf)]
    return await _answer(box, "dns", plan.inside.exec_in_prober("python", "-c", _DNS, _PUBLIC_HOST, *resolvers))


async def _internet_is_reachable(box: VmSandbox, plan: ProbePlan) -> str:
    return await _answer(box, "internet", plan.inside.exec_in_prober("python", "-c", _GET, _CONNECTIVITY_URL, "30"))


_DOOR = """
import http.client, sys, time
host, port, header, token = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]

def status(target, value):
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request("GET", target, headers={} if value is None else {header: value})
        return connection.getresponse().status
    finally:
        connection.close()

deadline = time.monotonic() + 30
while True:
    try:
        status("/", None)
        break
    except OSError:
        if time.monotonic() > deadline:
            raise
        time.sleep(1)
print(status("/", None), status("/", "wrong-" + token), status("/", token), status("//elsewhere/x", token))
"""


async def _door_takes_only_its_token(box: VmSandbox, plan: ProbePlan) -> str:
    slot = plan.other
    argv = ["python3", "-c", _DOOR, plan.facts.address, str(slot.door_port), SLOT_TOKEN_HEADER, slot.door_token]
    return await _answer(box, "door", shlex.join(argv))


async def _probe_slots_tear_down(box: VmSandbox, plan: ProbePlan) -> str:
    for slot in plan.slots:
        await tear_down_slot(box, slot.layout)
    return "torn down"


PROBES: tuple[Probe, ...] = (
    Probe("docker_info", frozenset({"ok"}), 30, _docker_runs_gvisor_in_a_user_namespace),
    Probe("firewall", frozenset({"ok"}), 60, _firewall_holds_the_walls),
    Probe("probe_slots_up", frozenset({"up"}), 180, _probe_slots_come_up),
    Probe("slots_under_gvisor", frozenset({"gvisor"}), 30, _slots_run_under_gvisor),
    Probe("slot_0_peer_by_name", frozenset({"200"}), 45, _peer_by_name(0)),
    Probe("slot_1_peer_by_name", frozenset({"200"}), 45, _peer_by_name(1)),
    _blocked_from_inside("a_to_b_container_ip", lambda plan: [
        f"{plan.other.peer_address}:{_PEER_PORT}", f"{plan.other.door_address}:{plan.other.door_port}"]),
    _blocked_from_inside("a_to_b_door_via_box_ip", lambda plan: [f"{plan.facts.address}:{plan.other.door_port}"]),
    _blocked_from_inside("a_to_box_ip", lambda plan: [f"{plan.facts.address}:{port}" for port in (22, 80, 2375)]),
    _blocked_from_inside("a_to_private_ranges", _private_range_targets),
    _blocked_from_inside("a_ipv6", lambda plan: list(_PUBLIC_IPV6)),
    Probe("dns", frozenset({"answers"}), 30, _dns_answers_through_the_mounted_resolv_conf, needs_open_egress=True),
    Probe("internet", frozenset({"204"}), 45, _internet_is_reachable, needs_open_egress=True),
    Probe("door", frozenset({"401 401 200 400"}), 45, _door_takes_only_its_token),
    Probe("probe_slots_torn_down", frozenset({"torn down"}), 60, _probe_slots_tear_down),
)
