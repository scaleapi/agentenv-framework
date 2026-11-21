"""A shared box is returned from ``create_shared_box`` with its walls verified only when every probe of its self-test
answers as expected, its probe slots are torn down, and only then; any other answer, error or timeout ends the box
through ``terminate_shared_box`` and raises, naming the probe. The probes run here against a fake box that scripts
each probe's answer, and their scripts' own logic runs here against a real door and real sockets."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import socket
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import uvicorn
from pytest_socket import enable_socket

from agent_env.env import gateway as gateway_package
from agent_env.env.gateway.constants import SLOT_TOKEN_HEADER
from agent_env.env.gateway.door import Door
from agent_env.providers.env_providers.stack_layout import StackLayout
from agent_env.providers.sandbox_providers import modal_vm_sandbox, shared_box_selftest
from agent_env.providers.sandbox_providers.modal_sandbox import WALLS_TAG, WALLS_VERSION
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox, ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.shared_box_selftest import PROBE_MARKER, PROBES, WallsNotVerified
from tst.unit.providers.sandbox_providers.modal_shared_box_test import BOX, _offline, modal_api, modal_sb, provider

__all__ = ["_offline", "modal_api"]

TEARDOWN = "probe_slots_torn_down"
TEARDOWNS = {StackLayout.for_slot(i).teardown_script for i in (0, 1)}
PASSING = {
    "box_facts": json.dumps({"address": "10.0.0.5", "kernel": "6.1.0-modal",
                             "resolv_conf": "nameserver 172.21.0.1\n"}),
    "docker_info": "ok",
    "firewall": "ok",
    "probe_slots_up": "up",
    "slots_under_gvisor": "gvisor",
    "slot_0_peer_by_name": "200",
    "slot_1_peer_by_name": "200",
    "a_to_b_container_ip": "blocked",
    "a_to_b_door_via_box_ip": "blocked",
    "a_to_box_ip": "blocked",
    "a_to_private_ranges": "blocked",
    "a_ipv6": "blocked",
    "dns": "answers",
    "internet": "204",
    "door": "401 401 200 400",
}
EVERY_PROBE = ["box_facts", *(probe.name for probe in PROBES)]


class FakeBox:
    """The execs of a shared box, each probe's script answered from ``answers`` by the probe it names, a teardown as
    the probe ``probe_slots_torn_down``, and every other exec, a file written, with success. An answer is the probe's
    printed line, an exit code and stderr, an exception to raise, or ``hang`` to never return. ``events`` is each probe
    run, in order, and ``tag`` once the box's tags are set."""

    def __init__(self, **answers):
        self.answers = {**PASSING, **answers}
        self.events: list[str] = []

    async def exec_with_output(self, box, *args: str) -> tuple[int, str, str]:
        name = None
        if args[:-1] in (("bash", "-c"), ("sudo", "bash", "-c")):
            script = args[-1]
            if script.startswith(PROBE_MARKER):
                name = script[len(PROBE_MARKER):].split("\n", 1)[0]
            elif script in TEARDOWNS:
                name = TEARDOWN
        if name is None:
            return 0, "", ""
        self.events.append(name)
        answer = self.answers.get(name, "")
        if answer == "hang":
            await asyncio.Event().wait()
        if isinstance(answer, BaseException):
            raise answer
        return answer if isinstance(answer, tuple) else (0, f"{answer}\n", "")


@pytest.fixture
def box_api(modal_api, monkeypatch):
    """A Modal VM whose dockerd starts, whose walls apply, and whose execs ``box_api.box`` answers; its set_tags is
    recorded among that box's events."""
    monkeypatch.setattr(ModalVmSandbox, "_ensure_dockerd_running", AsyncMock())
    monkeypatch.setattr(modal_vm_sandbox, "apply_walls_firewall", AsyncMock())
    modal_api.sb = modal_sb(BOX)
    modal_api.box = FakeBox()
    monkeypatch.setattr(ModalVmSandbox, "exec_with_output",
                        lambda self, *args: modal_api.box.exec_with_output(self, *args))
    modal_api.sb.set_tags.aio.side_effect = lambda tags: modal_api.box.events.append("tag")
    return modal_api


def spied_provider():
    p = provider(ModalVmSandboxProvider)
    p.terminate_shared_box = AsyncMock(wraps=p.terminate_shared_box)
    return p


@pytest.mark.asyncio
async def test_a_box_whose_probes_all_pass_is_tagged_verified_once_after_its_probe_slots_are_gone(box_api):
    p = spied_provider()

    box = await p.create_shared_box("box-1")

    assert box.walls_verified is True
    box_api.sb.set_tags.aio.assert_awaited_once_with({**BOX, WALLS_TAG: WALLS_VERSION})
    assert box_api.box.events == [*EVERY_PROBE[:-1], TEARDOWN, TEARDOWN, "tag"]
    p.terminate_shared_box.assert_not_called()


ANSWERS = {"unexpected answer": "something else", "script fails": (1, "", "boom"),
           "exec raises": ConnectionError("exec dropped")}


@pytest.mark.parametrize("probe, answer", [
    pytest.param(probe, answer, id=f"{probe}, {how}") for probe in EVERY_PROBE for how, answer in ANSWERS.items()
    # A teardown prints nothing it is judged by: only its failing counts.
    if not (probe == TEARDOWN and how == "unexpected answer")])
@pytest.mark.asyncio
async def test_a_box_whose_probe_answers_otherwise_is_ended_untagged_and_the_error_names_the_probe(
        box_api, probe, answer):
    box_api.box = FakeBox(**{probe: answer})
    p = spied_provider()

    with pytest.raises(WallsNotVerified, match=rf"probe {probe}:") as raised:
        await p.create_shared_box("box-1")

    box_api.sb.set_tags.aio.assert_not_called()
    p.terminate_shared_box.assert_awaited_once_with("sb-1", "box-1")
    box_api.sb.terminate.aio.assert_awaited_once()
    assert [r.name for r in raised.value.results] == EVERY_PROBE[:EVERY_PROBE.index(probe) + 1]
    assert [r.passed for r in raised.value.results] == [*([True] * EVERY_PROBE.index(probe)), False]


@pytest.mark.parametrize("probe", ["firewall", "door"])
@pytest.mark.parametrize("limit", ["probe", "whole run"])
@pytest.mark.asyncio
async def test_a_probe_that_doesnt_answer_in_time_leaves_the_box_untagged_and_ends_it(
        box_api, monkeypatch, probe, limit):
    box_api.box = FakeBox(**{probe: "hang"})
    if limit == "probe":
        monkeypatch.setattr(shared_box_selftest, "PROBES", tuple(
            dataclasses.replace(p, timeout_s=0.05) if p.name == probe else p for p in PROBES))
    else:
        monkeypatch.setattr(shared_box_selftest, "TOTAL_BUDGET_S", 0.2)
    p = spied_provider()

    with pytest.raises(WallsNotVerified, match=rf"probe {probe}: .*no answer within"):
        await p.create_shared_box("box-1")

    box_api.sb.set_tags.aio.assert_not_called()
    p.terminate_shared_box.assert_awaited_once_with("sb-1", "box-1")


@pytest.mark.asyncio
async def test_a_box_whose_egress_is_allowlisted_is_verified_without_the_internet_probes(box_api):
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",))

    box = await spied_provider().create_shared_box("box-1", network_policy=policy)

    assert box.walls_verified is True
    assert box_api.box.events == [e for e in EVERY_PROBE if e not in ("dns", "internet")][:-1] + [TEARDOWN] * 2 + ["tag"]


@pytest.mark.asyncio
async def test_a_box_not_set_up_for_a_gateway_runs_no_probe_and_stays_unverified(box_api):
    box = await spied_provider().create_shared_box("box-1", setup_for_gateway=False)

    assert box.walls_verified is False
    assert box_api.box.events == []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _python(code: str, *argv: str) -> str:
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", code, *argv,
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode()
    return stdout.decode().strip()


@pytest.mark.asyncio
async def test_the_door_probe_answers_as_expected_against_a_real_door():
    enable_socket()
    upstream_port, door_port = _free_port(), _free_port()
    upstream = await asyncio.create_subprocess_exec(sys.executable, "-m", "http.server", "--bind", "127.0.0.1",
                                                    str(upstream_port), stdout=asyncio.subprocess.DEVNULL,
                                                    stderr=asyncio.subprocess.DEVNULL)
    async with httpx.AsyncClient(trust_env=False) as client:
        door = uvicorn.Server(uvicorn.Config(Door(f"127.0.0.1:{upstream_port}", "the-token", client),
                                             host="127.0.0.1", port=door_port, lifespan="off", log_level="warning"))
        serving = asyncio.ensure_future(door.serve())
        try:
            answer = await _python(shared_box_selftest._DOOR, "127.0.0.1", str(door_port), SLOT_TOKEN_HEADER,
                                   "the-token")
        finally:
            door.should_exit = True
            await serving
            upstream.terminate()
            await upstream.wait()

    assert answer in next(probe.expected for probe in PROBES if probe.name == "door")


@pytest.mark.asyncio
async def test_the_reach_probe_counts_only_a_connection_that_opens_or_is_refused_as_reached():
    enable_socket()
    with socket.socket() as listening:
        listening.bind(("127.0.0.1", 0))
        listening.listen()
        open_port, closed_port = listening.getsockname()[1], _free_port()

        reached = await _python(shared_box_selftest._REACH, f"127.0.0.1:{open_port}", f"127.0.0.1:{closed_port}")
        unreachable = await _python(shared_box_selftest._REACH, "999.0.0.1:80")

    assert reached == f"reached 127.0.0.1:{open_port} open, 127.0.0.1:{closed_port} refused"
    assert unreachable == "blocked"


@pytest.mark.parametrize("info, nnp, answer", [
    pytest.param({"DefaultRuntime": "runsc", "SecurityOptions": ["name=seccomp", "name=userns",
                                                                 "name=no-new-privileges"]}, "1", "ok", id="all set"),
    pytest.param({"DefaultRuntime": "runsc", "SecurityOptions": ["name=userns"]}, "0", "ok", id="nnp not asked for"),
    pytest.param({"DefaultRuntime": "runc", "SecurityOptions": ["name=userns"]}, "0", "the default runtime is runc",
                 id="runc"),
    pytest.param({"DefaultRuntime": "runsc", "SecurityOptions": []}, "0", "no userns in []", id="no userns"),
    pytest.param({"DefaultRuntime": "runsc", "SecurityOptions": ["name=userns"]}, "1",
                 "no no-new-privileges in ['name=userns']", id="nnp missing"),
])
@pytest.mark.asyncio
async def test_the_docker_probe_passes_only_gvisor_by_default_in_a_user_namespace(info, nnp, answer):
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", shared_box_selftest._DOCKER_INFO, "runsc", nnp,
                                                   stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await process.communicate(json.dumps(info).encode())

    assert stdout.decode().strip() == answer


def test_every_probe_container_runs_the_base_the_gateway_image_is_built_on():
    dockerfile = Path(gateway_package.__file__).parent / "Dockerfile"

    assert dockerfile.read_text().splitlines()[0] == f"FROM {shared_box_selftest.PROBE_IMAGE}"
