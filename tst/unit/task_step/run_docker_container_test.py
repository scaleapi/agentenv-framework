"""Unit tests for RunDockerContainerTaskStep's keep_alive_with_base_command flag."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps.run_docker_container import (
    RunDockerContainerTaskStep as Step,
)


def _step(**kw):
    base = dict(
        id="s", version=None, sandbox_name="h",
        docker_context_artifact_id="u", docker_context_artifact_version=1,
        container_name="c",
    )
    base.update(kw)
    return Step(**base)


def test_keep_alive_and_command_override_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        _step(command_override="sleep infinity", keep_alive_with_base_command=True)


def test_keep_alive_roundtrips():
    step = _step(keep_alive_with_base_command=True)
    restored = Step.from_dict(step.to_dict())
    assert restored.keep_alive_with_base_command is True


def test_default_keep_alive_false():
    assert _step().keep_alive_with_base_command is False


@pytest.mark.parametrize("entrypoint,cmd,expected", [
    (["/bin/sh", "-c"], ["postgres -D /data"], ["/bin/sh", "-c", "postgres -D /data"]),
    (None, ["node", "server.js"], ["node", "server.js"]),
    (["/entry.sh"], None, ["/entry.sh"]),
    (None, None, []),
])
@pytest.mark.asyncio
async def test_inspect_base_command_combines_entrypoint_and_cmd(entrypoint, cmd, expected):
    import json

    class _FakeSandbox:
        async def exec_script(self, _script):
            ep = json.dumps(entrypoint) if entrypoint is not None else "null"
            cm = json.dumps(cmd) if cmd is not None else "null"
            return f"{ep}|{cm}\n"

    got = await _step()._inspect_base_command(_FakeSandbox(), "img:latest")
    assert got == expected


def test_network_and_ready_command_roundtrip():
    step = Step(
        id="s", version=None, sandbox_name="vm", docker_context_artifact_id="ctx",
        container_name="pg", network="task-net", ready_command="pg_isready -U agentenv",
    )
    restored = Step.from_dict(step.to_dict())
    assert restored.network == "task-net"
    assert restored.ready_command == "pg_isready -U agentenv"


def test_network_defaults_none():
    step = Step(
        id="s", version=None, sandbox_name="vm", docker_context_artifact_id="ctx",
    )
    assert step.network is None and step.ready_command is None
    assert Step.from_dict(step.to_dict()).network is None


def test_extra_networks_roundtrip():
    step = _step(network="dmz", extra_networks=["corp", "restricted"])
    restored = Step.from_dict(step.to_dict())
    assert restored.network == "dmz"
    assert restored.extra_networks == ["corp", "restricted"]


def test_extra_networks_defaults_empty():
    step = _step()
    assert step.extra_networks == []
    # None is normalized to an empty list on the way in and back out
    assert Step.from_dict(_step(extra_networks=None).to_dict()).extra_networks == []


def test_extra_networks_rejects_netns_modes():
    # host / none / container:<name> can't be attached with `docker network connect`
    for bad in ("host", "none", "container:estate"):
        with pytest.raises(ValueError, match="extra_networks"):
            _step(extra_networks=[bad])


def test_extra_networks_allows_bridge_and_user_bridges():
    step = _step(extra_networks=["dmz", "bridge"])
    assert step.extra_networks == ["dmz", "bridge"]


def test_device_and_cap_fields_roundtrip():
    step = _step(devices=["/dev/kvm", "/dev/net/tun"], cap_add=["NET_ADMIN"],
                 privileged=True, volumes=["/data:/storage"], shm_size="2g")
    restored = Step.from_dict(step.to_dict())
    assert restored.devices == ["/dev/kvm", "/dev/net/tun"]
    assert restored.cap_add == ["NET_ADMIN"]
    assert restored.privileged is True
    assert restored.volumes == ["/data:/storage"]
    assert restored.shm_size == "2g"


def test_device_fields_default_empty():
    step = _step()
    assert step.devices == [] and step.cap_add == [] and step.volumes == []
    assert step.privileged is False and step.shm_size is None
    # None normalizes to empty lists through a to_dict/from_dict roundtrip
    restored = Step.from_dict(_step(devices=None, cap_add=None, volumes=None).to_dict())
    assert restored.devices == [] and restored.cap_add == [] and restored.volumes == []
