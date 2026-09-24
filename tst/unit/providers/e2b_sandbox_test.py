"""Offline unit tests for the E2B VmSandbox adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.providers.e2b.sandbox import E2BSandbox
from agent_env.providers.sandbox import NetworkMode, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_provider import SANDBOX_MODE_VM


def _inner() -> MagicMock:
    sandbox = MagicMock()
    sandbox.sandbox_id = "e2b-test"
    sandbox.get_host.side_effect = lambda port: f"{port}-e2b-test.e2b.app"
    sandbox.kill = AsyncMock(return_value=True)
    sandbox.commands.run = AsyncMock(
        return_value=SimpleNamespace(exit_code=0, stdout="hello", stderr="warning")
    )
    return sandbox


def test_vm_identity_and_lazy_tunnel_urls():
    inner = _inner()
    sandbox = E2BSandbox(inner, exposed_ports=[18765])

    assert isinstance(sandbox, VmSandbox)
    assert sandbox.type == "e2b"
    assert sandbox.mode == SANDBOX_MODE_VM
    assert sandbox.sandbox_id == "e2b-test"
    assert sandbox.vnc_url is None
    assert sandbox.tunnel_urls[18765] == "https://18765-e2b-test.e2b.app"
    # A reconnect has no fixed port list; normal tunnel_urls indexing still
    # obtains and caches an E2B host dynamically.
    assert sandbox.tunnel_urls[8080] == "https://8080-e2b-test.e2b.app"
    assert inner.get_host.call_args_list[-1].args == (8080,)


def test_tunnel_urls_get_is_cache_only_but_indexing_is_lazy():
    inner = _inner()
    sandbox = E2BSandbox(inner)

    assert sandbox.tunnel_urls.get(8080) is None
    inner.get_host.assert_not_called()

    assert sandbox.tunnel_urls[8080] == "https://8080-e2b-test.e2b.app"
    assert sandbox.tunnel_urls.get(8080) == "https://8080-e2b-test.e2b.app"
    inner.get_host.assert_called_once_with(8080)


@pytest.mark.asyncio
async def test_exec_adapts_completed_command_result_and_preserves_sudo():
    inner = _inner()
    sandbox = E2BSandbox(inner)

    exit_code, stdout, stderr = await sandbox.exec_with_output(
        "sudo", "bash", "-c", "docker images && echo 'hello world'"
    )

    assert (exit_code, stdout, stderr) == (0, "hello", "warning")
    # E2B takes one shell command, and sudo must survive for Docker commands.
    assert inner.commands.run.await_args.args == (
        """sudo bash -c 'docker images && echo '"'"'hello world'"'"''""",
    )
    assert inner.commands.run.await_args.kwargs == {"timeout": 0}


@pytest.mark.asyncio
async def test_exec_converts_e2b_nonzero_command_exception_to_process_result():
    inner = _inner()
    inner.commands.run.side_effect = _CommandExit(7, "out", "bad")
    sandbox = E2BSandbox(inner)

    assert await sandbox.exec_with_output("sudo", "docker", "images") == (7, "out", "bad")
    assert inner.commands.run.await_args.args == ("sudo docker images",)


@pytest.mark.asyncio
async def test_setup_starts_docker_polls_until_ready_and_does_not_open_firewall(
    monkeypatch: pytest.MonkeyPatch,
):
    sandbox = E2BSandbox(_inner())
    sandbox._run_command = AsyncMock(
        return_value=SimpleNamespace(wait=AsyncMock(return_value=0)),
    )
    sandbox._exec_with_output_bounded = AsyncMock(
        side_effect=[
            (1, "", "not ready"),
            (0, "", ""),
            (0, "Docker Compose version v5.5.1", ""),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("agent_env.providers.e2b.sandbox.asyncio.sleep", sleep)

    await sandbox.setup_vm_for_gateway([18765])

    launch = sandbox._run_command.await_args.args[-1]
    assert "sudo systemctl start docker" in launch
    assert "sudo service docker start" in launch
    assert "sudo nohup dockerd" in launch
    assert "iptables" not in launch
    assert sandbox._exec_with_output_bounded.await_args_list == [
        (
            ("sudo", "docker", "info"),
            {"timeout": sandbox._DOCKER_READINESS_COMMAND_TIMEOUT},
        ),
        (
            ("sudo", "docker", "info"),
            {"timeout": sandbox._DOCKER_READINESS_COMMAND_TIMEOUT},
        ),
        (
            ("sudo", "docker", "compose", "version"),
            {"timeout": sandbox._DOCKER_READINESS_COMMAND_TIMEOUT},
        ),
    ]
    sleep.assert_awaited_once_with(sandbox._VM_READY_POLL_INTERVAL)


@pytest.mark.asyncio
async def test_setup_rejects_base_template_without_compose_v2():
    sandbox = E2BSandbox(_inner())
    sandbox._run_command = AsyncMock(
        return_value=SimpleNamespace(wait=AsyncMock(return_value=0)),
    )
    sandbox._exec_with_output_bounded = AsyncMock(
        side_effect=[
            (0, "", ""),
            (1, "", "docker: 'compose' is not a docker command"),
        ],
    )

    with pytest.raises(RuntimeError, match="must include the Docker Compose v2 plugin"):
        await sandbox.setup_vm_for_gateway([])


@pytest.mark.asyncio
async def test_docker_readiness_timeout_includes_daemon_log_tail():
    sandbox = E2BSandbox(_inner())
    sandbox._VM_READY_TIMEOUT = 0
    sandbox._run_command = AsyncMock(
        return_value=SimpleNamespace(wait=AsyncMock(return_value=0)),
    )
    sandbox._exec_with_output_bounded = AsyncMock(return_value=(0, "dockerd failure", ""))

    with pytest.raises(RuntimeError, match=r"(?s)Docker not ready.*dockerd failure"):
        await sandbox.wait_for_vm()

    assert "sudo systemctl start docker" in sandbox._run_command.await_args.args[-1]
    assert sandbox._exec_with_output_bounded.await_args.args == (
        "sudo",
        "tail",
        "-n",
        str(sandbox._DOCKER_LOG_TAIL_LINES),
        "/var/log/dockerd.log",
    )


@pytest.mark.asyncio
async def test_docker_readiness_probe_passes_a_finite_sdk_command_timeout():
    inner = _inner()
    sandbox = E2BSandbox(inner)
    sandbox._VM_READY_TIMEOUT = 1
    sandbox._DOCKER_READINESS_COMMAND_TIMEOUT = 0.25
    inner.commands.run.side_effect = [
        SimpleNamespace(exit_code=0, stdout="", stderr=""),
        TimeoutError("docker info timed out"),
        SimpleNamespace(exit_code=0, stdout="dockerd log", stderr=""),
    ]

    with pytest.raises(RuntimeError, match="Docker not ready"):
        await sandbox.wait_for_vm()

    assert inner.commands.run.await_args_list[1].args == ("sudo docker info",)
    assert inner.commands.run.await_args_list[1].kwargs == {"timeout": 0.25}
    assert inner.commands.run.await_args_list[-1].args == (
        "sudo tail -n 40 /var/log/dockerd.log",
    )
    assert inner.commands.run.await_args_list[-1].kwargs == {"timeout": 0.25}


@pytest.mark.asyncio
async def test_terminate_kills_the_sdk_sandbox():
    inner = _inner()
    sandbox = E2BSandbox(inner)

    await sandbox.terminate()

    inner.kill.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_reconnect_passes_explicit_key_and_rebuilds_dynamic_tunnels():
    reconnected = _inner()
    sandbox_cls = MagicMock()
    sandbox_cls.connect = AsyncMock(return_value=reconnected)

    sandbox = await E2BSandbox.reconnect(
        "e2b-reconnect", api_key="configured-key", sandbox_cls=sandbox_cls,
    )

    sandbox_cls.connect.assert_awaited_once_with("e2b-reconnect", api_key="configured-key")
    assert sandbox.tunnel_urls[3000] == "https://3000-e2b-test.e2b.app"


@pytest.mark.asyncio
async def test_restricted_image_loading_adds_signed_host_to_applied_policy(
    monkeypatch: pytest.MonkeyPatch,
):
    inner = _inner()
    inner.update_network = AsyncMock()
    policy = NetworkPolicy(
        mode=NetworkMode.ALLOWLIST,
        allow_hosts=("workload.example",),
        allow_cidrs=("10.0.0.0/8",),
    )
    sandbox = E2BSandbox(inner, network_policy=policy)
    artifact = SimpleNamespace(tar_gz_object_url="s3://bucket/image.tar.gz")
    store = MagicMock()
    store.signed_get_url.return_value = "https://downloads.example/path?signature=x"
    monkeypatch.setattr(
        "agent_env.config.get_config",
        lambda: SimpleNamespace(get_object_store=lambda: store),
    )
    base_load = AsyncMock()
    monkeypatch.setattr(VmSandbox, "load_docker_images", base_load)

    await sandbox.load_docker_images([artifact])

    inner.update_network.assert_awaited_once_with(
        {
            "allow_internet_access": True,
            "allow_out": ["workload.example", "downloads.example", "10.0.0.0/8"],
            "deny_out": ["0.0.0.0/0"],
        }
    )
    assert sandbox.network_policy == policy.with_hosts(["downloads.example"])
    base_load.assert_awaited_once_with([artifact])


@pytest.mark.asyncio
async def test_restricted_image_loading_retains_extended_policy_after_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    inner = _inner()
    inner.update_network = AsyncMock()
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("workload.example",))
    sandbox = E2BSandbox(inner, network_policy=policy)
    store = MagicMock()
    store.signed_get_url.return_value = "https://downloads.example/image.tar.gz"
    monkeypatch.setattr(
        "agent_env.config.get_config",
        lambda: SimpleNamespace(get_object_store=lambda: store),
    )

    async def fail_load(_self, _artifacts):
        raise RuntimeError("download failed")

    monkeypatch.setattr(VmSandbox, "load_docker_images", fail_load)

    with pytest.raises(RuntimeError, match="download failed"):
        await sandbox.load_docker_images([SimpleNamespace(tar_gz_object_url="s3://bucket/img")])

    inner.update_network.assert_awaited_once_with(
        {
            "allow_internet_access": True,
            "allow_out": ["workload.example", "downloads.example"],
            "deny_out": ["0.0.0.0/0"],
        }
    )
    assert sandbox.network_policy == policy.with_hosts(["downloads.example"])


@pytest.mark.asyncio
async def test_reconnected_sandbox_with_unknown_policy_refuses_image_load():
    sandbox = E2BSandbox(_inner(), network_policy=None)
    artifact = SimpleNamespace(tar_gz_object_url="s3://bucket/image.tar.gz")

    with pytest.raises(RuntimeError, match="applied network policy is unknown"):
        await sandbox.load_docker_images([artifact])


@pytest.mark.asyncio
async def test_allow_all_network_policy_explicitly_enables_internet_without_ingress_field():
    inner = _inner()
    inner.update_network = AsyncMock()
    sandbox = E2BSandbox(inner)

    await sandbox.apply_network_policy(NetworkPolicy(mode=NetworkMode.ALLOW_ALL))

    inner.update_network.assert_awaited_once_with({"allow_internet_access": True})


class _CommandExit(Exception):
    def __init__(self, exit_code: int, stdout: str, stderr: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
