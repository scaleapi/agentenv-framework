"""Offline unit tests for the CoreWeave sandboxes: exec, file transfer, stdin transfer, host setup and egress."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import httpx
import pytest
from cwsandbox import (
    CidrBlock,
    Container,
    EgressRule,
    SandboxError,
    SandboxExecutionError,
    SandboxNotRunningError,
    SandboxRequestTimeoutError,
    SandboxStreamTruncatedError,
    SandboxTimeoutError,
    SandboxUnavailableError,
    TenantScope,
)

from agent_env.providers.sandbox_providers.coreweave import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.coreweave.sandbox import (
    Client,
    CoreWeaveContainerSandbox,
    CoreWeaveSandbox,
    allows_download,
    host_setup_script,
    network_options,
    policy_from_egress,
    runs_docker_host,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, stage_files_into_container
from tst.unit.providers.sandbox_providers.coreweave_fakes import FakeProcess, Result, fake_sandbox_cls

ALLOWLIST = NetworkMode.ALLOWLIST


def _sandbox(*, policy=NetworkPolicy()) -> tuple[CoreWeaveSandbox, object]:
    cls = fake_sandbox_cls()
    raw = cls()
    raw.sandbox_id = "sb-1"
    cls.server.sandboxes["sb-1"] = raw
    client = Client(sandbox_cls=cls, auth="auth", base_url="https://api.test")
    return CoreWeaveSandbox(raw, client=client, tunnel_urls={}, network_policy=policy, command_timeout=600), raw


def _container() -> tuple[CoreWeaveContainerSandbox, object]:
    cls = fake_sandbox_cls()
    raw = cls(containers=[Container(image="registry.example/agent:1")])
    raw.sandbox_id = "sb-1"
    cls.server.sandboxes["sb-1"] = raw
    client = Client(sandbox_cls=cls, auth="auth", base_url="https://api.test")
    return CoreWeaveContainerSandbox(raw, client=client, tunnel_urls={}, network_policy=None, command_timeout=600), raw


async def _no_sleep(_seconds):
    return None


@pytest.mark.asyncio
async def test_exec_runs_with_the_lifetime_budget_and_drops_sudo():
    sandbox, raw = _sandbox()
    raw.results = [Result(stdout_bytes=b"hi\n")]
    assert await sandbox.exec_with_output("sudo", "echo", "hi") == (0, "hi\n", "")
    assert raw.exec_calls == [{"argv": ["echo", "hi"], "timeout": 600, "stdin": False}]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    SandboxUnavailableError("unavailable"),
    SandboxStreamTruncatedError("output lost in transit"),
    SandboxRequestTimeoutError("request timed out"),
    SandboxExecutionError("Command cat /big ended without an exit status from the sandbox; it may not have finished"),
])
async def test_a_lost_or_cut_stream_is_exit_minus_one_with_no_stdout(error):
    sandbox, raw = _sandbox()
    raw.results = [error]
    exit_code, stdout, stderr = await sandbox.exec_with_output("cat", "/big")
    assert (exit_code, stdout) == (-1, "")
    assert type(error).__name__ in stderr


@pytest.mark.asyncio
async def test_a_timed_out_command_is_exit_124():
    sandbox, raw = _sandbox()
    raw.results = [SandboxTimeoutError("deadline exceeded")]
    exit_code, _, stderr = await sandbox.exec_with_output("sleep", "9999")
    assert exit_code == 124 and "timed out after 600s" in stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    SandboxNotRunningError("gone"),
    SandboxError("rejected"),
    SandboxExecutionError("Command true failed with exit code 1", exec_result=object()),
])
async def test_other_sdk_errors_propagate(error):
    sandbox, raw = _sandbox()
    raw.results = [error]
    with pytest.raises(type(error)):
        await sandbox.exec_with_output("true")


@pytest.mark.asyncio
async def test_exec_script_retries_a_lost_stream(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    sandbox, raw = _sandbox()
    raw.results = [SandboxUnavailableError("unavailable"), Result(stdout_bytes=b"ok")]
    assert await sandbox.exec_script("echo ok", max_retries=1) == "ok"
    assert [call["argv"] for call in raw.exec_calls] == [["bash", "-c", "echo ok"]] * 2


@pytest.mark.asyncio
async def test_exec_with_stdin_streams_every_piece_then_closes():
    sandbox, raw = _sandbox()

    async def pieces():
        yield b"a"
        yield b"b"

    raw.results = [Result(stdout_bytes=b"digest  -\n")]
    assert await sandbox._exec_with_stdin("sha256sum", pieces()) == (0, "digest  -\n", "")
    process = raw.processes[0]
    assert raw.exec_calls[0] == {"argv": ["bash", "-c", "sha256sum"], "timeout": 600, "stdin": True}
    assert (process.stdin.pieces, process.stdin.closed) == ([b"a", b"b"], True)


@pytest.mark.asyncio
async def test_stdin_stops_once_the_command_ended_rather_than_wait_on_a_full_queue():
    """A command that exits before reading all its input leaves the SDK's bounded queue full; the next write would
    wait forever."""
    sandbox, raw = _sandbox()
    raw.results = [FakeProcess(Result(returncode=1, stderr_bytes=b"disk full"), stdin=True, read_limit=1)]
    read = []

    async def pieces():
        for piece in (b"a", b"b", b"c"):
            read.append(piece)
            yield piece

    process = raw.results[0]
    asyncio.get_running_loop().call_later(0.05, process.finish)
    assert await asyncio.wait_for(sandbox._exec_with_stdin("cat > /x", pieces()), 5) == (1, "", "disk full")
    assert read == [b"a", b"b"]


@pytest.mark.asyncio
async def test_a_failing_reader_closes_stdin_and_raises_its_error():
    sandbox, raw = _sandbox()

    async def pieces():
        yield b"a"
        raise RuntimeError("the object changed during the push")

    with pytest.raises(RuntimeError, match="changed during the push"):
        await sandbox._exec_with_stdin("cat > /x", pieces())
    assert raw.processes[0].stdin.closed


@pytest.mark.asyncio
async def test_write_host_file_goes_over_stdin_to_any_path():
    sandbox, raw = _sandbox()
    await sandbox.write_host_file(b"payload", "/opt/x/file.bin")
    assert raw.exec_calls[-1]["argv"] == ["bash", "-c", "cat > /opt/x/file.bin"]
    assert raw.processes[-1].stdin.pieces == [b"payload"]


@pytest.mark.asyncio
async def test_terminate_deletes_by_id_and_tolerates_a_gone_sandbox():
    sandbox, raw = _sandbox()
    await sandbox.terminate()
    await sandbox.terminate()
    assert raw.server.deleted == ["sb-1", "sb-1"]
    assert {call[1]["missing_ok"] for call in raw.server.calls if call[0] == "delete"} == {True}


@pytest.mark.asyncio
async def test_wait_for_vm_waits_with_sh_since_bash_may_not_be_installed_yet():
    sandbox, raw = _sandbox()
    await sandbox.wait_for_vm()
    assert raw.exec_calls[0]["argv"][:2] == ["sh", "-c"]


@pytest.mark.asyncio
async def test_wait_for_vm_reports_what_the_host_setup_logged():
    sandbox, raw = _sandbox()
    raw.results = [Result(returncode=1, stderr_bytes=b"the image lacks bash curl e2fsprogs")]
    with pytest.raises(RuntimeError, match="Docker not ready in CoreWeave sandbox sb-1: the image lacks bash"):
        await sandbox.wait_for_vm()


@pytest.mark.asyncio
async def test_wait_for_vm_rides_out_a_lost_stream():
    sandbox, raw = _sandbox()
    raw.results = [SandboxUnavailableError("unavailable"), Result()]
    await sandbox.wait_for_vm()
    assert len(raw.exec_calls) == 2


@pytest.mark.parametrize(("storage", "present", "absent"), [
    ("loop", ["apk add --no-cache bash curl e2fsprogs", "mount -o loop", "truncate -s 30G"], ["tmpfs"]),
    ("memory", ["apk add --no-cache bash curl", "mount -t tmpfs -o size=30g"], ["e2fsprogs", "mount -o loop"]),
])
def test_host_setup_mounts_dockers_storage_off_the_virtiofs_root(storage, present, absent):
    script = host_setup_script(storage, 30)
    assert all(text in script for text in present)
    assert not any(text in script for text in absent)
    assert "export DOCKER_HOST=unix:///var/run/docker.sock" in script


def test_network_options_for_allow_all_leaves_the_runner_default():
    assert network_options(NetworkPolicy()) is None


def test_network_options_for_an_empty_allowlist_denies_all_egress():
    assert network_options(NetworkPolicy(mode=ALLOWLIST)).deny_egress is True


def test_network_options_split_hostnames_from_addresses_and_ranges():
    options = network_options(NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org", "*.vercel.run", "10.1.2.3"),
                                            allow_cidrs=("1.1.1.0/24",)))
    assert [(rule.dns_name, rule.cidr.cidr if rule.cidr else None) for rule in options.egress] == [
        ("pypi.org", None), ("*.vercel.run", None), (None, "10.1.2.3/32"), (None, "1.1.1.0/24"),
    ]


@pytest.mark.parametrize("policy", [
    NetworkPolicy(mode=ALLOWLIST, allow_hosts=("not a host",)),
    NetworkPolicy(mode=ALLOWLIST, allow_hosts=("a.*.example.com",)),
    NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("not-a-cidr",)),
])
def test_network_options_refuse_an_entry_coreweave_cannot_take(policy):
    with pytest.raises(ValueError):
        network_options(policy)


@pytest.mark.parametrize(("rules", "policy"), [
    ((EgressRule(any=True),), NetworkPolicy()),
    ((EgressRule(cidr=CidrBlock(cidr="0.0.0.0/0", except_cidrs=("10.0.0.0/8",))),), NetworkPolicy()),
    ((), NetworkPolicy(mode=ALLOWLIST)),
    ((EgressRule(dns_name="pypi.org"), EgressRule(cidr="1.1.1.0/24")),
     NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("1.1.1.0/24",))),
    ((EgressRule(tenant=TenantScope.SAME_ORG),), None),
    ((EgressRule(cidr="1.1.1.0/24", ports=[443]),), None),
    ((EgressRule(cidr=CidrBlock(cidr="1.0.0.0/8", except_cidrs=("1.1.0.0/16",))),), None),
])
def test_policy_from_egress(rules, policy):
    assert policy_from_egress(rules) == policy


def test_policy_from_egress_reads_back_what_network_options_sends():
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("1.1.1.0/24",))
    assert policy_from_egress(network_options(policy).egress) == policy


@pytest.mark.parametrize(("policy", "url", "allowed"), [
    (NetworkPolicy(), "http://anything:8080/x", True),
    (None, "https://bucket.s3.amazonaws.com/o", False),
    (NetworkPolicy(), None, False),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("bucket.s3.amazonaws.com",)), "https://bucket.s3.amazonaws.com/o",
     True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("*.s3.amazonaws.com",)), "https://bucket.s3.amazonaws.com/o", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("*.amazonaws.com",)), "https://bucket.s3.amazonaws.com/o", False),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("store.test",)), "http://store.test/o", False),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("store.test",)), "https://store.test:8443/o", False),
    (NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("203.0.113.0/24",)), "http://203.0.113.7:9000/o", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("203.0.113.7",)), "http://203.0.113.7:9000/o", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("203.0.113.0/24",)), "http://198.51.100.1/o", False),
])
def test_allows_download_follows_coreweaves_rules(policy, url, allowed):
    """A hostname grant is HTTPS on 443 and a wildcard covers one label; a CIDR grant covers every port."""
    assert allows_download(policy, url) is allowed


@pytest.mark.asyncio
@pytest.mark.parametrize(("policy", "signed", "curled"), [
    (NetworkPolicy(), "https://bucket.s3.amazonaws.com/obj?sig", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",)), "https://bucket.s3.amazonaws.com/obj?sig", False),
    (None, "https://bucket.s3.amazonaws.com/obj?sig", False),
    (NetworkPolicy(), None, False),
])
async def test_a_download_the_sandbox_cannot_fetch_goes_over_stdin(monkeypatch, policy, signed, curled):
    store = SimpleNamespace(signed_get_url=lambda _url: signed)
    monkeypatch.setattr(sandbox_module, "get_config", lambda: SimpleNamespace(get_object_store_at=lambda _url: store))
    pushed = []

    async def push(sandbox, object_store, object_url, vm_path):
        pushed.append(vm_path)

    monkeypatch.setattr(sandbox_module, "push_object_over_stdin", push)
    sandbox, raw = _sandbox(policy=policy)
    await sandbox._download_object_to_vm("s3://b/obj", "/tmp/obj")
    assert any("curl" in " ".join(call["argv"]) for call in raw.exec_calls) is curled
    assert bool(pushed) is not curled


@pytest.mark.asyncio
async def test_image_tarballs_the_policy_blocks_are_pushed_rather_than_downloaded(monkeypatch):
    sandbox, _ = _sandbox(policy=NetworkPolicy(mode=ALLOWLIST, allow_hosts=("ok.test",)))
    signed = ["https://ok.test/a.tar.gz", "https://blocked.test/b.tar.gz"]
    loaded = []

    async def signed_urls(artifacts):
        return signed

    async def load(artifacts, urls):
        loaded.append(urls)

    monkeypatch.setattr(sandbox, "_signed_image_urls", signed_urls)
    monkeypatch.setattr(sandbox, "_load_docker_images", load)
    await sandbox._load_tarballs([object(), object()])
    assert loaded == [["https://ok.test/a.tar.gz", None]]


@pytest.mark.asyncio
async def test_a_container_runs_commands_as_given_in_its_own_container():
    sandbox, raw = _container()
    raw.results = [Result(stdout_bytes=b"hi\n"), SandboxUnavailableError("unavailable")]
    assert await sandbox.exec_with_output("sudo", "echo", "hi") == (0, "hi\n", "")
    process = await sandbox.exec("cat", "/big")
    assert (await process.wait(), await process.stdout.read()) == (-1, b"")
    assert [call["argv"] for call in raw.exec_calls] == [["sudo", "echo", "hi"], ["cat", "/big"]]
    assert {call["timeout"] for call in raw.exec_calls} == {600}


@pytest.mark.asyncio
async def test_a_container_takes_files_over_the_sandbox_api_rather_than_tools_its_image_may_lack(monkeypatch):
    store = SimpleNamespace(open=lambda url: io.BytesIO(b"object bytes"))
    monkeypatch.setattr(sandbox_module, "get_config", lambda: SimpleNamespace(get_object_store_at=lambda url: store))
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"url bytes"))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(sandbox_module.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    sandbox, raw = _container()
    await sandbox.write_file_from_text("text", "/app/a.txt")
    await sandbox.write_file_from_object("s3://b/obj", "/app/b.bin")
    await sandbox.write_file_from_url("https://files.example/c", "/app/c.bin")
    assert raw.files == {"/app/a.txt": b"text", "/app/b.bin": b"object bytes", "/app/c.bin": b"url bytes"}
    assert raw.exec_calls == []


@pytest.mark.asyncio
async def test_files_staged_into_a_container_are_written_into_it_directly(monkeypatch):
    store = SimpleNamespace(open=lambda url: io.BytesIO(url.encode()))
    monkeypatch.setattr(sandbox_module, "get_config", lambda: SimpleNamespace(get_object_store_at=lambda url: store))
    sandbox, raw = _container()
    loaded = await stage_files_into_container(sandbox, {"d/x.txt": SimpleNamespace(object_url="s3://b/x")}, "/work")
    assert loaded == {"d/x.txt": "/work/d/x.txt"}
    assert raw.files == {"/work/d/x.txt": b"s3://b/x"}
    assert all(call["argv"][:2] == ["mkdir", "-p"] for call in raw.exec_calls)


@pytest.mark.parametrize(("containers", "docker_host"), [
    ((Container(image="docker:dind", command="/bin/sh", args=("-c", host_setup_script("loop", 20)), primary=True),),
     True),
    ((Container(image="registry.example/agent:1", primary=True),), False),
    ((Container(image="registry.example/agent:1", command="/bin/sh", args=("-c", "serve")),), False),
    ((), None),
])
def test_a_docker_host_is_told_by_the_setup_script_it_was_started_with(containers, docker_host):
    assert runs_docker_host(SimpleNamespace(containers=containers)) is docker_host
