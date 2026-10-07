"""Unit tests for the VM-mode Modal backend (ModalVmSandbox / ModalVmSandboxProvider).

Offline: the real Modal sandbox is mocked. These cover the Modal-specific behavior the
inherited VmSandbox contract depends on — sudo stripping (Modal runs as root), dockerd
startup + readiness polling (Modal doesn't auto-start it), and that no iptables/firewall
step runs (ports are exposed via Modal encrypted_ports)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox, ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM


def _sandbox(sb: MagicMock | None = None) -> ModalVmSandbox:
    sb = sb or MagicMock()
    sb.object_id = "sb-test"
    return ModalVmSandbox(sb, {18765: "https://x.modal.host"})


def test_is_vm_sandbox_in_vm_mode():
    s = _sandbox()
    assert isinstance(s, VmSandbox)
    assert s.type == "modal_vm"
    assert s.mode == SANDBOX_MODE_VM
    assert s.sandbox_id == "sb-test"
    assert s.tunnel_urls == {18765: "https://x.modal.host"}


@pytest.mark.asyncio
async def test_exec_strips_sudo():
    # The inherited VmSandbox helpers call exec("sudo", "bash", "-c", ...) / exec("sudo",
    # "docker", "images"); Modal VM has no sudo (runs as root), so it must be stripped.
    sb = MagicMock()
    sb.object_id = "sb-test"
    captured: dict = {}

    async def fake_exec(*cmd, text=False):
        captured["cmd"] = cmd
        return MagicMock()

    sb.exec.aio = fake_exec
    s = ModalVmSandbox(sb, {})
    await s.exec("sudo", "bash", "-c", "echo hi")
    assert captured["cmd"] == ("bash", "-c", "echo hi")

    await s.exec("sudo", "docker", "images")
    assert captured["cmd"] == ("docker", "images")


@pytest.mark.asyncio
async def test_setup_starts_dockerd_polls_until_ready_and_skips_iptables():
    s = _sandbox()
    s._VM_READY_POLL_INTERVAL = 0  # don't actually sleep between polls
    s.exec_script = AsyncMock(return_value="")
    # `docker info`: down once, then ready.
    s.exec_with_output = AsyncMock(side_effect=[(1, "", "Cannot connect"), (0, "ok", "")])

    await s.setup_vm_for_gateway([18765])

    # One exec_script call: start dockerd. (The log forwarder is NOT started here — it
    # runs as a child of the sandbox main process; see test_main_script_* below.)
    assert s.exec_script.await_count == 1
    launch_script = s.exec_script.await_args_list[0].args[0]
    assert "dockerd" in launch_script
    assert "iptables" not in launch_script  # Modal exposes ports via encrypted_ports
    assert "modal_vm_log_forwarder" not in launch_script  # forwarder is in the main process
    # dockerd starts with json-file log rotation (daemon flags, no file written) so
    # container logs can't fill the rootfs.
    assert "--log-opt" in launch_script
    assert "max-size" in launch_script
    assert "max-file" in launch_script
    assert "daemon.json" not in launch_script  # passed as flags, not a written config file
    # polled `docker info` until it returned 0.
    assert s.exec_with_output.await_count == 2
    assert s.exec_with_output.await_args_list[0].args == ("docker", "info")


def test_main_script_streams_logs_to_stdout_and_is_a_bulletproof_keepalive():
    # The sandbox main process surfaces container logs in the Modal dashboard AND keeps the
    # VM alive. Two decoupled parts:
    from agent_env.providers.sandbox_providers.modal_vm_sandbox import _KEEPALIVE_CMD, _MAIN_SCRIPT

    assert _KEEPALIVE_CMD[:2] == ("sh", "-c")
    assert _KEEPALIVE_CMD[2] == _MAIN_SCRIPT
    # (1) a forwarder (backgrounded child of this process) that streams each container's
    #     logs, name-prefixed + unbuffered, to THIS process's stdout — which Modal captures.
    #     No spool file: nothing is written to disk, so no second on-disk copy and nothing
    #     to clean up.
    assert "docker logs -f" in _MAIN_SCRIPT
    assert "docker ps -aq" in _MAIN_SCRIPT  # -aq (not -q): also follow exited containers
    assert "docker ps -q " not in _MAIN_SCRIPT  # would drop crashed/short-lived containers
    assert "sed -u" in _MAIN_SCRIPT  # unbuffered -> lines flush immediately
    assert ") &" in _MAIN_SCRIPT  # forwarder backgrounded -> keepalive is the foreground
    assert "/var/log/agg.log" not in _MAIN_SCRIPT  # no spool file -> no extra disk copy
    assert "tail -F" not in _MAIN_SCRIPT  # not tailing a file anymore
    # (2) a keepalive that never exits, so a forwarder crash can't take the env down
    #     (equivalent to the old `sleep infinity` main process).
    assert "while true" in _MAIN_SCRIPT
    assert "sleep 3600" in _MAIN_SCRIPT


@pytest.mark.asyncio
async def test_setup_raises_with_dockerd_log_on_timeout():
    s = _sandbox()
    s._VM_READY_TIMEOUT = 0  # deadline already passed -> no polls, go straight to log+raise
    s._VM_READY_POLL_INTERVAL = 0
    s.exec_script = AsyncMock(return_value="")
    s.exec_with_output = AsyncMock(return_value=(0, "dockerd boom log", ""))

    with pytest.raises(RuntimeError, match="dockerd not ready"):
        await s.setup_vm_for_gateway([18765])
    # the failure surfaces the dockerd log tail for debugging
    assert s.exec_with_output.await_args.args[0] == "tail"


def test_provider_init_is_self_contained():
    # Standalone provider: constructs without touching Modal/network and is not coupled to
    # ModalSandboxProvider.
    from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider

    p = ModalVmSandboxProvider()
    assert not isinstance(p, ModalSandboxProvider)
    assert p._client is None and p._apps == {}


# ── object downloads: the aria2c range pool ───────────────────────────────────────

from agent_env.config import reset_config  # noqa: E402
from agent_env.providers.sandbox_providers import modal_vm_sandbox as mvs  # noqa: E402
from agent_env.store import set_object_store  # noqa: E402

URL = "https://bucket.s3.amazonaws.com/artifacts/file/x/1/x.zip?X-Amz-Signature=abc&X-Amz-Expires=3600"


def _summary(rc=0, nbytes=3, seconds=1, msg=""):
    return f"{mvs._DL_SUMMARY} rc={rc} bytes={nbytes} seconds={seconds} msg={msg}\n"


class _ScriptedVm(ModalVmSandbox):
    """Download scripts answer with the scripted summary lines in order; an int output is a transport drop."""

    def __init__(self, outputs):
        sb = MagicMock()
        sb.object_id = "sb-dl"
        super().__init__(sb, {})
        self.scripts: list[str] = []
        self._outputs = list(outputs)

    async def exec_with_output(self, *args):
        script = args[-1]
        self.scripts.append(script)
        if mvs._DL_SUMMARY in script:
            out = self._outputs.pop(0)
            return (out, "", "") if isinstance(out, int) else (0, out, "")
        return 0, "", ""

    def downloads(self):
        return [s for s in self.scripts if mvs._DL_SUMMARY in s]


class _Store:
    """Signs any object url and counts the URLs minted."""

    def __init__(self):
        self.minted = 0

    def signed_get_url(self, object_url, expires_in=3600):
        self.minted += 1
        return f"https://signed/{object_url.rsplit('/', 1)[-1]}?n={self.minted}"

    def get(self, object_url):  # pragma: no cover
        raise AssertionError("a signing store must not stream bytes through agent-env")


@pytest.fixture
def store():
    s = _Store()
    set_object_store(s)
    yield s
    reset_config()


def test_dl_settings_default_and_env_override(monkeypatch):
    assert mvs._dl_setting("CONNECTIONS", mvs._DL_CONNECTIONS) == 16
    monkeypatch.setenv("AGENT_ENV_MODAL_VM_DOWNLOAD_CONNECTIONS", "4")
    monkeypatch.setenv("AGENT_ENV_MODAL_VM_DOWNLOAD_ATTEMPTS", "")
    assert mvs._dl_setting("CONNECTIONS", mvs._DL_CONNECTIONS) == 4
    assert mvs._dl_setting("ATTEMPTS", mvs._DL_ATTEMPTS) == 3  # empty = unset
    monkeypatch.setenv("AGENT_ENV_MODAL_VM_DOWNLOAD_CONNECTIONS", "eight")
    with pytest.raises(ValueError, match="AGENT_ENV_MODAL_VM_DOWNLOAD_CONNECTIONS"):
        mvs._dl_setting("CONNECTIONS", mvs._DL_CONNECTIONS)


def test_download_script_shape(monkeypatch):
    script = mvs._render_download_script(URL, "/app/_artifact_staging/x.zip", fresh=False)
    assert script.count(URL) == 1 and "url=" in script and '"$url"' in script
    assert "timeout -k 15 1800 aria2c" in script
    assert "--max-connection-per-server=16 --split=16 --min-split-size=4M --lowest-speed-limit=0" in script
    assert "--show-console-readout=false" in script
    assert "--continue=true" in script and "--max-tries=8" in script and "curl" not in script
    assert script.rstrip().endswith("exit 0")
    assert f'echo "{mvs._DL_SUMMARY} rc=$rc bytes=$bytes' in script
    # URLs and long tokens are masked before the tail is taken
    assert "sed -E 's#https?://[^[:space:]]+#<url>#g; s#[A-Za-z0-9%+/=_-]{40,}#<token>#g' | tail -c 400" in script
    monkeypatch.setenv("AGENT_ENV_MODAL_VM_DOWNLOAD_CONNECTIONS", "40")  # clamped to aria2c's cap
    monkeypatch.setenv("AGENT_ENV_MODAL_VM_DOWNLOAD_DEADLINE_S", "900")
    script = mvs._render_download_script(URL, "/tmp/x", fresh=True)
    assert "timeout -k 15 900 aria2c" in script and "--max-connection-per-server=16 --split=16" in script


def test_first_attempt_clears_stale_state_and_later_attempts_resume():
    fresh = mvs._render_download_script(URL, "/tmp/_s3_x", fresh=True)
    resume = mvs._render_download_script(URL, "/tmp/_s3_x", fresh=False)
    assert 'rm -f "$out" "$out.aria2"' in fresh and fresh.index('rm -f "$out"') < fresh.index("aria2c")
    assert 'rm -f "$out" "$out.aria2"' not in resume and "--continue=true" in resume


def test_parse_download_summary():
    assert mvs._parse_download_summary("noise\n" + _summary(rc=22, nbytes=12, seconds=7, msg="status=403 <url>")) == (22, 12, 7, "status=403 <url>")
    assert mvs._parse_download_summary("nothing here\n") is None
    assert mvs._not_found(3, "") and mvs._not_found(4, "") and mvs._not_found(22, "status=404") and not mvs._not_found(22, "status=403")


@pytest.mark.asyncio
async def test_download_first_attempt_ok(store):
    """The URL signed to pick the path is the first attempt's."""
    vm = _ScriptedVm([_summary()])
    await vm.load_s3_file("s3://b/x.zip", "/app/x.zip")
    assert len(vm.downloads()) == 1
    assert store.minted == 1


@pytest.mark.asyncio
async def test_download_retries_with_a_fresh_url(store):
    vm = _ScriptedVm([_summary(rc=22, msg="status=403 <url>"), _summary()])
    await vm.load_s3_file("s3://b/x.zip", "/app/x.zip")
    first, second = vm.downloads()
    assert "?n=1" in first and "?n=2" in second
    assert 'rm -f "$out" "$out.aria2"' in first and 'rm -f "$out"' not in second  # attempt 2 resumes


@pytest.mark.asyncio
async def test_a_stale_probe_url_is_re_signed(store, monkeypatch):
    """A download that queued for a slot can outlive the URL signed to pick its path."""
    monkeypatch.setattr(mvs, "_FRESH_URL_SECONDS", -1)
    vm = _ScriptedVm([_summary()])
    await vm.load_s3_file("s3://b/x.zip", "/app/x.zip")
    assert "?n=2" in vm.downloads()[0]


@pytest.mark.asyncio
async def test_the_first_script_to_run_clears_stale_state_even_after_a_failed_sign(store, monkeypatch):
    """The path may hold another object's bytes, which a resumed download would splice onto."""
    monkeypatch.setattr(mvs, "_FRESH_URL_SECONDS", -1)
    signs = iter([store.signed_get_url, _refuse, store.signed_get_url])
    monkeypatch.setattr(store, "signed_get_url", lambda url, expires_in=3600: next(signs)(url))
    vm = _ScriptedVm([_summary()])
    await vm.load_s3_file("s3://b/x.zip", "/app/x.zip")
    [first] = vm.downloads()
    assert 'rm -f "$out" "$out.aria2"' in first


@pytest.mark.asyncio
async def test_a_failed_re_sign_costs_one_attempt(store, monkeypatch):
    signs = iter([store.signed_get_url, _refuse, store.signed_get_url])
    monkeypatch.setattr(store, "signed_get_url", lambda url, expires_in=3600: next(signs)(url))
    vm = _ScriptedVm([_summary(rc=1, msg="recv failure"), _summary()])
    await vm.load_s3_file("s3://b/x.zip", "/app/x.zip")
    assert len(vm.downloads()) == 2


def _refuse(url):
    raise ConnectionError("signer unreachable")


@pytest.mark.asyncio
async def test_download_not_found_raises_immediately(store):
    vm = _ScriptedVm([_summary(rc=3, msg="Resource not found"), _summary()])
    with pytest.raises(RuntimeError, match="not found"):
        await vm.load_s3_file("s3://b/missing.zip", "/app/missing.zip")
    assert len(vm.downloads()) == 1


@pytest.mark.asyncio
async def test_download_transport_drop_then_deadline_then_ok(store):
    vm = _ScriptedVm([-1, -1, _summary(rc=124, nbytes=1000), _summary()])  # exec_script retries -1 once itself
    await vm.load_s3_file("s3://b/x.zip", "/app/x.zip")
    assert len(vm.downloads()) == 4


@pytest.mark.asyncio
async def test_download_attempts_exhausted_cleans_up_and_raises(store, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_MODAL_VM_DOWNLOAD_ATTEMPTS", "2")
    vm = _ScriptedVm([_summary(rc=1, msg="recv failure"), _summary(rc=1, msg="recv failure again")])
    with pytest.raises(RuntimeError, match="failed after 2 attempts.*recv failure again"):
        await vm.load_s3_file("s3://b/x.zip", "/app/x.zip")
    assert any(s.startswith("rm -f /app/x.zip") for s in vm.scripts)
    assert any(s.startswith("rm -f /app/x.zip.aria2") for s in vm.scripts)


@pytest.mark.asyncio
async def test_an_object_the_store_cannot_sign_goes_over_stdin(monkeypatch):
    class _Local:
        def signed_get_url(self, object_url, expires_in=3600):
            return None

    pushed = []

    async def push(sandbox, store, object_url, vm_path):
        pushed.append((sandbox, store, object_url, vm_path))

    monkeypatch.setattr(mvs, "push_object_over_stdin", push)
    store = _Local()
    set_object_store(store)
    try:
        vm = _ScriptedVm([])
        await vm.load_s3_file("file:///store/x", "/app/x")
    finally:
        reset_config()
    assert pushed == [(vm, store, "file:///store/x", "/app/x")]
    assert not vm.downloads()


@pytest.mark.asyncio
async def test_a_stdin_exec_feeds_the_script_and_returns_what_it_printed():
    process = MagicMock()
    process.stdin.drain.aio = AsyncMock()
    process.wait.aio = AsyncMock(return_value=0)
    process.stdout.read.aio = AsyncMock(return_value=b"abc123  -\n")
    process.stderr.read.aio = AsyncMock(return_value=b"2+0 records in\n")
    sb = MagicMock()
    sb.exec.aio = AsyncMock(return_value=process)

    async def pieces():
        yield b"QUJD"
        yield b"REVG"

    result = await _sandbox(sb)._exec_with_stdin("base64 -d | dd of=/tmp/x && sha256sum /tmp/x", pieces())

    assert result == (0, "abc123  -\n", "2+0 records in\n")
    sb.exec.aio.assert_awaited_once_with("bash", "-c", "base64 -d | dd of=/tmp/x && sha256sum /tmp/x", text=False)
    assert [c.args for c in process.stdin.write.call_args_list] == [(b"QUJD",), (b"REVG",)]
    process.stdin.write_eof.assert_called_once()


@pytest.mark.asyncio
async def test_host_file_writes_stay_under_modals_exec_command_limit():
    import base64

    s = _sandbox()
    commands: list[tuple[str, ...]] = []

    async def record(*args):
        commands.append(args)
        return 0, "", ""

    s.exec_with_output = record
    data = b"z" * (300 * 1024)
    await s.write_host_file(data, "/tmp/run/input.json")

    assert max(sum(len(a) for a in c) for c in commands) < 48 * 1024
    appended = "".join(
        c[-1].split(" ", 2)[2].rsplit(" >> ", 1)[0].strip("'") for c in commands if c[-1].startswith("printf '%s' ")
    )
    assert appended == base64.b64encode(data).decode()
