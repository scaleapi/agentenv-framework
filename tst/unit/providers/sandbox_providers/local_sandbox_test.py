import asyncio
import os
import platform
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_env.providers.sandbox_providers.local_sandbox as ls
from agent_env.config import reset_config, set_object_store
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER
from agent_env.store import LocalFilesystemObjectStore
from agent_env.store.object_store.local_grants.tls import local_ca

_real_copy_into_container = ls._copy_into_container  # before the fixture below replaces it


@pytest.fixture(autouse=True)
def copies(monkeypatch):
    """What create_container would ``docker cp`` into its container, recorded instead of run."""
    recorded: list[tuple] = []
    monkeypatch.setattr(ls, "_copy_into_container", lambda *args: recorded.append(args))
    return recorded


class _StubImageStore:
    def auth(self, image_name):
        return None


class _StubConfig:
    def get_image_store(self):
        return _StubImageStore()


class _ScriptedProvider(LocalSandboxProvider):
    """Creates a recording sandbox under ``work_dir`` in place of a VM."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.made: list[_RecordingLocalSandbox] = []

    async def create_vm(self, **kwargs):
        sandbox = _RecordingLocalSandbox(work_dir=self.work_dir)
        self.made.append(sandbox)
        return sandbox


def test_rewrite_app_arg_only_rewrites_app_prefix():
    sandbox = LocalSandbox(work_dir=Path("/tmp/agent-env-work"))

    assert sandbox._rewrite_app_arg("/app") == "/tmp/agent-env-work"
    assert sandbox._rewrite_app_arg("/app/docker-compose.yml") == "/tmp/agent-env-work/docker-compose.yml"
    assert sandbox._rewrite_app_arg("/app:/container") == "/tmp/agent-env-work:/container"
    assert sandbox._rewrite_app_arg("/home/alice/app/repo") == "/home/alice/app/repo"
    assert sandbox._rewrite_app_arg("/tmp/application") == "/tmp/application"


def test_rewrite_app_script_keeps_local_paths_containing_app():
    sandbox = LocalSandbox(work_dir=Path("/tmp/agent-env-work"))

    script = "cd /app && cp '/home/alice/app/data.json' /app/data.json"

    assert sandbox._rewrite_app_script(script) == (
        "cd /tmp/agent-env-work && cp '/home/alice/app/data.json' "
        "/tmp/agent-env-work/data.json"
    )


def test_rewrite_app_script_handles_docker_volume_mounts():
    sandbox = LocalSandbox(work_dir=Path("/tmp/agent-env-work"))

    script = "docker run -v /app:/container image"

    assert sandbox._rewrite_app_script(script) == (
        "docker run -v /tmp/agent-env-work:/container image"
    )


def test_find_work_dir_only_returns_matching_sandbox_id(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))
    requested = tmp_path / "agent-env-local-requested-abc123"
    other = tmp_path / "agent-env-local-other-def456"
    requested.mkdir()
    other.mkdir()
    (other / "docker-compose.yml").write_text("")

    assert LocalSandbox.find_work_dir("local-requested") == requested
    assert LocalSandbox.find_work_dir("local-missing") is None


def test_find_work_dir_falls_back_to_legacy_tempdir(tmp_path: Path, monkeypatch):
    """A sandbox created before the work-dir relocation (under the temp dir) is still found."""
    new_root = tmp_path / "new"
    legacy = tmp_path / "legacy"
    new_root.mkdir()
    legacy.mkdir()
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(new_root))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(legacy))
    old = legacy / "agent-env-local-old123-deadbeef"
    old.mkdir()

    assert LocalSandbox.find_work_dir("local-old123") == old
    assert LocalSandbox.find_work_dir("local-missing") is None


def test_find_work_dir_survives_uncreatable_new_root(tmp_path: Path, monkeypatch):
    """Lookup must not depend on creating the new root: an uncreatable root still lets the legacy
    temp-dir scan find (and tear down) a pre-existing sandbox, and never mkdirs as a side effect."""
    blocker = tmp_path / "blocker"
    blocker.write_text("")  # a file — mkdir of a child would raise NotADirectoryError
    uncreatable = blocker / "root"
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(uncreatable))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(legacy))
    old = legacy / "agent-env-local-old999-cafe"
    old.mkdir()

    assert LocalSandbox.find_work_dir("local-old999") == old
    assert not uncreatable.exists()  # the lookup did not try to create the new root


@pytest.mark.asyncio
async def test_get_sandbox_raises_when_work_dir_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="Local sandbox work directory not found"):
        await LocalSandboxProvider().get_sandbox("local-missing")


class _RecordingLocalSandbox(LocalSandbox):
    def __init__(self, work_dir):
        super().__init__(work_dir=work_dir)
        self.scripts: list[str] = []

    async def exec_script(self, script, *, max_retries=0):
        self.scripts.append(script)
        return ""


@pytest.mark.asyncio
async def test_unsigned_object_is_written_in_place_on_this_host(tmp_path):
    """This host is the VM: an object the store cannot sign a URL for is written straight to the
    path (/app mapped to the work dir), with no curl or base64 over exec."""
    store = LocalFilesystemObjectStore(str(tmp_path / "store"))
    url = store.put("images/img.tar.gz", b"TARBALL")
    set_object_store(store)
    try:
        sandbox = _RecordingLocalSandbox(work_dir=tmp_path / "work")
        await sandbox._download_object_to_vm(url, "/app/staged/img.tar.gz")
    finally:
        reset_config()

    assert (tmp_path / "work" / "staged" / "img.tar.gz").read_bytes() == b"TARBALL"
    assert sandbox.scripts == []


@pytest.mark.asyncio
async def test_download_s3_object_url_falls_back_to_signed_curl(tmp_path):
    """An object the store can sign a URL for still goes through the signed-URL curl path."""
    class _Signing:
        def signed_get_url(self, url, expires_in=3600):
            return f"https://signed/{url.rsplit('/', 1)[-1]}"

    set_object_store(_Signing())
    try:
        sandbox = _RecordingLocalSandbox(work_dir=tmp_path)
        await sandbox._download_object_to_vm("s3://bucket/x.tar.gz", "/tmp/out.tar.gz")
    finally:
        reset_config()

    assert len(sandbox.scripts) == 1
    assert "curl" in sandbox.scripts[0] and "https://signed/x.tar.gz" in sandbox.scripts[0]


class _ImageLoadingLocalSandbox(_RecordingLocalSandbox):
    async def exec_with_output(self, *command):
        return 0, "svc latest", ""


@pytest.mark.asyncio
async def test_load_docker_images_stages_tarballs_per_sandbox(tmp_path):
    """Two local sandboxes load images on the same host at once: each stages its tarball under
    its own name, so neither copies over (or removes) the other's."""
    class _Local:
        def __init__(self):
            self.staged = []

        def signed_get_url(self, url, expires_in=3600):
            return None

        def download_to_file(self, url, dest_path):
            self.staged.append(dest_path)

    artifact = SimpleNamespace(tar_gz_object_url="file:///store/svc.tar.gz", image_name="svc:latest")
    store = _Local()
    set_object_store(store)
    try:
        for sandbox_id in ("local-aaaa", "local-bbbb"):
            sandbox = _ImageLoadingLocalSandbox(work_dir=tmp_path)
            sandbox.sandbox_id = sandbox_id
            await sandbox.load_docker_images([artifact])
    finally:
        reset_config()
    staged = store.staged

    assert staged[0] != staged[1]
    assert "local-aaaa" in staged[0] and "local-bbbb" in staged[1]


def test_local_container_name_is_per_sandbox():
    """Each local sandbox gets its own container name (not the fixed VmSandbox default), so
    concurrent local agents don't collide and teardown is ownership-scoped."""
    a = LocalSandbox(sandbox_id="local-aaa", work_dir=Path("/tmp/wd-a"))
    b = LocalSandbox(sandbox_id="local-bbb", work_dir=Path("/tmp/wd-b"))
    assert a.container_name == "agent-local-aaa"
    assert b.container_name == "agent-local-bbb"
    assert a.container_name != b.container_name


@pytest.mark.asyncio
async def test_terminate_removes_this_sandboxs_own_container():
    """A container-mode sandbox removes only its own per-sandbox container on teardown."""
    sandbox = _RecordingLocalSandbox(work_dir=Path("/tmp/agent-env-work"))
    sandbox.mode = SANDBOX_MODE_CONTAINER
    await sandbox.terminate()

    assert len(sandbox.scripts) == 1
    assert f"docker rm -f {sandbox.container_name}" in sandbox.scripts[0]
    assert sandbox.container_name != "agent-api"  # it's the unique local name


@pytest.mark.asyncio
async def test_terminate_compose_downs_vm_stack(tmp_path: Path):
    """A VM-mode sandbox with a compose file tears the stack down (scoped to its own work dir)."""
    (tmp_path / "docker-compose.yml").write_text("")
    sandbox = _RecordingLocalSandbox(work_dir=tmp_path)  # default mode is VM
    await sandbox.terminate()

    (compose_down,) = [script for script in sandbox.scripts if "docker compose down" in script]
    assert str(tmp_path) in compose_down


@pytest.mark.asyncio
async def test_terminate_surfaces_compose_down_failure(tmp_path: Path):
    """A failed `docker compose down` is not masked (no `|| true`): teardown raises so a leaked stack
    and its held host ports are detectable instead of silently left running."""
    (tmp_path / "docker-compose.yml").write_text("")

    class _FailingSandbox(_RecordingLocalSandbox):
        async def exec_script(self, script, *, max_retries=0):
            self.scripts.append(script)
            raise RuntimeError("compose down blew up")

    sandbox = _FailingSandbox(work_dir=tmp_path)  # VM mode + compose file → compose-down path
    with pytest.raises(RuntimeError, match="compose down blew up"):
        await sandbox.terminate()


@pytest.mark.asyncio
async def test_terminate_vm_without_compose_removes_only_an_agent_placed_on_it(tmp_path: Path):
    """A VM-mode sandbox that never rendered a compose stack removes only the agent container an agent placed
    on it runs as, if there is one."""
    sandbox = _RecordingLocalSandbox(work_dir=tmp_path)
    await sandbox.terminate()

    assert sandbox.scripts == [f"docker rm -f {sandbox.container_name} >/dev/null 2>&1 || true"]


@pytest.mark.asyncio
async def test_get_sandbox_restores_container_mode_from_marker(tmp_path: Path, monkeypatch):
    """The container-mode marker persisted at create time is restored so teardown cleans up."""
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))
    work_dir = tmp_path / "agent-env-local-agent1-abc123"
    work_dir.mkdir()
    (work_dir / ".agent-container-mode").write_text("agent-local-agent1")

    sandbox = await LocalSandboxProvider().get_sandbox("local-agent1")
    assert sandbox.mode == SANDBOX_MODE_CONTAINER


@pytest.mark.asyncio
@pytest.mark.parametrize("create", ["create_container", "create_sandbox"])  # the server provider calls the first, agents the second
async def test_a_started_container_is_removed_by_a_teardown_rebuilt_from_disk(tmp_path: Path, monkeypatch, create):
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))

    class _StubImageStore:
        def auth(self, image_name):
            return None

    class _StubConfig:
        def get_image_store(self):
            return _StubImageStore()

    import agent_env.config as cfg
    monkeypatch.setattr(cfg, "get_config", lambda: _StubConfig())

    scripts: list[str] = []

    async def record(self, script, *, max_retries=0):
        scripts.append(script)
        return ""

    monkeypatch.setattr(LocalSandbox, "exec_script", record)

    class _Provider(LocalSandboxProvider):
        async def create_vm(self, **kwargs):
            return LocalSandbox()

    created = await getattr(_Provider(), create)(image_name="img:v1", port=8000, env={})
    rebuilt = await LocalSandboxProvider().get_sandbox(created.sandbox_id)
    await rebuilt.terminate()
    assert rebuilt.mode == SANDBOX_MODE_CONTAINER and scripts[-1] == f"docker rm -f {created.container_name} >/dev/null 2>&1 || true"


@pytest.mark.asyncio
async def test_create_container_marker_failure_tears_down_and_raises(tmp_path: Path, monkeypatch):
    """If the marker can't be persisted, a reconstructed teardown couldn't tell this is a container
    and would leak it — so tear the just-started container down and fail loudly, never leak it."""
    blocker = tmp_path / "blocker"
    blocker.write_text("")  # a file — writing a marker under it raises NotADirectoryError
    bad_work_dir = blocker / "wd"

    class _StubImageStore:
        def auth(self, image_name):
            return None

    class _StubConfig:
        def get_image_store(self):
            return _StubImageStore()

    import agent_env.config as cfg
    monkeypatch.setattr(cfg, "get_config", lambda: _StubConfig())

    made: list = []

    class _RecordingProvider(LocalSandboxProvider):
        async def create_vm(self, **kwargs):
            sb = _RecordingLocalSandbox(work_dir=bad_work_dir)
            made.append(sb)
            return sb

    with pytest.raises(OSError):
        await _RecordingProvider().create_container(image_name="img:v1", port=8000, env={})
    # the started container was torn down before the error propagated
    assert made and any("docker rm -f" in s for s in made[0].scripts)
    assert not (bad_work_dir / ".agent-container-mode").exists()  # marker genuinely failed to write


@pytest.mark.asyncio
async def test_create_container_marker_failure_escalates_when_removal_also_fails(tmp_path: Path, monkeypatch, caplog):
    """Double failure (marker write AND docker rm): don't mask it — the original error still
    propagates and the un-removable container is escalated (ERROR) rather than silently leaked."""
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    bad_work_dir = blocker / "wd"

    class _StubImageStore:
        def auth(self, image_name):
            return None

    class _StubConfig:
        def get_image_store(self):
            return _StubImageStore()

    import agent_env.config as cfg
    monkeypatch.setattr(cfg, "get_config", lambda: _StubConfig())

    class _RmFailsSandbox(_RecordingLocalSandbox):
        async def exec_script(self, script, *, max_retries=0):
            self.scripts.append(script)
            if "docker rm" in script:
                raise RuntimeError("docker daemon unreachable")
            return ""

    class _RecordingProvider(LocalSandboxProvider):
        async def create_vm(self, **kwargs):
            return _RmFailsSandbox(work_dir=bad_work_dir)

    import logging
    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError):  # the original marker error, not swallowed by the rm failure
            await _RecordingProvider().create_container(image_name="img:v1", port=8000, env={})
    assert any("remove it manually" in r.message for r in caplog.records)  # escalated, not hidden


def test_local_never_shares_network_and_externalizes_localhost():
    """Local agent (docker-run bridge) and env (compose network) reach each other via the host."""
    assert LocalSandboxProvider.shares_network_with("local") is False
    assert (
        LocalSandboxProvider.get_external_url("http://localhost:18765/mcp")
        == "http://host.docker.internal:18765/mcp"
    )


def test_host_gateway_flag_only_where_nothing_provides_the_alias():
    """Only Linux lacks a native host.docker.internal; on Rancher an explicit mapping would break it."""
    expected = "--add-host host.docker.internal:host-gateway" if platform.system() == "Linux" else ""
    assert LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS == expected


@pytest.mark.asyncio
async def test_create_sandbox_splices_host_gateway_flag(tmp_path: Path, monkeypatch):
    """Whatever EXTRA_CONTAINER_RUN_ARGS holds is spliced into the agent container's docker run."""
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))
    monkeypatch.setattr(LocalSandboxProvider, "EXTRA_CONTAINER_RUN_ARGS", "--add-host host.docker.internal:host-gateway")

    class _StubImageStore:
        def auth(self, image_name):
            return None

    class _StubConfig:
        def get_image_store(self):
            return _StubImageStore()

    import agent_env.config as cfg
    monkeypatch.setattr(cfg, "get_config", lambda: _StubConfig())

    recorded: list[str] = []

    class _RecordingProvider(LocalSandboxProvider):
        async def create_vm(self, **kwargs):
            sb = LocalSandbox(exposed_ports=kwargs.get("exposed_ports"))

            async def _rec(script, *, max_retries=0):
                recorded.append(script)
                return ""

            sb.exec_script = _rec  # type: ignore[method-assign]
            return sb

    await _RecordingProvider().create_sandbox(image_name="img:v1", port=8000, env={"K": "v"})
    run_cmd = next(s for s in recorded if "docker create" in s)
    assert "--add-host host.docker.internal:host-gateway" in run_cmd


@pytest.mark.asyncio
async def test_create_vm_publishes_on_an_allocated_host_port_not_the_requested_one(monkeypatch):
    """The local backend packs every sandbox onto one Docker host, so a requested container
    port can only be published once — `create_vm` runs every exposed port through the free-port
    allocator instead. `tunnel_urls` (keyed by CONTAINER port) therefore names the HOST port,
    and the mapping is not identity. An integration test that assumed identity read the wrong
    URL and failed against a container that was serving fine.

    The allocator is stubbed rather than called: it binds a real socket, which the unit tier
    forbids, and what matters here is that every exposed port goes through it.
    """
    allocated = iter([41001, 41002])
    monkeypatch.setattr(ls, "_free_host_port", lambda: next(allocated))

    sandbox = await LocalSandboxProvider().create_vm(exposed_ports=[8000, 9001])

    assert sandbox.host_port(8000) == 41001
    assert sandbox.host_port(9001) == 41002
    assert sandbox.tunnel_urls[8000] == "http://localhost:41001"
    assert sandbox.tunnel_urls[9001] == "http://localhost:41002"
    # An unexposed port has nothing published, so it falls back to identity.
    assert sandbox.host_port(7777) == 7777


@pytest.mark.asyncio
async def test_create_container_publishes_the_allocated_host_port(monkeypatch, tmp_path):
    """The `docker run -p` left-hand side is the allocated host port, not the container port,
    so what `tunnel_urls` advertises is what Docker actually published."""
    import agent_env.config as cfg

    class _StubImageStore:
        def auth(self, image_name):
            return None

    class _StubConfig:
        def get_image_store(self):
            return _StubImageStore()

    monkeypatch.setattr(cfg, "get_config", lambda: _StubConfig())
    monkeypatch.setattr(ls, "_free_host_port", lambda: 41337)

    recorded: list[str] = []
    made: list[LocalSandbox] = []

    class _RecordingProvider(LocalSandboxProvider):
        async def create_vm(self, **kwargs):
            sb = await LocalSandboxProvider.create_vm(self, **kwargs)
            sb._work_dir = tmp_path  # type: ignore[attr-defined]

            async def _rec(script, *, max_retries=0):
                recorded.append(script)
                return ""

            sb.exec_script = _rec  # type: ignore[method-assign]
            made.append(sb)
            return sb

    await _RecordingProvider().create_sandbox(image_name="img:v1", port=8000, env={})

    assert made[0].host_port(8000) == 41337
    run_cmd = next(s for s in recorded if "docker create" in s)
    assert "-p 41337:8000" in run_cmd


class _CapturedProcess:
    def __init__(self):
        self.stdout = self.stderr = SimpleNamespace(read=self._empty)

    @staticmethod
    async def _empty():
        return b""

    async def wait(self):
        return 0


@pytest.fixture
def spawned(monkeypatch):
    argvs: list[tuple[str, ...]] = []

    async def fake_spawn(*argv, **_):
        argvs.append(argv)
        return _CapturedProcess()

    monkeypatch.setattr(ls.asyncio, "create_subprocess_exec", fake_spawn)
    return argvs


@pytest.mark.asyncio
async def test_exec_rewrites_app_inside_a_shell_script(spawned):
    sandbox = LocalSandbox(work_dir=Path("/tmp/agent-env-work"))

    await sandbox.exec("sudo", "bash", "-c", "cd /app/greeting && (python3 /app/greeting/check.py)")

    assert spawned == [(
        "bash", "-c", "cd /tmp/agent-env-work/greeting && (python3 /tmp/agent-env-work/greeting/check.py)",
    )]


@pytest.mark.asyncio
async def test_exec_leaves_a_script_for_another_program_alone(spawned):
    sandbox = LocalSandbox(work_dir=Path("/tmp/agent-env-work"))

    await sandbox.exec("docker", "exec", "c", "bash", "-c", "cat /app/x")

    assert spawned == [("docker", "exec", "c", "bash", "-c", "cat /app/x")]



@pytest.mark.asyncio
@pytest.mark.parametrize("command", [
    ("docker", "exec", "c", "test", "-e", "/app/out.txt"),
    ("docker", "exec", "-w", "/app", "c", "bash", "-c", "pytest -q"),
])
async def test_exec_leaves_a_docker_exec_naming_the_containers_app_alone(spawned, command):
    sandbox = LocalSandbox(work_dir=Path("/tmp/agent-env-work"))

    await sandbox.exec("sudo", *command)

    assert spawned == [command]

@pytest.mark.asyncio
async def test_exec_script_rewrites_app_exactly_once(spawned):
    sandbox = LocalSandbox(work_dir=Path("/app/sandboxes/wd"))

    await sandbox.exec_script("/app/run.sh && ls /app")

    assert spawned == [("bash", "-c", "/app/sandboxes/wd/run.sh && ls /app/sandboxes/wd")]


@pytest.mark.asyncio
async def test_exec_leaves_a_docker_exec_script_naming_the_containers_app_alone(spawned):
    sandbox = LocalSandbox(work_dir=Path("/tmp/agent-env-work"))
    script = "sudo docker exec c timeout 60 bash -c 'cd /app && pytest /tests'"

    await sandbox.exec("sudo", "bash", "-c", script)

    assert spawned == [("bash", "-c", script)]


@pytest.mark.parametrize("script", ["ls ~/app/x", "cat ${HOME}/app/x", "cat $(pwd)/app/x", "cat $APP/app/x"])
def test_rewrite_app_script_keeps_app_after_an_expansion(script):
    assert LocalSandbox(work_dir=Path("/tmp/agent-env-work"))._rewrite_app_script(script) == script


@pytest.mark.asyncio
async def test_cancelling_a_command_kills_everything_it_started(tmp_path: Path):
    sandbox = LocalSandbox(work_dir=tmp_path)
    running = asyncio.ensure_future(
        sandbox.exec_with_output("bash", "-c", "sleep 30 & echo $! > /app/child.pid; wait"))
    pid_file = tmp_path / "child.pid"
    while not pid_file.exists() or not pid_file.read_text().strip():
        await asyncio.sleep(0.01)
    child = int(pid_file.read_text())

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    for _ in range(200):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail(f"the command's child {child} outlived its cancelled call")



@pytest.mark.asyncio
async def test_a_command_stays_in_this_process_group_so_signals_to_the_group_reach_it(tmp_path: Path):
    sandbox = LocalSandbox(work_dir=tmp_path)

    _, stdout, _ = await sandbox.exec_with_output("bash", "-c", "ps -o pgid= -p $$")

    assert int(stdout) == os.getpgrp()


@pytest.mark.asyncio
async def test_a_command_cancelled_while_it_spawns_is_still_stopped(tmp_path: Path, monkeypatch):
    spawn = asyncio.create_subprocess_exec

    async def slow_spawn(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        await asyncio.sleep(0.2)  # the spawn is still finishing when the cancel lands
        return process

    monkeypatch.setattr(ls.asyncio, "create_subprocess_exec", slow_spawn)
    sandbox = LocalSandbox(work_dir=tmp_path)
    running = asyncio.ensure_future(sandbox.exec_with_output("bash", "-c", "echo $$ > /app/shell.pid; sleep 30"))
    pid_file = tmp_path / "shell.pid"
    while not pid_file.exists() or not pid_file.read_text().strip():
        await asyncio.sleep(0.01)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.asyncio
async def test_a_container_is_created_given_the_local_ca_then_started(tmp_path, monkeypatch, copies):
    """The CA's trust files are in place before the container's first process runs, and its TLS clients
    are pointed at them, so it can use the local object store's grants."""
    import agent_env.config as cfg

    monkeypatch.setattr(cfg, "get_config", lambda: _StubConfig())
    provider = _ScriptedProvider(tmp_path)
    sandbox = await provider.create_container(image_name="img:v1", port=8000, env={"K": "v"})

    create, start = [s for s in provider.made[0].scripts if s.startswith(("docker create", "docker start"))]
    assert "docker run" not in "".join(provider.made[0].scripts)
    for flag in (
        "-e SSL_CERT_FILE=/etc/agentenv/ca-bundle.pem",
        "-e REQUESTS_CA_BUNDLE=/etc/agentenv/ca-bundle.pem",
        "-e NODE_EXTRA_CA_CERTS=/etc/agentenv/ca.pem",
        "-e K=v",
    ):
        assert flag in create
    assert copies == [(local_ca().trust_dir, sandbox.container_name, "/etc/agentenv")]
    assert start == f"docker start {sandbox.container_name} > /dev/null"
    assert sandbox.mode == SANDBOX_MODE_CONTAINER


@pytest.mark.asyncio
async def test_a_trust_variable_the_caller_sets_wins(tmp_path, monkeypatch):
    import agent_env.config as cfg

    monkeypatch.setattr(cfg, "get_config", lambda: _StubConfig())
    provider = _ScriptedProvider(tmp_path)
    await provider.create_container(image_name="img:v1", port=8000, env={"SSL_CERT_FILE": "/own/roots.pem"})

    create = next(s for s in provider.made[0].scripts if s.startswith("docker create"))
    assert "-e SSL_CERT_FILE=/own/roots.pem" in create
    assert "SSL_CERT_FILE=/etc/agentenv" not in create


@pytest.mark.asyncio
async def test_a_failed_copy_removes_the_created_container(tmp_path, monkeypatch):
    import agent_env.config as cfg

    monkeypatch.setattr(cfg, "get_config", lambda: _StubConfig())

    def fail(*args):
        raise RuntimeError("no such container")

    monkeypatch.setattr(ls, "_copy_into_container", fail)
    provider = _ScriptedProvider(tmp_path)
    with pytest.raises(RuntimeError, match="no such container"):
        await provider.create_container(image_name="img:v1", port=8000, env={})
    scripts = provider.made[0].scripts
    assert not any(s.startswith("docker start") for s in scripts)
    assert scripts[-1].startswith(f"docker rm -f {provider.made[0].container_name}")


def test_the_copy_runs_docker_directly_and_reports_its_error(monkeypatch, tmp_path):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stderr="Error: No such container: agent-x\n")

    monkeypatch.setattr(ls.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="No such container"):
        _real_copy_into_container(tmp_path, "agent-x", "/etc/agentenv")
    assert calls == [["docker", "cp", f"{tmp_path}/.", "agent-x:/etc/agentenv"]]
