import asyncio
import hashlib
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.store import ImageStore, RegistryAuth, set_object_store
from agent_env.config import get_config, reset_config
from tst.unit.store.fakes import ConfiguredObjectStore
from tst.util.exec_scripts import script_run


def _image(tar_gz_object_url: str | None, image_name: str = "img:1") -> DockerImageArtifact:
    return DockerImageArtifact(id="img", description="", image_name=image_name, tar_gz_object_url=tar_gz_object_url)


class _SigningStore:
    """Fake object store that 'signs' any url into a fake https URL (the VM-curl path)."""

    def signed_get_url(self, object_url, expires_in=3600):
        return f"https://signed/{object_url.rsplit('/', 1)[-1]}"

    def get(self, object_url):
        return b"IMG"


@pytest.fixture
def signing_store():
    set_object_store(_SigningStore())
    yield
    reset_config()


class _RecordingVmSandbox(VmSandbox):
    """Concrete VmSandbox that records executed scripts instead of touching a VM."""

    def __init__(self, images_stdout: str = ""):
        self.sandbox_id = "vm-test"
        self.scripts: list[str] = []
        self._images_stdout = images_stdout

    async def terminate(self) -> None:  # pragma: no cover - not exercised
        pass

    async def exec(self, *command):  # pragma: no cover - not exercised
        return None

    async def exec_with_output(self, *args):
        # load_docker_images runs `sudo docker images` to verify; everything
        # else flows through exec_script as `sudo bash -c <script>`.
        if args[:2] == ("sudo", "docker") and "images" in args:
            return 0, self._images_stdout, ""
        if args[:2] == ("sudo", "bash"):
            self.scripts.append(script_run(args))
        return 0, "", ""

    async def write_file_from_text(self, content, destination_path):  # pragma: no cover
        pass


class _PushTargetVmSandbox(_RecordingVmSandbox):
    """Answers a push's sha256 check with the digest of ``pushed``, as a VM that received all of it would."""

    def __init__(self, pushed: bytes, images_stdout: str = ""):
        super().__init__(images_stdout)
        self._digest = hashlib.sha256(pushed).hexdigest()

    async def exec_with_output(self, *args):
        if args[:2] == ("sudo", "bash") and "sha256sum " in script_run(args):
            self.scripts.append(script_run(args))
            return 0, f"{self._digest}  pushed\n", ""
        return await super().exec_with_output(*args)


@pytest.mark.asyncio
async def test_load_docker_images_downloads_to_file_before_load(signing_store):
    """The docker-load path must download to a temp file before `gunzip | docker load`.

    Retrying `curl ... | gunzip | docker load` directly corrupts the stream
    because curl can't rewind bytes already piped to stdout, so the retry flags
    must only apply to a `-o file` download.
    """
    sandbox = _RecordingVmSandbox(images_stdout="myimage\n")
    artifact = _image("s3://bucket/img.tar.gz", "myimage:latest")

    await sandbox.load_docker_images([artifact])

    load_script = next(s for s in sandbox.scripts if "docker load" in s)
    # Download is retry-safe: the retry flags apply to a `-o file` curl.
    assert "--retry-all-errors" in load_script
    assert "-o " in load_script
    # The retried curl must NOT pipe straight into gunzip/docker load — the
    # download (everything up to `-o <file>`) contains no pipe into gunzip.
    assert "| gunzip" not in load_script.split("-o")[0]
    # gunzip reads from the downloaded file, not curl's stdout.
    assert "gunzip -c" in load_script


@pytest.mark.asyncio
async def test_load_docker_images_uses_unique_tmp_per_artifact(signing_store):
    sandbox = _RecordingVmSandbox(images_stdout="a\nb\n")
    artifacts = [
        _image("s3://bucket/a.tar.gz", "a:1"),
        _image("s3://bucket/b.tar.gz", "b:1"),
    ]

    await sandbox.load_docker_images(artifacts)

    load_script = next(s for s in sandbox.scripts if "docker load" in s)
    assert "/tmp/_docker_image_vm-test_0.tar.gz" in load_script
    assert "/tmp/_docker_image_vm-test_1.tar.gz" in load_script


@pytest.mark.asyncio
async def test_a_load_of_tarballs_runs_the_script_it_always_has(signing_store):
    sandbox = _RecordingVmSandbox(images_stdout="a\nb\n")

    await sandbox.load_docker_images([_image("s3://bucket/a.tar.gz", "a:1"), _image("s3://bucket/b.tar.gz", "b:1")])

    flags = "--retry 5 --retry-all-errors --retry-delay 1"
    assert sandbox.scripts == [
        f'(curl -fsSL {flags} "https://signed/a.tar.gz" -o /tmp/_docker_image_vm-test_0.tar.gz '
        "&& gunzip -c /tmp/_docker_image_vm-test_0.tar.gz | docker load && rm -f /tmp/_docker_image_vm-test_0.tar.gz) & "
        f'(curl -fsSL {flags} "https://signed/b.tar.gz" -o /tmp/_docker_image_vm-test_1.tar.gz '
        "&& gunzip -c /tmp/_docker_image_vm-test_1.tar.gz | docker load && rm -f /tmp/_docker_image_vm-test_1.tar.gz) & wait"
    ]


class _Registry(ImageStore):
    """Holds credentials for registry.example only."""

    def image_ref(self, repository, tag):
        return f"registry.example/{repository}:{tag}"

    def auth(self, ref):
        return RegistryAuth("registry.example", "user", "token") if ref.startswith("registry.example/") else None


@pytest.fixture
def registry():
    get_config().set_image_store(_Registry())
    yield
    reset_config()


PRIVATE = "registry.example/team/app@sha256:" + "0" * 64
PUBLIC = "ghcr.io/team/tool:v1"


@pytest.mark.asyncio
async def test_an_image_with_no_tarball_is_pulled_after_logging_in_to_the_registry_the_store_holds(registry):
    sandbox = _RecordingVmSandbox()

    await sandbox.load_docker_images([_image(None, PRIVATE), _image(None, PUBLIC), _image(None, PRIVATE)])

    login, *pulls = sandbox.scripts
    assert login == "echo token | docker login --username user --password-stdin registry.example"
    assert sorted(pulls) == sorted([f"docker pull {PRIVATE}", f"docker pull {PUBLIC}"])


@pytest.mark.asyncio
async def test_tarballs_are_loaded_and_the_rest_pulled(signing_store, registry):
    sandbox = _RecordingVmSandbox(images_stdout="a\n")

    await sandbox.load_docker_images([_image(None, PUBLIC), _image("s3://bucket/a.tar.gz", "a:1")])

    load, pull = sandbox.scripts
    assert '"https://signed/a.tar.gz"' in load and PUBLIC not in load
    assert pull == f"docker pull {PUBLIC}"


@pytest.mark.asyncio
async def test_an_image_no_sandbox_can_get_is_refused_before_anything_is_loaded(signing_store):
    sandbox = _RecordingVmSandbox(images_stdout="a\n")

    with pytest.raises(RuntimeError, match="Can't load images: 'img' v0 has no tar.gz, and its image name 'img:v1' doesn't name"):
        await sandbox.load_docker_images([_image("s3://bucket/a.tar.gz", "a:1"), _image(None, "img:v1")])

    assert sandbox.scripts == []


@pytest.mark.asyncio
async def test_a_login_is_minted_once_per_registry():
    minted = []

    class _Counting(_Registry):
        def auth(self, ref):
            minted.append(ref)
            return super().auth(ref)

    get_config().set_image_store(_Counting())
    try:
        sandbox = _RecordingVmSandbox()
        await sandbox.pull_images([PRIVATE, "registry.example/team/other:v1", PUBLIC, "ghcr.io/team/more:v2"])
    finally:
        reset_config()

    assert len(minted) == 2  # registry.example's and ghcr.io's
    assert sum("docker login" in script for script in sandbox.scripts) == 1
    assert sum(script.startswith("docker pull") for script in sandbox.scripts) == 4


class _OnePullFailsVm(_RecordingVmSandbox):
    """``docker pull`` of BAD fails; every other pull stalls until it's cancelled."""

    BAD = "ghcr.io/team/bad:v1"

    def __init__(self):
        super().__init__()
        self.cancelled: list[str] = []

    async def exec_script(self, script, *, max_retries=0):
        if script == f"docker pull {self.BAD}":
            await asyncio.sleep(0)
            raise RuntimeError("Script failed (exit 1):\nstderr: manifest unknown")
        if script.startswith("docker pull"):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(script)
                raise
        return ""


@pytest.mark.asyncio
async def test_the_first_pull_to_fail_cancels_the_rest_so_a_stalled_one_cant_hold_it_back(registry):
    sandbox = _OnePullFailsVm()

    with pytest.raises(RuntimeError, match="manifest unknown"):
        await sandbox.pull_images([PUBLIC, _OnePullFailsVm.BAD, PRIVATE])

    assert sorted(sandbox.cancelled) == sorted([f"docker pull {PUBLIC}", f"docker pull {PRIVATE}"])


class _ScriptRecorder(VmSandbox):
    """Records exec_script invocations, running the real write_file_from_text."""

    def __init__(self):
        self.scripts: list[str] = []

    async def terminate(self) -> None:  # pragma: no cover
        pass

    async def exec(self, *command):  # pragma: no cover
        return None

    async def exec_with_output(self, *args):
        if args[:3] == ("sudo", "bash", "-c"):
            self.scripts.append(script_run(args))
        return 0, "", ""


def _b64_from_chunk_scripts(scripts: list[str], b64_path: str) -> str:
    """Reconstruct the .b64 file's contents from the recorded `printf '%s' <chunk> >> path` appends."""
    out = []
    for s in scripts:
        if s.startswith("printf '%s' ") and s.endswith(f">> {b64_path}"):
            chunk = s[len("printf '%s' "):].rsplit(">>", 1)[0].strip()
            if chunk.startswith("'") and chunk.endswith("'"):  # base64 has no quotes to escape
                chunk = chunk[1:-1]
            out.append(chunk)
    return "".join(out)


@pytest.mark.asyncio
async def test_write_file_from_text_small_uses_single_heredoc():
    """Content <= the chunk bound takes the unchanged fast path: one base64 heredoc, no chunking."""
    import base64

    sandbox = _ScriptRecorder()
    content = "hello world"
    await sandbox.write_file_from_text(content, "/work/out.txt")

    heredocs = [s for s in sandbox.scripts if "base64 -d <<'ENDB64'" in s]
    assert len(heredocs) == 1
    assert base64.b64encode(content.encode()).decode() in heredocs[0]
    # Fast path doesn't build the chunked .b64 staging file.
    assert not any("printf '%s'" in s for s in sandbox.scripts)


@pytest.mark.asyncio
async def test_write_file_from_text_large_chunks_reconstruct_exactly():
    """Content > the chunk bound is appended in bounded chunks that reconstruct the exact base64
    (the >128 KiB MAX_ARG_STRLEN E2BIG path that broke writing large trajectories into a sandbox)."""
    import base64

    sandbox = _ScriptRecorder()
    # 200 KiB of content → base64 ~273 KiB, forcing several chunk appends.
    content = "x" * (200 * 1024)
    expected_b64 = base64.b64encode(content.encode()).decode()
    await sandbox.write_file_from_text(content, "/work/big.json")

    # No single heredoc (would overflow one bash -c arg); a chunked staging file instead.
    assert not any("base64 -d <<'ENDB64'" in s for s in sandbox.scripts)
    b64_path = next(s.removeprefix(": > ") for s in sandbox.scripts if s.startswith(": > "))
    assert b64_path.startswith("/tmp/_wft_") and b64_path.endswith("_work_big.json.b64")
    appends = [s for s in sandbox.scripts if s.startswith("printf '%s' ")]
    expected_chunks = -(-len(expected_b64) // VmSandbox._WFT_CHUNK_BYTES)  # ceil
    assert len(appends) == expected_chunks
    # The reconstructed base64 must equal the original, byte-for-byte.
    assert _b64_from_chunk_scripts(sandbox.scripts, b64_path) == expected_b64
    # And it's decoded into the real path before the docker cp.
    assert any(f"base64 -d {b64_path} > " in s for s in sandbox.scripts)


class _CleanupFailingSandbox(VmSandbox):
    """Sandbox whose exec API blips (exit -1) only on the trailing `rm -f` cleanup."""

    def __init__(self, fail_on: str = "rm -f"):
        self.scripts: list[str] = []
        self._fail_on = fail_on

    async def terminate(self) -> None:  # pragma: no cover
        pass

    async def exec(self, *command):  # pragma: no cover
        return None

    async def exec_with_output(self, *args):
        if args[:3] == ("sudo", "bash", "-c"):
            script = script_run(args)
            self.scripts.append(script)
            if script.startswith(self._fail_on):
                return -1, "", ""
        return 0, "", ""


@pytest.mark.asyncio
async def test_write_file_from_object_survives_cleanup_blip(signing_store):
    """A transient exec failure on the trailing `rm -f` must not fail the write —
    the file is already delivered into the container."""
    sandbox = _CleanupFailingSandbox()
    await sandbox.write_file_from_object("s3://bucket/data.json", "/work/data.json")
    # Delivery happened, and the cleanup was attempted despite the blip.
    assert any("docker cp" in s for s in sandbox.scripts)
    assert any(s.startswith("rm -f") for s in sandbox.scripts)


@pytest.mark.asyncio
async def test_write_file_from_url_survives_cleanup_blip():
    sandbox = _CleanupFailingSandbox()
    await sandbox.write_file_from_url("https://example.com/a.pdf", "/work/a.pdf")
    assert any("docker cp" in s for s in sandbox.scripts)
    assert any(s.startswith("rm -f") for s in sandbox.scripts)


@pytest.mark.asyncio
async def test_write_file_from_text_survives_cleanup_blip():
    sandbox = _CleanupFailingSandbox()
    await sandbox.write_file_from_text("hello", "/work/note.txt")
    assert any("docker cp" in s for s in sandbox.scripts)
    assert any(s.startswith("rm -f") for s in sandbox.scripts)


@pytest.mark.asyncio
async def test_write_file_from_object_still_fails_on_delivery_error(signing_store):
    """Failures on the actual delivery (docker cp) must still raise."""
    sandbox = _CleanupFailingSandbox(fail_on="docker cp")
    with pytest.raises(RuntimeError):
        await sandbox.write_file_from_object("s3://bucket/data.json", "/work/data.json")


@pytest.mark.asyncio
async def test_load_docker_images_streams_through_when_unsigned():
    """A non-signable backend (signed_get_url -> None, e.g. local FS) streams the bytes
    through agent-env — base64 push, no presigned curl."""

    class _LocalStore:
        def signed_get_url(self, object_url, expires_in=3600):
            return None

        def open(self, object_url):
            return io.BytesIO(b"IMGBYTES")

    set_object_store(_LocalStore())
    try:
        sandbox = _PushTargetVmSandbox(b"IMGBYTES", images_stdout="myimage\n")
        artifact = _image("file:///store/img.tar.gz", "myimage:latest")
        await sandbox.load_docker_images([artifact])
    finally:
        reset_config()

    assert not any("curl" in s for s in sandbox.scripts)      # no presigned download
    assert any("base64 -d" in s for s in sandbox.scripts)     # bytes streamed via base64
    assert any("docker load" in s for s in sandbox.scripts)


class _LoopCheckingStore:
    """Records, for each call, whether it ran on the event loop's thread."""

    def __init__(self, signs: bool) -> None:
        self.signs = signs
        self.on_loop: list[bool] = []

    def _record(self) -> None:
        try:
            asyncio.get_running_loop()
            self.on_loop.append(True)
        except RuntimeError:
            self.on_loop.append(False)

    def signed_get_url(self, object_url, expires_in=3600):
        self._record()
        return f"https://signed/{object_url.rsplit('/', 1)[-1]}" if self.signs else None

    def open(self, object_url):
        self._record()
        return io.BytesIO(b"IMG")


@pytest.mark.asyncio
@pytest.mark.parametrize("signs", [True, False], ids=["signing", "not-signing"])
async def test_the_store_is_called_off_the_event_loop(signs):
    """A remote store signs, and reads, over the network."""
    store = _LoopCheckingStore(signs)
    set_object_store(store)
    try:
        sandbox = _PushTargetVmSandbox(b"IMG", images_stdout="a\nb\n")
        artifacts = [_image(f"s3://bucket/{n}.tar.gz", f"{n}:1") for n in "ab"]
        await sandbox.load_docker_images(artifacts)
        await sandbox.load_object_file("s3://bucket/data.json", "/tmp/data.json")
    finally:
        reset_config()
    assert store.on_loop and not any(store.on_loop)


_WRITES = {
    "text": lambda sandbox, dest: sandbox.write_file_from_text("body", dest),
    "object": lambda sandbox, dest: sandbox.write_file_from_object("s3://bucket/body.json", dest),
    "url": lambda sandbox, dest: sandbox.write_file_from_url("https://example.com/body.json", dest),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("write", _WRITES.values(), ids=_WRITES.keys())
async def test_writes_to_one_destination_stage_at_different_host_paths(signing_store, write):
    """Local sandboxes share the host's /tmp, so concurrent writes of the same path must not meet."""
    sandbox = _ScriptRecorder()
    await asyncio.gather(*(write(sandbox, "/tmp/prompt_trajectory.json") for _ in range(2)))
    staged = {s.split(" ")[2] for s in sandbox.scripts if s.startswith("docker cp ")}
    assert len(staged) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("write", _WRITES.values(), ids=_WRITES.keys())
async def test_a_failed_write_removes_what_it_staged(signing_store, write):
    sandbox = _CleanupFailingSandbox(fail_on="docker cp")
    with pytest.raises(RuntimeError):
        await write(sandbox, "/work/out.json")
    staged = next(s.split(" ")[2] for s in sandbox.scripts if s.startswith("docker cp "))
    assert any(s.startswith("rm -f ") and staged in s for s in sandbox.scripts)


class _FailingSigner:
    def __init__(self) -> None:
        self.calls = 0

    def signed_get_url(self, object_url, expires_in=3600):
        self.calls += 1
        time.sleep(0.02)
        raise ConnectionError("signer unreachable")


@pytest.mark.asyncio
async def test_a_failed_sign_stops_the_signs_still_waiting():
    """Each waiting sign would otherwise take a worker thread for a whole retry window."""
    store = _FailingSigner()
    set_object_store(store)
    try:
        artifacts = [_image(f"s3://bucket/{n}.tar.gz", f"{n}:1") for n in range(20)]
        with pytest.raises(ConnectionError):
            await _RecordingVmSandbox().load_docker_images(artifacts)
        await asyncio.sleep(0.2)
    finally:
        reset_config()
    assert store.calls <= sandbox_module._CONCURRENT_SIGNS


class _CountingStore:
    """Signs slowly, recording how many signs run at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running = self.most = 0

    def signed_get_url(self, object_url, expires_in=3600):
        with self._lock:
            self.running += 1
            self.most = max(self.most, self.running)
        time.sleep(0.02)
        with self._lock:
            self.running -= 1
        return f"https://signed/{object_url.rsplit('/', 1)[-1]}"


@pytest.mark.asyncio
async def test_a_load_bounds_how_many_signs_run_at_once():
    """Each sign holds a worker thread, for a remote signer's whole retry window when it is down."""
    store = _CountingStore()
    set_object_store(store)
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=32))
    try:
        names = [f"i{n}" for n in range(20)]
        sandbox = _RecordingVmSandbox(images_stdout="\n".join(names))
        await sandbox.load_docker_images(
            [_image(f"s3://bucket/{n}.tar.gz", f"{n}:1") for n in names]
        )
    finally:
        reset_config()
    assert 1 < store.most <= sandbox_module._CONCURRENT_SIGNS


@pytest.mark.asyncio
async def test_write_host_file_writes_on_the_host_in_bounded_chunks():
    import base64

    sandbox = _ScriptRecorder()
    data = b"y" * (200 * 1024)
    await sandbox.write_host_file(data, "/tmp/agentenv_run_code/input.json")

    assert sandbox.scripts[0] == "mkdir -p /tmp/agentenv_run_code"
    assert not any("docker" in s for s in sandbox.scripts)
    b64_path = "/tmp/agentenv_run_code/input.json.b64"
    assert _b64_from_chunk_scripts(sandbox.scripts, b64_path) == base64.b64encode(data).decode()
    assert sandbox.scripts[-1] == f"base64 -d {b64_path} > /tmp/agentenv_run_code/input.json && rm -f {b64_path}"


class _ArgsRecorder(VmSandbox):
    def __init__(self, exit_code: int = 0):
        self.calls: list[tuple[str, ...]] = []
        self._exit_code = exit_code

    async def terminate(self) -> None:  # pragma: no cover
        pass

    async def exec(self, *command):  # pragma: no cover
        return None

    async def exec_with_output(self, *args):
        self.calls.append(args)
        return self._exit_code, "", "no such container"


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_source, script", [
    (False, 'docker cp "$1" "$2"'),
    (True, 'docker cp "$1" "$2" && rm -f "$1"'),
])
async def test_docker_cp_passes_its_paths_as_arguments(remove_source, script):
    sandbox = _ArgsRecorder()

    await sandbox.docker_cp("/tmp/a b", "c:/app/x", remove_source=remove_source)

    assert sandbox.calls == [("sudo", "bash", "-c", script, "docker-cp", "/tmp/a b", "c:/app/x")]


@pytest.mark.asyncio
async def test_a_failed_docker_cp_raises_with_its_stderr():
    with pytest.raises(RuntimeError, match="(?s)exit 1.*no such container"):
        await _ArgsRecorder(exit_code=1).docker_cp("/tmp/a", "c:/x")


@pytest.mark.asyncio
async def test_a_url_the_local_store_holds_is_read_from_it_not_the_configured_store(cli_routing, monkeypatch):
    """Under namespace routing, outside an @local run, the configured store owns only its own urls."""
    set_object_store(ConfiguredObjectStore())
    local = get_config().get_object_store_for("@local/~/t")
    data = local.put("objects/data.json", b"LOCAL-DATA")
    image = local.put("images/img.tar.gz", b"LOCAL-IMAGE")
    pushed = []

    async def push(sandbox, store, object_url, vm_path):
        pushed.append((store, object_url))

    monkeypatch.setattr(sandbox_module, "push_object_over_exec", push)
    sandbox = _RecordingVmSandbox(images_stdout="myimage\n")

    await sandbox.load_object_file(data, "/tmp/data.json")
    await sandbox.load_docker_images([_image(image, "myimage:latest")])

    assert pushed == [(local, data), (local, image)]
