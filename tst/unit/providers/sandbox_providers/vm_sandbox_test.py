import asyncio
import hashlib
import io
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_env.providers.sandbox_providers import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.store import set_object_store
from agent_env.config import get_config, reset_config
from tst.unit.store.fakes import ConfiguredObjectStore
from tst.util.exec_scripts import script_run


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

    def __init__(self):
        self.sandbox_id = "vm-test"
        self.scripts: list[str] = []

    async def terminate(self) -> None:  # pragma: no cover - not exercised
        pass

    async def exec(self, *command):  # pragma: no cover - not exercised
        return None

    async def exec_with_output(self, *args):
        # Image loading verifies exact refs through `docker image inspect`.
        if args[:4] == ("sudo", "docker", "image", "inspect"):
            return 0, "sha256:image", ""
        if args[:2] == ("sudo", "bash"):
            self.scripts.append(script_run(args))
        return 0, "", ""

    async def write_file_from_text(self, content, destination_path):  # pragma: no cover
        pass


class _PushTargetVmSandbox(_RecordingVmSandbox):
    """Answers a push's sha256 check with the digest of ``pushed``, as a VM that received all of it would."""

    def __init__(self, pushed: bytes):
        super().__init__()
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
    sandbox = _RecordingVmSandbox()
    artifact = SimpleNamespace(tar_gz_object_url="s3://bucket/img.tar.gz", image_name="myimage:latest")

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
    assert "set -o pipefail" in load_script
    assert "wait $pid_0 || status=1" in load_script
    assert "trap " not in load_script


@pytest.mark.asyncio
async def test_load_docker_images_uses_unique_tmp_per_artifact(signing_store):
    sandbox = _RecordingVmSandbox()
    artifacts = [
        SimpleNamespace(tar_gz_object_url="s3://bucket/a.tar.gz", image_name="a:1"),
        SimpleNamespace(tar_gz_object_url="s3://bucket/b.tar.gz", image_name="b:1"),
    ]

    await sandbox.load_docker_images(artifacts)

    load_script = next(s for s in sandbox.scripts if "docker load" in s)
    assert "/tmp/_docker_image_vm-test_0.tar.gz" in load_script
    assert "/tmp/_docker_image_vm-test_1.tar.gz" in load_script


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_stage", "expected_loads"),
    [("docker", ["fail", "ok"]), ("gunzip", ["ok", "ok"])],
)
async def test_load_image_worker_failure_propagates_and_cleans_staging(
    tmp_path, failure_stage, expected_loads,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker-load.log"
    _install_docker_load_stubs(
        bin_dir,
        fail_first=failure_stage == "docker",
        gunzip_fail_first=failure_stage == "gunzip",
    )
    sandbox = _ShellVmSandbox(bin_dir, docker_log, f"lifecycle-{tmp_path.name}")
    images = [
        SimpleNamespace(tar_gz_object_url=f"s3://bucket/{i}", image_name=f"repo:{i}")
        for i in range(2)
    ]
    with pytest.raises(RuntimeError, match=r"Script failed \(exit 1\)"):
        await sandbox._load_docker_images(images, ["https://signed/a", "https://signed/b"])
    assert sorted(docker_log.read_text().splitlines()) == expected_loads
    assert all(
        not Path(f"/tmp/_docker_image_{sandbox.sandbox_id}_{i}.tar.gz").exists()
        for i in range(2)
    )
    assert len([script for script in sandbox.scripts if "docker load" in script]) == 1
    assert not any("image" in call and "inspect" in call for call in sandbox.verify_calls)


@pytest.mark.asyncio
async def test_load_image_success_runs_every_worker_and_verifies_refs(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker-load.log"
    _install_docker_load_stubs(bin_dir)

    sandbox = _ShellVmSandbox(bin_dir, docker_log, f"lifecycle-{tmp_path.name}")
    images = [
        SimpleNamespace(tar_gz_object_url=f"s3://bucket/{i}", image_name=f"repo:{i}")
        for i in range(2)
    ]
    await sandbox._load_docker_images(images, ["https://signed/a", "https://signed/b"])
    assert sorted(docker_log.read_text().splitlines()) == ["ok", "ok"]
    assert sandbox.verify_calls == [(
        "sudo", "docker", "image", "inspect", "--format", "{{.Id}}", "repo:0", "repo:1",
    )]


@pytest.mark.asyncio
@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
async def test_load_image_retries_after_transient_exec_result_without_losing_archive(
    tmp_path, monkeypatch, signed,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker-load.log"
    _install_docker_load_stubs(bin_dir)
    sandbox = _ShellVmSandbox(bin_dir, docker_log, f"lifecycle-{tmp_path.name}")
    sandbox.transient_results = 1
    artifacts = [
        SimpleNamespace(tar_gz_object_url=f"s3://bucket/{idx}", image_name=f"repo:{idx}")
        for idx in range(2)
    ]

    if not signed:
        downloaded = []

        async def save_unsigned(_object_url, vm_path):
            downloaded.append(vm_path)
            Path(vm_path).write_text("ok")

        sandbox._download_object_to_vm = save_unsigned
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(sandbox_module.asyncio, "sleep", no_sleep)
    await sandbox._load_docker_images(
        artifacts,
        ["https://signed/a", "https://signed/b"] if signed else [None, None],
    )

    assert sorted(docker_log.read_text().splitlines()) == ["ok"] * 4
    assert len([script for script in sandbox.scripts if "docker load" in script]) == 2
    assert not any(Path(path).exists() for path in (
        f"/tmp/_docker_image_{sandbox.sandbox_id}_{idx}.tar.gz" for idx in range(2)
    ))
    if not signed:
        assert len(downloaded) == 2


class _ShellVmSandbox(_RecordingVmSandbox):
    def __init__(self, bin_dir, docker_log, sandbox_id):
        super().__init__()
        self.bin_dir = bin_dir
        self.docker_log = docker_log
        self.sandbox_id = sandbox_id
        self.verify_calls = []
        self.transient_results = 0

    async def exec_with_output(self, *args):
        if args[:4] == ("sudo", "docker", "image", "inspect"):
            self.verify_calls.append(args)
        if args[:3] == ("sudo", "bash", "-c"):
            script = args[3]
            self.scripts.append(script)
            env = dict(
                os.environ,
                PATH=f"{self.bin_dir}:{os.environ['PATH']}",
                DOCKER_LOG=str(self.docker_log),
            )
            result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
            if self.transient_results:
                self.transient_results -= 1
                return -1, result.stdout, result.stderr
            return result.returncode, result.stdout, result.stderr
        return await super().exec_with_output(*args)


def _install_docker_load_stubs(bin_dir, *, fail_first=False, gunzip_fail_first=False):
    curl_result = (
        'case "$url" in */a) printf fail > "$1" ;; *) printf ok > "$1" ;; esac\n'
        if fail_first else 'printf ok > "$1"\n'
    )
    gunzip_script = '#!/bin/sh\ncat "$2"\n'
    if gunzip_fail_first:
        gunzip_script = '#!/bin/sh\ncat "$2"\ncase "$2" in *_0.tar.gz) exit 24 ;; esac\n'
    for name, body in {
        "curl": (
            '#!/bin/sh\nurl=\nwhile [ "$1" != "-o" ]; do url=$1; shift; done\n'
            "shift\n" + curl_result
        ),
        "gunzip": gunzip_script,
        "docker": '#!/bin/sh\npayload=$(cat)\nprintf "%s\\n" "$payload" >> "$DOCKER_LOG"\n'
        '[ "$payload" = fail ] && exit 23\nexit 0\n',
    }.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)


@pytest.mark.asyncio
async def test_load_image_requires_exact_reference_even_when_old_tag_is_listed(signing_store):
    class StaleTagVm(_RecordingVmSandbox):
        async def exec_with_output(self, *args):
            if args[:4] == ("sudo", "docker", "image", "inspect"):
                existing_refs = {"registry.example:5000/team/repo:old-tag"}
                requested_refs = set(args[6:])
                if requested_refs <= existing_refs:
                    return 0, "sha256:old", ""
                return 1, "", "No such image"
            return await super().exec_with_output(*args)

    sandbox = StaleTagVm()
    image = SimpleNamespace(
        tar_gz_object_url="s3://bucket/img",
        image_name="registry.example:5000/team/repo:new-tag",
    )
    with pytest.raises(RuntimeError, match="registry.example:5000/team/repo:new-tag"):
        await sandbox.load_docker_images([image])


@pytest.mark.asyncio
async def test_unsigned_download_failure_cleans_staging_file(signing_store, tmp_path):
    cleaned = []

    class UnsignedStore(_SigningStore):
        def signed_get_url(self, object_url, expires_in=3600):
            return None

    set_object_store(UnsignedStore())

    class FailedDownloadVm(_RecordingVmSandbox):
        async def _download_object_to_vm(self, _object_url, vm_path):
            Path(vm_path).write_text("partial")
            raise RuntimeError("download failed")

        async def _remove_vm_temp_file(self, *vm_paths):
            cleaned.extend(vm_paths)
            for vm_path in vm_paths:
                Path(vm_path).unlink(missing_ok=True)

    sandbox = FailedDownloadVm()
    sandbox.sandbox_id = f"lifecycle-{tmp_path.name}"
    artifact = SimpleNamespace(tar_gz_object_url="obj://image", image_name="repo:tag")
    with pytest.raises(RuntimeError, match="download failed"):
        await sandbox._load_docker_images([artifact], [None])
    assert cleaned == [f"/tmp/_docker_image_{sandbox.sandbox_id}_0.tar.gz"]
    assert not Path(cleaned[0]).exists()


@pytest.mark.asyncio
async def test_failed_chunk_write_removes_base64_staging_file(tmp_path):
    sandbox = _ChunkWriteSandbox()
    destination = tmp_path / "payload.bin"
    staging = Path(f"{destination}.b64")
    with pytest.raises(RuntimeError, match=r"Script failed \(exit 17\)"):
        await sandbox._write_bytes_to_vm_path(b"x" * (VmSandbox._WFT_CHUNK_BYTES + 1), str(destination))
    assert not staging.exists()
    assert sandbox.scripts[-1] == f"rm -f {staging}"


@pytest.mark.asyncio
async def test_cancelled_chunk_write_removes_base64_staging_file(tmp_path):
    sandbox = _ChunkWriteSandbox(cancel_chunk=True)
    destination = tmp_path / "payload.bin"
    staging = Path(f"{destination}.b64")
    write = asyncio.create_task(
        sandbox._write_bytes_to_vm_path(b"x" * (VmSandbox._WFT_CHUNK_BYTES + 1), str(destination))
    )
    await asyncio.wait_for(sandbox.chunk_started.wait(), 1)
    write.cancel()
    with pytest.raises(asyncio.CancelledError):
        await write
    assert not staging.exists()
    assert sandbox.scripts[-1] == f"rm -f {staging}"


class _ChunkWriteSandbox(VmSandbox):
    def __init__(self, *, cancel_chunk=False):
        self.scripts = []
        self.cancel_chunk = cancel_chunk
        self.chunk_started = asyncio.Event()

    async def terminate(self):  # pragma: no cover
        pass

    async def exec(self, *command):  # pragma: no cover
        return None

    async def exec_with_output(self, *args):
        script = script_run(args)
        self.scripts.append(script)
        if script.startswith("printf '%s'"):
            if self.cancel_chunk:
                self.chunk_started.set()
                await asyncio.Future()
            return 17, "", "chunk write failed"
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr


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
        sandbox = _PushTargetVmSandbox(b"IMGBYTES")
        artifact = SimpleNamespace(tar_gz_object_url="file:///store/img.tar.gz", image_name="myimage:latest")
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
        sandbox = _PushTargetVmSandbox(b"IMG")
        artifacts = [SimpleNamespace(tar_gz_object_url=f"s3://bucket/{n}.tar.gz", image_name=f"{n}:1") for n in "ab"]
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
        artifacts = [SimpleNamespace(tar_gz_object_url=f"s3://bucket/{n}.tar.gz", image_name=f"{n}:1") for n in range(20)]
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
        sandbox = _RecordingVmSandbox()
        await sandbox.load_docker_images(
            [SimpleNamespace(tar_gz_object_url=f"s3://bucket/{n}.tar.gz", image_name=f"{n}:1") for n in names]
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
    assert f"base64 -d {b64_path} > /tmp/agentenv_run_code/input.json" in sandbox.scripts
    assert sandbox.scripts[-1] == f"rm -f {b64_path}"


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
async def test_the_s3_named_methods_are_deprecated_aliases(signing_store, caplog):
    """They delegate, and each use is counted: the log line, not the warning, is what reaches production logs."""
    sandbox = _RecordingVmSandbox()
    with caplog.at_level(logging.WARNING, logger="agent_env.utils.deprecation"):
        with pytest.warns(DeprecationWarning, match="use load_object_file"):
            await sandbox.load_s3_file("s3://bucket/a.json", "/tmp/a.json")
        with pytest.warns(DeprecationWarning, match="use write_file_from_object"):
            await sandbox.write_file_from_s3("s3://bucket/b.json", "/work/b.json")
    assert any("https://signed/a.json" in s for s in sandbox.scripts)
    assert any("https://signed/b.json" in s for s in sandbox.scripts)
    counted = [r.deprecated_symbol for r in caplog.records if getattr(r, "event", None) == "agent_env_deprecated_symbol"]
    assert counted == ["VmSandbox.load_s3_file", "Sandbox.write_file_from_s3"]


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
    sandbox = _RecordingVmSandbox()

    await sandbox.load_object_file(data, "/tmp/data.json")
    await sandbox.load_docker_images([SimpleNamespace(tar_gz_object_url=image, image_name="myimage:latest")])

    assert pushed == [(local, data), (local, image)]
