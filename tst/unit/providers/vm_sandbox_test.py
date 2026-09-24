from types import SimpleNamespace

import pytest

from agent_env.providers.sandbox import VmSandbox
from agent_env.store import set_object_store
from agent_env.config import reset_config


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
            self.scripts.append(args[-1])
        return 0, "", ""

    async def write_file_from_text(self, content, destination_path):  # pragma: no cover
        pass


@pytest.mark.asyncio
async def test_load_docker_images_downloads_to_file_before_load(signing_store):
    """The docker-load path must download to a temp file before `gunzip | docker load`.

    Retrying `curl ... | gunzip | docker load` directly corrupts the stream
    because curl can't rewind bytes already piped to stdout, so the retry flags
    must only apply to a `-o file` download (Greptile P1 on PR #430).
    """
    sandbox = _RecordingVmSandbox(images_stdout="myimage\n")
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


@pytest.mark.asyncio
async def test_load_docker_images_uses_unique_tmp_per_artifact(signing_store):
    sandbox = _RecordingVmSandbox(images_stdout="a\nb\n")
    artifacts = [
        SimpleNamespace(tar_gz_object_url="s3://bucket/a.tar.gz", image_name="a:1"),
        SimpleNamespace(tar_gz_object_url="s3://bucket/b.tar.gz", image_name="b:1"),
    ]

    await sandbox.load_docker_images(artifacts)

    load_script = next(s for s in sandbox.scripts if "docker load" in s)
    assert "/tmp/_docker_image_vm-test_0.tar.gz" in load_script
    assert "/tmp/_docker_image_vm-test_1.tar.gz" in load_script


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
            self.scripts.append(args[-1])
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
    (the >128 KiB MAX_ARG_STRLEN E2BIG path that broke large iOS trajectories)."""
    import base64

    sandbox = _ScriptRecorder()
    # 200 KiB of content → base64 ~273 KiB, forcing several chunk appends.
    content = "x" * (200 * 1024)
    expected_b64 = base64.b64encode(content.encode()).decode()
    await sandbox.write_file_from_text(content, "/work/big.json")

    # No single heredoc (would overflow one bash -c arg); a chunked staging file instead.
    assert not any("base64 -d <<'ENDB64'" in s for s in sandbox.scripts)
    b64_path = "/tmp/_wft_work_big.json.b64"
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
            script = args[-1]
            self.scripts.append(script)
            if script.startswith(self._fail_on):
                return -1, "", ""
        return 0, "", ""


@pytest.mark.asyncio
async def test_write_file_from_s3_survives_cleanup_blip(signing_store):
    """A transient exec failure on the trailing `rm -f` must not fail the write —
    the file is already delivered into the container."""
    sandbox = _CleanupFailingSandbox()
    await sandbox.write_file_from_s3("s3://bucket/data.json", "/work/data.json")
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
async def test_write_file_from_s3_still_fails_on_delivery_error(signing_store):
    """Failures on the actual delivery (docker cp) must still raise."""
    sandbox = _CleanupFailingSandbox(fail_on="docker cp")
    with pytest.raises(RuntimeError):
        await sandbox.write_file_from_s3("s3://bucket/data.json", "/work/data.json")


@pytest.mark.asyncio
async def test_load_docker_images_streams_through_when_unsigned():
    """A non-signable backend (signed_get_url -> None, e.g. local FS) streams the bytes
    through agent-env — base64 push, no presigned curl."""

    class _LocalStore:
        def signed_get_url(self, object_url, expires_in=3600):
            return None

        def get(self, object_url):
            return b"IMGBYTES"

    set_object_store(_LocalStore())
    try:
        sandbox = _RecordingVmSandbox(images_stdout="myimage\n")
        artifact = SimpleNamespace(tar_gz_object_url="file:///store/img.tar.gz", image_name="myimage:latest")
        await sandbox.load_docker_images([artifact])
    finally:
        reset_config()

    assert not any("curl" in s for s in sandbox.scripts)      # no presigned download
    assert any("base64 -d" in s for s in sandbox.scripts)     # bytes streamed via base64
    assert any("docker load" in s for s in sandbox.scripts)
