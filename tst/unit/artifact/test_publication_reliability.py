import gzip
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent_env.artifact.artifacts import docker_image as docker_image_module
from agent_env.artifact.artifacts.docker_image import _save_image_tar_gz
from agent_env.artifact.artifacts.file import FileArtifact

_SAVE_TIMEOUT_SECONDS = 0.25
_SAVE_SUCCESS_TIMEOUT_SECONDS = 30
_SAVE_FAILURE_TIMEOUT_SECONDS = 3
_FAKE_DOCKER_STDERR_BYTES = 1024 * 1024
_LARGE_STDERR_BYTES = 2 * 1024 * 1024
_STREAM_TEST_MIB = 16
_MIB = 1024 * 1024
_WATCHDOG_TEST_SECONDS = 2
_RETRY_PAYLOAD_MULTIPLIER = 1000
_DESCENDANT_MARKER_DELAY_SECONDS = 0.5
_DESCENDANT_SETTLE_SECONDS = 0.6
_LEADER_EXIT_POLL_SECONDS = 0.01
_MAX_DOCKER_ERROR_CHARS = 20 * 1024
_DESCENDANT_RELEASE_SECONDS = 3
_DESCENDANT_AUTO_EXIT_SECONDS = 5
_ESCAPED_DESCENDANT_TIMEOUT_SECONDS = 1.5
_ESCAPED_DESCENDANT_TEST_TIMEOUT_SECONDS = 4


def _install_fake_docker(tmp_path: Path, monkeypatch, body: str) -> None:
    executable = tmp_path / "docker"
    executable.write_text(f"#!/usr/bin/env python3\n{body}\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def detached_child_release(tmp_path, request):
    release = tmp_path / "release-descendant"
    request.addfinalizer(lambda: release.touch())
    return release


@pytest.mark.parametrize("api", ["path", "bytes"])
def test_file_artifact_retry_after_document_failure_isolated(local_stores, tmp_path, api):
    document_store = local_stores.get_document_store()
    original_insert = document_store.insert
    failed = False

    def insert(collection, document):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("temporary document failure")
        return original_insert(collection, document)

    document_store.insert = insert
    payload = b"retry payload"
    source = tmp_path / "payload.bin"
    source.write_bytes(payload)

    def publish():
        if api == "path":
            return FileArtifact.put("retry-file", description="retry", file_path=str(source))
        return FileArtifact.put_bytes(
            "retry-file", description="retry", filename=source.name, content=payload
        )

    with pytest.raises(RuntimeError, match="temporary document failure"):
        publish()
    artifact = publish()

    assert artifact.version == 1
    assert artifact.load() == payload
    objects = local_stores.get_object_store()
    keys = objects.list("artifacts/file/")
    assert len(keys) == 2
    assert len(set(keys)) == 2


def test_concurrent_file_artifacts_keep_each_payload(local_stores):
    payloads = [b"first" * _RETRY_PAYLOAD_MULTIPLIER, b"second" * _RETRY_PAYLOAD_MULTIPLIER]

    def publish(index):
        return FileArtifact.put_bytes(
            "concurrent-file", description=str(index), filename="data.bin", content=payloads[index]
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        artifacts = list(pool.map(publish, range(2)))

    assert sorted(artifact.version for artifact in artifacts) == [1, 2]
    assert {artifact.load() for artifact in artifacts} == set(payloads)


@pytest.mark.parametrize("write_stderr", [False, True])
def test_docker_save_timeout_covers_stuck_stdout(local_stores, tmp_path, monkeypatch, write_stderr):
    stderr_write = f"os.write(2, b'x' * {_FAKE_DOCKER_STDERR_BYTES})\n" if write_stderr else ""
    _install_fake_docker(
        tmp_path,
        monkeypatch,
        f"import os, time\nos.write(1, b'partial archive')\n{stderr_write}time.sleep(30)",
    )
    archive = tmp_path / "image.tar.gz"
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="timed out"):
        _save_image_tar_gz("example:latest", archive, _SAVE_TIMEOUT_SECONDS)

    assert time.monotonic() - started < _WATCHDOG_TEST_SECONDS
    assert not archive.exists()


def test_docker_save_timeout_kills_descendant_holding_stdout(local_stores, tmp_path, monkeypatch):
    child_script = "import time; time.sleep(30)"
    _install_fake_docker(
        tmp_path,
        monkeypatch,
        f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', {child_script!r}])",
    )
    archive = tmp_path / "image.tar.gz"
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="timed out"):
        _save_image_tar_gz("example:latest", archive, _SAVE_TIMEOUT_SECONDS)

    assert time.monotonic() - started < _WATCHDOG_TEST_SECONDS
    assert not archive.exists()


@pytest.mark.parametrize("held_stream", ["stdout", "stderr"])
@pytest.mark.timeout(_ESCAPED_DESCENDANT_TEST_TIMEOUT_SECONDS)
def test_docker_save_timeout_returns_when_escaped_descendant_holds_pipe(
    local_stores, tmp_path, monkeypatch, detached_child_release, held_stream
):
    child_pid_file = tmp_path / "detached-child.pid"
    child_exit_file = tmp_path / "detached-child-exited"
    child_script = (
        "import pathlib, sys, time\n"
        f"release = pathlib.Path({str(detached_child_release)!r})\n"
        f"deadline = time.monotonic() + {_DESCENDANT_AUTO_EXIT_SECONDS}\n"
        f"while not release.exists() and time.monotonic() < deadline: time.sleep({_LEADER_EXIT_POLL_SECONDS})\n"
        f"pathlib.Path({str(child_exit_file)!r}).touch()\n"
    )
    pipe_args = "stdout=None, stderr=subprocess.DEVNULL" if held_stream == "stdout" else "stdout=subprocess.DEVNULL, stderr=None"
    _install_fake_docker(
        tmp_path,
        monkeypatch,
        "import pathlib, subprocess, sys\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_script!r}], "
        f"start_new_session=True, {pipe_args})\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))",
    )
    archive = tmp_path / "image.tar.gz"
    thread_ids_before = {thread.ident for thread in threading.enumerate()}
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="timed out"):
        _save_image_tar_gz("example:latest", archive, _ESCAPED_DESCENDANT_TIMEOUT_SECONDS)

    assert time.monotonic() - started < _ESCAPED_DESCENDANT_TEST_TIMEOUT_SECONDS
    assert child_pid_file.exists()
    assert not archive.exists()
    detached_child_release.touch()
    exit_deadline = time.monotonic() + _DESCENDANT_RELEASE_SECONDS
    while not child_exit_file.exists() and time.monotonic() < exit_deadline:
        time.sleep(_LEADER_EXIT_POLL_SECONDS)
    assert child_exit_file.exists()
    assert {thread.ident for thread in threading.enumerate()} <= thread_ids_before


def test_docker_save_failure_removes_partial_archive(local_stores, tmp_path, monkeypatch):
    _install_fake_docker(
        tmp_path,
        monkeypatch,
        "import os\nos.write(1, b'partial archive')\nos.write(2, b'failure details')\nraise SystemExit(7)",
    )
    archive = tmp_path / "image.tar.gz"

    with pytest.raises(RuntimeError, match="failure details"):
        _save_image_tar_gz("example:latest", archive, _SAVE_FAILURE_TIMEOUT_SECONDS)

    assert not archive.exists()


def test_docker_save_bounds_captured_stderr_for_large_failure(local_stores, tmp_path, monkeypatch):
    _install_fake_docker(
        tmp_path,
        monkeypatch,
        "import os\n"
        "os.write(2, b'begin')\n"
        f"os.write(2, b'x' * {_LARGE_STDERR_BYTES})\n"
        "os.write(2, b'end-marker')\n"
        "raise SystemExit(7)",
    )
    archive = tmp_path / "image.tar.gz"

    with pytest.raises(RuntimeError, match="stderr truncated") as raised:
        _save_image_tar_gz("example:latest", archive, _SAVE_SUCCESS_TIMEOUT_SECONDS)

    assert "end-marker" in str(raised.value)
    assert "begin" not in str(raised.value)
    assert len(str(raised.value)) < _MAX_DOCKER_ERROR_CHARS
    assert not archive.exists()


def test_docker_save_does_not_signal_group_after_leader_is_reaped(local_stores, tmp_path, monkeypatch):
    _install_fake_docker(tmp_path, monkeypatch, "raise SystemExit(7)")
    archive = tmp_path / "image.tar.gz"
    signaled_groups = []
    real_killpg = docker_image_module.os.killpg

    def track_killpg(group_id, signal_number):
        signaled_groups.append(group_id)
        return real_killpg(group_id, signal_number)

    monkeypatch.setattr(docker_image_module.os, "killpg", track_killpg)
    with pytest.raises(RuntimeError, match="docker save"):
        _save_image_tar_gz("example:latest", archive, _SAVE_SUCCESS_TIMEOUT_SECONDS)

    assert signaled_groups == []
    assert not archive.exists()


def test_docker_save_compression_error_kills_child_and_cleans_archive(local_stores, tmp_path, monkeypatch):
    _install_fake_docker(tmp_path, monkeypatch, "import time\ntime.sleep(30)")
    archive = tmp_path / "image.tar.gz"

    def fail_compression(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(docker_image_module.gzip, "open", fail_compression)
    with pytest.raises(OSError, match="disk full"):
        _save_image_tar_gz("example:latest", archive, _SAVE_FAILURE_TIMEOUT_SECONDS)

    assert not archive.exists()


def test_docker_save_compression_error_kills_descendant_after_leader_exit(
    local_stores, tmp_path, monkeypatch
):
    leader_exit_marker = tmp_path / "leader-exited"
    survival_marker = tmp_path / "descendant-survived"
    child_script = (
        "import os, sys, time\n"
        "parent = int(sys.argv[1])\n"
        "while os.getppid() == parent: time.sleep(0.01)\n"
        "open(sys.argv[2], 'w').close()\n"
        f"time.sleep({_DESCENDANT_MARKER_DELAY_SECONDS})\n"
        "open(sys.argv[3], 'w').close()\n"
    )
    _install_fake_docker(
        tmp_path,
        monkeypatch,
        "import os, subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}, str(os.getpid()), "
        f"{str(leader_exit_marker)!r}, {str(survival_marker)!r}])",
    )
    archive = tmp_path / "image.tar.gz"

    def fail_after_leader_exit(*args, **kwargs):
        marker_deadline = time.monotonic() + _SAVE_FAILURE_TIMEOUT_SECONDS
        while not leader_exit_marker.exists() and time.monotonic() < marker_deadline:
            time.sleep(_LEADER_EXIT_POLL_SECONDS)
        assert leader_exit_marker.exists()
        raise OSError("disk full")

    monkeypatch.setattr(docker_image_module.gzip, "open", fail_after_leader_exit)
    with pytest.raises(OSError, match="disk full"):
        _save_image_tar_gz("example:latest", archive, _SAVE_FAILURE_TIMEOUT_SECONDS)

    time.sleep(_DESCENDANT_SETTLE_SECONDS)
    assert leader_exit_marker.exists()
    assert not survival_marker.exists()
    assert not archive.exists()


def test_docker_save_streams_large_payload_into_valid_gzip(local_stores, tmp_path, monkeypatch):
    _install_fake_docker(
        tmp_path,
        monkeypatch,
        f"import os\nchunk = b'z' * {_MIB}\nfor _ in range({_STREAM_TEST_MIB}): os.write(1, chunk)",
    )
    archive = tmp_path / "image.tar.gz"
    _save_image_tar_gz("example:latest", archive, _SAVE_SUCCESS_TIMEOUT_SECONDS)

    with gzip.open(archive, "rb") as compressed:
        payload = compressed.read()
    assert len(payload) == _STREAM_TEST_MIB * _MIB
    assert payload == b"z" * len(payload)


def test_docker_image_put_cleans_archive_after_save_failure(local_stores, tmp_path, monkeypatch):
    _install_fake_docker(tmp_path, monkeypatch, "raise SystemExit(7)")
    image_store = type("ImageStore", (), {
        "image_ref": lambda self, repository, tag: f"registry/{repository}:{tag}",
        "ensure_repository": lambda self, repository: None,
    })()
    monkeypatch.setattr(local_stores, "get_image_store_for", lambda artifact_id: image_store)
    monkeypatch.setattr(docker_image_module, "_push_local_image", lambda *args: None)
    temp_paths = []
    real_named_temp_file = docker_image_module.tempfile.NamedTemporaryFile

    def named_temp_file(*args, **kwargs):
        file = real_named_temp_file(*args, **kwargs)
        temp_paths.append(Path(file.name))
        return file

    monkeypatch.setattr(docker_image_module.tempfile, "NamedTemporaryFile", named_temp_file)

    with pytest.raises(RuntimeError, match="docker save"):
        docker_image_module.DockerImageArtifact.put(
            id="failed-image", description="failed", image_name="local:latest"
        )

    assert temp_paths and all(not path.exists() for path in temp_paths)
