"""``build_image``: the argv it runs, what a failed or impossible build raises, and that importing it pulls in
neither the CLI nor the providers, so library code can build too."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_env.utils import docker_build
from agent_env.utils.docker_build import DEFAULT_BUILD_PLATFORM, DockerBuildError, build_image


@pytest.fixture
def runs(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(docker_build.subprocess, "run", run)
    return calls


def test_the_default_platform_is_the_remote_sandboxes():
    assert DEFAULT_BUILD_PLATFORM == "linux/amd64"


@pytest.mark.parametrize("platform, flags", [
    ("linux/amd64", ["--platform", "linux/amd64"]),
    ("linux/arm64", ["--platform", "linux/arm64"]),
    ("", []),
    (None, []),
], ids=["amd64", "arm64", "empty-is-host-native", "none-is-host-native"])
def test_it_builds_the_context_with_the_dockerfile_into_the_tag(runs, platform, flags):
    build_image(Path("/ctx/Dockerfile"), Path("/ctx"), "my-image", platform=platform, build_args={"VERSION": "1.2"})

    (argv, kwargs), = runs
    assert argv == ["docker", "build", *flags, "--build-arg", "VERSION=1.2", "-f", "/ctx/Dockerfile", "-t", "my-image",
                    "/ctx"]
    assert kwargs["stderr"] is subprocess.STDOUT


def test_a_failed_build_raises_with_its_output(monkeypatch):
    output = "#5 [2/3] RUN pip install nope\n#5 ERROR: process did not complete\n"
    monkeypatch.setattr(docker_build.subprocess, "run", lambda argv, **_: SimpleNamespace(returncode=1, stdout=output))

    with pytest.raises(DockerBuildError) as failed:
        build_image(Path("Dockerfile"), Path("."), "my-image", platform=None)

    assert str(failed.value) == f"docker build of my-image failed (exit 1):\n{output.rstrip()}"


def test_the_error_carries_both_of_docker_s_streams_with_undecodable_bytes_replaced(tmp_path, monkeypatch):
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nprintf 'step 1\\n'\nprintf 'bad \\377 byte\\n' >&2\nexit 3\n")
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(DockerBuildError) as failed:
        build_image(Path("Dockerfile"), Path("."), "my-image", platform=None)

    assert str(failed.value) == "docker build of my-image failed (exit 3):\nstep 1\nbad � byte"


def test_a_missing_docker_raises_rather_than_crashing(monkeypatch):
    def run(argv, **_):
        raise FileNotFoundError(2, "No such file or directory", "docker")

    monkeypatch.setattr(docker_build.subprocess, "run", run)

    with pytest.raises(DockerBuildError, match="docker build of my-image failed: docker is not on PATH"):
        build_image(Path("Dockerfile"), Path("."), "my-image", platform=None)


def test_importing_it_pulls_neither_cli_nor_providers():
    code = (
        "import sys; import agent_env.utils.docker_build; "
        "assert 'agent_env.cli' not in sys.modules, 'cli imported'; "
        "assert 'agent_env.providers' not in sys.modules, 'providers imported'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
