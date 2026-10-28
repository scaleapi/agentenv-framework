"""The ubuntu-24.04 base computer, built from its folder once for the tests here and tagged for the local registry, and
``docker``, which runs the docker CLI for them."""

import subprocess
import uuid
from pathlib import Path

import pytest

import agent_env

BASE = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04"


@pytest.fixture(scope="session")
def base():
    tag = f"localhost:5000/agentenv-test-ubuntu-24.04:{uuid.uuid4().hex[:8]}"
    docker("build", "-q", "-t", tag, str(BASE))
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


def docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=True, capture_output=True)
