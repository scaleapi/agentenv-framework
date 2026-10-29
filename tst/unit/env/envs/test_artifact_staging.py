"""An env stages a loaded artifact's file through its sandbox's shell: a file or env name with shell syntax in it is
staged and copied as it is and runs nothing, and a file name that isn't one plain name is refused before anything runs.
A real LocalSandbox runs the scripts; docker is a stub that logs its arguments."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox

_SHELL_NAMES = [
    ("email", "x;touch pwned;#.json"),
    ("email", "$(touch pwned).json"),
    ("email", "`touch pwned`.json"),
    ("email", "x' ; touch pwned ; '.json"),
    ("email", "seed data.json"),
    ("e;touch pwned;#", "seed.json"),
]
_NOT_ONE_NAME = ["../seed.json", "a/b.json", "a\\b.json", "..", ".", "", "seed\n.json", "seed\x00.json"]


@pytest.fixture
def host(tmp_path, monkeypatch):
    log = tmp_path / "docker.log"
    stub = tmp_path / "bin" / "docker"
    stub.parent.mkdir()
    stub.write_text(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done >> {shlex.quote(str(log))}\necho --- >> {shlex.quote(str(log))}\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)
    sandbox = LocalSandbox(work_dir=tmp_path / "app")
    sandbox.work_dir.mkdir()

    async def load_object_file(object_url, path):
        Path(sandbox._rewrite_app_arg(path)).write_bytes(b"{}")

    sandbox.load_object_file = load_object_file

    def docker_calls() -> list[list[str]]:
        return [call.strip("\n").split("\n") for call in log.read_text().split("---\n")[:-1]] if log.exists() else []

    return SimpleNamespace(sandbox=sandbox, root=tmp_path, docker_calls=docker_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(("environment_name", "filename"), _SHELL_NAMES)
async def test_an_mcp_server_stages_a_shell_looking_name_as_it_is(host, environment_name, filename):
    env = _mcp(host, environment_name)

    assert await env._copy_artifact_into_container(_artifact(filename)) == f"/data/{filename}"

    exec_mkdir, cp = host.docker_calls()
    assert exec_mkdir == ["exec", "cid", "mkdir", "-p", "/data"]
    assert cp[0] == "cp" and cp[2] == f"cid:/data/{filename}"
    staged = Path(cp[1])
    assert staged.parent == host.sandbox.work_dir / "_artifact_staging" and staged.name.endswith(filename)
    assert not (host.root / "pwned").exists()
    assert not staged.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(("environment_name", "filename"), _SHELL_NAMES)
async def test_a_website_stages_a_shell_looking_name_as_it_is(host, environment_name, filename):
    env = _website(host, environment_name)

    assert await env._copy_artifact_into_container(_artifact(filename)) == f"/tmp/data/{filename}"

    exec_mkdir, cp = host.docker_calls()
    service = f"{environment_name}-website-backend"
    compose = ["compose", "-f", str(host.sandbox.work_dir / "docker-compose.yml")]
    assert exec_mkdir == [*compose, "exec", "-T", service, "mkdir", "-p", "/tmp/data"]
    assert cp[:4] == [*compose, "cp"] and cp[5] == f"{service}:/tmp/data/{filename}"
    assert not (host.root / "pwned").exists()
    assert cp[4].endswith(filename) and not Path(cp[4]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("make_env", [lambda h: _mcp(h, "email"), lambda h: _website(h, "email")], ids=["mcp", "website"])
@pytest.mark.parametrize("filename", _NOT_ONE_NAME)
async def test_a_name_that_is_not_one_plain_file_name_is_refused_before_anything_runs(host, make_env, filename):
    with pytest.raises(ValueError, match="must be a single file name"):
        await make_env(host)._copy_artifact_into_container(_artifact(filename))

    assert host.docker_calls() == [] and list(host.sandbox.work_dir.iterdir()) == []


def _mcp(host, environment_name: str) -> MCPServerEnv:
    env = MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=None, environment_name=environment_name)
    env._sandbox = host.sandbox
    env._env_provider = SimpleNamespace(_get_container_id=AsyncMock(return_value="cid"))
    return env


def _website(host, environment_name: str) -> WebsiteEnv:
    image = SimpleNamespace(id="img", version=1, type="docker_image", image_name="img:1")
    env = WebsiteEnv(id="w", version=1, backend_docker_image_artifact=image, frontend_docker_image_artifact=image,
                     environment_name=environment_name)
    env._sandbox = host.sandbox
    return env


def _artifact(filename: str) -> SimpleNamespace:
    return SimpleNamespace(filename=filename, object_url="file:///store/seed")
