"""An env stages a loaded artifact's file through its sandbox's shell: a file or env name with shell syntax in it is
staged and copied as it is and runs nothing, a file inside folders is staged under them, and a path that isn't plain
file names joined by '/' is refused before anything runs. A server whose VM sandbox is its container stages and removes
a load's folder through docker, never on the host. A real LocalSandbox runs the scripts; docker is a stub that logs its
arguments."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER

_SHELL_NAMES = [
    ("email", "x;touch pwned;#.json"),
    ("email", "$(touch pwned).json"),
    ("email", "`touch pwned`.json"),
    ("email", "x' ; touch pwned ; '.json"),
    ("email", "seed data.json"),
    ("e;touch pwned;#", "seed.json"),
]
_NOT_A_PATH = ["../seed.json", "/seed.json", "a//b.json", "a/./b.json", "a\\b.json", "..", ".", "", "seed\n.json",
               "seed\x00.json"]


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

    assert await env._copy_artifact_into_container(_artifact(filename), filename) == f"/data/{filename}"

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

    assert await env._copy_artifact_into_container(_artifact(filename), filename) == f"/tmp/data/{filename}"

    exec_mkdir, cp = host.docker_calls()
    service = f"{environment_name}-website-backend"
    compose = ["compose", "-f", str(host.sandbox.work_dir / "docker-compose.yml")]
    assert exec_mkdir == [*compose, "exec", "-T", service, "mkdir", "-p", "/tmp/data"]
    assert cp[:4] == [*compose, "cp"] and cp[5] == f"{service}:/tmp/data/{filename}"
    assert not (host.root / "pwned").exists()
    assert cp[4].endswith(filename) and not Path(cp[4]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(("make_env", "data"), [(lambda h: _mcp(h, "email"), "/data"),
                                                (lambda h: _website(h, "email"), "/tmp/data")], ids=["mcp", "website"])
async def test_a_file_inside_folders_is_staged_under_them(host, make_env, data):
    assert await make_env(host)._copy_artifact_into_container(_artifact("b.json"), "a/b.json") == f"{data}/a/b.json"

    exec_mkdir, cp = host.docker_calls()
    assert exec_mkdir[-3:] == ["mkdir", "-p", f"{data}/a"] and cp[-1].endswith(f":{data}/a/b.json")


@pytest.mark.asyncio
async def test_a_server_whose_vm_sandbox_is_its_container_stages_and_removes_a_loads_folder_through_docker(host):
    env = _mcp(host, "email")
    host.sandbox.mode = SANDBOX_MODE_CONTAINER
    server = host.sandbox.container_name

    await env._copy_artifact_into_container(_artifact("b.json"), ".agentenv-load-1/a/b.json")
    await env._remove_from_container(".agentenv-load-1")

    calls = host.docker_calls()
    assert ["exec", "-u", "0", server, "mkdir", "-p", "/data/.agentenv-load-1/a"] in calls
    assert calls[-1] == ["exec", "-u", "0", server, "rm", "-rf", "/data/.agentenv-load-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_env", [lambda h: _mcp(h, "email"), lambda h: _website(h, "email")], ids=["mcp", "website"])
@pytest.mark.parametrize("path", _NOT_A_PATH)
async def test_a_path_that_is_not_plain_file_names_is_refused_before_anything_runs(host, make_env, path):
    with pytest.raises(ValueError, match="must be plain file names joined by '/'"):
        await make_env(host)._copy_artifact_into_container(_artifact("seed.json"), path)

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
