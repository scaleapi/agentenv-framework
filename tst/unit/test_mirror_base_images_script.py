"""``.github/scripts/mirror_base_images.py``, which pulls the base images Dockerfiles name on public.ecr.aws from
mirror.gcr.io before a workflow builds them, falling back to public.ecr.aws for one the mirror doesn't give."""

from __future__ import annotations

import ast
import json
import subprocess
import sys

import pytest

from agent_env.env import bootstrap
from tst.util.scripts import SCRIPTS, load_script

REPO = SCRIPTS.parents[1]
SCRIPT = SCRIPTS / "mirror_base_images.py"
_DIGEST = "sha256:" + "c" * 64
_PINNED = f"public.ecr.aws/docker/library/python:3.11-slim@{_DIGEST}"
_TAGGED = "public.ecr.aws/docker/library/postgres:16-alpine"


@pytest.fixture
def script():
    return load_script(SCRIPT.name)


def test_it_imports_only_the_standard_library():
    """The release's image builds run it in a job that holds a token to push, which installs nothing."""
    imported = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert imported - {"__future__"} <= sys.stdlib_module_names, imported - sys.stdlib_module_names


def test_it_finds_each_public_ecr_base_once(script, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "Dockerfile").write_text(f"FROM {_PINNED}\nRUN true\nFROM ghcr.io/astral-sh/uv:latest AS uv\n")
    (tmp_path / "a" / "Dockerfile.db-mcp").write_text(f"from --platform=$BUILDPLATFORM {_TAGGED} AS build\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "Dockerfile").write_text(f"# FROM {_TAGGED}:old\nFROM {_PINNED}\n")
    (tmp_path / "b" / "README.md").write_text("FROM public.ecr.aws/docker/library/node:22\n")

    assert script._bases([tmp_path]) == [_TAGGED, _PINNED]
    assert script._bases([tmp_path / "a" / "Dockerfile.db-mcp"]) == [_TAGGED]


def test_it_finds_the_bases_of_the_images_agent_env_ships(script):
    bases = script._bases([image.dockerfile for image in bootstrap.PUBLISHED_IMAGES])

    assert bases and all(base.startswith(script.ECR) for base in bases)
    assert any("@sha256:" in base for base in bases)


@pytest.mark.parametrize("base, mirrored", [
    (_PINNED, f"mirror.gcr.io/library/python@{_DIGEST}"),
    (_TAGGED, "mirror.gcr.io/library/postgres:16-alpine"),
])
def test_a_base_is_pulled_by_its_digest_when_it_has_one(script, base, mirrored):
    assert script._mirrored(base) == mirrored


def _docker(monkeypatch, script, *, store="overlay2", pulls_fail=0):
    """Fakes the commands the script runs, recording them; the first ``pulls_fail`` pulls fail."""
    ran, failures, driver = [], iter(range(pulls_fail)), [store]

    def run(argv, **kwargs):
        ran.append(argv)
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps([["driver-type", driver[0]]]), "")
        if argv[:2] == ["sudo", "systemctl"]:
            driver[0] = "io.containerd.snapshotter.v1"
        if argv[:2] == ["sudo", "tee"]:
            ran.append(json.loads(kwargs["input"]))
        failed = argv[:2] == ["docker", "pull"] and next(failures, None) is not None
        return subprocess.CompletedProcess(argv, 1 if failed else 0, "", "")

    monkeypatch.setattr(script.subprocess, "run", run)
    monkeypatch.setattr(script.time, "sleep", lambda seconds: ran.append(["sleep", seconds]))
    return ran


def test_a_pinned_base_is_named_in_containerd_on_its_image_store(script, monkeypatch, tmp_path):
    config = tmp_path / "daemon.json"
    config.write_text('{"exec-opts": ["native.cgroupdriver=cgroupfs"]}\n')
    monkeypatch.setattr(script, "_DAEMON_CONFIG", config)
    ran = _docker(monkeypatch, script)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {_PINNED}\n")

    assert script.main([str(dockerfile)]) == 0

    mirrored = f"mirror.gcr.io/library/python@{_DIGEST}"
    assert ran == [
        ["docker", "info", "--format", "{{json .DriverStatus}}"],
        ["sudo", "tee", str(config)],
        {"exec-opts": ["native.cgroupdriver=cgroupfs"], "features": {"containerd-snapshotter": True}},
        ["sudo", "systemctl", "restart", "docker"],
        ["docker", "info", "--format", "{{json .DriverStatus}}"],
        ["docker", "pull", "--quiet", mirrored],
        ["docker", "tag", mirrored, "public.ecr.aws/docker/library/python:3.11-slim"],
        ["sudo", "ctr", "--namespace", "moby", "images", "tag", "--force", mirrored, _PINNED],
    ]


def test_dockerd_is_left_alone_when_no_base_is_pinned(script, monkeypatch, tmp_path):
    ran = _docker(monkeypatch, script)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {_TAGGED}\n")

    assert script.main([str(dockerfile)]) == 0

    assert ran == [["docker", "pull", "--quiet", "mirror.gcr.io/library/postgres:16-alpine"],
                   ["docker", "tag", "mirror.gcr.io/library/postgres:16-alpine", _TAGGED]]


def test_a_base_the_mirror_doesnt_give_is_left_to_its_build(script, monkeypatch, tmp_path, capsys):
    ran = _docker(monkeypatch, script, pulls_fail=3)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {_TAGGED}\n")

    assert script.main([str(dockerfile)]) == 0

    pull = ["docker", "pull", "--quiet", "mirror.gcr.io/library/postgres:16-alpine"]
    assert ran == [pull, ["sleep", 10], pull, ["sleep", 20], pull]
    assert f"::warning::couldn't pull mirror.gcr.io/library/postgres:16-alpine, so {_TAGGED}'s build" in capsys.readouterr().out


def test_a_pull_that_fails_once_is_tried_again(script, monkeypatch, tmp_path):
    ran = _docker(monkeypatch, script, pulls_fail=1)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {_TAGGED}\n")

    assert script.main([str(dockerfile)]) == 0

    assert ran[-1] == ["docker", "tag", "mirror.gcr.io/library/postgres:16-alpine", _TAGGED]
    assert ["sleep", 10] in ran and ["sleep", 20] not in ran
