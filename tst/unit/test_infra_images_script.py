"""The release workflow's infra images script, ``.github/scripts/infra_images.py``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_env.env import bootstrap
from agent_env.env.published_images import release_digests
from agent_env.store.image_store.registry_api import NoSuchImageError

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "infra_images.py"
_DIGEST = "sha256:" + "e" * 64


@pytest.fixture
def script(monkeypatch):
    spec = importlib.util.spec_from_file_location("infra_images", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("GHCR_TOKEN", raising=False)
    return module


def _run(script, monkeypatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["infra_images.py", *args])
    return script.main()


def _registry(script, monkeypatch, published: set[str]):
    """The registry serves ``_DIGEST`` under the refs in ``published`` and nothing else; returns the auth it saw."""
    seen = []

    def pin_digest(ref, auth):
        seen.append(auth)
        if ref not in published:
            raise NoSuchImageError(f"{ref}: ghcr.io has no such image")
        return f"{ref}@{_DIGEST}"

    monkeypatch.setattr(script, "pin_digest", pin_digest)
    return seen


def test_plan_lists_every_image_with_its_checkout_paths_and_tag(script, monkeypatch, capsys):
    assert _run(script, monkeypatch, "plan") == 0

    planned = json.loads(capsys.readouterr().out)
    assert [entry["name"] for entry in planned] == [image.name for image in bootstrap.PUBLISHED_IMAGES]
    gateway = planned[0]
    assert gateway == {"name": "agentenv-gateway", "repository": "ghcr.io/scaleapi/agentenv-gateway",
                       "tag": bootstrap.PUBLISHED_IMAGES[0].tag(), "dockerfile": "src/agent_env/env/gateway/Dockerfile",
                       "context": "src/agent_env/env/gateway"}


def test_plan_missing_lists_only_the_images_the_registry_lacks(script, monkeypatch, capsys):
    gateway, *rest = bootstrap.PUBLISHED_IMAGES
    _registry(script, monkeypatch, {image.ref() for image in rest})

    assert _run(script, monkeypatch, "plan", "--missing") == 0

    assert [entry["name"] for entry in json.loads(capsys.readouterr().out)] == [gateway.name]


def test_write_records_the_digest_served_under_every_tag(script, monkeypatch, tmp_path, capsys):
    _registry(script, monkeypatch, {image.ref() for image in bootstrap.PUBLISHED_IMAGES})
    path = tmp_path / "published_images.json"

    assert _run(script, monkeypatch, "write", str(path)) == 0

    assert release_digests(path) == {image.ref(): _DIGEST for image in bootstrap.PUBLISHED_IMAGES}


def test_write_records_nothing_when_an_image_is_missing(script, monkeypatch, tmp_path, capsys):
    db_mcp = next(image for image in bootstrap.PUBLISHED_IMAGES if image.name == "agentenv-db-mcp")
    _registry(script, monkeypatch, {image.ref() for image in bootstrap.PUBLISHED_IMAGES} - {db_mcp.ref()})
    path = tmp_path / "published_images.json"

    assert _run(script, monkeypatch, "write", str(path)) == 1

    assert not path.exists()
    assert f"::error::the registry has no image under {db_mcp.ref()}" in capsys.readouterr().out


def test_the_registry_is_read_with_the_token_when_one_is_given(script, monkeypatch, capsys):
    seen = _registry(script, monkeypatch, set())
    monkeypatch.setenv("GHCR_TOKEN", "a-token")
    monkeypatch.setenv("GITHUB_ACTOR", "release-bot")

    _run(script, monkeypatch, "plan", "--missing")

    assert {(auth.registry, auth.username, auth.password) for auth in seen} == {("ghcr.io", "release-bot", "a-token")}


def test_the_registry_is_read_anonymously_without_a_token(script, monkeypatch, capsys):
    seen = _registry(script, monkeypatch, set())

    _run(script, monkeypatch, "plan", "--missing")

    assert seen and set(seen) == {None}


@pytest.mark.parametrize("name, expected", [
    ("agentenv-gateway", ["docker", "run", "--rm", "--entrypoint", "python", "img", "-c", "import gateway.entrypoint"]),
    ("agentenv-db-mcp", ["docker", "run", "--rm", "img", "--help"]),
])
def test_a_smoke_runs_the_images_own_check(script, monkeypatch, name, expected):
    commands = []
    monkeypatch.setattr(script.subprocess, "run",
                        lambda command, **kw: commands.append(command) or subprocess.CompletedProcess(command, 0, "", ""))

    assert _run(script, monkeypatch, "smoke", name, "img") == 0

    assert commands == [expected]


def test_a_failing_smoke_names_the_command_and_its_output(script, monkeypatch, capsys):
    monkeypatch.setattr(script.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, 1, "", "ModuleNotFoundError: httpx"))

    assert _run(script, monkeypatch, "smoke", "agentenv-gateway", "img") == 1

    out = capsys.readouterr().out
    assert "::error::agentenv-gateway: `docker run --rm --entrypoint python img" in out and "httpx" in out


def _docker(statuses: list[str], calls: list[list[str]]):
    def run(command, **kw):
        calls.append(command)
        if command[:3] == ["docker", "run", "-d"]:
            return subprocess.CompletedProcess(command, 0, "c123\n", "")
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 0, statuses.pop(0) if statuses else "unhealthy", "")
        if command[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(command, 0, "FATAL: data directory has wrong ownership", "")
        return subprocess.CompletedProcess(command, 0, "", "")
    return run


def test_the_service_db_smoke_waits_for_its_health_check_and_removes_the_container(script, monkeypatch):
    calls = []
    monkeypatch.setattr(script.subprocess, "run", _docker(["starting", "starting", "healthy"], calls))
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)

    assert _run(script, monkeypatch, "smoke", "agentenv-service-db", "pg") == 0

    assert calls[0] == ["docker", "run", "-d", "pg"] and calls[-1] == ["docker", "rm", "-f", "c123"]


def test_a_service_db_that_never_turns_healthy_fails_with_its_logs(script, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(script.subprocess, "run", _docker([], calls))
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(script, "_HEALTHY_WITHIN_SECONDS", 0.01)

    assert _run(script, monkeypatch, "smoke", "agentenv-service-db", "pg") == 1

    out = capsys.readouterr().out
    assert "::error::pg wasn't healthy within 0.01s" in out and "wrong ownership" in out
    assert calls[-1] == ["docker", "rm", "-f", "c123"]


def test_notes_list_each_recorded_image_pinned_by_its_digest(script, monkeypatch, tmp_path, capsys):
    _registry(script, monkeypatch, {image.ref() for image in bootstrap.PUBLISHED_IMAGES})
    path = tmp_path / "published_images.json"
    _run(script, monkeypatch, "write", str(path))
    capsys.readouterr()

    assert _run(script, monkeypatch, "notes", str(path)) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Infra images this release pulls, by digest:"
    assert sorted(lines[1:]) == sorted(f"- `{image.ref()}@{_DIGEST}`" for image in bootstrap.PUBLISHED_IMAGES)
