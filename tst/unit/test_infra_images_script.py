"""The release workflow's read-only infra images script, ``.github/scripts/infra_images.py``."""

from __future__ import annotations

import json
import sys

import pytest

from agent_env.env import bootstrap
from agent_env.env.published_images import AGENT_IMAGES, release_digests
from agent_env.store.image_store.registry_api import NoSuchImageError, RegistryRefusedError
from tst.util.scripts import load_script

_DIGEST = "sha256:" + "e" * 64


@pytest.fixture
def script(monkeypatch):
    monkeypatch.delenv("GHCR_TOKEN", raising=False)
    return load_script("infra_images.py")


def _run(script, monkeypatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["infra_images.py", *args])
    return script.main()


def _registry(script, monkeypatch, served: dict[str, Exception | str]):
    """The registry serves ``_DIGEST`` under every ref but those ``served`` maps to an error; returns the auth it saw."""
    seen = []

    def pin_digest(ref, auth):
        seen.append(auth)
        if isinstance(answer := served.get(ref, _DIGEST), Exception):
            raise answer
        return f"{ref}@{answer}"

    monkeypatch.setattr(script, "pin_digest", pin_digest)
    return seen


def test_plan_lists_every_image_with_its_checkout_paths_and_tag(script, monkeypatch, capsys):
    assert _run(script, monkeypatch, "plan") == 0

    planned = json.loads(capsys.readouterr().out)
    assert [entry["name"] for entry in planned] == [image.name for image in bootstrap.PUBLISHED_IMAGES]
    assert planned[0] == {"name": "agentenv-gateway", "repository": "ghcr.io/scaleapi/agentenv-gateway",
                          "tag": bootstrap.PUBLISHED_IMAGES[0].tag(), "dockerfile": "src/agent_env/env/gateway/Dockerfile",
                          "context": "src/agent_env/env/gateway"}


def test_plan_only_lists_the_infra_images_or_the_agents_images(script, monkeypatch, capsys):
    """The release builds the infra images before it publishes the protocol, and the agents', which install it, after."""
    planned = {}
    for only in ("infra", "agents"):
        assert _run(script, monkeypatch, "plan", "--only", only) == 0
        planned[only] = [entry["name"] for entry in json.loads(capsys.readouterr().out)]

    agents = [image.name for image in AGENT_IMAGES]
    assert planned == {"infra": [image.name for image in bootstrap.PUBLISHED_IMAGES if image.name not in agents],
                       "agents": agents}
    assert "agentenv-gateway" in planned["infra"] and agents == [
        "agentenv-claude-code", "agentenv-codex", "agentenv-gemini", "agentenv-opencode", "agentenv-openhands",
        "agentenv-pi", "agentenv-proxy"]


def test_plan_missing_lists_the_images_the_registry_lacks_or_refuses_to_show(script, monkeypatch, capsys):
    gateway, service_db, *rest = bootstrap.PUBLISHED_IMAGES
    _registry(script, monkeypatch, {gateway.ref(): NoSuchImageError("no such image"),
                                    service_db.ref(): RegistryRefusedError("refused to serve it (HTTP 403)")})

    assert _run(script, monkeypatch, "plan", "--missing") == 0

    assert [entry["name"] for entry in json.loads(capsys.readouterr().out)] == [gateway.name, service_db.name]


def test_plan_fails_when_the_registry_cant_say(script, monkeypatch):
    _registry(script, monkeypatch, {bootstrap.PUBLISHED_IMAGES[0].ref(): ValueError("ghcr.io answered HTTP 500")})

    with pytest.raises(ValueError, match="HTTP 500"):
        _run(script, monkeypatch, "plan", "--missing")


def test_write_records_the_digest_served_under_every_tag(script, monkeypatch, tmp_path, capsys):
    _registry(script, monkeypatch, {})
    path = tmp_path / "published_images.json"

    assert _run(script, monkeypatch, "write", str(path)) == 0

    assert release_digests(path) == {image.ref(): _DIGEST for image in bootstrap.PUBLISHED_IMAGES}


def test_write_records_nothing_when_an_image_is_missing(script, monkeypatch, tmp_path, capsys):
    db_mcp = next(image for image in bootstrap.PUBLISHED_IMAGES if image.name == "agentenv-db-mcp")
    _registry(script, monkeypatch, {db_mcp.ref(): NoSuchImageError("no such image")})
    path = tmp_path / "published_images.json"

    assert _run(script, monkeypatch, "write", str(path)) == 1

    assert not path.exists()
    assert f"::error::the registry has no image under {db_mcp.ref()}" in capsys.readouterr().out


def test_the_registry_is_read_with_the_token_when_one_is_given(script, monkeypatch, capsys):
    seen = _registry(script, monkeypatch, {})
    monkeypatch.setenv("GHCR_TOKEN", "a-token")
    monkeypatch.setenv("GITHUB_ACTOR", "release-bot")

    _run(script, monkeypatch, "plan", "--missing")

    assert {(auth.registry, auth.username, auth.password) for auth in seen} == {("ghcr.io", "release-bot", "a-token")}


def test_the_registry_is_read_anonymously_without_a_token(script, monkeypatch, capsys):
    seen = _registry(script, monkeypatch, {})

    _run(script, monkeypatch, "plan", "--missing")

    assert seen and set(seen) == {None}


def test_public_warns_about_each_image_an_anonymous_pull_doesnt_get(script, monkeypatch, capsys):
    gateway, db_web, *rest = bootstrap.PUBLISHED_IMAGES
    seen = _registry(script, monkeypatch, {
        gateway.ref(): RegistryRefusedError(f"{gateway.ref()}: ghcr.io refused to serve it (HTTP 401); agent-env reaches"),
        db_web.ref(): NoSuchImageError(f"{db_web.ref()}: ghcr.io has no such image")})
    monkeypatch.setenv("GHCR_TOKEN", "a-token")

    assert _run(script, monkeypatch, "public") == 0

    warnings = [line for line in capsys.readouterr().out.splitlines() if line.startswith("::warning")]
    assert [line.split("::")[1] for line in warnings] == [f"warning title={gateway.name} isn't public",
                                                          f"warning title={db_web.name} isn't public"]
    assert f"{gateway.ref()}: ghcr.io refused to serve it (HTTP 401). Make the package public at " \
           "https://github.com/orgs/scaleapi/packages/container/agentenv-gateway/settings" in warnings[0]
    assert set(seen) == {None}
