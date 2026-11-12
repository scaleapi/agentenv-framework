"""agent-env env website put for a live site: live, its names, allow_hosts and secret settings read from the server's
@website(...) when no flag gives them, a flag the source contradicts refused, a secret's value refused, and a registry
image's stated with flags."""

from unittest.mock import patch

from click.testing import CliRunner

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.cli.env.website import website
from agent_env.env import Env

SOURCE = '''from pydantic import BaseModel, SecretStr
from agentenv_protocol.website import WebsiteEnvironment, website


@website(domains=["github.com"], live=True, allow_hosts=["*.githubassets.com"])
class GitHub(WebsiteEnvironment):
    class Settings(BaseModel):
        token: SecretStr
'''


def _stored(id, **_):
    return get_artifact_store().put_document(DockerImageArtifact(
        id=id, description="", image_name=f"registry.example/{id}@sha256:" + "a" * 64))


def _put(*args):
    return CliRunner().invoke(website, ["put", "--id", "github-site", "--environment-name", "github", *args,
                                        "--skip-validation"])


def _built(tmp_path, *args):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "site.py").write_text(SOURCE)
    with patch("agent_env.cli.env.website.build_image"), \
            patch("agent_env.artifact.artifacts.docker_image.DockerImageArtifact.put",
                  side_effect=lambda **k: _stored(k["id"])):
        return _put("--server-dockerfile", str(tmp_path / "Dockerfile"), *args)


def test_a_live_site_is_stored_as_its_source_declares_it(local_stores, tmp_path):
    result = _built(tmp_path, "--setting", "token=env:GH_TOKEN")

    assert result.exit_code == 0, result.output
    assert "A live site at github.com, its pages reaching *.githubassets.com" in result.output
    env = Env.get("github-site")
    assert (env.live, env.domains, env.allow_hosts, env.settings, env.secret_settings) == (
        True, ["github.com"], ["*.githubassets.com"], {"token": "env:GH_TOKEN"}, ["token"])


def test_put_refuses_a_flag_the_source_contradicts_and_a_secrets_value(local_stores, tmp_path):
    contradicted = _built(tmp_path, "--no-live")
    plain = _built(tmp_path, "--setting", "token=ghp_plain")

    assert contradicted.exit_code == 1 and "live is False here, but the site's @website(...) declares live=True" in (
        contradicted.output)
    assert plain.exit_code == 1 and "settings.token is a secret" in plain.output


def test_a_registry_images_live_site_is_stated_with_flags(local_stores):
    with patch("agent_env.cli.env.website.DockerImageArtifact.put_ref", side_effect=lambda id, **k: _stored(id)):
        result = _put("--server-image-ref", "ghcr.io/acme/github:3", "--live", "--domain", "github.com",
                      "--allow-host", "api.github.com")
        unnamed = _put("--server-image-ref", "ghcr.io/acme/github:3", "--live")

    assert result.exit_code == 0, result.output
    env = Env.get("github-site")
    assert (env.live, env.domains, env.allow_hosts) == (True, ["github.com"], ["api.github.com"])
    assert unnamed.exit_code == 1 and "a live website env names the domains the real site is at" in unnamed.output
