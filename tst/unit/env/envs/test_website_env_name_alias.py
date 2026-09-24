"""WebsiteEnv drops the transitional service_name shim (property + kwarg alias);
only environment_name remains. The persisted "service_name" key + from_dict dual-read stay
(the additive wire contract). Parallel to the MCPServerEnv coverage, kept independent."""

from types import SimpleNamespace

import pytest

from agent_env.artifact import Artifact
from agent_env.env.envs.website import WebsiteEnv


def _art(name="img"):
    return SimpleNamespace(id=name, version=1, type="docker_image", image_name=f"{name}:1")


def _kwargs(**over):
    base = dict(
        id="w",
        version=1,
        backend_docker_image_artifact=_art("be"),
        frontend_docker_image_artifact=_art("fe"),
        service_version=1,
    )
    base.update(over)
    return base


def test_environment_name_is_the_only_accessor():
    e = WebsiteEnv(**_kwargs(environment_name="shop"))
    assert e.environment_name == "shop"
    with pytest.raises(AttributeError):
        _ = e.service_name


def test_service_name_kwarg_is_rejected():
    with pytest.raises(TypeError):
        WebsiteEnv(**_kwargs(service_name="shop"))


def test_missing_name_raises():
    with pytest.raises(ValueError):
        WebsiteEnv(**_kwargs())


def test_missing_service_version_defaults_to_one():
    """__init__ coerces rather than tolerating: to_dict writes the key unconditionally."""
    kwargs = _kwargs(environment_name="shop")
    kwargs.pop("service_version")
    e = WebsiteEnv(**kwargs)
    assert e.service_version == 1
    assert e.to_dict()["service_version"] == 1


def test_to_dict_writes_both_name_keys():
    e = WebsiteEnv(**_kwargs(environment_name="shop", service_version=3))
    d = e.to_dict()
    assert d["service_name"] == "shop"
    assert d["environment_name"] == "shop"  # dual-write
    assert d["service_version"] == 3


def test_from_dict_dual_reads_both_keys(monkeypatch):
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: _art(id)))
    be = {"id": "be", "version": 1, "type": "docker_image"}
    fe = {"id": "fe", "version": 1, "type": "docker_image"}
    common = {
        "id": "w",
        "version": 1,
        "backend_docker_image_artifact": be,
        "frontend_docker_image_artifact": fe,
        "service_version": 1,
        "metadata": {},
    }
    legacy = WebsiteEnv.from_dict({**common, "service_name": "shop"})
    assert legacy.environment_name == "shop"
    new = WebsiteEnv.from_dict({**common, "environment_name": "blog"})
    assert new.environment_name == "blog"


@pytest.mark.asyncio
async def test_put_from_github_forwards_environment_name_no_service_name(monkeypatch):
    """Exercise the WebsiteEnv.put_from_github body: it forwards environment_name into put()
    and passes no service_name after the shim removal."""
    from agent_env.artifact.artifacts.docker_image import DockerImageArtifact

    build = SimpleNamespace(
        artifact=_art("be"), dockerfile_github_url="https://github.com/o/r/tree/main/Dockerfile",
        github_owner="o", github_repo="r", github_commit="abc",
        docker_context_github_url=None, github_ref=None,
    )

    async def _fake_di(cls, **kwargs):
        return build
    monkeypatch.setattr(DockerImageArtifact, "put_from_github", classmethod(_fake_di))

    captured: dict = {}
    def _fake_put(cls, **kwargs):
        captured.update(kwargs)
        return "ENV"
    monkeypatch.setattr(WebsiteEnv, "put", classmethod(_fake_put))

    result = await WebsiteEnv.put_from_github(
        id="w",
        backend_dockerfile_github_url="https://github.com/o/r/tree/main/b/Dockerfile",
        frontend_dockerfile_github_url="https://github.com/o/r/tree/main/f/Dockerfile",
        environment_name="shop", service_version=1,
    )
    assert result == "ENV"
    assert captured["environment_name"] == "shop"
    assert "service_name" not in captured
