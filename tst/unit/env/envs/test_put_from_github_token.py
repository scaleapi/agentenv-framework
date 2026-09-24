"""The env-level put_from_github classmethods forward github_token to every image build."""

from types import SimpleNamespace

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv


def _fake_build(id, dockerfile_github_url, **kwargs):
    return SimpleNamespace(
        artifact=SimpleNamespace(id=id, version=1),
        dockerfile_github_url=dockerfile_github_url,
        docker_context_github_url=kwargs.get("docker_context_github_url"),
        github_owner="o", github_repo="r", github_ref="main", github_commit="c" * 40,
    )


@pytest.fixture
def recorded_builds(monkeypatch):
    calls = []

    async def fake_put_from_github(cls, id, dockerfile_github_url, **kwargs):
        calls.append({"id": id, **kwargs})
        return _fake_build(id, dockerfile_github_url, **kwargs)

    monkeypatch.setattr(DockerImageArtifact, "put_from_github", classmethod(fake_put_from_github))
    for env_cls in (MCPServerEnv, WebsiteEnv):
        monkeypatch.setattr(env_cls, "put", classmethod(lambda cls, **kw: kw))
    return calls


@pytest.mark.asyncio
async def test_mcp_server_env_forwards_the_token(recorded_builds):
    await MCPServerEnv.put_from_github(
        id="e", dockerfile_github_url="https://github.com/o/r/blob/main/svc/Dockerfile", github_token="ghs_x"
    )
    assert [c["github_token"] for c in recorded_builds] == ["ghs_x"]


@pytest.mark.asyncio
async def test_mcp_server_env_defaults_to_no_token(recorded_builds):
    await MCPServerEnv.put_from_github(id="e", dockerfile_github_url="https://github.com/o/r/blob/main/svc/Dockerfile")
    assert [c["github_token"] for c in recorded_builds] == [None]


@pytest.mark.asyncio
async def test_website_env_forwards_the_token_to_both_builds(recorded_builds):
    await WebsiteEnv.put_from_github(
        id="w",
        backend_dockerfile_github_url="https://github.com/o/r/blob/main/be/Dockerfile",
        frontend_dockerfile_github_url="https://github.com/o/r/blob/main/fe/Dockerfile",
        github_token="ghs_y",
    )
    assert sorted(c["id"] for c in recorded_builds) == ["website-backend-w", "website-frontend-w"]
    assert {c["github_token"] for c in recorded_builds} == {"ghs_y"}
