"""The transitional service_name shim is removed. MCPServerEnv exposes only
environment_name — no service_name property, no service_name= kwarg alias. The persisted
Mongo key stays service_name and from_dict still dual-reads (the additive wire contract).

The dual-read section at the bottom covers the reader half of both env classes (MCPServerEnv
+ WebsiteEnv, whose from_dict behaves identically): only the NAME is dual-read, and to_dict
is frozen on the legacy service_name key. service_version was deleted rather than renamed, so
neither spelling is read or written; stored documents that still carry it load and drop it."""

from types import SimpleNamespace

import pytest

from agent_env.artifact import Artifact
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv


def _art():
    return SimpleNamespace(id="img", version=1, type="docker_image", image_name="img:1")


def test_environment_name_is_the_only_accessor():
    e = MCPServerEnv(id="e", version=1, docker_image_artifact=_art(), environment_name="email")
    assert e.environment_name == "email"
    with pytest.raises(AttributeError):
        _ = e.service_name


def test_service_name_kwarg_is_rejected():
    with pytest.raises(TypeError):
        MCPServerEnv(id="e", version=1, docker_image_artifact=_art(), service_name="email")


def test_missing_name_raises():
    with pytest.raises(ValueError):
        MCPServerEnv(id="e", version=1, docker_image_artifact=_art())


def test_to_dict_writes_both_name_keys():
    e = MCPServerEnv(id="e", version=1, docker_image_artifact=_art(), environment_name="email")
    d = e.to_dict()
    assert d["service_name"] == "email"
    assert d["environment_name"] == "email"  # dual-write


def test_from_dict_dual_reads_both_keys(monkeypatch):
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: _art()))
    ref = {"id": "img", "version": 1, "type": "docker_image"}
    legacy = MCPServerEnv.from_dict({"id": "e", "version": 1, "docker_image_artifact": ref, "service_name": "email", "metadata": {}})
    assert legacy.environment_name == "email"
    new = MCPServerEnv.from_dict({"id": "e", "version": 1, "docker_image_artifact": ref, "environment_name": "slack", "metadata": {}})
    assert new.environment_name == "slack"


@pytest.mark.asyncio
async def test_put_from_github_forwards_environment_name_no_service_name(monkeypatch):
    """Exercise the put_from_github body (not just the CLI wiring): it forwards environment_name
    into put() and passes no service_name after the shim removal."""
    from agent_env.artifact.artifacts.docker_image import DockerImageArtifact

    build = SimpleNamespace(
        artifact=_art(), dockerfile_github_url="https://github.com/o/r/tree/main/Dockerfile",
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
    monkeypatch.setattr(MCPServerEnv, "put", classmethod(_fake_put))

    result = await MCPServerEnv.put_from_github(
        id="e", dockerfile_github_url="https://github.com/o/r/tree/main/Dockerfile",
        environment_name="email",
    )
    assert result == "ENV"
    assert captured["environment_name"] == "email"
    assert "service_name" not in captured


# --- from_dict dual-reads environment_name; to_dict is frozen on the legacy name key ---
#
# service_version was deleted, not renamed, so nothing reads or writes it under either
# spelling. The store is append-only: documents written before the deletion keep the key.


def _ref():
    return {"id": "img", "version": 1, "type": "docker_image"}


def _mcp_doc(**name_keys):
    return {"id": "e", "version": 1, "docker_image_artifact": _ref(), "metadata": {}, **name_keys}


def _website_doc(**name_keys):
    return {"id": "w", "version": 1, "backend_docker_image_artifact": _ref(), "frontend_docker_image_artifact": _ref(), "metadata": {}, **name_keys}


_READERS = [pytest.param(MCPServerEnv, _mcp_doc, id="mcp_server"), pytest.param(WebsiteEnv, _website_doc, id="website")]


@pytest.fixture
def stub_artifact_get(monkeypatch):
    """Artifact.get is the only I/O from_dict does; the unit suite blocks sockets."""
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: _art()))


@pytest.mark.parametrize("cls,doc", _READERS)
def test_from_dict_reads_legacy_service_keys(cls, doc, stub_artifact_get):
    """The dominant case: most stored env documents spell both keys service_*."""
    e = cls.from_dict(doc(service_name="email", service_version=3))
    assert e.environment_name == "email"


@pytest.mark.parametrize("cls,doc", _READERS)
def test_from_dict_reads_environment_name(cls, doc, stub_artifact_get):
    """A document spelling the name environment_name loads to the same object as the legacy one."""
    legacy = cls.from_dict(doc(service_name="email", service_version=3))
    new = cls.from_dict(doc(environment_name="email"))
    assert new.environment_name == "email"
    assert new.to_dict() == legacy.to_dict()


@pytest.mark.parametrize("cls,doc", _READERS)
def test_from_dict_prefers_environment_name_when_both_present(cls, doc, stub_artifact_get):
    e = cls.from_dict(doc(service_name="email", environment_name="slack"))
    assert e.environment_name == "slack"


@pytest.mark.parametrize("cls,doc", _READERS)
def test_from_dict_missing_both_name_spellings_raises(cls, doc, stub_artifact_get):
    with pytest.raises(KeyError, match="service_name"):
        cls.from_dict(doc())


@pytest.mark.parametrize("keys", [{"service_name": "email"}, {"environment_name": "email"}], ids=["legacy_doc", "environment_name_doc"])
@pytest.mark.parametrize("cls,doc", _READERS)
def test_to_dict_writes_both_name_keys_whichever_spelling_was_read(cls, doc, keys, stub_artifact_get):
    """The legacy key is written no matter which spelling came in — that is what
    keeps an old SDK able to load a document this build wrote. The dual-write
    adds the new spelling beside it, so the read side can never re-key the doc in
    either direction. ``environment_version`` still never appears."""
    d = cls.from_dict(doc(**keys)).to_dict()
    assert d["service_name"] == "email"
    assert d["environment_name"] == "email"
    assert "environment_version" not in d


@pytest.mark.parametrize("cls,doc", _READERS)
def test_loads_a_legacy_doc_still_carrying_service_version(cls, doc, stub_artifact_get):
    e = cls.from_dict(doc(environment_name="email", service_version=9))
    assert e.environment_name == "email"
    assert not hasattr(e, "service_version")


@pytest.mark.parametrize("cls,doc", _READERS)
def test_service_version_is_not_written_back(cls, doc, stub_artifact_get):
    """A doc read with the stale key must not re-emit it, or reading and re-putting would
    resurrect the field one document at a time. No environment_version replaces it."""
    d = cls.from_dict(doc(service_name="email", service_version=9)).to_dict()
    assert "service_version" not in d
    assert "environment_version" not in d


_GITHUB = "https://github.com/o/r/tree/main/Dockerfile"


@pytest.mark.parametrize("call", [
    pytest.param(lambda: MCPServerEnv(id="e", version=1, docker_image_artifact=_art(), environment_name="email", service_version=1), id="mcp_server"),
    pytest.param(lambda: MCPServerEnv.put(id="e", docker_image_artifact=_art(), environment_name="email", service_version=1), id="mcp_server_put"),
    pytest.param(lambda: MCPServerEnv.put_from_github(id="e", dockerfile_github_url=_GITHUB, environment_name="email", service_version=1),
                 id="mcp_server_put_from_github"),
    pytest.param(lambda: WebsiteEnv(id="w", version=1, backend_docker_image_artifact=_art(), frontend_docker_image_artifact=_art(),
                                    environment_name="email", service_version=1), id="website"),
    pytest.param(lambda: WebsiteEnv.put(id="w", backend_docker_image_artifact=_art(), frontend_docker_image_artifact=_art(),
                                        environment_name="email", service_version=1), id="website_put"),
    pytest.param(lambda: WebsiteEnv.put_from_github(id="w", backend_dockerfile_github_url=_GITHUB, frontend_dockerfile_github_url=_GITHUB,
                                                    environment_name="email", service_version=1), id="website_put_from_github"),
])
def test_service_version_kwarg_is_rejected(call):
    with pytest.raises(TypeError, match="service_version"):
        call()


@pytest.mark.parametrize("call", [
    pytest.param(lambda: MCPServerEnv("e", 1, _art(), "email", 1, {}), id="mcp_server"),
    pytest.param(lambda: MCPServerEnv.put_from_github("e", _GITHUB, None, "email", 1, {}), id="mcp_server_put_from_github"),
    pytest.param(lambda: WebsiteEnv("w", 1, _art(), _art(), "email", 1, {}), id="website"),
    pytest.param(lambda: WebsiteEnv.put_from_github("w", _GITHUB, None, _GITHUB, None, "email", 1, {}), id="website_put_from_github"),
])
def test_a_positional_call_that_passed_service_version_raises(call):
    """Everything after the name is keyword-only, so a positional caller written against the old
    signature fails instead of shifting its version into metadata."""
    with pytest.raises(TypeError, match="positional"):
        call()


def test_positional_arguments_up_to_the_name_still_bind():
    assert MCPServerEnv("e", 1, _art(), "email").environment_name == "email"
    assert WebsiteEnv("w", 1, _art(), _art(), "email").environment_name == "email"
