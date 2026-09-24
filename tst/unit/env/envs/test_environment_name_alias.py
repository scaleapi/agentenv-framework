"""The transitional service_name shim is removed. MCPServerEnv exposes only
environment_name — no service_name property, no service_name= kwarg alias. The persisted
Mongo key stays service_name and from_dict still dual-reads (the additive wire contract).

The dual-read section at the bottom covers the reader half of both env classes (MCPServerEnv
+ WebsiteEnv, whose from_dict behaves identically): only the NAME is dual-read, and to_dict
is frozen on the legacy service_* keys. service_version has no dual-read on purpose — it is
deprecated rather than renamed, so it has no new spelling to read; absence now defaults to 1."""

from types import SimpleNamespace

import pytest

from agent_env.artifact import Artifact
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv


def _art():
    return SimpleNamespace(id="img", version=1, type="docker_image", image_name="img:1")


def test_environment_name_is_the_only_accessor():
    e = MCPServerEnv(id="e", version=1, docker_image_artifact=_art(), environment_name="email", service_version=1)
    assert e.environment_name == "email"
    with pytest.raises(AttributeError):
        _ = e.service_name


def test_service_name_kwarg_is_rejected():
    with pytest.raises(TypeError):
        MCPServerEnv(id="e", version=1, docker_image_artifact=_art(), service_name="email", service_version=1)


def test_missing_name_raises():
    with pytest.raises(ValueError):
        MCPServerEnv(id="e", version=1, docker_image_artifact=_art(), service_version=1)


def test_to_dict_writes_both_name_keys():
    e = MCPServerEnv(id="e", version=1, docker_image_artifact=_art(), environment_name="email", service_version=2)
    d = e.to_dict()
    assert d["service_name"] == "email"
    assert d["environment_name"] == "email"  # dual-write
    assert d["service_version"] == 2


def test_from_dict_dual_reads_both_keys(monkeypatch):
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: _art()))
    ref = {"id": "img", "version": 1, "type": "docker_image"}
    legacy = MCPServerEnv.from_dict({"id": "e", "version": 1, "docker_image_artifact": ref, "service_name": "email", "service_version": 1, "metadata": {}})
    assert legacy.environment_name == "email"
    new = MCPServerEnv.from_dict({"id": "e", "version": 1, "docker_image_artifact": ref, "environment_name": "slack", "service_version": 1, "metadata": {}})
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
        environment_name="email", service_version=1,
    )
    assert result == "ENV"
    assert captured["environment_name"] == "email"
    assert "service_name" not in captured


# --- from_dict dual-reads environment_name; to_dict is frozen on the legacy keys ---
#
# There is deliberately NO environment_version dual-read. service_version is slated for
# deprecation rather than rename, so nothing will ever write environment_version — a reader
# for it would be dead on arrival. Its absence defaults to 1 rather than raising.


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
    """The dominant case: every env document in Mongo today spells both keys service_*."""
    e = cls.from_dict(doc(service_name="email", service_version=3))
    assert (e.environment_name, e.service_version) == ("email", 3)


@pytest.mark.parametrize("cls,doc", _READERS)
def test_from_dict_reads_environment_name(cls, doc, stub_artifact_get):
    """A document spelling the name environment_name loads to the same object as the legacy one."""
    legacy = cls.from_dict(doc(service_name="email", service_version=3))
    new = cls.from_dict(doc(environment_name="email", service_version=3))
    assert (new.environment_name, new.service_version) == ("email", 3)
    assert new.to_dict() == legacy.to_dict()


@pytest.mark.parametrize("cls,doc", _READERS)
def test_from_dict_prefers_environment_name_when_both_present(cls, doc, stub_artifact_get):
    e = cls.from_dict(doc(service_name="email", environment_name="slack", service_version=1))
    assert (e.environment_name, e.service_version) == ("slack", 1)


@pytest.mark.parametrize("cls,doc", _READERS)
def test_from_dict_ignores_environment_version(cls, doc, stub_artifact_get):
    """service_version has no dual-read: environment_version is not consulted. Pins that nobody
    adds a reader for a key no writer will ever produce."""
    e = cls.from_dict(doc(environment_name="email", service_version=3, environment_version=99))
    assert e.service_version == 3


@pytest.mark.parametrize("cls,doc", _READERS)
def test_from_dict_defaults_missing_service_version(cls, doc, stub_artifact_get):
    """Absence is no longer fatal: 224 dev mcp_server docs predate the key, and the 84 that are
    otherwise well-formed go from KeyError to loading cleanly."""
    e = cls.from_dict(doc(environment_name="email", environment_version=99))
    assert e.service_version == 1

    explicit_null = cls.from_dict(doc(environment_name="email", service_version=None))
    assert explicit_null.service_version == 1
    assert explicit_null.to_dict()["service_version"] == 1
    assert e.to_dict()["service_version"] == 1


@pytest.mark.parametrize("cls,doc", _READERS)
def test_from_dict_missing_both_name_spellings_raises(cls, doc, stub_artifact_get):
    with pytest.raises(KeyError, match="service_name"):
        cls.from_dict(doc(service_version=1))


@pytest.mark.parametrize("keys", [{"service_name": "email", "service_version": 3}, {"environment_name": "email", "service_version": 3}], ids=["legacy_doc", "environment_name_doc"])
@pytest.mark.parametrize("cls,doc", _READERS)
def test_to_dict_writes_both_name_keys_whichever_spelling_was_read(cls, doc, keys, stub_artifact_get):
    """The legacy key is written no matter which spelling came in — that is what
    keeps an old SDK able to load a document this build wrote. The dual-write
    adds the new spelling beside it, so the read side can never re-key the doc in
    either direction. ``environment_version`` still never appears."""
    d = cls.from_dict(doc(**keys)).to_dict()
    assert d["service_name"] == "email"
    assert d["environment_name"] == "email"
    assert d["service_version"] == 3
    assert "environment_version" not in d
