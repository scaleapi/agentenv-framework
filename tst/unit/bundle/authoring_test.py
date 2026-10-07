"""What ``from_toml`` gets for a bundle entry, and the defaults that write an entity from its toml."""

from pathlib import Path
from typing import Literal

import pytest

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.vm_image import VMImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle.authoring import AuthoringContext
from agent_env.env.env import Env
from agent_env.env.store import reset_env_store
from agent_env.store import reset_config, set_document_store
from tst.unit.store.fakes import FakeDocumentStore

LAYOUT = {
    "envs/tickets/Dockerfile": "FROM scratch\n",
    "envs/tickets/env.toml": 'environment_name = "tickets"\n',
    "envs/both/env.toml": 'type = "multi"\n',
    "artifacts/golden/artifact.toml": 'type = "vm_image"\n',
}


class _NoteEnv(Env):
    type = "note_test"
    description = "an env that is only a document"

    def __init__(self, id, version, note, metadata=None):
        super().__init__(id, version, metadata=metadata)
        self.note = note

    def to_dict(self):
        return {**super().to_dict(), "note": self.note}

    @classmethod
    def from_dict(cls, data):
        return cls(id=data["id"], version=data.get("version"), note=data["note"], metadata=data.get("metadata"))


class _NoteArtifact(Artifact):
    type: Literal["note_artifact_test"] = "note_artifact_test"
    note: str = ""


@pytest.fixture(autouse=True)
def stores(monkeypatch, cli_routing):
    set_document_store(FakeDocumentStore())
    monkeypatch.setattr("agent_env.env.registry.get_env_registry", lambda: {"note_test": _NoteEnv})
    yield
    reset_artifact_store()
    reset_env_store()
    reset_config()


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """The context for one entry of a small bundle rooted at ``~/triage``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for rel, text in LAYOUT.items():
        path = tmp_path / "triage" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    bundle = parse_bundle(tmp_path / "triage")
    return lambda name: AuthoringContext(bundle, next(e for e in bundle.entries if e.name == name))


def problem(call) -> str:
    with pytest.raises(BundleError) as caught:
        call()
    (message,) = caught.value.problems
    return message


def test_an_entry_gives_its_id_name_and_folder(ctx):
    tickets = ctx("tickets")
    assert (tickets.id, tickets.name) == ("@local/~/triage/tickets", "tickets")
    assert tickets.dir == Path(tickets.bundle.root) / "envs" / "tickets"


def test_the_default_writes_a_document_only_artifact_as_put_would(ctx):
    authored = VMImageArtifact.from_toml({"type": "vm_image", "description": "golden", "ecr_url": "ecr/x"}, ctx("golden"))
    direct = VMImageArtifact.put(id="direct", description="golden", ecr_url="ecr/x")
    assert (authored.id, authored.version) == ("@local/~/triage/golden", 1)
    assert authored.model_dump(exclude={"id"}) == direct.model_dump(exclude={"id"})


def test_the_store_numbers_an_authored_artifact_whatever_its_toml_says(ctx):
    written = _NoteArtifact.from_toml({"type": "note_artifact_test", "note": "hi", "version": 7}, ctx("golden"))
    assert (written.version, written.note) == (1, "hi")


def test_the_default_writes_an_env_that_from_dict_reads_back(ctx):
    both = ctx("both")
    written = _NoteEnv.from_toml({"type": "note_test", "note": "hi"}, both)
    assert (written.id, written.version) == (both.id, 1)
    loaded = both.env(both.id, expect=_NoteEnv)
    assert (loaded.version, loaded.note) == (1, "hi")


def test_an_artifact_loads_at_latest_or_at_its_pin(ctx):
    VMImageArtifact.put(id="golden-image", description="v1", ecr_url="ecr/x")
    VMImageArtifact.put(id="golden-image", description="v2", ecr_url="ecr/x")
    both = ctx("both")
    assert both.artifact("golden-image").description == "v2"
    assert both.artifact({"artifact": "golden-image", "version": 1}, expect=VMImageArtifact).description == "v1"
    assert both.artifact({"artifact": "golden-image"}).description == "v2"


def test_a_loaded_ref_of_the_wrong_type_is_refused(ctx):
    VMImageArtifact.put(id="golden-image", description="v1", ecr_url="ecr/x")
    assert problem(lambda: ctx("tickets").artifact("golden-image", expect=DockerImageArtifact)) == (
        "envs/tickets: golden-image is a vm_image artifact, not a DockerImageArtifact"
    )


@pytest.mark.parametrize("load", [
    lambda both: both.env("tickets"),
    lambda both: both.env("Tickets"),
    lambda both: both.artifact("golden"),
    lambda both: both.artifact({"artifact": "golden", "version": 2}),
])
def test_a_bundle_name_that_reaches_from_toml_unresolved_is_refused(ctx, load):
    assert problem(lambda: load(ctx("both"))).endswith("wasn't resolved; declare its key in toml_refs")


@pytest.mark.parametrize("ref", [
    {"version": 2},
    {"env": "crm", "version": 0},
    {"env": "crm", "version": True},
    {"env": "crm", "version": 2, "extra": 1},
    {"artifact": "crm", "version": 2},
    "",
    3,
])
def test_a_malformed_ref_is_refused(ctx, ref):
    assert problem(lambda: ctx("both").env(ref)) == (
        f'envs/both: expected an id or {{ env = "<id>", version = <n> }}, the version optional, not {ref!r}'
    )
