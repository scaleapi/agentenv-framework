"""Base Artifact.put persists a doc-only artifact via put_document.

Regression guard: base Artifact.put previously called a non-existent
ArtifactStore.put (-> AttributeError). It's only reached by doc-only subclasses
(the built-ins all override put), i.e. exactly the custom Artifact types a user
can register. Network-free via an injected FakeDocumentStore.
"""

from typing import Literal

import pytest

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.store import reset_config, set_document_store
from tst.unit.store.fakes import FakeDocumentStore


class _DocOnlyArtifact(Artifact):
    type: Literal["doc_only_put_test"] = "doc_only_put_test"
    payload: str = ""


@pytest.fixture(autouse=True)
def _isolate():
    yield
    reset_artifact_store()
    reset_config()


def test_base_put_persists_doc_only_artifact():
    store = FakeDocumentStore()
    set_document_store(store)
    reset_artifact_store()

    saved = _DocOnlyArtifact.put(id="d1", payload="hello")

    # put_document ran: the persisted instance is returned with an allocated version.
    assert type(saved) is _DocOnlyArtifact
    assert saved.id == "d1"
    assert saved.version == 1
    assert saved.payload == "hello"

    # and the document actually landed in the store.
    docs = [d for d in store.docs if d["id"] == "d1"]
    assert len(docs) == 1
    assert docs[0]["type"] == "doc_only_put_test"
    assert docs[0]["payload"] == "hello"
