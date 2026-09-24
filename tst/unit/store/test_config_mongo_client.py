"""Asserts the Atlas resilience kwargs on MongoDocumentStore.from_config, and
that the retained Config.db shares the configured Mongo store's database instead of
building a second client."""

import pytest

from agent_env.config import Config
from agent_env.store import ConfigError
from agent_env.store.document_store import MongoDocumentStore


class _FakeAdmin:
    def command(self, *_args, **_kwargs):
        return {"ok": 1}


class _FakeClient:
    def __init__(self, uri, **kwargs):
        self.kwargs = kwargs
        self.uri = uri
        self.admin = _FakeAdmin()

    def __getitem__(self, name):
        return {"__dbname__": name}


def test_db_shares_the_configured_mongo_stores_database(monkeypatch):
    monkeypatch.setattr(
        "agent_env.store.document_store.mongo_document_store.MongoClient", _FakeClient
    )
    store = MongoDocumentStore.from_config(uri="mongodb://fake", database="agent_env_dev")
    cfg = Config()
    cfg.set_document_store(store)

    assert cfg.db is store.database
    assert cfg.db == {"__dbname__": "agent_env_dev"}


def test_db_raises_actionable_without_a_mongo_store(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_DOCUMENT_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match=r"\[stores\.document\]"):
        _ = Config().db


def test_from_config_builds_pinged_atlas_client(monkeypatch):
    captured: dict = {}

    def _factory(uri, **kwargs):
        captured["uri"] = uri
        captured["kwargs"] = kwargs
        return _FakeClient(uri, **kwargs)

    monkeypatch.setattr(
        "agent_env.store.document_store.mongo_document_store.MongoClient", _factory
    )
    store = MongoDocumentStore.from_config(uri="mongodb://fake", database="agent_env")

    assert isinstance(store, MongoDocumentStore)
    assert captured["uri"] == "mongodb://fake"
    assert captured["kwargs"]["retryReads"] is True
    assert captured["kwargs"]["retryWrites"] is True
    assert captured["kwargs"]["serverSelectionTimeoutMS"] >= 15000
    assert captured["kwargs"]["connectTimeoutMS"] >= 15000
    assert captured["kwargs"]["socketTimeoutMS"] >= 30000
