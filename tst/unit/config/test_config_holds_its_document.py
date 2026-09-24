"""A `Config` reads one document, resolved on first use and kept.

It read `config_snapshot.current()` on every access, so its stores, `[model]` endpoint and
local-store paths each answered from whichever file was current when they were built.
"""

from __future__ import annotations

import threading

import pytest

from agent_env.config import get_config, reset_config
from agent_env.config.runtime import Config


def _config_at(directory, default):
    path = directory / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[sandbox]\ndefault = "{default}"\n')
    return path


def test_a_config_keeps_the_document_it_first_read(tmp_path, monkeypatch):
    a = _config_at(tmp_path / "a", "local")
    _config_at(tmp_path / "b", "modal")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(a))
    config = Config()
    assert config.config_path() == a

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "b" / "config.toml"))

    assert config.config_path() == a
    assert config.config_file()["sandbox"]["default"] == "local"


def test_two_configs_can_hold_different_documents(tmp_path, monkeypatch):
    """The capability this buys: a worker running two stages in one process no longer has to
    re-point a global between them."""
    a = _config_at(tmp_path / "a", "local")
    b = _config_at(tmp_path / "b", "modal")

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(a))
    first = Config()
    first.config_path()                                   # resolve it now
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(b))
    second = Config()

    assert (first.config_path(), second.config_path()) == (a, b)


def test_constructing_a_config_still_reads_nothing(tmp_path, monkeypatch):
    """`config show` builds a Config to *report* a broken AGENT_ENV_CONFIG, so a constructor
    that resolved would raise before the report."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "nowhere.toml"))

    config = Config()                                     # no raise

    with pytest.raises(Exception):
        config.config_path()


def test_racing_first_callers_share_one_config(tmp_path, monkeypatch):
    """`get_config` had no lock — survivable while the Config held nothing, but now the loser
    of the race carries a document, and anything built from it goes too."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(tmp_path, "local")))
    reset_config()
    start, seen = threading.Barrier(8), []

    threads = [threading.Thread(target=lambda: (start.wait(timeout=5), seen.append(get_config())))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(seen) == 8 and len({id(c) for c in seen}) == 1


def test_configs_on_different_documents_are_not_equal(tmp_path, monkeypatch):
    """Every private field is `compare=False`, so field equality called two Configs on
    different files equal. Defensible before they held a document; not now."""
    a = _config_at(tmp_path / "a", "local")
    b = _config_at(tmp_path / "b", "modal")

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(a))
    first = Config()
    first.config_path()
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(b))
    second = Config()
    second.config_path()

    assert first != second
    assert first == first
