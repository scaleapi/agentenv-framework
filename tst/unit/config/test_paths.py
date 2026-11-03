"""The per-user state and cache roots: XDG first, then the platform default."""

import sys

import pytest

from agent_env.config.paths import cache_root, state_root


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    return tmp_path / "home"


def test_an_absolute_xdg_state_home_wins(home, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert state_root() == tmp_path / "xdg" / "agent-env"


@pytest.mark.parametrize("value", ["", "relative/state"])
def test_an_unset_or_relative_xdg_state_home_falls_back_to_the_home_default(home, monkeypatch, value):
    monkeypatch.setenv("XDG_STATE_HOME", value)
    assert state_root() == home / ".local" / "state" / "agent-env"


def test_windows_uses_local_app_data_unless_xdg_state_home_is_set(home, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    assert state_root() == tmp_path / "appdata" / "agent-env"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert state_root() == tmp_path / "xdg" / "agent-env"


@pytest.mark.parametrize("value", ["", "relative/cache"])
def test_the_cache_root_is_an_absolute_xdg_cache_homes_else_the_home_default(home, tmp_path, monkeypatch, value):
    monkeypatch.setenv("XDG_CACHE_HOME", value)
    assert cache_root() == home / ".cache" / "agentenv"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "xdg" / "agentenv"


def test_resolving_creates_nothing(home):
    state_root()
    cache_root()
    assert not home.exists()
