"""`reset_config()` drops the config *and* everything built from the document it resolved.

It used to drop only the `Config` and the parsed document, leaving the five registries built
from that document in place. So re-pointing `AGENT_ENV_CONFIG` after a reset gave you a fresh
`Config` reading the new file and a registry still holding the old file's impls — the two
halves of one process disagreeing about which config they are on.

The registries are fields on the `Config` now, so the two cannot come apart: dropping the
config drops them. These tests state that as behaviour rather than structure.
"""

from __future__ import annotations

import textwrap
from typing import Literal

import pytest

from agent_env.a2a_agent.store import get_a2a_agent_instance_store, get_a2a_agent_store
from agent_env.artifact.artifact import Artifact
from agent_env.artifact.store import get_artifact_store
from agent_env.config import get_config, reset_config
from agent_env.config.runtime import Config
from agent_env.env.env import Env
from agent_env.env.env_artifact_store import get_env_artifact_store
from agent_env.env.registry import get_env_registry
from agent_env.env.snapshot_store import get_env_snapshot_store
from agent_env.env.store import get_env_instance_store, get_env_store
from agent_env.eval.store import get_eval_store
from agent_env.providers.state.store import get_env_state_instance_store
from agent_env.task.store import get_task_instance_store, get_task_store
from agent_env.task_step.review_store import get_review_store
from agent_env.task_step.store import get_task_step_store

_HERE = "tst.unit.config.test_reset_config"


class _ResetProbeEnv(Env):
    type = "reset_probe_env"

    @classmethod
    def _create(cls, **kwargs):  # pragma: no cover - never deployed
        raise NotImplementedError


class _ProbeArtifact(Artifact):
    type: Literal["probe_artifact"] = "probe_artifact"


def _write_config(directory, body):
    path = directory / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return path


def test_a_re_point_after_a_reset_rebuilds_the_registry(monkeypatch, tmp_path):
    """The defect, through the reader rather than the private global. Without the registry
    reset the second assertion sees the first file's registry while `get_config()` has already
    moved on — one process, two answers."""
    plain = _write_config(tmp_path / "plain", "[envs]\nimpls = []\n")
    custom = _write_config(tmp_path / "custom", f"""
        [envs]
        impls = ["{_HERE}:_ResetProbeEnv"]
    """)

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(plain))
    assert "reset_probe_env" not in get_env_registry()

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(custom))
    reset_config()

    assert get_config().config_path() == custom
    assert get_env_registry()["reset_probe_env"] is _ResetProbeEnv


def test_the_config_and_its_registries_never_disagree_about_the_file(monkeypatch, tmp_path):
    """Stated as the invariant rather than the symptom, because the symptom is per-registry
    and the invariant is what a sixth registry would have to keep."""
    custom = _write_config(tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_ResetProbeEnv"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(custom))
    reset_config()

    assert ("reset_probe_env" in get_env_registry()) is (get_config().config_path() == custom)


_REGISTRIES = [
    "env_registry",
    "artifact_registry",
    "artifact_type_aliases",
    "task_step_registry",
    "sandbox_registry",
    "state_registry",
]


@pytest.mark.parametrize("name", _REGISTRIES)
def test_a_registry_is_memoized_on_the_config_that_built_it(name):
    """Once per Config, not once per process: the old globals outlived their Config."""
    config = get_config()

    first = getattr(config, name)()
    second = getattr(config, name)()

    assert first is second


def test_two_configs_on_different_files_hold_different_registries(monkeypatch, tmp_path):
    """The property module globals could not have: two Configs never share a registry."""
    plain = _write_config(tmp_path / "plain", "[envs]\nimpls = []\n")
    custom = _write_config(tmp_path / "custom", f"""
        [envs]
        impls = ["{_HERE}:_ResetProbeEnv"]
    """)

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(plain))
    on_plain = Config()
    assert "reset_probe_env" not in on_plain.env_registry()   # pins the plain document

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(custom))
    on_custom = Config()

    assert on_custom.env_registry()["reset_probe_env"] is _ResetProbeEnv
    assert "reset_probe_env" not in on_plain.env_registry()


def test_the_module_accessor_serves_the_process_config(monkeypatch, tmp_path):
    """A delegation now: memoizing here again would let it outvote the process Config."""
    custom = _write_config(tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_ResetProbeEnv"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(custom))
    reset_config()

    assert get_env_registry() is get_config().env_registry()


def test_artifact_aliases_are_checked_against_the_same_document(monkeypatch, tmp_path):
    """Reading the ambient aliases made a Config on an alias-free file raise about the
    other file's alias."""
    plain = _write_config(tmp_path / "plain", "[artifacts]\nimpls = []\n")
    aliased = _write_config(tmp_path / "aliased", f"""
        [artifacts]
        impls = ["{_HERE}:_ProbeArtifact"]
        type_aliases = {{ legacy_probe = "probe_artifact" }}
    """)

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(aliased))
    reset_config()
    assert get_config().artifact_type_aliases() == {"legacy_probe": "probe_artifact"}

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(plain))
    on_plain = Config()

    assert "probe_artifact" not in on_plain.artifact_registry()


def test_an_accessor_ignores_a_patched_re_export(monkeypatch, tmp_path):
    """Accessors use ``runtime.get_config``: consumer suites patch the re-exported name
    with doubles, and a MagicMock would answer ``env_registry()`` with a mock."""
    monkeypatch.setattr("agent_env.config.get_config", lambda: object())

    assert get_env_registry() is get_config().env_registry()


# Singletons that outlive any one Config, so nothing they cache may outlive a re-point.
_INSTANCE_STORES = [
    get_a2a_agent_instance_store, get_env_artifact_store, get_env_snapshot_store,
    get_env_instance_store, get_env_state_instance_store, get_review_store,
    get_task_instance_store,
]
_ENTITY_STORES = [
    get_a2a_agent_store, get_artifact_store, get_eval_store, get_env_store,
    get_task_step_store, get_task_store,
]


def _sqlite_config(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    return _write_config(directory, f"""
        [stores.document]
        impl = "agent_env.store.document_store.sqlite_document_store:LocalSqliteDocumentStore"
        [stores.document.config]
        path = "{directory}/documents.db"
    """)


@pytest.mark.parametrize("get_singleton", _INSTANCE_STORES, ids=lambda f: f.__name__)
def test_a_store_singleton_follows_the_config(get_singleton, monkeypatch, tmp_path):
    """A cached backend outlived `reset_config()`: the process read the new config and wrote
    the old database, saying nothing."""
    singleton = get_singleton()

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_sqlite_config(tmp_path, "first")))
    reset_config()
    first = singleton._doc_store

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_sqlite_config(tmp_path, "second")))
    reset_config()

    assert singleton._doc_store is get_config().get_document_store()
    assert singleton._doc_store is not first


@pytest.mark.parametrize("get_singleton", _ENTITY_STORES, ids=lambda f: f.__name__)
def test_an_entity_store_rebuilds_its_versioned_view(get_singleton, monkeypatch, tmp_path):
    """`get`/`put` read the versioned view, not `_doc_store`, and it is memoized because
    building one runs ensure_index — so it is the second thing a re-point has to invalidate."""
    singleton = get_singleton()

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_sqlite_config(tmp_path, "first")))
    reset_config()
    first = singleton._versioned
    assert singleton._versioned is first  # still memoized while the backend is unchanged

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_sqlite_config(tmp_path, "second")))
    reset_config()

    assert singleton._versioned is not first
    assert singleton._versioned._doc_store is get_config().get_document_store()


def test_resetting_a_config_imports_nothing(monkeypatch):
    """A process that never built a registry cannot have a stale one, and `reset_config()` is
    called per test by conftest and per `--stage` flag by the sdk — neither should pay to
    import ~60 task-step modules to clear caches that were never filled."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    imported = []
    monkeypatch.setattr("builtins.__import__",
                        lambda name, *a, **k: (imported.append(name), real_import(name, *a, **k))[1])

    reset_config()

    assert [name for name in imported if name.startswith("agent_env.")] == []
