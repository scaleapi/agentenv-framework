"""Test-side helpers for the config surface.

`config_with_document` lives here rather than in `agent_env.config`: nothing in the shipped
package needs a way to choose a document without writing a file, so shipping one would be
production surface with no production caller. A test is allowed to know internals the
package should not expose. `install_model_config` is the same: a test's `[model]` installed
as the process-wide config, which the package never needs to do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from agent_env.config import runtime, snapshot
from agent_env.config.model import ENV_API_KEY, ENV_BASE_URL, ModelConfig
from agent_env.store import LocalSecretStore


def config_with_document(document: Mapping[str, Any], *, path: Optional[Path] = None,
                         **kwargs: Any) -> runtime.Config:
    """A `Config` reading `document`, with no file on disk and no discovery.

    Returns the Config rather than installing one process-wide: a Config holds its own
    document, so a caller that built its own would not see a process-wide install.
    """
    config = runtime.Config(**kwargs)
    config._snapshot = snapshot.Snapshot(path=path, _document=document)
    return config


def install_model_config(monkeypatch: pytest.MonkeyPatch, *, secrets: Optional[Mapping[str, str]] = None,
                         **model: Any) -> runtime.Config:
    """The process-wide `Config` for this test, its `[model]` `ModelConfig(**model)` and its secrets `secrets` (the
    environment's otherwise), with `LITELLM_BASE_URL` and `LITELLM_API_KEY` cleared. Installed as the singleton, so
    every `get_config()` returns it, wherever it was imported from."""
    config = runtime.Config()
    if secrets is not None:
        config.set_secret_store(LocalSecretStore(values=dict(secrets), use_env=False))
    monkeypatch.setattr(config, "_model_cfg", ModelConfig(**model))
    monkeypatch.setattr(runtime, "_config", config)
    monkeypatch.delenv(ENV_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    return config
