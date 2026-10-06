"""The Sail SDK, imported on first use and authenticated with the provider's configured key.

The Python SDK takes its key only from ``SAIL_API_KEY``, read once when it builds its process-wide
client. The key is set for that one build and the variable restored, so subprocesses never inherit it
and a process holds one Sail key.
"""

from __future__ import annotations

import hashlib
import os
import threading
from types import ModuleType
from typing import Any

from agent_env.config.errors import ConfigError

API_KEY_ENV = "SAIL_API_KEY"
RUNTIME_THREADS_ENV = "SAIL_RUNTIME_THREADS"

_lock = threading.Lock()
_installed_key: str | None = None
_apps: dict[str, Any] = {}


def _fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def connect(api_key: str, app_name: str, *, runtime_threads: int | None = None, sdk: ModuleType | Any | None = None) -> tuple[Any, Any]:
    """The SDK module and the Sail App ``app_name`` (minted if missing), authenticating on first use.

    Blocking: call it through ``asyncio.to_thread``.
    """
    global _installed_key
    if sdk is None:
        import sail as sdk
    fingerprint = _fingerprint(api_key)
    with _lock:
        if _installed_key is None:
            previous = os.environ.get(API_KEY_ENV)
            os.environ[API_KEY_ENV] = api_key
            if runtime_threads is not None:
                os.environ[RUNTIME_THREADS_ENV] = str(runtime_threads)
            try:
                sdk.reset_transports()
                _apps[app_name] = sdk.App.find(name=app_name, mint_if_missing=True)
            finally:
                if previous is None:
                    os.environ.pop(API_KEY_ENV, None)
                else:
                    os.environ[API_KEY_ENV] = previous
            _installed_key = fingerprint
        elif _installed_key != fingerprint:
            raise ConfigError(
                "a process can use one Sail API key: another [sandbox.providers.sail] key is already in use"
            )
        elif app_name not in _apps:
            _apps[app_name] = sdk.App.find(name=app_name, mint_if_missing=True)
        return sdk, _apps[app_name]
