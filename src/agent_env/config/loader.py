"""Loader for the ``.agentenv/config.toml`` config surface.

Turns a store section (an ``impl`` dotted path + a ``config`` table) into a
constructed store: resolve ``env:`` / ``secret:`` references (each with an optional
``?default``), import the class, guard it against the target ABC, and call
``from_config``. Secret values never live in the toml, only references. See the
README for the schema and examples.
"""

from __future__ import annotations

import importlib
import logging
import os
import tomllib
from pathlib import Path
from typing import Any, Callable, Optional

from agent_env.config.errors import ConfigError

logger = logging.getLogger(__name__)

_CONFIG_DIR = ".agentenv"
_CONFIG_FILE = "config.toml"
ENV_CONFIG_PATH = "AGENT_ENV_CONFIG"

SecretResolver = Callable[[str], Optional[str]]

ENV_REF_PREFIX = "env:"
SECRET_REF_PREFIX = "secret:"


def discover_config_path(start: Optional[Path] = None) -> Optional[Path]:
    """Locate the config.toml: ``AGENT_ENV_CONFIG`` wins, else the nearest
    ``.agentenv/config.toml`` walking up from ``start`` (CWD by default).

    An explicit ``AGENT_ENV_CONFIG`` override must point to an existing file —
    fail loud rather than silently drop the whole config (and derive local-store
    paths from a non-existent directory) on a typo'd override path."""
    override = os.getenv(ENV_CONFIG_PATH)
    if override:
        path = Path(override)
        if not path.is_file():
            raise ConfigError(f"{ENV_CONFIG_PATH}={override!r} does not point to an existing file")
        return path
    for candidate in _walk_up_candidates(start):
        if candidate.is_file():
            return candidate
    return None


def _walk_up_candidates(start: Optional[Path] = None) -> list[Path]:
    here = (start or Path.cwd()).resolve()
    return [directory / _CONFIG_DIR / _CONFIG_FILE for directory in (here, *here.parents)]


def config_search_path(start: Optional[Path] = None) -> list[tuple[Path, str]]:
    """Every path ``discover_config_path`` would consider, in order, and why.

    Lives here because ``agent-env config debug`` exists to report discovery: a second copy
    of the walk would put the one command that must not lie about the search path in charge
    of guessing it.
    """
    override = os.getenv(ENV_CONFIG_PATH)
    if override:
        return [(Path(override), f"${ENV_CONFIG_PATH}")]
    return [(path, "walk-up") for path in _walk_up_candidates(start)]


_UTF8_BOM = b"\xef\xbb\xbf"


def load_config_file(path: Optional[Path]) -> dict:
    """Parse the config.toml at ``path`` (``{}`` when absent).

    Deliberately lenient about a path that is not there: this is a public helper and the
    sdk's exporter calls it with a freshly discovered path. Knowing that a path *was* just
    discovered — so that losing it is a race and not an absent config — belongs to the
    caller that discovered it, which is ``snapshot``; it uses ``read_config_file``.
    """
    if path is None or not path.is_file():
        return {}
    return read_config_file(path)


def read_config_file(path: Path) -> dict:
    """Parse the config.toml at ``path``, or raise.

    No existence check: the open *is* the check. Testing first and reading second leaves a
    window where the file can go, and a caller that has to ask "does it exist" before every
    read has a race rather than an answer.
    """
    with open(path, "rb") as f:
        raw = f.read()
    # An editor that writes a UTF-8 BOM produces a file correct to its author and rejected by
    # tomllib at line 1, column 1. The bytes carry no meaning here, so drop them.
    raw = raw[len(_UTF8_BOM):] if raw.startswith(_UTF8_BOM) else raw
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Malformed config.toml at {path}: {e}") from e
    except UnicodeDecodeError as e:
        raise ConfigError(f"config.toml at {path} is not valid UTF-8: {e}") from e


def interpolate(value: Any, *, secret_resolver: Optional[SecretResolver] = None) -> Any:
    """Resolve ``env:NAME`` / ``secret:KEY`` string references (each with an optional
    ``?default`` suffix), recursing into dicts/lists; pass any other value through
    unchanged. ``secret:`` needs a ``secret_resolver``; when none is available (e.g.
    while building the secret store itself) a ``secret:`` reference fails loud."""
    if isinstance(value, dict):
        return {k: interpolate(v, secret_resolver=secret_resolver) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, secret_resolver=secret_resolver) for v in value]
    if not isinstance(value, str):
        return value
    if value.startswith(ENV_REF_PREFIX):
        return _resolve_ref(ENV_REF_PREFIX, value, os.getenv)
    if value.startswith(SECRET_REF_PREFIX):
        if secret_resolver is None:
            raise ConfigError(f"Cannot resolve {value!r}: no secret store is available for secret: references here")
        return _resolve_ref(SECRET_REF_PREFIX, value, secret_resolver)
    return value


def _resolve_ref(prefix: str, value: str, resolve: SecretResolver) -> str:
    ref, sep, default = value[len(prefix):].partition("?")
    resolved = resolve(ref)
    if resolved is not None:
        return resolved
    if sep:
        return default
    raise ConfigError(f"Unresolved {prefix}{ref} reference (no value and no default)")


def load_impl(impl: str | type, abc: type) -> type:
    """Import the ``module.path:ClassName`` pointer ``impl`` and guard the class
    against its target interface ``abc``. Shared by every ``.agentenv`` seam that
    resolves a dotted impl pointer (stores, task steps). An already-loaded class, which is
    what an entry-point plugin registers, is only guarded."""
    if isinstance(impl, type):
        if not issubclass(impl, abc):
            raise ConfigError(f"impl {impl.__qualname__!r} is not a subclass of {abc.__name__}")
        return impl
    module_path, sep, attr = impl.partition(":")
    if not sep or not attr:
        raise ConfigError(f"impl must be 'module.path:ClassName', got {impl!r}")
    try:
        cls = getattr(importlib.import_module(module_path), attr)
    except (ImportError, AttributeError) as e:
        raise ConfigError(f"Cannot import impl {impl!r}: {e}") from e
    if not (isinstance(cls, type) and issubclass(cls, abc)):
        raise ConfigError(f"impl {impl!r} is not a subclass of {abc.__name__}")
    return cls


def build_store(section: dict, abc: type, *, secret_resolver: Optional[SecretResolver] = None):
    """Import the ``impl`` class named by ``section``, guard it against ``abc``,
    and construct it via ``from_config`` with the interpolated ``config`` table."""
    impl = section.get("impl")
    if not impl:
        raise ConfigError(f"Store section is missing an 'impl' dotted path: {section!r}")
    for key in section:
        if key not in ("impl", "config"):
            logger.warning("Ignoring unknown key %r in store section (expected 'impl'/'config')", key)
    cls = load_impl(impl, abc)
    config = interpolate(section.get("config", {}), secret_resolver=secret_resolver)
    return cls.from_config(**config)
