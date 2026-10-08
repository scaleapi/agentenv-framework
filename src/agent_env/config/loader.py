"""Loader for the ``.agentenv/config.toml`` config surface.

Turns a store section (an ``impl`` dotted path + a ``config`` table) into a
constructed store: resolve ``env:`` / ``secret:`` references (each with an optional
``?default``), import the class, guard it against the target ABC, and call
``from_config``. Secret values never live in the toml, only references. See the
README for the schema and examples.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import re
import tomllib
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path
from typing import Any, Callable, Optional

from packaging.requirements import Requirement

from agent_env.config.errors import ConfigError

logger = logging.getLogger(__name__)

DISTRIBUTION = "agentenv-framework"
_CANNOT_IMPORT_NAME = re.compile(r"cannot import name '([\w.]+)' from '([\w.]+)'")
_NAME_SEPARATORS = re.compile(r"[-_.]+")
_CONFIG_DIR = ".agentenv"
_CONFIG_FILE = "config.toml"
ENV_CONFIG_PATH = "AGENT_ENV_CONFIG"

SecretResolver = Callable[[str], Optional[str]]

ENV_REF_PREFIX = "env:"
SECRET_REF_PREFIX = "secret:"

# The keys a seam table may hold: the class, and the kwargs it is built with.
SEAM_KEYS = ("impl", "config")


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
    except ImportError as e:
        raise ConfigError(f"Cannot import impl {impl!r}: {e}{install_hint(e)}") from e
    except AttributeError as e:
        raise ConfigError(f"Cannot import impl {impl!r}: {e}") from e
    if not (isinstance(cls, type) and issubclass(cls, abc)):
        raise ConfigError(f"impl {impl!r} is not a subclass of {abc.__name__}")
    return cls


def _missing_module(error: ImportError) -> str | None:
    """The module an import failed to find. ``from pkg import mod`` on an absent ``mod`` names
    ``pkg``, and ``mod`` only as ``name_from`` (3.12+) or in the message (3.11)."""
    name_from = getattr(error, "name_from", None)
    if name_from is None and (match := _CANNOT_IMPORT_NAME.match(str(error))):
        return f"{match[2]}.{match[1]}"
    return f"{error.name}.{name_from}" if error.name and name_from else error.name


def install_hint(error: ImportError) -> str:
    """How to install the module ``error`` failed to find, when an extra of this distribution has it."""
    extra = _missing_extra(error)
    return f"; it needs the {extra!r} extra, as in pip install '{DISTRIBUTION}[{extra}]'" if extra else ""


def _missing_extra(error: ImportError) -> str | None:
    """The smallest extra of this distribution with a requirement named for the module ``error``
    failed to find: by name, because the missing distribution's files are not installed to read.
    Separators are dropped before comparing, as ``google-cloud-secret-manager`` installs
    ``google.cloud.secretmanager``."""
    module = _missing_module(error)
    if not module:
        return None
    try:
        package = metadata(DISTRIBUTION)
    except PackageNotFoundError:
        return None
    wanted = _NAME_SEPARATORS.sub("", module).lower()
    sizes: dict[str, int] = {}
    matches: set[str] = set()
    for line in package.get_all("Requires-Dist") or []:
        requirement = Requirement(line)
        if requirement.marker is None:
            continue
        for extra in package.get_all("Provides-Extra") or []:
            if requirement.marker.evaluate({"extra": extra}):
                sizes[extra] = sizes.get(extra, 0) + 1
                if _NAME_SEPARATORS.sub("", requirement.name).lower() == wanted:
                    matches.add(extra)
    return min(matches, key=lambda e: (sizes[e], e)) if matches else None


def build_store(section: dict, abc: type, *, secret_resolver: Optional[SecretResolver] = None):
    """Import the ``impl`` class named by ``section``, guard it against ``abc``,
    and construct it via ``from_config`` with the interpolated ``config`` table."""
    impl = section.get("impl")
    if not impl:
        raise ConfigError(f"Store section is missing an 'impl' dotted path: {section!r}")
    for key in section:
        if key not in SEAM_KEYS:
            logger.warning("Ignoring unknown key %r in store section (expected 'impl'/'config')", key)
    cls = load_impl(impl, abc)
    config = interpolate(section.get("config", {}), secret_resolver=secret_resolver)
    _check_config_keys(cls, abc, config, impl)
    return cls.from_config(**config)


def _check_config_keys(cls: type, abc: type, config: dict, impl: str | type) -> None:
    """Refuse a config table the store can't take as a ConfigError naming the keys, where Python
    would raise a TypeError that reads as a bug in the store. A class on its interface's default
    ``from_config`` hands the table to its constructor, so that is what the table is checked
    against; any other ``from_config`` is checked against its own parameters."""
    target = cls if getattr(cls.from_config, "__func__", None) is getattr(abc.from_config, "__func__", ...) else cls.from_config
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return
    named = [p for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    takes_any = any(p.kind is p.VAR_KEYWORD for p in parameters)
    unknown = [] if takes_any else sorted(set(config) - {p.name for p in named})
    missing = [p.name for p in named if p.default is p.empty and p.name not in config]
    if not unknown and not missing:
        return
    name = impl if isinstance(impl, str) else impl.__qualname__
    problems = [f"unknown key {', '.join(map(repr, unknown))}"] if unknown else []
    problems += [f"no value for {', '.join(map(repr, missing))}"] if missing else []
    raise ConfigError(
        f"The config table for {name!r} has {' and '.join(problems)}; "
        f"it takes {', '.join(p.name for p in named) or 'no keys'}"
    )
