"""Resolving the config document: discover a path, parse it, answer sections from it.

The result is held by a ``Config``, which resolves once and keeps it. Nothing is cached here,
so two Configs can hold different documents on purpose and neither can drift from its own.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from agent_env.config import loader
from agent_env.config.errors import ConfigError


@dataclass(frozen=True)
class Snapshot:
    """Where the document came from, and what it said."""

    path: Optional[Path]
    _document: Mapping[str, Any]
    error: Optional[Exception] = None

    def section(self, *names: str) -> Mapping[str, Any]:
        """The table at ``names`` as a copy; ``{}`` when absent, an error when mis-shaped.

        Absent is ``{}`` because every caller has a built-in default. *Present but not a
        table* raises, because someone wrote it and meant something by it: ``sandbox = 5``
        used to leave the reader on its built-in ``local`` while `config show` printed the
        5 back, which is the report lying about a value the process never used. The aliased
        sections in ``runtime`` have always raised here; this is the same rule.

        Copied, not frozen: fifteen ``isinstance(x, dict)`` sites read a non-dict as absent,
        so a ``MappingProxyType`` would turn nested tables into silent defaults.
        """
        if self.error is not None:
            raise self.error
        node: Any = self._document
        walked: list[str] = []
        for name in names:
            if node is None:
                return {}
            self._require_table(node, walked)
            walked.append(name)
            node = node.get(name)
        if node is None:
            return {}
        self._require_table(node, walked)
        return copy.deepcopy(node)

    @staticmethod
    def _require_table(node: Any, walked: list) -> None:
        if not isinstance(node, dict):
            raise ConfigError(
                f"config.toml [{'.'.join(walked)}] must be a table, got {type(node).__name__}"
            )


def _parse(path: Optional[Path]) -> Snapshot:
    """A parse error rides on the snapshot so the path stays reportable — `config show` needs
    to name the file most when the file is the broken thing."""
    if path is None:
        return Snapshot(path=None, _document={})
    try:
        # A strict read, with no existence check in front of it: discovery returned this
        # path a moment ago, so a missing file is a race and must be loud. Checking first
        # and reading second would only narrow the window, which is not the same as closing
        # it — the open is the check.
        return Snapshot(path=path, _document=loader.read_config_file(path))
    except FileNotFoundError:
        return Snapshot(path=path, _document={}, error=ConfigError(
            f"config.toml at {path} vanished between discovery and read"))
    except (ConfigError, OSError) as e:
        return Snapshot(path=path, _document={}, error=e)


def resolve() -> Snapshot:
    """Discover and parse, with no caching — the Config holds the result."""
    return _parse(loader.discover_config_path())
