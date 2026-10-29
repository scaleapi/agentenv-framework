"""Read-only inspection of what the config surface actually resolved to.

Answers "what am I pointed at" for both audiences the config serves: a bare install that
gets local stores with nothing said about it, and a deployment whose stage is whichever
file ``AGENT_ENV_CONFIG`` landed on. It reports the *resolution*, not the constructed
stores — no client is built, no network call is made, and nothing is written to disk.

Secret values cannot leak from here because nothing is interpolated: an ``env:`` /
``secret:`` reference is reported as the reference it is. Literal values under a
secret-shaped key, and the userinfo of a connection URI, are masked anyway — the schema
says secrets live behind references, but a report is the wrong place to find out otherwise.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Union

from packaging.utils import canonicalize_name

from agent_env.config import loader as config_loader
from agent_env.config import plugin_tables
from agent_env.config.errors import ConfigError
from agent_env.config.model import ModelConfig
from agent_env.config.provenance import (
    KIND_DEFAULT,
    KIND_ENV,
    KIND_FILE,
    Layer,
    class_name,
)
from agent_env.config.runtime import (
    ALIASED_SECTIONS,
    RESOLVED_KEYS,
    Config,
    validated_agents_section,
)

MASK = "***"

# How the winning file was found. A discriminator, not prose: `--json` exists so a tool can
# read the provenance, and prose it has to parse back is not provenance.
ConfigSource = Literal["env", "walk-up", "none", "error", "installed"]

SOURCE_ENV: ConfigSource = "env"
SOURCE_WALK_UP: ConfigSource = "walk-up"
SOURCE_NONE: ConfigSource = "none"
SOURCE_ERROR: ConfigSource = "error"
# A Config carrying its own document: discovery would not have found this path, so no
# discovery source may be claimed for it.
SOURCE_INSTALLED: ConfigSource = "installed"

# Substrings that make a key secret-shaped. Deliberately broad: over-masking costs a
# debugging round-trip, under-masking prints a credential into a terminal and a paste. On
# that trade `auth` is in even though it also catches `author` — a store `config` table is
# where `authorization = "Bearer ..."` lives, and a masked author is the cheaper mistake.
# `headers` masks a whole table of HTTP headers: a name like `X-Api-Key` doesn't say it holds a key.
_SECRET_KEY_PARTS = ("password", "passwd", "passphrase", "secret", "token", "credential",
                     "api_key", "private_key", "auth", "bearer", "headers")
_URI_USERINFO = re.compile(r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<userinfo>[^/@\s]+)@")


@dataclass(frozen=True)
class SectionReport:
    """One section: the layer that supplied it, the layers it beat, and the value it names.

    Masks itself on construction. Doing it at each call site is what this module tried
    first, and three separate leaks shipped that way — a scalar layer's `summary()` prints
    the value, so a single forgotten call publishes a secret. Here it cannot be forgotten.

    `value` is the masked section as resolved — an impl table for an aliased section, the
    file's table for a directly-read one, `None` when the winning layer had nothing to give.
    There is no combination of fields that describes a section twice.
    """

    name: str
    winner: Layer
    shadowed: tuple[Layer, ...] = ()
    value: Any = None
    unresolved: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "winner", _masked_layer(self.winner))
        object.__setattr__(self, "shadowed", tuple(_masked_layer(b) for b in self.shadowed))
        object.__setattr__(self, "value", _mask_table(self.value))

    @property
    def error(self) -> Optional[str]:
        """Why this section has no value — the winning layer would not read, or its value
        would not resolve. One question with one answer, wherever the failure happened."""
        return self.unresolved or self.winner.error

    @property
    def impl(self) -> Optional[str]:
        return self.value.get("impl") if isinstance(self.value, dict) else None

    @property
    def impl_name(self) -> str:
        return class_name(self.impl) or "(unresolved)"

    @property
    def config(self) -> dict:
        under = self.value.get("config") if isinstance(self.value, dict) else None
        return under if isinstance(under, dict) else {}


@dataclass(frozen=True)
class PluginOwner:
    """The distribution a ``[plugins.<name>]`` table belongs to, as installed, if it is."""

    name: str
    installed: Optional[plugin_tables.Installed] = None

    @property
    def version(self) -> Optional[str]:
        return self.installed.version if self.installed else None

    @property
    def plugin(self) -> bool:
        return bool(self.installed and self.installed.plugin)

    @property
    def core(self) -> bool:
        return bool(self.installed and self.installed.core)

    @property
    def label(self) -> str:
        if self.installed is None:
            return f"{self.name} (not installed)"
        version = f"{self.name} {self.version}".strip()
        if self.core:
            return f"{version} (agent-env itself)"
        return version if self.plugin else f"{version} (declares no agent_env entry point)"

    @property
    def note(self) -> str:
        """What core can say about a key in this table: whose it is, never whether it is read."""
        if self.plugin:
            return f"in the table of the plugin {self.label}; agent-env does not read or check it"
        if self.core:
            return (f"in the table of {self.name} {self.version}, which is agent-env itself, not a plugin; "
                    "agent-env does not read it")
        if self.installed:
            return (f"in the table of {self.name} {self.version}, which is installed but declares no "
                    "agent_env entry point; agent-env does not read it")
        return f"in the table of {self.name}, which is not installed; agent-env does not read it"


@dataclass(frozen=True)
class PluginTableReport:
    """One plugin's own settings table, as written; agent-env does not read it.

    Masked on construction, one table at a time: masking ``[plugins]`` whole would scope on the
    distribution names, and ``agentenv-oauth-bridge`` would hide its every setting.
    """

    owner: PluginOwner
    keys: tuple[str, ...]
    value: Any = None
    error: Optional[str] = None

    @property
    def where(self) -> str:
        return ", ".join(plugin_tables.where(key) for key in self.keys)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _mask_table(self.value))


@dataclass(frozen=True)
class PluginsReport:
    """Every ``[plugins.<name>]`` table, or why ``[plugins]`` itself could not be read."""

    tables: tuple[PluginTableReport, ...] = ()
    error: Optional[str] = None


@dataclass(frozen=True)
class ConfigReport:
    """The whole resolution chain: the file that won, and each section under it."""

    config_path: Optional[Path]
    config_source: ConfigSource
    search_root: Path
    sections: list[SectionReport] = field(default_factory=list)
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    plugins: PluginsReport = field(default_factory=PluginsReport)


def is_secret_key(key: str) -> bool:
    return any(part in key.lower() for part in _SECRET_KEY_PARTS)


def mask_value(key: str, value: Any, *, secret_scope: bool = False) -> Any:
    """Mask a config value for display. A reference passes through — it names a secret
    rather than carrying one, and the name is the useful half when debugging.

    ``secret_scope`` masks the value regardless of its own key, for a value that sits
    somewhere under a secret-shaped key rather than directly at one."""
    # Absence is not a secret. Masking it invents one: `"value": "***"` where the honest
    # answer is `null` tells an operator a key is configured when nothing supplies it.
    if value is None:
        return None
    if isinstance(value, str) and value.startswith(
        (config_loader.ENV_REF_PREFIX, config_loader.SECRET_REF_PREFIX)
    ):
        # The reference names a secret, but a `?default` after it is a literal like any other.
        ref, sep, default = value.partition("?")
        return f"{ref}?{mask_value(key, default, secret_scope=secret_scope)}" if sep else value
    # Before the string check: a secret is no less a secret for being written as a number.
    if secret_scope or is_secret_key(key):
        return MASK
    if not isinstance(value, str):
        return value
    return _URI_USERINFO.sub(lambda m: f"{m.group('scheme')}{MASK}@", value)


def _mask_table(table: Any, *, key: str = "", secret_scope: bool = False) -> Any:
    """Mask a resolved config subtree.

    A secret-shaped key masks *everything* beneath it, not just a scalar sitting directly
    at it: a backend's ``config`` is an arbitrary TOML table, so a credential arrives just
    as easily as ``tokens = ["…"]`` or ``credentials = { pw = "…" }``, and a mask that only
    consulted the leaf's own key would print both.
    """
    if isinstance(table, dict):
        return {k: _mask_table(v, key=k, secret_scope=secret_scope or is_secret_key(k))
                for k, v in table.items()}
    if isinstance(table, list):
        # A list carries no keys of its own — its items stay in the enclosing key's scope.
        return [_mask_table(v, key=key, secret_scope=secret_scope) for v in table]
    return mask_value(key, table, secret_scope=secret_scope)


@dataclass(frozen=True)
class Named:
    """A table whose keys the file chooses, such as ``[sandbox.providers]``: each holds ``shape``."""

    shape: "Shape"


# The keys a table may hold, each with its own shape. None is a value, or a table of free-form
# keys, that the unread-key check does not descend into: a seam's `config`, `[model.params]`.
Shape = Union[Mapping[str, Any], Named, None]

_SEAM: Mapping[str, Shape] = dict.fromkeys(config_loader.SEAM_KEYS)


def _resolved_keys(table: str) -> dict[str, Shape]:
    return dict.fromkeys(key.toml_path[1] for key in RESOLVED_KEYS if key.toml_path[0] == table)


# Sections read straight out of the file. Their fallbacks live in the reader, not here, so
# the report names where to look rather than guessing a value it would have to keep in sync.
@dataclass(frozen=True)
class FileSection:
    """A section with no env var and no alias — a table the reader takes as written.

    A type rather than a tuple so everything about a section lives on it: without one,
    its check, its lookup key, its prefix and its keys each needed a parallel table to live in.
    """

    name: str
    toml_path: tuple[str, ...]
    owner: str
    keys: Mapping[str, Shape]
    check: Optional[Callable[[Mapping[str, Any]], Any]] = None


_FILE_SECTIONS: tuple[FileSection, ...] = (
    FileSection("model", ("model",), "config.runtime", dict.fromkeys(f.name for f in fields(ModelConfig))),
    FileSection("conversations", ("conversations",), "config.runtime", _resolved_keys("conversations")),
    FileSection("agents", ("agents",), "config.runtime", _resolved_keys("agents"), validated_agents_section),
    FileSection("sandbox", ("sandbox",), "providers.sandbox_providers.sandbox_provider",
                {"default": None, "agent_default": None, "attribution": None, "providers": Named(_SEAM)}),
    FileSection("state", ("state",), "providers.env_state.env_state_provider", {"providers": Named(_SEAM)}),
    FileSection("envs", ("envs",), "env.registry", {"impls": None}),
    FileSection("artifacts", ("artifacts",), "artifact.registry", {"impls": None, "type_aliases": None}),
    FileSection("task_steps", ("task_steps",), "task_step.registry", {"impls": None}),
    FileSection("explorer", ("explorer",), "explorer.app",
                {**dict.fromkeys(("port", "cors_origins", "allowed_hosts", "static_dir")),
                 "plugins": {"impls": None}}),
)

_TOP_LEVEL_SECTIONS = ({s.toml_path[0] for s in _FILE_SECTIONS}
                       | {section.toml_path[0] for section in ALIASED_SECTIONS})


# The core tables with a key of their own named `plugins`: [explorer.plugins] lists explorer plugins.
_OWN_PLUGINS_KEY = frozenset({"explorer"})


def _misplaced(table: str, key: str) -> bool:
    if key == plugin_tables.TABLE:
        return table not in _OWN_PLUGINS_KEY
    return key in _TOP_LEVEL_SECTIONS


def _table(*path: str) -> str:
    """A table's header as the file writes it, each part quoted where TOML needs it."""
    return "[" + ".".join(plugin_tables.toml_key(part) for part in path) + "]"


def _misplaced_sections(document: Mapping[str, Any]) -> list[str]:
    """Section names appearing as a key *inside* another table.

    TOML binds a bare key to the table above it, so `envs = "..."` typed at the foot of a
    file becomes `[<last table>].envs`: a well-formed document that configures nothing.
    Nothing else can catch it — the section is simply absent — so the command reached for
    when config "is not taking effect" is the place to say so.
    """
    found = []
    for table, body in document.items():
        # Every key in [plugins] is a package name, even one spelled like a core section.
        if not isinstance(body, dict) or table == plugin_tables.TABLE:
            continue
        for key in body:
            if _misplaced(table, key):
                found.append(
                    f"{_table(table)} has a key named {key!r}, which is also a top-level section — "
                    f"a bare key binds to the table above it. Did you mean [{key}]?"
                )
    return found


def _inner(shape: Any, key: str) -> Any:
    if isinstance(shape, Named):
        return shape.shape
    return shape.get(key) if isinstance(shape, Mapping) else None


def _keys_shadowed_by_config(document: Mapping[str, Any], path: tuple = (), shape: Any = None) -> list[str]:
    """Keys set both beside a `config` table and inside it.

    Every reader takes `config`, so the one beside it is dead — and silently, because the
    section is well-formed and the value that does nothing looks like the value that does.
    """
    found = []
    for key, body in document.items():
        if not isinstance(body, dict):
            continue
        here = path + (key,)
        inner = _inner(shape, key)
        if inner is None:
            continue  # free-form, or a table nothing reads: a `config` in it is just a key
        config = body.get("config")
        if isinstance(config, dict) and isinstance(inner, Mapping) and "config" in inner:
            # not `impl`: the one beside the table names the class, the one inside is a
            # from_config kwarg. Both are read, which is why `lines` excludes it too.
            for duplicate in sorted(set(body) & set(config) - {"impl"}):
                found.append(
                    f"{_table(*here)} sets {duplicate!r} both beside its config table and "
                    f"inside it; only {_table(*here, 'config')} {duplicate} is read."
                )
        found += _keys_shadowed_by_config(body, here, inner)
    return found


def _masked_layer(layer: Layer, key: str = "") -> Layer:
    """A layer safe to publish. `summary()` on a table yields a class name *or* its
    scalars, and on a scalar the value itself — so any layer a report prints or serializes
    must be masked first, not just the resolved value beside it."""
    return layer if layer.error is not None else replace(layer, raw=_mask_table(layer.raw, key=key))


def _unresolved(config: Config, name: str, error: str) -> SectionReport:
    """A section whose winning layer could not be turned into one — still naming that layer.

    Reporting the default here would be the report telling a reassuring story about a layer
    that had no part in the failure.
    """
    try:
        candidates = config.section_candidates(name)
    except ConfigError:
        candidates = []
    winner = candidates[0] if candidates else Layer(KIND_DEFAULT, "(unresolved)")
    # Not when the winner already carries it: an unreadable layer and a layer whose value
    # would not resolve are different failures, but one layer cannot be both.
    return SectionReport(name=name, winner=winner, shadowed=tuple(candidates[1:]),
                         unresolved=None if winner.error else error)


def _describe_section(config: Config, name: str) -> SectionReport:
    try:
        traced = config.trace_section(name)
    except ConfigError as e:
        return _unresolved(config, name, str(e))
    return SectionReport(name=name, winner=traced.winner, shadowed=traced.shadowed,
                         value=traced.value)


def _describe_file_section(config: Config, section: "FileSection") -> SectionReport:
    """A section read straight out of the file — no env var, no alias. Its fallback lives in
    `section.owner`, so an absent table names that module rather than guessing a value."""
    layer = config.file_candidate(section.toml_path)
    if layer is None:
        return SectionReport(name=section.name,
                             winner=Layer(KIND_DEFAULT, f"default in {section.owner}"))
    if layer.error is not None:
        return SectionReport(name=section.name, winner=layer)
    try:
        # The shape rule lives in `section`, so the report cannot disagree with the reader
        # about it. Echoing `sandbox = 5` back while the reader refuses it is this command
        # confirming a value the process never used.
        value = config.section(*section.toml_path)
        if section.check is not None:
            section.check(value)
    except ConfigError as e:
        return SectionReport(name=section.name, winner=layer, unresolved=str(e))
    return SectionReport(name=section.name, winner=layer, value=value)


@dataclass(frozen=True)
class Candidate:
    """One path the loader would consider, and whether it is there."""

    path: Path
    why: str
    exists: bool


def search_path() -> list[Candidate]:
    """Every path discovery would consider, in the order it considers them.

    Answers "why is agent-env reading that file" without resolving anything, so it still
    reports when resolution itself is what is failing.
    """
    return [Candidate(path, why, path.is_file())
            for path, why in config_loader.config_search_path()]


# The live object each store section's getter returns, when one has been built or
# installed. `set_*_store()` writes straight to these.
_LIVE_STORES = {"document": "_document_store", "object": "_object_store",
                "image": "_image_store", "secret": "_secret_store"}


def _live_store_disagreements(config: Config, sections: list[SectionReport]) -> list[str]:
    """Sections whose live object is not what the config resolves to.

    `get_document_store()` returns an installed object verbatim, so a report naming the
    file's backend would describe a store the process is not using -- and that is the
    failure this whole module exists to catch, so it cannot be allowed in through the
    report's own back door.
    """
    found = []
    by_name = {s.name: s for s in sections}
    for name, attr in _LIVE_STORES.items():
        live = getattr(config, attr, None)
        section = by_name.get(name)
        if live is None or section is None:
            continue
        resolved = class_name(section.impl)
        actual = type(live).__name__
        if resolved is not None and resolved != actual:
            found.append(
                f"[stores.{name}] resolves to {resolved}, but this config already holds a "
                f"live {actual} -- an installed store, or one built before the config moved."
            )
    return found


def describe_config(config: Optional[Config] = None) -> ConfigReport:
    """Report what the config surface resolves to, without side effects.

    With no argument this builds its own `Config`, so the report describes what a *fresh*
    resolution would pick rather than any store a caller installed. That is the right answer
    for a CLI, which is asked what a command *would* use.

    A caller that passes its own `Config` gets a report about **that object**, installed
    state and all. A long-running service needs this and a fresh resolution actively
    misleads there: it calls `configure()` at startup, so a fresh report describes a
    process that does not exist rather than the one serving the request — the same
    reassuring story this module refuses everywhere else, one level out.

    The report then reflects installed doubles, deliberately.
    """
    config = config if config is not None else Config()
    search_root = Path.cwd().resolve()
    # Shared with `sources`, so the two reports cannot disagree about whether a file is in
    # play. A file that won't parse comes back as the report's error rather than a
    # traceback — this command is precisely what gets reached for then — and the sections
    # are still reported, because a higher layer may be resolving over the top of it.
    config_path, config_source, file_error = _classify_file(config)
    if config_source == SOURCE_ERROR:
        return ConfigReport(config_path=None, config_source=SOURCE_ERROR,
                            search_root=search_root, error=file_error,
                            plugins=PluginsReport(error=file_error))

    document = {} if file_error else config.config_file()
    # A plugin's table is the plugin's to read, so no claim is made about a `config` key in it.
    core = {k: v for k, v in document.items() if k != plugin_tables.TABLE}
    warnings = (_misplaced_sections(document) + _keys_shadowed_by_config(core, (), _KNOWN)
                + _misspelled_plugins_table(document) + _unread(document, _KNOWN))

    sections = [_describe(config, section) for section in _SECTIONS]
    warnings += _live_store_disagreements(config, sections)
    return ConfigReport(config_path=config_path, config_source=config_source, warnings=warnings,
                        search_root=search_root, sections=sections, error=file_error,
                        plugins=_describe_plugins(config))


def _near_plugins(key: str) -> bool:
    return key != plugin_tables.TABLE and bool(
        difflib.get_close_matches(key.lower(), [plugin_tables.TABLE], cutoff=0.85))


def _misspelled_plugins_table(document: Mapping[str, Any]) -> list[str]:
    """A top-level table one letter from ``[plugins]``: nothing reads it, and the plugin whose
    settings it holds silently gets none."""
    return [
        f"agent-env does not read {_table(key)}: a plugin's settings go under "
        f"[{plugin_tables.TABLE}.<package>]. Did you mean [{plugin_tables.TABLE}]?"
        for key in document
        if _near_plugins(key)
    ]


# How close a key must be to a known one to be offered as the one meant: `implz` is 0.8 from
# `impls`, and `export`, 0.71 from `explorer`, gets no advice rather than wrong advice.
_NEAR = 0.75


def _fits(body: Any, shape: Shape) -> bool:
    """Whether what the file wrote could be what a known key takes: a table where one is taken,
    sharing a key with it, or for a seam the string that names its class or backend."""
    if isinstance(shape, Named):
        return isinstance(body, dict)
    if shape is _SEAM and isinstance(body, str):
        return True
    if isinstance(shape, Mapping):
        return isinstance(body, dict) and (not body or bool(set(body) & set(shape)))
    return True


def _near(key: str, known: Mapping[str, Any], body: Any = None) -> Optional[str]:
    folded = {k.lower().replace("-", "_"): k for k in known if _fits(body, known[k])}
    match = difflib.get_close_matches(key.lower().replace("-", "_"), list(folded), n=1, cutoff=_NEAR)
    return folded[match[0]] if match else None


def _reported_elsewhere(table: Mapping[str, Any], path: tuple[str, ...], key: str,
                        shape: Mapping[str, Shape]) -> bool:
    """Whether another warning already names this key, so it is reported once."""
    if not path:
        return _near_plugins(key)
    if len(path) == 1 and _misplaced(path[0], key):
        return True
    # the shadowed-by-config warning covers only a table that takes `config`
    config = table.get("config")
    return "config" in shape and isinstance(config, dict) and key in config and key != "impl"


def _listed(names: list[str]) -> str:
    quoted = [repr(n) for n in names]
    if not quoted:
        return "no keys"
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _tables_taking(key: str) -> list[str]:
    """The tables that take a key named ``key``: where a key written above every header belongs."""
    return [f"[{name}]" for name, shape in _KNOWN.items() if isinstance(shape, Mapping) and key in shape]


def _unread_message(path: tuple[str, ...], key: str, shape: Mapping[str, Shape], body: Any = None) -> str:
    if path and (key in _TOP_LEVEL_SECTIONS or key == plugin_tables.TABLE):
        return (f"{_table(*path)} has a key named {key!r}, which is also a top-level section — "
                f"a bare key binds to the table above it. Did you mean [{key}]?")
    near = _near(key, shape, body)
    if not path and not isinstance(body, dict):
        # A bare key above the first header: a key, not a table.
        if near and near != plugin_tables.TABLE:
            return f"agent-env does not read the top-level key {key!r}. Did you mean {near!r}?"
        tables = _tables_taking(key)
        takes = "takes" if len(tables) == 1 else "take"
        where = f"; {' and '.join(tables)} {takes} a key of that name" if tables else ""
        return f"The top-level key {key!r} is outside every table, and agent-env does not read it{where}."
    if not path:
        if near:
            return f"agent-env does not read {_table(key)}: it has no such table. Did you mean [{near}]?"
        return (f"agent-env does not read {_table(key)}: it has no such table, and a plugin's own settings "
                f"go under [{plugin_tables.TABLE}.<package>].")
    table = _table(*path)
    said = f"{table} has a key named {key!r}, which agent-env does not read"
    if near:
        return f"{said}. Did you mean {near!r}?"
    if shape is _SEAM:
        return (f"{said}: {table} takes {_listed(list(shape))}, and the class's own settings go "
                f"under {_table(*path, 'config')}.")
    return f"{said}: {table} takes {_listed(list(shape))}."


def _unread(document: Mapping[str, Any], shape: Mapping[str, Shape], path: tuple[str, ...] = ()) -> list[str]:
    """Tables and keys no reader takes, checked against the keys each section declares.

    Checks the file rather than what won, since a typo in a table an env var replaces is still
    a typo. A free-form table is not descended into, and a key another warning names is skipped.
    """
    found = []
    for key, body in document.items():
        if key not in shape:
            if not _reported_elsewhere(document, path, key, shape):
                found.append(_unread_message(path, key, shape, body))
            continue
        inner = shape[key]
        if isinstance(body, dict) and isinstance(inner, Named):
            for name, entry in body.items():
                if isinstance(entry, dict):
                    found += _unread(entry, inner.shape, path + (key, name))
        elif isinstance(body, dict) and isinstance(inner, Mapping):
            found += _unread(body, inner, path + (key,))
    return found


def _plugins_table(config: Config) -> tuple[Mapping[str, Any], Optional[str]]:
    try:
        return config.section(plugin_tables.TABLE), None
    except (ConfigError, OSError) as e:
        return {}, str(e)


def _plugin_table(config: Config, owner: PluginOwner, keys: list[str]) -> PluginTableReport:
    """One plugin's table, found as ``agent_env.plugins.settings`` finds it, so the two agree."""
    if len(keys) > 1:
        return PluginTableReport(owner, tuple(keys), error=plugin_tables.duplicate(owner.name, keys))
    try:
        return PluginTableReport(owner, tuple(keys), value=config.section(plugin_tables.TABLE, keys[0]))
    except ConfigError as e:
        return PluginTableReport(owner, tuple(keys), error=str(e))


def _describe_plugins(config: Config) -> PluginsReport:
    table, error = _plugins_table(config)
    if error is not None:
        return PluginsReport(error=error)
    names = plugin_tables.spellings(table)
    installed = plugin_tables.installed() if names else {}
    return PluginsReport(tables=tuple(
        _plugin_table(config, PluginOwner(name, installed.get(name)), keys)
        for name, keys in names.items()
    ))


# Every environment variable the config package reads, and the TOML path it shadows.
# Declared once so `config sources` cannot promise a complete checklist and then omit a
# variable, and so a new one is a test failure rather than a silent hole in the report.
@dataclass(frozen=True)
class EnvSource:
    name: str
    shadows: Optional[tuple[str, ...]] = None
    note: Optional[str] = None


@lru_cache(maxsize=1)
def env_sources() -> tuple[EnvSource, ...]:
    """Every environment variable agent-env reads, and the TOML path it shadows."""
    return tuple(
        [EnvSource(s.env_var, s.toml_path) for s in ALIASED_SECTIONS]
        + [
            EnvSource("LITELLM_API_KEY", ("model", "api_key")),
            EnvSource("LITELLM_BASE_URL", ("model", "base_url")),
            EnvSource("AGENT_ENV_HUMAN_A2A_URL", ("conversations", "default_human_a2a_url")),
        # Read by the resolver but shadowing nothing in the file — listed so the checklist
        # is complete, with no `shadows` because claiming one would be false.
            EnvSource("AGENT_SANDBOX_MODE", None, "no file equivalent"),
            EnvSource("AGENT_ENV_MODAL_REGION", None, "no file equivalent"),
            EnvSource("AGENT_ENV_MODAL_APP_NAME", None, "no file equivalent"),
            EnvSource("AGENT_ENV_FIXTURE_PREFIX", None, "no file equivalent"),
            EnvSource("MODAL_TOKEN_ID", None, "credential, no file equivalent"),
            EnvSource("MODAL_TOKEN_SECRET", None, "credential, no file equivalent"),
    ]
    )


_RESOLVED_BY_KEY = {key.name: key for key in RESOLVED_KEYS}


@dataclass(frozen=True)
class SourceReport:
    """One layer of the resolution chain, and whether it is contributing anything."""

    kind: str
    where: str
    present: bool
    detail: Optional[str] = None
    shadows: Optional[str] = None


def _classify_file(config: Config) -> tuple[Optional[Path], ConfigSource, Optional[str]]:
    """The winning file, how it was found, and why it could not be read. One place, because
    `describe_config` and `sources` disagreeing about whether a file is in play would be
    the reports contradicting each other."""
    try:
        path = config.config_path()
    except ConfigError as e:
        return None, SOURCE_ERROR, str(e)
    if path is None:
        return None, SOURCE_NONE, None
    # Against what discovery would find, not against the environment variable alone: a
    # supplied Config can carry a document discovery never chose, and crediting
    # $AGENT_ENV_CONFIG for that path would name a file the object never read.
    try:
        discovered = config_loader.discover_config_path()
    except (ConfigError, OSError):
        discovered = None
    if discovered != path:
        source: ConfigSource = SOURCE_INSTALLED
    else:
        source = SOURCE_ENV if os.getenv(config_loader.ENV_CONFIG_PATH) else SOURCE_WALK_UP
    try:
        config.config_file()
    except (ConfigError, OSError) as e:
        return path, source, f"{path}: {e}"
    return path, source, None


def sources(config: Optional[Config] = None) -> list[SourceReport]:
    """Every layer the resolver consults, **lowest precedence first**.

    `search_path` answers "which file", `describe_config` answers "what did each section
    resolve to"; this answers the question between them — which layers are in play at all.
    A layer contributing nothing is still listed, because "the file I edited is not being
    read" and "a variable I forgot about is overriding it" are what this is reached for,
    and neither is visible from a list of what won.

    Ordering is the array's own, not a numbered layer id: the numbering of the target
    precedence chain is not settled, and publishing one that later renumbers would be worse
    than publishing none.
    """
    config = config if config is not None else Config()
    out = [SourceReport(KIND_DEFAULT, "built-in defaults", True)]

    path, source, error = _classify_file(config)
    if source == SOURCE_ERROR:
        out.append(SourceReport(KIND_FILE, "(discovery failed)", False, error))
    elif path is None:
        out.append(SourceReport(KIND_FILE, "(no config file)", False,
                                "none found on the search path"))
    else:
        # How it was found is a layer distinction, not trivia: $AGENT_ENV_CONFIG is terminal
        # over a discovered file, so conflating the two hides which one could still apply.
        how = (f"via ${config_loader.ENV_CONFIG_PATH}" if source == SOURCE_ENV
               else "discovered by walking up")
        detail = how if error is None else f"{how}; unreadable: {error}"
        out.append(SourceReport(KIND_FILE, str(path), error is None, detail))

    # One row per variable: a name can be both a declared override and referenced by the
    # file, and printing it twice would read as two different sources.
    referenced = _env_references(config)
    for env in env_sources():
        value = os.getenv(env.name)
        detail = env.note if value is None else mask_value(env.name, value)
        where = referenced.pop(env.name, None)
        if where is not None:
            detail = "; ".join(filter(None, [detail, f"also referenced by {where}"]))
        out.append(SourceReport(KIND_ENV, f"${env.name}", value is not None, detail,
                                None if env.shadows is None else ".".join(env.shadows)))
    for name, where in sorted(referenced.items()):
        present = os.getenv(name) is not None
        # No `shadows`: a reference does not override that path, it supplies it — the
        # distinction the precedence chain turns on.
        out.append(SourceReport(KIND_ENV, f"${name}", present,
                                f"referenced by {where}" if present
                                else f"referenced by {where}, unset"))
    return out


def _env_references(config: Config) -> dict[str, str]:
    """`env:NAME` references the active file makes, name -> every path that references it.

    Not a precedence layer — a reference resolves *inside* the document that wrote it — but
    it is still an environment variable supplying a value, and a checklist that omitted it
    would leave an operator hunting for an override that is really a missing variable.
    """
    try:
        document = config.config_file()
    except (ConfigError, OSError):
        return {}
    found: dict[str, list[str]] = {}

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + (k,))
        elif isinstance(node, list):
            for item in node:
                walk(item, path)
        elif isinstance(node, str) and node.startswith(config_loader.ENV_REF_PREFIX):
            name = node[len(config_loader.ENV_REF_PREFIX):].partition("?")[0]
            # Every site, not the first: one variable feeding two keys is exactly the
            # coupling worth seeing, and naming one of them would hide the other.
            found.setdefault(name, []).append(".".join(path))

    walk(document, ())
    return {name: ", ".join(f"[{p}]" for p in sorted(set(paths)))
            for name, paths in found.items()}


@dataclass(frozen=True)
class PathReport:
    """One dotted path: what it resolves to, and which layer supplied it."""

    path: str
    value: Any = None
    winner: Optional[Layer] = None
    shadowed: tuple[Layer, ...] = ()
    section: Optional[str] = None
    error: Optional[str] = None
    file_only: bool = False
    plugin: Optional[PluginOwner] = None

    def __post_init__(self) -> None:
        # Masked on construction, never by remembering: doing it at each call site is what
        # this module tried first, and three separate leaks shipped that way — a scalar
        # layer's `summary()` prints the value, so one forgotten call publishes a secret.
        leaf = self.path.rsplit(".", 1)[-1]
        if self.winner is not None:
            object.__setattr__(self, "winner", _masked_layer(self.winner, leaf))
        object.__setattr__(self, "shadowed", tuple(_masked_layer(b, leaf) for b in self.shadowed))
        object.__setattr__(self, "value", _mask_table(self.value, key=leaf))


@dataclass(frozen=True)
class SectionsReport:
    """A path that is an *ancestor* of real sections, such as `stores`.

    Its own type rather than a `children` field on `PathReport`: such a path has no single
    winner and no value of its own, so every field describing one would have to be read as
    "not applicable here" — a second shape wearing the first one's clothes.
    """

    path: str
    children: tuple[PathReport, ...] = ()


Explained = Union[PathReport, SectionsReport]


# Both spellings resolve: an operator reads the TOML path off a config file and should not
# have to know the internal section name exists.
_SECTIONS: tuple = ALIASED_SECTIONS + _FILE_SECTIONS
_SECTION_BY_KEY = {k: s for s in _SECTIONS
                   for k in (s.name, ".".join(s.toml_path))}


def _known_tables() -> dict[str, Shape]:
    """Every table agent-env reads, nested as the file nests them, plus `[plugins]`, which it
    reads none of."""
    known: dict[str, Any] = {plugin_tables.TABLE: None}
    for section in _SECTIONS:
        *parents, leaf = section.toml_path
        node = known
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = section.keys if isinstance(section, FileSection) else _SEAM
    return known


_KNOWN = _known_tables()


def _describe(config: Config, section) -> SectionReport:
    """Whichever kind of section it is, described the way that kind resolves."""
    return (_describe_file_section(config, section) if isinstance(section, FileSection)
            else _describe_section(config, section.name))


def explain_path(path: str, config: Optional[Config] = None) -> Explained:
    """Where one dotted path's value came from — the same layers the resolver would take.

    A path below a section is read out of whichever layer *won that section*, not out of
    the file: reporting the file's nested value while an environment variable replaced the
    whole section is the reassuring story this module exists to refuse.
    """
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        return PathReport(path=path, error="empty path")
    key = ".".join(parts)
    config = config if config is not None else Config()
    if parts[0] == plugin_tables.TABLE:
        return _explain_plugins(config, path, _segments(path)[1:])

    section = _SECTION_BY_KEY.get(key)
    if section is not None:
        return _as_path_report(path, section.name, _describe(config, section))
    resolved = _RESOLVED_BY_KEY.get(key)
    if resolved is not None:
        return _explain_resolved_key(config, path, resolved)

    below = [s for s in _SECTIONS
             if len(parts) > len(s.toml_path) and parts[:len(s.toml_path)] == s.toml_path]
    if below:
        return _explain_below_section(config, path, below[0], parts[len(below[0].toml_path):])

    above = [s for s in _SECTIONS
             if len(s.toml_path) > len(parts) and s.toml_path[:len(parts)] == parts]
    if above:
        return SectionsReport(path=path, children=tuple(
            _as_path_report(".".join(s.toml_path), s.name, _describe(config, s)) for s in above))

    layer = config.file_candidate(parts)
    if layer is None:
        return PathReport(path=path, file_only=True)
    if layer.error is not None:
        return PathReport(path=path, winner=layer, error=layer.error, file_only=True)
    return PathReport(path=path, value=layer.raw, winner=layer, file_only=True)


def _segments(path: str) -> list[str]:
    """``path`` split on the dots TOML splits on: a ``"quoted"`` segment is one key, so a table
    label copied from ``config show``, such as ``plugins."AgentEnv.Toy"``, names that table."""
    out, current, quoted, escaped = [], "", False, False
    for char in path:
        if escaped:
            current, escaped = current + char, False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == "." and not quoted:
            out.append(current)
            current = ""
        else:
            current += char
    return [segment for segment in (*out, current) if segment]


def _plugin_named(rest: list[str], *known: Mapping[str, Any]) -> tuple[str, list[str]]:
    """The distribution a path below ``plugins.`` names, and the key path under its table. The
    first segment is the table, as TOML reads it; failing that, a dotted name as ``plugin list``
    prints one (``AgentEnv.Toy``), longest first."""
    first = canonicalize_name(rest[0])
    if any(first in names for names in known):
        return first, rest[1:]
    for names in known:
        for end in range(len(rest), 1, -1):
            if (name := canonicalize_name(".".join(rest[:end]))) in names:
                return name, rest[end:]
    return first, rest[1:]


def _explain_plugins(config: Config, path: str, rest: list[str]) -> Explained:
    """A path in ``[plugins]``: the file is its only layer, and agent-env does not read it; the
    report says whose table it is. Looked up by canonical name, as the plugin's own read is."""
    table, error = _plugins_table(config)
    if error is not None:
        return PathReport(path=path, section=plugin_tables.TABLE, error=error,
                          winner=Layer(KIND_FILE, f"[{plugin_tables.TABLE}]", error=error))
    names = plugin_tables.spellings(table)
    if not rest and not names:
        return PathReport(path=path, section=plugin_tables.TABLE)
    installed = plugin_tables.installed()
    if not rest:
        return SectionsReport(path=path, children=tuple(
            _explain_plugin(config, f"{plugin_tables.TABLE}.{name}",
                            PluginOwner(name, installed.get(name)), keys, [])
            for name, keys in names.items()))
    name, below = _plugin_named(rest, names, installed)
    owner = PluginOwner(name, installed.get(name))
    return _explain_plugin(config, path, owner, names.get(name, []), below)


def _explain_plugin(config: Config, path: str, owner: PluginOwner, keys: list[str],
                    below: list[str]) -> PathReport:
    if not keys:
        return PathReport(path=path, section=plugin_tables.TABLE, plugin=owner)
    found = _plugin_table(config, owner, keys)
    if found.error is not None:
        return PathReport(path=path, section=plugin_tables.TABLE, plugin=owner, error=found.error,
                          winner=Layer(KIND_FILE, found.where, error=found.error))
    # Walked in the masked table, so a key beneath a secret-shaped one stays masked.
    node: Any = found.value
    for part in below:
        node = node.get(part) if isinstance(node, dict) else None
    return PathReport(path=path, value=node, winner=Layer(KIND_FILE, found.where, found.value),
                      section=plugin_tables.TABLE, plugin=owner)


def _explain_resolved_key(config: Config, path: str, key) -> PathReport:
    """A key the file is not the only source for, resolved through the same
    `Config.key_candidates` the getter takes — so the two cannot disagree about the winner."""
    candidates = config.key_candidates(key)
    if not candidates:
        return PathReport(path=path)
    winner = candidates[0]
    if winner.error is not None:
        return PathReport(path=path, winner=winner, shadowed=tuple(candidates[1:]),
                          error=winner.error)
    return PathReport(path=path, value=winner.raw, winner=winner,
                      shadowed=tuple(candidates[1:]))


def _explain_below_section(config: Config, path: str, section, rest: tuple[str, ...]) -> PathReport:
    """A path inside a section: read out of the layer that won the section, never the file.

    Walking the *resolved* section is what keeps this honest — when an environment variable
    replaced the section, the file's nested key is not what the process reads, and a report
    naming it would be describing a value nothing uses.
    """
    described = _describe(config, section)
    if described.error is not None:
        return PathReport(path=path, winner=described.winner, shadowed=described.shadowed,
                          section=section.name, error=described.error)
    node: Any = described.value
    for part in rest:
        if not isinstance(node, dict) or part not in node:
            node = None
            break
        node = node[part]
    return PathReport(path=path, value=node, winner=described.winner,
                      shadowed=described.shadowed, section=section.name)


def _as_path_report(path: str, name: str, report: SectionReport) -> PathReport:
    return PathReport(path=path, value=report.value, winner=report.winner,
                      shadowed=report.shadowed, section=name, error=report.error)


def as_dict(report: ConfigReport) -> dict:
    """The report as JSON-able data. Provenance is preserved — every surveyed tool drops it
    from its machine-readable form, which makes that form useless for the one question
    worth asking.

    Lives here rather than in the CLI so a service can serve the same shape without
    importing a command module.
    """
    return {
        "config_path": str(report.config_path) if report.config_path else None,
        "config_source": report.config_source,
        "search_root": str(report.search_root),
        "error": report.error,
        "warnings": report.warnings,
        "sections": [
            {
                "name": s.name,
                "winner": {"kind": s.winner.kind, "where": s.winner.where, "value": s.winner.summary()},
                "shadowed": [{"kind": b.kind, "where": b.where, "value": b.summary(), "error": b.error}
                             for b in s.shadowed],
                "impl": s.impl,
                "config": s.config,
                "value": s.value,
                "error": s.error,
            }
            for s in report.sections
        ],
        "plugins": {
            "error": report.plugins.error,
            "tables": [
                {
                    "name": t.owner.name,
                    "installed": t.owner.installed is not None,
                    "version": t.owner.version,
                    "plugin": t.owner.plugin,
                    "keys": list(t.keys),
                    "value": t.value,
                    "error": t.error,
                }
                for t in report.plugins.tables
            ],
        },
    }
