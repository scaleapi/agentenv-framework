"""The plugin report's format version and reason codes.

Internal: the contract is the PLUGINS.md section "Plugin report format". A code says why a
contribution has its status, or why a group, the config or discovery failed. Codes are never
removed or reused; a new one needs no format bump.
"""

from __future__ import annotations

# The `format_version` of `agent-env plugin list / show / check --json`. Raised only for a change
# a consumer cannot ignore: a key removed, renamed or retyped, a new status, or a code redefined.
FORMAT_VERSION = 1

LOAD_FAILED = "load-failed"
INVALID_PLUGIN = "invalid-plugin"
INCOMPATIBLE_CORE = "incompatible-core"
BUILTIN_NAME = "builtin-name"
NAME_CONFLICT = "name-conflict"
REPLACED_BY_CONFIG = "replaced-by-config"
ALREADY_ATTACHED = "already-attached"
NOT_LOADED = "not-loaded"
STATUS_UNKNOWN = "status-unknown"
CONFIG_NOT_FOUND = "config-not-found"
CONFIG_UNREADABLE = "config-unreadable"
CONFIG_INVALID = "config-invalid"
GROUP_BUILD_FAILED = "group-build-failed"
ENTRY_POINTS_UNREADABLE = "entry-points-unreadable"
QUALIFIED_ONLY = "qualified-only"

# Where each code may appear: a contribution status, or "error" for a group, config or discovery error.
CODES: dict[str, frozenset[str]] = {
    LOAD_FAILED: frozenset({"failed"}),
    INVALID_PLUGIN: frozenset({"failed"}),
    INCOMPATIBLE_CORE: frozenset({"failed"}),
    BUILTIN_NAME: frozenset({"skipped"}),
    NAME_CONFLICT: frozenset({"conflict", "skipped"}),
    REPLACED_BY_CONFIG: frozenset({"replaced"}),
    ALREADY_ATTACHED: frozenset({"active"}),
    NOT_LOADED: frozenset({"unloaded"}),
    STATUS_UNKNOWN: frozenset({"unloaded"}),
    CONFIG_NOT_FOUND: frozenset({"blocked", "error"}),
    CONFIG_UNREADABLE: frozenset({"blocked", "error"}),
    CONFIG_INVALID: frozenset({"blocked", "error"}),
    GROUP_BUILD_FAILED: frozenset({"blocked", "error"}),
    ENTRY_POINTS_UNREADABLE: frozenset({"error"}),
    QUALIFIED_ONLY: frozenset({"active"}),
}
