"""Interface Manifest: a structured description of a server's data model that
interface renderers (CLI, GUI, ...) are generated from.

Servers build manifests from their API spec, validate them against their live
MCP tools, and serve them at ``INTERFACE_MANIFEST_PATH``, advertised on the
EnvironmentCard under ``GET_INTERFACES_EXTENSION_URI``. Producers and
consumers both import the schema, constants, and version rules from here.

Two layers: a structural core (:class:`InterfaceManifest`) and one projection
per interface (:class:`CliManifest` for the CLI). Manifests are structure
only — no environment data or per-environment behavior — and every operation
references an MCP tool by name, so renderers interact with a server purely
through the protocol and tool calls.

Skew tolerance: models ignore unknown fields, and :func:`manifest_compatible`
permits compatible version skew, so producers and consumers can deploy
independently.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Bumped when the manifest shape changes in a way renderers must notice.
MANIFEST_VERSION = "0.1.0"

# Where a server serves its manifest index (GET -> list of interface names);
# one interface's manifest is at f"{INTERFACE_MANIFEST_PATH}/{interface}".
INTERFACE_MANIFEST_PATH = "/agentenv/interface-manifest"

# EnvironmentCard extension advertising the (gateway-rewritten) index endpoint
# in `params.endpoint`. Absent = pre-extension image; `[]` from the endpoint =
# definitively no manifests.
GET_INTERFACES_EXTENSION_URI = "urn:agentenv:get-interfaces/v1"

OperationKind = Literal["list", "read", "create", "update", "delete"]

# CLI command name per core operation kind (``read`` renders as ``get``).
CLI_COMMANDS: Dict[str, str] = {
    "list": "list",
    "read": "get",
    "create": "create",
    "update": "update",
    "delete": "delete",
}

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def manifest_compatible(served: object, supported: str = MANIFEST_VERSION) -> bool:
    """Whether a manifest at version ``served`` is consumable by a renderer built for ``supported``.

    Same major is compatible; while the major is 0 (pre-1.0 semver) the minor
    must match too, so only patch-level skew is tolerated. A missing or
    unparseable version is incompatible.
    """
    if not isinstance(served, str):
        return False
    served_match = _VERSION_RE.match(served)
    supported_match = _VERSION_RE.match(supported)
    if not served_match or not supported_match:
        return False
    if served_match.group(1) != supported_match.group(1):
        return False
    if served_match.group(1) == "0" and served_match.group(2) != supported_match.group(2):
        return False
    return True


class ManifestModel(BaseModel):
    """Base for all manifest models: unknown fields from a newer producer are ignored."""

    model_config = ConfigDict(extra="ignore")


class FieldSpec(ManifestModel):
    """One field of an entity, as seen through the server's tools."""

    name: str
    type: str
    format: Optional[str] = None
    enum: Optional[List[str]] = None
    description: Optional[str] = None
    required: bool = False
    read_only: bool = False


class Relationship(ManifestModel):
    """A field that points at another entity (for detail-view linking)."""

    field: str
    references: str
    many: bool = False


class ParamSpec(ManifestModel):
    """A single input parameter of an operation/action, as the tool accepts it.

    Carries everything an interface projection needs to render the input
    without re-reading the spec: its type, array item type, whether it is
    required, and (for enum fields) the constrained set of allowed values.
    """

    name: str
    type: str
    item_type: Optional[str] = None
    required: bool = False
    enum: Optional[List[str]] = None


class Operation(ManifestModel):
    """A CRUD operation on an entity, backed by exactly one MCP tool."""

    kind: OperationKind
    tool: str
    summary: Optional[str] = None
    description: Optional[str] = None
    required: List[str] = Field(default_factory=list)
    params: List[ParamSpec] = Field(default_factory=list)


class Action(ManifestModel):
    """A non-CRUD tool (rendered as a button/command), backed by one MCP tool."""

    tool: str
    description: Optional[str] = None
    params: List[ParamSpec] = Field(default_factory=list)


class Entity(ManifestModel):
    """A record type: its fields, relationships, and per-entity operations."""

    name: str
    fields: List[FieldSpec] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    operations: List[Operation] = Field(default_factory=list)


class InterfaceManifest(ManifestModel):
    """The full structural description of a server."""

    service: str
    manifest_version: str = MANIFEST_VERSION
    entities: List[Entity] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)

    def to_json(self) -> str:
        """Deterministic, human-diffable JSON (trailing newline for POSIX).

        ``exclude_none`` drops absent optionals (``format``/``enum``/
        ``description``) so committed goldens stay compact; Pydantic consumers
        see absent == default anyway.
        """
        return self.model_dump_json(indent=2, exclude_none=True) + "\n"


# ── CLI interface projection ────────────────────────────────────────────────
#
# The command-oriented view of the structural core: per-entity commands
# (list/get/create/update/delete) + actions, each backed by one MCP tool with
# its required/optional params (enum fields carry their constrained options).


class CliCommand(ManifestModel):
    """One CLI command: an entity operation or action, backed by one MCP tool."""

    tool: str
    summary: Optional[str] = None
    description: Optional[str] = None
    params: List[ParamSpec] = Field(default_factory=list)


class CliEntity(ManifestModel):
    """An entity's CLI commands, keyed by verb (``list``/``get``/...)."""

    entity: str
    commands: Dict[str, CliCommand] = Field(default_factory=dict)


class CliAction(ManifestModel):
    """A non-CRUD tool exposed as its own top-level CLI command."""

    name: str
    tool: str
    params: List[ParamSpec] = Field(default_factory=list)
    description: Optional[str] = None


class CliManifest(ManifestModel):
    """The CLI projection of a server's structural core."""

    service: str
    manifest_version: str = MANIFEST_VERSION
    interface: Literal["cli"] = "cli"
    entities: List[CliEntity] = Field(default_factory=list)
    actions: List[CliAction] = Field(default_factory=list)

    def to_json(self) -> str:
        """Deterministic, human-diffable JSON (see :meth:`InterfaceManifest.to_json`)."""
        return self.model_dump_json(indent=2, exclude_none=True) + "\n"
