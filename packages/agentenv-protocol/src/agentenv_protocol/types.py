"""Wire types for the v1 data-plane protocol."""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


def error_body(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


PROTOCOL_VERSION = "1.0"

WELL_KNOWN_PATH = "/.well-known/agent-env.json"
RPC_PATH = "/agentenv"

# A card declares its MCP endpoint as an `additionalInterfaces` entry with this transport; a card
# without one is reached at MCP_PATH by convention.
MCP_TRANSPORT = "mcp"
MCP_PATH = "/mcp"

METHOD_RESET = "data/reset"
METHOD_ADD = "data/add"
METHOD_GET = "data/get"

# The vehicle for the data-plane intake declaration: a declaration-only extension
# whose `params` carry an `IntakeDeclaration`. No endpoint — it only describes what `data/add`
# accepts and what `data/get` returns, for discoverability + pre-deploy fit checks. Absence of the
# extension means "no claim" (skip fit-checks), matching the `supports_v1` 404->legacy precedent.
INTAKE_EXTENSION_URI = "urn:agentenv:intake/v1"


class FileWithBytes(BaseModel):
    bytes: str
    mimeType: Optional[str] = None
    name: Optional[str] = None


class FileWithUri(BaseModel):
    uri: str
    mimeType: Optional[str] = None
    name: Optional[str] = None


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str
    metadata: Optional[dict] = None


class FilePart(BaseModel):
    kind: Literal["file"] = "file"
    file: Union[FileWithBytes, FileWithUri]
    metadata: Optional[dict] = None


class DataPart(BaseModel):
    kind: Literal["data"] = "data"
    data: dict
    metadata: Optional[dict] = None


Part = Annotated[Union[TextPart, FilePart, DataPart], Field(discriminator="kind")]


class AddDataRequest(BaseModel):
    parts: list[Part] = Field(min_length=1)


class ResetDataResponse(BaseModel):
    pass


class AddDataResponse(BaseModel):
    pass


class GetDataResponse(BaseModel):
    parts: list[Part]


class EnvironmentInterface(BaseModel):
    url: str
    transport: str


class EnvironmentExtension(BaseModel):
    uri: str
    description: Optional[str] = None
    params: Optional[dict] = None
    required: Optional[bool] = None


class IntakeFormat(BaseModel):
    """One accepted (on `data/add`) or returned (on `data/get`) content shape.

    Describes a single way data crosses the fixed `AddDataRequest` envelope: which `Part` `kind`
    carries it (`data`/`file`/`text`), the content `format` (e.g. `"json"`, `"zip-bundle"`,
    `"csv"`), the `mimeTypes` that select it, and whether loading `replace`s existing data or is
    `additive`. `bundleLayout` names a non-schema-able file-tree convention (e.g.
    `"data-json+root/v1"`); `tables` optionally carries per-table JSON-Schemas of the loadable
    shape (layer 4 — populate only where derivable, e.g. from a server's SQLAlchemy models).
    """

    part: Literal["data", "file", "text"]
    format: str
    mimeTypes: Optional[list[str]] = None
    load: Optional[Literal["additive", "replace"]] = None
    bundleLayout: Optional[str] = None
    tables: Optional[dict[str, dict]] = None


class IntakeDeclaration(BaseModel):
    """What a server's v1 data plane accepts on `data/add` and returns on `data/get`.

    Advertised on the `EnvironmentCard` under `INTAKE_EXTENSION_URI`. Layers 1-3 (part
    kind, format + mimeTypes, load semantics) are enough for fit-checks; layer 4 (`tables`) is
    optional and highest value. Absence of the whole declaration = "no claim".
    """

    add: Optional[list[IntakeFormat]] = None
    get: Optional[list[IntakeFormat]] = None


def intake_extension(
    declaration: IntakeDeclaration, description: Optional[str] = None
) -> "EnvironmentExtension":
    """Wrap an `IntakeDeclaration` as the declaration-only `INTAKE_EXTENSION_URI` extension.

    `None` params are dropped so the advertised card stays terse; read it back with
    `client.intake_declaration(card)`.
    """
    return EnvironmentExtension(
        uri=INTAKE_EXTENSION_URI,
        description=description
        or "Declares what the v1 data plane accepts on data/add and returns on data/get.",
        params=declaration.model_dump(exclude_none=True),
    )


class EnvironmentTool(BaseModel):
    name: str
    description: Optional[str] = None
    inputSchema: Optional[dict] = None


class EnvironmentCapabilities(BaseModel):
    extensions: Optional[list[EnvironmentExtension]] = None
    tools: Optional[list[EnvironmentTool]] = None
    # Wire methods served at /agentenv. [] = none; absent = pre-advertisement card (full data trio).
    operations: Optional[list[str]] = None


class EnvironmentCard(BaseModel):
    name: str
    protocolVersion: str = PROTOCOL_VERSION
    url: str = RPC_PATH
    preferredTransport: str = "JSONRPC"
    additionalInterfaces: list[EnvironmentInterface] = Field(default_factory=list)
    capabilities: EnvironmentCapabilities = Field(default_factory=EnvironmentCapabilities)
    children_environments: Optional[list[EnvironmentCard]] = None


EnvironmentCard.model_rebuild()
