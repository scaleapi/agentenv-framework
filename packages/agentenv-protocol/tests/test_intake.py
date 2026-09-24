"""Unit tests for the data-plane intake declaration."""

from __future__ import annotations

from agentenv_protocol import (
    INTAKE_EXTENSION_URI,
    EnvironmentCapabilities,
    EnvironmentCard,
    IntakeDeclaration,
    IntakeFormat,
    client as protocol_v1,
    intake_extension,
)


def _json_bundle_declaration() -> IntakeDeclaration:
    return IntakeDeclaration(
        add=[
            IntakeFormat(part="data", format="json", load="additive"),
            IntakeFormat(
                part="file",
                format="zip-bundle",
                mimeTypes=["application/zip"],
                load="replace",
                bundleLayout="data-json+root/v1",
            ),
        ],
        get=[IntakeFormat(part="data", format="json")],
    )


def test_intake_extension_uses_uri_and_drops_none_params():
    ext = intake_extension(_json_bundle_declaration())
    assert ext.uri == INTAKE_EXTENSION_URI
    assert ext.description
    add = ext.params["add"]
    # exclude_none keeps the declared fields terse: no null mimeTypes/tables leak onto the card.
    assert add[0] == {"part": "data", "format": "json", "load": "additive"}
    assert add[1]["bundleLayout"] == "data-json+root/v1"
    assert "tables" not in add[1] and "mimeTypes" not in add[0]
    assert ext.params["get"] == [{"part": "data", "format": "json"}]


def test_intake_declaration_reads_back_from_card():
    card = EnvironmentCard(
        name="items",
        capabilities=EnvironmentCapabilities(extensions=[intake_extension(_json_bundle_declaration())]),
    )
    declared = protocol_v1.intake_declaration(card.model_dump())
    assert declared is not None
    # round-trips through the typed model
    parsed = IntakeDeclaration.model_validate(declared)
    assert [f.format for f in parsed.add] == ["json", "zip-bundle"]
    assert parsed.add[1].load == "replace"


def test_intake_declaration_absent_is_no_claim():
    # No extension at all, or the intake extension without params, both read as "no claim" (None) —
    # never {}, which would read as "declares an empty intake".
    assert protocol_v1.intake_declaration({"name": "items"}) is None
    assert protocol_v1.intake_declaration({"capabilities": None}) is None
    bare = {"capabilities": {"extensions": [{"uri": INTAKE_EXTENSION_URI}]}}
    assert protocol_v1.intake_declaration(bare) is None


def test_empty_declaration_is_no_claim():
    # A declaration with no formats must serialize to {} (both fields None-defaulted and stripped
    # by exclude_none) so it reads back as "no claim" (None), never a truthy {"add": []} that a
    # fit-check consumer would mistake for a vacuous-but-present intake.
    assert intake_extension(IntakeDeclaration()).params == {}
    card = EnvironmentCard(
        name="items",
        capabilities=EnvironmentCapabilities(extensions=[intake_extension(IntakeDeclaration())]),
    )
    assert protocol_v1.intake_declaration(card.model_dump()) is None


def test_intake_tables_layer_is_optional_and_preserved():
    decl = IntakeDeclaration(
        add=[IntakeFormat(part="data", format="json", tables={"users": {"type": "object"}})]
    )
    ext = intake_extension(decl)
    assert ext.params["add"][0]["tables"] == {"users": {"type": "object"}}
