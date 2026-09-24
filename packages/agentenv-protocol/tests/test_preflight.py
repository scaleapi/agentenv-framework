"""Unit tests for the data-plane intake fit-check."""
from __future__ import annotations

from agentenv_protocol import (
    IntakeDeclaration,
    IntakeFormat,
    intake_fit_check,
)


def _decl_with_tables() -> dict:
    return IntakeDeclaration(
        add=[
            IntakeFormat(
                part="data",
                format="json",
                load="additive",
                tables={
                    "users": {"required": ["id", "email"]},
                    "notes": {"required": ["id", "body"]},
                },
            ),
            IntakeFormat(part="file", format="zip-bundle", mimeTypes=["application/zip"]),
        ]
    ).model_dump(exclude_none=True)


def test_absent_declaration_is_no_claim():
    assert intake_fit_check(None, {"users": [{"id": 1}]}) == []


def test_matching_data_fits():
    data = {
        "users": [{"id": "u1", "email": "a@b.com"}],
        "notes": [{"id": "n1", "body": "hi"}],
    }
    assert intake_fit_check(_decl_with_tables(), data) == []


def test_extra_fields_tolerated():
    # export is a superset of the loadable shape — extra fields must not fail the check.
    data = {"users": [{"id": "u1", "email": "a@b.com", "computed_rank": 7}]}
    assert intake_fit_check(_decl_with_tables(), data) == []


def test_unknown_collection_flagged():
    data = {"slack_messages": [{"id": "m1"}]}
    issues = intake_fit_check(_decl_with_tables(), data)
    assert any("not accepted" in i and "slack_messages" in i for i in issues)


def test_missing_required_field_flagged():
    data = {"users": [{"id": "u1"}]}  # missing 'email'
    issues = intake_fit_check(_decl_with_tables(), data)
    assert any("missing required field" in i and "email" in i for i in issues)


def test_scalar_top_level_extras_tolerated():
    # Real universes ship scalar export extras (e.g. contacts.current_user_id) alongside
    # collections; those are not tables and must not be flagged as unaccepted collections.
    data = {"users": [{"id": "u1", "email": "a@b.com"}], "current_user_id": "u1"}
    assert intake_fit_check(_decl_with_tables(), data) == []


def test_format_not_accepted_flagged():
    file_only = IntakeDeclaration(
        add=[IntakeFormat(part="file", format="zip-bundle")]
    ).model_dump(exclude_none=True)
    issues = intake_fit_check(file_only, {"users": [{"id": "u1"}]})
    assert issues and "does not accept part=data" in issues[0]
