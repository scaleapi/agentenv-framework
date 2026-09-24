"""Interface Manifest contract: version skew rules + model round trips."""
import pytest

from agentenv_protocol.manifest import (
    MANIFEST_VERSION,
    CliManifest,
    InterfaceManifest,
    manifest_compatible,
)

CLI_MANIFEST = {
    "service": "crm",
    "manifest_version": MANIFEST_VERSION,
    "interface": "cli",
    "entities": [
        {
            "entity": "CRMContact",
            "commands": {
                "list": {
                    "tool": "crm_search_contacts",
                    "params": [{"name": "full_name", "type": "string", "required": True}],
                },
            },
        },
    ],
    "actions": [
        {
            "name": "show_data",
            "tool": "crm_show_data",
            "params": [{"name": "mode", "type": "string", "enum": ["raw", "summary"]}],
        },
    ],
}


@pytest.mark.parametrize(
    "served,supported,compatible",
    [
        ("0.1.0", "0.1.0", True),
        ("0.1.7", "0.1.0", True),  # patch skew tolerated
        ("0.2.0", "0.1.0", False),  # pre-1.0 minor bump is breaking
        ("1.0.0", "0.1.0", False),
        ("1.2.3", "1.0.0", True),  # post-1.0 minor skew tolerated
        ("2.0.0", "1.0.0", False),
        (None, "0.1.0", False),
        ("", "0.1.0", False),
        ("0.1", "0.1.0", False),
        ("garbage", "0.1.0", False),
    ],
)
def test_manifest_compatible(served, supported, compatible):
    assert manifest_compatible(served, supported) is compatible


def test_manifest_compatible_defaults_to_current_version():
    assert manifest_compatible(MANIFEST_VERSION)
    assert not manifest_compatible("9.9.9")


def test_cli_manifest_round_trips_deterministically():
    model = CliManifest.model_validate(CLI_MANIFEST)
    serialized = model.to_json()
    assert serialized.endswith("\n")
    assert CliManifest.model_validate_json(serialized) == model
    # Absent optionals are dropped so committed goldens stay compact.
    assert '"summary":' not in serialized
    assert '"description":' not in serialized


def test_unknown_fields_are_ignored_for_skew_tolerance():
    extended = {
        **CLI_MANIFEST,
        "color_scheme": "dark",
        "entities": [
            {**CLI_MANIFEST["entities"][0], "icon": "person"},
        ],
    }
    model = CliManifest.model_validate(extended)
    assert not hasattr(model, "color_scheme")
    assert not hasattr(model.entities[0], "icon")


def test_structural_core_defaults():
    manifest = InterfaceManifest(service="crm")
    assert manifest.manifest_version == MANIFEST_VERSION
    assert manifest.entities == [] and manifest.actions == []
