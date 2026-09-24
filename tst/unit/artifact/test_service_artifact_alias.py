"""EnvironmentArtifact drops the read-only service_name property and the
EnvironmentArtifact.put(service_name=) kwarg alias. The pydantic Field(alias="service_name")
stays — construction via service_name= and the serialized key remain service_name
(the additive wire contract) — but there is no longer a .service_name attribute.

These are the three distinct mechanisms that must not be conflated: the Field alias
(kept), the property (removed), and the put() kwarg alias (removed)."""

import pytest

from agent_env.artifact.artifacts.environment import EnvironmentArtifact


def test_construct_via_alias_and_field_name():
    a = EnvironmentArtifact(id="x", service_name="foo")
    assert a.environment_name == "foo"
    b = EnvironmentArtifact(id="y", environment_name="bar")
    assert b.environment_name == "bar"


def test_no_service_name_attribute():
    a = EnvironmentArtifact(id="x", environment_name="foo")
    with pytest.raises(AttributeError):
        _ = a.service_name


def test_serialized_doc_carries_both_name_spellings():
    """The dual-write adds ``environment_name`` beside the legacy wire key. The
    legacy key must survive verbatim — a dual-write that drops it is a rename,
    and old SDKs hard-read it."""
    a = EnvironmentArtifact(id="x", environment_name="foo")
    d = a.model_dump(by_alias=True)
    assert d["service_name"] == "foo"
    assert d["environment_name"] == "foo"


def test_loads_legacy_doc_keyed_service_name():
    c = EnvironmentArtifact.model_validate(
        {"id": "z", "type": "environment", "service_name": "svc", "service_version": 3}
    )
    assert c.environment_name == "svc"


def test_put_without_a_name_raises_valueerror_not_validationerror():
    """Same guard as MCPServerEnv/WebsiteEnv: name the missing argument rather than
    letting environment_name=None reach the model as a Pydantic ValidationError."""
    with pytest.raises(ValueError, match="environment_name cannot be empty"):
        EnvironmentArtifact.put(id="x", file_artifact=object())  # type: ignore[arg-type]


def test_put_rejects_service_name_kwarg():
    with pytest.raises(TypeError):
        EnvironmentArtifact.put(
            id="x",
            service_name="b",
            file_artifact=object(),  # type: ignore[arg-type]
        )



def test_loads_a_legacy_doc_still_carrying_service_version():
    """The field is deleted, but stored docs carry the key for ever; pydantic
    extra='ignore' must drop it rather than choke on it."""
    a = EnvironmentArtifact.model_validate(
        {"id": "z", "type": "environment", "service_name": "svc", "service_version": 9}
    )
    assert a.environment_name == "svc"
    assert not hasattr(a, "service_version")


def test_service_version_is_not_written_back():
    """A doc read with the stale key must not re-emit it — otherwise reading and
    re-putting would resurrect the field one document at a time."""
    a = EnvironmentArtifact.model_validate(
        {"id": "z", "type": "environment", "service_name": "svc", "service_version": 9}
    )
    assert "service_version" not in a.model_dump(by_alias=True)


def test_only_environment_name_is_required():
    """`id` comes from Artifact; nothing else about this model is required."""
    required = {n for n, f in EnvironmentArtifact.model_fields.items() if f.is_required()}
    assert required == {"environment_name", "id"}


def test_put_rejects_a_reintroduced_service_version():
    """Guard against the kwarg creeping back: every consumer dropped it before this
    parameter was removed, so re-adding it here would be a TypeError in prod."""
    with pytest.raises(TypeError):
        EnvironmentArtifact.put(
            id="x", environment_name="n", service_version=1, file_artifact=object()  # type: ignore[call-arg]
        )
