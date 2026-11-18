"""The extra.agentenv profile: every field optional, unknown keys kept, and PROFILE.md naming every field and value."""
import json
from importlib.resources import files

import pytest

from agentenv_protocol.atif import (
    PROFILES,
    VOCABULARIES,
    AtifError,
    StepProfile,
    dump_profile,
    load,
    profile_of,
    read_profile,
)
from agentenv_protocol.atif.profile import AdapterRef, Arrival, MediaInfo, ReasoningBlock, StepError

SPEC = files("agentenv_protocol.atif").joinpath("PROFILE.md").read_text(encoding="utf-8")
NESTED = (AdapterRef, Arrival, MediaInfo, ReasoningBlock, StepError)


@pytest.mark.parametrize("model", [*PROFILES.values(), *NESTED])
def test_every_profile_field_is_optional(model):
    assert model() is not None
    assert all(not f.is_required() for f in model.model_fields.values())


def test_a_reader_keeps_an_unknown_profile_key():
    step = {"step_id": 1, "source": "user", "message": "x",
            "extra": {"agentenv": {"turn": 0, "later_field": {"a": [1, None]}, "state": "paused"}}}
    profile = read_profile(step, StepProfile)
    assert profile.turn == 0 and profile.state == "paused"
    assert dump_profile(profile) == step["extra"]["agentenv"]
    document = {"agent": {"name": "a", "version": "1"}, "steps": [step], "extra": {"agentenv": {"later": True}}}
    again = load(json.dumps(document))
    assert profile_of(again["steps"][0])["later_field"] == {"a": [1, None]} and profile_of(again) == {"later": True}


def test_a_profile_of_the_wrong_shape_is_refused():
    with pytest.raises(AtifError):
        read_profile({"extra": {"agentenv": {"turn": "zero"}}}, StepProfile)


def test_profile_md_names_every_field_and_vocabulary_value():
    fields = {name for model in [*PROFILES.values(), *NESTED] for name in model.model_fields}
    missing = sorted(name for name in fields if f"`{name}`" not in SPEC)
    assert not missing, f"PROFILE.md doesn't name: {missing}"
    for field, values in VOCABULARIES.items():
        assert f"`{field.split('.')[1]}`" in SPEC
        assert all(f"`{value}`" in SPEC for value in values), field
