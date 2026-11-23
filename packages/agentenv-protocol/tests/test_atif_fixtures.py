"""The profile's canonical fixtures: rebuilt through the adapter framework, valid ATIF with or without the profile,
unchanged by Harbor's own load and save, and a prefix of themselves at every live page."""
import json

import pytest

from agentenv_protocol.atif import Trajectory, check, is_prefix, strip_profile
from atif_cases import CASES, FIXTURES, _text, build, fixtures


def test_the_checked_in_fixtures_are_what_the_cases_build():
    built = fixtures()
    shipped = {str(p.relative_to(FIXTURES)) for p in FIXTURES.rglob("*") if p.is_file()}
    assert shipped == set(built), "rewrite them with: python packages/agentenv-protocol/tests/atif_cases.py"
    for rel, value in built.items():
        assert (FIXTURES / rel).read_bytes() == _text(value), (
            f"{rel} is stale; rewrite with: python packages/agentenv-protocol/tests/atif_cases.py")


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_fixture_is_atif_with_and_without_its_profile_and_survives_harbors_save(name):
    document = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    check(document)
    stripped = strip_profile(document)
    check(stripped)
    assert Trajectory.model_validate(document).to_json_dict() == document
    assert Trajectory.model_validate(stripped).to_json_dict() == stripped


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_live_page_validates_and_is_a_prefix_of_the_final_document(name):
    document, pages, _ = build(name)
    for page in pages:
        if page is not None:
            check(page)
        assert is_prefix(page, document)
    assert pages[-1] == document


@pytest.mark.parametrize("name", sorted(CASES))
def test_an_all_text_message_is_one_string_as_harbors_tools_read_it(name):
    document = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    for trajectory in [document, *document.get("subagent_trajectories", [])]:
        for step in trajectory["steps"]:
            message = step["message"]
            assert isinstance(message, str) or any(p["type"] != "text" for p in message)
