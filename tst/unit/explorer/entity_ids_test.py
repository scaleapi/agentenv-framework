from urllib.parse import quote, unquote

import pytest

from agent_env.explorer.entity_ids import route_path

IDS = [
    "hello-flat",
    "@local/agentenv-framework/hello/hello",
    "@local/~/stuff/instances/foo",
    "@local/~/Dropbox (Personal)/a&b+c,d@e/tickets",
    "@local/~/Été/Straße/tâche",
    "weird%2Fbare",
    "100%25done",
    "odd#id?x",
]


@pytest.mark.parametrize("entity_id", IDS)
def test_an_encoded_id_stays_one_segment_and_decodes_exactly(entity_id):
    parts = route_path(f"/api/v1/tasks/{quote(entity_id, safe='')}/runs").split("/")

    assert parts[:4] == ["", "api", "v1", "tasks"] and parts[5:] == ["runs"]
    assert unquote(parts[4]) == entity_id


def test_everything_but_slash_and_percent_is_decoded():
    assert route_path("/a%20b/%40local%2Fx/%25/%2f/%C3%A9") == "/a b/@local%2Fx/%25/%2f/é"
