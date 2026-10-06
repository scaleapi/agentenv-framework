"""The validator's URI probes name the fixtures' own object URLs and leave making them readable to
prompt_agent, which sends each as an HTTPS URL the agent it deploys can read."""

from types import SimpleNamespace

import pytest

from agent_env.a2a_agent.validator import A2AAgentValidator
from agent_env.config import set_object_store
from tst.util.granting_object_store import GrantingObjectStore

_AGENT = SimpleNamespace(id="solver", version=3)


@pytest.fixture
def store(tmp_path):
    granting = GrantingObjectStore(str(tmp_path))
    set_object_store(granting)
    return granting


def test_the_uri_probes_name_the_uploaded_fixtures(store):
    fixtures = A2AAgentValidator._upload_probe_fixtures(_AGENT, "file:///skill")
    steps, probes = A2AAgentValidator._build_modality_steps("t", fixtures, "t-deploy-agent")

    uris = {
        probe["modality"]: part["file"]["uri"]
        for step, probe in zip(steps, probes)
        for part in step.parts
        if part["kind"] == "file" and "uri" in part["file"]
    }
    assert uris == {"image/png+uri-https": fixtures.png_object_uri, "video/mp4": fixtures.mp4_object_uri}
    assert store.granted == []
