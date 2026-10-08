"""The digests the ledger records stay byte-identical across releases unless its SCHEME changes: a bundle run on a
new release reuses every version an older one wrote. Each value here was recorded by an earlier release."""

import json

import pytest

from agent_env.bundle.ledger import Ledger
from tst.unit.bundle._support import layout, plan_of

LAYOUT = {
    "artifacts/greeting/hello.txt": "hello\n",
    "artifacts/shelf/a.txt": "a\n",
    "artifacts/shelf/sub/b.txt": "b\n",
    "artifacts/note/artifact.toml": 'description = "a note"\n',
    "artifacts/note/note.md": "# note\n",
    "envs/tickets/Dockerfile": "FROM scratch\nCOPY . /app\n",
    "envs/tickets/server.py": "print('tickets')\n",
    "envs/tickets/env.toml": 'environment_name = "tickets"\n',
    "agents/solver/Dockerfile": "FROM scratch\n",
    "agents/solver/agent.py": "print('solver')\n",
    "tasks/t.json": json.dumps([
        {"id": "env", "type": "deploy_env", "env_id": "tickets"},
        {"id": "load", "type": "load_artifact", "env_id": "tickets", "artifact_id": "greeting", "depends_on": ["env"]},
    ]),
    "tasks/u.json": json.dumps([
        {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"},
        {"id": "shelf", "type": "load_artifact", "env_id": "tickets", "artifact_id": "shelf"},
        {"id": "note", "type": "load_artifact", "env_id": "tickets", "artifact_id": "note"},
    ]),
}

# Recorded by agentenv-framework 0.9.1278.
RECORDED = {
    "@local/~/triage/tickets__env_image": "sha256:b381bf1c841f024fe2dfeabb08e5a10cf2b3cc596563ba3e6ea5c6798af1c868",
    "@local/~/triage/tickets": "sha256:724cfb95eed4f78c7f77dc1c94507277217860bcce6a773bc3156e77bc2f8521",
    "@local/~/triage/solver__agent_image": "sha256:00fb2164bc6e1d8230ef6e2d437f640ba853acde7ab7c4f13485d3fb1b48d5d6",
    "@local/~/triage/solver": "sha256:54bcabc92b285434ad23af91e2dcf1b04700d73a819016b1e62b66f7d0f3ec0e",
    "@local/~/triage/greeting": "sha256:e0047a71d6e34c6f390f6b6bf416ff54765bbbc390b2bd7f34432ee79a691e66",
    "@local/~/triage/note": "sha256:3e9de21964b7c4e78a240f8f7aaafb66a0f04a9a430b33b6e54b9869c9001a9d",
    "@local/~/triage/shelf": "sha256:ab0dd7c0b9c41312caeb9e012a31e4b640d0c626ddf005a5d7607824f36aa674",
    "@local/~/triage/t": "sha256:056ad6e1030f251c2a6f9e144270fd53d9053c5d4264e0654ba8e547b10abe7a",
    "@local/~/triage/u": "sha256:f8d5445a95fed8bf4d03b713fa8f545b1a8da847db02c22a3806eec39e26db29",
}


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = layout(tmp_path / "triage", LAYOUT)
    for rel in LAYOUT:  # a built image hashes permission bits other than 0644, which the umask would decide
        (root / rel).chmod(0o644)
    return root


def test_every_digest_an_earlier_release_recorded_is_unchanged(bundle_dir):
    plan = plan_of(bundle_dir)
    ledger = Ledger.for_plan(plan)
    digests = {write.id: ledger.digest(write, {need: 1 for need in write.needs}) for write in plan.writes}

    assert {id: digest.value for id, digest in digests.items() if digest is not None} == RECORDED
