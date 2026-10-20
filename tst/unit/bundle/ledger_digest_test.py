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

# Recorded at SCHEME 2, which writes every image the bundle builds as its build context.
RECORDED = {
    "@local/~/triage/tickets__env_image": "sha256:c76d8bd3706e9785ff464cc56fd7cf0b2afd968003a1661568d36e2abefa70dd",
    "@local/~/triage/tickets": "sha256:0732f87377c46dc5e0238adb4cadc705dc814fb89825536cdc83b28b086a9b33",
    "@local/~/triage/solver__agent_image": "sha256:6d78d7ba2266a8144537f8493099b5b5b9440a20d4d5d806f9cddb39de443436",
    "@local/~/triage/solver": "sha256:ef703cb708fba45b796f9413654cd0c8579ca4f6268cdedf177068619b1ad645",
    "@local/~/triage/greeting": "sha256:386ee0e6beef93a08f3816336de4ae136ab76a6c2136159c430599728ca91c0e",
    "@local/~/triage/note": "sha256:f427863a7d913875a1b87f10ea136e9bc50ec496bcaec277b11927f19ae3e6cd",
    "@local/~/triage/shelf": "sha256:53359d8af25aec8090b42560071e31d8a15023be27148189f8376fedf3062955",
    "@local/~/triage/t": "sha256:3477937456f6da7403b62c98f2e76d695349887aa8d17076542f21be277c141d",
    "@local/~/triage/u": "sha256:b264dad989f8de065f0ea4026cab0dc67d8a51ad73621380bb9ed22ada0e51f2",
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
