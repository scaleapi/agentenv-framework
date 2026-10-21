"""What a put records about how an image was built, which a new version of an infra env records afresh rather than
keeps (``detected``)."""

from pathlib import Path

from agent_env.utils.docker import metadata

REPO = Path(__file__).resolve().parents[4]


def test_every_key_a_put_records_about_the_build_is_one_it_records_afresh(monkeypatch):
    """So a key added to the detection is never carried over from an env's earlier version."""
    monkeypatch.setenv("USER", "me")

    recorded = metadata.detect_env_metadata(REPO / "src/agent_env/env/gateway/Dockerfile", REPO)

    assert {"created_by", "agent_env_version", "dockerfile_path"} <= set(recorded)
    assert [key for key in recorded if not metadata.detected(key)] == []
    assert not metadata.detected("owner")
