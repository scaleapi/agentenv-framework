"""The three capture primitives shared by ``snapshot_agent_state`` and
``prompt_agent``'s periodic capture.

``capture_workspace`` raises on any failure (no tar → nothing gradable);
``read_partial_trajectory`` never raises (the bundle is what makes a row
gradable, so a degraded trajectory read must not discard the point).
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent_env.task_step.snapshot_utils import agent_state_capture as mod

from .capture_stubs import (
    BUCKET,
    CONTEXT_ID_GET,
    agent_card,
    install_capture_stubs,
)


def _save_card(*variants: list[str], **params) -> dict:
    return agent_card(
        snapshot={
            "methods": {
                "save": {
                    "request": {
                        "required": ["context_id"],
                        "oneOf": [{"required": fields} for fields in variants],
                    }
                }
            }
        },
        **params,
    )


async def _capture(**overrides):
    kwargs = dict(
        a2a_url="https://agent",
        a2a_card=_save_card(["objects"]),
        agent_name="solver",
        a2a_context_id="ctx-1",
        artifact_id="wsp",
        timeout_seconds=30,
    )
    kwargs.update(overrides)
    return await mod.capture_workspace(**kwargs)


def _granted(rec) -> dict[str, str]:
    return {
        grant["object_url"].rsplit("/", 1)[-1]: grant["object_url"]
        for grant in rec.object_store.write_grants
    }


# ---- capture_workspace ------------------------------------------------------

@pytest.mark.asyncio
async def test_posts_context_id_and_a_write_grant_for_each_snapshot_object(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    result = await _capture()

    (sent,) = rec.save_requests
    assert sent["url"] == "https://agent/ext/snapshot"
    assert set(sent["json"]) == {"context_id", "objects"}
    assert sent["json"]["context_id"] == "ctx-1"
    assert set(sent["json"]["objects"]) == {"trajectory", "workspace"}
    assert result.capture_prefix.startswith(f"s3://{BUCKET}/agent_snapshots/wsp/")
    assert _granted(rec) == {
        name: f"{result.capture_prefix}{name}" for name in ("trajectory", "workspace")
    }
    # Registered where the grants pointed, wrapped as a universe.
    assert result.bundle_object_url == result.capture_prefix
    assert result.universe_id == "wsp"


@pytest.mark.asyncio
async def test_honours_a_card_pinned_endpoint(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    await _capture(a2a_card=_save_card(["objects"], snapshot_endpoint="/custom/snap"))
    assert rec.save_requests[0]["url"] == "https://agent/custom/snap"


@pytest.mark.asyncio
async def test_each_capture_gets_its_own_prefix(monkeypatch):
    """The random suffix is what isolates two captures, NOT the version.

    Pinned to one version on purpose: with the stub's incrementing version the
    prefixes differ even if the suffix is deleted, so this passed while asserting
    nothing. Two concurrent captures really can peek the same version.
    """
    rec = install_capture_stubs(monkeypatch)
    monkeypatch.setattr(
        rec, "next_version",
        lambda artifact_id: (rec.next_version_calls.append(artifact_id), 7)[1],
    )
    first = await _capture()
    second = await _capture()
    assert first.capture_prefix != second.capture_prefix
    assert "/7-" in first.capture_prefix and "/7-" in second.capture_prefix
    assert rec.next_version_calls == ["wsp", "wsp"]


@pytest.mark.asyncio
async def test_raises_when_the_card_does_not_advertise_snapshot(monkeypatch):
    install_capture_stubs(monkeypatch)
    with pytest.raises(RuntimeError, match="does not advertise the snapshot extension"):
        await _capture(a2a_card=agent_card(snapshot=False))


@pytest.mark.asyncio
async def test_raises_on_an_http_error(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    rec.save_status = 500
    rec.save_body = {"detail": "boom"}
    with pytest.raises(httpx.HTTPStatusError, match="snapshot save failed with HTTP 500"):
        await _capture()
    assert not rec.universes


@pytest.mark.asyncio
async def test_a_capture_is_issued_under_the_fixture_prefix(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    rec.key_prefix = "fx/"
    result = await _capture()
    assert result.capture_prefix.startswith(f"s3://{BUCKET}/fx/agent_snapshots/wsp/")
    assert all(url.startswith(result.capture_prefix) for url in _granted(rec).values())


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact_id, segment", [
    ("wsp", "wsp"),
    ("@local/~/work/triage/tasks/t-abcd1234__solve-workspace", "local/work-triage-tasks-t-abcd1234-solve-workspace-606593d6050a"),
], ids=["bare", "local"])
async def test_a_capture_is_issued_under_the_workspaces_key_segment(monkeypatch, artifact_id, segment):
    install_capture_stubs(monkeypatch)
    result = await _capture(artifact_id=artifact_id)
    assert result.capture_prefix.startswith(f"s3://{BUCKET}/agent_snapshots/{segment}/")
    assert result.universe_id == artifact_id


@pytest.mark.asyncio
async def test_a_card_offering_both_save_forms_gets_the_object_one(monkeypatch):
    rec = install_capture_stubs(monkeypatch)

    result = await _capture(a2a_card=_save_card(["s3_prefix"], ["objects"]))

    assert set(rec.save_requests[0]["json"]) == {"context_id", "objects"}
    assert set(_granted(rec)) == {"trajectory", "workspace"}
    assert result.bundle_object_url == result.capture_prefix


@pytest.mark.asyncio
async def test_snapshot_save_requires_every_supplied_object(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    rec.object_store.withheld = {"workspace"}

    with pytest.raises(RuntimeError, match=r"left \['workspace'\] out of the object store"):
        await _capture()
    assert not rec.universes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card",
    [_save_card(["s3_prefix"]), agent_card()],
    ids=["s3-prefix-only", "no-declared-request"],
)
async def test_an_agent_without_the_object_save_form_is_refused_before_any_request(
    monkeypatch, card
):
    rec = install_capture_stubs(monkeypatch)

    with pytest.raises(RuntimeError, match="does not advertise the object form"):
        await _capture(a2a_card=card)

    assert (rec.save_requests, rec.object_store.write_grants, rec.universes) == ([], [], [])


@pytest.mark.asyncio
async def test_a_save_on_a_store_without_grants_is_refused(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    monkeypatch.setattr(rec.object_store, "supports_transfer_grants", False)

    with pytest.raises(RuntimeError, match="does not issue transfer grants"):
        await _capture()

    assert rec.save_requests == []


# ---- read_partial_trajectory ------------------------------------------------

PARTIAL_PREFIX = f"s3://{BUCKET}/partial-trajectories/run-1/"
CONTEXT_OBJECTS_GET = {
    "request": {
        "oneOf": [
            {"required": ["context_id"]},
            {"required": ["context_id", "objects"]},
        ]
    }
}


async def _read(card, **overrides):
    kwargs = dict(
        a2a_url="https://agent", a2a_card=card, context_id="ctx-1", timeout_seconds=30,
        trajectory_output_prefix=PARTIAL_PREFIX,
    )
    kwargs.update(overrides)
    return await mod.read_partial_trajectory(**kwargs)


@pytest.mark.asyncio
async def test_reads_the_trajectory_keyed_on_context_id(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    result = await _read(agent_card(trajectory=CONTEXT_ID_GET))

    (sent,) = rec.trajectory_requests
    assert sent["url"] == "https://agent/ext/trajectory"
    # context_id, never task_id: task_id is one finished turn and 404s mid-run.
    assert sent["json"] == {"context_id": "ctx-1"}
    assert result.trajectory == [{"role": "user"}]
    assert result.reason is None


@pytest.mark.asyncio
async def test_context_trajectory_prefers_advertised_object_mode(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    rec.trajectory_body = {
        "objects": {"trajectory": {"size_bytes": 42}}
    }

    result = await _read(agent_card(trajectory=CONTEXT_OBJECTS_GET))

    assert result.trajectory is None
    assert result.reason is None
    assert result.object_url.startswith(f"{PARTIAL_PREFIX}trajectory-")
    sent = rec.trajectory_requests[0]["json"]
    assert sent["context_id"] == "ctx-1"
    assert sent["objects"]["trajectory"]["write"]["kind"] == "http-put"


@pytest.mark.asyncio
async def test_context_trajectory_is_inline_on_a_store_without_grants(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    monkeypatch.setattr(rec.object_store, "supports_transfer_grants", False)

    result = await _read(agent_card(trajectory=CONTEXT_OBJECTS_GET))

    assert [r["json"] for r in rec.trajectory_requests] == [{"context_id": "ctx-1"}]
    assert rec.object_store.write_grants == []
    assert result.trajectory == [{"role": "user"}]


@pytest.mark.asyncio
async def test_a_grant_that_cannot_be_issued_is_a_reason_not_a_raise(monkeypatch):
    rec = install_capture_stubs(monkeypatch)

    def refuse(*args, **kwargs):
        raise ValueError("cannot sign")

    monkeypatch.setattr(rec.object_store, "issue_write_grant", refuse)

    result = await _read(agent_card(trajectory=CONTEXT_OBJECTS_GET))

    assert (result.reason, result.object_url) == ("trajectory_grant_unavailable", None)
    assert rec.trajectory_requests == []


@pytest.mark.parametrize(
    "card, reason",
    [
        (agent_card(), "trajectory_ext_unavailable"),
        (agent_card(trajectory={"request": {"required": ["task_id"]}}),
         "trajectory_context_unsupported"),
        # A bare `{}` advertises `get` with no request contract at all, so the
        # context_id mode is not claimed — treat it as task_id-only.
        (agent_card(trajectory={}), "trajectory_context_unsupported"),
    ],
)
@pytest.mark.asyncio
async def test_an_unsupported_card_costs_no_round_trip(monkeypatch, card, reason):
    rec = install_capture_stubs(monkeypatch)
    result = await _read(card)
    assert (result.reason, result.trajectory) == (reason, None)
    assert rec.trajectory_requests == []  # refused off the card, not via a 400


@pytest.mark.asyncio
async def test_concurrent_captures_on_one_artifact_id_never_share_a_prefix(monkeypatch):
    """Registration is deliberately unserialized: ``prompt_agent`` derives an
    artifact id per rollout, so nothing else allocates versions of it, and a
    genuine cross-process race is ``put_document``'s retry to absorb.

    What must still hold is byte isolation — every capture writes to its own
    random-suffixed prefix, so even captures that peek the same version cannot
    overwrite each other's objects.
    """
    rec = install_capture_stubs(monkeypatch)
    results = await asyncio.gather(*(_capture() for _ in range(4)))

    assert len({r.capture_prefix for r in results}) == 4
    assert len(rec.save_requests) == 4


@pytest.mark.asyncio
async def test_a_get_advertised_without_methods_is_unadvertised(monkeypatch):
    install_capture_stubs(monkeypatch)
    card = {"capabilities": {"extensions": [
        {"uri": "urn:agentenv:trajectory/v1", "params": {"methods": {"save": {}}}}
    ]}}
    assert (await _read(card)).reason == "trajectory_get_unadvertised"


@pytest.mark.parametrize(
    "status, body, reason",
    [
        (404, {}, "trajectory_session_missing"),
        (500, {}, "trajectory_http_500"),
        (200, {"trajectory": []}, "trajectory_empty"),
    ],
)
@pytest.mark.asyncio
async def test_a_degraded_read_is_a_reason_not_a_raise(monkeypatch, status, body, reason):
    rec = install_capture_stubs(monkeypatch)
    rec.trajectory_status, rec.trajectory_body = status, body
    result = await _read(agent_card(trajectory=CONTEXT_ID_GET))
    assert (result.reason, result.trajectory) == (reason, None)


@pytest.mark.asyncio
async def test_a_transport_failure_is_a_reason_not_a_raise(monkeypatch):
    install_capture_stubs(monkeypatch)

    async def boom(self, method, url, **kwargs):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx.AsyncClient, "request", boom)
    assert (await _read(agent_card(trajectory=CONTEXT_ID_GET))).reason == "trajectory_read_failed"


# ---- upload_trajectory ------------------------------------------------------

def test_uploads_under_the_prefix_with_a_unique_key(monkeypatch):
    rec = install_capture_stubs(monkeypatch)
    prefix = f"s3://{BUCKET}/prompt_agent_trajectories/prompt_id=p1/"

    first = mod.upload_trajectory([{"a": 1}], prefix)
    second = mod.upload_trajectory([{"a": 1}], prefix)

    assert first != second, "two uploads must not collide on one key"
    assert first.startswith(f"{prefix}trajectory-") and first.endswith(".json")
    assert json.loads(rec.object_store.puts[0][1]) == [{"a": 1}]


def test_a_prefix_without_a_trailing_slash_still_nests(monkeypatch):
    install_capture_stubs(monkeypatch)
    url = mod.upload_trajectory([], f"s3://{BUCKET}/traj")
    assert url.startswith(f"s3://{BUCKET}/traj/trajectory-")


def test_a_foreign_bucket_raises_rather_than_writing_to_the_configured_one(monkeypatch):
    install_capture_stubs(monkeypatch)
    # Derived through the object store, not urlparse().path — otherwise this
    # silently writes to the configured bucket under the same key path.
    with pytest.raises(ValueError, match="not in the configured bucket"):
        mod.upload_trajectory([], "s3://someone-elses-bucket/traj/")
