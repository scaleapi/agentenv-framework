"""Tier B integration: real per-turn trajectory delivery into a judge sandbox.

Validates what unit tests structurally can't — that
``RubricsVerifierTaskStep._load_per_turn_trajectories`` actually places
``turn_NN.json`` into a real container filesystem at the right paths, and that
they are intact and parseable *from inside* the sandbox (the deterministic
ceiling for "loaded by the agent").

Uses a **bare Modal sandbox**: ``ModalSandbox.write_file_from_s3`` streams into
the sandbox FS directly, so no agent needs to be deployed. The VM-provider path
(``docker cp … agent-api:…`` + the container ``mkdir -p`` fix) requires a
deployed agent's container and is covered by the full-agent e2e / manual runs.
"""
from __future__ import annotations

import json
import uuid
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from agent_env.config import get_config
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import TrajectoryFilter
from tst.util.capabilities import skip_without_remote_sandbox



pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("modal"),
]

# Minimal public image with sh / base64 / mkdir / python3 (for the read-back parse).
_IMAGE = "python:3.11-slim"


def _tool_span(tool: str, output) -> dict:
    return {
        "name": tool,
        "attributes": {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.prompt": json.dumps({"tool": tool, "input": {}}),
            "gen_ai.completion": json.dumps({"output": output}),
        },
        "start_time": "1",
        "end_time": "2",
    }


# Both turns carry a >1KB tool result so both externalize — exercising the
# per-turn ``turn_NN_`` namespacing (they'd collide as tool_call_result_1.json).
TURN1_SPANS = [_tool_span("email_send", {"result": "x" * 2000})]
TURN2_SPANS = [_tool_span("reminder_move", {"result": "y" * 2000})]
_TOTAL_EVENTS = len(TURN1_SPANS) + len(TURN2_SPANS)


@pytest.fixture(scope="module")
def per_turn_uris():
    """Upload two per-turn OTel-span trajectories through the configured object store.

    The verifier hands them to the sandbox as S3 objects (``Sandbox.write_file_from_s3``), so a store
    that mints another scheme fails here rather than inside the container. The objects stay behind:
    the ObjectStore API has no delete, and the prefix is unique per run."""
    store = get_config().get_object_store()
    prefix = f"agent_snapshots/it-mt-{uuid.uuid4().hex[:12]}"
    uris = [
        store.put(f"{prefix}/turn_{i}.json", json.dumps(spans).encode(), content_type="application/json")
        for i, spans in enumerate((TURN1_SPANS, TURN2_SPANS), start=1)
    ]
    assert all(urlparse(u).scheme == "s3" for u in uris), f"per-turn files need an S3-backed store; got {uris!r}"
    return uris


@pytest_asyncio.fixture(scope="module")
async def modal_sandbox():
    """A bare Modal sandbox (no agent), reused across this module's tests."""
    from agent_env.providers import ModalSandboxProvider

    provider = ModalSandboxProvider()
    sandbox = await provider.create_container(
        image_name=_IMAGE, port=8000, env={}, expose_externally=False,
    )
    try:
        yield sandbox
    finally:
        try:
            await sandbox.terminate()
        except Exception:
            pass


def _verifier() -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(id="v", version=1, criteria=[], prompt_id="p", use_agent_judge=True)


def _fresh_dir(v: RubricsVerifierTaskStep) -> str:
    return f"{v.TRAJECTORY_CONTAINER_DIR}/{uuid.uuid4().hex[:12]}"


async def test_no_filter_delivers_readable_turn_files(modal_sandbox, per_turn_uris):
    v = _verifier()
    d = _fresh_dir(v)

    await v._load_per_turn_trajectories(modal_sandbox, per_turn_uris, d, None)

    rc, out, err = await modal_sandbox.exec_with_output("ls", d)
    assert rc == 0, err
    assert "turn_01.json" in out and "turn_02.json" in out

    rc, body, err = await modal_sandbox.exec_with_output("cat", f"{d}/turn_01.json")
    assert rc == 0, err
    assert json.loads(body) == TURN1_SPANS          # intact + parseable in-container


async def test_default_filter_compacts_and_namespaces_in_container(modal_sandbox, per_turn_uris):
    v = _verifier()
    d = _fresh_dir(v)

    await v._load_per_turn_trajectories(modal_sandbox, per_turn_uris, d, TrajectoryFilter())

    bodies = {}
    for turn in ("turn_01", "turn_02"):
        rc, body, err = await modal_sandbox.exec_with_output("cat", f"{d}/{turn}.json")
        assert rc == 0, err
        events = json.loads(body)
        assert any(e.get("type") == "tool_result" for e in events)   # compacted, not raw
        bodies[turn] = events

    # Both large results externalized under distinct, turn-namespaced names
    # (would collide as tool_call_result_1.json without the prefix).
    rc, listing, err = await modal_sandbox.exec_with_output("ls", "/tmp/tool_call_results")
    assert rc == 0, err
    assert "turn_01_tool_call_result_1.json" in listing
    assert "turn_02_tool_call_result_1.json" in listing

    # Every in-event output_file reference resolves to a real file in the container.
    for events in bodies.values():
        ref = next(e["output_file"] for e in events if e.get("type") == "tool_result")
        rc, _, err = await modal_sandbox.exec_with_output("test", "-f", ref)
        assert rc == 0, f"referenced tool-result file missing in container: {ref} ({err})"


async def test_all_turns_readable_from_container(modal_sandbox, per_turn_uris):
    """Deterministic proxy for agent-side consumption: read every turn file from
    the directory the judge is pointed at and confirm they collectively parse."""
    v = _verifier()
    d = _fresh_dir(v)

    await v._load_per_turn_trajectories(modal_sandbox, per_turn_uris, d, None)

    script = (
        "import json,glob;"
        f"print(sum(len(json.load(open(f))) for f in sorted(glob.glob('{d}/*.json'))))"
    )
    rc, out, err = await modal_sandbox.exec_with_output("python3", "-c", script)
    assert rc == 0, err
    assert int(out.strip()) == _TOTAL_EVENTS
