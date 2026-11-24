"""What a turn hands Hermes and when it runs: turns of different contexts at once and a context's own one at a time, the
session a context is bound to when its turn starts, the task's config, files and MCP servers, a failed turn, and a
snapshot of a context's conversation and workspace restored into another context."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest
from agentenv_protocol.a2a_agent import ModelUsage, Uploaded

from tst.unit.agents.hermes.agent import a2a_server, hermes_bridge, hermes_kit
from tst.unit.agents.hermes.conftest import result


class Turns:
    """Stand in for run_hermes: each call's arguments, held until ``release`` when it is set."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.release: asyncio.Event | None = None
        self.answer = lambda kwargs: result(session_id=kwargs["session_id"] or f"s-{kwargs['context_id']}")

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.release is not None:
            await self.release.wait()
        return self.answer(kwargs)


@pytest.fixture
def turns(monkeypatch) -> Turns:
    stand_in = Turns()
    monkeypatch.setattr(a2a_server, "run_hermes", stand_in)
    return stand_in


async def _until(condition, seconds: float = 5) -> None:
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_turns_of_different_contexts_run_at_once(hermes, turns):
    turns.release = asyncio.Event()
    async with hermes.client() as client:
        sent = [asyncio.create_task(hermes.send(client, "go", context_id)) for context_id in ("ctx-a", "ctx-b")]
        await _until(lambda: len(turns.calls) == 2)
        turns.release.set()
        tasks = await asyncio.gather(*sent)

    assert {call["context_id"] for call in turns.calls} == {"ctx-a", "ctx-b"}
    assert [task["status"]["state"] for task in tasks] == ["completed", "completed"]


@pytest.mark.asyncio
async def test_a_contexts_turns_run_one_at_a_time_each_resuming_the_last(hermes, turns):
    turns.release = asyncio.Event()
    async with hermes.client() as client:
        first = asyncio.create_task(hermes.send(client, "one"))
        await _until(lambda: len(turns.calls) == 1)
        second = asyncio.create_task(hermes.send(client, "two"))
        await asyncio.sleep(0.1)
        running_together = len(turns.calls)
        turns.release.set()
        await asyncio.gather(first, second)

    assert running_together == 1
    assert [call["session_id"] for call in turns.calls] == [None, "s-ctx-1"]


@pytest.mark.asyncio
async def test_the_turns_config_servers_and_workspace_reach_hermes(monkeypatch, hermes, turns, folders):
    monkeypatch.setenv("A2A_PORT", "8000")
    config = {"model": "provider/model", "effort": "high", "system_prompt": "be brief", "project_id": "p-1",
              "task_id": "t-1", "role": "solver", "timeout_seconds": 42}
    async with hermes.client() as client:
        assert (await client.post("/ext/agent-config", json=config)).status_code == 200
        await client.post("/ext/mcp-config", json={"name": "env", "url": "https://mcp.example/mcp",
                                                   "headers": {"X-Key": "k"}})
        task = await hermes.send(client, "go")

    (call,) = turns.calls
    assert {key: call[key] for key in ("prompt", "model", "effort", "system_prompt", "project_id", "task_id", "role",
                                       "timeout_s", "workspace", "skills_dir")} == {
        "prompt": "go", "model": "provider/model", "effort": "high", "system_prompt": "be brief",
        "project_id": "p-1", "task_id": "t-1", "role": "solver", "timeout_s": 42, "workspace": str(folders),
        "skills_dir": a2a_server.SKILLS_DIR}
    assert call["mcp_servers"]["env"] == {"url": "https://mcp.example/mcp", "headers": {"X-Key": "k"}}
    assert call["mcp_servers"]["peer_agents"]["url"] == "http://127.0.0.1:8000/ext/peer-agents/mcp"
    assert task["status"]["state"] == "completed"


@pytest.mark.asyncio
async def test_a_file_part_is_written_under_the_task_and_named_in_the_prompt(hermes, turns, folders):
    parts = [{"kind": "text", "text": "describe"},
             {"kind": "file", "file": {"name": "../a b.png", "mimeType": "image/png", "bytes": "cG5n"}}]
    async with hermes.client() as client:
        task = await hermes.send(client, parts)

    attached = folders / "inputs" / task["id"] / "a_b.png"
    assert attached.read_bytes() == b"png"
    (call,) = turns.calls
    assert call["prompt"].startswith("describe\n\nAttached files")
    assert f"- {attached}" in call["prompt"]


@pytest.mark.asyncio
async def test_two_pdfs_whose_names_share_a_stem_keep_their_own_pages(monkeypatch, tmp_path, hermes, turns):
    # A pdftoppm that renders as many pages as the file's text says, each naming the file it came from.
    fake = tmp_path / "pdfbin" / "pdftoppm"
    fake.parent.mkdir()
    fake.write_text(f"#!{sys.executable}\nimport pathlib, sys\nsource, prefix = sys.argv[-2:]\n"
                    "for page in range(1, int(pathlib.Path(source).read_text()) + 1):\n"
                    "    pathlib.Path(f'{prefix}-{page}.png').write_text(pathlib.Path(source).name)\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    parts = [{"kind": "text", "text": "compare"},
             {"kind": "file", "file": {"name": "report.pdf", "mimeType": "application/pdf", "bytes": "Mw=="}},
             {"kind": "file", "file": {"name": "report", "mimeType": "application/pdf", "bytes": "MQ=="}}]
    async with hermes.client() as client:
        await hermes.send(client, parts)

    (call,) = turns.calls
    listed = [Path(line[2:]) for line in call["prompt"].splitlines() if line.startswith("- ")]
    pages = [path for path in listed if path.suffix == ".png"]
    assert [path.read_text() for path in pages] == ["report.pdf"] * 3 + ["report"]
    assert len(set(pages)) == 4


@pytest.mark.asyncio
async def test_a_pdf_with_a_name_near_the_filesystems_limit_renders(monkeypatch, tmp_path, hermes, turns):
    fake = tmp_path / "pdfbin" / "pdftoppm"
    fake.parent.mkdir()
    fake.write_text(f"#!{sys.executable}\nimport pathlib, sys\npathlib.Path(sys.argv[-1] + '-1.png').write_text('p')\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    name = "r" * 238 + ".pdf"
    parts = [{"kind": "text", "text": "read"},
             {"kind": "file", "file": {"name": name, "mimeType": "application/pdf", "bytes": "MQ=="}}]
    async with hermes.client() as client:
        task = await hermes.send(client, parts)

    assert task["status"]["state"] == "completed"
    (call,) = turns.calls
    assert sum(line.endswith(".png") for line in call["prompt"].splitlines()) == 1


@pytest.mark.asyncio
async def test_a_failed_turn_is_a_failed_task_that_keeps_its_session(hermes, turns):
    turns.answer = lambda kwargs: (
        result(response="", error="HTTP 401: invalid key", error_code="401", session_id="s-started")
        if not kwargs["session_id"] else result(session_id=kwargs["session_id"]))
    async with hermes.client() as client:
        failed = await hermes.send(client, "one")
        await hermes.send(client, "two")

    assert failed["status"]["state"] == "failed"
    assert "HTTP 401: invalid key" in str(failed["status"]["message"])
    assert [call["session_id"] for call in turns.calls] == [None, "s-started"]


@pytest.mark.asyncio
async def test_a_first_turn_cancelled_after_hermes_named_its_session_keeps_it_for_the_next(monkeypatch, hermes):
    calls: list[dict] = []
    started = asyncio.Event()

    async def run(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs["event_callback"]({"type": "on_session_start", "session_id": "s-started"})
            started.set()
            await asyncio.Event().wait()
        return result(session_id=kwargs["session_id"])

    monkeypatch.setattr(a2a_server, "run_hermes", run)
    async with hermes.client() as client:
        task = await hermes.send(client, "one", blocking=False)
        await asyncio.wait_for(started.wait(), 5)
        await client.post("/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/cancel", "params": {"id": task["id"]}})
        await _until(lambda: hermes.agent.session_ref_for_context("ctx-1") == "s-started")
        await hermes.send(client, "two")

    assert [call["session_id"] for call in calls] == [None, "s-started"]


def test_session_references_outlive_the_frameworks_bounded_record():
    assert issubclass(a2a_server.HermesAgent, hermes_kit.DurableSessionRefs)


_GRANT = {"expires_at": "2099-01-01T00:00:00Z", "url": "https://objects.example/o"}
_WRITE = {"media_type": "application/octet-stream", "max_bytes": 2**20, "write": {"kind": "http-put", **_GRANT}}


@pytest.mark.asyncio
async def test_a_snapshot_restores_the_conversation_and_workspace_into_another_context(
        monkeypatch, hermes, turns, folders, tmp_path):
    objects = tmp_path / "objects"
    objects.mkdir()

    async def upload(target, source):
        name = "workspace" if "workspace" in str(source) else "trajectory"
        shutil.copyfile(source, objects / name)
        return Uploaded(size_bytes=(objects / name).stat().st_size)

    async def download(source, destination):
        shutil.copyfile(objects / ("workspace" if "workspace" in str(destination) else "trajectory"), destination)

    monkeypatch.setattr(a2a_server, "upload", upload)
    monkeypatch.setattr(a2a_server, "download", download)
    async with hermes.client() as client:
        await hermes.send(client, "go", "ctx-src")
        source = hermes_bridge.profile_dir("ctx-src")
        source.mkdir(parents=True)
        (source / "state.db").write_bytes(b"the conversation")
        (source / "memories").mkdir()
        (source / "memories" / "MEMORY.md").write_text("a memory")
        (source / "config.yaml").write_text("rewritten each turn")
        (folders / "result.txt").write_text("42")
        saved = await client.post("/ext/snapshot", json={"context_id": "ctx-src", "objects": {
            "trajectory": _WRITE, "workspace": _WRITE}})
        (folders / "result.txt").write_text("changed since")
        read = {"media_type": "application/octet-stream", "max_bytes": 2**20, "size_bytes": 1,
                "read": {"kind": "http-get", **_GRANT}}
        loaded = await client.put("/ext/snapshot", json={"target_context_id": "ctx-dst", "objects": {
            "trajectory": read, "workspace": read}})
        await hermes.send(client, "go on", "ctx-dst")

    assert saved.status_code == 200, saved.text
    assert loaded.status_code == 200, loaded.text
    target = hermes_bridge.profile_dir("ctx-dst")
    assert (target / "state.db").read_bytes() == b"the conversation"
    assert (target / "memories" / "MEMORY.md").read_text() == "a memory"
    assert not (target / "config.yaml").exists()
    assert (folders / "result.txt").read_text() == "42"
    assert turns.calls[-1]["session_id"] == "s-ctx-src"


def test_a_turns_usage_adds_the_cache_back_to_the_input():
    """GPT's two calls of a smoke run through a LiteLLM proxy, as Hermes reported them: its input leaves out the 10,496
    cached tokens the second call read."""
    report = {"input_tokens": 10750, "output_tokens": 48, "total_tokens": 21294, "cache_read_tokens": 10496,
              "cache_write_tokens": 0, "reasoning_tokens": 0, "api_calls": 2, "estimated_cost_usd": 0.0,
              "cost_status": "unknown"}

    assert a2a_server._usage(result(usage=report, tool_call_count=1)).to_dict() == {
        "tool_call_count": 1, "input_tokens": 21246, "output_tokens": 48, "total_tokens": 21294,
        "cache_read_tokens": 10496, "cache_write_tokens": 0, "input_includes_cache": True,
        "provider_details": {"api_calls": 2, "reasoning_tokens": 0, "cost_status": "unknown"}}


def test_hermes_cost_is_kept_only_when_it_priced_the_calls():
    """Hermes reports 0.0 with ``cost_status`` unknown when it couldn't price a call; that isn't a price."""
    report = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "cache_read_tokens": 0,
              "cache_write_tokens": 0, "api_calls": 1}

    unknown = a2a_server._usage(result(usage={**report, "estimated_cost_usd": 0.0, "cost_status": "unknown"}))
    estimated = a2a_server._usage(result(usage={**report, "estimated_cost_usd": 0.0123, "cost_status": "estimated"}))

    assert unknown.cost_usd is None and estimated.cost_usd == 0.0123


def test_a_turns_usage_counts_its_auxiliary_calls():
    """hermes-baseline-gpt, recorded on 2026-11-20 through Scale's LiteLLM proxy: two main calls and a title-generation
    call, which Hermes reports only in its usage file's auxiliary block; the proxy saw 21,523 in and 64 out."""
    report = {"input_tokens": 10739, "output_tokens": 54, "total_tokens": 21289, "cache_read_tokens": 10496,
              "cache_write_tokens": 0, "api_calls": 2, "estimated_cost_usd": 0.0, "cost_status": "unknown",
              "auxiliary": {"api_calls": 1, "input_tokens": 288, "output_tokens": 10, "cache_read_tokens": 0,
                            "cache_write_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.0,
                            "total_tokens": 298, "by_task": {"title_generation": {"api_calls": 1}}},
              "total_including_auxiliary": {"estimated_cost_usd": 0.0, "total_tokens": 21587, "api_calls": 3}}

    usage = a2a_server._usage(result(usage=report))

    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (21523, 64, 10496)
    assert usage.provider_details["api_calls"] == 3 and usage.cost_usd is None


def _events(*events):
    parser = hermes_bridge._EventParser()
    for event in events:
        parser.on_line(json.dumps(event))
    return parser


def _call(turn, session, prompt, output, model=None):
    return {"type": "post_api_request", "turn_id": turn, "session_id": session, "model": model,
            "usage": {"input_tokens": prompt, "prompt_tokens": prompt, "output_tokens": output,
                      "cache_read_tokens": 0, "cache_write_tokens": 0}}


def test_a_delegate_task_childs_calls_are_counted_and_the_session_stays_the_turns():
    """hermes-subagent-claude, recorded on 2026-11-20: the turn's two calls (26,616 in, 86 out, as its usage file
    counts them) and its title call (456, 17) beside a child's two (9,161 and 9,251 in), which the usage file leaves
    out; the proxy saw 45,484 in and 172 out."""
    parser = _events({"type": "on_session_start", "session_id": "parent"}, _call("A", "parent", 13081, 82),
                     {"type": "on_session_start", "session_id": "child"}, _call("B", "child", 9161, 60),
                     _call("B", "child", 9251, 9), _call("A", "parent", 13535, 4))
    report = {"input_tokens": 26616, "output_tokens": 86, "total_tokens": 26702, "cache_read_tokens": 0,
              "cache_write_tokens": 0, "api_calls": 2, "cost_status": "unknown",
              "auxiliary": {"api_calls": 1, "input_tokens": 456, "output_tokens": 17, "cache_read_tokens": 0,
                            "cache_write_tokens": 0, "total_tokens": 473},
              "total_including_auxiliary": {"api_calls": 3}}

    turn = parser.result(b"", b"", 0, report)
    usage = a2a_server._usage(turn)

    assert turn.session_id == "parent" and parser.session_id == "parent"
    assert (usage.input_tokens, usage.output_tokens, usage.provider_details["api_calls"]) == (45484, 172, 5)


def test_a_priced_turn_with_a_delegate_task_child_has_no_cost():
    """Hermes prices only the calls its usage file counts, which leave out a child's, so its figure is short."""
    parser = _events(_call("A", "parent", 13081, 82), _call("B", "child", 9161, 60), _call("A", "parent", 13535, 4))
    report = {"input_tokens": 26616, "output_tokens": 86, "total_tokens": 26702, "cache_read_tokens": 0,
              "cache_write_tokens": 0, "api_calls": 2, "estimated_cost_usd": 0.0271, "cost_status": "estimated"}

    alone = a2a_server._usage(result(usage=report))
    delegated = a2a_server._usage(parser.result(b"", b"", 0, report))

    assert (alone.cost_usd, delegated.cost_usd, delegated.input_tokens) == (0.0271, None, 26616 + 9161)


def test_calls_that_reported_counts_are_kept_beside_those_that_reported_none():
    """A streamed main answer without usage leaves the turn's title call and a child's call counted, and only the main
    calls unreported."""
    parser = _events(_call("A", "parent", 0, 0), _call("B", "child", 9161, 60))
    report = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0,
              "cache_write_tokens": 0, "api_calls": 2, "estimated_cost_usd": 0.0, "cost_status": "estimated",
              "auxiliary": {"api_calls": 1, "input_tokens": 456, "output_tokens": 17, "cache_read_tokens": 0,
                            "cache_write_tokens": 0, "total_tokens": 473},
              "total_including_auxiliary": {"api_calls": 3}}

    usage = a2a_server._usage(parser.result(b"", b"", 0, report))

    assert usage.to_dict() == {
        "tool_call_count": 0, "input_tokens": 456 + 9161, "output_tokens": 17 + 60, "total_tokens": 456 + 9161 + 17 + 60,
        "cache_read_tokens": 0, "cache_write_tokens": 0, "input_includes_cache": True,
        "provider_details": {"api_calls": 4, "unreported_calls": 2, "cost_status": "estimated"}}


def test_a_killed_turn_keeps_the_calls_it_made_as_a_lower_bound():
    """hermes-timeout-claude, recorded on 2026-11-20: two calls (13,073 and 13,156 in, 53 and 81 out) before the turn
    ran out of time, killed before it wrote its usage file."""
    parser = _events(_call("A", "s", 13073, 53), _call("A", "s", 13156, 81))

    usage = a2a_server._usage(result(usage={}, killed_calls=parser.all_calls))

    assert (usage.input_tokens, usage.output_tokens, usage.input_includes_cache) == (26229, 134, True)
    assert usage.provider_details == {"api_calls": 2, "unreported_calls": 1}


def test_a_turns_counts_are_its_models_its_title_call_and_childs_included():
    """Every call a turn makes runs on its model: the title call routes to the main model and a child inherits it."""
    haiku = "claude-haiku-4-5"
    parser = _events(_call("A", "parent", 13081, 82, haiku), _call("B", "child", 9161, 60, haiku),
                     _call("A", "parent", 13535, 4, haiku))
    report = {"model": "claude-haiku-4-5", "input_tokens": 26616, "output_tokens": 86, "total_tokens": 26702,
              "cache_read_tokens": 0, "cache_write_tokens": 0, "api_calls": 2, "estimated_cost_usd": 0.0271,
              "cost_status": "estimated",
              "auxiliary": {"api_calls": 1, "input_tokens": 456, "output_tokens": 17, "cache_read_tokens": 0,
                            "cache_write_tokens": 0, "total_tokens": 473},
              "total_including_auxiliary": {"api_calls": 3, "estimated_cost_usd": 0.0275}}

    alone = a2a_server._usage(result(usage=report))
    delegated = a2a_server._usage(parser.result(b"", b"", 0, report))

    assert alone.models == (ModelUsage(model="claude-haiku-4-5", calls=3, input_tokens=26616 + 456,
                                       output_tokens=86 + 17, cache_read_tokens=0, cache_write_tokens=0,
                                       input_includes_cache=True, cost_usd=0.0275),)
    (row,) = delegated.models
    assert (row.calls, row.input_tokens, row.output_tokens, row.cost_usd) == (4, 26616 + 456 + 9161, 86 + 17 + 60, None)


def test_a_turn_whose_calls_named_several_models_gives_no_rows():
    parser = _events(_call("A", "parent", 13081, 82, "claude-haiku-4-5"), _call("B", "child", 9161, 60, "gpt-4.1-mini"))
    report = {"model": "claude-haiku-4-5", "input_tokens": 13081, "output_tokens": 82, "total_tokens": 13163,
              "cache_read_tokens": 0, "cache_write_tokens": 0, "api_calls": 1}

    assert a2a_server._usage(parser.result(b"", b"", 0, report)).models == ()
    assert a2a_server._usage(result(usage={k: v for k, v in report.items() if k != "model"})).models == ()


def test_a_killed_turns_counts_are_the_model_its_calls_named():
    parser = _events(_call("A", "s", 13073, 53, "claude-haiku-4-5"), _call("A", "s", 13156, 81, "claude-haiku-4-5"))

    (row,) = a2a_server._usage(result(usage={}, killed_calls=parser.all_calls)).models

    assert (row.model, row.calls, row.input_tokens, row.output_tokens) == ("claude-haiku-4-5", 2, 26229, 134)


def test_the_profile_turns_hermes_review_fork_off():
    config = hermes_bridge.build_config("m", None, {}, None, Path("/skills"), environ={"LITELLM_BASE_URL": "http://x"})

    assert config["auxiliary"] == {"background_review": {"enabled": False}}


def test_a_turn_whose_calls_reported_no_usage_has_unknown_counts():
    """Claude's streamed answers through a LiteLLM proxy carry no usage, so Hermes reports zeros for them."""
    report = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0,
              "cache_write_tokens": 0, "api_calls": 3}

    assert a2a_server._usage(result(usage=report, tool_call_count=2)).to_dict() == {
        "tool_call_count": 2, "provider_details": {"api_calls": 3, "unreported_calls": 3}}


def test_a_turns_usage_follows_what_its_total_says_its_input_holds():
    """A total of the input and output alone says Hermes counted the cache in the input already; one that is neither
    sum keeps Hermes' rule, unconfirmed."""
    held = {"input_tokens": 21246, "output_tokens": 48, "total_tokens": 21294, "cache_read_tokens": 10496,
            "cache_write_tokens": 0, "api_calls": 2}
    neither = {**held, "input_tokens": 10750, "total_tokens": 30000}

    assert a2a_server._usage(result(usage=held, tool_call_count=1)).to_dict() == {
        "tool_call_count": 1, "input_tokens": 21246, "output_tokens": 48, "total_tokens": 21294,
        "cache_read_tokens": 10496, "cache_write_tokens": 0, "input_includes_cache": True,
        "provider_details": {"api_calls": 2}}
    assert a2a_server._usage(result(usage=neither, tool_call_count=1)).to_dict() == {
        "tool_call_count": 1, "input_tokens": 21246, "output_tokens": 48, "total_tokens": 30000,
        "cache_read_tokens": 10496, "cache_write_tokens": 0, "provider_details": {"api_calls": 2}}
