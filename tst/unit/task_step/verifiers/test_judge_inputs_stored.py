"""Each judge call's record names exactly what it was sent, stored by its sha256: a direct judge's messages, response
format and settings (the config's with the run's over them, credentials left out); an agent judge's message, every file
its step staged in its sandbox and, for a judge the task deployed, what the task's steps loaded there as the run recorded
it. The uploads run in the background and the grade waits for them, marking any that failed or hadn't finished. A
judge's settings say which trajectory source it read for each turn."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import litellm
import pytest

from agent_env.a2a_agent import protocol
from agent_env.task_step.context import DeployedAgent, PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils import judge_inputs, judge_usage
from tst.unit.task_step import test_always_on_agent_verifier as A
from tst.unit.task_step.verifiers.test_rubrics_verifier_golden import _binary, _context, _direct_judge, _step
from tst.unit.task_step.verifiers.test_rubrics_verifier_judge_settings import _grade, _llm

ANSWER = json.dumps({"results": [{"id": "c1", "score": 1.0, "justification": "it asked"}]})


class _Store:
    """An object store in memory; ``fail`` refuses every put, or only those of bytes holding it; ``gate`` holds each put
    until it's set."""

    def __init__(self, fail=False, gate: threading.Event | None = None):
        self.fail, self.gate, self.objects, self.puts = fail, gate, {}, 0

    def exists(self, key):
        return key in self.objects

    def object_url(self, key):
        return f"mem://{key}"

    def put(self, key, data, content_type=None, allow_overwrite=False):
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.fail is True or (isinstance(self.fail, bytes) and self.fail in data):
            raise ConnectionError("store unreachable")
        self.puts += 1
        self.objects[key] = data
        return f"mem://{key}"


def _using(monkeypatch, store: _Store) -> _Store:
    monkeypatch.setattr(judge_inputs, "get_config", lambda: SimpleNamespace(
        get_artifact_key_prefix=lambda: "pfx/", get_object_store=lambda: store))
    return store


@pytest.fixture
def store(monkeypatch):
    return _using(monkeypatch, _Store())


def _opened(store: _Store, ref: dict) -> dict:
    data = store.objects[ref["object_url"].removeprefix("mem://")]
    assert (hashlib.sha256(data).hexdigest(), len(data)) == (ref["sha256"], ref["bytes"])
    assert ref["object_url"] == f"mem://pfx/judge-inputs/{ref['sha256']}"
    return json.loads(data)


@pytest.mark.asyncio
async def test_a_direct_calls_record_opens_to_exactly_the_messages_format_and_settings_it_sent(monkeypatch, store):
    requests = _llm(monkeypatch, params={"temperature": 0, "max_tokens": 512, "aws_secret_access_key": "hush",
                                         "extra_body": {"top_k": 5, "auth": "hush",
                                                        "providers": [{"api_key": "hush", "order": 1}],
                                                        "relay": "https://user:hush@relay.test/v1"}},
                    headers={"X-Team-Key": "team-secret"})

    grade = await _grade(_step(default_model="claude-sonnet-4-6", effort="high"), _context())

    [call] = grade["judge_calls"]
    sent = _opened(store, call["input"])
    assert sent["messages"] == requests[0]["messages"]
    assert sent["response_format"] == requests[0]["response_format"]
    assert sent["settings"] == {"temperature": 0, "max_tokens": 512, "reasoning_effort": "high", "extra_body": {
        "top_k": 5, "providers": [{"order": 1}], "relay": "https://***@relay.test/v1"}}
    assert not any(secret in data for data in store.objects.values() for secret in (b"hush", b"team-secret"))
    assert (requests[0]["extra_headers"], requests[0]["aws_secret_access_key"]) == ({"X-Team-Key": "team-secret"}, "hush")


@pytest.mark.asyncio
async def test_a_request_cancelled_mid_call_keeps_what_it_was_sent(monkeypatch, store):
    _llm(monkeypatch)
    calling = asyncio.Event()

    async def hangs(**kwargs):
        calling.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(litellm, "acompletion", hangs)
    context, step = _context(), _step(default_model="gpt-4o-mini")
    with judge_usage.scope():
        judge = judge_usage.start(context, step.id)
        judging = asyncio.create_task(step._judge_directly("grade it", model="gpt-4o-mini", context=context,
                                                           image_blocks=None, usage=judge))
        await calling.wait()
        judging.cancel()
        with pytest.raises(asyncio.CancelledError):
            await judging
        await judge.settle()

    [call] = judge.calls
    assert (call["outcome"], call["endpoint_host"]) == ("cut_off", "gw.test")
    assert _opened(store, call["input"])["messages"] == [{"role": "user", "content": "grade it"}]


@pytest.mark.asyncio
async def test_a_request_retried_with_the_same_input_shares_one_stored_object(monkeypatch, store):
    _llm(monkeypatch, failures=[litellm.RateLimitError("slow down", llm_provider="openai", model="gpt-4o-mini")])

    grade = await _grade(_step(default_model="gpt-4o-mini"), _context())

    refused, answered = grade["judge_calls"]
    assert (refused["outcome"], answered["outcome"]) == ("refused", "answered")
    assert refused["input"] == answered["input"] and store.puts == 1


@pytest.mark.asyncio
async def test_an_input_the_store_refuses_is_recorded_and_the_judge_still_grades(monkeypatch):
    _llm(monkeypatch)
    _using(monkeypatch, _Store(fail=True))

    grade = await _grade(_step(default_model="gpt-4o-mini"), _context())

    [call] = grade["judge_calls"]
    assert "input" not in call and call["input_error"] == "ConnectionError: store unreachable"
    assert grade["verifier_status"] == "graded"


class _Sandbox:
    def __init__(self):
        self.files = {}

    async def write_file_from_text(self, text, path, **kwargs):
        self.files[path] = text

    async def write_file_from_object(self, object_url, path, **kwargs):
        self.files[path] = object_url

    async def terminate(self):
        pass


@pytest.mark.asyncio
async def test_the_staging_recorder_keeps_each_file_put_in_the_judges_sandbox(store):
    judge = judge_usage.JudgeUsage(TaskStepContext(), "verify")
    sandbox = _Sandbox()
    recorder = judge.recording(sandbox)

    await recorder.write_file_from_text("turn one", "/tmp/run/turn_01.json")
    await recorder.write_file_from_object("mem://trajectories/2.json", "/tmp/run/turn_02.json")
    await recorder.terminate()
    await judge.settle()

    text = "turn one".encode()
    assert sandbox.files == {"/tmp/run/turn_01.json": "turn one", "/tmp/run/turn_02.json": "mem://trajectories/2.json"}
    assert judge.staged == [
        {"path": "/tmp/run/turn_01.json", "object_url": f"mem://pfx/judge-inputs/{hashlib.sha256(text).hexdigest()}",
         "bytes": len(text), "sha256": hashlib.sha256(text).hexdigest()},
        {"path": "/tmp/run/turn_02.json", "object_url": "mem://trajectories/2.json"}]


@pytest.mark.asyncio
async def test_an_agent_calls_record_opens_to_its_message_and_the_files_staged_for_it(monkeypatch, store):
    async def send(url, **kwargs):
        return "task-1", None

    async def poll(url, task_id, **kwargs):
        return {"status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": ANSWER}]}}}

    monkeypatch.setattr(protocol, "send_a2a_message", send)
    monkeypatch.setattr(protocol, "poll_a2a_task", poll)
    context = TaskStepContext()
    with judge_usage.scope():
        judge = judge_usage.start(context, "verify")
        judge.staged.append({"path": "/tmp/run/turn_01.json", "object_url": "mem://trajectories/1.json"})
        verifier = _step(use_agent_judge=True, criteria=[{"id": "c1", "title": "asked"}])
        await verifier._invoke_judge_a2a(eval_prompt="grade it", judge_a2a_url="http://judge.test",
                                         judge_agent_card={}, judge_agent=SimpleNamespace(sandbox_id=None, control_headers={}),
                                         model="m", usage=judge)
        await judge.settle()

    [call] = judge.calls
    assert _opened(store, call["input"]) == {
        "message": "grade it", "files": [{"path": "/tmp/run/turn_01.json", "object_url": "mem://trajectories/1.json"}],
        "loaded": []}


def test_the_trajectory_sources_name_each_turns_format_and_an_atif_copys_adapter_version():
    from agent_env.task_step.task_steps.verifiers.rubrics_verifier import _trajectory_sources

    response = PromptResponse(prompt_id="p1", response="done", target_agent_per_turn_trajectories=[
        {"format": "claude-code-stream-json", "atif": None},
        {"format": "atif/1", "atif": {"adapter_version": "claude-code@0.3", "status": "ok"}}])

    assert _trajectory_sources(response) == [
        {"turn": 1, "format": "claude-code-stream-json"},
        {"turn": 2, "format": "atif/1", "adapter_version": "claude-code@0.3"}]


@pytest.mark.asyncio
async def test_a_judge_that_reads_the_trajectory_records_its_sources_in_its_settings(store):
    context = _context(target_agent_per_turn_trajectories=[{"format": "claude-code-stream-json", "atif": None}],
                       target_agent_per_turn_trajectory_object_urls=["mem://traj/1.json"])
    step = _direct_judge(_step(use_trajectory=True), lambda ids, n: _binary(ids))

    grade = await _grade(step, context)

    assert grade["verifier_settings"]["trajectory_sources"] == [{"turn": 1, "format": "claude-code-stream-json"}]


def _answering_judge(monkeypatch) -> list[str]:
    """The judge agent's A2A calls answered passing for every criterion asked about; each message it was sent kept."""
    sent: list[str] = []

    async def send(url, **kwargs):
        sent.append(kwargs["message"] if "message" in kwargs else str(kwargs))
        return "task-1", None

    async def poll(url, task_id, **kwargs):
        return {"status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": _binary(["c1", "c2"])}]}}}

    monkeypatch.setattr(protocol, "send_a2a_message", send)
    monkeypatch.setattr(protocol, "poll_a2a_task", poll)
    return sent


@pytest.mark.asyncio
async def test_a_judge_the_task_deployed_lists_what_the_runs_steps_loaded_into_it(monkeypatch, store):
    _answering_judge(monkeypatch)
    context = _context()
    context.deployed_agents.append(DeployedAgent(agent_name="judge", api_url="http://judge.test",
                                                 a2a_url="http://judge.test", sandbox_id="sb-judge"))
    outputs = {"id": "outputs", "version": 3, "artifact_type": "file_artifact_universe", "agent_name": "judge",
               "destination_path": "/tmp/outputs", "files": {"report.md": "/tmp/outputs/report.md"}}
    context.metadata.update(
        loaded_file_artifact_universes=[outputs, {**outputs, "agent_name": "default-agent", "destination_path": "/app"}],
        loaded_environment_artifacts=[{"id": "inbox", "version": 1, "agent_name": "judge",
                                       "destination_path": "/data/env_after", "files": ["mail/inbox.json"]}],
        loaded_urls=[{"step_id": "load-sheet", "agent_name": "judge", "destination_path": "/tmp/inputs",
                      "files": ["sheet.xlsx"]}])
    step = _step(use_agent_judge=True, agent_name="judge")

    async def configured(**kwargs):
        return None

    step._configure_judge_a2a = configured
    grade = await _grade(step, context)

    [call] = grade["judge_calls"]
    sent = _opened(store, call["input"])
    assert sent["files"] == []
    assert sent["loaded"] == [
        {"record": "loaded_file_artifact_universes", **outputs},
        {"record": "loaded_environment_artifacts", "id": "inbox", "version": 1, "agent_name": "judge",
         "destination_path": "/data/env_after", "files": ["mail/inbox.json"]},
        {"record": "loaded_urls", "step_id": "load-sheet", "agent_name": "judge", "destination_path": "/tmp/inputs",
         "files": ["sheet.xlsx"]}]


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_stores")
async def test_an_always_on_judge_keeps_each_file_it_staged_and_names_the_formats_it_guessed(monkeypatch, store):
    run, sandbox, sources = A._staged_run(monkeypatch)
    verifier = A._verifier(use_agent_judge=True, use_trajectory=True, agent_name="judge")
    seen: dict = {}

    async def judge(*, eval_prompt, **kwargs):
        record = judge_usage.current()
        seen.update(staged=list(record.staged), sources=list(record.sources or []))
        return ([{**verifier.criteria[0], "score": 1.0, "result": True}], 0, [], None)

    def read(url):
        if url not in sources:
            raise FileNotFoundError(url)
        return sources[url]

    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", judge)
    monkeypatch.setattr(verifier, "_read_trajectory_text", read)
    grade = await _grade(verifier, run)

    assert sandbox.files and {entry["path"] for entry in seen["staged"]} == set(sandbox.files)
    for entry in seen["staged"]:
        assert store.objects[f"pfx/judge-inputs/{entry['sha256']}"] == sandbox.files[entry["path"]].encode()
    assert seen["sources"] and all(source["guessed"] for source in seen["sources"])
    assert grade["verifier_settings"]["trajectory_sources"] == seen["sources"]
    assert {source["format"] for source in seen["sources"]} >= {"claude-code-stream-json"}


@pytest.mark.asyncio
async def test_the_judge_is_sent_its_message_while_its_staged_files_are_still_uploading(monkeypatch):
    released = threading.Event()
    store = _using(monkeypatch, _Store(gate=released))
    sent_while_uploading: list[bool] = []

    async def send(url, **kwargs):
        sent_while_uploading.append(not store.objects)
        released.set()
        return "task-1", None

    async def poll(url, task_id, **kwargs):
        return {"status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": ANSWER}]}}}

    monkeypatch.setattr(protocol, "send_a2a_message", send)
    monkeypatch.setattr(protocol, "poll_a2a_task", poll)
    with judge_usage.scope():
        judge = judge_usage.start(TaskStepContext(), "verify")
        await judge.recording(_Sandbox()).write_file_from_text("turn one", "/tmp/run/turn_01.json")
        verifier = _step(use_agent_judge=True, criteria=[{"id": "c1", "title": "asked"}])
        await verifier._invoke_judge_a2a(eval_prompt="grade it", judge_a2a_url="http://judge.test",
                                         judge_agent_card={}, judge_agent=SimpleNamespace(sandbox_id=None, control_headers={}),
                                         model="m", usage=judge)
        await judge.settle()

    assert sent_while_uploading == [True]
    [call] = judge.calls
    assert _opened(store, call["input"])["files"][0]["sha256"] == hashlib.sha256(b"turn one").hexdigest()


@pytest.mark.asyncio
async def test_an_upload_still_running_when_the_grade_is_written_is_marked_and_the_grade_goes_on(monkeypatch):
    released = threading.Event()
    _using(monkeypatch, _Store(gate=released))
    monkeypatch.setattr(judge_inputs, "SETTLE_SECONDS", 0.2)
    _llm(monkeypatch)

    grade = await _grade(_step(default_model="gpt-4o-mini"), _context())
    released.set()

    [call] = grade["judge_calls"]
    assert "input" not in call and call["input_error"] == "TimeoutError: still uploading after 0s"
    assert grade["verifier_status"] == "graded"


@pytest.mark.asyncio
async def test_a_staged_file_that_wasnt_stored_is_marked_in_the_manifest_and_in_the_calls_input(monkeypatch):
    store = _using(monkeypatch, _Store(fail=b"secret turn"))
    _answering_judge(monkeypatch)
    with judge_usage.scope():
        judge = judge_usage.start(TaskStepContext(), "verify")
        recorder = judge.recording(_Sandbox())
        await recorder.write_file_from_text("secret turn", "/tmp/run/turn_01.json")
        await recorder.write_file_from_text("turn two", "/tmp/run/turn_02.json")
        verifier = _step(use_agent_judge=True)
        await verifier._invoke_judge_a2a(eval_prompt="grade it", judge_a2a_url="http://judge.test",
                                         judge_agent_card={}, judge_agent=SimpleNamespace(sandbox_id=None, control_headers={}),
                                         model="m", usage=judge)
        await judge.settle()

    failed, stored = judge.staged
    assert "object_url" not in failed and failed["input_error"] == "ConnectionError: store unreachable"
    assert stored["object_url"] == f"mem://pfx/judge-inputs/{stored['sha256']}"
    [call] = judge.calls
    assert _opened(store, call["input"])["files"] == [failed, stored]


@pytest.mark.asyncio
async def test_a_store_that_cant_be_had_leaves_input_error_and_the_judge_still_grades(monkeypatch):
    def no_store():
        raise PermissionError("no credentials for the bucket")

    monkeypatch.setattr(judge_inputs, "get_config", lambda: SimpleNamespace(
        get_artifact_key_prefix=lambda: "pfx/", get_object_store=no_store))
    _llm(monkeypatch)

    grade = await _grade(_step(default_model="gpt-4o-mini"), _context())

    [call] = grade["judge_calls"]
    assert "input" not in call and call["input_error"] == "PermissionError: no credentials for the bucket"
    assert grade["verifier_status"] == "graded"


@pytest.mark.asyncio
async def test_settling_stops_at_its_deadline_even_when_an_input_is_slow_to_encode(monkeypatch, store):
    _answering_judge(monkeypatch)
    monkeypatch.setattr(judge_inputs, "SETTLE_SECONDS", 0.2)
    encode = judge_inputs._encoded
    monkeypatch.setattr(judge_inputs, "_encoded", lambda body: (time.sleep(1), encode(body))[1])
    with judge_usage.scope():
        judge = judge_usage.start(TaskStepContext(), "verify")
        verifier = _step(use_agent_judge=True)
        await verifier._invoke_judge_a2a(eval_prompt="grade it", judge_a2a_url="http://judge.test",
                                         judge_agent_card={}, judge_agent=SimpleNamespace(sandbox_id=None, control_headers={}),
                                         model="m", usage=judge)
        started = time.monotonic()
        await judge.settle()

    assert time.monotonic() - started < 0.8
    [call] = judge.calls
    assert "input" not in call and call["input_error"] == "TimeoutError: not encoded within 0s"


@pytest.mark.asyncio
async def test_uploads_held_up_by_a_slow_store_leave_the_event_loops_threads_free(monkeypatch):
    released = threading.Event()
    _using(monkeypatch, _Store(gate=released))
    uploads = judge_inputs.Uploads()
    try:
        for n in range(judge_inputs.UPLOAD_CONCURRENCY + 8):
            uploads.put(f"file {n}".encode(), "text/plain")
        assert await asyncio.wait_for(asyncio.gather(*(asyncio.to_thread(lambda: 1) for _ in range(64))), 5) == [1] * 64
    finally:
        released.set()
        await uploads.wait(uploads.deadline())
