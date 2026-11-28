"""Hermes' ATIF adapter on hook events shaped as the agent's plugin writes them, the server's ``turn`` records beside
them. Each case runs through the protocol's adapter kit, so the live rule holds after every record and the document
validates as a producer's, and its counts are held to the usage the turn reports, as the bridge and server count it
from the same events."""
from __future__ import annotations

import base64
import json
import random
import re
import struct
import zlib
from datetime import datetime, timezone

import pytest
from agentenv_protocol.atif import Fed, RecordedTask, TurnContext, check_adapter, errors_in, profile_of, validate

from tst.unit.agents.hermes.agent import AGENT_DIR, a2a_server, hermes_atif, hermes_bridge
from tst.unit.agents.hermes.conftest import result

HermesAtif = hermes_atif.HermesAtif
MODEL, SERVED = "anthropic/claude-haiku-4-5", "claude-haiku-4-5"
SESSION = "20261127_101500_a1b2c3"
T0 = datetime(2026, 11, 27, 10, 15, tzinfo=timezone.utc).timestamp()


def png(width=4, height=3) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def counts(prompt=100, output=20, read=0, write=0, reasoning=0) -> dict:
    """A call's usage as Hermes normalizes it: ``input_tokens`` the prompt less its cache, which ``prompt_tokens``
    counts."""
    return {"input_tokens": prompt - read - write, "output_tokens": output, "cache_read_tokens": read,
            "cache_write_tokens": write, "reasoning_tokens": reasoning, "request_count": 1, "prompt_tokens": prompt,
            "total_tokens": prompt + output}


class Feed:
    """One task's records in arrival order, a quarter second apart: the plugin's events for a session's turn, and the
    server's ``turn`` records."""

    clock = 0.0
    turns = 0

    def __init__(self, session=SESSION):
        self.session, self.records = session, []
        self.turn = self.new_turn(session)
        self.calls: dict[str, int] = {}

    @staticmethod
    def new_turn(session):
        Feed.turns += 1
        return f"{session}:task-{Feed.turns}:{Feed.turns:08x}"

    def now(self) -> float:
        Feed.clock += 0.25
        return T0 + Feed.clock

    def event(self, hook, session=None, turn=None, **fields):
        at = self.now()
        record = {"type": hook, "timestamp": int(at * 1000), "session_id": session or self.session}
        if turn is not False:
            record["turn_id"] = turn or self.turn
        record.update(fields)
        self.records.append(Fed(record))
        return self

    def aux(self, record):
        self.records.append(Fed(record, aux="turn"))
        return self

    def at(self):
        return self.records[-1].record

    def start(self, session=None):
        return self.event("on_session_start", session=session, turn=False, model=MODEL)

    def call(self, text=None, calls=(), usage=None, reasoning=None, old=False, session=None, turn=None):
        """A model call's post_api_request: its calls ``(id, name, arguments)``, usage and text; ``old`` as the plugin
        wrote it before it kept the message whole, its usage only in Hermes' record of the response."""
        turn = turn or self.turn
        number = self.calls[turn] = self.calls.get(turn, 0) + 1
        usage = usage if usage is not None else counts()
        tool_calls = [{"id": call_id, "name": name, "arguments": json.dumps(arguments)}
                      for call_id, name, arguments in calls]
        finish = "tool_calls" if calls else "stop"
        cut = text if text is None or len(text) <= 8000 else text[:8000] + f"...[truncated {len(text) - 8000} chars]"
        response = {"model": SERVED, "upstream_provider": None, "finish_reason": finish, "usage": usage,
                    "assistant_message": {"role": "assistant", "content": cut,
                                          "tool_calls": [{**call, "provider_data": None} for call in tool_calls]}}
        fields = {"api_call_count": number, "model": MODEL, "finish_reason": finish, "response": response}
        if not old:
            message = {key: value for key, value in (("content", text), ("reasoning", reasoning),
                                                     ("tool_calls", tool_calls)) if value}
            fields = {"api_request_id": f"{turn}:api:{number}", **fields, "response_model": SERVED,
                      "started_at": T0 + Feed.clock + 0.1, "usage": usage, "message": message}
        return self.event("post_api_request", session=session, turn=turn, **fields)

    def error(self, message="Connection error.", kind="APIConnectionError", status=None, reason="timeout",
              retryable=True, retry=1, session=None, turn=None):
        turn = turn or self.turn
        number = self.calls.get(turn, 0) + 1
        fields = {"api_request_id": f"{turn}:api:{number}", "api_call_count": number, "retry_count": retry,
                  "model": MODEL, "started_at": T0 + Feed.clock + 0.1, "reason": reason, "retryable": retryable,
                  "error": {"type": kind, "message": message}}
        if status is not None:
            fields["status_code"] = status
        return self.event("api_request_error", session=session, turn=turn, **fields)

    def pre(self, call_id, name="terminal", args=None, session=None, turn=None):
        return self.event("pre_tool_call", session=session, turn=turn, api_request_id="", tool_call_id=call_id,
                          tool_name=name, args=args if args is not None else {"command": "ls"})

    def post(self, call_id, name="terminal", output="ok", status="ok", session=None, turn=None):
        fields = {"api_request_id": "", "tool_call_id": call_id, "tool_name": name, "result": output,
                  "status": status, "duration_ms": 12}
        if status == "error":
            fields.update(error_type="tool_error", error_message="failed")
        return self.event("post_tool_call", session=session, turn=turn, **fields)

    def shell(self, call_id, command="python3 hello.py", output="5050", exit_code=0, **where):
        result = json.dumps({"output": output, "exit_code": exit_code, "error": None})
        return self.pre(call_id, args={"command": command}, **where).post(call_id, output=result, **where)

    def auxiliary(self, task="title_generation", usage=None, content="Sum script", error=None, session=None,
                  turn=None):
        fields = {"api_request_id": "", "aux_task": task, "model": MODEL, "response_model": SERVED,
                  "started_at": T0 + Feed.clock + 0.1}
        if error is None:
            fields.update(usage=usage if usage is not None else counts(300, 10), content=content)
        else:
            fields["error"] = error
        return self.event("post_auxiliary_call", session=session, turn=turn, **fields)

    def answer(self, text, session=None, turn=None):
        return self.event("post_llm_call", session=session, turn=turn, assistant_response=text)

    def end(self, failed=False, session=None, turn=None):
        return self.event("on_session_end", session=session, turn=turn, completed=not failed, failed=failed,
                          interrupted=False, turn_exit_reason="text_response(finish_reason=stop)")

    def spawn(self, child, subagent_id="sa-0-1a2b3c4d", goal="count the lines"):
        at = self.now()
        self.records.append(Fed({"type": "subagent_start", "timestamp": int(at * 1000), "parent_session_id":
                                 self.session, "child_session_id": child, "child_subagent_id": subagent_id,
                                 "child_role": "leaf", "child_goal": goal}))
        return self

    def stop(self, child, summary="19", status="completed"):
        at = self.now()
        self.records.append(Fed({"type": "subagent_stop", "timestamp": int(at * 1000), "parent_session_id":
                                 self.session, "child_session_id": child, "child_status": status,
                                 "child_summary": summary}))
        return self


def context(index=0):
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=["do it"],
                       agent_name="Hermes Agent", agent_version="1.0.0", agent_model=MODEL,
                       started_at=datetime.fromtimestamp(T0 + index, tz=timezone.utc))


def run(*feeds, finals=None, states=None):
    tasks = []
    for index, feed in enumerate(feeds):
        final = (finals or {}).get(index, _answer(feed))
        tasks.append(RecordedTask(context=context(index), records=feed.records,
                                  state=(states or {}).get(index, "completed"),
                                  final=[final] if final is not None else None))
    kit = check_adapter(HermesAtif, tasks)
    kit.raise_for_errors()
    assert not errors_in(validate(kit.document, media=kit.media))
    return kit


def _answer(feed):
    answers = [fed.record["assistant_response"] for fed in feed.records
               if fed.aux is None and fed.record["type"] == "post_llm_call"]
    return answers[-1].strip() if answers else None


def reported(feed, *, killed=False):
    """The usage the server reports for the task: the bridge's counts of its events, the usage file Hermes writes from
    the turn's own main and auxiliary calls, and the server's sum of both."""
    parser = hermes_bridge._EventParser()
    natives = [fed.record for fed in feed.records if fed.aux is None]
    for record in natives:
        parser.on_line(json.dumps(record))
    if killed:
        return a2a_server._usage(result(usage={}, killed_calls=parser.all_calls))
    main = [r["usage"] for r in natives if r["type"] == "post_api_request" and r.get("turn_id") == parser.turn_id]
    aux = [r["usage"] for r in natives if r["type"] == "post_auxiliary_call" and not r.get("error")
           and r.get("session_id") in (None, "", parser.session_id)]

    def total(calls, key):
        return sum(call[key] for call in calls)

    report = {"model": MODEL, "api_calls": len(main), "cost_status": "unknown", "estimated_cost_usd": 0.0,
              **{key: total(main, key) for key in ("input_tokens", "output_tokens", "cache_read_tokens",
                                                   "cache_write_tokens", "total_tokens")}}
    if aux:
        report["auxiliary"] = {"api_calls": len(aux), **{key: total(aux, key) for key in (
            "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens")}}
        report["total_including_auxiliary"] = {"api_calls": len(main) + len(aux)}
    return a2a_server._usage(parser.result(b"", b"", 0, report))


def assert_sums_to(document, *usages):
    final = document["final_metrics"]
    assert final["total_prompt_tokens"] == sum(usage.input_tokens for usage in usages)
    assert final["total_completion_tokens"] == sum(usage.output_tokens for usage in usages)
    assert final["total_cached_tokens"] == sum(usage.cache_read_tokens for usage in usages)
    assert profile_of(final)["total_cache_write_tokens"] == sum(usage.cache_write_tokens for usage in usages)


def agent_steps(trajectory):
    return [step for step in trajectory["steps"] if step["source"] == "agent"]


def subagents(document):
    return {sub["trajectory_id"]: sub for sub in document.get("subagent_trajectories") or ()}


def every_call_answered(document):
    for trajectory in [document, *subagents(document).values()]:
        for step in agent_steps(trajectory):
            answered = {r["source_call_id"] for r in (step.get("observation") or {}).get("results") or ()}
            assert {call["tool_call_id"] for call in step.get("tool_calls") or ()} <= answered, step["step_id"]


def coding(feed=None, *, old=False):
    feed = feed or Feed().start()
    feed.call("I'll write it.", calls=[("toolu_1", "write_file", {"path": "hello.py", "content": "print(5050)"})],
              usage=counts(13365, 65, write=13362), old=old)
    feed.pre("toolu_1", "write_file", {"path": "hello.py", "content": "print(5050)"})
    feed.post("toolu_1", "write_file", json.dumps({"bytes_written": 11}))
    feed.call(calls=[("toolu_2", "terminal", {"command": "python3 hello.py"})], usage=counts(13500, 40, read=13362),
              old=old)
    feed.shell("toolu_2")
    feed.call("5050\n", usage=counts(13600, 4, read=13450, write=50), old=old)
    return feed.answer("5050\n").end()


def test_a_model_call_is_one_step_with_its_model_start_counts_and_the_calls_hermes_ran_for_it():
    feed = coding()
    first_call = feed.records[1].record
    kit = run(feed)
    steps = agent_steps(kit.document)

    assert len(steps) == 3
    assert steps[0]["model_name"] == MODEL
    assert steps[0]["timestamp"] == datetime.fromtimestamp(first_call["started_at"], tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")
    assert steps[0]["message"] == "I'll write it."
    assert steps[0]["metrics"] == {"prompt_tokens": 13365, "cached_tokens": 0, "completion_tokens": 65,
                                   "extra": {"agentenv": {"cache_write_tokens": 13362}}}
    assert steps[0]["tool_calls"] == [{"tool_call_id": "toolu_1", "function_name": "write_file",
                                       "arguments": {"path": "hello.py", "content": "print(5050)"}}]
    (shell,) = steps[1]["observation"]["results"]
    assert json.loads(shell["content"])["output"] == "5050"
    assert profile_of(shell)["exit_code"] == 0
    assert_sums_to(kit.document, reported(feed))


def test_the_last_response_is_the_turns_answer_so_the_sdk_adds_no_step_of_its_own():
    kit = run(coding())
    steps = agent_steps(kit.document)

    assert steps[-1]["message"] == "5050" and steps[-1]["llm_call_count"] == 1
    assert profile_of(steps[-1])["state"] == "completed"


def test_a_record_from_before_the_plugin_kept_the_message_reads_the_response_hermes_recorded():
    new, old = run(coding()).document, run(coding(old=True)).document

    def view(document):
        return [(step["message"], step["metrics"], step.get("tool_calls")) for step in agent_steps(document)]

    assert view(old) == view(new)
    assert all("timestamp" in step for step in agent_steps(old))


def test_a_long_answer_is_whole_though_hermes_cut_its_record_of_the_response():
    text = "x" * 9000
    feed = Feed().start().call(calls=[("toolu_1", "terminal", {"command": "yes"})]).shell("toolu_1")
    feed.call(text + " and more", calls=[("toolu_2", "terminal", {"command": "ls"})]).shell("toolu_2")
    feed.call("done").answer("done").end()
    steps = agent_steps(run(feed).document)

    assert steps[1]["message"] == text + " and more"


def test_the_tool_search_wrapper_and_a_connector_batch_are_the_calls_hermes_ran():
    """Hermes unwraps ``tool_call``: the first entry keeps the model's id, a later one gets an id of Hermes' own."""
    wrapper = {"calls": [{"name": "mcp__env__search_emails", "arguments": {"query": "invoice"}},
                         {"name": "mcp__env__get_time", "arguments": {}}]}
    feed = Feed().start().call("Searching.", calls=[("toolu_1", "tool_call", wrapper)])
    feed.pre("toolu_1", "mcp__env__search_emails", {"query": "invoice"}).pre("call_cc9323bd604c", "mcp__env__get_time",
                                                                            {})
    feed.post("toolu_1", "mcp__env__search_emails", '{"result": "[]"}')
    feed.post("call_cc9323bd604c", "mcp__env__get_time", '{"result": "2027-02-01T12:00:14Z"}')
    feed.call("No invoices.").answer("No invoices.").end()
    first = agent_steps(run(feed).document)[0]

    assert [(call["tool_call_id"], call["function_name"], call["arguments"]) for call in first["tool_calls"]] == [
        ("toolu_1", "mcp__env__search_emails", {"query": "invoice"}), ("call_cc9323bd604c", "mcp__env__get_time", {})]
    assert [result["source_call_id"] for result in first["observation"]["results"]] == ["toolu_1", "call_cc9323bd604c"]


def test_a_tool_error_keeps_its_status():
    feed = Feed().start().call(calls=[("toolu_1", "tool_call", {"calls": [{"name": "nope", "arguments": {}}]})])
    feed.pre("toolu_1", "tool_call", {"calls": [{"name": "nope", "arguments": {}}]})
    feed.post("toolu_1", "tool_call", '{"error": "\'nope\' is not a known tool name."}', status="error")
    feed.call("I can't.").answer("I can't.").end()
    (result_,) = agent_steps(run(feed).document)[0]["observation"]["results"]

    assert profile_of(result_)["status"] == "error"


def test_a_batch_entry_hermes_refused_before_its_pre_tool_call_is_a_call_in_its_step_with_the_refusal():
    """As recorded: the second entry of a tool search batch lacked an argument, so Hermes' scope check refused it before
    the pre_tool_call hook, under an id of its own and the wrapper's name; the model's next call answers the refusal."""
    wrapper = {"calls": [{"name": "mcp__browser__browser_navigate", "arguments": {"url": "https://example.com"}},
                         {"name": "mcp__browser__browser_take_screenshot", "arguments": {}}]}
    refusal = '{"error": "tool_call to \'mcp__browser__browser_take_screenshot\' is missing required argument(s): scale"}'
    feed = Feed().start().call("Opening it.", calls=[("toolu_1", "tool_call", wrapper)])
    feed.pre("toolu_1", "mcp__browser__browser_navigate", {"url": "https://example.com"})
    feed.post("toolu_1", "mcp__browser__browser_navigate", '{"result": "### Page"}')
    feed.event("post_tool_call", api_request_id="", tool_call_id="call_a7b35e644820", tool_name="tool_call",
               result=refusal, status="blocked", duration_ms=0, error_type="tool_scope_block", error_message="missing")
    feed.call("With the scale, then.").answer("With the scale, then.").end()
    kit = run(feed)
    first = agent_steps(kit.document)[0]

    assert [(call["tool_call_id"], call["function_name"], call["arguments"]) for call in first["tool_calls"]] == [
        ("toolu_1", "mcp__browser__browser_navigate", {"url": "https://example.com"}),
        ("call_a7b35e644820", "tool_call", {})]
    refused = first["observation"]["results"][1]
    assert (refused["source_call_id"], refused["content"], profile_of(refused)["status"]) == (
        "call_a7b35e644820", refusal, "error")
    assert "unmatched" not in profile_of(kit.document)


def test_a_call_hermes_refused_under_the_models_own_id_keeps_the_models_name_and_arguments():
    feed = Feed().start().call(calls=[("toolu_1", "terminal", {"command": "rm -rf /"}), ("toolu_2", "terminal", {})])
    feed.event("post_tool_call", api_request_id="", tool_call_id="toolu_1", tool_name="terminal",
               result='{"error": "Blocked by guardrail policy"}', status="blocked", duration_ms=0,
               error_type="guardrail_block", error_message="Blocked by guardrail policy")
    feed.shell("toolu_2", "ls", "hello.py")
    feed.call("I won't.").answer("I won't.").end()
    first = agent_steps(run(feed).document)[0]

    assert [(call["tool_call_id"], call["arguments"]) for call in first["tool_calls"]] == [
        ("toolu_1", {"command": "rm -rf /"}), ("toolu_2", {"command": "ls"})]
    assert [r["source_call_id"] for r in first["observation"]["results"]] == ["toolu_1", "toolu_2"]


def test_an_execute_code_scripts_own_tool_calls_which_carry_no_id_are_left_to_the_native_log():
    """As Hermes runs a script's calls through its tool RPC: handle_function_call with no ids, so each fires
    pre_tool_call with an empty id, session and turn, and no post_tool_call; the script's own result says what it
    made."""
    code = {"code": "from hermes_tools import terminal, read_file\nterminal('ls'); read_file('a.py')"}
    feed = Feed().start().call(calls=[("toolu_1", "execute_code", code)]).pre("toolu_1", "execute_code", code)
    for name in ("terminal", "read_file"):
        feed.records.append(Fed({"type": "pre_tool_call", "timestamp": int(feed.now() * 1000), "session_id": "",
                                 "turn_id": "", "api_request_id": "", "tool_call_id": "", "tool_name": name,
                                 "args": {}}))
    feed.post("toolu_1", "execute_code", json.dumps({"status": "success", "output": "a.py\n", "tool_calls_made": 2}))
    kit = run(feed.call("Done.").answer("Done.").end())

    assert [call["tool_call_id"] for call in agent_steps(kit.document)[0]["tool_calls"]] == ["toolu_1"]
    every_call_answered(kit.document)
    assert profile_of(kit.document)["unmatched"] == {"pre_tool_call": 2}


@pytest.mark.parametrize(("status", "expected"), [
    ("ok", None), ("error", "error"), ("blocked", "error"), ("cancelled", "canceled"), ("timeout", "timeout")])
def test_a_results_status_is_hermes_tool_executors_in_atifs_words(status, expected):
    feed = Feed().start().call(calls=[("toolu_1", "terminal", {"command": "sleep 900"})]).pre("toolu_1")
    feed.event("post_tool_call", api_request_id="", tool_call_id="toolu_1", tool_name="terminal", result="{}",
               status=status, duration_ms=12)
    (result_,) = agent_steps(run(feed.call("Stopped.").answer("Stopped.").end()).document)[0]["observation"]["results"]

    assert profile_of(result_).get("status") == expected


def test_a_screenshot_hermes_hands_the_model_by_its_cache_path_is_a_media_file_from_the_bytes_the_plugin_read():
    """As recorded on the endpoint's route, which takes no images: the result names the cached file, and the model
    reads it later through vision_analyze."""
    image, path = png(), "/data/.hermes/profiles/ctx-1/cache/images/img_7d5bbce47a8d.png"
    text = json.dumps({"result": f"### Result\n- [Screenshot of viewport](page.png)\nMEDIA:{path}"})
    feed = Feed().start().call(calls=[("toolu_1", "mcp__browser__browser_take_screenshot", {"scale": "css"})])
    feed.pre("toolu_1", "mcp__browser__browser_take_screenshot", {"scale": "css"})
    feed.event("post_tool_call", api_request_id="", tool_call_id="toolu_1",
               tool_name="mcp__browser__browser_take_screenshot", result=text, status="ok", duration_ms=51,
               media=[{"path": path, "media_type": "image/png", "data": base64.b64encode(image).decode()}])
    feed.call("Example Domain").answer("Example Domain").end()
    kit = run(feed)

    (result_,) = agent_steps(kit.document)[0]["observation"]["results"]
    said, picture = result_["content"]
    assert said == {"type": "text", "text": text}
    assert picture["type"] == "image" and kit.media[picture["source"]["path"]] == image
    assert profile_of(result_)["media"][picture["source"]["path"]]["width"] == 4


def test_an_image_a_tool_hands_the_model_is_a_media_file_named_by_its_bytes():
    image = png()
    envelope = {"_multimodal": True, "text_summary": "screenshot", "content": [
        {"type": "text", "text": "### Result"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}}]}
    feed = Feed().start().call(calls=[("toolu_1", "mcp__browser__browser_take_screenshot", {})])
    feed.pre("toolu_1", "mcp__browser__browser_take_screenshot", {})
    feed.event("post_tool_call", api_request_id="", tool_call_id="toolu_1",
               tool_name="mcp__browser__browser_take_screenshot", result=envelope, status="ok", duration_ms=80)
    feed.call("Example Domain").answer("Example Domain").end()
    kit = run(feed)

    (result_,) = agent_steps(kit.document)[0]["observation"]["results"]
    text, picture = result_["content"]
    assert text == {"type": "text", "text": "### Result"}
    assert picture["type"] == "image" and kit.media[picture["source"]["path"]] == image
    assert profile_of(result_)["media"][picture["source"]["path"]]["width"] == 4


@pytest.mark.parametrize(("reasoning", "tokens", "expected"), [
    ("The sum is n(n+1)/2.", 12, ("The sum is n(n+1)/2.", None)),
    (None, 40, (None, [{"chars": 0, "visibility": "omitted"}])),
    (None, 0, (None, None)),
])
def test_reasoning_is_its_text_or_omitted_when_the_call_only_counted_it(reasoning, tokens, expected):
    feed = Feed().start().call("5050", usage=counts(200, 50, reasoning=tokens), reasoning=reasoning)
    step = agent_steps(run(feed.answer("5050").end()).document)[0]

    assert (step.get("reasoning_content"), profile_of(step).get("reasoning")) == expected


def test_failed_attempts_are_steps_with_their_errors_and_the_retry_that_answered_is_its_own():
    """Nine connection errors on one call, as the endpoint dropped them, then the call that answered."""
    feed = Feed().start().call(calls=[("toolu_1", "terminal", {"command": "echo first"})]).shell("toolu_1", "echo first",
                                                                                               "first")
    for retry in range(1, 10):
        feed.error(retry=retry)
    feed.call("first").answer("first").end()
    kit = run(feed)
    steps = agent_steps(kit.document)

    errors = steps[1:10]
    assert all(step["llm_call_count"] == 0 and "metrics" not in step for step in errors)
    assert profile_of(errors[0])["error"] == {"kind": "infra_error", "message": "Connection error.", "code": "timeout"}
    assert steps[-1]["message"] == "first" and steps[-1]["metrics"]["prompt_tokens"] == 100
    assert_sums_to(kit.document, reported(feed))


def test_a_refused_call_fails_the_task_after_its_error_step():
    feed = Feed().start().error("Error code: 400 - invalid model", "BadRequestError", status=400, reason="bad_request",
                                retryable=False).end(failed=True)
    kit = run(feed, finals={0: "Error code: 400 - invalid model"}, states={0: "failed"})
    error, final = agent_steps(kit.document)

    assert profile_of(error)["error"] == {"kind": "infra_error", "message": "Error code: 400 - invalid model",
                                          "code": "400"}
    assert final["llm_call_count"] == 0 and profile_of(final)["state"] == "failed"


def test_an_auxiliary_call_is_a_step_of_its_usage_alone_and_the_turns_last_response_stays_its_last_step():
    """The title call ends after the turn's last response, in the thread Hermes runs it in."""
    feed = Feed().start().call(calls=[("toolu_1", "terminal", {"command": "ls"})]).shell("toolu_1")
    feed.auxiliary("vision", counts(900, 30), content="A chart.").call("42")
    feed.auxiliary("title_generation", counts(288, 10), content="Answer: 42").answer("42").end()
    kit = run(feed)
    steps = agent_steps(kit.document)

    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in steps] == [
        ("", 100), ("", 900), ("", 288), ("42", 100)]
    assert all(step["model_name"] == MODEL and step["timestamp"] for step in steps)
    assert_sums_to(kit.document, reported(feed))


def test_a_failed_auxiliary_attempt_is_a_step_with_its_error():
    feed = Feed().start().auxiliary(error="APITimeoutError: Request timed out.").call("42").answer("42").end()
    first = agent_steps(run(feed).document)[0]

    assert first["llm_call_count"] == 0
    assert profile_of(first)["error"]["message"] == "title_generation: APITimeoutError: Request timed out."


def test_a_summary_hermes_committed_is_a_compaction_on_its_compression_calls_step():
    feed = Feed().start().call(calls=[("toolu_1", "terminal", {"command": "cat big.log"})]).shell("toolu_1")
    feed.auxiliary("compression", counts(40000, 700), content="## Summary\nThe log was read.")
    feed.event("post_compaction", turn=False, old_session_id="", kind="summarize", reason="pre_api",
               before_tokens=40000, after_tokens=3100, messages_removed=12)
    feed.call("The log is clean.").answer("The log is clean.").end()
    kit = run(feed)
    summary = agent_steps(kit.document)[1]

    assert summary["message"] == "## Summary\nThe log was read." and summary["metrics"]["prompt_tokens"] == 40000
    assert profile_of(summary)["compaction"] == {"written_by": "assistant", "trigger": "auto", "tokens_before": 40000,
                                                 "tokens_after": 3100}
    assert_sums_to(kit.document, reported(feed))


def delegated(feed, child="20261127_101530_c0ffee", *, announce=True):
    """The turn delegating to a child that runs one command and answers, its events under its own session and turn."""
    feed.call("Delegating.", calls=[("toolu_d", "delegate_task", {"goal": "count the lines"})])
    feed.pre("toolu_d", "delegate_task", {"goal": "count the lines"})
    if announce:
        feed.spawn(child)
    sub = Feed.new_turn(child)
    feed.start(session=child)
    feed.call(calls=[("toolu_c1", "terminal", {"command": "wc -l /etc/passwd"})], usage=counts(9161, 60),
              session=child, turn=sub)
    feed.shell("toolu_c1", "wc -l /etc/passwd", "19 /etc/passwd", session=child, turn=sub)
    feed.auxiliary("vision", counts(500, 8), session=child, turn=sub)
    feed.call("19", usage=counts(9251, 9), session=child, turn=sub).answer("19", session=child, turn=sub)
    feed.end(session=child, turn=sub)
    if announce:
        feed.stop(child)
    feed.post("toolu_d", "delegate_task", json.dumps({"results": [{"status": "completed", "summary": "19"}]}))
    return feed.call("19").answer("19").end()


@pytest.mark.parametrize("announce", [True, False])
def test_a_delegate_task_child_is_a_sub_agent_under_the_call_that_spawned_it(announce):
    child = "20261127_101530_c0ffee"
    feed = delegated(Feed().start(), child, announce=announce)
    kit = run(feed)

    spawn = agent_steps(kit.document)[0]["observation"]["results"][0]
    assert spawn["subagent_trajectory_ref"] == [{"trajectory_id": child}]
    sub = subagents(kit.document)[child]
    assert sub["agent"]["name"] == ("sa-0-1a2b3c4d" if announce else child)
    assert [(step["message"], step["metrics"]["prompt_tokens"]) for step in agent_steps(sub)] == [
        ("", 9161), ("", 500), ("19", 9251)]
    assert agent_steps(sub)[0]["observation"]["results"][0]["source_call_id"] == "toolu_c1"
    assert_sums_to(kit.document, reported(feed))
    every_call_answered(kit.document)


def test_a_batchs_later_children_are_each_under_a_call_of_its_own_beside_the_one_the_model_made():
    first, second = "20261127_101530_c0ffee", "20261127_101531_beef00"
    tasks = {"tasks": [{"goal": "count the lines"}, {"goal": "list the users"}]}
    feed = Feed().start().call("Delegating both.", calls=[("toolu_d", "delegate_task", tasks)])
    feed.pre("toolu_d", "delegate_task", tasks).spawn(first).spawn(second, "sa-1-5e6f7a8b", "list the users")
    for child, answer in ((first, "19"), (second, "root, agent")):
        sub = Feed.new_turn(child)
        feed.call(answer, usage=counts(9000, 5), session=child, turn=sub).answer(answer, session=child, turn=sub)
    feed.stop(first).stop(second, "root, agent")
    feed.post("toolu_d", "delegate_task", json.dumps({"results": [{"summary": "19"}, {"summary": "root, agent"}]}))
    feed.call("19; root, agent").answer("19; root, agent").end()
    kit = run(feed)
    step = agent_steps(kit.document)[0]

    assert [(call["tool_call_id"], call["arguments"]) for call in step["tool_calls"]] == [
        ("toolu_d", tasks), (f"toolu_d:{second}", {"goal": "list the users"})]
    results = {r["source_call_id"]: r for r in step["observation"]["results"]}
    assert results[f"toolu_d:{second}"]["content"] == "root, agent"
    assert {r["subagent_trajectory_ref"][0]["trajectory_id"] for r in results.values()} == {first, second}
    assert_sums_to(kit.document, reported(feed))


@pytest.mark.parametrize(("status", "expected"), [
    ("completed", None), ("failed", "error"), ("error", "error"), ("interrupted", "canceled"), ("timeout", "timeout")])
def test_a_batchs_later_child_ends_its_call_with_the_status_hermes_gave_it(status, expected):
    """Hermes' delegation ends a child completed, failed, interrupted, or, when it raised or outran its timeout, error
    or timeout (tools/delegate_tool_child_run.py)."""
    first, second = "20261127_101530_c0ffee", "20261127_101531_beef00"
    tasks = {"tasks": [{"goal": "count the lines"}, {"goal": "list the users"}]}
    feed = Feed().start().call("Delegating both.", calls=[("toolu_d", "delegate_task", tasks)])
    feed.pre("toolu_d", "delegate_task", tasks).spawn(first).spawn(second, "sa-1-5e6f7a8b", "list the users")
    for child in (first, second):
        sub = Feed.new_turn(child)
        feed.call("19", usage=counts(9000, 5), session=child, turn=sub).answer("19", session=child, turn=sub)
    feed.stop(first).stop(second, "", status)
    feed.post("toolu_d", "delegate_task", json.dumps({"results": [{"summary": "19"}, {"status": status}]}))
    step = agent_steps(run(feed.call("19").answer("19").end()).document)[0]

    results = {r["source_call_id"]: r for r in step["observation"]["results"]}
    assert profile_of(results[f"toolu_d:{second}"]).get("status") == expected


def test_a_childs_events_after_a_compaction_rotated_its_session_stay_in_its_own_trajectory():
    """Hermes moves a compacted transcript to a new session (agent/conversation_compression.py), and the child's later
    events, and its subagent_stop, name that one."""
    first, second, rotated = "20261127_101530_c0ffee", "20261127_101531_beef00", "20261127_101540_d00d00"
    tasks = {"tasks": [{"goal": "count the lines"}, {"goal": "list the users"}]}
    feed = Feed().start().call("Delegating both.", calls=[("toolu_d", "delegate_task", tasks)])
    feed.pre("toolu_d", "delegate_task", tasks).spawn(first).spawn(second, "sa-1-5e6f7a8b", "list the users")
    sub = Feed.new_turn(second)
    feed.call(calls=[("toolu_c1", "terminal", {"command": "cat /etc/passwd"})], session=second, turn=sub)
    feed.shell("toolu_c1", "cat /etc/passwd", "root:x:0:0", session=second, turn=sub)
    feed.auxiliary("compression", counts(40000, 700), content="## Summary\nRead it.", session=second, turn=sub)
    feed.event("post_compaction", session=rotated, turn=False, old_session_id=second, kind="summarize",
               reason="pre_api", before_tokens=40000, after_tokens=3100)
    feed.call(calls=[("toolu_c2", "terminal", {"command": "cut -d: -f1 /etc/passwd"})], session=rotated, turn=sub)
    feed.shell("toolu_c2", "cut -d: -f1 /etc/passwd", "root", session=rotated, turn=sub)
    feed.call("root, agent", session=rotated, turn=sub).answer("root, agent", session=rotated, turn=sub)
    other = Feed.new_turn(first)
    feed.call("19", usage=counts(9000, 5), session=first, turn=other).answer("19", session=first, turn=other)
    feed.stop(first).stop(rotated, "root, agent")
    feed.post("toolu_d", "delegate_task", json.dumps({"results": [{"summary": "19"}, {"summary": "root, agent"}]}))
    kit = run(feed.call("19; root, agent").answer("19; root, agent").end())

    assert set(subagents(kit.document)) == {first, second}
    steps = agent_steps(subagents(kit.document)[second])
    assert [call["tool_call_id"] for step in steps for call in step.get("tool_calls") or ()] == ["toolu_c1", "toolu_c2"]
    assert [bool(profile_of(step).get("compaction")) for step in steps] == [False, True, False, False]
    results = {r["source_call_id"]: r for r in agent_steps(kit.document)[0]["observation"]["results"]}
    assert results[f"toolu_d:{second}"]["content"] == "root, agent"
    assert_sums_to(kit.document, reported(feed))
    every_call_answered(kit.document)


def test_a_child_no_call_waits_for_is_left_to_the_native_log():
    feed = Feed().start().call("Thinking.")
    feed.call("ok", usage=counts(50, 2), session="20261127_101530_c0ffee", turn=Feed.new_turn("x"))
    kit = run(feed.call("done").answer("done").end())

    assert not subagents(kit.document)
    assert profile_of(kit.document)["unmatched"] == {"post_api_request": 1}


def test_a_resumed_turn_starts_its_own_steps_with_no_session_start():
    first, second = coding(), Feed()
    second.call(calls=[("toolu_9", "terminal", {"command": "python3 hello.py"})], usage=counts(14000, 30, 13600))
    second.shell("toolu_9", output="500500").call("500500", usage=counts(14100, 6, 14000)).answer("500500").end()
    kit = run(first, second)
    steps = agent_steps(kit.document)

    assert [step["message"] for step in steps] == ["I'll write it.", "", "5050", "", "500500"]
    assert [profile_of(step)["turn"] for step in kit.document["steps"] if step["source"] == "user"] == [0, 1]
    assert_sums_to(kit.document, reported(first), reported(second))


def test_a_killed_turns_steps_are_the_calls_its_events_counted_and_a_call_hermes_never_ran_stays_unanswered():
    feed = Feed().start().auxiliary(usage=counts(288, 10))
    feed.call("Running.", calls=[("toolu_1", "terminal", {"command": "sleep 600"}), ("toolu_2", "terminal", {})])
    feed.pre("toolu_1", args={"command": "sleep 600"})
    kit = run(feed, finals={0: "hermes timed out after 3s"}, states={0: "failed"})
    step = next(step for step in agent_steps(kit.document) if step.get("tool_calls"))

    assert [call["tool_call_id"] for call in step["tool_calls"]] == ["toolu_1", "toolu_2"]
    assert "observation" not in step
    assert_sums_to(kit.document, reported(feed, killed=True))


def test_the_turn_records_give_the_root_its_effort_and_the_task_hermes_cost():
    feed = Feed().aux({"type": "start", "effort": "high"})
    coding(feed.start())
    feed.aux({"type": "end", "cost_usd": 0.0271})
    kit = run(feed)

    assert {step.get("reasoning_effort") for step in agent_steps(kit.document) if step["llm_call_count"]} == {"high"}
    assert kit.document["final_metrics"]["total_cost_usd"] == 0.0271
    assert profile_of(kit.document["final_metrics"])["cost_source"] == "harness"


def test_an_unpriced_turn_has_no_cost():
    feed = Feed().aux({"type": "start", "effort": None})
    coding(feed.start()).aux({"type": "end", "cost_usd": None})
    document = run(feed).document

    assert "total_cost_usd" not in document["final_metrics"] and "reasoning_effort" not in agent_steps(document)[0]


def test_from_atif_gives_back_events_the_adapter_reads_to_the_same_messages_and_counts():
    document = run(coding()).document
    native = HermesAtif.from_atif(document)
    again = run(_replayed(native.log)).document

    assert native.session is None
    assert [(s["message"], s.get("metrics", {}).get("prompt_tokens")) for s in agent_steps(again)] == [
        (s["message"], s.get("metrics", {}).get("prompt_tokens")) for s in agent_steps(document)]


def _replayed(log):
    feed = Feed()
    feed.records = [Fed(record) for record in log]
    return feed


def test_the_tested_hermes_is_the_one_the_image_pins():
    pinned = re.search(r"^ARG HERMES_COMMIT=(\S+)$", (AGENT_DIR / "Dockerfile").read_text(), re.M).group(1)
    assert hermes_atif.TESTED_WITH == (pinned,)


def seeded_conversation(seed: int) -> tuple[list[Feed], list[dict]]:
    """A random conversation of one to three turns as the plugin writes its events: responses with text, reasoning and
    calls, some wrapped in the tool search bridge with a connector batch, some returning images or failing; failed
    attempts before a call; auxiliary calls between and after; a delegate child, alone or a batch, its events among
    the turn's. With each turn's events, to count what it reports."""
    rng = random.Random(seed)
    n = iter(range(10**6))
    image = {"_multimodal": True, "content": [{"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(png()).decode()}}]}

    def usage():
        prompt = rng.randint(50, 20000)
        read = rng.randint(0, prompt // 2)
        return counts(prompt, rng.randint(1, 300), read, rng.randint(0, prompt - read))

    def response(feed, session=None, turn=None, final=False):
        for retry in range(rng.choice([0, 0, 0, 1, 2])):
            feed.error(retry=retry + 1, session=session, turn=turn)
        calls = [] if final else [(f"toolu_{next(n)}", rng.choice(["terminal", "read_file", "mcp__env__get"]), {})
                                  for _ in range(rng.choice([1, 1, 2]))]
        text = rng.choice([None, "Looking.", "  done  "]) if not final else "answer"
        feed.call(text, calls=calls, usage=usage(), reasoning=rng.choice([None, "plan"]), session=session, turn=turn)
        for call_id, name, _ in calls:
            feed.pre(call_id, name, {}, session=session, turn=turn)
            if rng.random() < 0.3:
                feed.pre(f"call_{next(n):012x}", "mcp__env__batch", {}, session=session, turn=turn)
                feed.post(feed.at()["tool_call_id"], "mcp__env__batch", "b", session=session, turn=turn)
            output = image if name == "mcp__env__get" and rng.random() < 0.5 else "out"
            if isinstance(output, dict):
                feed.event("post_tool_call", session=session, turn=turn, api_request_id="", tool_call_id=call_id,
                           tool_name=name, result=output, status="ok", duration_ms=3)
            else:
                feed.post(call_id, name, output, status=rng.choice(["ok", "ok", "error"]), session=session, turn=turn)
        if rng.random() < 0.25:
            feed.auxiliary(rng.choice(["vision", "compression"]), usage(), session=session, turn=turn)

    def delegate(feed):
        children = [f"child-{next(n)}" for _ in range(rng.choice([1, 1, 2]))]
        call_id = f"toolu_{next(n)}"
        feed.call(calls=[(call_id, "delegate_task", {"goal": "go"})], usage=usage())
        feed.pre(call_id, "delegate_task", {"goal": "go"})
        announce = rng.random() < 0.7 or len(children) > 1
        for child in children:
            if announce:
                feed.spawn(child, f"sa-{next(n)}-x", "go")
        for child in children:
            sub = Feed.new_turn(child)
            for _ in range(rng.randint(0, 2)):
                response(feed, child, sub)
            response(feed, child, sub, final=True)
            feed.answer("answer", session=child, turn=sub)
            if announce:
                feed.stop(child, "answer")
        feed.post(call_id, "delegate_task", "{}")

    feeds = []
    for task in range(rng.randint(1, 3)):
        feed = Feed()
        if task == 0:
            feed.start()
        if rng.random() < 0.5:
            feed.auxiliary(usage=usage())
        for _ in range(rng.randint(0, 3)):
            delegate(feed) if rng.random() < 0.2 else response(feed)
        response(feed, final=True)
        if rng.random() < 0.5:
            feed.auxiliary(usage=usage())
        feeds.append(feed.answer("answer").end())
    return feeds


@pytest.mark.parametrize("seed", range(80))
def test_a_seeded_conversations_copy_holds_every_call_where_it_happened_and_sums_to_what_the_turns_report(seed):
    feeds = seeded_conversation(seed)
    kit = run(*feeds)

    assert_sums_to(kit.document, *(reported(feed) for feed in feeds))
    every_call_answered(kit.document)
    users = [index for index, step in enumerate(kit.document["steps"]) if step["source"] == "user"]
    for end in [*users[1:], len(kit.document["steps"])]:
        last = kit.document["steps"][end - 1]
        assert last["message"] == "answer" and last["llm_call_count"] == 1
