"""Codex's ATIF adapter on rollout records shaped as the CLI writes them, fed as the bridge feeds them
(``rollout_aux``), beside the exec JSON events that are the native log. Each case runs through the protocol's adapter
kit, so the live rule holds after every record and the document validates as a producer's."""
from __future__ import annotations

import base64
import json
import struct
import zlib

import pytest
from agentenv_protocol.atif import Fed, RecordedTask, TurnContext, check_adapter, profile_of

from tst.unit.agents.codex.agent import AGENT_DIR, codex_atif, codex_bridge

CodexAtif = codex_atif.CodexAtif
MAIN, SUB = "01a2-main", "01a2-sub"
MODEL = "gpt-5.5"


def png(width=4, height=3) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class Feed:
    """One task's records in arrival order: rollout records through ``rollout_aux``, each a millisecond after the
    last, and exec JSON events as natives."""

    clock = 0

    def __init__(self):
        self.records: list[Fed] = []

    def raw(self, kind, payload, thread=MAIN, parent=None):
        Feed.clock += 1
        record = {"timestamp": f"2026-11-25T16:15:{Feed.clock // 1000:02d}.{Feed.clock % 1000:03d}Z", "type": kind,
                  "payload": payload}
        fed = codex_bridge.rollout_aux(record, thread, parent)
        if fed is not None:
            self.records.append(Fed(fed, aux="rollout"))
        return self

    def at(self):
        return self.records[-1].record["timestamp"]

    def native(self, event):
        self.records.append(Fed(event))
        return self

    def meta(self, thread=MAIN, parent=None, path=None, nickname=None):
        source = {"subagent": {"thread_spawn": {"parent_thread_id": parent, "depth": 1, "agent_path": path}}} \
            if parent else "exec"
        return self.raw("session_meta", {"id": thread, "parent_thread_id": parent, "agent_path": path,
                                         "agent_nickname": nickname, "source": source, "cli_version": "0.162.1",
                                         "base_instructions": {"text": "You are Codex."}}, thread, parent)

    def started(self, thread=MAIN, parent=None):
        return self.raw("event_msg", {"type": "task_started", "turn_id": "turn-1"}, thread, parent)

    def context(self, thread=MAIN, parent=None, model=MODEL, effort=None):
        payload = {"model": model, "cwd": "/app", "approval_policy": "never"}
        if effort:
            payload["effort"] = effort
        return self.raw("turn_context", payload, thread, parent)

    def user(self, text, thread=MAIN, parent=None):
        return self.raw("response_item", {"type": "message", "role": "user",
                                          "content": [{"type": "input_text", "text": text}]}, thread, parent)

    def reasoning(self, item_id, summary=(), encrypted="gAAAA-cipher", thread=MAIN, parent=None):
        payload = {"type": "reasoning", "id": item_id, "summary": [{"type": "summary_text", "text": s}
                                                                   for s in summary]}
        if encrypted:
            payload["encrypted_content"] = encrypted
        return self.raw("response_item", payload, thread, parent)

    def message(self, item_id, text, thread=MAIN, parent=None):
        return self.raw("response_item", {"type": "message", "id": item_id, "role": "assistant", "phase": "final",
                                          "content": [{"type": "output_text", "text": text}]}, thread, parent)

    def call(self, item_id, call_id, name="exec_command", arguments=None, namespace=None, thread=MAIN, parent=None):
        payload = {"type": "function_call", "id": item_id, "name": name, "call_id": call_id,
                   "arguments": json.dumps(arguments if arguments is not None else {"cmd": "ls"})}
        if namespace:
            payload["namespace"] = namespace
        return self.raw("response_item", payload, thread, parent)

    def output(self, call_id, output="ok", kind="function_call_output", thread=MAIN, parent=None):
        return self.raw("response_item", {"type": kind, "id": f"fco-{call_id}", "call_id": call_id, "output": output},
                        thread, parent)

    def usage(self, response_id, input_tokens=100, cached=40, output=7, thread=MAIN, parent=None):
        counts = {"input_tokens": input_tokens, "cached_input_tokens": cached, "cache_write_input_tokens": 0,
                  "output_tokens": output, "reasoning_output_tokens": 0, "total_tokens": input_tokens + output}
        return self.raw("token_usage_record", {"thread_id": thread, "response_id": response_id, "usage": counts,
                                               "turn_token_usage": counts, "thread_token_usage": counts},
                        thread, parent)

    def complete(self, error=None, thread=MAIN, parent=None):
        payload = {"type": "task_complete", "last_agent_message": None if error else "done"}
        if error:
            payload["error"] = {"message": error, "codex_error_info": "other"}
        return self.raw("event_msg", payload, thread, parent)

    def spawned(self, call_id, path, thread=MAIN):
        return self.output(call_id, json.dumps({"task_name": path}), thread=thread)


def context(index=0):
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=["do it"],
                       agent_name="Codex", agent_version="1.0.0", agent_model=MODEL,
                       started_at=f"2026-11-25T16:00:0{index}Z")


def run(*feeds: Feed):
    kit = check_adapter(CodexAtif, [RecordedTask(context=context(i), records=feed.records)
                                    for i, feed in enumerate(feeds)])
    kit.raise_for_errors()
    return kit


def agent_steps(trajectory):
    return [step for step in trajectory["steps"] if step["source"] == "agent"]


def subagents(document):
    return {sub["trajectory_id"]: sub for sub in document.get("subagent_trajectories") or ()}


def opening(feed=None):
    return (feed or Feed()).meta().started().context(effort="high").user("do it")


def test_a_response_is_one_step_with_its_items_its_first_items_time_and_its_own_usage():
    feed = opening().reasoning("rs_1")
    first = feed.at()
    feed.call("fc_1", "call_1", arguments={"cmd": "python3 hello.py"}).usage("resp_1").output("call_1", "5050\n")
    feed.message("msg_2", "5050").usage("resp_2", input_tokens=120, cached=100, output=3).complete()
    feed.native({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": "5050"}})
    steps = agent_steps(run(feed).document)

    assert len(steps) == 2
    assert steps[0]["timestamp"] == first and steps[0]["model_name"] == MODEL
    assert steps[0]["reasoning_effort"] == "high"
    assert steps[0]["tool_calls"] == [{"tool_call_id": "call_1", "function_name": "exec_command",
                                       "arguments": {"cmd": "python3 hello.py"}}]
    assert steps[0]["observation"]["results"][0]["content"] == "5050\n"
    assert steps[0]["metrics"] == {"prompt_tokens": 100, "cached_tokens": 40, "completion_tokens": 7,
                                   "extra": {"agentenv": {"cache_write_tokens": 0}}}
    assert steps[1]["message"] == "5050" and steps[1]["metrics"]["prompt_tokens"] == 120


@pytest.mark.parametrize(("summary", "encrypted", "expected"), [
    ((), "gAAAA-cipher", {"visibility": "redacted", "data": "gAAAA-cipher"}),
    (("Plan the sum.", "Then run it."), "gAAAA-cipher", {"visibility": "summary"}),
    ((), None, {"visibility": "omitted"}),
])
def test_reasoning_keeps_its_summary_or_its_ciphertext(summary, encrypted, expected):
    feed = opening().reasoning("rs_1", summary=summary, encrypted=encrypted).message("msg_1", "ok").usage("r1")
    step = agent_steps(run(feed.complete()).document)[0]

    (reasoning,) = profile_of(step)["reasoning"]
    assert {key: reasoning[key] for key in expected} == expected
    assert step.get("reasoning_content") == ("Plan the sum.\nThen run it." if summary else None)


def test_an_mcp_tools_name_keeps_its_namespace_and_a_patch_keeps_its_text():
    patch = "*** Begin Patch\n*** Update File: hello.py\n*** End Patch\n"
    feed = opening().raw("response_item", {"type": "tool_search_call", "id": "tsc_1", "call_id": "call_s",
                                           "arguments": {"query": "browser", "limit": 8}})
    feed.usage("r1").raw("response_item", {"type": "tool_search_output", "call_id": "call_s",
                                           "tools": [{"type": "namespace", "name": "mcp__browser"}]})
    feed.call("fc_2", "call_n", "browser_navigate", {"url": "https://example.com"}, namespace="mcp__browser")
    feed.usage("r2").output("call_n", [{"type": "input_text", "text": "navigated"}])
    feed.raw("response_item", {"type": "custom_tool_call", "id": "ctc_3", "call_id": "call_p", "name": "apply_patch",
                               "input": patch})
    feed.usage("r3").output("call_p", "Success.", kind="custom_tool_call_output").complete()
    steps = agent_steps(run(feed).document)

    calls = [step["tool_calls"][0] for step in steps]
    assert [call["function_name"] for call in calls] == ["tool_search", "mcp__browser__browser_navigate",
                                                         "apply_patch"]
    assert calls[0]["arguments"] == {"query": "browser", "limit": 8}
    assert calls[2]["arguments"] == {"input": patch}
    assert json.loads(steps[0]["observation"]["results"][0]["content"]) == [{"type": "namespace",
                                                                              "name": "mcp__browser"}]
    assert [step["observation"]["results"][0]["content"] for step in steps[1:]] == ["navigated", "Success."]


def test_an_image_a_tool_returns_is_a_media_file_named_by_its_bytes():
    image = png()
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    feed = opening().call("fc_1", "call_1", "browser_take_screenshot", {}, namespace="mcp__browser").usage("r1")
    feed.output("call_1", [{"type": "input_text", "text": "### Result"},
                           {"type": "input_image", "image_url": url, "detail": "high"}]).complete()
    kit = run(feed)

    (result,) = agent_steps(kit.document)[0]["observation"]["results"]
    text, picture = result["content"]
    assert text == {"type": "text", "text": "### Result"}
    assert picture["type"] == "image" and picture["source"]["media_type"] == "image/png"
    assert kit.media[picture["source"]["path"]] == image


def sub_agent(feed, call_id="call_spawn", meta_first=True):
    """The main thread spawning sub-agent SUB, which runs one command and answers; its spawn result comes before or
    after its rollout's first record."""
    feed.reasoning("rs_1").call("fc_1", call_id, "spawn_agent", {"task_name": "wc", "message": "count"},
                                namespace="collaboration").usage("r1")
    if meta_first:
        feed.meta(SUB, MAIN, "/root/wc", "Mendel").spawned(call_id, "/root/wc")
    else:
        feed.spawned(call_id, "/root/wc").meta(SUB, MAIN, "/root/wc", "Mendel")
    feed.started(SUB, MAIN).context(SUB, MAIN).call("fc_s1", "call_s1", thread=SUB, parent=MAIN)
    feed.usage("rs1", input_tokens=50, cached=0, output=4, thread=SUB, parent=MAIN)
    feed.output("call_s1", "19 /etc/passwd", thread=SUB, parent=MAIN)
    feed.message("msg_s2", "19", thread=SUB, parent=MAIN).usage("rs2", thread=SUB, parent=MAIN)
    feed.complete(thread=SUB, parent=MAIN)
    return feed.call("fc_2", "call_wait", "wait_agent", {"targets": ["/root/wc"]}, namespace="collaboration")


@pytest.mark.parametrize("meta_first", [True, False])
def test_a_sub_agent_is_embedded_under_the_call_that_spawned_it(meta_first):
    feed = sub_agent(opening(), meta_first=meta_first).usage("r2").output("call_wait", "Wait completed.")
    feed.message("msg_3", "19").usage("r3").complete()
    document = run(feed).document

    spawn = agent_steps(document)[0]["observation"]["results"][0]
    assert spawn["content"] == json.dumps({"task_name": "/root/wc"})
    assert spawn["subagent_trajectory_ref"] == [{"trajectory_id": SUB}]
    sub = subagents(document)[SUB]
    assert sub["agent"]["name"] == "Mendel"
    sub_steps = agent_steps(sub)
    assert [step["message"] for step in sub_steps] == ["", "19"]
    assert sub_steps[0]["tool_calls"][0]["tool_call_id"] == "call_s1"
    assert sub_steps[0]["metrics"]["prompt_tokens"] == 50 and sub_steps[0]["model_name"] == MODEL


def test_a_spawn_whose_sub_agent_never_shows_keeps_its_result():
    feed = opening().call("fc_1", "call_spawn", "spawn_agent", {"task_name": "wc"}, namespace="collaboration")
    feed.usage("r1").spawned("call_spawn", "/root/wc").message("msg_2", "gave up").usage("r2").complete()
    document = run(feed).document

    result = agent_steps(document)[0]["observation"]["results"][0]
    assert result["content"] == json.dumps({"task_name": "/root/wc"}) and "subagent_trajectory_ref" not in result
    assert not subagents(document)


def test_a_sub_agent_continued_in_a_later_task_is_left_to_the_native_log():
    first = sub_agent(opening()).usage("r2").output("call_wait", "done").message("msg_3", "19").usage("r3").complete()
    second = Feed().started().context().user("again?").call("fc_4", "call_send", "send_input",
                                                            {"target": "/root/wc"}, namespace="collaboration")
    second.usage("r4").output("call_send", "sent").started(SUB, MAIN).message("msg_s9", "still 19", thread=SUB,
                                                                              parent=MAIN)
    second.usage("rs9", thread=SUB, parent=MAIN).message("msg_5", "19").usage("r5").complete()
    kit = run(first, second)

    assert [step["message"] for step in agent_steps(subagents(kit.document)[SUB])] == ["", "19"]
    assert profile_of(kit.document)["unmatched"] == {"event_msg": 1, "response_item": 1, "token_usage_record": 1}
    assert [finding.rule for finding in kit.findings] == ["kit-unmatched"]


def test_a_refused_call_is_a_step_with_no_model_call_holding_its_error():
    feed = opening().complete(error='{"error": {"message": "refused", "code": 400}}')
    (step,) = agent_steps(run(feed).document)

    assert step["llm_call_count"] == 0 and "metrics" not in step
    assert profile_of(step)["error"] == {"kind": "infra_error", "message": '{"error": {"message": "refused", '
                                                                           '"code": 400}}'}


def test_a_resumed_attempt_follows_its_failed_one_and_its_steps_carry_its_number():
    feed = opening().call("fc_1", "call_1", arguments={"cmd": "echo first"}).usage("r1").output("call_1", "first")
    feed.complete(error="unexpected status 503").started().context().user("Your previous turn was interrupted")
    feed.call("fc_2", "call_2", arguments={"cmd": "echo second"}).usage("r2").output("call_2", "second")
    feed.message("msg_3", "first second").usage("r3").complete()
    steps = agent_steps(run(feed).document)

    assert [profile_of(step).get("attempt") for step in steps] == [None, None, 2, 2]
    assert profile_of(steps[1])["error"]["message"] == "unexpected status 503"
    assert steps[-1]["message"] == "first second"


def test_a_resumed_threads_next_task_starts_its_own_steps():
    first = opening().message("msg_1", "5050").usage("r1").complete()
    second = Feed().started().context().user("now to 1000").message("msg_2", "500500").usage("r2").complete()
    kit = run(first, second)

    assert [step["message"] for step in agent_steps(kit.document)] == ["5050", "500500"]
    assert [profile_of(step).get("attempt") for step in agent_steps(kit.document)] == [None, None]


def test_a_response_that_wrote_nothing_gets_a_step_for_its_usage():
    feed = opening().usage("resp_empty", input_tokens=90, cached=0, output=0).message("msg_1", "ok").usage("r2")
    steps = agent_steps(run(feed.complete()).document)

    assert steps[0]["metrics"]["prompt_tokens"] == 90 and not steps[0]["message"]
    assert steps[1]["message"] == "ok"


def test_the_tested_cli_is_the_one_the_image_pins():
    pinned = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["@openai/codex"]
    assert codex_atif.TESTED_WITH == (pinned,)
