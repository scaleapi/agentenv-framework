"""The adapter framework and the step builder: declarations checked at import, records routed to the most specific
handler, each task's user step and final response written from its TurnContext, results in their call's step, usage
in ATIF's terms, and a live view that only ever grows."""
import math

import pytest

from agentenv_protocol.atif import (
    Adapter,
    AdapterRun,
    AtifError,
    Fed,
    Native,
    RecordedTask,
    Step,
    TurnContext,
    atif_adapter,
    check,
    finish,
    is_prefix,
    native,
    profile_of,
    replay,
    timestamp,
)


def adapter(includes_cache=False):
    @atif_adapter(native_format="toy-events/1", version="0.1.0", input_includes_cache=includes_cache)
    class Toy(Adapter):
        @native(type="say")
        def say(self, r, turn):
            turn.message(step=r["step"], text=r["text"])

        @native(type="call")
        def call(self, r, turn):
            turn.tool_call(step=r["step"], tool_call_id=r["id"], function_name="Bash", arguments=r.get("args", {}))

        @native(type="result")
        def result(self, r, turn):
            turn.tool_result(source_call_id=r["id"], text=r.get("out", "done"), time=r.get("at"))

        @native(type="usage")
        def usage(self, r, turn):
            turn.step(key=r["step"], input_tokens=r.get("in"), cache_read_tokens=r.get("read"),
                      cache_write_tokens=r.get("write"), output_tokens=r.get("out"))
    return Toy


CTX = TurnContext(index=0, context_id="ctx", task_id="t1", message_id="m1", inbound=["hello"], agent_name="toy",
                  agent_version="1.0", started_at="2026-11-18T10:00:00Z")


def run_task(records, *, final=None, state="completed", cls=None):
    run = AdapterRun(cls or adapter())
    pages = [run.start_task(CTX)] + [run.native(r) for r in records]
    pages.append(run.end_task(final=final, state=state, ended_at="2026-11-18T10:05:00Z"))
    return run, pages


def test_the_user_step_comes_from_the_turn_context():
    run, _ = run_task([{"type": "say", "step": "s1", "text": "hi"}])
    user = run.document()["steps"][0]
    assert user["source"] == "user" and user["message"] == "hello" and user["timestamp"] == "2026-11-18T10:00:00Z"
    assert profile_of(user) == {"context_id": "ctx", "task_id": "t1", "message_id": "m1", "turn": 0}
    assert run.document()["agent"] == {"name": "toy", "version": "1.0"}


def test_a_final_response_the_last_step_holds_is_written_once():
    run, _ = run_task([{"type": "say", "step": "s1", "text": "all done"}], final=["all done"])
    steps = run.document()["steps"]
    assert len(steps) == 2 and profile_of(steps[-1])["state"] == "completed"


def test_a_final_response_the_last_step_lacks_gets_its_own_step_with_no_model_call():
    run, _ = run_task([{"type": "say", "step": "s1", "text": "working"}], final=["{\"ok\": true}"], state="failed")
    last = run.document()["steps"][-1]
    assert last["message"] == "{\"ok\": true}" and last["llm_call_count"] == 0 and "metrics" not in last
    assert profile_of(last)["state"] == "failed" and last["timestamp"] == "2026-11-18T10:05:00Z"
    check(run.document())


def test_the_terminal_state_is_never_served_before_the_task_ends():
    run, pages = run_task([{"type": "say", "step": "s1", "text": "a"}, {"type": "say", "step": "s2", "text": "b"}])
    for page in pages[:-1]:
        assert all("state" not in profile_of(s) for s in (page or {}).get("steps", []))
    assert profile_of(pages[-1]["steps"][-1])["state"] == "completed"
    with pytest.raises(AtifError):
        AdapterRun(adapter()).end_task()


def test_a_late_result_lands_in_its_calls_step_and_holds_back_the_steps_after_it():
    records = [{"type": "call", "step": "A", "id": "c1"}, {"type": "call", "step": "B", "id": "c2"},
               {"type": "result", "id": "c2"}, {"type": "say", "step": "C", "text": "waiting"},
               {"type": "result", "id": "c1", "out": "312 passed", "at": "2026-11-18T10:04:00.5Z"},
               {"type": "say", "step": "D", "text": "done"}]
    run, pages = run_task(records)
    served = [len(p["steps"]) if p else 0 for p in pages]
    assert served == [0, 1, 1, 1, 1, 3, 4, 5]
    step_a = run.document()["steps"][1]
    assert [r["source_call_id"] for r in step_a["observation"]["results"]] == ["c1"]
    assert profile_of(step_a["observation"]["results"][0]) == {"arrived": {"at": "2026-11-18T10:04:00.500Z",
                                                                           "step_id": 4}}
    assert all(is_prefix(p, run.document()) for p in pages)


def test_a_step_still_waiting_on_a_result_is_served_when_its_task_ends():
    run, pages = run_task([{"type": "call", "step": "A", "id": "c1"}, {"type": "say", "step": "B", "text": "x"}])
    assert [len(p["steps"]) if p else 0 for p in pages] == [0, 1, 1, 3]
    assert "observation" not in pages[-1]["steps"][1]
    check(pages[-1])


@pytest.mark.parametrize("includes,expected", [(False, 95), (True, 10)])
def test_usage_is_written_in_atifs_terms(includes, expected):
    run, _ = run_task([{"type": "usage", "step": "s1", "in": 10, "read": 80, "write": 5, "out": 7}],
                      cls=adapter(includes))
    metrics = run.document()["steps"][1]["metrics"]
    assert metrics["prompt_tokens"] == expected and metrics["cached_tokens"] == 80
    assert metrics["completion_tokens"] == 7 and metrics["extra"] == {"agentenv": {"cache_write_tokens": 5}}
    with pytest.raises(AtifError):
        run_task([{"type": "usage", "step": "s1", "in": 10}], cls=adapter(None))


@pytest.mark.parametrize("value,expected", [
    ("2026-11-18T01:02:03.12345Z", "2026-11-18T01:02:03.123450Z"),
    ("2026-11-18T01:02:03.123456789Z", "2026-11-18T01:02:03.123456Z"),
    ("2026-11-18T01:02:03.1+05:30", "2026-11-17T19:32:03.100Z"),
    ("2026-11-18T01:02:03.123Z", "2026-11-18T01:02:03.123Z"),
    ("2026-11-18 01:02:03-0700", "2026-11-18T08:02:03Z"),
])
def test_timestamps_are_normalized_to_what_atif_reads_everywhere(value, expected):
    assert timestamp(value) == expected
    Step.model_validate({"step_id": 1, "source": "user", "message": "", "timestamp": timestamp(value)})


@pytest.mark.parametrize("value", ["2026-11-18T01:02:03", "yesterday", "2026-11-18T25:00:00Z", 1700000000])
def test_a_time_without_a_zone_or_that_isnt_one_is_refused(value):
    with pytest.raises(AtifError):
        timestamp(value)


@pytest.mark.parametrize("arguments,typed", [
    ("*** Begin Patch", {"input": "*** Begin Patch"}), (None, {}), ([1, 2], {"value": [1, 2]}), (7, {"value": 7}),
])
def test_arguments_that_arent_an_object_are_wrapped_as_harbor_wraps_them(arguments, typed):
    run, _ = run_task([{"type": "call", "step": "s1", "id": "c1", "args": arguments}])
    call = run.document()["steps"][1]["tool_calls"][0]
    assert call["arguments"] == typed and call["extra"] == {"raw_arguments": arguments}
    run, _ = run_task([{"type": "call", "step": "s1", "id": "c1", "args": {"command": "ls"}}])
    assert "extra" not in run.document()["steps"][1]["tool_calls"][0]


def test_a_duplicate_call_id_and_a_result_for_no_call_are_refused():
    with pytest.raises(AtifError, match="already"):
        run_task([{"type": "call", "step": "s1", "id": "c1"}, {"type": "call", "step": "s2", "id": "c1"}])
    with pytest.raises(AtifError, match="no call"):
        run_task([{"type": "result", "id": "nope"}])


def test_values_that_arent_json_are_refused_as_they_are_written():
    with pytest.raises(AtifError, match="JSON"):
        run_task([{"type": "call", "step": "s1", "id": "c1", "args": {"x": math.nan}}])
    with pytest.raises(AtifError, match="JSON"):
        run_task([{"type": "call", "step": "s1", "id": "c1", "args": {"x": object()}}])


def test_reads_are_fresh_copies():
    run, _ = run_task([{"type": "say", "step": "s1", "text": "a"}])
    run.document()["steps"].clear()
    run.page()["steps"][0]["message"] = "rewritten"
    assert len(run.document()["steps"]) == 2 and run.page()["steps"][0]["message"] == "hello"


def test_reasoning_blocks_keep_their_boundaries_signatures_and_ciphertext():
    @atif_adapter(native_format="toy-think/1", version="1", input_includes_cache=None)
    class Thinker(Adapter):
        @native(type="think")
        def think(self, r, turn):
            turn.reasoning(step="s1", text=r.get("text"), visibility=r["vis"], signature=r.get("sig"),
                           data=r.get("data"))

    run = AdapterRun(Thinker)
    run.start_task(CTX)
    for r in [{"vis": "raw", "text": "first", "sig": "S1"}, {"vis": "omitted", "sig": "S2"},
              {"vis": "redacted", "data": "Q0lQSEVS"}, {"vis": "summary", "text": "second"}]:
        run.native({"type": "think", **r})
    run.end_task()
    step = run.document()["steps"][1]
    blocks = profile_of(step)["reasoning"]
    assert step["reasoning_content"] == "first\n\nsecond"
    assert [b["chars"] for b in blocks] == [5, 0, 0, 6] and blocks[2]["data"] == "Q0lQSEVS"
    text, pieces = step["reasoning_content"], []
    for block in blocks:
        pieces.append(text[:block["chars"]])
        text = text[block["chars"] + 2:] if block["chars"] else text
    assert pieces == ["first", "", "", "second"]


def test_media_go_to_the_sink_once_and_types_atif_lacks_become_notes():
    @atif_adapter(native_format="toy-media/1", version="1", input_includes_cache=None)
    class Shots(Adapter):
        @native(type="shot")
        def shot(self, r, turn):
            turn.tool_call(step=r["step"], tool_call_id=r["id"], function_name="computer", arguments={})
            turn.tool_result(source_call_id=r["id"], content=[turn.media(r["data"], r["mt"], width=2, height=1)])

    sunk = []
    run = AdapterRun(Shots, media=lambda path, data: sunk.append(path))
    run.start_task(CTX)
    for i, mt in enumerate(["image/png", "image/png", "video/mp4"]):
        run.native({"type": "shot", "step": f"s{i}", "id": f"c{i}", "data": b"same" if i < 2 else b"clip", "mt": mt})
    run.end_task()
    document = run.document()
    check(document)
    png, mp4 = (r["observation"]["results"][0] for r in document["steps"][1::2])
    path = png["content"][0]["source"]["path"]
    assert path.startswith("media/") and path.endswith(".png") and len(sunk) == 2
    assert profile_of(png)["media"][path] == {"sha256": path[6:-4], "bytes": 4, "width": 2, "height": 1}
    assert mp4["content"].startswith("[video: media/") and mp4["content"].endswith(", 4 B]")
    assert list(profile_of(mp4)["media"].values())[0]["media_type"] == "video/mp4"


def test_records_no_method_takes_are_counted_by_type():
    run, _ = run_task([{"type": "system"}, {"type": "system"}, {"kind": "no type"}])
    assert profile_of(run.document())["unmatched"] == {"dict": 1, "system": 2}


def test_a_sub_agent_is_an_embedded_trajectory_its_spawning_result_points_at():
    @atif_adapter(native_format="toy-sub/1", version="1", input_includes_cache=True)
    class Parent(Adapter):
        @native(type="go")
        def go(self, r, turn):
            turn.tool_call(step="p1", tool_call_id="task", function_name="Task", arguments={})
            turn.subagent("sub", spawned_by="task", name="explore", model_name="small")
            turn.message(step="x1", text="looking", agent="sub")
            turn.step(key="x1", input_tokens=5, output_tokens=1, agent="sub")
            turn.step(key="p1", input_tokens=10, output_tokens=2)
            turn.tool_result(source_call_id="task", text="found it")

    run = AdapterRun(Parent)
    run.start_task(CTX)
    run.native({"type": "go"})
    run.end_task()
    document = run.document()
    check(document)
    sub = document["subagent_trajectories"][0]
    assert sub["trajectory_id"] == "sub" and sub["agent"] == {"name": "explore", "version": "unknown",
                                                             "model_name": "small"}
    result = document["steps"][1]["observation"]["results"][0]
    assert result["subagent_trajectory_ref"] == [{"trajectory_id": "sub"}]
    assert document["final_metrics"]["total_prompt_tokens"] == 15 and document["final_metrics"]["total_steps"] == 2


def test_a_step_with_no_model_call_takes_no_usage():
    @atif_adapter(native_format="toy-dispatch/1", version="1", input_includes_cache=True)
    class Dispatch(Adapter):
        @native(type="d")
        def d(self, r, turn):
            turn.step(key="s1", llm_call_count=0, input_tokens=3)

    with pytest.raises(AtifError):
        run_task([{"type": "d"}], cls=Dispatch)


def test_finish_raising_still_closes_the_task():
    @atif_adapter(native_format="toy-fail/1", version="1", input_includes_cache=None)
    class Fails(Adapter):
        @finish
        def close(self, turn):
            raise RuntimeError("boom")

    run = AdapterRun(Fails)
    run.start_task(CTX)
    with pytest.raises(RuntimeError):
        run.end_task(state="failed")
    assert profile_of(run.document()["steps"][-1])["state"] == "failed"
    run.start_task(TurnContext(index=1, context_id="ctx", task_id="t2", inbound=["again"]))


def test_replaying_a_contexts_stored_tasks_builds_what_it_built_live():
    records = [{"type": "call", "step": "s1", "id": "c1"}, {"type": "usage", "step": "s1", "in": 1, "out": 1},
               {"type": "result", "id": "c1"}, {"type": "say", "step": "s2", "text": "ok"}]
    live, _ = run_task(records, final=["ok"])
    stored = [RecordedTask(context=CTX, records=[Fed(r) for r in records], final=["ok"],
                           ended_at="2026-11-18T10:05:00Z")]
    assert replay(adapter(), stored) == live.document()


def test_declarations_are_checked_at_import():
    with pytest.raises(AtifError):
        atif_adapter(native_format="atif/1", version="1", input_includes_cache=True)
    with pytest.raises(AtifError):
        atif_adapter(native_format="no-version", version="1", input_includes_cache=True)
    with pytest.raises(AtifError):
        atif_adapter(native_format="toy/1", version="", input_includes_cache=True)
    with pytest.raises(AtifError):
        atif_adapter(native_format="toy/1", version="1", input_includes_cache="yes")
    with pytest.raises(AtifError):
        @atif_adapter(native_format="toy/1", version="1", input_includes_cache=True)
        class TwoFinishes(Adapter):
            @finish
            def a(self, turn):
                pass

            @finish
            def b(self, turn):
                pass
    with pytest.raises(AtifError):
        AdapterRun(Adapter)


def test_the_most_specific_handler_takes_a_record_and_a_tie_is_refused():
    @atif_adapter(native_format="toy-pick/1", version="1", input_includes_cache=None)
    class Pick(Adapter):
        @native(type="e")
        def general(self, r, turn):
            turn.message(step="g", text="general")

        @native(type="e", where={"kind": "retry"})
        def retry(self, r, turn):
            turn.message(step="r", text="retry")

    run, _ = run_task([{"type": "e", "kind": "retry"}, {"type": "e", "kind": "other"}], cls=Pick)
    assert [s["message"] for s in run.document()["steps"][1:]] == ["retry", "general"]

    @atif_adapter(native_format="toy-tie/1", version="1", input_includes_cache=None)
    class Tie(Adapter):
        @native(where={"a": 1})
        def one(self, r, turn):
            pass

        @native(where={"b": 1})
        def two(self, r, turn):
            pass

    with pytest.raises(AtifError, match="equally"):
        run_task([{"a": 1, "b": 1}], cls=Tie)


def test_from_atif_is_each_adapters_own():
    with pytest.raises(NotImplementedError):
        adapter().from_atif({})
    assert Native(log=[{"type": "x"}]).session is None
