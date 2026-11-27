"""One turn of the Antigravity SDK (antigravity_bridge): the config the SDK's ``Agent`` is started with, its model calls
through the turn's own relay with the endpoint's key and the run's attribution, a task's selection of the SDK's tools,
the trajectory the SDK's streamed steps and hooks make, a tool result tied to its streamed call, and how a turn ends."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.task_steps.deploy_agent import _check_builtin_tools
from tst.unit.agents.antigravity import fake_sdk
from tst.unit.agents.antigravity.agent import AGENT_DIR, a2a_server, antigravity_bridge, antigravity_date
from tst.unit.agents.antigravity.fake_sdk import StopReason, Text, Thought, ToolCall, ToolResult, Turn

TOOLS = fake_sdk.BuiltinTools
ALWAYS_OFF = [TOOLS.ASK_QUESTION, TOOLS.GENERATE_IMAGE]


class ToolError(Exception):
    """What the SDK gives its ``on_tool_error`` hook."""

    def __init__(self, message: str, call_id: str, tool_name: str, server_name: str | None = None) -> None:
        super().__init__(message)
        self.call_id, self.tool_name, self.server_name = call_id, tool_name, server_name


def _run(turn: Turn | None = None, **kwargs: Any) -> tuple[antigravity_bridge.AntigravityResult, list[dict]]:
    """One turn through the bridge, its trajectory as appended."""
    if turn is not None:
        fake_sdk.Agent.turns.append(turn)
    events: list[dict] = []
    arguments = dict(model="m", session_id=None, mcp_servers=None, system_prompt=None, timeout_s=30, role=None,
                     builtin_tools=None, web_tools=True, max_turns=None, workspace="/w", event_callback=events.append)
    result = asyncio.run(antigravity_bridge.run_antigravity("go", **{**arguments, **kwargs}))
    return result, events


def _config() -> dict:
    (config,) = fake_sdk.Agent.configs
    return dict(config.__dict__)


def test_the_sdk_is_started_with_the_relay_the_workspace_and_every_call_allowed(monkeypatch):
    monkeypatch.delenv("AGENTENV_EGRESS_PROXY", raising=False)
    _run(Turn())

    config = _config()
    hooks = config.pop("hooks")
    assert config == {
        "base_url": "http://relay.test/token/v1", "model": "m", "save_dir": str(antigravity_bridge.SAVE_DIR),
        "workspaces": ["/w"],
        "capabilities": fake_sdk.CapabilitiesConfig(agent_behavior=fake_sdk.AgentBehavior.AUTONOMOUS,
                                                    disabled_tools=ALWAYS_OFF),
        "policies": ["allow_all"],
        "retry_config": fake_sdk.RetryConfig(api_retry=fake_sdk.ModelAPIRetryConfig(
            max_retries=6, initial_sleep_duration_ms=5000, exponential_multiplier=2.0)),
        "env": None, "mcp_servers": [], "skills_paths": [],
    }
    assert [hook.kind for hook in hooks] == ["pre", "post", "error"]
    assert fake_sdk.Agent.prompts == ["go"]


def test_the_task_settings_reach_the_sdk():
    antigravity_bridge.SKILLS_DIR.mkdir(parents=True)
    servers = {"env4821": {"url": "https://gw/mcp", "headers": {"agentenv-role": "seller", "x-k": "v"}},
               "peer_agents": {"url": "http://127.0.0.1:8000/ext/peer-agents/mcp"}}

    _run(Turn(), mcp_servers=servers, role="buyer", system_prompt="be brief", max_turns=5)

    config = _config()
    assert config["mcp_servers"] == [
        fake_sdk.McpStreamableHttpServer(name="env4821", url="https://gw/mcp",
                                         headers={"x-k": "v", "AgentEnv-Role": "buyer"}, timeout=60.0,
                                         sse_read_timeout=300.0),
        fake_sdk.McpStreamableHttpServer(name="peer_agents", url="http://127.0.0.1:8000/ext/peer-agents/mcp",
                                         headers={"AgentEnv-Role": "buyer"}, timeout=60.0, sse_read_timeout=300.0)]
    assert config["skills_paths"] == [str(antigravity_bridge.SKILLS_DIR)]
    assert config["system_instructions"] == "be brief"
    assert config["budget_config"] == fake_sdk.BudgetConfig(max_model_calls=5,
                                                            scope=fake_sdk.BudgetScope.FORWARD_LOOKING)


def test_a_server_keeps_its_own_role_when_the_task_has_none():
    _run(Turn(), mcp_servers={"env": {"url": "https://gw/mcp", "headers": {"AgentEnv-Role": "seller"}}})

    (server,) = _config()["mcp_servers"]
    assert server.headers == {"AgentEnv-Role": "seller"}


def test_a_conversation_is_resumed_by_its_id_when_its_store_is_there():
    antigravity_bridge.SAVE_DIR.mkdir(parents=True)
    (antigravity_bridge.SAVE_DIR / "conv-1.db").write_bytes(b"")

    result, _ = _run(Turn(conversation_id="conv-1"), session_id="conv-1")

    assert _config()["conversation_id"] == "conv-1"
    assert result.session_id == "conv-1"


def test_a_conversation_whose_store_is_gone_starts_afresh():
    result, _ = _run(Turn(conversation_id="conv-new"), session_id="conv-gone")

    assert "conversation_id" not in _config()
    assert result.session_id == "conv-new"


def test_on_an_armed_clock_the_harness_goes_out_through_the_egress_proxy(monkeypatch):
    monkeypatch.setenv("AGENTENV_EGRESS_PROXY", "http://127.0.0.1:3128")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gw.example/v1")
    monkeypatch.setattr(antigravity_bridge, "clock_armed", lambda: True)

    _run(Turn(), mcp_servers={"env": {"url": "http://env.internal:9000/mcp"}})

    env = _config()["env"]
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:3128"
    assert env["NO_PROXY"].split(",")[-1] == "env.internal"


def test_model_calls_carry_the_endpoints_key_its_headers_and_the_runs_attribution(monkeypatch, relay):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gw.example/v1/")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    monkeypatch.setenv("LITELLM_EXTRA_HEADERS", json.dumps({"X-LiteLLM-Tags": "team:envs", "X-Extra": "1"}))

    _run(Turn(), project_id="p-1", task_id="t-1")

    (opened,) = relay
    assert opened["upstream"] == "https://gw.example"
    assert opened["headers"] == {"authorization": "Bearer sk-test", "x-extra": "1", "x-litellm-customer-id": "p-1",
                                 "x-litellm-tags": "team:envs,projectId:p-1,taskId:t-1"}
    assert antigravity_bridge.get_last_attribution_snapshot()["project_id"] == "p-1"


def test_the_turns_usage_is_the_relays_count_of_its_calls(monkeypatch, relay):
    async def answered(_prompt, **_kwargs):
        relay[0]["calls"].add(200, antigravity_bridge.Usage(input_tokens=10, output_tokens=2), "m")
        relay[0]["calls"].add(200, None, "m")
        relay[0]["calls"].add(502, None, "m")
        return antigravity_bridge.AntigravityResult(response="ok", tool_call_count=3)

    monkeypatch.setattr(antigravity_bridge, "_turn", answered)
    result, _ = _run()

    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.tool_call_count) == (10, 2, 3)
    assert result.usage.provider_details == {"model_calls": 2, "unreported_calls": 1}


@pytest.mark.parametrize("selection, expected", [
    (None, {"disabled_tools": ALWAYS_OFF}),
    ({"only": ["view_file", "run_command"]}, {"enabled_tools": [TOOLS.VIEW_FILE, TOOLS.RUN_COMMAND, TOOLS.FINISH]}),
    ({"only": []}, {"enabled_tools": [TOOLS.FINISH]}),
    ({"disable": ["search_web", "read_url_content"]},
     {"disabled_tools": [*ALWAYS_OFF, TOOLS.SEARCH_WEB, TOOLS.READ_URL_CONTENT]}),
    ({"disable": "*"}, {"disabled_tools": [*ALWAYS_OFF, TOOLS.LIST_DIR, TOOLS.SEARCH_DIR, TOOLS.FIND_FILE,
                                           TOOLS.VIEW_FILE, TOOLS.CREATE_FILE, TOOLS.EDIT_FILE, TOOLS.RUN_COMMAND,
                                           TOOLS.INVOKE_SUBAGENT, TOOLS.RUN_WORKFLOW, TOOLS.SEARCH_WEB,
                                           TOOLS.READ_URL_CONTENT, TOOLS.SCHEDULE]}),
])
def test_a_selection_of_the_sdks_tools_is_its_capabilities(selection, expected):
    tools = a2a_server.AntigravityConfig(builtin_tools=selection).builtin_tools

    _run(Turn(), builtin_tools=tools)

    assert _config()["capabilities"] == fake_sdk.CapabilitiesConfig(
        agent_behavior=fake_sdk.AgentBehavior.AUTONOMOUS, **expected)


@pytest.mark.parametrize("selection, expected", [
    (None, {"disabled_tools": [*ALWAYS_OFF, TOOLS.SEARCH_WEB, TOOLS.READ_URL_CONTENT]}),
    ({"only": ["view_file", "search_web"]}, {"enabled_tools": [TOOLS.VIEW_FILE, TOOLS.FINISH]}),
    ({"disable": ["run_command", "search_web"]},
     {"disabled_tools": [*ALWAYS_OFF, TOOLS.RUN_COMMAND, TOOLS.SEARCH_WEB, TOOLS.READ_URL_CONTENT]}),
], ids=["no-selection", "only-with-search-web", "disable-list"])
def test_web_tools_off_turns_the_sdks_web_tools_off_whatever_the_selection_keeps(selection, expected):
    config = a2a_server.AntigravityConfig(builtin_tools=selection, web_tools=False)

    _run(Turn(), builtin_tools=config.builtin_tools, web_tools=config.web_tools)

    assert _config()["capabilities"] == fake_sdk.CapabilitiesConfig(
        agent_behavior=fake_sdk.AgentBehavior.AUTONOMOUS, **expected)


def test_the_card_takes_web_tools_so_deploy_agent_can_turn_them_off(ag):
    extension = A2AAgent.find_extension(ag.card, A2AAgent.EXT_AGENT_CONFIG)

    assert "web_tools" in extension["params"]["methods"]["set"]["request"]["supported"]


@pytest.mark.parametrize("selection, match", [
    ({"only": ["ask_question"]}, "ask_question"),
    ({"disable": ["finish"]}, "finish"),
    ({"only": ["generate_image"]}, "generate_image"),
    ({"disable": "*", "only": ["view_file"]}, "exactly one"),
])
def test_a_selection_of_a_tool_the_task_cant_choose_is_refused(selection, match):
    with pytest.raises(ValueError, match=match):
        a2a_server.AntigravityConfig(builtin_tools=selection)


def test_the_card_takes_builtin_tools_so_deploy_agent_accepts_a_selection(ag):
    extension = A2AAgent.find_extension(ag.card, A2AAgent.EXT_AGENT_CONFIG)

    assert "builtin_tools" in extension["params"]["methods"]["set"]["request"]["supported"]
    _check_builtin_tools(extension, "antigravity", {"disable": ["search_web", "read_url_content"]})
    _check_builtin_tools(extension, "antigravity", {"only": ["view_file", "run_command"]})
    with pytest.raises(RuntimeError, match="can't take builtin_tools"):
        _check_builtin_tools(extension, "antigravity", {"disable": ["bash"]})


def test_the_selectable_tools_are_the_sdks_but_the_ones_a_headless_run_cant_use():
    assert set(antigravity_bridge.AntigravityTool.__args__) == (
        {tool.value for tool in TOOLS} - {"ask_question", "generate_image", "finish"})


def test_the_tool_names_are_those_of_the_sdk_version_requirements_pin():
    pinned = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
              if line.startswith("google-antigravity==")]

    assert pinned == [f"google-antigravity=={antigravity_bridge.SDK_VERSION}"], (
        "the SDK version changed: record the tools the new SDK offers in AntigravityTool, then SDK_VERSION; "
        "tst/integration/agent/antigravity_env_date_test.py checks the new SDK's tool names and local-time line")


def test_the_answer_is_the_last_text_after_a_tool_call_and_the_steps_are_the_trajectory():
    result, events = _run(Turn(steps=[
        ("chunk", Thought(step_index=0, text="thinking")),
        ("chunk", Text(step_index=0, text="Let me ")),
        ("chunk", Text(step_index=0, text="check.")),
        ("call", ToolCall(id="h1", name="run_command")),
        ("chunk", ToolCall(id="s1", name="run_command", step_index=0)),
        ("result", ToolResult(id="h1", name="run_command", result="42")),
        ("chunk", Text(step_index=1, text="42")),
    ], conversation_id="conv-7"))

    assert (result.response, result.session_id, result.tool_call_count, result.error) == ("42", "conv-7", 1, None)
    assert events == [
        {"type": "Thought", "step_index": 0, "text": "thinking"},
        {"type": "Text", "step_index": 0, "text": "Let me check."},
        {"type": "ToolCall", "id": "s1", "name": "run_command", "step_index": 0},
        {"kind": "tool_result", "id": "h1", "name": "run_command", "result": "42", "call_ref": "s1"},
        {"type": "Text", "step_index": 1, "text": "42"},
    ]


def test_a_turn_that_ends_on_a_tool_call_answers_with_its_last_text():
    result, _ = _run(Turn(steps=[
        ("chunk", Text(step_index=0, text="Done: 42")),
        ("call", ToolCall(id="h1", name="finish")),
        ("chunk", ToolCall(id="s1", name="finish", step_index=0)),
    ]))

    assert result.response == "Done: 42"


def test_the_answer_is_a_paragraph_per_step_after_the_last_tool_call():
    result, _ = _run(Turn(steps=[
        ("call", ToolCall(id="h1", name="run_command")),
        ("chunk", ToolCall(id="s1", name="run_command", step_index=0)),
        ("chunk", Text(step_index=1, text="It runs in the ")),
        ("chunk", Text(step_index=1, text="background.")),
        ("chunk", Text(step_index=3, text="2031-04-01")),
    ]))

    assert result.response == "It runs in the background.\n\n2031-04-01"


def test_a_result_that_comes_before_its_streamed_call_follows_it():
    _, events = _run(Turn(steps=[
        ("call", ToolCall(id="h1", name="view_file")),
        ("call", ToolCall(id="h2", name="view_file")),
        ("result", ToolResult(id="h2", name="view_file", result="b")),
        ("chunk", ToolCall(id="s1", name="view_file", step_index=0)),
        ("chunk", ToolCall(id="s2", name="view_file", step_index=0)),
        ("result", ToolResult(id="h1", name="view_file", result="a")),
    ]))

    assert [(event.get("type") or event["kind"], event.get("call_ref")) for event in events] == [
        ("ToolCall", None), ("ToolCall", None), ("tool_result", "s2"), ("tool_result", "s1")]


def test_a_result_of_the_harnesss_own_tool_is_tied_to_its_undecided_call():
    """The SDK's skill lookup streams and gives its result to the hooks, but never asks them to decide it."""
    _, events = _run(Turn(steps=[
        ("call", ToolCall(id="m1", name="run_command")),
        ("chunk", ToolCall(id="c:1", name="run_command", step_index=0)),
        ("result", ToolResult(id="m1", name="run_command", result="42")),
        ("chunk", ToolCall(id="c:2", name="skill_lookup", step_index=1)),
        ("result", ToolResult(id="m2", name="list_skills", result="[]")),
        ("call", ToolCall(id="m3", name="view_file")),
        ("chunk", ToolCall(id="c:3", name="view_file", step_index=2)),
        ("result", ToolResult(id="m3", name="view_file", result="text")),
    ]))

    assert [(event["id"], event["call_ref"]) for event in events if event.get("kind") == "tool_result"] == [
        ("m1", "c:1"), ("m2", "c:2"), ("m3", "c:3")]


def test_a_call_binds_to_the_next_streamed_call_of_its_name():
    _, events = _run(Turn(steps=[
        ("call", ToolCall(id="m1", name="view_file")),
        ("chunk", ToolCall(id="c:1", name="skill_lookup", step_index=0)),
        ("chunk", ToolCall(id="c:2", name="view_file", step_index=0)),
        ("result", ToolResult(id="m1", name="view_file", result="text")),
        ("result", ToolResult(id="m2", name="list_skills", result="[]")),
    ]))

    assert [(event["id"], event["call_ref"]) for event in events if event.get("kind") == "tool_result"] == [
        ("m1", "c:2"), ("m2", "c:1")]


def test_a_failed_tool_call_is_its_result_with_the_error():
    _, events = _run(Turn(steps=[
        ("call", ToolCall(id="h1", name="run_command")),
        ("chunk", ToolCall(id="s1", name="run_command", step_index=0)),
        ("error", ToolError("exit status 1", call_id="h1", tool_name="run_command")),
    ]))

    assert events[-1] == {"kind": "tool_result", "id": "h1", "name": "run_command", "server_name": None,
                          "result": None, "error": "exit status 1", "call_ref": "s1"}


def test_a_result_whose_call_never_streamed_is_kept_unlinked():
    _, events = _run(Turn(steps=[
        ("call", ToolCall(id="h1", name="run_command")),
        ("result", ToolResult(id="h1", name="run_command", result="x")),
        ("result", ToolResult(id="h9", name="peer_agents_send", result="y")),
    ]))

    assert [(event["id"], event["call_ref"]) for event in events] == [("h9", None), ("h1", None)]


def test_a_turn_past_its_time_fails_with_what_it_streamed_kept_and_its_conversation_for_the_next():
    antigravity_bridge.SAVE_DIR.mkdir(parents=True)
    (antigravity_bridge.SAVE_DIR / "conv-new.db").write_bytes(b"")

    result, events = _run(Turn(steps=[("chunk", Text(step_index=0, text="working")), ("sleep", 5)],
                               conversation_id="conv-new"), timeout_s=0.2)

    assert (result.error_code, result.error) == ("timeout", "the turn ran past its 0.2 s")
    assert events == [{"type": "Text", "step_index": 0, "text": "working"}]
    assert result.session_id == "conv-new"


@pytest.mark.parametrize("session_id", [None, "conv-1"])
def test_a_turn_that_fails_before_its_first_step_keeps_the_conversation_it_resumed(session_id):
    antigravity_bridge.SAVE_DIR.mkdir(parents=True)
    (antigravity_bridge.SAVE_DIR / "conv-1.db").write_bytes(b"")

    result, _ = _run(Turn(steps=[("raise", fake_sdk.AntigravityExecutionError("harness exited 2"))],
                          conversation_id="conv-2"), session_id=session_id)

    assert result.session_id == session_id


@pytest.mark.parametrize("message, code, infra", [
    ("MCP load failed: env4821: 502 Bad Gateway", "mcp_load_failed", True),
    ("localharness exited with status 2", "antigravity_error", False),
])
def test_a_harness_that_fails_fails_the_turn(message, code, infra):
    result, _ = _run(Turn(steps=[("raise", fake_sdk.AntigravityExecutionError(message))]))

    assert (result.error_code, result.error, result.infra) == (code, message, infra)


@pytest.mark.parametrize("stop, code, error", [
    (StopReason.MAX_MODEL_CALLS_EXCEEDED, "max_turns_exceeded", "the turn stopped at its max_turns, 3"),
    (StopReason.QUOTA_EXHAUSTED, "stopped", "the SDK stopped the turn: QUOTA_EXHAUSTED"),
])
def test_a_turn_the_sdk_stops_fails_with_its_answer_so_far(stop, code, error):
    result, _ = _run(Turn(steps=[("chunk", Text(step_index=0, text="partial"))], stop_reason=stop), max_turns=3)

    assert (result.response, result.error_code, result.error) == ("partial", code, error)


@pytest.mark.parametrize("effort, sent", [("low", "low"), ("high", "high"), ("xhigh", "high"), ("max", "high"),
                                          (None, None)])
def test_the_effort_goes_as_reasoning_effort(effort, sent):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()

    rewritten = json.loads(asyncio.run(antigravity_bridge.turn_rewrite(effort, [])(body)))

    assert rewritten.get("reasoning_effort") == sent


def _rewritten(rewrite, messages: list[dict]) -> list[dict]:
    return json.loads(asyncio.run(rewrite(json.dumps({"model": "m", "messages": messages}).encode())))["messages"]


def test_an_image_goes_with_its_turns_message_once_and_as_it_was_given_in_the_turns_after(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG")
    part = antigravity_bridge.media_part(str(image))
    first = antigravity_bridge.turn_rewrite(None, [str(image)])
    asked = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"},
             {"role": "user", "content": "look"}]

    once = _rewritten(first, asked)
    twice = _rewritten(first, once)
    image.write_bytes(b"GIF89a, edited by the agent")
    later = _rewritten(antigravity_bridge.turn_rewrite(None, []), [
        *asked, {"role": "assistant", "content": "a chart"}, {"role": "user", "content": "its other label?"}])

    assert part == {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}}
    assert once[0]["content"] == "earlier"
    assert once[2]["content"] == [{"type": "text", "text": "look"}, part]
    assert twice == once
    assert later[2]["content"] == [{"type": "text", "text": "look"}, part]
    assert later[4]["content"] == "its other label?"


def test_a_turns_records_are_filed_under_its_conversation(monkeypatch, tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG")

    @asynccontextmanager
    async def relay(upstream, calls, **kwargs):
        await kwargs["rewrite"](json.dumps({"model": "m", "messages": [{"role": "user", "content": "look"}]}).encode())
        yield "http://relay.test/token"

    monkeypatch.setattr(antigravity_bridge, "model_relay", relay)
    antigravity_bridge.file_records("conv-1", {"earlier#1"}, set())

    _run(Turn(steps=[("chunk", Text(step_index=0, text="ok"))], conversation_id="conv-1"),
         attachment_paths=[str(image)])

    assert antigravity_bridge.CONVERSATION_RECORDS.get("conv-1") == {
        "env_times": ["earlier#1"], "turn_images": [antigravity_date.message_digest("look")]}


def test_a_cancelled_turn_files_its_records_under_the_conversation_it_reached(monkeypatch, tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG")
    antigravity_bridge.SAVE_DIR.mkdir(parents=True)
    (antigravity_bridge.SAVE_DIR / "conv-9.db").write_bytes(b"")

    @asynccontextmanager
    async def relay(upstream, calls, **kwargs):
        await kwargs["rewrite"](json.dumps({"model": "m", "messages": [{"role": "user", "content": "look"}]}).encode())
        yield "http://relay.test/token"

    monkeypatch.setattr(antigravity_bridge, "model_relay", relay)
    fake_sdk.Agent.turns.append(Turn(steps=[("chunk", Text(step_index=0, text="working")), ("sleep", 5)],
                                     conversation_id="conv-9"))

    async def cancelled() -> None:
        turn = asyncio.create_task(antigravity_bridge.run_antigravity(
            "look", model="m", session_id=None, mcp_servers=None, system_prompt=None, timeout_s=30, role=None,
            builtin_tools=None, web_tools=True, max_turns=None, workspace="/w", event_callback=lambda _: None,
            attachment_paths=[str(image)]))
        await asyncio.sleep(0.2)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

    asyncio.run(cancelled())

    assert antigravity_bridge.CONVERSATION_RECORDS.get("conv-9") == {
        "env_times": [], "turn_images": [antigravity_date.message_digest("look")]}


def test_an_image_gone_before_its_turn_sends_it_is_left_out(tmp_path):
    asked = [{"role": "user", "content": "look"}]

    once = _rewritten(antigravity_bridge.turn_rewrite(None, [str(tmp_path / "chart.png")]), asked)
    later = _rewritten(antigravity_bridge.turn_rewrite(None, []), asked)

    assert once == later == asked
    assert antigravity_bridge.media_part(str(tmp_path / "notes.txt")) is None


def test_an_attachment_is_named_in_the_prompt(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("n")

    _run(Turn(), attachment_paths=[str(notes)])

    assert fake_sdk.Agent.prompts == [f"go\n\nAttached files, also on disk for your tools:\n- {notes}"]
