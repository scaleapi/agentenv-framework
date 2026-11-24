"""How the chat-completions bridge threads gemini-cli's tool calls: each functionResponse becomes a tool message
answering a call of the assistant message before it, by id where gemini-cli gives one, and a binary result keeps
its bytes. LiteLLM's own Gemini conversion checks every request as the gateway would."""

from __future__ import annotations

import json

from litellm.llms.vertex_ai.gemini.transformation import _gemini_convert_messages_with_history

from tst.unit.agents.gemini.agent import gemini_bridge

MODEL = "gemini/gemini-3.1-pro-preview"
PDF = {"inlineData": {"mimeType": "application/pdf", "data": "JVBERi0xLjQ="}}


def _user(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


def _call(name: str, call_id: str | None = None, **extra) -> dict:
    fc = {"name": name, "args": {"path": "a.pdf"}}
    if call_id:
        fc["id"] = call_id
    return {"functionCall": fc, **extra}


def _response(name: str, call_id: str | None = None, output: str = "ok", parts: list | None = None) -> dict:
    fr = {"name": name, "response": {"output": output}}
    if call_id:
        fr["id"] = call_id
    if parts:
        fr["parts"] = parts
    return {"functionResponse": fr}


def _translate(*contents: dict) -> list[dict]:
    messages = gemini_bridge.gemini_request_to_openai({"contents": list(contents)}, MODEL)["messages"]
    _gemini_convert_messages_with_history(messages=messages, model=MODEL)
    return messages


def _calls_and_answers(messages: list[dict]) -> list[tuple[list[str], list[str]]]:
    """Each assistant message's call ids, with the ids the tool messages right after it answer."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            answers = []
            for later in messages[i + 1:]:
                if later["role"] != "tool":
                    break
                answers.append(later["tool_call_id"])
            out.append(([c["id"] for c in m["tool_calls"]], answers))
    return out


def test_ids_pair_parallel_calls_of_one_tool_whatever_the_order_of_their_responses():
    messages = _translate(
        _user("read both"),
        {"role": "model", "parts": [_call("read_file", "c1"), _call("read_file", "c2")]},
        {"role": "user", "parts": [_response("read_file", "c2", "second"), _response("read_file", "c1", "first")]},
    )

    tools = [m for m in messages if m["role"] == "tool"]
    assert [(m["tool_call_id"], json.loads(m["content"])["output"]) for m in tools] == [
        ("c2", "second"), ("c1", "first")]
    assert _calls_and_answers(messages) == [(["c1", "c2"], ["c2", "c1"])]


def test_a_call_without_an_id_pairs_by_name_with_a_response_that_has_one():
    """gemini-cli makes up an id for a call that came without one and echoes it only on the response."""
    messages = _translate(
        _user("read it"),
        {"role": "model", "parts": [_call("read_file")]},
        {"role": "user", "parts": [_response("read_file", "read_file_1700000000000_0")]},
    )

    assert _calls_and_answers(messages) == [(["call_0"], ["call_0"])]


def test_a_history_without_ids_translates_as_before():
    greq = {"contents": [
        _user("go"),
        {"role": "model", "parts": [_call("ls"), _call("read_file", thoughtSignature="sig")]},
        {"role": "user", "parts": [_response("read_file", output="r"), _response("ls", output="l")]},
        {"role": "model", "parts": [{"text": "done"}]},
        _user("thanks"),
    ]}

    assert gemini_bridge.gemini_request_to_openai(greq, MODEL)["messages"] == [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "ls", "arguments": '{"path": "a.pdf"}'}},
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.pdf"}'},
             "provider_specific_fields": {"thought_signature": "sig"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"output": "r"}'},
        {"role": "tool", "tool_call_id": "call_0", "content": '{"output": "l"}'},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
    ]


def test_a_binary_read_keeps_its_bytes_in_the_tool_message():
    messages = _translate(
        _user("summarize a.pdf"),
        {"role": "model", "parts": [_call("read_file", "c1")]},
        {"role": "user", "parts": [
            _response("read_file", "c1", "Binary content provided (1 item(s)).", parts=[PDF])]},
    )

    tool = next(m for m in messages if m["role"] == "tool")
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "text", "text": '{"output": "Binary content provided (1 item(s))."}'},
        {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,JVBERi0xLjQ="}},
    ]}
    contents = _gemini_convert_messages_with_history(messages=messages, model=MODEL)
    function_response = next(
        p["function_response"] for c in contents for p in c["parts"] if "function_response" in p)
    assert function_response["parts"] == [
        {"inline_data": {"mime_type": "application/pdf", "data": "JVBERi0xLjQ="}}]


def test_a_response_whose_call_is_missing_reaches_the_model_as_user_content():
    """The call_read_file case: a resumed or trimmed history holds a read_file result but not its call."""
    messages = _translate(
        _user("summarize a.pdf"),
        {"role": "model", "parts": [{"text": "Reading it."}]},
        {"role": "user", "parts": [_response("read_file", "gone", "Binary content provided (1 item(s)).",
                                             parts=[PDF])]},
    )

    assert not any(m["role"] == "tool" for m in messages)
    assert messages[-1] == {"role": "user", "content": [
        {"type": "text", "text": "Result of a read_file tool call missing from this conversation:\n"
                                 '{"output": "Binary content provided (1 item(s))."}'},
        {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,JVBERi0xLjQ="}},
    ]}


def test_a_response_pairs_only_with_the_latest_model_turn():
    """An earlier turn's unanswered call is answered as having no result, not by a later response."""
    messages = _translate(
        _user("go"),
        {"role": "model", "parts": [_call("read_file")]},
        {"role": "model", "parts": [_call("read_file")]},
        {"role": "user", "parts": [_response("read_file")]},
    )

    assert _calls_and_answers(messages) == [(["call_0"], ["call_0"]), (["call_1"], ["call_1"])]
    assert json.loads(messages[2]["content"]) == {"error": "No result was recorded for this tool call."}
    assert json.loads(messages[4]["content"]) == {"output": "ok"}


def test_a_call_left_unanswered_by_a_resumed_history_gets_a_no_result_answer():
    messages = _translate(
        _user("go"),
        {"role": "model", "parts": [_call("read_file", "c1"), _call("ls", "c2")]},
        {"role": "user", "parts": [_response("read_file", "c1")]},
        {"role": "model", "parts": [{"text": "ls never answered"}]},
        _user("ok"),
    )

    assert _calls_and_answers(messages) == [(["c1", "c2"], ["c1", "c2"])]


def test_a_response_id_names_the_call_it_answers_so_the_next_request_pairs_by_it():
    gresp = gemini_bridge.openai_response_to_gemini({"choices": [{"message": {"tool_calls": [{
        "id": "call_abc__thought__c2ln", "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
        "provider_specific_fields": {"thought_signature": "c2ln"},
    }]}}]})

    assert gresp["candidates"][0]["content"]["parts"] == [
        {"functionCall": {"name": "read_file", "args": {}, "id": "call_abc"}, "thoughtSignature": "c2ln"}]
