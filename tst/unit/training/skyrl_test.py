"""``write_dataset`` on the OpenEnv server ``build_app`` makes, served on loopback over the Episode API's fake steps,
with a local tokenizer whose template renders tools as Qwen3's does; and the parsers the env reads replies with."""

import json

import httpx
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from agent_env.openenv.environment import END_TURN
from agent_env.training.skyrl.__main__ import main
from agent_env.training.skyrl.calls import Call, Unparsed, parse
from agent_env.training.skyrl.prepare import ROW, write_dataset
from tst.unit.task.episode.run_test import (
    ALICE,
    ITEMS,
    ITEMS_MCP,
    PANTRY,
    PANTRY_MCP,
    REPLIES,
    TURN,
    _tool_sessions,
)
from tst.unit.training.nemo_gym_test import PLAYED

ADD_ITEM = {"type": "function", "function": {"name": "add_item", "description": "add_item.",
                                             "parameters": {"type": "object"}}}
FUNCTION_END_TURN = {"type": "function", "function": {"name": "end_turn", "description": END_TURN.description,
                                                      "parameters": END_TURN.input_schema}}
#: A chat template that renders tools in the system turn, after the system prompt, as Qwen3's does.
SYSTEM_TOOLS = (
    "{% if tools or messages[0]['role'] == 'system' %}{{ '<|im_start|>system\\n' }}"
    "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{% if tools %}{{ '\\n\\n' }}{% endif %}"
    "{% endif %}{% if tools %}{{ '# Tools\\n<tools>' }}{% for tool in tools %}{{ '\\n' + (tool | tojson) }}{% endfor %}"
    "{{ '\\n</tools>' }}{% endif %}{{ '<|im_end|>\\n' }}{% endif %}"
    "{% for message in messages %}{% if not (loop.first and message['role'] == 'system') %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    "{% if enable_thinking is defined and not enable_thinking %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}"
    "{% endif %}"
)
#: One that renders them in the user's turn instead.
USER_TOOLS = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] }}"
    "{% if message['role'] == 'user' and tools %}{{ '\\n' + (tools | tojson) }}{% endif %}{{ '<|im_end|>\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
SCHEDULE = {"name": "schedule", "parameters": {"type": "object", "properties": {
    "hours": {"type": "array"}, "count": {"type": "integer"}, "price": {"type": "number"}, "sell": {"type": "boolean"},
    "note": {"type": "string"}, "plan": {"type": "object"}, "either": {"type": ["integer", "string"]}, "loose": {}}}}
TOOLS = {"schedule": SCHEDULE, "end_turn": {"name": "end_turn", "parameters": {"type": "object"}}}


@pytest.fixture
def tokenizer(tmp_path):
    """A local tokenizer of whitespace-separated words with ``template`` as its chat template, saved where
    ``AutoTokenizer`` loads it."""

    def saved(template=SYSTEM_TOOLS):
        words = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        words.pre_tokenizer = pre_tokenizers.Whitespace()
        folder = tmp_path / "tokenizer"
        PreTrainedTokenizerFast(tokenizer_object=words, unk_token="[UNK]", eos_token="<|im_end|>",
                                additional_special_tokens=["<|im_start|>"], chat_template=template
                                ).save_pretrained(folder)
        return str(folder)

    return saved


def test_a_servers_tasks_become_a_row_per_task_with_its_tools_in_its_system_prompt(serve, tmp_path, tokenizer,
                                                                                   teardowns):
    server = serve({"a": PLAYED, "b": [ITEMS, ALICE, {**TURN, "prompt": "Add a barrel."}, REPLIES]}, {"smoke": ["b"]},
                   name="Dock Yard")

    written = write_dataset(f"{server.url}/", tmp_path / "data", tokenizer=tokenizer(), tool_format="qwen3_coder",
                            tool_timeout=30.0, reset_timeout=90.0, attempts=2)
    server.stop()

    tools = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in (ADD_ITEM, FUNCTION_END_TURN))
    [row] = _rows(written.folder / "smoke.parquet")
    assert row == {
        "prompt": [{"role": "system", "content": f"You keep the items.\n\n# Tools\n<tools>\n{tools}\n</tools>"},
                   {"role": "user", "content": "Add a barrel."}],
        "env_class": "agentenv",
        "data_source": "agentenv/Dock Yard",
        "episode": {"split": "smoke", "index": 0, "task_id": "b", "tools": [ADD_ITEM, FUNCTION_END_TURN]},
        "server": {"url": server.url, "tool_timeout": 30.0, "reset_timeout": 90.0, "attempts": 2,
                   "tool_format": "qwen3_coder", "token_var": None},
    }
    template = AutoTokenizer.from_pretrained(tokenizer())
    assert template.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False) == \
        template.apply_chat_template(row["prompt"][1:], tools=[ADD_ITEM, FUNCTION_END_TURN],
                                     add_generation_prompt=True, tokenize=False).replace(
            "<|im_start|>system\n", "<|im_start|>system\nYou keep the items.\n\n")
    assert [(row["episode"]["index"], row["episode"]["task_id"]) for row in _rows(written.folder / "all.parquet")] == [
        (0, "a"), (1, "b")]
    assert written.longest_prompt == max(
        len(template.apply_chat_template(row["prompt"], add_generation_prompt=True, return_dict=True)["input_ids"])
        for row in _rows(written.folder / "all.parquet"))
    assert len(teardowns.finished) == 1


def test_a_task_without_a_system_prompt_gets_one_of_its_tools(serve, tmp_path, tokenizer):
    server = serve({"a": [ITEMS, {**ALICE, "system_prompt": None}, TURN, REPLIES]})

    out = write_dataset(server.url, tmp_path, tokenizer=tokenizer()).folder
    server.stop()

    [row] = _rows(out / "all.parquet")
    tools = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in (ADD_ITEM, FUNCTION_END_TURN))
    assert row["prompt"] == [{"role": "system", "content": f"# Tools\n<tools>\n{tools}\n</tools>"},
                             {"role": "user", "content": "Add a crate."}]


def test_each_toolset_is_probed_once_and_a_rows_tools_are_its_tasks(serve, tmp_path, tokenizer, monkeypatch,
                                                                    teardowns):
    _tool_sessions(monkeypatch, {ITEMS_MCP: ["add_item"], PANTRY_MCP: ["take_item"]})
    server = serve({"a": PLAYED, "p": [PANTRY, {**ALICE, "env_ids": ["pantry"]}, TURN, REPLIES], "c": PLAYED})

    out = write_dataset(server.url, tmp_path, tokenizer=tokenizer()).folder
    server.stop()

    rows = _rows(out / "all.parquet")
    assert [[tool["function"]["name"] for tool in row["episode"]["tools"]] for row in rows] == [
        ["add_item", "end_turn"], ["take_item", "end_turn"], ["add_item", "end_turn"]]
    assert ['"name": "take_item"' in row["prompt"][0]["content"] for row in rows] == [False, True, False]
    assert len(teardowns.finished) == 2


def test_a_split_the_server_keeps_empty_has_a_rows_columns(serve, tmp_path, tokenizer):
    server = serve({"a": PLAYED}, {"empty": []})

    out = write_dataset(server.url, tmp_path, tokenizer=tokenizer()).folder
    server.stop()

    empty = pq.read_table(out / "empty.parquet")
    assert (empty.num_rows, empty.schema) == (0, ROW)
    assert pq.read_table(out / "all.parquet").schema == ROW


def test_a_server_that_asks_for_a_token_is_read_with_the_one_its_variable_holds_and_the_rows_name_it(
        serve, tmp_path, tokenizer, monkeypatch, teardowns):
    server = serve({"a": PLAYED}, token="s3cret")
    monkeypatch.setenv("DOCK_TOKEN", "s3cret")

    out = write_dataset(server.url, tmp_path / "data", tokenizer=tokenizer(), token_var="DOCK_TOKEN").folder
    monkeypatch.setenv("DOCK_TOKEN", "guess")
    with pytest.raises(httpx.HTTPStatusError, match="401 Unauthorized"):
        write_dataset(server.url, tmp_path / "refused", tokenizer=tokenizer(), token_var="DOCK_TOKEN")
    server.stop()

    [row] = _rows(out / "all.parquet")
    assert row["server"]["token_var"] == "DOCK_TOKEN"
    assert b"s3cret" not in (out / "all.parquet").read_bytes()
    assert not (tmp_path / "refused").exists()


def test_rows_over_the_prompt_limit_are_refused_counted_with_and_without_the_template_settings(serve, tmp_path,
                                                                                            tokenizer):
    server = serve({"a": PLAYED})
    plain = write_dataset(server.url, tmp_path / "plain", tokenizer=tokenizer()).longest_prompt
    unthinking = write_dataset(server.url, tmp_path / "unthinking", tokenizer=tokenizer(),
                               chat_template_kwargs={"enable_thinking": False}).longest_prompt

    with pytest.raises(ValueError, match=rf"1 rows' prompts are over max_prompt_length={unthinking - 1} tokens with "
                                         rf"their tools, the longest task a of split all at {unthinking}, which "
                                         "SkyRL's prompt filter would drop"):
        write_dataset(server.url, tmp_path / "data", tokenizer=tokenizer(), max_prompt_length=unthinking - 1,
                      chat_template_kwargs={"enable_thinking": False})
    server.stop()

    assert unthinking > plain
    assert not (tmp_path / "data").exists()


def test_a_template_that_renders_tools_outside_the_system_prompt_is_refused(serve, tmp_path, tokenizer):
    server = serve({"a": PLAYED})

    with pytest.raises(ValueError, match="chat template doesn't render tools in the system prompt, where this writer "
                                         "puts them, since SkyRL's generator passes the template no tools"):
        write_dataset(server.url, tmp_path / "data", tokenizer=tokenizer(USER_TOOLS))
    server.stop()

    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(("options", "message"), [
    ({"attempts": 0}, "attempts must be at least 1, not 0"),
    ({"tool_format": "pythonic"}, "tool_format must be one of hermes, qwen3_coder, not 'pythonic'"),
    ({"token_var": "UNSET_DOCK_TOKEN"}, "UNSET_DOCK_TOKEN isn't set: it holds the token the OpenEnv server asks for"),
])
def test_what_cant_be_written_is_refused_before_the_server_is_read(tmp_path, monkeypatch, options, message):
    monkeypatch.delenv("UNSET_DOCK_TOKEN", raising=False)

    with pytest.raises(ValueError, match=message):
        write_dataset("http://127.0.0.1:1", tmp_path / "data", tokenizer="unread", **options)
    assert list(tmp_path.iterdir()) == []


def test_the_command_writes_the_rows(serve, tmp_path, tokenizer, teardowns):
    server = serve({"a": PLAYED})

    result = CliRunner().invoke(main, [server.url, str(tmp_path / "data"), "--tokenizer", tokenizer(),
                                       "--tool-format", "qwen3_coder", "--tool-timeout", "15", "--reset-timeout",
                                       "120", "--attempts", "4", "--chat-template-kwargs",
                                       '{"enable_thinking": false}'])
    server.stop()

    assert result.exit_code == 0, result.output
    [row] = _rows(tmp_path / "data" / "all.parquet")
    assert {key: row["server"][key] for key in ("tool_format", "tool_timeout", "reset_timeout", "attempts")} == {
        "tool_format": "qwen3_coder", "tool_timeout": 15.0, "reset_timeout": 120.0, "attempts": 4}
    assert len(teardowns.finished) == 1


@pytest.mark.parametrize(("reply", "calls", "text"), [
    ('Scheduling.\n<tool_call>\n{"name": "schedule", "arguments": {"hours": [1, -1], "count": 2}}\n</tool_call>',
     [Call("schedule", {"hours": [1, -1], "count": 2})], "Scheduling."),
    ('<tool_call>{"name": "schedule", "arguments": "{\\"count\\": 2}"}</tool_call>\n'
     '<tool_call>{"name": "end_turn", "arguments": {"response": "Done."}}</tool_call>',
     [Call("schedule", {"count": 2}), Call("end_turn", {"response": "Done."})], ""),
    ('<tool_call>{"name": "schedule", "arguments": {}}', [Call("schedule", {})], ""),
    ("<think>Idle is safest.</think>Idling all day.", [], "Idling all day."),
    ('<think>Not yet: <tool_call>{"name": "end_turn", "arguments": {}}</tool_call></think>Idling all day.', [],
     "Idling all day."),
    ('Charge first, then <tool_call>{"name": "end_turn", "arguments": {}}</tool_call></think>\n'
     '<tool_call>{"name": "schedule", "arguments": {"count": 1}}</tool_call>', [Call("schedule", {"count": 1})], ""),
    ('Done.<tool_call>{"name": "end_turn", "arguments": {}}</tool_call><think>Or <tool_call>{"name": "schedule"}',
     [Call("end_turn", {})], "Done."),
    ('<tool_call>{"name": "schedule", "arguments": {"note": "keep </think> and <think> as written"}}</tool_call>',
     [Call("schedule", {"note": "keep </think> and <think> as written"})], ""),
    ('<think>Quote it.</think><tool_call>{"name": "schedule", "arguments": {"note": "</think>"}}</tool_call>',
     [Call("schedule", {"note": "</think>"})], ""),
    ('<tool_call>{"name": "sell"}</tool_call>', [Unparsed("there's no tool 'sell'; the tools are end_turn, schedule")],
     ""),
    ("<tool_call>schedule(count=2)</tool_call>", [Unparsed(
        "the tool call isn't JSON (Expecting value: line 1 column 1 (char 0)): write it as "
        '{"name": <tool name>, "arguments": <arguments as a JSON object>}')], ""),
    ('<tool_call>{"arguments": {}}</tool_call>',
     [Unparsed('the tool call names no tool: write it as {"name": <tool name>, "arguments": <arguments>}')], ""),
    ('<tool_call>{"name": "schedule", "arguments": [2]}</tool_call>',
     [Unparsed("the arguments of schedule aren't a JSON object")], ""),
])
def test_hermes_calls_are_a_json_object_a_block(reply, calls, text):
    assert parse(reply, "hermes", TOOLS) == (calls, text)


def test_qwen3_coder_calls_are_typed_by_the_tools_schema():
    reply = ("I'll sell at the peak.\n<tool_call>\n<function=schedule>\n<parameter=hours>\n[0, 2, -1]\n</parameter>\n"
             "<parameter=count>\n3\n</parameter>\n<parameter=price>\n41.5\n</parameter>\n<parameter=sell>\ntrue\n"
             "</parameter>\n<parameter=note>\n42\n</parameter>\n<parameter=plan>\n{\"at\": 18}\n</parameter>\n"
             "<parameter=either>\nseven\n</parameter>\n<parameter=loose>\n[1]\n</parameter>\n</function>\n"
             "</tool_call>\n<tool_call>\n<function=end_turn>\n<parameter=response>\nSold.\n</parameter>\n</function>"
             "\n</tool_call>")

    calls, text = parse(reply, "qwen3_coder", TOOLS)

    assert calls == [Call("schedule", {"hours": [0, 2, -1], "count": 3, "price": 41.5, "sell": True, "note": "42",
                                       "plan": {"at": 18}, "either": "seven", "loose": [1]}),
                     Call("end_turn", {"response": "Sold."})]
    assert text == "I'll sell at the peak."


def test_a_qwen3_coder_calls_argument_may_hold_the_reasoning_tags():
    reply = ("<think>Write it down.</think>\n<tool_call>\n<function=schedule>\n<parameter=note>\nkeep </think> and "
             "<think> as written\n</parameter>\n</function>\n</tool_call>")

    assert parse(reply, "qwen3_coder", TOOLS) == ([Call("schedule", {"note": "keep </think> and <think> as written"})], "")


@pytest.mark.parametrize(("reply", "calls"), [
    ("<tool_call>\n<function=schedule>\n<parameter=count>\nmany\n</parameter>\n</function>\n</tool_call>",
     [Call("schedule", {"count": "many"})]),
    ("<tool_call>\n<function=schedule>\n<parameter=count>\n4\n<parameter=sell>\nfalse\n",
     [Call("schedule", {"count": 4, "sell": False})]),
    ("<tool_call>\n<function=refund>\n</function>\n</tool_call>",
     [Unparsed("there's no tool 'refund'; the tools are end_turn, schedule")]),
    ("<tool_call>\nschedule\n</tool_call>", [Unparsed(
        "the tool call names no tool: write it as <function=tool name> with a <parameter=argument name> block per "
        "argument")]),
])
def test_qwen3_coder_calls_that_dont_fit_their_schema_are_passed_as_written_or_unparsed(reply, calls):
    assert parse(reply, "qwen3_coder", TOOLS)[0] == calls


def test_an_unknown_format_is_refused():
    with pytest.raises(ValueError, match="tool_format must be one of hermes, qwen3_coder, not 'pythonic'"):
        parse("", "pythonic", TOOLS)


def _rows(path):
    return [{**row, "episode": {**row["episode"], "tools": json.loads(row["episode"]["tools"])}}
            for row in pq.read_table(path).to_pylist()]
