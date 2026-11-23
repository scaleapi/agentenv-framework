"""``Usage``'s cache counts and includes-cache flag: validated like its other fields, written by ``to_dict`` only when
set, read back by ``from_dict`` from any protocol version, and carried by a failed result as by a successful one."""

import pytest

from agentenv_protocol.a2a_agent.tasks.v1 import TaskResult, Usage

FULL = Usage(tool_call_count=2, input_tokens=120, output_tokens=7, total_tokens=127, cache_read_tokens=100,
             cache_write_tokens=20, input_includes_cache=True, cost_usd=0.01, provider_details={"model": "m"})


@pytest.mark.parametrize("usage", [FULL, Usage(), Usage(input_tokens=5), Usage(input_includes_cache=False),
                                   Usage(cache_read_tokens=0, cache_write_tokens=0)],
                         ids=["full", "empty", "no-cache-fields", "flag-false", "zero-cache"])
def test_usage_round_trips_with_and_without_the_cache_fields(usage):
    assert Usage.from_dict(usage.to_dict()) == usage


def test_to_dict_writes_the_cache_fields_only_when_set():
    assert FULL.to_dict() == {"tool_call_count": 2, "input_tokens": 120, "output_tokens": 7, "total_tokens": 127,
                              "cache_read_tokens": 100, "cache_write_tokens": 20, "input_includes_cache": True,
                              "cost_usd": 0.01, "provider_details": {"model": "m"}}
    assert Usage(input_tokens=5, input_includes_cache=False).to_dict() == {"input_tokens": 5,
                                                                          "input_includes_cache": False}


def test_an_older_agents_usage_reads_with_no_cache_fields_and_a_newer_ones_unknown_keys_are_ignored():
    older = Usage.from_dict({"tool_call_count": 1, "input_tokens": 9, "output_tokens": 4,
                             "provider_details": {"cache_read_tokens": 3}})
    assert (older.cache_read_tokens, older.cache_write_tokens, older.input_includes_cache) == (None, None, None)
    assert older.provider_details == {"cache_read_tokens": 3}
    assert Usage.from_dict({"input_tokens": 9, "reasoning_tokens": 2, "cache_read_tokens": None}) == Usage(
        input_tokens=9)


@pytest.mark.parametrize("field", ["cache_read_tokens", "cache_write_tokens"])
@pytest.mark.parametrize("value", [-1, 1.5, "3", True])
def test_a_cache_count_that_isnt_a_non_negative_integer_is_refused(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        Usage(**{field: value})
    with pytest.raises(ValueError, match=field):
        Usage.from_dict({field: value})


@pytest.mark.parametrize("value", [1, 0, "true", "false"])
def test_an_includes_cache_flag_that_isnt_a_bool_is_refused(value):
    with pytest.raises(ValueError, match="input_includes_cache must be true, false or null"):
        Usage(input_includes_cache=value)


def test_from_dict_refuses_what_isnt_a_mapping():
    with pytest.raises(TypeError, match="usage must be a mapping"):
        Usage.from_dict([("input_tokens", 1)])


@pytest.mark.parametrize("details", [[], False, 0, "", "x", [("a", 1)]])
def test_from_dict_refuses_provider_details_that_arent_a_mapping_even_when_empty(details):
    with pytest.raises(TypeError, match="provider_details must be a mapping"):
        Usage.from_dict({"input_tokens": 1, "provider_details": details})
    assert Usage.from_dict({"input_tokens": 1, "provider_details": None}) == Usage(input_tokens=1)


@pytest.mark.parametrize("usage", [Usage(cache_read_tokens=0), Usage(cache_write_tokens=3),
                                   Usage(input_includes_cache=False)], ids=["read", "write", "flag"])
def test_a_usage_with_only_cache_fields_isnt_empty(usage):
    assert not usage.is_empty
    assert Usage().is_empty


def test_a_failed_result_carries_its_usage():
    result = TaskResult.failure("agent.boom", "it broke", usage=FULL)
    assert (result.outcome, result.usage) == ("failed", FULL)
    assert TaskResult.failure("agent.boom", "it broke").usage == Usage()


# Real responses recorded through a LiteLLM proxy: one Claude prompt of 7,818 tokens, 7,809 of them cacheable, and one
# GPT prompt, through Chat Completions, Anthropic's Messages and OpenAI's Responses; the LiteLLM client's shape is as
# LiteLLM 1.83's client reads Anthropic's, and the Responses read with a cache hit is OpenAI's documented shape.
RESPONSES = {
    "litellm-proxy-claude-writes": ({"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14,
                                     "cache_creation_input_tokens": 7809, "cache_read_input_tokens": 0},
                                    Usage(input_tokens=7818, output_tokens=5, total_tokens=7823, cache_read_tokens=0,
                                          cache_write_tokens=7809, input_includes_cache=True)),
    "litellm-proxy-claude-reads": ({"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14,
                                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 7809},
                                   Usage(input_tokens=7818, output_tokens=5, total_tokens=7823, cache_read_tokens=7809,
                                         cache_write_tokens=0, input_includes_cache=True)),
    "anthropic-messages-reads": ({"input_tokens": 9, "output_tokens": 5, "cache_creation_input_tokens": 0,
                                  "cache_read_input_tokens": 7809, "cache_creation": {"ephemeral_5m_input_tokens": 0}},
                                 Usage(input_tokens=7818, output_tokens=5, total_tokens=7823, cache_read_tokens=7809,
                                       cache_write_tokens=0, input_includes_cache=True,
                                       provider_details={"cache_creation": {"ephemeral_5m_input_tokens": 0}})),
    "anthropic-messages-writes-for-an-hour": (
        {"input_tokens": 9, "output_tokens": 5, "cache_creation_input_tokens": 7813, "cache_read_input_tokens": 0,
         "cache_creation": {"ephemeral_1h_input_tokens": 7813, "ephemeral_5m_input_tokens": 0}, "speed": "standard"},
        Usage(input_tokens=7822, output_tokens=5, total_tokens=7827, cache_read_tokens=0, cache_write_tokens=7813,
              input_includes_cache=True,
              provider_details={"cache_creation": {"ephemeral_1h_input_tokens": 7813, "ephemeral_5m_input_tokens": 0}})),
    "openai-responses-gpt": (
        {"input_tokens": 7823, "output_tokens": 3, "total_tokens": 7826, "cost": None,
         "input_tokens_details": {"audio_tokens": None, "cached_tokens": 0, "cache_write_tokens": 0, "text_tokens": None,
                                  "cache_write_tokens_details": {"ttl_12h_tokens": 0, "ttl_30m_tokens": 0}},
         "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": None}},
        Usage(input_tokens=7823, output_tokens=3, total_tokens=7826, cache_read_tokens=0, cache_write_tokens=0,
              input_includes_cache=True,
              provider_details={"cache_write_tokens_details": {"ttl_12h_tokens": 0, "ttl_30m_tokens": 0}})),
    "openai-responses-claude": (
        {"input_tokens": 7819, "output_tokens": 5, "total_tokens": 7824, "cost": None,
         "input_tokens_details": {"audio_tokens": None, "cached_tokens": 0, "text_tokens": 7819},
         "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": 5}},
        Usage(input_tokens=7819, output_tokens=5, total_tokens=7824, cache_read_tokens=0, input_includes_cache=True)),
    "openai-responses-reads": ({"input_tokens": 7823, "output_tokens": 3, "total_tokens": 7826,
                                "input_tokens_details": {"cached_tokens": 7680}},
                               Usage(input_tokens=7823, output_tokens=3, total_tokens=7826, cache_read_tokens=7680,
                                     input_includes_cache=True)),
    "litellm-client-claude-reads": ({"prompt_tokens": 7818, "completion_tokens": 5, "total_tokens": 7823,
                                     "prompt_tokens_details": {"cached_tokens": 7809},
                                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 7809},
                                    Usage(input_tokens=7818, output_tokens=5, total_tokens=7823,
                                          cache_read_tokens=7809, cache_write_tokens=0, input_includes_cache=True)),
    "openai-gpt-reads": ({"prompt_tokens": 7822, "completion_tokens": 2, "total_tokens": 7824,
                          "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 7680}},
                         Usage(input_tokens=7822, output_tokens=2, total_tokens=7824, cache_read_tokens=7680,
                               input_includes_cache=True)),
    "openai-no-details": ({"prompt_tokens": 9, "completion_tokens": 4},
                          Usage(input_tokens=9, output_tokens=4, total_tokens=13, input_includes_cache=True)),
    "details-below-the-cache": ({"prompt_tokens": 9, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 50}},
                                Usage(input_tokens=59, output_tokens=1, total_tokens=60, cache_read_tokens=50,
                                      input_includes_cache=True)),
    "nothing": ({}, Usage()),
    "unreadable": ({"prompt_tokens": "9", "completion_tokens": True, "prompt_tokens_details": "x"}, Usage()),
    "bedrock-camel-case": ({"inputTokens": 5, "outputTokens": 2, "cacheReadInputTokens": 1},
                           Usage(provider_details={"unrecognized": ["cacheReadInputTokens", "inputTokens",
                                                                    "outputTokens"]})),
}


@pytest.mark.parametrize("response, expected", RESPONSES.values(), ids=RESPONSES.keys())
def test_a_model_responses_usage_reads_with_input_counting_the_cache(response, expected):
    assert Usage.from_model_response(response) == expected
