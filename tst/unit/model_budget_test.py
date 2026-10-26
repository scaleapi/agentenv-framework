"""``agent_env.model_budget``: the key a budgeted agent gets, what is asked of LiteLLM to make it, and how a teardown
records what it spent and deletes it — against a fake LiteLLM key API (``tst.util.fake_litellm``)."""

from __future__ import annotations

import json
import logging
import math
import re

import pytest

from agent_env import model_budget
from agent_env.config.errors import ConfigError
from agent_env.config.model import ModelConfig
from agent_env.task_step.context import TaskStepContext
from tst.util.fake_litellm import PROXY, serve


@pytest.fixture
def litellm(monkeypatch, local_stores):
    return serve(monkeypatch, local_stores, headers={"X-Bill-To": "acme"})


async def _make(**kwargs) -> model_budget.BudgetKey:
    args = {"agent_name": "solver", "max_usd": 2.5, "ttl_seconds": 600,
            "attribution": {"run_id": "run-1", "pipeline_step": "task_deploy"}}
    return await model_budget.make_key(**{**args, **kwargs})


@pytest.mark.asyncio
async def test_a_key_is_made_with_the_budget_and_the_admin_key_and_outlives_the_agent_by_an_hour(litellm):
    key = await _make()

    [request] = litellm.requests
    assert (request.method, str(request.url)) == ("POST", f"{PROXY}/key/generate")
    assert request.headers["authorization"] == "Bearer sk-admin"
    assert request.headers["x-bill-to"] == "acme"  # [model.headers] go with every call to the endpoint
    body = json.loads(request.content)
    assert body["max_budget"] == 2.5 and body["duration"] == "4200s"
    assert re.fullmatch(r"agentenv/run-1/solver/[0-9a-f]{8}", body["key_alias"])
    assert body["metadata"] == {"run_id": "run-1", "pipeline_step": "task_deploy", "agent_name": "solver"}
    made = litellm.keys[key.key_id]
    assert (made.key, made.alias, key.alias, key.max_usd) == (key.key, body["key_alias"], body["key_alias"], 2.5)
    assert key.key not in repr(key)


@pytest.mark.asyncio
async def test_an_alias_keeps_to_the_characters_litellm_takes(litellm):
    key = await _make(agent_name="my agent!", attribution={})

    assert re.fullmatch(r"agentenv/my-agent/[0-9a-f]{8}", key.alias)


@pytest.mark.asyncio
async def test_an_agent_pointed_at_another_endpoint_gets_no_key_and_the_admin_key_goes_nowhere(litellm):
    with pytest.raises(ValueError, match="configured model endpoint"):
        await _make(base_url="https://elsewhere.test")

    assert litellm.requests == []
    await _make(base_url=PROXY + "/")  # the configured endpoint, however it is written


@pytest.mark.asyncio
async def test_without_an_admin_key_no_key_is_made(litellm, local_stores, monkeypatch):
    monkeypatch.setattr(local_stores, "_model_cfg", ModelConfig(base_url=PROXY, api_key="sk-shared"))

    with pytest.raises(ConfigError, match=r"\[model\] admin_api_key"):
        await _make()
    assert litellm.requests == []


@pytest.mark.asyncio
async def test_a_refusal_says_what_litellm_said(litellm):
    litellm.refuse = (401, "Authentication Error, Only proxy admin can be used to generate keys")

    with pytest.raises(RuntimeError, match="answered 401: Authentication Error, Only proxy admin"):
        await _make()


@pytest.mark.asyncio
async def test_a_key_that_belongs_to_a_user_is_deleted_and_refused(litellm):
    """A user's key could make its agent more keys, each with a budget of its own."""
    litellm.user_id = "user-1"

    with pytest.raises(RuntimeError, match="belongs to a user"):
        await _make()
    assert litellm.keys == {} and len(litellm.calls("/key/delete")) == 1


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True, "5"])
def test_a_budget_is_a_positive_number_of_dollars(value):
    with pytest.raises(ValueError, match="model_budget_usd must be a positive number"):
        model_budget.check_budget(value, step_id="deploy")


def test_no_budget_and_a_whole_number_budget():
    assert model_budget.check_budget(None, step_id="deploy") is None
    assert model_budget.check_budget(5, step_id="deploy") == 5.0


@pytest.mark.asyncio
async def test_a_teardown_blocks_the_key_and_litellm_keeps_it(litellm, caplog):
    """Blocked, not deleted: LiteLLM writes spend every 10 to 15 seconds, and a blocked key still takes in what the
    agent's last calls cost."""
    key = await _make()
    context = TaskStepContext()
    model_budget.record(context, "solver", key)

    with caplog.at_level(logging.INFO, logger="agent_env.model_budget"):
        await model_budget.revoke(context)

    assert context.metadata["model_budgets"] == {"solver": {
        "max_usd": 2.5, "key_alias": key.alias, "key_id": key.key_id, "key_blocked": True}}
    assert litellm.keys[key.key_id].blocked
    assert f"Blocked the model budget key of agent 'solver' ({key.alias})" in caplog.text
    asked = len(litellm.requests)
    await model_budget.revoke(context)  # a second teardown has nothing left to do
    assert len(litellm.requests) == asked


def test_a_rolled_back_budget_is_one_whose_key_the_undo_stripped():
    def stored(**keys):
        return {"metadata": {"model_budgets": {name: {"key_id": key_id} for name, key_id in keys.items()}}}

    before = stored(solver="k1", critic="k2")
    assert model_budget.rolled_back(before, stored(critic="k2")) == {"solver": {"key_id": "k1"}}
    # redeployed under the same name, the agent has a new key, and the old one is still rolled back
    assert model_budget.rolled_back(before, stored(solver="k3", critic="k2")) == {"solver": {"key_id": "k1"}}
    assert model_budget.rolled_back(before, before) == {} and model_budget.rolled_back({}, {}) == {}


@pytest.mark.asyncio
async def test_a_teardown_blocks_the_keys_a_retry_rolled_back_too(litellm):
    old, new = await _make(), await _make()
    context, stripped = TaskStepContext(), TaskStepContext()
    model_budget.record(context, "solver", new)
    model_budget.record(stripped, "solver", old)
    context.metadata["retry_resets"] = [{"attempt": 2, model_budget.ROLLED_BACK_KEY: stripped.metadata["model_budgets"]}]

    await model_budget.revoke(context)

    assert litellm.keys[old.key_id].blocked and litellm.keys[new.key_id].blocked


@pytest.mark.asyncio
async def test_a_key_that_cannot_be_blocked_is_left_to_expire_and_the_others_are_blocked(litellm, caplog):
    context = TaskStepContext()
    for name in ("a", "b"):
        model_budget.record(context, name, await _make(agent_name=name))
    budgets = context.metadata["model_budgets"]
    litellm.fail_block.add(budgets["a"]["key_id"])

    with caplog.at_level(logging.WARNING, logger="agent_env.model_budget"):
        await model_budget.revoke(context)

    assert (budgets["a"]["key_blocked"], budgets["b"]["key_blocked"]) == (False, True)
    assert ("Could not block the model budget key of agent 'a': proxy.test answered 500: database unavailable; "
            "it expires on its own") in caplog.text


@pytest.mark.asyncio
async def test_a_key_already_gone_counts_as_blocked(litellm):
    key = await _make()
    context = TaskStepContext()
    model_budget.record(context, "solver", key)
    await model_budget.delete_key(key.key_id)

    await model_budget.revoke(context)

    assert context.metadata["model_budgets"]["solver"]["key_blocked"] is True


@pytest.mark.asyncio
async def test_the_context_never_holds_the_key(litellm):
    key = await _make()
    context = TaskStepContext()
    model_budget.record(context, "solver", key)

    assert key.key not in json.dumps(context.to_dict()) and key.key not in json.dumps(context.to_safe_dict())


@pytest.mark.asyncio
async def test_a_key_already_gone_deletes_quietly(litellm):
    key = await _make()
    await model_budget.delete_key(key.key_id)

    await model_budget.delete_key(key.key_id)  # LiteLLM answers 404 for a key it no longer has
    assert litellm.keys == {}
