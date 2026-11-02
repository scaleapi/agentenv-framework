"""``AgentEnvWorkflow`` running a bundle's task with its own agents, as AReaL's agent workflows run it in inline mode:
the solver's deploy alone is given AReaL's proxy, a stand-in here, as its model endpoint, under the key of the session
AReaL records the calls in, with no endpoint configured; the task's verifier scores the run."""

import asyncio
import json

import pytest

from agent_env.training import areal
from agent_env.training.areal import AgentEnvWorkflow, task_rows
from tst.unit.bundle._support import layout
from tst.unit.conftest import local_stores  # noqa: F401
from tst.unit.training.harness_test import ASK, DEPLOY, asked, bundle, engine, hanging, registries  # noqa: F401

#: What AReaL passes an inline agent besides its proxy and session key.
EXTRA = {"http_client": None, "proxy_gateway_api_key": "gateway"}


def _row(rows, name):
    return next(row for row in rows if row["task_id"].endswith(f"/{name}"))


def test_the_solver_alone_calls_the_proxy_under_the_sessions_key_and_the_run_returns_its_reward(engine, bundle):
    rows = task_rows(bundle)

    reward = asyncio.run(AgentEnvWorkflow(bundle).run(_row(rows, "sum"), base_url=engine.url, api_key="session-1",
                                                      **EXTRA))

    [call] = asked
    assert sorted(row["task_id"].rsplit("/", 1)[1] for row in rows) == ["broken", "hanging", "sum"]
    assert _row(rows, "sum")["messages"] == [{"role": "user", "content": "What is 6 * 7?"}]
    assert (call["model"], call["key"], call["answer"]) == ("default", "session-1", "It is 42.")
    assert call["overrides"]["step_params"] == {"agent": {"litellm_base_url": engine.url,
                                                          "litellm_api_key": "session-1"}}
    assert not {"litellm_base_url", "litellm_api_key"} & call["overrides"].keys()
    assert reward == 1.0 and isinstance(reward, float)


def test_the_agent_named_is_the_one_the_solver_deploys(engine, bundle):
    row = _row(task_rows(bundle, agent="mine"), "sum")

    asyncio.run(AgentEnvWorkflow(bundle, agent="mine").run(row, base_url=engine.url, api_key="session-1"))

    [call] = asked
    assert call["overrides"]["step_params"]["agent"] == {
        "a2a_agent_id": "mine", "a2a_agent_version": 1, "litellm_base_url": engine.url, "litellm_api_key": "session-1"}


def test_a_run_that_fails_raises_so_areal_leaves_the_sample_out(engine, bundle):
    with pytest.raises(RuntimeError, match="the agent crashed"):
        asyncio.run(AgentEnvWorkflow(bundle).run(_row(task_rows(bundle), "broken"), base_url=engine.url, api_key="k"))


def test_a_run_that_records_no_score_raises_so_areal_leaves_the_sample_out(engine, bundle, tmp_path):
    unchecked = layout(tmp_path / "unchecked", {"tasks/sum.json": json.dumps([DEPLOY, ASK])})

    with pytest.raises(ValueError, match="records no score to train on"):
        asyncio.run(AgentEnvWorkflow(unchecked).run(_row(task_rows(unchecked), "sum"), base_url=engine.url,
                                                    api_key="k"))
    assert len(asked) == 1


@pytest.fixture
def torn(monkeypatch):
    """The task ids of the runs torn down, each recorded once its teardown is done."""
    done, teardown = [], areal.teardown_run

    async def tears_down(context):
        report = await teardown(context)
        done.append(context.metadata["task_id"])
        return report

    monkeypatch.setattr(areal, "teardown_run", tears_down)
    return done


def test_a_cancelled_run_is_stopped_and_torn_down_before_the_cancel_returns(engine, bundle, torn):
    workflow = AgentEnvWorkflow(bundle)
    row = _row(task_rows(bundle), "hanging")

    async def cancelled():
        running = asyncio.ensure_future(workflow.run(row, base_url=engine.url, api_key="k"))
        assert await asyncio.to_thread(hanging.wait, 30)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        return list(torn)

    assert [task_id.rsplit("/", 1)[1] for task_id in asyncio.run(cancelled())] == ["hanging"]


def test_a_row_naming_no_task_of_the_workflows_is_refused(engine, bundle):
    with pytest.raises(KeyError, match="build the dataset with task_rows"):
        asyncio.run(AgentEnvWorkflow(bundle).run({"task_id": "another-bundle/sum"}, base_url=engine.url, api_key="k"))


def test_a_process_prepares_the_bundle_once_however_often_areal_builds_the_workflow(bundle, monkeypatch):
    prepared, trained = [], areal.trained
    monkeypatch.setattr(areal, "trained", lambda *args: prepared.append(args) or trained(*args))

    AgentEnvWorkflow(bundle, tasks=["sum"])
    AgentEnvWorkflow(bundle, tasks=["sum"])
    AgentEnvWorkflow(bundle, tasks=["sum", "broken"])

    assert [args[1] for args in prepared] == [("sum",), ("sum", "broken")]


def test_a_failed_sample_leaves_its_group_alone_unless_areal_found_the_model_at_fault(bundle):
    workflow = AgentEnvWorkflow(bundle)

    assert workflow.should_reject_failed_sample(RuntimeError("the agent crashed"), "unknown_failure_reject")
    assert workflow.should_reject_failed_sample(RuntimeError("the sandbox went"), None)
    assert not workflow.should_reject_failed_sample(RuntimeError("context overflow"), "model_failure_zero")
