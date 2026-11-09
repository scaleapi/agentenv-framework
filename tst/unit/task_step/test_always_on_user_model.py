"""The user sim agent's model: a run's user_model (task run / eval run --user-model, a bundle's user_model) wins over the
user_sim step's model, as prompt_agent honours it; the user_sim step's own step_params win over both. The record says
which model was used and where it came from."""

from __future__ import annotations

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import USER_SIM, Rig

pytestmark = pytest.mark.usefixtures("local_stores")

CONFIG_CARD = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
    "endpoint": "/ext/agent-config",
    "methods": {"set": {"request": {"supported": ["model", "system_prompt", "output_format"]}}}}}]}}


@pytest.mark.parametrize("overrides,model,source", [
    ({}, "sim-model", "user_sim_step"),
    ({"user_model": "run-model"}, "run-model", "run_user_model"),
    ({"user_model": "run-model", "step_params": {"linda-step": {"model": "step-model"}}}, "step-model",
     "user_sim_step_params"),
], ids=["the user_sim step's", "the run's", "the user_sim step's step_params"])
@pytest.mark.asyncio
async def test_the_run_s_user_model_sets_the_user_sim_agent_s_model(monkeypatch, tmp_path, overrides, model, source):
    posted: list[tuple[str, dict]] = []

    async def post(url, fields):
        posted.append((url, fields))

    monkeypatch.setattr(ao.protocol, "post_agent_config", post)
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    ctx = rig.context(user={"step_id": "linda-step"})
    next(a for a in ctx.deployed_agents if a.agent_name == "user-sim").a2a_card = CONFIG_CARD
    ctx.metadata["user_overrides"] = overrides
    ctx = await rig.run(rig.step(), ctx)

    [(url, fields)] = [(u, f) for u, f in posted if u.startswith(USER_SIM)]
    assert fields["model"] == model
    record = rig.record(ctx)["user_sim"]
    assert (record["model"], record["model_from"], record["user_sim_step_model"]) == (model, source, "sim-model")
    assert record["model_set"] is True
    assert ctx.metadata["models"]["user simulator"]["week"] == model
    assert rig.events(ctx, "start")[0]["user_model"] == model
