"""always_on sends its user sim's own-tool and peer-tools settings again, as deploy_agent stamped them on its card,
for a user sim deploy_user_sim_agent set up (its prompt and model already set) as for one it sets up itself; so a user
sim that restarted since its deploy doesn't get its tools back."""

from __future__ import annotations

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import USER_SIM, Rig

pytestmark = pytest.mark.usefixtures("local_stores")

SAVED = {"builtin_tools": {"disable": "*"}, "peer_tools": False}
CARD = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
    "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": [
        "model", "system_prompt", "output_format", "builtin_tools", "peer_tools"]}}}}}]}, **SAVED}
PROMPT_SET = {"prompt_set": True, "multi_day": False, "model_source": "user_sim_step",
              "setup": {"model": "sim-model", "model_from": "user_sim_step", "accepted_fields": ["model"]}}


@pytest.mark.parametrize("user", [PROMPT_SET, {}], ids=["deploy_user_sim_agent set it up", "always_on sets it up"])
@pytest.mark.asyncio
async def test_the_user_sim_is_sent_its_saved_tool_settings(monkeypatch, tmp_path, user):
    posted: list[tuple[str, dict]] = []

    async def post(url, fields, headers=None):
        posted.append((url, fields))

    monkeypatch.setattr(ao.protocol, "post_agent_config", post)
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    ctx = rig.context(user=user)
    next(a for a in ctx.deployed_agents if a.agent_name == "user-sim").a2a_card = CARD
    await rig.run(rig.step(max_wakes=1), ctx)

    sent = [fields for url, fields in posted if url.startswith(USER_SIM)]
    assert sent and all(fields.items() >= SAVED.items() for fields in sent)
    if user.get("prompt_set"):  # its prompt and model stay deploy_user_sim_agent's: only the tool settings go
        assert sent == [SAVED]
