"""Integration test: the triggers extension urn:agentenv:triggers/v1 against a REAL deployed agent.

Deploys the echo agent (tst/data/a2a_agent; the protocol framework serves its triggers extension) on
a dev sandbox VM, then — with NO stubs — drives the live agent over HTTP to prove every agent-trigger
path works end to end:
- the AgentCard advertises the triggers extension (endpoint + register/decide/state methods);
- `POST /ext/triggers` registers a script covering every condition type (step / env_trigger /
  conversational / all / any) and both actions (say / end);
- `POST /ext/triggers/decide` fires the right beats for crafted (turn, solver_message, env_triggers) —
  deterministic (no LLM, no env), including `once`, `once:false` recurrence, and turn-regression reset;
- registration is fail-loud (HTTP 400) and decide validates its turn;
- `GET /ext/triggers` reads back the config + trigger summary + firing log;
- the real `RegisterAgentTriggersStep` task step registers a script against the live agent's card.

Takes ~5 min (image build + deploy). Run with the repo venv:
    .venv/bin/python -m pytest tst/integration/agent/a2a_agent/agent_triggers_e2e_test.py -v -m integration
"""
from __future__ import annotations

import httpx
import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.register_agent_triggers import RegisterAgentTriggersStep
from tst.util.a2a_test_agent import put_test_agent
from tst.util.capabilities import skip_without_remote_sandbox

# Builds an image and deploys a real sandbox VM (~5 min); excluded from the fast tier.
pytestmark = [pytest.mark.int_test_slow]

_AGENT_ID = "test-agent-triggers"

# One trigger per condition type + both actions. Registration order matters (decide evaluates in order).
_SCRIPT = {
    "triggers": [
        # step (eq): the first stakeholder reaction (decide turn 1 = reply to the solver's first response; the opening ask is the authored prompt_text, not a trigger).
        {"id": "open", "when": {"type": "step", "turn": 1},
         "actions": [{"type": "say", "text": "[PM]: Look into LeapBlock."}]},
        # conversational: react to what the solver said.
        {"id": "answer-format",
         "when": {"type": "conversational", "where": {"message": {"regex": "(?i)format|which sections"}}},
         "actions": [{"type": "say", "text": "[PM]: Title 'LeapBlock Recommendation'; sections A/B/C."}]},
        # env_trigger: the cross-side link — fires when the env gateway reports v6-rate fired.
        {"id": "re-sweep",
         "when": {"type": "env_trigger", "env_id": "env-x", "trigger_id": "v6-rate", "status": "fired"},
         "actions": [{"type": "say", "text": "[PM]: Vendor news landed — re-sweep."}]},
        # any + once:false: a recurring nudge on either the solver stalling or turn >= 6.
        {"id": "nudge", "once": False,
         "when": {"type": "any", "of": [
             {"type": "conversational", "where": {"message": {"regex": "(?i)stuck"}}},
             {"type": "step", "turn": 6, "cmp": "gte"}]},
         "actions": [{"type": "say", "text": "[PM]: Want help?"}]},
        # all + say + end: guarded acceptance that closes the conversation.
        {"id": "accept",
         "when": {"type": "all", "of": [
             {"type": "env_trigger", "env_id": "env-x", "trigger_id": "v6-accept", "status": "fired"},
             {"type": "step", "turn": 2, "cmp": "gte"}]},
         "actions": [{"type": "say", "text": "[PM]: Looks good — sending to Arthur."}, {"type": "end"}]},
    ],
}
_SCRIPT_IDS = {"open", "answer-format", "re-sweep", "nudge", "accept"}


@pytest.fixture(scope="module")
def agent() -> A2AAgent:
    return put_test_agent(_AGENT_ID)


@pytest.mark.integration
@pytest.mark.asyncio
@skip_without_remote_sandbox("modal_vm")
async def test_agent_triggers_e2e(agent):
    # Pinned to modal_vm: the configured default backend is unreachable from CI, so CI deploys on Modal.
    deployed = await agent.deploy(sandbox_type="modal_vm", ttl_seconds=1200)

    # 1. The card advertises the triggers extension with register/decide/state methods.
    ext = A2AAgent.find_extension(deployed.agent_card, A2AAgent.EXT_TRIGGERS)
    assert ext is not None, "triggers extension not advertised on the card"
    endpoint = ext["params"]["endpoint"]
    assert endpoint == "/ext/triggers"
    methods = ext["params"]["methods"]
    assert set(methods) >= {"register", "decide", "state"}
    triggers_url = deployed.a2a_url + endpoint
    decide_url = deployed.a2a_url + methods["decide"]["endpoint"]

    async with httpx.AsyncClient(timeout=30) as client:
        # 2. Register the full script (every condition type + both actions).
        resp = await client.post(triggers_url, json=_SCRIPT)
        resp.raise_for_status()
        body = resp.json()
        assert body["ok"] is True
        assert set(body["added"]) == _SCRIPT_IDS
        assert set(body["all"]) >= _SCRIPT_IDS

        # 3. Fail-loud: an unknown when.type rejects the whole batch (HTTP 400).
        bad = await client.post(triggers_url, json={"triggers": [
            {"when": {"type": "bogus"}, "actions": [{"type": "say", "text": "x"}]}]})
        assert bad.status_code == 400
        # say with no text is also rejected.
        bad2 = await client.post(triggers_url, json={"triggers": [
            {"when": {"type": "step", "turn": 1}, "actions": [{"type": "say"}]}]})
        assert bad2.status_code == 400

        async def decide(turn, message="", env=None, ctx="default"):
            r = await client.post(decide_url, json={
                "turn": turn, "solver_message": message, "context_id": ctx, "env_triggers": env or {}})
            r.raise_for_status()
            return r.json()

        # 4. Deterministic firing sequence on context c1 (step -> conversational -> env_trigger -> all/end).
        r = await decide(1, "hello", ctx="c1")
        assert r["fired"] == ["open"] and r["done"] is False
        assert r["parts"] == [{"kind": "text", "text": "[PM]: Look into LeapBlock."}]

        r = await decide(2, "what document format?", ctx="c1")
        assert r["fired"] == ["answer-format"] and r["done"] is False  # conversational regex

        r = await decide(3, "I drafted it", env={"env-x": {"v6-rate": "fired"}}, ctx="c1")
        assert r["fired"] == ["re-sweep"] and r["done"] is False  # env_trigger

        r = await decide(4, "ok done", env={"env-x": {"v6-rate": "fired", "v6-accept": "fired"}}, ctx="c1")
        assert r["fired"] == ["accept"] and r["done"] is True  # all -> say + end
        assert "sending to Arthur" in r["parts"][0]["text"]

        # 5. Turn-regression reset: replaying turn 1 clears c1's fired-set -> `open` re-fires (retry-safe).
        r = await decide(1, "hello", ctx="c1")
        assert r["fired"] == ["open"]

        # 6. once:false recurrence + the step branch of `any` on a fresh context.
        r = await decide(6, ctx="c2")
        assert r["fired"] == ["nudge"] and r["done"] is False  # any[step >= 6]
        r = await decide(7, ctx="c2")
        assert r["fired"] == ["nudge"]  # once:false -> fires again

        # 7. The conversational branch of `any` (fires on "stuck").
        r = await decide(2, "I am stuck", ctx="c3")
        assert "nudge" in r["fired"]

        # 8. decide validates its turn (non-positive -> 400).
        bad_turn = await client.post(decide_url, json={"turn": 0})
        assert bad_turn.status_code == 400

        # 9. State read-back: config + trigger summary + firing log.
        state = (await client.get(triggers_url)).json()
        summary = {t["id"]: t for t in state["triggers"]}
        assert _SCRIPT_IDS <= set(summary)
        assert summary["open"]["when_type"] == "step"
        assert summary["answer-format"]["when_type"] == "conversational"
        assert summary["re-sweep"]["when_type"] == "env_trigger"
        assert summary["nudge"]["when_type"] == "any" and summary["nudge"]["once"] is False
        assert summary["accept"]["when_type"] == "all"
        kinds = {e["kind"] for e in state["firing_log"]}
        assert {"registered", "fired", "reset"} <= kinds
        fired_ids = {e["trigger_id"] for e in state["firing_log"] if e["kind"] == "fired"}
        assert {"open", "answer-format", "re-sweep", "accept", "nudge"} <= fired_ids

    # 10. The real RegisterAgentTriggersStep drives registration against the live agent's card.
    da = DeployedAgent(
        agent_name="stakeholder",
        api_url=deployed.a2a_url,
        a2a_url=deployed.a2a_url,
        a2a_card=deployed.agent_card,
        sandbox_id=deployed.sandbox_id,
    )
    context = TaskStepContext(deployed_agents=[da])
    step = RegisterAgentTriggersStep(
        id="reg-step", version=None, agent_name="stakeholder",
        triggers=[{"id": "step-reg", "when": {"type": "step", "turn": 1},
                   "actions": [{"type": "say", "text": "registered via the task step"}]}],
    )
    await step.execute(context)
    regs = context.metadata["agent_trigger_registrations"]
    assert regs[-1]["agent_name"] == "stakeholder"
    assert "step-reg" in regs[-1]["added"]

    async with httpx.AsyncClient(timeout=30) as client:
        state = (await client.get(triggers_url)).json()
    assert "step-reg" in {t["id"] for t in state["triggers"]}
