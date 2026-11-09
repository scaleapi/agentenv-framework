"""A harness for the always_on step: an in-process gateway (clock, triggers, notifications, user chat) behind a real
DeployedGatewayEnv, an agent whose turns are scripted and act on that gateway as its role, and a user sim plug-in whose
answers are scripted. Shared by the always_on step's tests."""

from __future__ import annotations

from typing import Any, Callable, Optional

import httpx

from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.constants import AGENT_ENV_ROLE_HEADER, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.user_sim import USER_SIMS_KEY

GATEWAY = "http://gateway.test"
SOLVER = "http://solver.test"
USER_SIM = "http://user-sim.test"
T0 = "2026-11-09T14:00:00Z"  # Mon Nov 9, 9:00 AM in New York
ROLE = "default"

Turn = Callable[[int, list[str], "Rig"], Optional[str]]


class Rig:
    """The world, the agent and the user sim for one always_on run.

    ``agent(n, texts, rig)`` plays the agent's nth turn (nudges included) given the texts it was sent, acting through
    ``message``, ``quiet`` and ``schedule``, and returns its closing text; by default it stays quiet. ``answers`` are the
    user sim's answers to its
    reads, in order; once they run out it answers no_reply. Every message the agent got is in ``sent``, every read the
    user sim got in ``reads``. A ``plugin`` replaces the scripted one; what it sends the user sim agent with ``ask`` is
    in ``user_calls``, answered in turn by ``user_texts`` (then no_reply)."""

    def __init__(self, monkeypatch, tmp_path, *, agent: Optional[Turn] = None, answers: tuple = (), rate: float = 0,
                 t0: str = T0, plugin: Any = None, user_texts: tuple = ()):
        monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
        self.gw = Gateway(host="127.0.0.1", port=0, server_name="AgentEnvGateway", internal_mcp_servers=[])
        self.gw._tools_discovered = True
        real_client = httpx.AsyncClient
        transport = httpx.ASGITransport(app=self.gw._mcp.streamable_http_app())
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_client(*a, **{**kw, "transport": transport}))
        self.http = lambda: real_client(transport=transport, base_url=GATEWAY)
        self.t0, self.rate = t0, rate
        self.agent = agent or (lambda n, texts, rig: (rig.quiet(), "")[1])
        self.answers = list(answers)
        self.sent: list[list[str]] = []
        self.parts: list[list[dict]] = []
        self.reads: list[dict] = []
        self.setups: list[dict] = []
        self.plugin = plugin
        self.user_texts = list(user_texts)
        self.user_calls: list[dict] = []
        monkeypatch.setattr(ao, "POLL_SECONDS", 0)
        monkeypatch.setattr(ao, "send_and_wait", self._send_and_wait)
        monkeypatch.setattr(ao._Run, "_load_plugin", self._load_plugin)

    # ---- the world -----------------------------------------------------------------------------------------------
    async def arm(self) -> None:
        """Arm the gateway clock the way sync_env_clock does."""
        async with self.http() as client:
            r = await client.put("/clock/set-time", json={"virtual_time": self.t0,
                                                          "virtual_seconds_per_real_second": self.rate})
            r.raise_for_status()

    def triggers(self, *triggers: dict) -> None:
        self.gw._trigger_engine.register({"watch_roles": [ROLE], "triggers": list(triggers)})

    def notify_at(self, trigger_id: str, at: str, text: str, *, channel: str = "Messages", to: str = ROLE) -> dict:
        return {"id": trigger_id, "when": {"type": "time", "at": at},
                "actions": [{"type": "notify", "channel": channel, "text": text, "to": to}]}

    def now(self) -> str:
        return self.gw._clock.read()["virtual_time"]

    # ---- the agent's tools -----------------------------------------------------------------------------------------
    def message(self, text: str) -> dict:
        return self.gw._user_chat.send(text, role=ROLE)

    def quiet(self) -> dict:
        return self.gw._user_chat.quiet(role=ROLE)

    def schedule(self, note: str, at: str) -> dict:
        return self.gw._notifications.schedule(note, at=at, to=ROLE)

    async def tools(self, role: str = ROLE) -> set[str]:
        async with self.http() as client:
            r = await client.post("/step", json={"action": "list_tools"}, headers={AGENT_ENV_ROLE_HEADER: role})
            return {t["name"] for t in r.json()["tools"]}

    # ---- the run ---------------------------------------------------------------------------------------------------
    def context(self, *, user: Optional[dict] = None) -> TaskStepContext:
        card = {"capabilities": {"extensions": list(GATEWAY_EXTENSIONS)}}
        env = DeployedGatewayEnv(env_id="world", env_version=1, gateway_url=GATEWAY, mcp_url="", sandbox_id="sb",
                                 environment_card_url=f"{GATEWAY}{WELL_KNOWN_PATH}", environment_card=card,
                                 metadata={"deploy_step_id": "deploy"})
        ctx = TaskStepContext(instance_id="ti-1", deployed_envs=[env])
        ctx.deployed_agents += [DeployedAgent(agent_name="solver", api_url=SOLVER, a2a_url=SOLVER, a2a_card={}),
                                DeployedAgent(agent_name="user-sim", api_url=USER_SIM, a2a_url=USER_SIM, a2a_card={})]
        ctx.metadata[USER_SIMS_KEY] = {"linda": {
            "name": "linda", "model": "sim-model", "agent_name": "user-sim", "user_version": "ours",
            "display_name": "Linda Murphy", "first_name": "Linda", "time_zone": "America/New_York",
            "description": "Linda, planning Noor's 30th.", **(user or {})}}
        return ctx

    def step(self, **settings: Any) -> ao.AlwaysOnTaskStep:
        return ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt="sort the cdmx weekend", prompt_id="ask",
                                   agent_name="solver", **settings)

    async def run(self, step: ao.AlwaysOnTaskStep, ctx: Optional[TaskStepContext] = None) -> TaskStepContext:
        ctx = ctx or self.context()
        await step.execute(ctx)
        return ctx

    def record(self, ctx: TaskStepContext) -> dict:
        return ctx.metadata[ao.ALWAYS_ON_KEY]["week"]

    def events(self, ctx: TaskStepContext, name: str) -> list[dict]:
        return [e for e in self.record(ctx)["timeline"] if e["event"] == name]

    # ---- fakes -----------------------------------------------------------------------------------------------------
    async def _send_and_wait(self, url, parts, *, before_send=None, **kw):
        if url == USER_SIM:
            self.user_calls.append({"parts": list(parts), "context_id": kw.get("context_id")})
            text = self.user_texts.pop(0) if self.user_texts else '{"outcome": "no_reply", "message": ""}'
            return f"user-{len(self.user_calls)}", {"status": {"state": "completed", "message": {
                "parts": [{"kind": "text", "text": text}]}}}
        assert url == SOLVER, url
        if before_send:
            before_send()
        texts = [p["text"] for p in parts if p.get("kind") == "text"]
        self.sent.append(texts)
        self.parts.append(list(parts))
        closing = self.agent(len(self.sent), texts, self)
        reply = [{"kind": "text", "text": closing}] if closing else []
        return f"task-{len(self.sent)}", {"status": {"state": "completed", "message": {"parts": reply}}}

    async def _load_plugin(self, spec):
        if self.plugin is not None:
            return self.plugin
        rig = self

        class Plugin:
            @staticmethod
            def setup(version, person):
                rig.setups.append(person)
                return {"system_prompt": f"you are {person['name']}", "output_format": None}

            @staticmethod
            async def answer(version, person, history, read, ask):
                rig.reads.append(read)
                return rig.answers.pop(0) if rig.answers else {"outcome": "no_reply", "message": ""}

        return Plugin
