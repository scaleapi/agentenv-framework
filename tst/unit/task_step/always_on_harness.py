"""A harness for the always_on step: an in-process gateway (clock, triggers, notifications, user chat) behind a real
DeployedGatewayEnv, an agent whose turns are scripted and act on that gateway as its role, and a user sim plug-in whose
answers are scripted. Shared by the always_on step's tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

import httpx
from mcp.types import CallToolResult, TextContent, Tool

from agent_env.a2a_agent import A2AAgent, conversation_store
from agent_env.config import get_config
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.constants import AGENT_ENV_ROLE_HEADER, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway
from agent_env.env.gateway.triggers import TriggerEngine
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY

GATEWAY = "http://gateway.test"
SOLVER = "http://solver.test"
USER_SIM = "http://user-sim.test"
T0 = "2026-11-09T14:00:00Z"  # Mon Nov 9, 9:00 AM in New York
ROLE = "default"

Turn = Callable[[int, list[str], "Rig"], Optional[str]]
# An agent card whose agent-config extension takes the settings the step sends.
CONFIG_CARD = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
    "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": [
        "model", "system_prompt", "effort", "output_format", "timeout_seconds", "max_turns"]}}}}}]}}


class Fail:
    """What an agent's turn returns to end in the failed state, as a harness reports a turn that broke; or in another
    state that isn't completed (``state``: canceled, rejected)."""

    def __init__(self, message: str = "the CLI crashed", error_type: Optional[str] = None,
                 state: str = "failed") -> None:
        self.message, self.error_type, self.state = message, error_type, state


class Rig:
    """The world, the agent and the user sim for one always_on run.

    ``agent(n, texts, rig)`` plays the agent's nth turn (nudges included) given the texts it was sent, acting through
    ``message``, ``quiet`` and ``schedule``, and returns its closing text; by default it stays quiet. ``answers`` are the
    user sim's answers to its
    reads, in order; once they run out it answers no_reply. Every message the agent got is in ``sent``, every read the
    user sim got in ``reads``. A ``plugin`` replaces the scripted one; what it sends the user sim agent with ``ask`` is
    in ``user_calls``, answered in turn by ``user_texts`` (then no_reply). ``during`` has, for each of the agent's
    turns, the gateway's settings and the agent's tools as the agent saw them. With ``takes_config`` the agent's card
    takes its config (``CONFIG_CARD``), whose posts land in ``configs``."""

    def __init__(self, monkeypatch, tmp_path, *, agent: Optional[Turn] = None, answers: tuple = (), rate: float = 0,
                 t0: str = T0, plugin: Any = None, user_texts: tuple = (), takes_config: bool = False):
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
        self.contact_calls: list[dict] = []
        self.during: list[dict] = []
        self.solver_contexts: list[Optional[str]] = []
        self.takes_config = takes_config
        self.configs: list[tuple[str, dict]] = []
        self._contacts: dict[str, dict] = {}
        self._monkeypatch = monkeypatch
        monkeypatch.setattr(ao, "POLL_SECONDS", 0)
        monkeypatch.setattr(ao, "send_and_wait", self._send_and_wait)
        monkeypatch.setattr(ao._Run, "_load_plugin", self._load_plugin)
        if takes_config:
            async def post(url, fields):
                self.configs.append((url, fields))
            monkeypatch.setattr(ao.protocol, "post_agent_config", post)

    # ---- the world -----------------------------------------------------------------------------------------------
    async def arm(self) -> None:
        """Arm the gateway clock the way sync_env_clock does."""
        async with self.http() as client:
            r = await client.put("/clock/set-time", json={"virtual_time": self.t0,
                                                          "virtual_seconds_per_real_second": self.rate})
            r.raise_for_status()

    def triggers(self, *triggers: dict, executors: Optional[dict] = None) -> None:
        self.gw._trigger_engine.register({"watch_roles": [ROLE], "triggers": list(triggers),
                                          **({"executors": executors} if executors else {})})

    def contact(self, name: str, *, seconds: float, reply: Callable[["Rig", int], None],
                timeout_seconds: Optional[float] = None, keeps_to_its_limit: bool = True) -> dict:
        """An AI-played contact (a trigger's executor) whose every reply takes ``seconds`` of real time, then acts with
        ``reply(rig, n)`` for its nth reply. Past its time limit the call fails, though the reply still lands, as a
        real contact's does; one that doesn't ``keep_to_its_limit`` holds its call open until it is done. Returns its
        executor spec, for ``triggers(executors={name: spec})``."""
        url = f"http://{name}.test"
        self._contacts[url] = {"name": name, "seconds": seconds, "reply": reply, "keeps": keeps_to_its_limit,
                               "replies": 0}
        rig = self

        async def call(engine, executor, instruction, context_id):
            contact = rig._contacts[executor["a2a_url"]]
            rig.contact_calls.append({"contact": contact["name"], "timeout_seconds": executor["timeout_seconds"],
                                      "context_id": context_id, "at": rig.now()})
            contact["replies"] += 1
            n, limit = contact["replies"], executor["timeout_seconds"]
            if contact["keeps"] and contact["seconds"] > limit:
                await asyncio.sleep(limit)
                asyncio.get_running_loop().call_later(contact["seconds"] - limit, contact["reply"], rig, n)
                return None
            await asyncio.sleep(contact["seconds"])
            contact["reply"](rig, n)
            return "replied"

        self._monkeypatch.setattr(TriggerEngine, "_call_executor", call)
        return {"a2a_url": url, "role": f"contact-{name}",
                **({"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {})}

    def alert(self, text: str, *, channel: str = "Gmail") -> dict:
        """Something reaching the agent the way a contact's email alert does."""
        return self.gw._notifications.notify(channel, text, to=ROLE)

    def keep_changelog(self) -> None:
        """Give the gateway an app changelog (urn:agentenv:changes/v1), held here; ``app_write`` adds to it."""
        self.changelog: list[dict] = []
        self.gw._db_conn = object()  # the gateway serves changes only for an env with a database
        self.gw._query_changelog_id = lambda: self.changelog[-1]["id"] if self.changelog else None
        self.gw._read_changelog = lambda after_id, count: [r for r in self.changelog if r["id"] > after_id][:count]

    def app_write(self, table: str, summary: str = "", *, tool: str = "app_tool", role: Optional[str] = ROLE,
                  trigger_id: Optional[str] = None) -> int:
        """One change to an app's data: by ``role``'s tool call, by trigger ``trigger_id``'s tool action, or, with
        neither, by nothing the gateway served (a website reached directly). Returns its changelog id."""
        before = self.changelog[-1]["id"] if self.changelog else 0
        self.changelog.append({"id": before + 1, "timestamp": "2026-11-09T14:00:00", "schema_name": "public",
                               "table_name": table, "operation": "INSERT", "row_id": str(before + 1),
                               "summary": summary, "changed_fields": None})
        changes, n = self.gw._changes, len(self.changelog)
        if trigger_id is not None:
            changes.observe({"event_type": "internal_tool_call", "event_id": f"w{n}", "trigger_id": trigger_id,
                             "action_index": 0, "tool": tool, "as": None, "changelog_id_before": before,
                             "changelog_id_after": before + 1})
        elif role is not None:
            changes.observe({"event_type": "tool_call", "event_id": f"c{n}", "role": role,
                             "tool_call": {"function_name": tool}})
            changes.observe({"event_type": "tool_call_result", "event_id": f"r{n}", "tool_call_event_id": f"c{n}",
                             "changelog_id_before": before, "changelog_id": before + 1})
        return before + 1

    def app_tools(self, *names: str) -> None:
        """App tools the gateway knows of, so a trigger may watch them; nothing serves them."""
        self.gw._server_tools.setdefault("apps", []).extend(Tool(name=n, inputSchema={"type": "object"}) for n in names)

    def provoke(self, tool: str, args: Optional[dict] = None) -> None:
        """The agent's tool call as the gateway's triggers see it."""
        self.gw._trigger_engine.on_tool_call(ROLE, tool, args or {},
                                             CallToolResult(content=[TextContent(type="text", text="{}")]))

    async def close(self) -> None:
        """Cancel what the gateway's triggers still have running."""
        await self.gw._trigger_engine.cancel_scheduled()

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
        ctx.deployed_agents += [DeployedAgent(agent_name="solver", api_url=SOLVER, a2a_url=SOLVER,
                                              a2a_card=CONFIG_CARD if self.takes_config else {}),
                                DeployedAgent(agent_name="user-sim", api_url=USER_SIM, a2a_url=USER_SIM, a2a_card={})]
        ctx.metadata[USER_SIMS_KEY] = {"linda": {
            "name": "linda", "model": "sim-model", "agent_name": "user-sim", "user_version": "ours",
            "display_name": "Linda Murphy", "first_name": "Linda", "time_zone": "America/New_York",
            "description": "Linda, planning Noor's 30th.", "opening": "sort the cdmx weekend", **(user or {})}}
        return ctx

    def step(self, **settings: Any) -> ao.AlwaysOnTaskStep:
        return ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt_id="ask",
                                   agent_name="solver", **settings)

    async def run(self, step: ao.AlwaysOnTaskStep, ctx: Optional[TaskStepContext] = None) -> TaskStepContext:
        ctx = ctx or self.context()
        await step.execute(ctx)
        return ctx

    def record(self, ctx: TaskStepContext) -> dict:
        """The step's summary in the run's metadata."""
        return ctx.metadata[ao.ALWAYS_ON_KEY]["week"]

    def stored(self, ctx: TaskStepContext) -> dict:
        """The step's full record, its timeline included, as the object store holds it."""
        url = self.record(ctx)["object_url"]
        return json.loads(get_config().get_object_store_at(url).get(url))

    def timeline(self, ctx: TaskStepContext) -> list[dict]:
        return self.stored(ctx)["timeline"]

    def chat(self, ctx: TaskStepContext) -> list[dict]:
        """The graded conversation as stored: each message's role and text parts."""
        doc = conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["week"])
        return [{"role": m["role"], "texts": [p["text"] for p in m["parts"] if p.get("kind") == "text"],
                 "parts": m["parts"]} for m in doc["messages"]]

    def events(self, ctx: TaskStepContext, name: str) -> list[dict]:
        return [e for e in self.timeline(ctx) if e["event"] == name]

    # ---- fakes -----------------------------------------------------------------------------------------------------
    async def _send_and_wait(self, url, parts, *, before_send=None, **kw):
        if url == USER_SIM:
            self.user_calls.append({"parts": list(parts), "context_id": kw.get("context_id")})
            text = self.user_texts.pop(0) if self.user_texts else '{"outcome": "no_reply", "message": ""}'
            return f"user-{len(self.user_calls)}", {"status": {"state": "completed", "message": {
                "parts": [{"kind": "text", "text": text}]}}}
        assert url == SOLVER, url
        self.solver_contexts.append(kw.get("context_id"))
        if before_send:
            before_send()
        chat, notes = self.gw._user_chat, self.gw._notifications
        self.during.append({"user_chat_roles": chat.roles, "stay_quiet": chat.stay_quiet, "check_roles": notes.roles,
                            "min_interval": notes.min_interval, "max_active": notes.max_active,
                            "trigger_settings": dict(self.gw._trigger_engine.state()["settings"]),
                            "clock_rate": self.gw._clock.rate, "tools": await self.tools()})
        texts = [p["text"] for p in parts if p.get("kind") == "text"]
        self.sent.append(texts)
        self.parts.append(list(parts))
        closing = self.agent(len(self.sent), texts, self)  # a TimeoutError it raises is the turn timing out
        if isinstance(closing, Fail):
            data = {"error_message": closing.message, **({"error_type": closing.error_type}
                                                          if closing.error_type else {})}
            return f"task-{len(self.sent)}", {"status": {"state": closing.state, "message": {"parts": [
                {"kind": "text", "text": closing.message}, {"kind": "data", "data": data}]}}}
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
