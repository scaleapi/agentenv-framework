"""NeMo Gym's resources server for a bundle's tasks served as an OpenEnv env (``agent_env.openenv.build_app``, or the
package ``openenv import`` writes). Gym's ``simple_agent`` seeds a session with a row, which starts the row's episode,
calls the episode's tools at their routes, and verifies the session, which ends the turns the model left open, its last
reply ending the first, and answers with the episode's reward; a rollout the environment failed is masked out of
training. It runs in the venv Gym builds for it, with one worker, since a rollout's session lives in the process."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import ConfigDict, PrivateAttr, model_validator

try:
    from nemo_gym.base_resources_server import (
        BaseResourcesServerConfig,
        BaseSeedSessionRequest,
        BaseSeedSessionResponse,
        BaseVerifyRequest,
        BaseVerifyResponse,
        ResourcesCloseSessionRequest,
        ResourcesCloseSessionResponse,
        SimpleResourcesServer,
    )
    from nemo_gym.server_utils import SESSION_ID_KEY
    from openenv.core.generic_client import GenericEnvClient  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name not in ("nemo_gym", "openenv"):
        raise
    raise ImportError("agent_env.training.nemo_gym.server requires pip install 'agentenv-framework[nemo-gym]', on "
                      "Python 3.13.14 or later as NeMo Gym is") from exc

from agent_env.openenv import build_app
from agent_env.training._openenv import STOP_SECONDS, Server, Session, Shared

logger = logging.getLogger(__name__)


class AgentEnvResourcesServerConfig(BaseResourcesServerConfig):
    """Where the bundle is served: ``url``, an OpenEnv server's, or ``bundle``, a folder or an installed bundle's name,
    which this server serves on a loopback port, deploying where ``sandbox`` names, at most ``max_concurrent_envs``
    episodes at once, which must cover the rollouts Gym runs at once. ``tool_timeout`` is one tool call's deadline in
    seconds, and ``reset_timeout`` how long an episode may take to deploy and open its first turn, and each turn to be
    ended; the server ends an episode that misses one, and the rollout is masked."""

    url: str | None = None
    bundle: str | None = None
    sandbox: str | None = None
    reward: str | None = None
    max_concurrent_envs: int = 8
    tool_timeout: float = 60.0
    reset_timeout: float = 600.0

    @model_validator(mode="after")
    def _one_source(self) -> "AgentEnvResourcesServerConfig":
        if (self.url is None) == (self.bundle is None):
            raise ValueError("name the bundle's server as url, or the bundle this server serves as bundle; not both")
        return self


class AgentEnvSeedSessionRequest(BaseSeedSessionRequest):
    """A row: the tools it shows the model, and the task it plays."""

    model_config = ConfigDict(extra="allow")

    responses_create_params: dict[str, Any]
    split: str
    index: int
    task_id: str | None = None


class AgentEnvVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class AgentEnvVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class AgentEnvResourcesServer(SimpleResourcesServer):
    config: AgentEnvResourcesServerConfig
    ray_enabled = False
    _sessions: dict[str, "_Session"] = PrivateAttr(default_factory=dict)
    _served: Server | None = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{tool}")(self.call_tool)
        lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def serving(app: FastAPI) -> AsyncIterator[Any]:
            if self.config.bundle is not None:
                self._served = await asyncio.to_thread(self._serve)
            try:
                async with lifespan(app) as state:
                    yield state
            finally:
                await self._shut_down()

        app.router.lifespan_context = serving
        return app

    async def seed_session(self, request: Request, body: AgentEnvSeedSessionRequest) -> BaseSeedSessionResponse:
        """Starts the row's episode and returns before it is deployed, so the deploy overlaps the model's first turn;
        the first tool call or the verify waits for it. A seed on a session that holds an episode ends that one."""
        session_id = request.session[SESSION_ID_KEY]
        offered = frozenset(tool["name"] for tool in body.responses_create_params.get("tools") or ())
        session = _Session(Shared(self._url, offered, self.config.tool_timeout, self.config.reset_timeout))
        if (previous := self._sessions.pop(session_id, None)) is not None:
            await previous._close()
        self._sessions[session_id] = session
        await session._start(body.split, body.index, body.task_id)
        return BaseSeedSessionResponse()

    async def call_tool(self, tool: str, request: Request, arguments: dict[str, Any] = Body()) -> PlainTextResponse:
        """What the model sees of the call: the tool's text or error, or for ``end_turn`` the next turn's prompt."""
        session = self._sessions.get(request.session.get(SESSION_ID_KEY))
        if session is None:
            raise HTTPException(status_code=400, detail="This session has no episode: seed it first.")
        return PlainTextResponse(await session._step(tool, arguments))

    async def verify(self, request: Request, body: AgentEnvVerifyRequest) -> AgentEnvVerifyResponse:
        """The episode's reward, once the turns the model left open are ended; a rollout with none, as one the
        environment failed or that offered tools its row doesn't show the model has, is masked."""
        session = self._sessions.pop(request.session.get(SESSION_ID_KEY), None)
        if session is None:
            reward, failure = None, "the rollout has no episode: its session wasn't seeded, or was closed"
        else:
            reward, failure = await session.result(_reply(body.response))
        if failure is not None:
            logger.warning("rollout masked: %s", failure)
        return AgentEnvVerifyResponse(**body.model_dump(), reward=reward or 0.0, mask_sample=failure is not None,
                                      failure_reason=failure)

    async def close_resources_session(self, request: Request,
                                      body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        if (session := self._sessions.pop(request.session.get(SESSION_ID_KEY), None)) is not None:
            await session._close()
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    @property
    def _url(self) -> str:
        return self._served.url if self._served is not None else self.config.url

    def _serve(self) -> Server:
        return Server(build_app(self.config.bundle, reward=self.config.reward, sandbox=self.config.sandbox,
                                max_concurrent_envs=self.config.max_concurrent_envs))

    async def _shut_down(self) -> None:
        """Ends the episodes left running, then stops the server this one served the bundle on, which tears down its
        episodes."""
        sessions, self._sessions = list(self._sessions.values()), {}
        await asyncio.gather(*(session._close() for session in sessions))
        served, self._served = self._served, None
        if served is not None and not await asyncio.to_thread(served.stop):
            logger.warning("the bundle's server at %s was still tearing down its episodes after %gs; their sandboxes "
                           "may be left running", served.url, STOP_SECONDS)


class _Session(Session):
    """A rollout's session, which shows the model the tools its row lists."""

    offered_by = "the row shows the model"

    def __init__(self, shared: Shared):
        super().__init__()
        self._shared = shared

    def _misses(self, offered: frozenset[str]) -> bool:
        return not offered <= self._shared.offered

    async def result(self, reply: str) -> tuple[float | None, str | None]:
        """The episode's reward, or why it has none, once the turns left open are ended, ``reply`` ending the first;
        closes the session."""
        rollout = self._rollout
        if rollout is None:
            return None, "the rollout has no episode"
        try:
            await self._finish(rollout, reply)
        finally:
            await self._close()
        if rollout.failure is None and rollout.reward is None:
            return None, "the episode ended without a reward"
        return rollout.reward, rollout.failure


def _reply(response: Any) -> str:
    """The text of the model's last message, which ends its turn as an agent's reply does."""
    replies = [item for item in response.output if item.type == "message"]
    return "".join(part.text for part in replies[-1].content if part.type == "output_text") if replies else ""
