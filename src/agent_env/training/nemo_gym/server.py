"""NeMo Gym's resources server for a bundle's tasks served as an OpenEnv env (``agent_env.openenv.build_app``, or the
package ``openenv import`` writes). Gym's ``simple_agent`` seeds a session with a row, which starts the row's episode,
calls the episode's tools at their routes, and verifies the session, which ends the turns the model left open, its last
reply ending the first, and answers with the episode's reward; a rollout the environment failed is masked out of
training. It runs in the venv Gym builds for it, or on its own through ``serve``, with one worker, since a rollout's
session lives in the process."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

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
    from nemo_gym.config_types import BaseServerConfig
    from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
    from omegaconf import OmegaConf
    from openenv.core.generic_client import GenericEnvClient  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name not in ("nemo_gym", "openenv"):
        raise
    raise ImportError("agent_env.training.nemo_gym.server requires pip install 'agentenv-framework[nemo-gym]', on "
                      "Python 3.13.14 or later as NeMo Gym is") from exc

from agent_env.openenv import build_app
from agent_env.training._openenv import STOP_SECONDS, Server, Session, Shared
from agent_env.training.nemo_gym.prepare import REWARD_AND_COMPONENTS, summed

logger = logging.getLogger(__name__)

#: How often the server looks for idle sessions, in seconds: every ``REAP_SECONDS``, or ``REAP_SHARE`` of
#: ``idle_timeout`` when that is sooner.
REAP_SECONDS, REAP_SHARE = 30.0, 0.25


class AgentEnvResourcesServerConfig(BaseResourcesServerConfig):
    """Where the bundle is served: ``url``, an OpenEnv server's, or ``bundle``, a folder or an installed bundle's name,
    which this server serves on a loopback port, deploying where ``sandbox`` names, at most ``max_concurrent_envs``
    episodes at once, which should cover a step's rollouts: the rest wait their turn. ``tool_timeout`` is one tool call's deadline in
    seconds, and ``reset_timeout`` how long an episode may take to deploy and open its first turn, and each turn to be
    ended; the server ends an episode that misses one, and the rollout is masked. ``idle_timeout`` is how long a
    session may go without a request before its episode is ended, freeing its slot: Gym never verifies a rollout whose
    run failed or was cancelled. ``reward_components`` has verify also answer with each of the task's scores by step id,
    as NeMo RL's GDPO takes them, and makes the reward their sum, as NeMo RL requires of a reward with components; it
    takes no ``reward``. ``masked_as_error`` answers verify for a masked rollout with HTTP 503 and why, rather than with
    ``mask_sample``, for a client that doesn't read ``mask_sample``: verifiers' NeMo Gym taskset then fails the
    rollout, which prime-rl leaves out of the loss and its group's baseline. It runs with one worker, since a rollout's
    session lives in its process."""

    url: str | None = None
    bundle: str | None = None
    sandbox: str | None = None
    reward: str | None = None
    max_concurrent_envs: int = 8
    tool_timeout: float = 60.0
    reset_timeout: float = 600.0
    idle_timeout: float = Field(600.0, gt=0, allow_inf_nan=False)
    reward_components: bool = False
    masked_as_error: bool = False

    @model_validator(mode="after")
    def _checked(self) -> "AgentEnvResourcesServerConfig":
        if (self.url is None) == (self.bundle is None):
            raise ValueError("name the bundle's server as url, or the bundle this server serves as bundle; not both")
        if self.reward_components and self.reward is not None:
            raise ValueError(REWARD_AND_COMPONENTS)
        if (self.num_workers or 1) != 1:
            raise ValueError("the server runs with one worker, since a rollout's session lives in its process; "
                             "set num_workers to 1")
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
            reaping = asyncio.create_task(self._reap())
            try:
                async with lifespan(app) as state:
                    yield state
            finally:
                reaping.cancel()
                await self._shut_down()

        app.router.lifespan_context = serving
        return app

    async def seed_session(self, request: Request, body: AgentEnvSeedSessionRequest) -> BaseSeedSessionResponse:
        """Starts the row's episode and returns before it is deployed, so the deploy overlaps the model's first turn;
        the first tool call or the verify waits for it. A seed on a session that holds an episode ends that one."""
        session_id = request.session[SESSION_ID_KEY]
        offered = frozenset(tool["name"] for tool in body.responses_create_params.get("tools") or ())
        session = _Session(Shared(self._url, offered, self.config.tool_timeout, self.config.reset_timeout,
                                  patience=self.config.reset_timeout))
        if (previous := self._sessions.pop(session_id, None)) is not None:
            await previous.close()
        self._sessions[session_id] = session
        await session.start(body.split, body.index, body.task_id)
        return BaseSeedSessionResponse()

    async def call_tool(self, tool: str, request: Request, arguments: dict[str, Any] = Body()) -> PlainTextResponse:
        """What the model sees of the call: the tool's text or error, or for ``end_turn`` the next turn's prompt."""
        session = self._sessions.get(request.session.get(SESSION_ID_KEY))
        if session is None:
            raise HTTPException(status_code=400, detail="This session has no episode: seed it first.")
        return PlainTextResponse(await session.call(tool, arguments))

    async def verify(self, request: Request, body: AgentEnvVerifyRequest) -> AgentEnvVerifyResponse:
        """The episode's reward, once the turns the model left open are ended, with its scores as the reward's
        components when the config asks for them; a rollout with none, as one the environment failed or that offered
        tools its row doesn't show the model has, is masked."""
        session = self._sessions.pop(request.session.get(SESSION_ID_KEY), None)
        if session is None:
            reward, scores, failure = None, None, "the rollout has no episode: its session wasn't seeded, or was closed"
        else:
            reward, scores, failure = await session.result(_reply(body.response))
        if failure is not None:
            logger.warning("rollout masked: %s", failure)
            if self.config.masked_as_error:
                raise HTTPException(status_code=503, detail=failure)
        components = scores if self.config.reward_components and failure is None else None
        if components:
            reward = sum(components.values())
        return AgentEnvVerifyResponse(**body.model_dump(), reward=reward or 0.0, mask_sample=failure is not None,
                                      failure_reason=failure, **({"reward_components": components} if components else {}))

    async def close_resources_session(self, request: Request,
                                      body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        if (session := self._sessions.pop(request.session.get(SESSION_ID_KEY), None)) is not None:
            await session.close()
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def _reap(self) -> None:
        """Ends the episode of each session with no request in flight and none for ``idle_timeout`` seconds."""
        while True:
            await asyncio.sleep(min(REAP_SECONDS, self.config.idle_timeout * REAP_SHARE))
            now = time.monotonic()
            for session_id, session in list(self._sessions.items()):
                if session.idle_for(now) <= self.config.idle_timeout or self._sessions.get(session_id) is not session:
                    continue
                del self._sessions[session_id]
                logger.warning("ended the episode of a rollout with no request for %gs: Gym stopped its run without "
                               "verifying it", self.config.idle_timeout)
                await session.close()

    @property
    def _url(self) -> str:
        return self._served.url if self._served is not None else self.config.url

    def _serve(self) -> Server:
        reward = summed if self.config.reward_components else self.config.reward
        return Server(build_app(self.config.bundle, reward=reward, sandbox=self.config.sandbox,
                                max_concurrent_envs=self.config.max_concurrent_envs))

    async def _shut_down(self) -> None:
        """Ends the episodes left running, then stops the server this one served the bundle on, which tears down its
        episodes."""
        sessions, self._sessions = list(self._sessions.values()), {}
        await asyncio.gather(*(session.close() for session in sessions))
        served, self._served = self._served, None
        if served is not None and not await asyncio.to_thread(served.stop):
            logger.warning("the bundle's server at %s was still tearing down its episodes after %gs; their sandboxes "
                           "may be left running", served.url, STOP_SECONDS)


def serve(folder: str | Path, *, host: str, port: int, masked_as_error: bool = False,
          idle_timeout: float | None = None) -> None:
    """Runs the resources server ``write_server`` wrote into ``folder`` on its own, at ``host``:``port``, without Gym's
    head server or the agent servers its config names, for a trainer that plays the folder's rows against the server's
    URL. ``host``, ``port`` and an ``idle_timeout`` win over any the folder's config names, and ``masked_as_error``, as
    the config takes it, is on when either it or the config sets it. As verifiers' launcher for Gym's own servers does,
    its client names a head server nothing calls."""
    folder = Path(folder).resolve()
    written = yaml.safe_load((folder / "configs" / f"{folder.name}.yaml").read_text())
    config = written[folder.name]["resources_servers"][folder.name]
    settings = {"name": folder.name, "host": host, "port": port,
                "masked_as_error": masked_as_error or config.get("masked_as_error", False),
                **({"idle_timeout": idle_timeout} if idle_timeout is not None else {})}
    resources = AgentEnvResourcesServer(
        config=AgentEnvResourcesServerConfig(**{**config, **settings}),
        server_client=ServerClient(head_server_config=BaseServerConfig(host=host, port=port),
                                   global_config_dict=OmegaConf.create({})),
    )
    app = resources.setup_webserver()
    resources.setup_liveness(app)
    resources.setup_exception_middleware(app)
    uvicorn.run(app, host=host, port=port)


class _Session(Session):
    """A rollout's session, which shows the model the tools its row lists. The server gives it the reset deadline to
    wait for room in, since Gym runs every rollout of a step at once and a server at its limit makes the rest take
    turns."""

    offered_by = "the row shows the model"
    start = Session._start
    close = Session._close

    def __init__(self, shared: Shared):
        super().__init__()
        self._shared = shared
        self._seen = time.monotonic()

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        self._seen = time.monotonic()
        try:
            return await self._step(name, arguments)
        finally:
            self._seen = time.monotonic()

    def idle_for(self, now: float) -> float:
        """How long the session has gone with no request, in seconds; none while one is in flight."""
        return 0.0 if self._lock.locked() else now - self._seen

    async def result(self, reply: str) -> tuple[float | None, dict[str, float] | None, str | None]:
        """The episode's reward and scores, or why it has none, once the turns left open are ended, ``reply`` ending
        the first; closes the session."""
        rollout = self._rollout
        if rollout is None:
            return None, None, "the rollout has no episode"
        try:
            await self._finish(rollout, reply)
        finally:
            await self._close()
        if rollout.failure is None and rollout.reward is None:
            return None, None, "the episode ended without a reward"
        return rollout.reward, rollout.scores, rollout.failure


def _reply(response: Any) -> str:
    """The text of the model's last message, which ends its turn as an agent's reply does."""
    replies = [item for item in response.output if item.type == "message"]
    return "".join(part.text for part in replies[-1].content if part.type == "output_text") if replies else ""
