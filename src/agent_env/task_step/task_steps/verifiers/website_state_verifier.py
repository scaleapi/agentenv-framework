"""Verify the state of deployed websites, a website env or a multi env's: each check reads one value from one session
of one site, as the site reports it ({initial_state, current_state, state_diff}), and compares it with an expected
value. The sites are the deployed env's card's (``website_sites``), and each one's state is its card's
``website/v1`` ``state`` (``website_state``). A site whose card has no ``state`` is read, deprecated, at the path its
``params.state.path`` names, else at ``/go?sid=``, the verifier endpoint CUA-Gym sites serve; a record with no card
to read, an older one, reads each site its env definition names at the gateway's ``/svc/<name>/go``, deprecated too."""

from __future__ import annotations

import logging
import re
import uuid

import httpx
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Optional

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.client import ChildEnv
from agentenv_protocol.website import WEBSITE_EXTENSION_URI, website_sites, website_state

from agent_env.entity.refs import EntityRef
from agent_env.env import legacy_protocol
from agent_env.env.env import Env, gateway_url_of
from agent_env.store.base import NotFoundError
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency, attach_retry_config
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

logger = logging.getLogger(__name__)

SOURCES = ("current_state", "initial_state", "state_diff")
OPS = ("eq", "ne", "ge", "gt", "le", "lt", "in", "contains")
REDUCES = ("len", "keys", "bool")
DEFAULT_SID = "default"
GO_TIMEOUT_S = 30
LEGACY_STATE_PATH = "/go"


@dataclass(frozen=True)
class StateCheck:
    """One check: ``op`` of the value at ``path`` (dot-separated, a key's ``.`` and ``\\`` escaped with ``\\``, a list
    index by number; empty for the whole source) in session ``sid`` of ``site``, a website env's environment_name,
    reduced by ``reduce`` first when given, and ``value``. A path that isn't there reads as None."""

    site: str
    path: str
    op: str
    value: Any = None
    sid: str = DEFAULT_SID
    source: str = "current_state"
    reduce: Optional[str] = None
    weight: float = 1.0

    def problems(self) -> list[str]:
        named = f"check of {self.site!r} {self.path!r}"
        return ([f"{named}: op {self.op!r} is none of {', '.join(OPS)}"] if self.op not in OPS else []) + (
            [f"{named}: source {self.source!r} is none of {', '.join(SOURCES)}"] if self.source not in SOURCES else []) + (
            [f"{named}: reduce {self.reduce!r} is none of {', '.join(REDUCES)}"]
            if self.reduce is not None and self.reduce not in REDUCES else [])

    def result(self, sessions: dict) -> dict:
        """The check's result row against ``sessions``, each site's ``/go`` answer by its sid."""
        session = sessions.get(self.site, {}).get(self.sid)
        actual = None if session is None else _read(session.get(self.source), self.path, self.source == "state_diff")
        if self.reduce is not None:
            actual = _reduce(actual, self.reduce)
        try:
            passed = _compare(actual, self.op, self.value)
        except TypeError:
            passed = False
        return {"criterion": f"{self.site}[{self.sid}].{self.source}{'.' + self.path if self.path else ''}"
                             f"{' |' + self.reduce if self.reduce else ''} {self.op} {self.value!r}",
                "result": passed, "actual": actual, "weight": self.weight,
                **({} if session is not None else {"detail": f"{self.site!r} has no session {self.sid!r}"})}


def _read(source: Any, path: str, whole_keys: bool) -> Any:
    if not path:
        return source
    if whole_keys:
        return source.get(path) if isinstance(source, dict) else None
    value = source
    for part in _parts(path):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and re.fullmatch(r"-?\d+", part) and -len(value) <= int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def _parts(path: str) -> list[str]:
    """``path``'s keys, split at each ``.`` that no ``\\`` escapes, as the runtime escapes a key's ``.`` and ``\\``."""
    parts, part, chars = [], [], iter(path)
    for char in chars:
        if char == "\\":
            part.append(next(chars, ""))
        elif char == ".":
            parts.append("".join(part))
            part = []
        else:
            part.append(char)
    return [*parts, "".join(part)]


def _reduce(value: Any, how: str) -> Any:
    if how == "bool":
        return bool(value)
    if how == "keys":
        return sorted(value) if isinstance(value, dict) else []
    return len(value) if isinstance(value, (dict, list, str)) else 0


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "contains":
        return actual is not None and expected in actual
    if actual is None or expected is None:
        return False
    return {"ge": actual >= expected, "gt": actual > expected, "le": actual <= expected, "lt": actual < expected}[op]


class WebsiteStateVerifierTaskStep(TaskStep):
    """Runs ``checks`` against the state the websites of the deployed env ``env_id``, a website env or a multi env,
    report through their cards' ``website/v1`` ``state``, one result row each, and scores them with ``score_aggregator``. The sites checks name are
    checked before a run against ``env_version``, the version the task deploys, when it's given."""

    type: ClassVar[str] = "website_state_verifier"
    entity_refs = (EntityRef.env("env_id", version_field="env_version"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, checks: list[StateCheck | dict],
                 score_aggregator: Optional[ScoreAggregator | str] = None, verifier_id: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True,
                 env_version: Optional[int] = None):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.env_version = env_version
        self.checks = [check if isinstance(check, StateCheck) else StateCheck(**check) for check in checks]
        self.score_aggregator = ScoreAggregator(score_aggregator) if isinstance(score_aggregator, str) else (
            score_aggregator or ScoreAggregator.ALL_PASS)
        self.verifier_id = verifier_id or uuid.uuid4().hex
        if problems := [problem for check in self.checks for problem in check.problems()]:
            raise ValueError(f"website_state_verifier {id!r}: " + "; ".join(problems))

    def to_dict(self) -> dict:
        return {**super().to_dict(), "env_id": self.env_id, "env_version": self.env_version,
                "checks": [asdict(check) for check in self.checks], "score_aggregator": self.score_aggregator.value,
                "verifier_id": self.verifier_id}

    @classmethod
    def from_dict(cls, data: dict) -> WebsiteStateVerifierTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], env_version=data.get("env_version"),
                   checks=data["checks"], score_aggregator=data.get("score_aggregator"),
                   verifier_id=data.get("verifier_id"))

    def preflight(self) -> list[str]:
        """The env ``env_id`` when it holds no website, and the sites checks name that it doesn't serve, read from
        ``env_version``; with no ``env_version`` the sites are left to the run, which reads the version it deploys."""
        try:
            env = Env.get(self.env_id, self.env_version)
        except NotFoundError:
            return []
        if not (sites := _sites(env)):
            return [f"website_state_verifier {self.id!r}: {self.env_id} is a {env.type} env with no website; name a "
                    "website env or a multi env of websites"]
        if self.env_version is None:
            return []
        if unknown := sorted({check.site for check in self.checks} - sites):
            return [f"website_state_verifier {self.id!r}: {', '.join(map(repr, unknown))} "
                    f"{'is not a site' if len(unknown) == 1 else 'are not sites'} of {self.env_id} "
                    f"v{self.env_version}, whose sites are {sorted(sites)}"]
        return []

    @classmethod
    def put(cls, **kwargs: Any) -> WebsiteStateVerifierTaskStep:
        """Store the step once its preflight passes. Refused, naming every problem, before anything is stored."""
        from agent_env.task_step.store import get_task_step_store

        kwargs.setdefault("version", None)
        retry_config = kwargs.pop("retry_config", None)
        step = cls(**kwargs)
        if problems := step.preflight():
            raise ValueError("; ".join(problems))
        if retry_config is not None:
            attach_retry_config(step, {"retry_config": retry_config})
        return get_task_step_store().put_document(step)

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        wanted = sorted({(check.site, check.sid) for check in self.checks})
        card, address = await _card(deployed)
        if card is None:
            sessions = await self._definition_states(deployed, wanted)
        else:
            sessions = await self._card_states(deployed, card, address, wanted)
        results = [check.result(sessions) for check in self.checks]
        score = aggregate_score(results, self.score_aggregator)
        context.metadata.setdefault("verifications", {})[self.verifier_id] = {"results": results, "score": score}
        logger.info(f"Verification '{self.verifier_id}' complete: {len(results)} checks, score={score}")
        return context


    async def _card_states(self, deployed: Any, card: dict, address: str, wanted: list[tuple[str, str]]) -> dict:
        """Each wanted session's state, by site and sid, as the card's sites give it: ``website/v1``'s ``state``, else,
        deprecated, ``GET`` of the site's state path. A site the env definition names that the card doesn't list as a
        website, one serving no card, is read as a record with no card is (``_definition_states``), deprecated."""
        sites = website_sites(card)
        listed = {child.name for child in sites}
        if unlisted := [(site, sid) for site, sid in wanted if site not in listed]:
            defined = _sites(Env.get(deployed.env_id, deployed.env_version))
            if missing := sorted({site for site, _ in unlisted} - defined):
                raise RuntimeError(f"env '{self.env_id}' serves no website {missing[0]!r}; its sites are "
                                   f"{sorted(listed | defined)}")
        sessions: dict[str, dict] = await self._definition_states(deployed, unlisted) if unlisted else {}
        async with httpx.AsyncClient() as client:
            for site, sid in wanted:
                if site not in listed:
                    continue
                child = self._site(sites, site)
                state = await website_state(address, child.card, sid=sid, timeout=GO_TIMEOUT_S,
                                            headers=deployed.control_headers_for(address))
                if state is not None:
                    sessions.setdefault(site, {})[sid] = state.model_dump()
                    continue
                path = _state_path(child)
                logger.warning(f"website {site!r} of env '{self.env_id}' has no {WEBSITE_EXTENSION_URI} state on its "
                               f"card, so its state is read at {path}?sid=, which is deprecated")
                sessions.setdefault(site, {})[sid] = await _get_state(client, f"{child.base_url(address)}{path}", sid)
        return sessions

    def _site(self, sites: list[ChildEnv], name: str) -> ChildEnv:
        found = [child for child in sites if child.name == name]
        if not found:
            raise RuntimeError(f"env '{self.env_id}' serves no website {name!r}; its sites are "
                               f"{sorted(str(child.name) for child in sites)}")
        if len(found) > 1:
            raise RuntimeError(f"env '{self.env_id}' serves {len(found)} websites named {name!r} "
                               f"({', '.join(str(child.url) for child in found)}), which a check can't tell apart")
        return found[0]

    async def _definition_states(self, deployed: Any, wanted: list[tuple[str, str]]) -> dict:
        """Each wanted session's state for a site with no card to read: each site the env definition names, at the
        gateway's ``/svc/<name>/go?sid=``, deprecated."""
        sites = _sites(Env.get(deployed.env_id, deployed.env_version))
        gateway_url = gateway_url_of(deployed)
        logger.warning(f"env '{self.env_id}' has no card listing {sorted({site for site, _ in wanted})}, so they are "
                       f"the sites its definition names, each read at /svc/<name>{LEGACY_STATE_PATH}?sid=, which is "
                       "deprecated")
        sessions: dict[str, dict] = {}
        async with httpx.AsyncClient() as client:
            for site, sid in wanted:
                if site not in sites:
                    raise RuntimeError(f"env '{self.env_id}' serves no website {site!r}; its sites are {sorted(sites)}")
                base_url = legacy_protocol.environment_base_url(gateway_url, site, mcp=False)
                sessions.setdefault(site, {})[sid] = await _get_state(client, f"{base_url}{LEGACY_STATE_PATH}", sid)
        return sessions


async def _card(deployed: Any) -> tuple[Optional[dict], Optional[str]]:
    """The deployed env's card and the address its paths resolve against: the stored card, else the one it serves now;
    None for a record with no card URL, which nothing can be read at."""
    address = deployed.environment_url
    if not address:
        return None, None
    if deployed.environment_card:
        return deployed.environment_card, address
    return await protocol_v1.get_card(address), address


def _state_path(site: ChildEnv) -> str:
    """Where a site with no ``state`` op serves its state: its ``website/v1`` ``params.state.path``, else ``/go``."""
    state = (protocol_v1.extension_params(site.card, WEBSITE_EXTENSION_URI)).get("state")
    path = state.get("path") if isinstance(state, dict) else None
    return path if isinstance(path, str) and path.startswith("/") else LEGACY_STATE_PATH


async def _get_state(client: httpx.AsyncClient, url: str, sid: str) -> Any:
    response = await client.get(url, params={"sid": sid}, timeout=GO_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def _sites(env: Env) -> set[str]:
    """The environment_names of the website envs ``env`` is or holds."""
    from agent_env.env.envs.multi_env import MultiEnv
    from agent_env.env.envs.website import WebsiteEnv

    if isinstance(env, WebsiteEnv):
        return {env.environment_name}
    return {child.environment_name for child in env.website_envs} if isinstance(env, MultiEnv) else set()
