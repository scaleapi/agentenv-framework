"""Per-agent model budgets: a key of its own for each agent whose step sets ``model_budget_usd``.

An agent otherwise gets the shared ``[model] api_key``, which nothing limits. An agent with a budget gets a key made
for it on the configured endpoint, a LiteLLM proxy, with ``[model] admin_api_key``, before its sandbox exists. The
budget is the key's ``max_budget``, so the proxy refuses the agent's model calls once the key has spent it. The key
belongs to no user, so the agent can't make, change or delete keys with it, and so can't lift its own budget; it
expires an hour after the agent's TTL.

A run records each budget in ``context.metadata["model_budgets"]``, by agent name: the budget, the key's alias and
LiteLLM's id for it (the key's hash, which authenticates nothing), never the key itself. A retry that rolls a
deployment back moves its budget into the ``retry_resets`` marker, as it does the deployment's sandbox. ``teardown_run``
blocks every key the run made, so one an agent leaked is no use after the run. LiteLLM keeps it, and what it spent:
LiteLLM writes spend to its database every 10 to 15 seconds, so a figure read at teardown could miss the agent's last
calls, while the blocked key's own count takes them in.
"""

from __future__ import annotations

import logging
import math
import re
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

import httpx

from agent_env.attribution import RUN_ID_KEY, Attribution
from agent_env.config import get_config

if TYPE_CHECKING:
    from agent_env.task_step.context import TaskStepContext

logger = logging.getLogger(__name__)

BUDGETS_KEY = "model_budgets"
#: Where a retry rolls back a deployment's budget, in its ``retry_resets`` marker (``agent_env.task.task``).
ROLLED_BACK_KEY = "rolled_back_model_budgets"
#: How long a key outlives its agent's TTL: its clock starts before the agent's sandbox is made, which can take a
#: while (an image build, say). A teardown blocks it long before.
KEY_LIFETIME_MARGIN_SECONDS = 3600
REQUEST_TIMEOUT_SECONDS = 30
# A LiteLLM key alias is letters, digits and _-./@, and starts and ends with a letter or digit.
_ALIAS_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_ALIAS_PART_LENGTH = 100
_ERROR_LENGTH = 300


def check_budget(value: Any, *, step_id: str) -> Optional[float]:
    """``value`` as a budget in US dollars: None for none, else a finite positive number."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Step '{step_id}': model_budget_usd must be a positive number of US dollars, not {value!r}")
    return float(value)


@dataclass(frozen=True)
class BudgetKey:
    """A key made for one agent. ``key`` goes to the agent and nowhere else; ``key_id`` is LiteLLM's hash of it, which
    names it in LiteLLM's key API and authenticates nothing."""

    key: str = field(repr=False)
    key_id: str
    alias: str
    max_usd: float

    async def close(self) -> None:
        """Delete the key, for a step that fails before its agent is up. A failure is logged: the key expires anyway."""
        try:
            await delete_key(self.key_id)
        except Exception as e:
            logger.warning("Could not delete model budget key %s, which expires on its own: %s", self.alias, e)


async def make_key(
    *, agent_name: str, max_usd: float, ttl_seconds: int, attribution: Attribution, base_url: Optional[str] = None,
) -> BudgetKey:
    """Make ``agent_name``'s key, which may spend ``max_usd`` US dollars, on the configured endpoint. ``base_url`` is
    the endpoint the agent will call, None for the configured one; any other is refused, since the key is good only
    where it is made, and the admin key goes nowhere else."""
    configured = get_config().get_litellm_base_url()
    if base_url and base_url.rstrip("/") != configured.rstrip("/"):
        raise ValueError(
            f"Agent '{agent_name}' has a model budget, so it has to call the configured model endpoint "
            f"([model] base_url), where its key is made, not {_host(base_url)}"
        )
    parts = ("agentenv", attribution.get(RUN_ID_KEY), agent_name, secrets.token_hex(4))  # an alias is unique in LiteLLM
    alias = "/".join(filter(None, (_alias_part(part) for part in parts)))
    made = await _admin(
        "POST", "/key/generate", f"make a ${max_usd:g} model budget key for agent '{agent_name}'",
        json={
            "max_budget": max_usd,
            "duration": f"{ttl_seconds + KEY_LIFETIME_MARGIN_SECONDS}s",
            "key_alias": alias,
            "metadata": {**attribution, "agent_name": agent_name},
        },
    )
    key, key_id = made.get("key"), made.get("token_id") or made.get("token")
    if not isinstance(key, str) or not isinstance(key_id, str):
        raise RuntimeError(f"{_host(configured)} answered /key/generate for agent '{agent_name}' without a key")
    budget_key = BudgetKey(key=key, key_id=key_id, alias=alias, max_usd=max_usd)
    if made.get("user_id") is not None:
        # A user's key acts with the user's role, which can let it make keys of its own, each with a budget of its own.
        await budget_key.close()
        raise RuntimeError(
            f"[model] admin_api_key made agent '{agent_name}' a key that belongs to a user, which can let the agent "
            "make itself more keys: use the proxy's master key or a proxy admin's, whose keys belong to no user"
        )
    logger.info("Made model budget key %s for agent '%s': $%g", alias, agent_name, max_usd)
    return budget_key


def record(context: TaskStepContext, agent_name: str, key: BudgetKey) -> None:
    """Note ``agent_name``'s budget in ``context``, for ``revoke`` to find; never the key itself."""
    context.metadata.setdefault(BUDGETS_KEY, {})[agent_name] = {
        "max_usd": key.max_usd, "key_alias": key.alias, "key_id": key.key_id, "key_blocked": False,
    }


def rolled_back(before: dict, after: dict) -> dict:
    """The budgets in the stored context ``before`` an undo whose key the context ``after`` it no longer records, by
    agent name, for the retry's ``retry_resets`` marker: the undo strips them, and ``revoke`` must still find them."""
    kept = {budget.get("key_id") for budget in ((after.get("metadata") or {}).get(BUDGETS_KEY) or {}).values()}
    return {name: budget for name, budget in ((before.get("metadata") or {}).get(BUDGETS_KEY) or {}).items()
            if budget.get("key_id") not in kept}


def _recorded(context: TaskStepContext) -> list[tuple[str, dict]]:
    """Every budget ``context`` records: its agents', and those a retry rolled back."""
    budgets = list((context.metadata.get(BUDGETS_KEY) or {}).items())
    for reset in context.metadata.get("retry_resets") or []:
        budgets.extend((reset.get(ROLLED_BACK_KEY) or {}).items())
    return budgets


async def revoke(context: TaskStepContext) -> None:
    """Block each budget key ``context`` records, a rolled-back deployment's too, that isn't blocked yet. Best-effort,
    as a teardown is: a failure is logged, and the key expires on its own."""
    for agent_name, budget in _recorded(context):
        if not budget.get("key_id") or budget.get("key_blocked"):
            continue
        try:  # a key already gone is as good as blocked
            await _admin("POST", "/key/block", f"block the model budget key of agent '{agent_name}'",
                         json={"key": budget["key_id"]}, gone_ok=True)
        except Exception as e:
            logger.warning("%s; it expires on its own", e)
            continue
        budget["key_blocked"] = True
        logger.info("Blocked the model budget key of agent '%s' (%s); LiteLLM keeps what it spent", agent_name,
                    budget.get("key_alias"))


async def delete_key(key_id: str) -> None:
    """Delete the key ``key_id`` names; one already gone is fine."""
    await _admin("POST", "/key/delete", "delete a model budget key", json={"keys": [key_id]}, gone_ok=True)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)


async def _admin(
    method: str, path: str, what: str, *, json: Optional[dict] = None, params: Optional[dict] = None,
    gone_ok: bool = False,
) -> dict:
    """A call to the configured endpoint's key API with ``[model] admin_api_key``, and ``[model.headers]``, which go
    with every call to it."""
    config = get_config()
    base = config.get_litellm_base_url().rstrip("/")
    headers = {name: value for name, value in config.get_model_headers(base).items() if name.lower() != "authorization"}
    headers["Authorization"] = f"Bearer {config.get_model_admin_api_key()}"
    try:
        async with _client() as client:
            response = await client.request(method, base + path, json=json, params=params, headers=headers)
    except httpx.HTTPError as e:
        raise RuntimeError(f"Could not {what}: {_host(base)} did not answer ({type(e).__name__}: {e})") from e
    if gone_ok and response.status_code == httpx.codes.NOT_FOUND:
        return {}
    if response.status_code >= 400:
        raise RuntimeError(f"Could not {what}: {_host(base)} answered {response.status_code}: {_error(response)}")
    try:
        answer = response.json()
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        raise RuntimeError(f"Could not {what}: {_host(base)} answered {path} with something other than a JSON object")
    return answer


def _error(response: httpx.Response) -> str:
    """LiteLLM's message for a refused call, else the start of its body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:_ERROR_LENGTH]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
    else:
        message = body.get("detail") if isinstance(body, dict) else None
    return str(message or body)[:_ERROR_LENGTH]


def _alias_part(part: Optional[str]) -> str:
    return _ALIAS_UNSAFE.sub("-", part or "")[:_ALIAS_PART_LENGTH].strip("-_.")


def _host(url: str) -> str:
    return urlparse(url).hostname or url
