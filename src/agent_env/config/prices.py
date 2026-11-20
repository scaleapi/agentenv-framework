"""Prices for model usage, from the price map the installed LiteLLM bundles, and the one rule that turns usage into a
cost (``price``), with ``unpriced``, which names the roles of a plan the map can't price. A cost keeps its status, its
source and the unit prices it used, so a LiteLLM upgrade never changes a cost already recorded."""

from __future__ import annotations

import functools
import importlib.metadata
import json
import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

PER_MILLION = 1_000_000
LITELLM = "litellm"
# The map a LiteLLM release ships; LiteLLM itself loads a newer one from the network at import unless told not to.
_MAP_FILE = "litellm/model_prices_and_context_window_backup.json"
_MAP_KEYS = {"input": "input_cost_per_token", "output": "output_cost_per_token",
             "cache_read": "cache_read_input_token_cost", "cache_write": "cache_creation_input_token_cost"}


class CostStatus(StrEnum):
    """Where a cost came from: a figure the agent or the endpoint reported, LiteLLM's price map, or neither."""

    REPORTED = "reported"
    PRICED = "priced"
    UNPRICED = "unpriced"


class ReportedBy(StrEnum):
    """Who reported a ``reported`` cost, as ATIF's ``cost_source`` names them: the agent's harness, or the endpoint that
    billed the call (a LiteLLM proxy's ``x-litellm-response-cost``)."""

    HARNESS = "harness"
    PROVIDER = "provider"


class Assumed(StrEnum):
    """A guess that makes a priced cost an upper bound."""

    CACHE_READ_RATE = "cache_read_rate"
    CACHE_WRITE_RATE = "cache_write_rate"
    INPUT_EXCLUDES_CACHE = "input_excludes_cache"


class Unpriced(StrEnum):
    """Why a cost has no figure: no model, a URL other than the configured endpoint (or one not known to be it), no
    price in the map for the model, or no input and output token counts to price."""

    NO_MODEL = "no_model"
    OTHER_ENDPOINT = "other_endpoint"
    NO_PRICE = "no_price"
    NO_TOKENS = "no_tokens"


@dataclass(frozen=True)
class ModelPrice:
    """A model's prices in USD per million tokens; a cache rate None when the map gives none."""

    input: float
    output: float
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None


@dataclass(frozen=True)
class PriceTable:
    """Each model's prices, keyed by its name, and the ``source`` a cost priced from them names."""

    entries: Mapping[str, ModelPrice] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def from_cost_map(cls, cost_map: Mapping[str, Any], source: str) -> PriceTable:
        """LiteLLM's model cost map, in USD per token, as prices per million tokens: each entry with an input and an
        output price, and its cache prices when it has them."""
        entries = {}
        for name, info in cost_map.items():
            rates = {rate: _rate(info.get(key)) for rate, key in _MAP_KEYS.items()} if isinstance(info, dict) else {}
            if rates.get("input") is not None and rates.get("output") is not None:
                entries[name] = ModelPrice(**rates)
        return cls(entries, source)

    def lookup(self, model: str) -> Optional[ModelPrice]:
        """``model``'s prices: the entry named as called, else the one named without its first ``/`` segment, so
        ``openai/gpt-4o-mini`` finds ``gpt-4o-mini``; None, never a zero price, when there's neither."""
        if model in self.entries:
            return self.entries[model]
        _, slash, name = model.partition("/")
        return self.entries.get(name) if slash else None


@functools.cache
def litellm_prices() -> PriceTable:
    """The price map the installed LiteLLM bundles, read without importing LiteLLM, so a cost's source,
    ``litellm@<version>``, names the one map it came from. A map that can't be read prices nothing, with a warning,
    rather than failing the step whose usage is being recorded."""
    try:
        dist = importlib.metadata.distribution(LITELLM)
        cost_map = json.loads(Path(dist.locate_file(_MAP_FILE)).read_text())
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError) as error:
        logger.warning("Usage stays unpriced, since LiteLLM's price map can't be read: %s", error)
        return PriceTable()
    return PriceTable.from_cost_map(cost_map, f"{LITELLM}@{dist.version}")


def _rate(per_token: object) -> Optional[float]:
    """A per-token price as USD per million tokens; None for anything but a finite, non-negative number."""
    if isinstance(per_token, bool) or not isinstance(per_token, (int, float)):
        return None
    return round(per_token * PER_MILLION, 9) if math.isfinite(per_token) and per_token >= 0 else None


@dataclass(frozen=True)
class Tokens:
    """The tokens a usage record is charged for, each at its own rate: its input outside the cache, its output, and
    its cache reads and writes. They add up across records, so a step's turns are each read by their own
    ``input_includes_cache`` before they're summed; ``assumed_uncached`` says some record's cache tokens were taken as
    outside its input, an upper bound."""

    uncached_input: int
    output: int
    cache_read: int = 0
    cache_write: int = 0
    assumed_uncached: bool = False

    @classmethod
    def of(cls, usage: Mapping[str, Any]) -> Optional[Tokens]:
        """``usage``'s tokens (``Usage.to_dict``'s keys); None without input and output counts.
        ``input_includes_cache`` says whether ``input_tokens`` counts the cache tokens. Unknown, it's taken not to (an
        upper bound) unless the counts prove it, input being fewer than the cache tokens."""
        inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
        if inputs is None or outputs is None:
            return None
        read, write = usage.get("cache_read_tokens") or 0, usage.get("cache_write_tokens") or 0
        cached = read + write
        includes = usage.get("input_includes_cache")
        if includes is True and inputs >= cached:
            return cls(inputs - cached, outputs, read, write)
        return cls(inputs, outputs, read, write, includes is None and 0 < cached <= inputs)

    def __add__(self, other: Tokens) -> Tokens:
        return Tokens(self.uncached_input + other.uncached_input, self.output + other.output,
                      self.cache_read + other.cache_read, self.cache_write + other.cache_write,
                      self.assumed_uncached or other.assumed_uncached)


@dataclass(frozen=True)
class Cost:
    """A usage record's cost: its figure (None when unpriced), where it came from, and the agent's or endpoint's own
    figure kept beside it; a priced one also has the uncached input and the unit prices it was charged at, the assumed
    rates filled in, and what was assumed, which makes it an upper bound."""

    usd: Optional[float]
    status: CostStatus
    source: Optional[str]
    reported_usd: Optional[float]
    uncached_input: Optional[int] = None
    unit_prices: Optional[dict[str, float]] = None
    assumed: tuple[Assumed, ...] = ()
    unpriced: Optional[Unpriced] = None

    def fields(self) -> dict[str, Any]:
        """The cost as a usage entry's keys."""
        record: dict[str, Any] = {"cost_usd": self.usd, "cost_status": self.status.value, "cost_source": self.source,
                                  "reported_cost_usd": self.reported_usd}
        if self.status is CostStatus.PRICED:
            record["uncached_input_tokens"] = self.uncached_input
            record["unit_prices"] = self.unit_prices
        if self.assumed:
            record["cost_assumed"] = [assumed.value for assumed in self.assumed]
        if self.unpriced is not None:
            record["unpriced_reason"] = self.unpriced.value
        return record


def price(
    tokens: Optional[Tokens],
    reported: Optional[float],
    *,
    model: Optional[str],
    prices: PriceTable,
    on_endpoint: bool,
    reported_first: bool,
    reported_by: ReportedBy,
) -> Cost:
    """The cost of ``tokens`` on ``model``, given the ``reported`` figure. ``prices`` price them only for calls
    through the configured endpoint (``on_endpoint``), and their price comes before the reported figure, which is kept
    beside it, unless ``reported_first``: a judge's proxy-billed figure comes first. Neither gives ``unpriced``."""
    listed = prices.lookup(model) if model and on_endpoint else None
    as_reported = Cost(reported, CostStatus.REPORTED, reported_by.value, reported) if reported is not None else None
    as_priced = _priced(tokens, listed, prices.source, reported) if listed and tokens else None
    first, second = (as_reported, as_priced) if reported_first else (as_priced, as_reported)
    chosen = first or second
    if chosen is not None:
        return chosen
    why = (Unpriced.NO_MODEL if not model else Unpriced.OTHER_ENDPOINT if not on_endpoint
           else Unpriced.NO_PRICE if listed is None else Unpriced.NO_TOKENS)
    return Cost(None, CostStatus.UNPRICED, None, None, unpriced=why)


def _priced(tokens: Tokens, rates: ModelPrice, source: str, reported: Optional[float]) -> Cost:
    """``tokens`` at ``rates``, a cache rate the map lacks taken as the input rate (an upper bound, marked)."""
    assumed = []
    read_rate, write_rate = rates.cache_read, rates.cache_write
    if read_rate is None:
        read_rate = rates.input
        if tokens.cache_read:
            assumed.append(Assumed.CACHE_READ_RATE)
    if write_rate is None:
        write_rate = rates.input
        if tokens.cache_write:
            assumed.append(Assumed.CACHE_WRITE_RATE)
    if tokens.assumed_uncached:
        assumed.append(Assumed.INPUT_EXCLUDES_CACHE)
    usd = (tokens.uncached_input * rates.input + tokens.output * rates.output + tokens.cache_read * read_rate
           + tokens.cache_write * write_rate) / PER_MILLION
    unit = {"input": rates.input, "output": rates.output, "cache_read": read_rate, "cache_write": write_rate}
    return Cost(usd, CostStatus.PRICED, source, reported, tokens.uncached_input, unit, tuple(assumed))


@dataclass(frozen=True)
class PlannedModel:
    """A role's model in a plan, and whether its calls go through the configured endpoint."""

    role: str
    model: Optional[str]
    on_endpoint: bool = True


@dataclass(frozen=True)
class UnpricedModels:
    """The roles of a plan the prices can't price, each with why."""

    missing: tuple[tuple[PlannedModel, Unpriced], ...] = ()
    source: str = ""

    def message(self) -> str:
        """Each unpriced role and why, for a person to read."""
        return "\n".join(f"{planned.role}: {_WHY[why].format(model=planned.model, source=self.source)}"
                         for planned, why in self.missing)


_WHY = {
    Unpriced.NO_MODEL: "no model is set, so nothing can price it",
    Unpriced.OTHER_ENDPOINT: "{model} is called on a URL other than the configured endpoint, whose prices aren't known",
    Unpriced.NO_PRICE: "{model} isn't in LiteLLM's price map ({source})",
}


def unpriced(planned: Iterable[PlannedModel], prices: PriceTable) -> UnpricedModels:
    """The roles of a plan whose usage ``prices`` can't price: no model, a URL other than the configured endpoint, or
    a model the map lacks. A cost an agent reports isn't a price, since a budget must hold before the run."""
    missing = []
    for role in planned:
        if not role.model:
            missing.append((role, Unpriced.NO_MODEL))
        elif not role.on_endpoint:
            missing.append((role, Unpriced.OTHER_ENDPOINT))
        elif prices.lookup(role.model) is None:
            missing.append((role, Unpriced.NO_PRICE))
    return UnpricedModels(tuple(missing), prices.source)
