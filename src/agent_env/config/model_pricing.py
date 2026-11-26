"""Prices for model usage, from the price map the installed LiteLLM bundles, and the one rule that turns usage into a
cost (``price``). A cost keeps its status, its source and the unit prices it used, so a LiteLLM upgrade never changes a
cost already recorded."""

from __future__ import annotations

import functools
import importlib.metadata
import json
import logging
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional

logger = logging.getLogger(__name__)

PER_MILLION = 1_000_000
LITELLM = "litellm"
# The map a LiteLLM release ships; LiteLLM itself loads a newer one from the network at import unless told not to.
_MAP_FILE = "litellm/model_prices_and_context_window_backup.json"
_MAP_KEYS = {"input": "input_cost_per_token", "output": "output_cost_per_token",
             "cache_read": "cache_read_input_token_cost", "cache_write": "cache_creation_input_token_cost",
             "cache_write_1h": "cache_creation_input_token_cost_above_1hr"}
# A long-context tier's rate: one of those keys, then the prompt size past which a call is charged it, as
# ``_above_272k_tokens`` or ``_above_272000_tokens``. Service-tier and batch rates end otherwise, so they never match.
_TIER_KEY = re.compile(rf"^({'|'.join(map(re.escape, _MAP_KEYS.values()))})_above_(\d+)(k?)_tokens$")


class CostStatus(StrEnum):
    """Where a cost came from: a figure the agent or the endpoint reported, LiteLLM's price map, or neither."""

    REPORTED = "reported"
    PRICED = "priced"
    UNPRICED = "unpriced"


class ReportedBy(StrEnum):
    """Who reports a cost, as ATIF's ``cost_source`` names them: an agent's harness, whose figure comes after the map's
    price, or the endpoint that billed the call (a LiteLLM proxy's ``x-litellm-response-cost``, a judge's), whose
    figure comes before it."""

    HARNESS = "harness"
    PROVIDER = "provider"


class Assumed(StrEnum):
    """A guess a priced cost made. Most make it an upper bound; those that ``errs_low``, a lower bound: a sum of calls
    one of which may have passed a long-context tier, priced at the base rates, and cache writes that may have been
    kept an hour, priced at the five-minute rate."""

    CACHE_READ_RATE = "cache_read_rate"
    CACHE_WRITE_RATE = "cache_write_rate"
    INPUT_EXCLUDES_CACHE = "input_excludes_cache"
    BELOW_CONTEXT_TIER = "below_context_tier"
    CACHE_WRITE_5M = "cache_write_5m"

    @property
    def errs_low(self) -> bool:
        return self in (Assumed.BELOW_CONTEXT_TIER, Assumed.CACHE_WRITE_5M)


class Unpriced(StrEnum):
    """Why a cost has no figure: no model, no price in the map for the model, or no input and output token counts to
    price."""

    NO_MODEL = "no_model"
    NO_PRICE = "no_price"
    NO_TOKENS = "no_tokens"


@dataclass(frozen=True)
class ContextTier:
    """The rates a call whose prompt, cache included, is over ``threshold`` tokens is charged; a rate the map doesn't
    give for the tier stays at the base rate."""

    threshold: int
    input: Optional[float] = None
    output: Optional[float] = None
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None
    cache_write_1h: Optional[float] = None


@dataclass(frozen=True)
class ModelPrice:
    """A model's prices in USD per million tokens; a cache rate None when the map gives none. ``cache_write_1h`` is
    the rate of cache writes kept an hour, and ``tiers`` the rates past each prompt size, lowest first."""

    input: float
    output: float
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None
    cache_write_1h: Optional[float] = None
    tiers: tuple[ContextTier, ...] = ()

    def past(self, tier: ContextTier) -> ModelPrice:
        """The rates a call past ``tier`` is charged: the tier's own, else these. A call past several tiers takes each
        in turn, lowest first, so a rate a higher tier lacks stays at the one below it, as LiteLLM bills it."""
        return ModelPrice(*(own if (own := getattr(tier, rate)) is not None else getattr(self, rate)
                            for rate in _MAP_KEYS))


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
                entries[name] = ModelPrice(**rates, tiers=_tiers(info))
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


def _tiers(info: Mapping[str, Any]) -> tuple[ContextTier, ...]:
    """A map entry's long-context tiers, lowest first: each prompt size one of its tier keys names, with that size's
    rates."""
    rate_of = {key: rate for rate, key in _MAP_KEYS.items()}
    by_threshold: dict[int, dict[str, float]] = {}
    for key, value in info.items():
        if (match := _TIER_KEY.match(key)) and (rate := _rate(value)) is not None:
            threshold = int(match[2]) * (1000 if match[3] else 1)
            by_threshold.setdefault(threshold, {})[rate_of[match[1]]] = rate
    return tuple(ContextTier(threshold, **rates) for threshold, rates in sorted(by_threshold.items()))


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
    outside its input, an upper bound. ``cache_write_1h`` are the writes records said were kept an hour, and
    ``unsplit_write`` those of records that didn't say how long theirs were kept. ``prompt`` is the largest record's
    prompt, cache included, which decides whether a call may have passed a long-context tier, and ``one_call`` says the
    tokens are one model call's, so a tier prices them exactly."""

    uncached_input: int
    output: int
    cache_read: int = 0
    cache_write: int = 0
    assumed_uncached: bool = False
    cache_write_1h: int = 0
    unsplit_write: int = 0
    prompt: int = 0
    one_call: bool = False

    @classmethod
    def of(cls, usage: Mapping[str, Any], *, one_call: bool = False) -> Optional[Tokens]:
        """``usage``'s tokens (``Usage.to_dict``'s keys); None without input and output counts.
        ``input_includes_cache`` says whether ``input_tokens`` counts the cache tokens. Unknown, it's taken not to (an
        upper bound) unless the counts prove it, input being fewer than the cache tokens. Its writes kept an hour are
        ``provider_details.cache_creation``'s (``cache_write_split``), and the writes that split doesn't cover, all of
        them without one, are ``unsplit_write``."""
        inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
        if inputs is None or outputs is None:
            return None
        read, write = usage.get("cache_read_tokens") or 0, usage.get("cache_write_tokens") or 0
        cached = read + write
        details = usage.get("provider_details")
        five, hour = cache_write_split(details.get("cache_creation") if isinstance(details, Mapping) else None) or (0, 0)
        hour = min(hour, write)
        written = (hour, write - min(write, five + hour))
        includes = usage.get("input_includes_cache")
        if includes is True and inputs >= cached:
            return cls(inputs - cached, outputs, read, write, False, *written, inputs, one_call)
        return cls(inputs, outputs, read, write, includes is None and 0 < cached <= inputs, *written,
                   inputs + cached, one_call)

    def __add__(self, other: Tokens) -> Tokens:
        return Tokens(self.uncached_input + other.uncached_input, self.output + other.output,
                      self.cache_read + other.cache_read, self.cache_write + other.cache_write,
                      self.assumed_uncached or other.assumed_uncached, self.cache_write_1h + other.cache_write_1h,
                      self.unsplit_write + other.unsplit_write, max(self.prompt, other.prompt))


def cache_write_split(split: object) -> Optional[tuple[int, int]]:
    """A usage's split of its cache writes by how long they're kept, as (five-minute, one-hour) writes, from Anthropic's
    ``ephemeral_5m_input_tokens`` and ``ephemeral_1h_input_tokens``. A count that isn't a non-negative integer is
    unknown, and the split None when neither count is known."""
    if not isinstance(split, Mapping):
        return None
    five, hour = (_count(split.get(key)) for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"))
    return None if five is None and hour is None else (five or 0, hour or 0)


def _count(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass(frozen=True)
class Cost:
    """A usage record's cost: its figure (None when unpriced), where it came from, and the agent's or endpoint's own
    figure kept beside it; a priced one also has the uncached input, the cache writes charged at the hour's rate and
    the unit prices it was charged at, the assumed rates filled in, and what was assumed (``Assumed``)."""

    usd: Optional[float]
    status: CostStatus
    source: Optional[str]
    reported_usd: Optional[float]
    uncached_input: Optional[int] = None
    unit_prices: Optional[dict[str, float]] = None
    assumed: tuple[Assumed, ...] = ()
    unpriced: Optional[Unpriced] = None
    cache_write_1h: Optional[int] = None

    #: Every key ``fields`` can write, so a record built from them keeps a call's whole cost.
    KEYS: ClassVar[tuple[str, ...]] = ("cost_usd", "cost_status", "cost_source", "reported_cost_usd",
                                       "uncached_input_tokens", "unit_prices", "cost_assumed", "unpriced_reason",
                                       "cache_write_1h_tokens")

    def fields(self) -> dict[str, Any]:
        """The cost as a usage entry's keys; the uncached input and unit prices only when it has them, as a price from
        the map does and a sum of several models' costs doesn't."""
        record: dict[str, Any] = {"cost_usd": self.usd, "cost_status": self.status.value, "cost_source": self.source,
                                  "reported_cost_usd": self.reported_usd}
        if self.uncached_input is not None:
            record["uncached_input_tokens"] = self.uncached_input
        if self.unit_prices is not None:
            record["unit_prices"] = self.unit_prices
        if self.assumed:
            record["cost_assumed"] = [assumed.value for assumed in self.assumed]
        if self.unpriced is not None:
            record["unpriced_reason"] = self.unpriced.value
        if self.cache_write_1h:
            record["cache_write_1h_tokens"] = self.cache_write_1h
        return record


def price(
    tokens: Optional[Tokens],
    reported: Optional[float],
    *,
    model: Optional[str],
    prices: PriceTable,
    reported_by: ReportedBy,
) -> Cost:
    """The cost of ``tokens`` on ``model``, given the ``reported`` figure. The map's price comes before a harness's
    figure, which is kept beside it, and after the figure the endpoint billed. Neither gives ``unpriced``."""
    listed = prices.lookup(model) if model else None
    as_reported = Cost(reported, CostStatus.REPORTED, reported_by.value, reported) if reported is not None else None
    as_priced = _priced(tokens, listed, prices.source, reported) if listed and tokens else None
    first, second = ((as_reported, as_priced) if reported_by is ReportedBy.PROVIDER
                     else (as_priced, as_reported))
    chosen = first or second
    if chosen is not None:
        return chosen
    why = Unpriced.NO_MODEL if not model else Unpriced.NO_PRICE if listed is None else Unpriced.NO_TOKENS
    return Cost(None, CostStatus.UNPRICED, None, None, unpriced=why)


def _priced(tokens: Tokens, rates: ModelPrice, source: str, reported: Optional[float]) -> Cost:
    """``tokens`` at ``rates``. A call past a long-context tier is charged the tier's rates; a sum of calls one of which
    may have passed one, at the base rates, marked. Writes kept an hour are charged the hour's rate, and writes no
    record split, the five-minute rate, marked when the hour's is higher. A cache rate the map lacks is taken as the
    input rate, marked."""
    assumed = []
    passed = [tier for tier in rates.tiers if tokens.prompt > tier.threshold]
    if passed and tokens.one_call:
        for tier in passed:
            rates = rates.past(tier)
    elif passed:
        assumed.append(Assumed.BELOW_CONTEXT_TIER)
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
    hour_rate = rates.cache_write_1h if rates.cache_write_1h is not None else write_rate
    if tokens.unsplit_write and hour_rate > write_rate:
        assumed.append(Assumed.CACHE_WRITE_5M)
    usd = (tokens.uncached_input * rates.input + tokens.output * rates.output + tokens.cache_read * read_rate
           + (tokens.cache_write - tokens.cache_write_1h) * write_rate + tokens.cache_write_1h * hour_rate) / PER_MILLION
    unit = {"input": rates.input, "output": rates.output, "cache_read": read_rate, "cache_write": write_rate}
    if tokens.cache_write_1h:
        unit["cache_write_1h"] = hour_rate
    return Cost(usd, CostStatus.PRICED, source, reported, tokens.uncached_input, unit, tuple(assumed),
                cache_write_1h=tokens.cache_write_1h or None)

