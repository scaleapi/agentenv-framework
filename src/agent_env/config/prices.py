"""The ``[model.prices]`` table and the one rule that turns usage into a cost (``price``), with ``unpriced``, which
names the models a plan would leave without a price. A cost keeps its status, its source and the unit prices it used,
so changing the table never changes a cost already recorded."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Optional

from agent_env.config.errors import ConfigError

PER_MILLION = 1_000_000
PRICES_SOURCE = "agentenv-prices"
_RATES = ("input", "output", "cache_read", "cache_write")
_REQUIRED = ("input", "output")
_VERSION_CHARS = 12
_EXAMPLE = '"gpt-4o-mini" = { input = 0.15, output = 0.6, cache_read = 0.075 }'


class CostStatus(StrEnum):
    """Where a cost came from: a figure the agent or the endpoint reported, the price table, or neither."""

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
    price in the table for the model, or no input and output token counts to price."""

    NO_MODEL = "no_model"
    OTHER_ENDPOINT = "other_endpoint"
    NO_PRICE = "no_price"
    NO_TOKENS = "no_tokens"


@dataclass(frozen=True)
class ModelPrice:
    """A model's prices in USD per million tokens; a cache rate None when the table gives none."""

    input: float
    output: float
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None


@dataclass(frozen=True)
class PriceTable:
    """``[model.prices]``: each model's prices, keyed by its name without a routing prefix."""

    entries: Mapping[str, ModelPrice] = field(default_factory=dict)

    @classmethod
    def from_section(cls, section: object) -> PriceTable:
        """Parse ``[model.prices]``, refusing an entry without ``input`` and ``output``, a rate that isn't a
        non-negative number, or a key that isn't a rate, naming the entry."""
        if not isinstance(section, dict):
            raise ConfigError(f"[model.prices] maps each model to its prices, such as {_EXAMPLE}")
        return cls({name: _model_price(name, rates) for name, rates in section.items()})

    @property
    def version(self) -> str:
        """A hash of the entries, so any change to a price gives a new version, and writing a price as ``1`` or
        ``1.0``, or the entries in another order, doesn't."""
        canonical = {name: {rate: float(value) for rate in _RATES if (value := getattr(price, rate)) is not None}
                     for name, price in self.entries.items()}
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())
        return digest.hexdigest()[:_VERSION_CHARS]

    @property
    def source(self) -> str:
        """What a priced cost names as its source: this table at its version."""
        return f"{PRICES_SOURCE}@{self.version}"

    def lookup(self, model: str) -> Optional[ModelPrice]:
        """``model``'s prices: the entry named as called, else the one named without its first ``/`` segment, so
        ``openai/gpt-4o-mini`` finds ``gpt-4o-mini``; None, never a zero price, when there's neither."""
        if model in self.entries:
            return self.entries[model]
        _, slash, name = model.partition("/")
        return self.entries.get(name) if slash else None


@dataclass(frozen=True)
class Cost:
    """A usage record's cost: its figure (None when unpriced), where it came from, and the agent's or endpoint's own
    figure kept beside it; a priced one also has the unit prices it used, the assumed ones filled in, and what was
    assumed, which makes it an upper bound."""

    usd: Optional[float]
    status: CostStatus
    source: Optional[str]
    reported_usd: Optional[float]
    unit_prices: Optional[dict[str, float]] = None
    assumed: tuple[Assumed, ...] = ()
    unpriced: Optional[Unpriced] = None

    def fields(self) -> dict[str, Any]:
        """The cost as a usage entry's keys."""
        record: dict[str, Any] = {"cost_usd": self.usd, "cost_status": self.status.value, "cost_source": self.source,
                                  "reported_cost_usd": self.reported_usd}
        if self.unit_prices is not None:
            record["unit_prices"] = self.unit_prices
        if self.assumed:
            record["cost_assumed"] = [assumed.value for assumed in self.assumed]
        if self.unpriced is not None:
            record["unpriced_reason"] = self.unpriced.value
        return record


def price(
    usage: Mapping[str, Any],
    *,
    model: Optional[str],
    prices: PriceTable,
    on_endpoint: bool,
    reported_first: bool,
    reported_by: ReportedBy,
) -> Cost:
    """The cost of ``usage`` (``Usage.to_dict``'s keys, ``cost_usd`` the reported figure) on ``model``. The table
    prices it only for calls through the configured endpoint (``on_endpoint``). The table's price comes first and the
    reported figure second, or the other way round with ``reported_first``: an agent's harness figure follows
    ``[model] prefer_reported_cost``, and a judge's proxy figure always comes first. Neither gives ``unpriced``."""
    reported = usage.get("cost_usd")
    listed = prices.lookup(model) if model and on_endpoint else None
    computed = _priced(usage, listed) if listed else None
    as_reported = (Cost(reported, CostStatus.REPORTED, reported_by.value, reported)
                   if reported is not None else None)
    as_priced = (Cost(computed[0], CostStatus.PRICED, prices.source, reported, computed[1], computed[2])
                 if computed else None)
    first, second = (as_reported, as_priced) if reported_first else (as_priced, as_reported)
    chosen = first or second
    if chosen is not None:
        return chosen
    why =(Unpriced.NO_MODEL if not model else Unpriced.OTHER_ENDPOINT if not on_endpoint
           else Unpriced.NO_PRICE if listed is None else Unpriced.NO_TOKENS)
    return Cost(None, CostStatus.UNPRICED, None, None, unpriced=why)


def _priced(usage: Mapping[str, Any], rates: ModelPrice) -> Optional[tuple[float, dict[str, float], tuple]]:
    """``usage`` at ``rates``: the cost, the unit prices used and what was assumed; None without input and output
    counts. Cached input is charged once, at its own rate: ``input_includes_cache`` says whether ``input_tokens``
    counts it. Unknown, it's taken not to (an upper bound, marked) unless the counts prove it, input being fewer than
    the cache tokens; and a cache rate the table lacks is the input rate (an upper bound, marked)."""
    inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
    if inputs is None or outputs is None:
        return None
    read, write = usage.get("cache_read_tokens") or 0, usage.get("cache_write_tokens") or 0
    assumed = []
    read_rate, write_rate = rates.cache_read, rates.cache_write
    if read_rate is None:
        read_rate = rates.input
        if read:
            assumed.append(Assumed.CACHE_READ_RATE)
    if write_rate is None:
        write_rate = rates.input
        if write:
            assumed.append(Assumed.CACHE_WRITE_RATE)
    cached = read + write
    includes = usage.get("input_includes_cache")
    uncached = inputs - cached if includes is True and inputs >= cached else inputs
    if includes is None and cached and inputs >= cached:
        assumed.append(Assumed.INPUT_EXCLUDES_CACHE)
    usd = (uncached * rates.input + outputs * rates.output + read * read_rate + write * write_rate) / PER_MILLION
    unit = {"input": rates.input, "output": rates.output, "cache_read": read_rate, "cache_write": write_rate}
    return usd, unit, tuple(assumed)


@dataclass(frozen=True)
class PlannedModel:
    """A role's model in a plan, and whether its calls go through the configured endpoint."""

    role: str
    model: Optional[str]
    on_endpoint: bool = True


@dataclass(frozen=True)
class UnpricedModels:
    """The roles of a plan the table can't price, each with why, and the priced models whose entry has no cache rate,
    whose cache tokens would be priced at the input rate."""

    missing: tuple[tuple[PlannedModel, Unpriced], ...] = ()
    no_cache_rates: tuple[str, ...] = ()

    def toml(self) -> str:
        """``[model.prices]`` lines for every model with no price, to fill in and add under that table."""
        names = dict.fromkeys(planned.model for planned, why in self.missing if why is Unpriced.NO_PRICE)
        return "\n".join(f"{json.dumps(name)} = {{ input = 0.0, output = 0.0 }}  # USD per million tokens"
                         for name in names)

    def message(self) -> str:
        """What a person reads: each unpriced role and why, the lines that would price the models the table lacks,
        and the cache rates worth adding."""
        lines = [f"{planned.role}: {_WHY[why].format(model=planned.model)}" for planned, why in self.missing]
        if toml := self.toml():
            lines += ["Add their prices under [model.prices] in .agentenv/config.toml:", toml]
        lines += [f"{model} has no cache_read or cache_write rate, so its cache tokens are priced at its input rate, "
                  "an upper bound; add them to its [model.prices] entry" for model in self.no_cache_rates]
        return "\n".join(lines)


_WHY = {
    Unpriced.NO_MODEL: "no model is set, so nothing can price it",
    Unpriced.OTHER_ENDPOINT: "{model} is called on a URL other than the configured endpoint, which [model.prices] "
                             "doesn't price",
    Unpriced.NO_PRICE: "{model} has no price in [model.prices]",
}


def unpriced(planned: Iterable[PlannedModel], prices: PriceTable) -> UnpricedModels:
    """The roles of a plan whose usage the table can't price: no model, a URL other than the configured endpoint, or a
    model the table lacks. A cost an agent reports isn't a price, since a budget must hold before it runs."""
    missing, no_cache = [], []
    for role in planned:
        listed = prices.lookup(role.model) if role.model and role.on_endpoint else None
        if listed is None:
            missing.append((role, Unpriced.NO_MODEL if not role.model
                            else Unpriced.OTHER_ENDPOINT if not role.on_endpoint else Unpriced.NO_PRICE))
        elif listed.cache_read is None or listed.cache_write is None:
            no_cache.append(role.model)
    return UnpricedModels(tuple(missing), tuple(dict.fromkeys(no_cache)))


def _model_price(name: object, rates: object) -> ModelPrice:
    if not isinstance(name, str) or not name:
        raise ConfigError(f"[model.prices] keys are model names, such as {_EXAMPLE}")
    if not isinstance(rates, dict):
        raise ConfigError(f"[model.prices] {name!r} is a table of USD per million tokens, such as {_EXAMPLE}")
    unknown = set(rates) - set(_RATES)
    if unknown:
        raise ConfigError(f"[model.prices] {name!r} has unknown keys {sorted(unknown)}; allowed: {list(_RATES)}")
    missing = [rate for rate in _REQUIRED if rate not in rates]
    if missing:
        raise ConfigError(f"[model.prices] {name!r} needs {' and '.join(missing)}, in USD per million tokens")
    for rate, value in rates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ConfigError(f"[model.prices] {name!r} {rate} is {value!r}; a price is a number of USD per million "
                              "tokens, 0 or more")
    return ModelPrice(**{rate: float(value) for rate, value in rates.items()})
