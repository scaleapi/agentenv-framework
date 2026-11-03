"""The website spec: one declarative description of a site that every way of serving one produces or is checked
against. A spec is a TOML file of fixed sections, each optional: ``[site]`` (name, domains, health path),
``[source]`` (exactly one of ``image``, ``sdk``, ``static``, ``live``, ``replay``), ``[data]``, ``[users]``,
``[roles]``, ``[routes]``, ``[state]``, ``[clock]`` and ``[[uses]]``, with ``extends`` naming a spec it overlays.

``load(path)`` reads one, resolves ``extends`` (a deep merge: tables merge key by key, arrays and scalars replace,
``[source]`` replaces whole, ``false`` removes an inherited route, role or collection, and ``[[uses]]`` entries are
added, one with the same ``as`` replacing the inherited one) and composes ``[[uses]]`` (each component's routes
mounted under its path prefix and named ``<as>.<name>``, its collections namespaced ``<as>.<name>`` unless mapped
into one of the site's, its personas merged), and validates the result, raising ``SpecError`` with every problem.
A resolved spec has no ``extends``, ``uses`` or ``routes.openapi`` left, the document's operations being in
``routes.ops`` (less any an overlay removed); ``components`` says what was mounted.

``to_card_extension()`` is the ``urn:agentenv:website/v1`` params a server of the spec advertises, ``from_card``
reads a running site's card back into a spec, ``from_environment`` derives one from a WebsiteEnvironment class's
decorators, and ``name_table`` builds a multi env's one table of host names, synthetic before live before
allowlisted. ``python -m agentenv_protocol.website.spec validate|show|schema`` is its command line."""
from __future__ import annotations
from ..readiness import WEBSITE_EXTENSION_URI

import argparse
import copy
import dataclasses
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator, model_validator

from ...agent_env_environment import _CARD_CONFIG_ATTR
from ...client import find_extension
from ..routes import openapi_path, reserved_path
from ..session import session_cookie_name

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

DEFAULT_HEALTH_PATH = "/api/health"
SESSION_PATH = "/api/session"
STATE_PATH = "/go"
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
SOURCE_KINDS = ("image", "sdk", "static", "live", "replay")
NameKind = Literal["synthetic", "live", "allowlist"]
_PRECEDENCE: dict[str, int] = {"synthetic": 0, "live": 1, "allowlist": 2}
_HOST = re.compile(r"(\*\.)?([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*(\.[A-Za-z_][A-Za-z0-9_-]*)*")
_SDK_REF = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_<][\w.<>]*")
_SECRET_REF = re.compile(r"(secret|env):[A-Za-z_][A-Za-z0-9_./-]*")
_PATH = re.compile(r"/(?:[A-Za-z0-9._~!$&'()*+,;=:@/{}-]|%[0-9A-Fa-f]{2})*")
_PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("source", "image", "dockerfile"), ("source", "image", "context"), ("source", "static"), ("source", "assets"),
    ("source", "replay", "capture"), ("data", "schema"), ("routes", "openapi"))
_REMOVABLE = {("routes", "ops"), ("roles",), ("data", "collections")}


class SpecError(ValueError):
    """A spec that can't be read or doesn't hold together; ``problems`` lists each, ``<where>: <why>``."""

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Site(_Section):
    """Who the site is: its name (the card's), the host names it answers for, ``*.`` as a leftmost label matching
    any subdomain, and the path that answers 200 once it's usable."""

    name: Optional[str] = None
    domains: tuple[str, ...] = ()
    health: Optional[str] = None

    @field_validator("domains")
    @classmethod
    def _domains(cls, domains: tuple[str, ...]) -> tuple[str, ...]:
        bad = [d for d in domains if not _HOST.fullmatch(d)]
        if bad:
            raise ValueError(f"{bad} aren't host names in lower case, '*.' only as the leftmost label")
        return domains

    @field_validator("health")
    @classmethod
    def _health(cls, health: Optional[str]) -> Optional[str]:
        if health is not None and (not _PATH.fullmatch(health) or health == "/" or reserved_path(health)):
            raise ValueError(f"health is a path of its own under '/', not {health!r}")
        return health


class ImageSource(_Section):
    """A server image built from a Dockerfile, both relative to the spec."""

    dockerfile: str
    context: str = "."


class LiveSource(_Section):
    """The real site, reached through the env: the hosts besides its domains it may call, and the credentials it
    signs in with, each a ``secret:`` or ``env:`` reference, never a value."""

    allow_hosts: tuple[str, ...] = ()
    credentials: dict[str, str] = {}

    @field_validator("allow_hosts")
    @classmethod
    def _hosts(cls, hosts: tuple[str, ...]) -> tuple[str, ...]:
        return Site._domains(hosts)

    @field_validator("credentials")
    @classmethod
    def _credentials(cls, credentials: dict[str, str]) -> dict[str, str]:
        bad = sorted(name for name, ref in credentials.items() if not _SECRET_REF.fullmatch(ref))
        if bad:
            raise ValueError(f"credentials {bad} must be secret:<name> or env:<name> references, not values")
        return credentials


class ReplaySource(_Section):
    """A capture of the site served back: the artifact, relative to the spec, and the hosts still let through."""

    capture: str
    allow_hosts: tuple[str, ...] = ()

    @field_validator("allow_hosts")
    @classmethod
    def _hosts(cls, hosts: tuple[str, ...]) -> tuple[str, ...]:
        return Site._domains(hosts)


class Source(_Section):
    """What serves the site, exactly one of: an ``image`` (a reference, or a table building one), an ``sdk``
    WebsiteEnvironment class as ``module:Class`` with its static ``assets`` folder, a ``static`` folder, the
    ``live`` real site, or a ``replay`` of a capture."""

    image: Union[str, ImageSource, None] = None
    sdk: Optional[str] = None
    static: Optional[str] = None
    live: Optional[LiveSource] = None
    replay: Optional[ReplaySource] = None
    assets: Optional[str] = None

    @model_validator(mode="after")
    def _one(self) -> Source:
        given = [kind for kind in SOURCE_KINDS if getattr(self, kind) is not None]
        if len(given) != 1:
            raise ValueError(f"names exactly one of {', '.join(SOURCE_KINDS)}, not {given or 'none'}")
        if self.sdk is not None and not _SDK_REF.fullmatch(self.sdk):
            raise ValueError(f"sdk is module:Class, not {self.sdk!r}")
        if self.assets is not None and self.sdk is None:
            raise ValueError("assets is the static folder of an sdk source")
        return self

    @property
    def kind(self) -> str:
        return next(kind for kind in SOURCE_KINDS if getattr(self, kind) is not None)


class Collection(_Section):
    """One collection of the site's data: the field naming each object (``key``), the field naming the user who
    owns one (``owner``), and a JSON Schema of an object, inline or a file relative to the spec."""

    key: str = "id"
    owner: Optional[str] = None
    schema_: Union[str, dict, None] = Field(default=None, alias="schema")


class Data(_Section):
    """The site's data: its collections, or one JSON Schema of the whole state, or ``models = "mcp:<server>"``
    binding it to an MCP server's schema; and the seed files it starts with, in order."""

    collections: dict[str, Collection] = {}
    schema_: Union[str, dict, None] = Field(default=None, alias="schema")
    models: Optional[str] = None
    seed: tuple[str, ...] = ()

    @field_validator("seed", mode="before")
    @classmethod
    def _seed(cls, seed: Any) -> Any:
        return (seed,) if isinstance(seed, str) else seed

    @field_validator("models")
    @classmethod
    def _models(cls, models: Optional[str]) -> Optional[str]:
        if models is not None and not re.fullmatch(r"mcp:[A-Za-z0-9_.-]+", models):
            raise ValueError(f"models is mcp:<server>, not {models!r}")
        return models

    @model_validator(mode="after")
    def _schema_or_models(self) -> Data:
        if self.schema_ is not None and self.models is not None:
            raise ValueError("data takes a schema or models, not both")
        if bad := sorted(name for name in self.collections if not _NAME.fullmatch(name)):
            raise ValueError(f"collection names {bad} aren't names")
        return self


class Persona(_Section):
    id: str
    name: Optional[str] = None
    role: Optional[str] = None


class Users(_Section):
    """Who uses the site: its personas, the acting user a browser with no session is signed in as, and how a
    browser signs in: ``cookie`` (the site's signed session cookie, as ``session_path`` hands it out), ``none``, or
    ``live``, the real site's own sign-in."""

    personas: tuple[Persona, ...] = ()
    acting_user: Optional[str] = None
    sign_in: Literal["cookie", "none", "live"] = "cookie"
    session_cookie: Optional[str] = None
    session_path: Optional[str] = None

    @field_validator("personas", mode="before")
    @classmethod
    def _personas(cls, personas: Any) -> Any:
        return [{"id": p} if isinstance(p, str) else p for p in personas] if isinstance(personas, list) else personas

    @model_validator(mode="after")
    def _consistent(self) -> Users:
        ids = [p.id for p in self.personas]
        if twice := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"personas {twice} are declared twice")
        if self.acting_user is not None and ids and self.acting_user not in ids:
            raise ValueError(f"acting_user {self.acting_user!r} is none of the personas {ids}")
        if self.sign_in != "cookie" and (self.session_cookie or self.session_path):
            raise ValueError("session_cookie and session_path are a cookie sign-in's")
        return self


class Route(_Section):
    """One named operation: its verb, its OpenAPI-style path, the roles it answers (any when empty), and, for a site
    served by a live or replayed source, the ``module:Class`` whose route of the same name answers it instead."""

    method: str
    path: str
    roles: tuple[str, ...] = ()
    served_by: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, value: Any) -> Any:
        if isinstance(value, str):
            method, _, path = value.strip().partition(" ")
            value = {"method": method, "path": path.strip()}
        if isinstance(value, dict):
            value = dict(value)
            if isinstance(value.get("method"), str):
                value["method"] = value["method"].upper()
            if isinstance(value.get("path"), str):
                value["path"] = openapi_path(value["path"])
        return value

    @model_validator(mode="after")
    def _route(self) -> Route:
        if self.method not in METHODS:
            raise ValueError(f"method is one of {', '.join(METHODS)}, not {self.method!r}")
        if not _PATH.fullmatch(self.path):
            raise ValueError(f"path starts with '/', not {self.path!r}")
        if reserved_path(self.path):
            raise ValueError(f"path {self.path!r} is the data plane's or the card's")
        if self.served_by is not None and not _SDK_REF.fullmatch(self.served_by):
            raise ValueError(f"served_by is module:Class, not {self.served_by!r}")
        return self

    def advertised(self, name: str) -> dict:
        return {"method": self.method, "path": self.path, "name": name, "roles": list(self.roles) or None}


class Routes(_Section):
    """The site's named operations, ``name = "METHOD /path"`` or a table, and an OpenAPI document whose operations,
    by ``operationId``, are routes too; the names are what triggers and records call ``op``."""

    openapi: Optional[str] = None
    ops: dict[str, Route] = {}

    @field_validator("ops")
    @classmethod
    def _names(cls, ops: dict[str, Route]) -> dict[str, Route]:
        if bad := sorted(name for name in ops if not _NAME.fullmatch(name)):
            raise ValueError(f"route names {bad} aren't names")
        return ops


class State(_Section):
    """How a grader reads the site's state: ``data_plane`` (``data/get``), ``go`` (``GET <path>``, the
    ``{initial_state, current_state, state_diff}`` of a session), or ``diff`` (the writes a run made, as the env's
    proxy recorded them, for a site whose data plane isn't its own)."""

    read: Literal["data_plane", "go", "diff"] = "go"
    path: Optional[str] = None

    @model_validator(mode="after")
    def _path(self) -> State:
        if self.path is not None and self.read != "go":
            raise ValueError("path is where a go read reads")
        if self.path is not None and not _PATH.fullmatch(self.path):
            raise ValueError(f"path starts with '/', not {self.path!r}")
        return self


class Clock(_Section):
    """How the site follows the universe's time: ``header`` (its answers carry ``X-Universe-Now`` and its pages read
    it), ``date_patch`` (its pages' ``Date`` is shifted too), or ``none`` (wall time)."""

    follow: Literal["header", "date_patch", "none"] = "header"


class Use(_Section):
    """A component: another spec, relative to this one, mounted under ``mount``, named ``as`` (its site's name when
    left out), with ``data`` mapping a collection of the component's onto one of this site's."""

    spec: str
    mount: str = "/"
    as_: Optional[str] = Field(default=None, alias="as")
    data: dict[str, str] = {}

    @field_validator("mount")
    @classmethod
    def _mount(cls, mount: str) -> str:
        if not _PATH.fullmatch(mount) or "{" in mount or reserved_path(mount):
            raise ValueError(f"mount is a plain path under '/', not {mount!r}")
        return mount.rstrip("/") or "/"


@dataclasses.dataclass(frozen=True)
class Component:
    """A component a resolved spec mounted: its name, its path prefix, and its own resolved spec."""

    name: str
    mount: str
    spec: WebsiteSpec


class WebsiteSpec(_Section):
    """A site as one declaration. ``load`` reads and resolves one; a spec built from a dict is resolved by
    ``resolve``. Paths in it are relative to ``base_dir``."""

    extends: Optional[str] = None
    site: Site = Site()
    source: Optional[Source] = None
    data: Data = Data()
    users: Users = Users()
    roles: dict[str, tuple[str, ...]] = {}
    routes: Routes = Routes()
    state: Optional[State] = None
    clock: Optional[Clock] = None
    uses: tuple[Use, ...] = ()

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)
    _components: tuple[Component, ...] = PrivateAttr(default=())
    _declared: Optional[dict] = PrivateAttr(default=None)

    def __eq__(self, other: object) -> bool:
        """Two specs are equal when they resolve to the same site in the same folder, however each was declared."""
        if not isinstance(other, WebsiteSpec):
            return NotImplemented
        return (type(self), self.__dict__, self._base_dir, self._components) == (
            type(other), other.__dict__, other._base_dir, other._components)

    __hash__ = None

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def components(self) -> tuple[Component, ...]:
        return self._components

    @property
    def name(self) -> Optional[str]:
        return self.site.name

    @property
    def health(self) -> str:
        return self.site.health or DEFAULT_HEALTH_PATH

    def path(self, relative: str) -> Path:
        """``relative``, a path the spec names, as a path on disk."""
        given = Path(relative)
        return given if given.is_absolute() else self._base_dir / given

    def route_table(self) -> dict[str, Route]:
        """Every route by name, each with the roles ``[roles]`` grants it added to its own."""
        granted: dict[str, list[str]] = {}
        for role, names in self.roles.items():
            for name in names:
                granted.setdefault(name, []).append(role)
        return {name: route.model_copy(update={"roles": tuple(dict.fromkeys([*route.roles, *granted.get(name, [])]))})
                for name, route in self.routes.ops.items()}

    def to_dict(self) -> dict:
        """The spec as a JSON-able dict, empty sections left out, which ``resolve`` reads back as the same spec."""
        return _pruned(self.model_dump(mode="json", by_alias=True, exclude_none=True))

    def overlay(self, changes: dict, *, base_dir: Path | str | None = None) -> WebsiteSpec:
        """This spec with ``changes`` merged over it as an ``extends`` overlay merges, then resolved again; paths in
        ``changes`` are relative to ``base_dir``, this spec's own when left out. A spec ``resolve`` or ``load`` made
        is merged into as it was declared, its ``[[uses]]`` not yet mounted, so a component of the same ``as``
        replaces the mounted one as it would in a file overlay."""
        here = Path(base_dir) if base_dir is not None else self._base_dir
        declared = self._declared if self._declared is not None else self.to_dict()
        merged = _merge(declared, _expanded(_rebased(changes, here, self._base_dir)))
        return resolve(merged, self._base_dir)

    def to_card_extension(self) -> dict:
        """The ``urn:agentenv:website/v1`` params a server of this spec advertises: its domains, health path and
        routes by name, its session cookie and path for a cookie sign-in, and how its state is read and its clock
        followed, when declared."""
        params: dict = {"domains": list(self.site.domains), "health": self.health,
                        "routes": [route.advertised(name) for name, route in self.route_table().items()]}
        if self.users.sign_in == "cookie":
            params["session_cookie"] = self.users.session_cookie or session_cookie_name(self.site.name)
            params["session_path"] = self.users.session_path or SESSION_PATH
        if self.state is not None:
            params["state"] = {"read": self.state.read, **({"path": self.state_path} if self.state.read == "go" else {})}
        if self.clock is not None:
            params["clock"] = {"follow": self.clock.follow}
        return params

    @property
    def state_path(self) -> Optional[str]:
        if self.state is None or self.state.read != "go":
            return None
        return self.state.path or STATE_PATH


def load(path: str | Path) -> WebsiteSpec:
    """The spec in the TOML file at ``path``, resolved: its ``extends`` merged, its ``uses`` composed, checked."""
    return _load(Path(path).resolve(), ())


def resolve(raw: dict, base_dir: str | Path, *, origin: str = "spec") -> WebsiteSpec:
    """The spec ``raw`` declares, paths in it relative to ``base_dir``, resolved as ``load`` resolves a file's."""
    return _resolve(copy.deepcopy(raw), Path(base_dir).resolve(), origin, ())


def _load(path: Path, stack: tuple[Path, ...]) -> WebsiteSpec:
    return _resolve(_read(path), path.parent, str(path), stack + (path,))


def _read(path: Path) -> dict:
    if tomllib is None:
        raise SpecError([f"{path}: reading a TOML spec needs Python 3.11's tomllib, or tomli before it"])
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SpecError([f"{path}: can't be read: {e.strerror or e}"]) from None
    except tomllib.TOMLDecodeError as e:
        raise SpecError([f"{path}: isn't TOML: {e}"]) from None


def _resolve(raw: dict, base_dir: Path, origin: str, stack: tuple[Path, ...]) -> WebsiteSpec:
    """The spec ``raw`` declares, resolved. A spec that uses components has its acting user checked only once their
    personas are merged into its own, since the acting user may be one only a component declares."""
    raw = _extended(raw, base_dir, origin, stack)
    declared = _expanded(copy.deepcopy(raw))
    removed = _removals(raw)
    problems: list[str] = []
    imported = _imported(raw, base_dir, removed, problems)
    users = raw.get("users")
    spec = _validated({**raw, "users": {k: v for k, v in users.items() if k != "acting_user"}}
                      if raw.get("uses") and isinstance(users, dict) else raw, origin)
    spec._base_dir = base_dir
    ops = dict(imported)
    for name, route in spec.routes.ops.items():
        if name in ops and (ops[name].method, ops[name].path) != (route.method, route.path):
            problems.append(f"routes.ops.{name}: {route.method} {route.path} isn't the openapi document's "
                            f"{ops[name].method} {ops[name].path}")
        ops[name] = route
    raw_routes = raw.setdefault("routes", {})
    raw_routes["ops"] = {name: route.model_dump(mode="json", exclude_none=True, exclude_defaults=True) | {
        "method": route.method, "path": route.path} for name, route in ops.items()}
    raw_routes.pop("openapi", None)
    components = []
    for index, use in enumerate(spec.uses):
        component = _component(use, index, spec, stack, problems)
        if component is not None:
            components.append(component)
            _mount(raw, component, use, base_dir, f"uses[{index}]", problems)
    raw.pop("uses", None)
    if problems:
        raise SpecError(f"{origin}: {p}" for p in problems)
    resolved = _validated(raw, origin)
    resolved._base_dir, resolved._components, resolved._declared = base_dir, tuple(components), declared
    problems = _check(resolved)
    if problems:
        raise SpecError(f"{origin}: {p}" for p in problems)
    return resolved


def _extended(raw: dict, base_dir: Path, origin: str, stack: tuple[Path, ...]) -> dict:
    """``raw`` merged over the spec its ``extends`` names, recursively, that spec's paths made relative to
    ``base_dir``."""
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise SpecError([f"{origin}: extends is the path of a spec, not {parent!r}"])
    path = (base_dir / parent).resolve()
    if path in stack:
        raise SpecError([f"{origin}: extends {parent} is a cycle: {' -> '.join(str(p) for p in (*stack, path))}"])
    base = _extended(_read(path), path.parent, str(path), stack + (path,))
    return _merge(_expanded(_rebased(base, path.parent, base_dir)), _expanded(raw))


def _expanded(raw: dict) -> dict:
    """``raw`` with each ``"METHOD /path"`` route written as its table, so an overlay can change one key of it."""
    ops = (raw.get("routes") or {}).get("ops") if isinstance(raw.get("routes"), dict) else None
    if isinstance(ops, dict):
        for name, route in ops.items():
            if isinstance(route, str):
                method, _, path = route.strip().partition(" ")
                ops[name] = {"method": method, "path": path.strip()}
    return raw


def _removals(raw: dict) -> set[str]:
    """The names of the routes ``raw``'s merged overlays removed, each a ``false`` left in ``routes.ops`` so it
    removes an operation of the OpenAPI document too, taken out of ``raw``."""
    routes = raw.get("routes")
    ops = routes.get("ops") if isinstance(routes, dict) else None
    if not isinstance(ops, dict):
        return set()
    removed = {name for name, route in ops.items() if route is False}
    for name in removed:
        del ops[name]
    return removed


def _validated(raw: dict, origin: str) -> WebsiteSpec:
    try:
        return WebsiteSpec.model_validate(raw)
    except ValidationError as e:
        raise SpecError(f"{origin}: {_where(err['loc'])}: {_why(err)}" for err in e.errors()) from None


def _where(loc: tuple) -> str:
    where = ""
    for part in loc:
        if isinstance(part, int):
            where += f"[{part}]"
        elif part not in ("str", "dict", "ImageSource") and "[" not in part:
            where += f".{part}" if where else part
    return where or "spec"


def _why(err: dict) -> str:
    message = err["msg"]
    return message.removeprefix("Value error, ") if err["type"] != "extra_forbidden" else "isn't a key the spec takes"


def _merge(base: Any, over: Any, at: tuple[str, ...] = ()) -> Any:
    """``over`` merged over ``base``: tables key by key, ``[source]`` whole, ``false`` removing an entry of
    routes, roles or collections, ``[[uses]]`` added by ``as``, and anything else replaced. A removed route stays a
    ``false``, so that resolving the merge removes it from the OpenAPI document's operations as well."""
    if not (isinstance(base, dict) and isinstance(over, dict)) or at == ("source",):
        return copy.deepcopy(over)
    merged = copy.deepcopy(base)
    for key, value in over.items():
        if value is False and at == ("routes", "ops"):
            merged[key] = False
        elif value is False and at in _REMOVABLE:
            merged.pop(key, None)
        elif key == "uses" and not at:
            kept = {_use_id(u): u for u in merged.get("uses", [])}
            kept.update({_use_id(u): copy.deepcopy(u) for u in value})
            merged["uses"] = list(kept.values())
        elif key in merged:
            merged[key] = _merge(merged[key], value, (*at, key))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _use_id(use: Any) -> str:
    return str(use.get("as") or use.get("spec")) if isinstance(use, dict) else repr(use)


def _rebased(raw: dict, from_dir: Path, to_dir: Path) -> dict:
    """``raw`` with each relative path it names, relative to ``from_dir``, made relative to ``to_dir``; an image
    source's context is moved even when left to its default, so it stays the folder its own spec meant."""
    if from_dir == to_dir:
        return copy.deepcopy(raw)
    raw = copy.deepcopy(raw)

    def moved(value: Any) -> Any:
        return _moved(value, from_dir, to_dir)

    image = raw.get("source", {}).get("image") if isinstance(raw.get("source"), dict) else None
    if isinstance(image, dict):
        image.setdefault("context", ImageSource.model_fields["context"].default)
    for keys in _PATH_FIELDS:
        holder = raw
        for key in keys[:-1]:
            holder = holder.get(key) if isinstance(holder, dict) else None
        if isinstance(holder, dict) and keys[-1] in holder:
            holder[keys[-1]] = moved(holder[keys[-1]])
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    if isinstance(data.get("seed"), (str, list)):
        data["seed"] = moved(data["seed"]) if isinstance(data["seed"], str) else [moved(s) for s in data["seed"]]
    for collection in (data.get("collections") or {}).values():
        if isinstance(collection, dict) and "schema" in collection:
            collection["schema"] = moved(collection["schema"])
    for use in raw.get("uses") or []:
        if isinstance(use, dict) and "spec" in use:
            use["spec"] = moved(use["spec"])
    return raw


def _moved(value: Any, from_dir: Path, to_dir: Path) -> Any:
    """``value``, a path relative to ``from_dir``, made relative to ``to_dir``; anything else as it is."""
    if not isinstance(value, str) or Path(value).is_absolute():
        return value
    return os.path.relpath(from_dir / value, to_dir)


def _imported(raw: dict, base_dir: Path, removed: set[str], problems: list[str]) -> dict[str, Route]:
    """The routes of ``raw``'s OpenAPI document that no overlay removed, each of ``raw``'s ``routes.ops`` tables
    naming one of them written over it, so an overlay can change one key of an imported operation as it can of a
    declared one."""
    routes = raw.get("routes") if isinstance(raw.get("routes"), dict) else {}
    document = routes.get("openapi")
    if not isinstance(document, str):
        return {}
    imported = {name: route for name, route in _openapi_routes(document, base_dir, problems).items()
                if name not in removed}
    ops = routes.get("ops")
    for name, route in (ops.items() if isinstance(ops, dict) else ()):
        if isinstance(route, dict) and name in imported:
            ops[name] = imported[name].model_dump(mode="json", exclude_none=True) | route
    return imported


def _openapi_routes(document: str, base_dir: Path, problems: list[str]) -> dict[str, Route]:
    """The routes of the OpenAPI document at ``document``, relative to ``base_dir``, by operationId, roles from
    ``x-agentenv-roles``; a document of another shape, an operation without a name and a name two operations share
    are problems."""
    try:
        parsed = json.loads((base_dir / document).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        problems.append(f"routes.openapi: {document} isn't a readable JSON document: {e}")
        return {}
    paths = (parsed.get("paths") or {}) if isinstance(parsed, dict) else None
    if not isinstance(paths, dict):
        problems.append(f"routes.openapi: {document} isn't an OpenAPI document, an object whose paths "
                        "are an object")
        return {}
    routes: dict[str, Route] = {}
    for path, operations in paths.items():
        if not isinstance(operations, dict):
            problems.append(f"routes.openapi: path {path} isn't an object of operations")
            continue
        for method, operation in operations.items():
            if method.upper() not in METHODS or not isinstance(operation, dict):
                continue
            name = operation.get("operationId")
            if not name or not isinstance(name, str):
                problems.append(f"routes.openapi: {method.upper()} {path} has no operationId to name it by")
                continue
            if name in routes:
                problems.append(f"routes.openapi: {name} names both {routes[name].method} {routes[name].path} and "
                                f"{method.upper()} {path}")
                continue
            try:
                routes[name] = Route(method=method, path=path, roles=operation.get("x-agentenv-roles") or ())
            except ValidationError as e:
                problems.append(f"routes.openapi: {name}: {_why(e.errors()[0])}")
    return routes


def _component(use: Use, index: int, host: WebsiteSpec, stack: tuple[Path, ...],
               problems: list[str]) -> Optional[Component]:
    path = host.path(use.spec).resolve()
    if path in stack:
        problems.append(f"uses[{index}]: {use.spec} is a cycle: {' -> '.join(str(p) for p in (*stack, path))}")
        return None
    try:
        spec = _load(path, stack)
    except SpecError as e:
        problems.extend(f"uses[{index}]: {p}" for p in e.problems)
        return None
    name = use.as_ or spec.site.name
    if name is None:
        problems.append(f"uses[{index}]: {use.spec} has no site.name; name it with as")
        return None
    return Component(name, use.mount, spec)


def _mount(raw: dict, component: Component, use: Use, base_dir: Path, where: str, problems: list[str]) -> None:
    """Merge ``component`` into ``raw``, whose paths are relative to ``base_dir``: its routes prefixed and
    namespaced, its collections namespaced or mapped, their schema files made relative to ``base_dir``, its personas
    added; every clash is a problem."""
    ns, spec = component.name, component.spec
    ops = raw["routes"]["ops"]
    taken = {(r["method"], r["path"]): name for name, r in ops.items()}
    if use.mount != "/":
        for name, r in ops.items():
            if r["path"] == use.mount or r["path"].startswith(use.mount + "/"):
                problems.append(f"{where}: route {name} ({r['method']} {r['path']}) is under {use.mount}, where "
                                f"{ns} is mounted")
    for name, route in spec.route_table().items():
        full, path = f"{ns}.{name}", _prefixed(use.mount, route.path)
        if full in ops:
            problems.append(f"{where}: route {full} is declared by the site and by component {ns}")
        elif (route.method, path) in taken:
            problems.append(f"{where}: {ns}'s route {name} is {route.method} {path}, which {taken[(route.method, path)]} "
                            "already serves")
        else:
            ops[full] = {**route.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
                         "method": route.method, "path": path}
            taken[(route.method, path)] = full
    data = raw.setdefault("data", {})
    collections = data.setdefault("collections", {})
    if missing := sorted(set(use.data) - set(spec.data.collections)):
        problems.append(f"{where}: data maps {missing}, which {ns} doesn't declare")
    for name, collection in spec.data.collections.items():
        target = use.data.get(name, f"{ns}.{name}")
        theirs = collection.model_dump(mode="json", by_alias=True, exclude_none=True)
        if "schema" in theirs:
            theirs["schema"] = _moved(theirs["schema"], spec.base_dir, base_dir)
        ours = collections.get(target)
        if ours is None:
            collections[target] = theirs
            continue
        merged = {**theirs, **ours}
        clash = [k for k in ("key", "owner", "schema") if k in ours and k in theirs and ours[k] != theirs[k]]
        clash += ["key"] if "key" not in ours and theirs.get("key", "id") != "id" else []
        if clash:
            problems.append(f"{where}: {ns}'s collection {name} maps onto {target}, whose {', '.join(clash)} differ")
        collections[target] = merged
    users = raw.setdefault("users", {})
    personas = {(p["id"] if isinstance(p, dict) else p): ({"id": p} if isinstance(p, str) else p)
                for p in users.get("personas", [])}
    for persona in spec.users.personas:
        theirs, ours = persona.model_dump(mode="json", exclude_none=True), personas.get(persona.id, {})
        if any(key in ours and ours[key] != value for key, value in theirs.items()):
            problems.append(f"{where}: persona {persona.id} is declared otherwise by {ns}")
        personas[persona.id] = {**theirs, **ours}
    if personas:
        users["personas"] = list(personas.values())
    sign_in = users.get("sign_in", "cookie")
    if spec.users.sign_in not in ("none", sign_in):
        problems.append(f"{where}: {ns} signs in by {spec.users.sign_in}, the site by {sign_in}")


def _prefixed(mount: str, path: str) -> str:
    return path if mount == "/" else mount + path


def _check(spec: WebsiteSpec) -> list[str]:
    """What doesn't hold together across sections of a resolved spec."""
    problems = []
    table = spec.route_table()
    seen: dict[tuple[str, str], str] = {}
    for name, route in table.items():
        if (route.method, route.path) in seen:
            problems.append(f"routes.ops: {name} and {seen[route.method, route.path]} are both {route.method} {route.path}")
        seen.setdefault((route.method, route.path), name)
    if ("GET", spec.health) in seen:
        problems.append(f"site.health: {spec.health} is served by route {seen['GET', spec.health]}")
    for role, names in spec.roles.items():
        if unknown := [n for n in names if n not in table]:
            problems.append(f"roles.{role}: names no route {unknown}")
    kind = spec.source.kind if spec.source is not None else None
    if spec.users.sign_in == "live" and kind not in ("live", "replay"):
        problems.append("users.sign_in: live is the sign-in of a live or replayed source")
    if spec.state is not None and spec.state.read == "diff" and kind not in ("live", "replay", None):
        problems.append(f"state.read: diff is how a live or replayed site is read, not a site of source {kind}")
    if kind in ("live", "replay") and spec.state is not None and spec.state.read != "diff":
        problems.append(f"state.read: a {kind} site has no data plane of its own; its state is read as diff")
    if kind == "sdk" and spec.clock is not None and spec.clock.follow == "none":
        problems.append("clock.follow: an sdk site follows the universe's clock; none is a live site's")
    if kind == "static" and table:
        problems.append("routes: a static site serves no routes")
    if served := [n for n, r in table.items() if r.served_by is not None and kind not in ("live", "replay")]:
        problems.append(f"routes.ops: served_by swaps a route of a live or replayed site, not of {kind or 'no'} "
                        f"source ({served})")
    for name, collection in spec.data.collections.items():
        if collection.owner is not None and spec.users.sign_in == "none":
            problems.append(f"data.collections.{name}: owner names a user, and the site signs nobody in")
    return problems


def _pruned(value: Any, at: tuple[str, ...] = ()) -> Any:
    """``value`` without its empty tables and arrays, but for a JSON Schema, which is kept as it is."""
    if _is_schema(at):
        return value
    if isinstance(value, dict):
        pruned = {k: _pruned(v, (*at, k)) for k, v in value.items()}
        return {k: v for k, v in pruned.items() if _is_schema((*at, k)) or v not in ({}, [], ())}
    if isinstance(value, list):
        return [_pruned(v, at) for v in value]
    return value


def _is_schema(at: tuple[str, ...]) -> bool:
    return at == ("data", "schema") or (len(at) == 4 and at[:2] == ("data", "collections") and at[3] == "schema")


def from_card(card: dict) -> WebsiteSpec:
    """The spec a running site's card declares: its name, domains, health path, routes and sign-in, and its state
    and clock when the card says them. A card says nothing of a site's source, data or personas."""
    extension = find_extension(card, WEBSITE_EXTENSION_URI)
    if extension is None:
        raise SpecError([f"the card declares no {WEBSITE_EXTENSION_URI}"])
    params = extension.get("params") or {}
    raw: dict = {"site": {"name": card.get("name"), "domains": params.get("domains") or [],
                          "health": params.get("health")},
                 "routes": {"ops": {r["name"]: {"method": r["method"], "path": r["path"], "roles": r.get("roles") or []}
                                    for r in params.get("routes") or []}}}
    if params.get("session_cookie"):
        raw["users"] = {"sign_in": "cookie"}
        if params["session_cookie"] != session_cookie_name(card.get("name")):
            raw["users"]["session_cookie"] = params["session_cookie"]
        if params.get("session_path") not in (None, SESSION_PATH):
            raw["users"]["session_path"] = params["session_path"]
    else:
        raw["users"] = {"sign_in": "none"}
    if isinstance(params.get("state"), dict):
        raw["state"] = {k: v for k, v in params["state"].items() if k != "path" or v != STATE_PATH}
    if isinstance(params.get("clock"), dict):
        raw["clock"] = params["clock"]
    return resolve(_pruned(raw), Path.cwd(), origin=f"card {card.get('name')!r}")


def from_environment(cls: type) -> WebsiteSpec:
    """The spec a WebsiteEnvironment class's decorators declare: ``@environment_card``'s name, ``@website``'s
    domains, health path, static folder, seed and clock patch, its ``@api`` routes, a cookie sign-in, and its state
    read at ``/go`` when it keeps the default store or serves ``/go`` itself, else through the data plane."""
    declaration = cls.declaration()
    if declaration.spec is not None:
        base = declaration.spec
        raw = base.to_dict()
    else:
        base, raw = None, {}
    card_name = (getattr(cls, _CARD_CONFIG_ATTR, None) or {}).get("name")
    routes = cls.routes()
    serves_go = any(r.method == "GET" and openapi_path(r.path) == STATE_PATH for r in routes)
    site_file = Path(inspect.getfile(cls)).resolve().parent
    raw["site"] = {**raw.get("site", {}), "domains": list(declaration.domains), "health": declaration.health}
    if card_name or raw["site"].get("name") is None:
        raw["site"]["name"] = card_name or cls.__name__
    raw["source"] = {"sdk": f"{cls.__module__}:{cls.__qualname__}",
                     **({"assets": str(cls._site_path(declaration.static))} if declaration.static else {})}
    seeds = [str(path) for path in cls._seed_files()]
    if seeds:
        raw.setdefault("data", {})["seed"] = seeds
        acting = _seed_user(seeds)
        if acting is not None and "acting_user" not in raw.get("users", {}):
            raw.setdefault("users", {})["acting_user"] = acting
    raw.setdefault("users", {})["sign_in"] = "cookie"
    raw["routes"] = {"ops": {r.name: {"method": r.method, "path": openapi_path(r.path), "roles": list(r.roles)}
                             for r in routes}}
    raw.pop("roles", None)
    raw["state"] = {"read": "go"} if cls._store_get is not None or serves_go else {"read": "data_plane"}
    raw["clock"] = {"follow": "date_patch" if declaration.clock_patch else "header"}
    return resolve(raw, base.base_dir if base is not None else site_file, origin=cls.__qualname__)


def _seed_user(paths: Iterable[str | Path]) -> Optional[str]:
    """The ``current_user`` the store is left with once the seed files at ``paths`` are loaded in order, each that
    has one taking its place."""
    user = None
    for path in paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "current_user" in data:
            user = data["current_user"] if isinstance(data["current_user"], str) else None
    return user


def problems_with_environment(spec: WebsiteSpec, cls: type) -> list[str]:
    """Where a WebsiteEnvironment class doesn't serve what ``spec`` declares: its source, routes by name, verb,
    path and roles, sign-in, state read and clock."""
    problems = []
    if spec.source is not None and spec.source.kind != "sdk":
        problems.append(f"source: a WebsiteEnvironment is an sdk source, and the spec names {spec.source.kind}")
    elif spec.source is not None and spec.source.sdk.rsplit(":", 1)[1] not in (cls.__qualname__, cls.__name__):
        problems.append(f"source.sdk: {spec.source.sdk} names another class than {cls.__name__}")
    derived = from_environment(cls)
    ours = {name: (r.method, r.path, tuple(sorted(r.roles))) for name, r in spec.route_table().items()}
    theirs = {name: (r.method, r.path, tuple(sorted(r.roles))) for name, r in derived.route_table().items()}
    if spec.routes.ops:
        for name in sorted(set(ours) | set(theirs)):
            if name not in theirs:
                problems.append(f"routes.ops.{name}: {' '.join(ours[name][:2])} isn't an @api route of {cls.__name__}")
            elif name not in ours:
                problems.append(f"routes.ops: {cls.__name__} serves {name} ({' '.join(theirs[name][:2])}), "
                                "which the spec doesn't declare")
            elif ours[name] != theirs[name]:
                problems.append(f"routes.ops.{name}: the spec says {_shown(ours[name])}, the class serves "
                                f"{_shown(theirs[name])}")
    if spec.users.sign_in != "cookie":
        problems.append(f"users.sign_in: a WebsiteEnvironment signs in by cookie, not {spec.users.sign_in}")
    cookie = session_cookie_name(spec.site.name)
    if spec.users.session_cookie not in (None, cookie) or spec.users.session_path not in (None, SESSION_PATH):
        problems.append(f"users: a WebsiteEnvironment's session is {cookie} at {SESSION_PATH}")
    if spec.state is not None and (spec.state.read, spec.state_path) != (derived.state.read, derived.state_path):
        problems.append(f"state: the spec reads {spec.state.read} {spec.state_path or ''}".rstrip()
                        + f", the class is read by {derived.state.read}")
    acting = _seed_user(cls._seed_files())
    if spec.users.acting_user is not None and acting is not None and spec.users.acting_user != acting:
        problems.append(f"users.acting_user: the spec says {spec.users.acting_user}, the seed's current_user {acting}")
    return problems


def _shown(route: tuple) -> str:
    method, path, roles = route
    return f"{method} {path}" + (f" for {list(roles)}" if roles else "")


@dataclasses.dataclass(frozen=True)
class NameEntry:
    """One host name pattern of a multi env's name table: the site it routes to and how that site is reached."""

    pattern: str
    site: str
    kind: NameKind

    def matches(self, host: str) -> bool:
        host = host.lower().rstrip(".")
        if self.pattern.startswith("*."):
            return host.endswith(self.pattern[1:])
        return host == self.pattern

    @property
    def specificity(self) -> tuple[int, int]:
        return (0, len(self.pattern)) if self.pattern.startswith("*.") else (1, len(self.pattern))


@dataclasses.dataclass(frozen=True)
class NameTable:
    """A multi env's host names: each pattern's winning entry, and each entry a higher kind shadowed, beside the
    entry it lost to."""

    entries: tuple[NameEntry, ...]
    shadowed: tuple[tuple[NameEntry, NameEntry], ...] = ()

    def resolve(self, host: str) -> Optional[NameEntry]:
        """The entry ``host`` routes to: of those matching it, the highest kind's (synthetic, live, allowlist),
        and among those the most specific pattern (an exact name, then the longest wildcard)."""
        matching = [e for e in self.entries if e.matches(host)]
        if not matching:
            return None
        return min(matching, key=lambda e: (_PRECEDENCE[e.kind], tuple(-n for n in e.specificity)))

    def as_dict(self) -> dict[str, dict]:
        return {e.pattern: {"site": e.site, "kind": e.kind} for e in self.entries}


def name_table(specs: Iterable[WebsiteSpec]) -> NameTable:
    """The one table of host names of a multi env's sites: each site's domains, ``synthetic`` for one the env
    serves (image, sdk, static, replay) and ``live`` for a live source, and its source's ``allow_hosts`` as
    ``allowlist``. A pattern two kinds claim goes to the higher, synthetic before live before allowlist; two sites
    claiming one pattern at one kind is a conflict, but for an allowlist, which only lets traffic through. Raises
    SpecError listing every conflict."""
    claims: dict[str, list[NameEntry]] = {}
    problems = []
    names: set[str] = set()
    for spec in specs:
        if spec.site.name is None:
            problems.append(f"a site with domains {list(spec.site.domains)} has no site.name")
            continue
        if spec.site.name in names:
            problems.append(f"two sites are named {spec.site.name}")
        names.add(spec.site.name)
        source = spec.source
        kind: NameKind = "live" if source is not None and source.kind == "live" else "synthetic"
        allowed = (source.live.allow_hosts if source is not None and source.live is not None
                   else source.replay.allow_hosts if source is not None and source.replay is not None else ())
        for pattern in spec.site.domains:
            claims.setdefault(pattern, []).append(NameEntry(pattern, spec.site.name, kind))
        for pattern in allowed:
            claims.setdefault(pattern, []).append(NameEntry(pattern, spec.site.name, "allowlist"))
    entries, shadowed = [], []
    for pattern, claimed in claims.items():
        clashes = [(kind, sorted({e.site for e in claimed if e.kind == kind})) for kind in ("synthetic", "live")]
        clashes = [(kind, sites) for kind, sites in clashes if len(sites) > 1]
        problems.extend(f"{pattern}: claimed as {kind} by {', '.join(sites)}" for kind, sites in clashes)
        if clashes:
            continue
        best = min(_PRECEDENCE[e.kind] for e in claimed)
        top = list(dict.fromkeys(e for e in claimed if _PRECEDENCE[e.kind] == best))
        winner = top[0]
        entries.append(winner)
        shadowed.extend((e, winner) for e in dict.fromkeys(claimed) if e != winner and e.site != winner.site)
    if problems:
        raise SpecError(problems)
    return NameTable(tuple(entries), tuple(shadowed))


def json_schema() -> dict:
    """The JSON Schema of a spec file, with the string shorthands a route and a persona take, the partial route
    table an overlay changes an inherited route's keys with, and the ``false`` that removes an inherited route, role
    or collection."""
    schema = WebsiteSpec.model_json_schema(by_alias=True)
    defs = schema.get("$defs", {})
    route_ref = {"$ref": "#/$defs/Route"}
    shorthand = {"type": "string", "pattern": f"^({'|'.join(METHODS)}) /", "description": "METHOD /path"}
    removed = {"const": False, "description": "removes the inherited entry"}
    partial = {**{k: v for k, v in defs["Route"].items() if k != "required"},
               "title": "RouteChange", "description": "keys of an inherited route an overlay changes"}
    defs["Routes"]["properties"]["ops"]["additionalProperties"] = {"anyOf": [route_ref, shorthand, removed, partial]}
    for holder in (schema["properties"]["roles"], defs["Data"]["properties"]["collections"]):
        holder["additionalProperties"] = {"anyOf": [holder["additionalProperties"], removed]}
    defs["Users"]["properties"]["personas"]["items"] = {"anyOf": [{"$ref": "#/$defs/Persona"}, {"type": "string"}]}
    defs["Data"]["properties"]["seed"] = {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                                          "default": []}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "urn:agentenv:website/v1:spec",
            **schema}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentenv_protocol.website.spec",
                                     description="Validate, show or describe a website/v1 spec.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="check a spec and everything it extends and uses").add_argument("path")
    show = commands.add_parser("show", help="print the resolved spec as JSON")
    show.add_argument("path")
    show.add_argument("--card", action="store_true", help="print the website/v1 card params it advertises instead")
    commands.add_parser("schema", help="print the JSON Schema of a spec file")
    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(json_schema(), indent=2))
        return 0
    try:
        spec = load(args.path)
    except SpecError as e:
        print("\n".join(e.problems), file=sys.stderr)
        return 1
    if args.command == "show":
        print(json.dumps(spec.to_card_extension() if args.card else spec.to_dict(), indent=2))
        return 0
    source = spec.source.kind if spec.source is not None else "no source"
    mounted = f", {len(spec.components)} component(s)" if spec.components else ""
    print(f"{args.path}: ok: {spec.site.name or 'unnamed site'}, {source}, {len(spec.routes.ops)} route(s), "
          f"{len(spec.data.collections)} collection(s){mounted}")
    return 0
