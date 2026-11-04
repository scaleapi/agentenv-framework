"""The website spec: loading and validating one, ``extends`` overlays, ``uses`` composition and its conflicts, a multi
env's name table, the card round trip, a spec derived from a WebsiteEnvironment class and a class declared from one,
and the conformance checker run against a spec."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from agentenv_protocol import environment_card
from agentenv_protocol.website import WebsiteEnvironment, api, conformance, website
from agentenv_protocol.website import spec as website_spec
from agentenv_protocol.website.spec import SpecError, from_card, from_environment, load, name_table, resolve

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "website"
NOTES_DIR = EXAMPLES / "notes"
SPECS_DIR = EXAMPLES / "specs"


def _notes_module():
    spec = importlib.util.spec_from_file_location("notes_spec_example_site", NOTES_DIR / "site.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


notes_site = _notes_module()


def write(folder: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)
    return folder / next(iter(files))


def problems(path_or_raw, base_dir: Path | None = None) -> list[str]:
    with pytest.raises(SpecError) as caught:
        load(path_or_raw) if base_dir is None else resolve(path_or_raw, base_dir)
    return caught.value.problems


def card_of(client: TestClient) -> dict:
    return client.get("/.well-known/agent-env.json").json()


def website_params(card: dict) -> dict:
    params = next(e for e in card["capabilities"]["extensions"] if e["uri"] == website_spec.WEBSITE_EXTENSION_URI)["params"]
    return {k: v for k, v in params.items() if k != "methods"}


@pytest.fixture
def notes_client(tmp_path: Path) -> TestClient:
    (tmp_path / "index.html").write_text("<!doctype html><html><head></head><body></body></html>")
    spec = notes_site.Notes.declaration().spec.overlay({"source": {"sdk": "site:Notes", "assets": str(tmp_path)}})
    return TestClient(website(spec=spec)(type("Notes", (notes_site.Notes,), {}))().create_app())


# Loading


def test_the_notes_spec_loads_with_its_routes_roles_personas_and_data() -> None:
    spec = load(NOTES_DIR / "website.toml")
    assert (spec.site.name, spec.site.domains, spec.health) == ("notes", ("notes.test",), "/healthz")
    assert (spec.source.kind, spec.source.sdk, spec.path(spec.source.assets)) == ("sdk", "site:Notes",
                                                                                  NOTES_DIR / "web" / "dist")
    assert [p.id for p in spec.users.personas] == ["ann", "ben"] and spec.users.acting_user == "ann"
    assert spec.data.collections["notes"].owner == "owner" and spec.data.seed == ("data/seed.json",)
    table = spec.route_table()
    assert list(table)[:3] == ["list_users", "sign_in", "list_notes"]
    assert (table["get_note"].method, table["get_note"].path) == ("GET", "/api/notes/{note_id}")
    assert table["all_notes"].roles == ("admin",) and table["list_notes"].roles == ()
    assert spec.clock.follow == "header"


def test_every_section_is_optional_and_a_route_takes_a_table_or_a_shorthand(tmp_path: Path) -> None:
    assert load(write(tmp_path, {"empty.toml": ""})).to_card_extension() == {
        "domains": [], "health": "/api/health", "routes": [], "session_cookie": "agentenv_session",
        "session_path": "/api/session"}
    spec = load(write(tmp_path, {"r.toml": '[routes.ops]\na = "get /x/{id:int}"\n'
                                           'b = { method = "post", path = "/x", roles = ["admin"] }\n'}))
    assert [r.advertised(n) for n, r in spec.route_table().items()] == [
        {"method": "GET", "path": "/x/{id}", "name": "a", "roles": None},
        {"method": "POST", "path": "/x", "name": "b", "roles": ["admin"]}]


def test_routes_come_from_an_openapi_document_by_operation_id(tmp_path: Path) -> None:
    write(tmp_path, {"openapi.json": json.dumps({"paths": {
        "/api/items": {"get": {"operationId": "list_items"}, "post": {"operationId": "add_item",
                                                                      "x-agentenv-roles": ["admin"]}}}})})
    spec = load(write(tmp_path, {"s.toml": '[routes]\nopenapi = "openapi.json"\n[routes.ops]\nping = "GET /ping"\n'}))
    assert {n: (r.method, r.path, r.roles) for n, r in spec.route_table().items()} == {
        "list_items": ("GET", "/api/items", ()), "add_item": ("POST", "/api/items", ("admin",)),
        "ping": ("GET", "/ping", ())}
    clash = write(tmp_path, {"c.toml": '[routes]\nopenapi = "openapi.json"\n[routes.ops]\nlist_items = "GET /other"\n'})
    assert any("list_items: GET /other isn't the openapi document's GET /api/items" in p for p in problems(clash))


def test_an_openapi_document_of_another_shape_or_with_a_name_twice_is_refused(tmp_path: Path) -> None:
    def found(document) -> list[str]:
        write(tmp_path, {"openapi.json": json.dumps(document)})
        return problems(write(tmp_path, {"s.toml": '[routes]\nopenapi = "openapi.json"\n'}))

    shape = "openapi.json isn't an OpenAPI document, an object whose paths are an object"
    assert any(shape in p for p in found([]))
    assert any(shape in p for p in found({"paths": ["/a"]}))
    assert any("path /a isn't an object of operations" in p for p in found({"paths": {"/a": ["get"]}}))
    assert any("routes.openapi: items names both GET /a and GET /b" in p for p in found(
        {"paths": {"/a": {"get": {"operationId": "items"}}, "/b": {"get": {"operationId": "items"}}}}))


def test_false_removes_an_operation_of_the_openapi_document(tmp_path: Path) -> None:
    write(tmp_path, {"openapi.json": json.dumps({"paths": {"/api/items": {
        "get": {"operationId": "list_items"}, "post": {"operationId": "add_item"}}}}),
        "base.toml": '[routes]\nopenapi = "openapi.json"\n'})
    spec = load(tmp_path / "base.toml")
    assert set(spec.route_table()) == {"list_items", "add_item"} and spec.routes.openapi is None
    assert set(spec.overlay({"routes": {"ops": {"add_item": False}}}).route_table()) == {"list_items"}
    child = load(write(tmp_path, {"child.toml": 'extends = "base.toml"\n[routes.ops]\nadd_item = false\n'}))
    assert set(child.route_table()) == {"list_items"}
    assert set(resolve(child.to_dict(), tmp_path).route_table()) == {"list_items"}


@pytest.mark.parametrize("toml, problem", [
    ("[site]\ncolor = 1\n", "site.color: isn't a key the spec takes"),
    ("[pages]\n", "pages: isn't a key the spec takes"),
    ('[source]\nsdk = "a:B"\nstatic = "dist"\n', "source: names exactly one of image, sdk, static, live, replay, "
                                                  "not ['sdk', 'static']"),
    ("[source]\n", "source: names exactly one of image, sdk, static, live, replay, not none"),
    ('[source]\nstatic = "dist"\nassets = "x"\n', "source: assets is the static folder of an sdk source"),
    ('[source]\nsdk = "nocolon"\n', "source: sdk is module:Class, not 'nocolon'"),
    ('[site]\ndomains = ["ok.test", "Bad.test", "a.*.test"]\n', "site.domains: ['Bad.test', 'a.*.test'] aren't host"),
    ('[site]\nhealth = "/agentenv"\n', "site.health: health is a path of its own under '/', not '/agentenv'"),
    ('[source.live]\ncredentials = { token = "xoxp-123" }\n', "source.live.credentials: credentials ['token'] must be"),
    ('[users]\npersonas = ["ann"]\nacting_user = "zed"\n', "users: acting_user 'zed' is none of the personas ['ann']"),
    ('[users]\npersonas = ["ann", "ann"]\n', "users: personas ['ann'] are declared twice"),
    ('[users]\nsign_in = "sso"\n', "users.sign_in: Input should be 'cookie', 'none' or 'live'"),
    ('[routes.ops]\na = "FETCH /x"\n', "routes.ops.a: method is one of GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS, not 'FETCH'"),
    ('[routes.ops]\na = "GET /agentenv/x"\n', "routes.ops.a: path '/agentenv/x' is the data plane's or the card's"),
    ('[routes.ops]\na = "GET /x"\nb = "get /x"\n', "routes.ops: b and a are both GET /x"),
    ('[site]\nhealth = "/up"\n[routes.ops]\nup = "GET /up"\n', "site.health: /up is served by route up"),
    ('[roles]\nadmin = ["nope"]\n', "roles.admin: names no route ['nope']"),
    ('[source]\nsdk = "a:B"\n[users]\nsign_in = "live"\n', "users.sign_in: live is the sign-in of a live or replayed"),
    ('[source]\nsdk = "a:B"\n[clock]\nfollow = "none"\n', "clock.follow: an sdk site follows the universe's clock"),
    ('[source]\nstatic = "dist"\n[routes.ops]\na = "GET /a"\n', "routes: a static site serves no routes"),
    ('[source]\nsdk = "a:B"\n[routes.ops]\na = { method = "GET", path = "/a", served_by = "x:Y" }\n',
     "routes.ops: served_by swaps a route of a live or replayed site"),
    ('[data]\nschema = "s.json"\nmodels = "mcp:mail"\n', "data: data takes a schema or models, not both"),
    ('[data]\nmodels = "mail"\n', "data.models: models is mcp:<server>, not 'mail'"),
    ('[users]\nsign_in = "none"\n[data.collections.notes]\nowner = "owner"\n',
     "data.collections.notes: owner names a user, and the site signs nobody in"),
    ('[[uses]]\nspec = "c.toml"\nmount = "/.well-known/agent-env.json"\n', "uses[0].mount: mount is a plain path"),
    ("[site\n", "isn't TOML"),
], ids=lambda value: value[:24] if isinstance(value, str) else value)
def test_a_spec_that_does_not_hold_together_is_refused_naming_where(tmp_path: Path, toml: str, problem: str) -> None:
    found = problems(write(tmp_path, {"s.toml": toml}))
    assert any(problem in p for p in found), found
    assert all(p.startswith(str(tmp_path / "s.toml")) for p in found)


def test_a_missing_spec_file_is_a_problem(tmp_path: Path) -> None:
    assert "can't be read" in problems(tmp_path / "nope.toml")[0]


def test_to_dict_reads_back_as_the_same_spec() -> None:
    spec = load(NOTES_DIR / "website.toml")
    again = resolve(spec.to_dict(), NOTES_DIR)
    assert again == spec and again.to_card_extension() == spec.to_card_extension()


def test_to_dict_and_an_overlay_keep_a_json_schema_as_it_is(tmp_path: Path) -> None:
    closed = {"type": "object", "additionalProperties": False, "properties": {"id": {}}, "required": []}
    spec = resolve({"data": {"schema": {"properties": {}}, "collections": {"items": {"schema": closed},
                                                                            "any": {"schema": {}}}}}, tmp_path)
    data = spec.to_dict()["data"]
    assert (data["schema"], data["collections"]["items"]["schema"], data["collections"]["any"]["schema"]) == (
        {"properties": {}}, closed, {})
    assert resolve(spec.to_dict(), tmp_path) == spec
    assert spec.overlay({"site": {"name": "x"}}).data.collections["items"].schema_ == closed


# extends


BASE = """
[site]
name = "shop"
domains = ["shop.test", "www.shop.test"]
health = "/healthz"

[source]
sdk = "shop:Shop"
assets = "dist"

[data]
seed = ["seed.json"]

[data.collections.orders]
owner = "buyer"

[data.collections.carts]

[users]
personas = ["ann", "ben"]
acting_user = "ann"

[roles]
admin = ["refund"]
staff = ["refund"]

[routes.ops]
list_orders = "GET /api/orders"
refund = "POST /api/orders/{id}/refund"
"""


def test_an_overlay_changes_one_key_of_an_operation_of_the_openapi_document(tmp_path: Path) -> None:
    write(tmp_path, {"openapi.json": json.dumps({"paths": {"/api/items": {
        "get": {"operationId": "list_items"}, "post": {"operationId": "add_item", "x-agentenv-roles": ["admin"]}}}}),
        "base.toml": '[routes]\nopenapi = "openapi.json"\n'})
    child = load(write(tmp_path, {"child.toml": 'extends = "base.toml"\n[source.live]\n[routes.ops]\n'
                                                'add_item = { roles = ["staff"] }\n'
                                                'list_items = { served_by = "shop:Shop" }\n'}))
    routes = child.route_table()
    assert (routes["add_item"].method, routes["add_item"].path, routes["add_item"].roles) == (
        "POST", "/api/items", ("staff",))
    assert (routes["list_items"].method, routes["list_items"].path, routes["list_items"].served_by) == (
        "GET", "/api/items", "shop:Shop")
    assert load(tmp_path / "base.toml").overlay({"routes": {"ops": {"add_item": {"roles": ["staff"]}}}}).route_table()[
        "add_item"].roles == ("staff",)
    moved = write(tmp_path, {"moved.toml": 'extends = "base.toml"\n[routes.ops]\nadd_item = { path = "/api/other" }\n'})
    assert any("add_item: POST /api/other isn't the openapi document's POST /api/items" in p for p in problems(moved))


def test_extends_merges_tables_replaces_arrays_and_source_and_false_removes(tmp_path: Path) -> None:
    write(tmp_path, {"base/shop.toml": BASE})
    child = write(tmp_path, {"variants/ben.toml": """
extends = "../base/shop.toml"

[site]
domains = ["shop.example"]

[source.image]
dockerfile = "Dockerfile"

[data.collections]
orders = false

[data.collections.carts]
key = "cart_id"

[users]
personas = ["ben", "cy"]
acting_user = "ben"

[roles]
staff = false

[routes.ops]
refund = { roles = ["manager"] }
list_orders = false
place_order = "POST /api/orders"
"""})
    spec = load(child)
    assert (spec.site.name, spec.site.domains, spec.health) == ("shop", ("shop.example",), "/healthz")
    assert spec.source.kind == "image" and spec.source.assets is None
    assert spec.source.image.dockerfile == "Dockerfile" and spec.base_dir == (tmp_path / "variants").resolve()
    assert spec.path(spec.data.seed[0]).resolve() == (tmp_path / "base" / "seed.json").resolve()
    assert list(spec.data.collections) == ["carts"] and spec.data.collections["carts"].key == "cart_id"
    assert [p.id for p in spec.users.personas] == ["ben", "cy"] and spec.users.acting_user == "ben"
    assert list(spec.roles) == ["admin"]
    assert {n: (r.method, r.path, r.roles) for n, r in spec.route_table().items()} == {
        "refund": ("POST", "/api/orders/{id}/refund", ("manager", "admin")),
        "place_order": ("POST", "/api/orders", ())}
    assert spec.extends is None


def test_an_inherited_image_keeps_its_default_context_in_its_own_folder(tmp_path: Path) -> None:
    write(tmp_path, {"base/site.toml": '[site]\nname = "shop"\n[source.image]\ndockerfile = "Dockerfile"\n'})
    implicit = load(write(tmp_path, {"variants/ben.toml": 'extends = "../base/site.toml"\n'}))
    write(tmp_path, {"base/explicit.toml": '[site]\nname = "shop"\n[source.image]\ndockerfile = "Dockerfile"\n'
                                           'context = "."\n'})
    explicit = load(write(tmp_path, {"variants/cy.toml": 'extends = "../base/explicit.toml"\n'}))
    assert implicit.path(implicit.source.image.context).resolve() == (tmp_path / "base").resolve()
    assert implicit.source.image == explicit.source.image


def test_an_extends_cycle_is_refused(tmp_path: Path) -> None:
    write(tmp_path, {"a.toml": 'extends = "b.toml"\n', "b.toml": 'extends = "a.toml"\n'})
    assert "is a cycle" in problems(tmp_path / "a.toml")[0]


@pytest.mark.parametrize(("files", "problem"), [
    ({"s.toml": 'extends = "base.toml"\nuses = 1\n', "base.toml": '[[uses]]\nspec = "c.toml"\n'}, "uses: "),
    ({"s.toml": 'extends = "base/base.toml"\n', "base/base.toml": "[data]\ncollections = [1]\n"}, "data.collections: "),
], ids=["uses", "collections"])
def test_an_extends_whose_fields_have_the_wrong_type_is_refused_naming_them(tmp_path: Path, files: dict,
                                                                           problem: str) -> None:
    found = problems(write(tmp_path, files))
    assert any(problem in p for p in found), found


def test_overlay_merges_over_a_loaded_spec_as_extends_does(tmp_path: Path) -> None:
    spec = load(write(tmp_path, {"shop.toml": BASE}))
    changed = spec.overlay({"users": {"acting_user": "ben"}, "routes": {"ops": {"refund": {"roles": []}}}})
    assert changed.users.acting_user == "ben" and changed.route_table()["refund"].roles == ("admin", "staff")
    assert spec.users.acting_user == "ann"


# uses


SSO = """
[site]
name = "sso"
domains = ["sso.test"]

[source]
sdk = "sso:FakeSso"

[data.collections.accounts]

[data.collections.grants]
owner = "user"

[users]
personas = [{ id = "ann", name = "Ann" }]

[roles]
admin = ["revoke"]

[routes.ops]
login = "POST /login"
userinfo = "GET /userinfo"
revoke = "POST /revoke"
"""

HOST = """
[site]
name = "intranet"
domains = ["intranet.test"]

[source.image]
dockerfile = "Dockerfile"

[data.collections.users]

[users]
personas = ["ann", "ben"]

[roles]
editor = ["sso.login"]

[routes.ops]
pages = "GET /api/pages"
"""


def test_uses_mounts_a_component_prefixing_its_routes_and_namespacing_its_data(tmp_path: Path) -> None:
    write(tmp_path, {"components/sso.toml": SSO})
    spec = load(write(tmp_path, {"site.toml": HOST + '[[uses]]\nspec = "components/sso.toml"\nmount = "/auth"\n'
                                                     'data = { accounts = "users" }\n'}))
    table = spec.route_table()
    assert {n: (r.method, r.path, r.roles) for n, r in table.items()} == {
        "pages": ("GET", "/api/pages", ()), "sso.login": ("POST", "/auth/login", ("editor",)),
        "sso.userinfo": ("GET", "/auth/userinfo", ()), "sso.revoke": ("POST", "/auth/revoke", ("admin",))}
    assert list(spec.data.collections) == ["users", "sso.grants"]
    assert spec.data.collections["sso.grants"].owner == "user"
    assert [p.id for p in spec.users.personas] == ["ann", "ben"]
    assert spec.site.domains == ("intranet.test",) and spec.uses == ()
    (component,) = spec.components
    assert (component.name, component.mount, component.spec.site.name) == ("sso", "/auth", "sso")
    assert [r["name"] for r in spec.to_card_extension()["routes"]] == ["pages", "sso.login", "sso.userinfo",
                                                                       "sso.revoke"]


def test_uses_names_a_component_by_as_and_mounts_it_at_the_root(tmp_path: Path) -> None:
    write(tmp_path, {"sso.toml": SSO})
    spec = load(write(tmp_path, {"site.toml": HOST.replace("sso.login", "auth.login")
                                 + '[[uses]]\nspec = "sso.toml"\nas = "auth"\n'}))
    assert spec.route_table()["auth.login"].path == "/login" and "auth.accounts" in spec.data.collections


def test_a_mounted_route_keeps_its_trailing_slash(tmp_path: Path) -> None:
    write(tmp_path, {"sso.toml": SSO.replace('revoke = "POST /revoke"', 'revoke = "POST /revoke"\nhome = "GET /"\n'
                                                                       'items = "GET /items/"')})
    table = load(write(tmp_path, {"site.toml": HOST + '[[uses]]\nspec = "sso.toml"\nmount = "/auth"\n'})).route_table()
    assert (table["sso.home"].path, table["sso.items"].path, table["sso.login"].path) == ("/auth/", "/auth/items/",
                                                                                          "/auth/login")


def test_a_sites_acting_user_may_be_a_persona_only_its_component_declares(tmp_path: Path) -> None:
    write(tmp_path, {"sso.toml": SSO})
    host = HOST.replace('personas = ["ann", "ben"]', 'personas = ["ben"]\nacting_user = "ann"')
    spec = load(write(tmp_path, {"site.toml": host + '[[uses]]\nspec = "sso.toml"\nmount = "/auth"\n'}))
    assert ([p.id for p in spec.users.personas], spec.users.acting_user) == (["ben", "ann"], "ann")
    write(tmp_path, {"site.toml": host.replace('acting_user = "ann"', 'acting_user = "zed"')
                     + '[[uses]]\nspec = "sso.toml"\nmount = "/auth"\n'})
    assert problems(tmp_path / "site.toml") == [
        f"{tmp_path / 'site.toml'}: users: acting_user 'zed' is none of the personas ['ben', 'ann']"]


@pytest.mark.parametrize("host, component, problem", [
    (HOST + '[routes.ops.mine]\nmethod = "POST"\npath = "/auth/login"\n', SSO,
     "uses[0]: route mine (POST /auth/login) is under /auth, where sso is mounted"),
    (HOST.replace('pages = "GET /api/pages"', 'pages = "GET /userinfo"'), SSO.replace("[site]", "[site]\n"),
     "uses[0]: sso's route userinfo is GET /userinfo, which pages already serves"),
    (HOST + '[routes.ops."sso.login"]\nmethod = "GET"\npath = "/x"\n', SSO,
     "uses[0]: route sso.login is declared by the site and by component sso"),
    (HOST + '[data.collections.people]\nkey = "email"\n', SSO,
     "uses[0]: sso's collection accounts maps onto people, whose key differ"),
    (HOST, SSO.replace('[data.collections.grants]\nowner = "user"', '[data.collections.grants]\nowner = "owner"'),
     "uses[0]: sso's collection grants maps onto sso.grants"),
    (HOST.replace('personas = ["ann", "ben"]', 'personas = ["ann", "ben"]\nsign_in = "none"'), SSO,
     "uses[0]: sso signs in by cookie, the site by none"),
    (HOST.replace('personas = ["ann", "ben"]', 'personas = [{ id = "ann", name = "Annie" }]'), SSO,
     "uses[0]: persona ann is declared otherwise by sso"),
    (HOST.replace('editor = ["sso.login"]', 'editor = ["sso.nope"]'), SSO, "roles.editor: names no route ['sso.nope']"),
    (HOST, SSO.replace('name = "sso"\n', ""), "uses[0]: components/sso.toml has no site.name; name it with as"),
    (HOST, SSO.replace('login = "POST /login"', 'login = "FETCH /login"'), "routes.ops.login: method is one of"),
], ids=["host-route-under-mount", "same-path", "same-name", "collection-key", "collection-owner", "sign-in",
        "persona", "role-of-component-route", "unnamed", "component-invalid"])
def test_a_composition_that_clashes_is_refused(tmp_path: Path, host: str, component: str, problem: str) -> None:
    mount = "/" if "already serves" in problem else "/auth"
    mapping = 'data = { accounts = "people" }\n' if "people" in problem else ""
    if "grants maps onto sso.grants" in problem:
        host += '[data.collections."sso.grants"]\nowner = "user"\n'
    write(tmp_path, {"components/sso.toml": component})
    found = problems(write(tmp_path, {"site.toml": host + f'[[uses]]\nspec = "components/sso.toml"\nmount = "{mount}"\n'
                                                          + mapping}))
    assert any(problem in p for p in found), found


def test_a_components_schema_file_stays_its_own_once_mounted(tmp_path: Path) -> None:
    write(tmp_path, {"components/sso.toml": SSO.replace("[data.collections.accounts]",
                                                        '[data.collections.accounts]\nschema = "accounts.json"')})
    spec = load(write(tmp_path, {"site.toml": HOST + '[[uses]]\nspec = "components/sso.toml"\n'}))
    assert spec.path(spec.data.collections["sso.accounts"].schema_).resolve() == tmp_path / "components" / "accounts.json"
    mapped = load(write(tmp_path, {"mapped.toml": HOST.replace("[data.collections.users]",
                                                               '[data.collections.users]\nschema = "components/accounts.json"')
                                   + '[[uses]]\nspec = "components/sso.toml"\ndata = { accounts = "users" }\n'}))
    assert mapped.data.collections["users"].schema_ == "components/accounts.json"


def test_a_mapping_of_a_collection_the_component_lacks_and_a_uses_cycle_are_refused(tmp_path: Path) -> None:
    write(tmp_path, {"sso.toml": SSO})
    assert any("data maps ['nope'], which sso doesn't declare" in p for p in problems(
        write(tmp_path, {"a.toml": HOST + '[[uses]]\nspec = "sso.toml"\ndata = { nope = "users" }\n'})))
    write(tmp_path, {"x.toml": '[site]\nname = "x"\n[[uses]]\nspec = "y.toml"\n',
                     "y.toml": '[site]\nname = "y"\n[[uses]]\nspec = "x.toml"\n'})
    assert any("is a cycle" in p for p in problems(tmp_path / "x.toml"))


def test_an_overlay_adds_a_component_and_replaces_one_of_the_same_name(tmp_path: Path) -> None:
    write(tmp_path, {"sso.toml": SSO, "base.toml": HOST + '[[uses]]\nspec = "sso.toml"\nmount = "/a"\nas = "sso"\n'})
    spec = load(write(tmp_path, {"child.toml": 'extends = "base.toml"\n[[uses]]\nspec = "sso.toml"\nmount = "/b"\n'
                                              'as = "sso"\n[[uses]]\nspec = "sso.toml"\nmount = "/c"\nas = "backup"\n'}))
    assert [(c.name, c.mount) for c in spec.components] == [("sso", "/b"), ("backup", "/c")]
    assert (spec.route_table()["sso.login"].path, spec.route_table()["backup.login"].path) == ("/b/login", "/c/login")


def test_overlay_replaces_a_mounted_component_as_a_file_overlay_does(tmp_path: Path) -> None:
    write(tmp_path, {"sso.toml": SSO, "base.toml": HOST + '[[uses]]\nspec = "sso.toml"\nmount = "/a"\nas = "sso"\n'})
    by_file = load(write(tmp_path, {"child.toml": 'extends = "base.toml"\n[[uses]]\nspec = "sso.toml"\nmount = "/b"\n'
                                                 'as = "sso"\n'}))
    by_code = load(tmp_path / "base.toml").overlay({"uses": [{"spec": "sso.toml", "mount": "/b", "as": "sso"}]})
    assert [(c.name, c.mount) for c in by_code.components] == [("sso", "/b")]
    assert by_code.to_dict() == by_file.to_dict()


# The name table


def _site(name: str, domains: list[str], source: dict | None = None) -> website_spec.WebsiteSpec:
    raw: dict = {"site": {"name": name, "domains": domains}}
    if source is not None:
        raw["source"] = source
        if "live" in source or "replay" in source:
            raw["users"] = {"sign_in": "live"}
    return resolve(raw, Path.cwd())


def test_the_name_table_puts_synthetic_before_live_before_allowlisted() -> None:
    slack = load(SPECS_DIR / "slack.toml")
    status = _site("status", ["status.slack.com", "*.slack-edge.com"], {"live": {"allow_hosts": ["cdn.status.io"]}})
    notes = load(NOTES_DIR / "website.toml")
    table = name_table([slack, status, notes])
    assert table.as_dict() == {
        "slack.com": {"site": "slack", "kind": "synthetic"}, "*.slack.com": {"site": "slack", "kind": "synthetic"},
        "*.slack-edge.com": {"site": "status", "kind": "live"}, "status.slack.com": {"site": "status", "kind": "live"},
        "cdn.status.io": {"site": "status", "kind": "allowlist"}, "notes.test": {"site": "notes", "kind": "synthetic"}}
    assert [(lost.site, lost.kind, won.site) for lost, won in table.shadowed] == [("slack", "allowlist", "status")]
    assert table.resolve("status.slack.com").site == "slack"
    assert table.resolve("files.slack-edge.com").kind == "live"
    assert table.resolve("NOTES.test.").site == "notes" and table.resolve("slack.org") is None


def test_the_name_table_prefers_the_most_specific_pattern_of_one_kind() -> None:
    table = name_table([_site("mail", ["*.example.test"]), _site("inbox", ["inbox.example.test"]),
                        _site("deep", ["*.b.example.test"])])
    assert [table.resolve(h).site for h in ("x.example.test", "inbox.example.test", "a.b.example.test")] == [
        "mail", "inbox", "deep"]
    assert table.resolve("example.test") is None


def test_the_name_table_refuses_two_sites_claiming_one_name_at_one_kind() -> None:
    with pytest.raises(SpecError) as caught:
        name_table([_site("a", ["shop.test", "*.x.test"]), _site("b", ["shop.test"]), _site("c", ["*.x.test"]),
                    _site("a", []), resolve({"site": {"domains": ["q.test"]}}, Path.cwd()),
                    _site("l1", ["live.test"], {"live": {}}), _site("l2", ["live.test"], {"live": {}})])
    assert caught.value.problems == [
        "two sites are named a", "a site with domains ['q.test'] has no site.name",
        "shop.test: claimed as synthetic by a, b", "*.x.test: claimed as synthetic by a, c",
        "live.test: claimed as live by l1, l2"]



def test_the_name_table_refuses_two_live_sites_on_one_name_a_synthetic_site_also_claims() -> None:
    with pytest.raises(SpecError) as caught:
        name_table([_site("l1", ["shop.test"], {"live": {}}), _site("l2", ["shop.test"], {"live": {}}),
                    _site("s", ["shop.test"])])
    assert caught.value.problems == ["shop.test: claimed as live by l1, l2"]


def test_two_sites_may_allowlist_one_host() -> None:
    table = name_table([_site("a", [], {"live": {"allow_hosts": ["cdn.test"]}}),
                        _site("b", [], {"replay": {"capture": "c.har", "allow_hosts": ["cdn.test"]}})])
    assert table.as_dict() == {"cdn.test": {"site": "a", "kind": "allowlist"}}


# The card


def test_a_spec_round_trips_through_the_card_it_advertises() -> None:
    spec = load(NOTES_DIR / "website.toml")
    card = {"name": "notes", "capabilities": {"extensions": [
        {"uri": website_spec.WEBSITE_EXTENSION_URI, "params": spec.to_card_extension()}]}}
    back = from_card(card)
    assert back.to_card_extension() == spec.to_card_extension()
    assert (back.site, back.users.sign_in, back.clock) == (spec.site, "cookie", spec.clock)
    assert back.source is None and back.data.collections == {}


def test_from_card_reads_a_running_sites_card_and_a_card_without_a_session() -> None:
    card = {"name": "docs", "capabilities": {"extensions": [{"uri": website_spec.WEBSITE_EXTENSION_URI, "params": {
        "domains": ["docs.test"], "health": "/up", "routes": [{"method": "GET", "path": "/a", "name": "a"}]}}]}}
    spec = from_card(card)
    assert (spec.site.name, spec.health, spec.users.sign_in, spec.clock) == ("docs", "/up", "none", None)
    assert spec.to_card_extension() == card["capabilities"]["extensions"][0]["params"] | {
        "routes": [{"method": "GET", "path": "/a", "name": "a", "roles": None}]}
    with pytest.raises(SpecError, match="declares no urn:agentenv:website/v1"):
        from_card({"name": "x"})


def test_the_notes_sites_card_is_its_spec(notes_client: TestClient) -> None:
    from agentenv_protocol.website.routes import openapi_path

    card = card_of(notes_client)
    params = website_params(card)
    assert card["name"] == "notes"
    assert {r["name"]: r["path"] for r in params["routes"]}["get_note"] == "/api/notes/{note_id:int}"
    templated = [{**r, "path": openapi_path(r["path"])} for r in params["routes"]]
    assert {**params, "routes": templated} == load(NOTES_DIR / "website.toml").to_card_extension()
    assert from_card(card).to_card_extension() == {**params, "routes": templated}


# WebsiteEnvironment


def test_from_environment_derives_the_notes_spec_from_its_class() -> None:
    derived = from_environment(notes_site.Notes)
    declared = load(NOTES_DIR / "website.toml")
    assert derived.to_card_extension() == declared.to_card_extension()
    assert (derived.source.kind, derived.source.sdk) == ("sdk", f"{notes_site.__name__}:Notes")
    assert derived.users.acting_user == "ann" and derived.path(derived.data.seed[0]) == NOTES_DIR / "data" / "seed.json"


def test_from_environment_derives_a_spec_from_decorators_alone(tmp_path: Path) -> None:
    (tmp_path / "seed.json").write_text('{"current_user": "zoe", "items": []}')

    @environment_card(name="shop")
    @website(domains=["shop.test"], health="/up", data=str(tmp_path / "seed.json"), clock_patch=True)
    class Shop(WebsiteEnvironment):
        @api.get("/api/items", roles=["admin"])
        def items(self) -> list:
            return []

    spec = from_environment(Shop)
    assert (spec.site.name, spec.site.domains, spec.health, spec.users.acting_user) == ("shop", ("shop.test",), "/up",
                                                                                         "zoe")
    assert (spec.clock.follow, spec.route_table()["items"].roles) == ("date_patch", ("admin",))
    assert website_params(card_of(TestClient(Shop().create_app()))) == spec.to_card_extension()


def test_from_environment_takes_any_url_path_a_route_serves() -> None:
    @environment_card(name="chat")
    @website(domains=["chat.test"])
    class Chat(WebsiteEnvironment):
        @api.get("/api/users/@me")
        def me(self) -> dict:
            return {"id": "me"}

        @api.get("/api/tags/c++,%2F")
        def tags(self) -> list:
            return []

    assert {r.path for r in from_environment(Chat).route_table().values()} == {"/api/users/@me", "/api/tags/c++,%2F"}
    assert TestClient(Chat().create_app()).get("/api/users/@me").json() == {"id": "me"}


def test_from_environment_keeps_any_name_an_api_route_takes() -> None:
    @environment_card(name="shop")
    @website(domains=["shop.test"])
    class Shop(WebsiteEnvironment):
        @api.get("/api/items", name="items:list")
        def items(self) -> list:
            return []

    assert list(from_environment(Shop).route_table()) == ["items:list"]
    card = card_of(TestClient(Shop().create_app()))
    assert [r["name"] for r in website_params(card)["routes"]] == ["items:list"]
    assert list(from_card(card).route_table()) == ["items:list"]
    for derived in (from_environment(Shop), from_card(card)):
        assert derived.overlay({"site": {"domains": ["shop.example"]}}).site.domains == ("shop.example",)
    assert "routes.ops: route names ['items:list'] aren't names" in problems(
        {"routes": {"ops": {"items:list": "GET /api/items"}}}, Path.cwd())[0]


def test_a_site_whose_domains_are_in_upper_case_starts_and_its_spec_names_them_in_lower_case() -> None:
    @environment_card(name="notes")
    @website(domains=["Notes.TEST", "*.Notes.Test"])
    class Notes(WebsiteEnvironment):
        pass

    card = card_of(TestClient(Notes().create_app()))
    assert from_environment(Notes).site.domains == ("notes.test", "*.notes.test")
    assert from_card(card).site.domains == ("notes.test", "*.notes.test")


def test_from_environment_keeps_a_head_or_options_route_an_api_route_takes() -> None:
    @environment_card(name="shop")
    @website(domains=["shop.test"])
    class Shop(WebsiteEnvironment):
        @api.route("HEAD", "/ping")
        def ping(self) -> dict:
            return {}

        @api.route("OPTIONS", "/items")
        def items_options(self) -> dict:
            return {}

    assert {name: (r.method, r.path) for name, r in from_environment(Shop).route_table().items()} == {
        "ping": ("HEAD", "/ping"), "items_options": ("OPTIONS", "/items")}
    routes = website_params(card_of(TestClient(Shop().create_app())))["routes"]
    assert [(r["method"], r["name"]) for r in routes] == [("HEAD", "ping"), ("OPTIONS", "items_options")]


def test_a_class_declared_from_a_spec_must_serve_what_it_declares(tmp_path: Path) -> None:
    path = write(tmp_path, {"site.toml": '[site]\nname = "wiki"\ndomains = ["wiki.test"]\n[source]\nsdk = "m:Wiki"\n'
                                         '[routes.ops]\npages = "GET /api/pages"\nedit = "PUT /api/pages/{id}"\n'
                                         '[roles]\nadmin = ["edit"]\n'})

    class Wiki(WebsiteEnvironment):
        @api.get("/api/pages")
        def pages(self) -> list:
            return []

        @api.put("/api/pages/{id}", name="edit")
        def edit_page(self, id: str) -> dict:
            return {}

        @api.get("/api/extra")
        def extra(self) -> dict:
            return {}

    with pytest.raises(SpecError) as caught:
        website(spec=path)(Wiki)
    assert [p.split(": ", 1)[1] for p in caught.value.problems] == [
        "routes.ops.edit: the spec says PUT /api/pages/{id} for ['admin'], the class serves PUT /api/pages/{id}",
        "routes.ops: Wiki serves extra (GET /api/extra), which the spec doesn't declare"]
    with pytest.raises(SpecError, match="source.sdk: m:Wiki names another class than Other"):
        website(spec=load(path).overlay({"routes": {"ops": {"edit": False}}, "roles": {"admin": False}}))(
            type("Other", (WebsiteEnvironment,), {}))
    with pytest.raises(TypeError, match="takes the site from its spec"):
        website(spec=path, domains=["x.test"])


def test_a_class_declared_from_a_spec_is_named_and_seeded_by_it(tmp_path: Path) -> None:
    write(tmp_path, {"a.json": '{"current_user": "ann", "pages": [{"id": 1}]}', "b.json": '{"pages": [{"id": 2}]}',
                     "dist/index.html": "<html><head></head></html>"})
    path = write(tmp_path, {"site.toml": '[site]\nname = "wiki"\ndomains = ["wiki.test"]\nhealth = "/up"\n'
                                         '[source]\nsdk = "m:Wiki"\nassets = "dist"\n[data]\nseed = ["a.json", "b.json"]\n'
                                         '[clock]\nfollow = "date_patch"\n'})

    @website(spec=path)
    class Wiki(WebsiteEnvironment):
        pass

    env = Wiki()
    client = TestClient(env.create_app())
    assert card_of(client)["name"] == "wiki" and client.get("/up").status_code == 200
    assert env.store["pages"] == [{"id": 1}, {"id": 2}] and env.store.current_user == "ann"
    assert Wiki.declaration().clock_patch and "<script>" in client.get("/").text
    assert website_params(card_of(client))["clock"] == {"follow": "date_patch"}

    @environment_card(name="other")
    @website(spec=path)
    class Wiki(WebsiteEnvironment):
        pass

    with pytest.raises(SpecError, match="@environment_card names it 'other', its spec 'wiki'"):
        Wiki().create_app()


def test_a_specs_acting_user_must_be_who_its_seeds_sign_in(tmp_path: Path) -> None:
    write(tmp_path, {"a.json": '{"current_user": "ann", "pages": []}', "b.json": '{"current_user": "ben"}',
                     "c.json": '{"pages": []}'})
    path = write(tmp_path, {"site.toml": '[site]\nname = "wiki"\n[source]\nsdk = "m:Wiki"\n'
                                         '[data]\nseed = ["a.json", "b.json", "c.json"]\n[users]\nacting_user = "ann"\n'})

    class Wiki(WebsiteEnvironment):
        pass

    with pytest.raises(SpecError, match="users.acting_user: the spec says ann, the seed's current_user ben"):
        website(spec=path)(Wiki)
    assert website(spec=load(path).overlay({"users": {"acting_user": "ben"}}))(Wiki).site_spec().users.acting_user == \
        "ben"


# Conformance against a spec


def answering(client: TestClient, monkeypatch, change) -> None:
    """Make ``client``'s answers ``change(path, response)``'s."""
    original = client.request

    def request(method, url, **kwargs):
        response = original(method, url, **kwargs)
        return change(httpx.URL(str(url)).path, response)

    monkeypatch.setattr(client, "request", request)


def test_the_notes_site_passes_the_conformance_check_against_its_spec(notes_client: TestClient) -> None:
    results = conformance.check("http://testserver", client=notes_client, spec=load(NOTES_DIR / "website.toml"))
    assert [r.name for r in results][-2:] == ["spec", "spec_routes"]
    assert all(r.status == "passed" for r in results), results


def test_the_conformance_check_fails_a_server_that_does_not_serve_its_spec(notes_client: TestClient,
                                                                         tmp_path: Path) -> None:
    spec = load(NOTES_DIR / "website.toml").overlay({
        "site": {"domains": ["notes.example"], "health": "/api/health"},
        "routes": {"ops": {"archive": "POST /api/archive", "all_notes": {"path": "/api/notes/all"}}}})
    results = {r.name: r for r in conformance.check("http://testserver", client=notes_client, spec=spec)}
    assert results["health"].status == "passed"  # readiness is the card's website/v1 ready, not the spec's path
    assert results["spec"].status == "failed"
    assert "the card's health is '/healthz', the spec's '/api/health'" in results["spec"].detail
    assert "the card's domains are ['notes.test'], the spec's ['notes.example']" in results["spec"].detail
    assert "the card lacks route archive" in results["spec"].detail
    assert "the card's all_notes is GET /api/admin/notes for ['admin'], the spec's GET /api/notes/all" in \
        results["spec"].detail
    assert results["spec_routes"].status == "failed"
    assert results["spec_routes"].detail == ("all_notes: nothing is at GET /api/notes/all; "
                                             "archive: nothing is at POST /api/archive")


def test_the_spec_check_fails_a_cookie_sign_in_when_the_spec_signs_nobody_in() -> None:
    @environment_card(name="shop")
    @website(domains=["shop.test"])
    class Shop(WebsiteEnvironment):
        pass

    spec = from_environment(Shop).overlay({"users": {"sign_in": "none"}})
    results = {r.name: r for r in conformance.check("http://testserver", client=TestClient(Shop().create_app()),
                                                    spec=spec)}
    assert results["spec"].status == "failed"
    assert "the card's session_cookie is 'agentenv_session_shop', a cookie sign-in, and the spec's sign-in is none" \
        in results["spec"].detail
    assert "the card's session_path is" in results["spec"].detail


def test_the_spec_check_fails_a_card_with_another_session_path(
        notes_client: TestClient, monkeypatch) -> None:
    def change(path: str, response: httpx.Response) -> httpx.Response:
        if path != "/.well-known/agent-env.json":
            return response
        card = response.json()
        params = next(e for e in card["capabilities"]["extensions"]
                      if e["uri"] == website_spec.WEBSITE_EXTENSION_URI)["params"]
        params["session_path"] = "/api/login"
        return httpx.Response(200, json=card, request=response.request)

    answering(notes_client, monkeypatch, change)
    results = {r.name: r for r in conformance.check("http://testserver", client=notes_client,
                                                    spec=load(NOTES_DIR / "website.toml"))}
    assert results["spec"].status == "failed"
    assert results["spec"].detail == "the card's session_path is '/api/login', the spec's '/api/session'"


def test_spec_routes_fails_a_server_whose_advertised_openapi_document_cannot_be_read(notes_client: TestClient,
                                                                                    monkeypatch) -> None:
    endpoint = conformance.find_extension_method(card_of(notes_client), website_spec.WEBSITE_EXTENSION_URI,
                                                 "openapi")["endpoint"]
    answering(notes_client, monkeypatch, lambda path, response: httpx.Response(500, request=response.request)
              if path == endpoint else response)
    results = {r.name: r for r in conformance.check("http://testserver", client=notes_client,
                                                    spec=load(NOTES_DIR / "website.toml"))}
    assert results["spec_routes"].status == "failed"
    assert results["spec_routes"].detail == f"the card's openapi, GET {endpoint}, answered 500 and no OpenAPI document"


def test_spec_routes_probes_a_path_parameter_with_a_value_its_converter_takes(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><html><head></head><body></body></html>")

    @environment_card(name="shop")
    @website(domains=["shop.test"], static=str(tmp_path))
    class Shop(WebsiteEnvironment):
        @api.get("/api/items/{id:uuid}")
        def item(self, id: str) -> dict:
            return {}

        @api.get("/api/files/{rest:path}")
        def file(self, rest: str) -> dict:
            return {}

    spec = resolve({"site": {"name": "shop", "domains": ["shop.test"]},
                    "routes": {"ops": {"item": "GET /api/items/{id}", "file": "GET /api/files/{rest}"}}}, tmp_path)
    results = {r.name: r for r in conformance.check("http://testserver", client=TestClient(Shop().create_app()),
                                                    spec=spec)}
    assert results["spec_routes"].status == "passed", results["spec_routes"]


def _starlette_site(options_status: int | None = None) -> TestClient:
    """A server with no OpenAPI document serving ``POST /api/items`` and ``GET /api/items/{id}``; with
    ``options_status``, a catch-all answering every OPTIONS that status without an ``Allow``."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def handler(request):
        return JSONResponse({})

    app = Starlette(routes=[Route("/api/items", handler, methods=["POST"]),
                            Route("/api/items/{id}", handler, methods=["GET"])])

    async def catch_all(scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "OPTIONS" and options_status is not None:
            return await Response(status_code=options_status)(scope, receive, send)
        return await app(scope, receive, send)

    return TestClient(catch_all)


def test_spec_routes_without_an_openapi_document_reads_the_allow_of_the_options(tmp_path: Path) -> None:
    def spec_routes(client: TestClient, ops: dict) -> conformance.Result:
        spec = resolve({"site": {"name": "shop", "domains": ["shop.test"]}, "routes": {"ops": ops}}, tmp_path)
        return {r.name: r for r in conformance.check("http://testserver", client=client, spec=spec)}["spec_routes"]

    served = {"add_item": "POST /api/items", "item": "GET /api/items/{id}"}
    assert spec_routes(_starlette_site(), served).status == "passed"
    missing = spec_routes(_starlette_site(), {**served, "archive": "POST /api/archive"})
    assert (missing.status, missing.detail) == ("failed", "archive: nothing is at POST /api/archive")
    for site, ops, answered in (
            (_starlette_site(), {"drop_items": "DELETE /api/items"}, "405 allowing POST"),
            (_starlette_site(200), {"archive": "POST /api/archive"}, "200 without an Allow"),
            (_starlette_site(500), {"archive": "POST /api/archive"}, "500 without an Allow")):
        name, route = next(iter(ops.items()))
        path = route.split(" ")[1]
        unknown = spec_routes(site, ops)
        assert (unknown.status, unknown.detail) == (
            "warning", f"unverified: {name}: OPTIONS {path} answered {answered}, which doesn't show {route} is served")
        assert not unknown.fails() and unknown.fails(strict=True)


def test_conformance_main_takes_a_spec_and_refuses_one_that_does_not_load(monkeypatch, capsys,
                                                                       notes_client: TestClient,
                                                                       tmp_path: Path) -> None:
    seen = {}

    def fake_check(url, **kwargs):
        seen.update(kwargs)
        return [conformance.Result("spec", "passed")]

    monkeypatch.setattr(conformance, "check", fake_check)
    assert conformance.main(["http://x", "--spec", str(NOTES_DIR / "website.toml")]) == 0
    assert seen["spec"].site.name == "notes"
    assert conformance.main(["http://x", "--spec", str(write(tmp_path, {"bad.toml": "[pages]\n"}))]) == 2
    assert "pages: isn't a key the spec takes" in capsys.readouterr().err


# Examples and the command line


def test_the_composed_examples_load() -> None:
    slack = load(SPECS_DIR / "slack.toml")
    assert (slack.source.kind, slack.users.sign_in, slack.clock.follow) == ("replay", "live", "date_patch")
    assert slack.route_table()["post_message"].served_by == "slack_synthetic:SlackChat"
    assert slack.route_table()["post_message"].path == "/api/chat.postMessage"
    assert load(SPECS_DIR / "slack-live.toml").source.live.credentials == {"token": "secret:SLACK_USER_TOKEN"}
    intranet = load(SPECS_DIR / "intranet.toml")
    assert [n for n in intranet.route_table() if n.startswith("sso.")] == ["sso.authorize", "sso.login", "sso.token",
                                                                          "sso.userinfo"]
    assert list(intranet.data.collections) == ["users", "pages", "sso.grants"]
    ben = load(SPECS_DIR / "intranet-ben.toml")
    assert (ben.users.acting_user, [p.id for p in ben.users.personas], ben.roles) == ("ben", ["ben", "cy"], {})


def test_the_command_line_validates_shows_and_describes_a_spec(capsys, tmp_path: Path) -> None:
    assert website_spec.main(["validate", str(SPECS_DIR / "intranet.toml")]) == 0
    assert "ok: intranet, image, 6 route(s), 3 collection(s), 1 component(s)" in capsys.readouterr().out
    assert website_spec.main(["show", str(NOTES_DIR / "website.toml")]) == 0
    assert json.loads(capsys.readouterr().out)["site"]["name"] == "notes"
    assert website_spec.main(["show", "--card", str(NOTES_DIR / "website.toml")]) == 0
    assert json.loads(capsys.readouterr().out)["session_cookie"] == "agentenv_session_notes"
    assert website_spec.main(["validate", str(write(tmp_path, {"bad.toml": '[source]\nsdk = "x"\n'}))]) == 1
    assert "source: sdk is module:Class" in capsys.readouterr().err
    assert website_spec.main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert set(schema["properties"]) == {"extends", "site", "source", "data", "users", "roles", "routes", "clock",
                                         "uses"}
    assert schema["$defs"]["Routes"]["properties"]["ops"]["additionalProperties"]["anyOf"][1]["type"] == "string"
    assert schema["$defs"]["Use"]["properties"]["as"]


def test_the_json_schema_takes_the_false_that_removes_an_inherited_entry() -> None:
    schema = website_spec.json_schema()
    removed = {"const": False, "description": "removes the inherited entry"}
    assert removed in schema["$defs"]["Routes"]["properties"]["ops"]["additionalProperties"]["anyOf"]
    assert removed in schema["properties"]["roles"]["additionalProperties"]["anyOf"]
    assert removed in schema["$defs"]["Data"]["properties"]["collections"]["additionalProperties"]["anyOf"]
    jsonschema = pytest.importorskip("jsonschema")
    overlay = {"extends": "base.toml", "roles": {"staff": False, "admin": ["refund"]},
               "routes": {"ops": {"list_orders": False, "place_order": "POST /api/orders"}},
               "data": {"collections": {"orders": False, "carts": {"key": "cart_id"}}}}
    jsonschema.validate(overlay, schema)
    jsonschema.validate(website_spec._read(SPECS_DIR / "slack.toml"), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"roles": {"staff": True}}, schema)


@pytest.mark.parametrize("route", ["get /items", "GET /items", "Patch /items/{id}", "head /ping", "OPTIONS /items"])
def test_the_json_schema_takes_a_route_shorthand_in_the_case_the_loader_takes(route: str, tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate({"routes": {"ops": {"a": route}}}, website_spec.json_schema())
    assert list(resolve({"routes": {"ops": {"a": route}}}, tmp_path).route_table()) == ["a"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"routes": {"ops": {"a": "FETCH /items"}}}, website_spec.json_schema())
