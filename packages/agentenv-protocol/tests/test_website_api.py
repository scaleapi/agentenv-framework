"""The website SDK's @api routes: binding from type hints, JSON answers, errors, roles, and what a class may not
declare."""

from __future__ import annotations

import datetime
import enum
from typing import Optional, Union

import pytest
from pydantic import BaseModel, computed_field
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from agentenv_protocol.website import (
    ROLE_KEY_ENV,
    ROLE_KEY_HEADER,
    ApiContext,
    BadRequest,
    Conflict,
    NotFound,
    WebsiteEnvironment,
    api,
    reserved_path,
    website,
)


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class NewItem(BaseModel):
    title: str
    price: float = 0


@website(domains=["shop.example"])
class Shop(WebsiteEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.items: dict[int, dict] = {1: {"id": 1, "title": "mug", "price": 3.5}}

    @api.get("/api/items/{item_id}", name="get_item")
    async def get_item(self, item_id: int) -> dict:
        """Read one item."""
        if item_id not in self.items:
            raise NotFound(f"no item {item_id}")
        return self.items[item_id]

    @api.get("/api/items")
    def list_items(self, limit: int = 10, cheap: bool = False, color: Optional[Color] = None,
                   since: Optional[datetime.date] = None, tags: list[str] = []) -> dict:
        return {"limit": limit, "cheap": cheap, "color": color.value if color else None,
                "since": since.isoformat() if since else None, "tags": tags}

    @api.post("/api/items", name="create_item", status=201)
    async def create_item(self, item: NewItem) -> NewItem:
        if any(i["title"] == item.title for i in self.items.values()):
            raise Conflict(f"{item.title} exists")
        self.items[len(self.items) + 1] = {"id": len(self.items) + 1, **item.model_dump()}
        return item

    @api.patch("/api/items/{item_id}")
    async def patch_item(self, item_id: int, changes: dict) -> dict:
        self.items[item_id].update(changes)
        return self.items[item_id]

    @api.delete("/api/items/{item_id}", roles=["admin"])
    async def delete_item(self, item_id: int) -> None:
        self.items.pop(item_id, None)

    @api.put("/api/teapot")
    async def teapot(self) -> None:
        raise BadRequest("short and stout", code="teapot", status=418)

    @api.get("/api/whoami")
    async def whoami(self, ctx: ApiContext) -> dict:
        return {"user": ctx.user, "role": ctx.role, "base_url": ctx.base_url, "q": ctx.query.get("q"),
                "header": ctx.headers.get("x-probe"), "cookie": ctx.cookies.get("c")}

    @api.get("/api/plain")
    async def plain(self) -> PlainTextResponse:
        return PlainTextResponse("as is")

    @api.get("/api/boom")
    async def boom(self) -> dict:
        raise RuntimeError("secret detail")

    @api.get("/api/picked")
    def picked(self, ids: Optional[list[int]] = None) -> dict:
        return {"ids": ids}

    @api.delete("/api/cart", status=204)
    async def empty_cart(self) -> None:
        return None


@pytest.fixture
def client() -> TestClient:
    return TestClient(Shop().create_app())


def test_path_query_and_body_parameters_are_bound_from_type_hints(client: TestClient) -> None:
    assert client.get("/api/items/1").json() == {"id": 1, "title": "mug", "price": 3.5}
    assert client.get("/api/items").json() == {"limit": 10, "cheap": False, "color": None, "since": None, "tags": []}
    listed = client.get("/api/items?limit=3&cheap=yes&color=blue&since=2030-01-02&tags=a&tags=b").json()
    assert listed == {"limit": 3, "cheap": True, "color": "blue", "since": "2030-01-02", "tags": ["a", "b"]}
    created = client.post("/api/items", json={"title": "pen", "price": 1})
    assert created.status_code == 201 and created.json() == {"title": "pen", "price": 1.0}
    assert client.patch("/api/items/1", json={"price": 4}).json()["price"] == 4


def test_a_request_its_parameters_cannot_be_bound_from_answers_400(client: TestClient) -> None:
    for response in (client.get("/api/items/abc"), client.get("/api/items?limit=many"),
                     client.get("/api/items?color=green"), client.post("/api/items", json={"price": 2}),
                     client.post("/api/items", content=b"not json"), client.post("/api/items"),
                     client.patch("/api/items/1", json=[1, 2])):
        assert response.status_code == 400, response.text
        assert response.json()["code"] == "invalid_params"
    assert "title" in client.post("/api/items", json={"price": 2}).json()["error"]


class Pin(BaseModel):
    lat: float


class Place(BaseModel):
    name: str
    pin: Pin


class Note(BaseModel):
    text: str


@website(domains=["map.example"])
class Map(WebsiteEnvironment):
    @api.post("/api/places")
    def add_place(self, place: Place) -> list[Place]:
        return [place]

    @api.post("/api/marks")
    def mark(self, mark: Union[Note, Pin]) -> dict:
        return {"kind": type(mark).__name__}

    @api.put("/api/settings")
    def settings(self, settings: Optional[dict]) -> dict:
        return {"settings": settings}


def test_a_body_is_validated_against_its_whole_annotation() -> None:
    client = TestClient(Map().create_app())
    assert client.put("/api/settings", content=b"null", headers={"content-type": "application/json"}).json() == {
        "settings": None}
    assert client.post("/api/marks", json={"lat": 1.5}).json() == {"kind": "Pin"}
    assert client.post("/api/marks", json={"text": "hi"}).json() == {"kind": "Note"}
    assert client.post("/api/marks", json={"other": 1}).status_code == 400


def test_nested_models_are_the_openapi_document_s_components_its_schemas_reference() -> None:
    document = TestClient(Map().create_app()).get("/agentenv/ext/website/openapi").json()
    schemas = document["components"]["schemas"]
    assert set(schemas) == {"Pin", "Place", "Note"}
    assert schemas["Place"]["properties"]["pin"] == {"$ref": "#/components/schemas/Pin"}
    body = document["paths"]["/api/places"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    answer = document["paths"]["/api/places"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert body == {"$ref": "#/components/schemas/Place"}
    assert answer == {"type": "array", "items": {"$ref": "#/components/schemas/Place"}}
    refs = [v for v in _walk(document) if isinstance(v, str) and v.startswith("#/")]
    assert refs and all(ref.split("/")[-1] in schemas and ref.startswith("#/components/schemas/") for ref in refs)


class Cart(BaseModel):
    prices: list[float]

    @computed_field
    @property
    def total(self) -> float:
        return sum(self.prices)


@website(domains=["till.example"])
class Till(WebsiteEnvironment):
    @api.post("/api/carts")
    def price(self, cart: Cart) -> Cart:
        return cart


def test_an_answer_s_schema_describes_it_as_serialized_and_a_body_s_as_validated() -> None:
    client = TestClient(Till().create_app())
    assert client.post("/api/carts", json={"prices": [1, 2]}).json() == {"prices": [1, 2], "total": 3}
    document = client.get("/agentenv/ext/website/openapi").json()
    schemas = document["components"]["schemas"]
    operation = document["paths"]["/api/carts"]["post"]
    body = operation["requestBody"]["content"]["application/json"]["schema"]
    answer = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert body == {"$ref": "#/components/schemas/Cart-Input"}
    assert answer == {"$ref": "#/components/schemas/Cart-Output"}
    assert "total" not in schemas["Cart-Input"]["properties"] and "total" in schemas["Cart-Output"]["properties"]


def _walk(value: object):
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)
    else:
        yield value


def test_api_errors_answer_with_their_status_and_code(client: TestClient) -> None:
    missing = client.get("/api/items/9")
    assert (missing.status_code, missing.json()) == (404, {"error": "no item 9", "code": "not_found"})
    assert client.post("/api/items", json={"title": "mug"}).json() == {"error": "mug exists", "code": "conflict"}
    teapot = client.put("/api/teapot")
    assert (teapot.status_code, teapot.json()["code"]) == (418, "teapot")
    boom = client.get("/api/boom")
    assert (boom.status_code, boom.json()) == (500, {"error": "internal error", "code": "internal"})


def test_responses_pass_through_and_none_is_json_null(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROLE_KEY_ENV, "k3y")
    assert client.get("/api/plain").text == "as is"
    deleted = client.delete("/api/items/1", headers={"AgentEnv-Role": "admin", ROLE_KEY_HEADER: "k3y"})
    assert (deleted.status_code, deleted.json()) == (200, None)


def test_an_optional_list_query_parameter_is_a_list_of_its_coerced_values(client: TestClient) -> None:
    assert client.get("/api/picked?ids=1&ids=2").json() == {"ids": [1, 2]}
    assert client.get("/api/picked").json() == {"ids": None}
    assert client.get("/api/picked?ids=x").status_code == 400
    [ids] = client.get("/agentenv/ext/website/openapi").json()["paths"]["/api/picked"]["get"]["parameters"]
    assert ids["schema"] == {"type": "array", "items": {"type": "integer"}, "default": None}


def test_a_route_declared_204_answers_with_no_body(client: TestClient) -> None:
    emptied = client.delete("/api/cart")
    assert (emptied.status_code, emptied.content) == (204, b"")
    assert "content-length" not in emptied.headers and "content-type" not in emptied.headers


def test_a_204_answer_keeps_a_real_server_connection_usable() -> None:
    import socket
    import threading
    import time

    uvicorn = pytest.importorskip("uvicorn")
    server = uvicorn.Server(uvicorn.Config(Shop().create_app(), host="127.0.0.1", port=0, log_level="error",
                                          ws="none"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            conn.sendall(b"DELETE /api/cart HTTP/1.1\r\nHost: x\r\n\r\nGET /api/items/1 HTTP/1.1\r\nHost: x\r\n\r\n")
            received = b""
            while b"mug" not in received:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                received += chunk
        assert received.startswith(b"HTTP/1.1 204") and b"HTTP/1.1 200" in received and b"mug" in received
    finally:
        server.should_exit = True
        thread.join(5)


def test_a_session_cookie_that_is_not_ascii_is_no_session(client: TestClient) -> None:
    for path in ("/api/whoami", "/api/session"):
        response = client.get(path, headers={"Cookie": "agentenv_session=abc.\xe9".encode("latin-1")})
        assert response.status_code == 200, response.text
        assert response.json()["user"] is None


def test_roles_refuse_other_callers_with_403(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROLE_KEY_ENV, "k3y")
    for headers in ({}, {"AgentEnv-Role": "default"}, {"AgentEnv-Role": "agent", ROLE_KEY_HEADER: "k3y"}):
        refused = client.delete("/api/items/1", headers=headers)
        assert refused.status_code == 403 and refused.json()["code"] == "forbidden"
    assert client.get("/api/items/1").status_code == 200
    assert client.delete("/api/items/1", headers={"AgentEnv-Role": "admin", ROLE_KEY_HEADER: "k3y"}).status_code == 200


def test_a_role_is_believed_only_with_the_site_s_role_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ROLE_KEY_ENV, raising=False)
    claim = {"AgentEnv-Role": "admin"}
    assert client.delete("/api/items/1", headers=claim).status_code == 403
    assert client.delete("/api/items/1", headers={**claim, ROLE_KEY_HEADER: ""}).status_code == 403
    monkeypatch.setenv(ROLE_KEY_ENV, "k3y")
    for key in (None, "", "wrong", "k3y "):
        headers = claim if key is None else {**claim, ROLE_KEY_HEADER: key}
        assert client.delete("/api/items/1", headers=headers).status_code == 403
        assert client.get("/api/whoami", headers=headers).json()["role"] == "default"
    assert client.get("/api/whoami", headers={**claim, ROLE_KEY_HEADER: "k3y"}).json()["role"] == "admin"


def test_the_context_carries_the_request_role_and_forwarded_address(client: TestClient,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROLE_KEY_ENV, "k3y")
    client.cookies.set("c", "crumb")
    seen = client.get("/api/whoami?q=1", headers={"x-probe": "p", "AgentEnv-Role": "agent", ROLE_KEY_HEADER: "k3y"}).json()
    assert seen == {"user": None, "role": "agent", "base_url": "http://testserver", "q": "1", "header": "p",
                    "cookie": "crumb"}
    forwarded = client.get("/api/whoami", headers={"X-Forwarded-Host": "shop.example, proxy",
                                                   "X-Forwarded-Proto": "https"}).json()
    assert (forwarded["base_url"], forwarded["role"]) == ("https://shop.example", "default")


def test_a_redirect_names_the_forwarded_address(client: TestClient) -> None:
    redirected = client.get("/api/items/", follow_redirects=False,
                            headers={"Host": "internal-x.invalid", "X-Forwarded-Host": "shop.example",
                                     "X-Forwarded-Proto": "https"})
    assert (redirected.status_code, redirected.headers["location"]) == (307, "https://shop.example/api/items")
    assert client.get("/api/items/", follow_redirects=False).headers["location"] == "http://testserver/api/items"


def test_a_site_behind_a_forwarded_prefix_names_it_in_its_address_and_redirects_and_routes_its_own_paths(
        client: TestClient) -> None:
    behind = {"Host": "internal-x.invalid", "X-Forwarded-Host": "gw.example", "X-Forwarded-Proto": "https",
              "X-Forwarded-Prefix": "/website/shop/"}
    assert client.get("/api/whoami", headers=behind).json()["base_url"] == "https://gw.example/website/shop"
    assert client.get("/api/items/1", headers=behind).json()["title"] == "mug"
    redirected = client.get("/api/items/", follow_redirects=False, headers=behind)
    assert (redirected.status_code, redirected.headers["location"]) == (307, "https://gw.example/website/shop/api/items")
    for unusable in ("//evil.example", "website", ""):
        seen = client.get("/api/whoami", headers={**behind, "X-Forwarded-Prefix": unusable}).json()
        assert seen["base_url"] == "https://gw.example", unusable


def test_parameterized_routes_never_take_the_data_plane_or_card() -> None:
    @website(domains=["wiki.example"])
    class Wiki(WebsiteEnvironment):
        @api.get("/{slug}")
        def page(self, slug: str) -> dict:
            return {"slug": slug}

        @api.post("/{collection}")
        def add(self, collection: str) -> dict:
            return {"captured": collection}

        @api.get("/.well-known/{name}")
        def well_known(self, name: str) -> dict:
            return {"captured": name}

        @api.get("/files/{rest:path}")
        def files(self, rest: str) -> dict:
            return {"rest": rest}

    client = TestClient(Wiki().create_app())
    assert client.get("/hello").json() == {"slug": "hello"}
    assert client.post("/notes").json() == {"captured": "notes"}
    assert client.get("/.well-known/other.json").json() == {"captured": "other.json"}
    assert client.get("/files/a/b.txt").json() == {"rest": "a/b.txt"}
    answer = client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get", "params": {}}).json()
    assert answer["jsonrpc"] == "2.0" and answer["id"] == 1 and "captured" not in answer
    assert client.get("/.well-known/agent-env.json").json()["name"] == "Wiki"


@pytest.mark.parametrize("path", ["/agentenv", "/agentenv/ext/x", "/.well-known/agent-env.json",
                                  "/api/../agentenv/x", "//agentenv"])
def test_api_paths_under_the_reserved_ones_are_refused(path: str) -> None:
    assert reserved_path(path)
    with pytest.raises(ValueError, match="data plane"):
        api.get(path)


def test_a_class_serving_a_name_or_route_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        class Twice(WebsiteEnvironment):
            @api.get("/a", name="same")
            def a(self) -> None: ...

            @api.get("/b", name="same")
            def b(self) -> None: ...

    with pytest.raises(ValueError, match="route"):
        class Again(WebsiteEnvironment):
            @api.post("/a")
            def a(self) -> None: ...

            @api.post("/a")
            def b(self) -> None: ...


def test_a_route_must_take_its_path_parameters_and_at_most_one_body() -> None:
    with pytest.raises(TypeError, match="item_id"):
        api.get("/api/{item_id}")(lambda self: None)

    class Bodies(WebsiteEnvironment):
        @api.post("/api/x")
        def x(self, a: dict, b: NewItem) -> None: ...

    with pytest.raises(TypeError, match="one JSON body"):
        Bodies().create_app()


@pytest.mark.parametrize(("annotation", "expected", "schema"), [
    (set[int], {1, 2}, {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}),
    (frozenset[int], frozenset({1, 2}), {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}),
    (tuple[int, ...], (1, 2, 2), {"type": "array", "items": {"type": "integer"}}),
])
def test_a_set_frozenset_or_tuple_query_parameter_binds_every_value_coerced(annotation, expected, schema) -> None:
    class Picks(WebsiteEnvironment):
        def picks(self, ids) -> dict:
            assert isinstance(ids, type(expected))
            return {"ids": sorted(ids)}

        picks.__annotations__["ids"] = annotation
        picks = api.get("/api/picks")(picks)

    client = TestClient(Picks().create_app())
    assert client.get("/api/picks?ids=2&ids=1&ids=2").json() == {"ids": sorted(expected)}
    spec = client.get("/agentenv/ext/website/openapi").json()
    assert spec["paths"]["/api/picks"]["get"]["parameters"][0]["schema"] == schema


def test_a_fixed_length_tuple_query_parameter_fails_the_apps_build() -> None:
    class Pair(WebsiteEnvironment):
        @api.get("/api/pair")
        def pair(self, point: tuple[int, int]) -> dict:
            return {}

    with pytest.raises(TypeError, match="fixed-length tuple"):
        Pair().create_app()


@pytest.mark.parametrize("domains", ["shop.example", ["https://shop.example"], ["shop.example:8080"], ["shop.example/x"],
                                     [""], ["-shop.example"], ["shop..example"], ["*shop.example"], [None]])
def test_a_sites_domains_are_a_list_of_host_names(domains: object) -> None:
    with pytest.raises(ValueError, match="domains"):
        website(domains=domains)


def test_a_wildcard_domain_names_a_sites_subdomains() -> None:
    @website(domains=["shop.example", "*.shop.example", "intranet.localhost"])
    class Shop(WebsiteEnvironment):
        pass

    assert Shop.declaration().domains == ("shop.example", "*.shop.example", "intranet.localhost")


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), "60", True])
def test_a_ready_timeout_is_a_number_of_seconds_above_zero(timeout: object) -> None:
    with pytest.raises(ValueError, match="ready_timeout_s"):
        website(ready_timeout_s=timeout)


@website(domains=["shelf.example"])
class Shelf(WebsiteEnvironment):
    @api.delete("/api/books/{book_id}", status=204)
    def drop(self, book_id: int) -> Pin:
        return Pin(lat=1.0)


def test_a_bodyless_status_publishes_no_body_and_answers_none() -> None:
    client = TestClient(Shelf().create_app())
    answered = client.delete("/api/books/1")
    assert answered.status_code == 204 and answered.content == b""
    responses = client.get("/agentenv/ext/website/openapi").json()["paths"]["/api/books/{book_id}"]["delete"]["responses"]
    assert "content" not in responses["204"]
