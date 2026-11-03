"""A site of ``@website(models=...)``: its state is a database's, SQLAlchemy models' tables on DATABASE_URL (sqlite
here) or an object's own load/export/reset, which its data plane and /go read and write, and its routes reach
through ``self.db``."""

from __future__ import annotations

import datetime
import sys
import types
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from agentenv_protocol.website import ApiContext, WebsiteEnvironment, api, conformance, website

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import DateTime, ForeignKey, Integer, String, select  # noqa: E402
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column  # noqa: E402

RPC = "/agentenv"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)


class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner: Mapped[str] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String)
    opened_at: Mapped[datetime.datetime] = mapped_column(DateTime)


SEED = {"current_user": "ann", "users": [{"id": "ann", "name": "Ann"}],
        "tickets": [{"id": 1, "owner": "ann", "title": "printer", "opened_at": "2031-01-02T03:04:05"}]}


def rpc(client: TestClient, method: str, params: dict | None = None) -> dict:
    body = client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).json()
    assert "error" not in body, body
    return body["result"]


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    (tmp_path / "index.html").write_text("<html><head></head><body>desk</body></html>")
    return tmp_path


@pytest.fixture
def database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    url = f"sqlite:///{tmp_path / 'service.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    return url


def helpdesk(dist: Path, models: object) -> type[WebsiteEnvironment]:
    @website(domains=["desk.example"], static=str(dist), models=models)
    class Helpdesk(WebsiteEnvironment):
        @api.get("/api/tickets")
        def tickets(self, ctx: ApiContext) -> list:
            with self.db() as session:
                return [{"id": t.id, "title": t.title} for t in session.scalars(
                    select(Ticket).where(Ticket.owner == ctx.user).order_by(Ticket.id))]

        @api.post("/api/tickets", status=201)
        def open_ticket(self, body: dict, ctx: ApiContext) -> dict:
            with self.db.begin() as session:
                ticket = Ticket(owner=ctx.user, title=body["title"], opened_at=self.now().replace(tzinfo=None))
                session.add(ticket)
                session.flush()
                return {"id": ticket.id}

    return Helpdesk


def test_the_data_plane_loads_exports_and_resets_the_models_tables(dist: Path, database: str) -> None:
    env = helpdesk(dist, Base)()
    client = TestClient(env.create_app())
    rpc(client, "data/add", {"parts": [{"kind": "data", "data": SEED}]})
    assert rpc(client, "data/get")["parts"][0]["data"] == SEED
    assert client.get("/api/session").json()["user"] == "ann"
    assert client.get("/api/tickets").json() == [{"id": 1, "title": "printer"}]
    rpc(client, "data/reset")
    assert rpc(client, "data/get")["parts"][0]["data"] == {"users": [], "tickets": []}


def test_rows_that_leave_an_optional_column_out_load_as_given_in_either_order(dist: Path, database: str) -> None:
    class Threads(DeclarativeBase):
        pass

    class Note(Threads):
        __tablename__ = "notes"
        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        parent_id: Mapped[int | None] = mapped_column(ForeignKey("notes.id"), nullable=True)

    client = TestClient(helpdesk(dist, Threads)().create_app())
    for seed in ([{"id": 1}, {"id": 2, "parent_id": 1}, {"id": 3, "parent_id": 1}, {"id": 4}],
                 [{"id": 1, "parent_id": None}, {"id": 2}, {"id": 3, "parent_id": 2}]):
        rpc(client, "data/reset")
        rpc(client, "data/add", {"parts": [{"kind": "data", "data": {"notes": seed}}]})
        assert rpc(client, "data/get")["parts"][0]["data"]["notes"] == [{"parent_id": None, **row} for row in seed]


def test_a_row_the_gui_adds_is_the_mcp_servers_and_go_reports_it(dist: Path, database: str) -> None:
    env = helpdesk(dist, Base)()
    client = TestClient(env.create_app())
    rpc(client, "data/add", {"parts": [{"kind": "data", "data": SEED}]})
    assert client.post("/api/tickets", json={"title": "badge"}).json() == {"id": 2}
    server = sqlalchemy.create_engine(database)
    with server.connect() as connection:
        assert connection.execute(select(Ticket.title).order_by(Ticket.id)).scalars().all() == ["printer", "badge"]
        connection.execute(sqlalchemy.update(Ticket).where(Ticket.id == 1).values(title="printer jam"))
        connection.commit()
    state = client.get("/go", params={"sid": "default"}).json()
    assert state["initial_state"] == SEED
    diff = state["state_diff"]["tickets"]
    assert [t["title"] for t in diff["added"]] == ["badge"] and diff["removed"] == []
    assert diff["changed"][0]["before"]["title"] == "printer" and diff["changed"][0]["after"]["title"] == "printer jam"
    assert client.get("/go", params={"sid": "other"}).json() == state


def test_a_models_site_passes_the_conformance_check(dist: Path, database: str) -> None:
    client = TestClient(helpdesk(dist, Base)().create_app())
    rpc(client, "data/add", {"parts": [{"kind": "data", "data": SEED}]})
    results = conformance.check("http://testserver", client=client)
    assert all(r.status == "passed" for r in results), results
    assert rpc(client, "data/get")["parts"][0]["data"] == SEED


def test_models_may_be_a_metadata_or_a_module_and_refuse_anything_else(dist: Path, database: str) -> None:
    module = types.ModuleType("helpdesk_models")
    module.Base, module.Ticket = Base, Ticket
    for models in (Base.metadata, module):
        assert helpdesk(dist, models)()._tables.metadata is Base.metadata
    with pytest.raises(TypeError, match="models="):
        helpdesk(dist, object())
    with pytest.raises(ValueError, match="data/add"):
        website(models=Base, data="seed.json")


def test_unknown_tables_and_a_missing_database_url_are_refused(dist: Path, monkeypatch: pytest.MonkeyPatch,
                                                               database: str) -> None:
    client = TestClient(helpdesk(dist, Base)().create_app())
    body = client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "data/add",
                                  "params": {"parts": [{"kind": "data", "data": {"orders": []}}]}}).json()
    assert "error" in body
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        helpdesk(dist, Base)().db()


def test_a_database_url_whose_driver_isnt_installed_names_it(dist: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    monkeypatch.setitem(sys.modules, "psycopg", None)
    monkeypatch.setenv("DATABASE_URL", "postgresql://desk@localhost/service")
    with pytest.raises(ImportError, match=r"postgresql:// needs the database driver psycopg, .* carries psycopg"):
        helpdesk(dist, Base)().db()


def test_an_object_with_its_own_hooks_is_the_sites_data(dist: Path) -> None:
    class Rows:
        def __init__(self) -> None:
            self.rows: list = []

        def load(self, data: dict) -> None:
            self.rows += data["rows"]

        async def export(self) -> dict:
            return {"rows": list(self.rows)}

        def reset(self) -> None:
            self.rows.clear()

    rows = Rows()
    env = helpdesk(dist, rows)()
    client = TestClient(env.create_app())
    rpc(client, "data/add", {"parts": [{"kind": "data", "data": {"current_user": "bo", "rows": [{"id": 1}]}}]})
    rows.rows.append({"id": 2})
    assert rpc(client, "data/get")["parts"][0]["data"] == {"current_user": "bo", "rows": [{"id": 1}, {"id": 2}]}
    assert client.get("/go").json()["state_diff"] == {"rows": {"added": [{"id": 2}], "removed": [], "changed": []}}
    assert env.acting_user() == "bo"
    assert env.db is None


def test_editing_a_row_a_hook_still_owns_leaves_the_initial_state(dist: Path) -> None:
    class Rows:
        def __init__(self) -> None:
            self.rows: list = []

        def load(self, data: dict) -> None:
            self.rows += data["rows"]

        def export(self) -> dict:
            return {"rows": list(self.rows)}

        def reset(self) -> None:
            self.rows.clear()

    rows = Rows()
    client = TestClient(helpdesk(dist, rows)().create_app())
    rpc(client, "data/add", {"parts": [{"kind": "data", "data": {"rows": [{"id": 1, "status": "open"}]}}]})
    rows.rows[0]["status"] = "closed"
    assert client.get("/go").json()["state_diff"] == {"rows": {"added": [], "removed": [], "changed": [
        {"id": 1, "before": {"id": 1, "status": "open"}, "after": {"id": 1, "status": "closed"}}]}}
