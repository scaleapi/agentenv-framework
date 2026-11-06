"""agent-env env website dev's pieces, each without a long-running process: the site class it serves, the boot every
answer of a server carries, the local clock standing in for the gateway's, when web/ is rebuilt, priming a server
(clock/v1's sync_time and the seed through the data plane) once per boot with one conformance check, and the command
wiring uvicorn's reload, the clock and the supervisor together."""

import datetime
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient

from agent_env.cli.env import website_dev
from agent_env.cli.env.website import website
from agent_env.cli.env.website_scaffold import scaffold
from agentenv_protocol.types import WELL_KNOWN_PATH
from agentenv_protocol.website import UNIVERSE_NOW_HEADER, conformance, website as declare
from agentenv_protocol.website.clock import parse_time

TWO_SITES = '''
from agentenv_protocol.website import WebsiteEnvironment, website


class Base(WebsiteEnvironment):
    pass


@website(domains=["a.test"])
class Site(Base):
    pass
'''


@pytest.fixture
def site_dir(tmp_path: Path) -> Path:
    scaffold(tmp_path / "shop", "shop")
    page = tmp_path / "shop" / "web" / "dist"
    page.mkdir()
    (page / "index.html").write_text("<!doctype html><html><head><title>shop</title></head><body></body></html>")
    return tmp_path / "shop"


@pytest.fixture
def clock(socket_enabled):
    started = website_dev.DevClock(datetime.datetime(2030, 1, 1, 9, tzinfo=datetime.timezone.utc)).start()
    yield started
    started.close()


def _app(site_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(website_dev.SITE_ENV, str(site_dir / "site.py"))
    return website_dev.create_app()


def test_the_site_class_is_the_one_its_file_defines_or_the_one_declared_with_website(tmp_path, site_dir):
    assert website_dev.site_class(site_dir / "site.py").__name__ == "ShopSite"
    (tmp_path / "two.py").write_text(TWO_SITES)
    assert website_dev.site_class(tmp_path / "two.py").__name__ == "Site"
    (tmp_path / "none.py").write_text("x = 1\n")
    with pytest.raises(website_dev.DevError, match="defines 0 WebsiteEnvironment classes"):
        website_dev.site_class(tmp_path / "none.py")


def test_a_site_imports_the_files_beside_it(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "shop_models_beside.py").write_text("DOMAIN = 'shop.test'\n")
    (tmp_path / "shop" / "site.py").write_text(
        "from agentenv_protocol.website import WebsiteEnvironment, website\n"
        "from shop_models_beside import DOMAIN\n\n\n@website(domains=[DOMAIN])\nclass Shop(WebsiteEnvironment):\n    pass\n")
    monkeypatch.chdir(tmp_path)
    assert website_dev.site_class(Path("shop") / "site.py").__name__ == "Shop"


def test_every_answer_carries_the_servers_boot_and_the_supervisors_polls_leave_the_access_log(site_dir, monkeypatch):
    first, second = TestClient(_app(site_dir, monkeypatch)), TestClient(_app(site_dir, monkeypatch))
    boot = first.get("/healthz").headers[website_dev.BOOT_HEADER]
    assert first.get("/api/items").headers[website_dev.BOOT_HEADER] == boot
    assert second.get("/healthz").headers[website_dev.BOOT_HEADER] != boot

    def line(path):
        return logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                                 ("127.0.0.1:1", "GET", path, "1.1", 200), None)
    assert not website_dev._unpolled(line(f"/.well-known/agent-env.json?{website_dev.POLL_QUERY}"))
    assert website_dev._unpolled(line("/api/items"))


def test_the_local_clock_answers_as_the_gateways_and_is_set_and_advanced(clock):
    base = clock.url.rsplit("/clock/", 1)[0]
    assert parse_time(httpx.get(clock.url).json()["virtual_time"]) - parse_time("2030-01-01T09:00:00Z") < datetime.timedelta(seconds=5)
    state = httpx.put(base + "/clock/set-time", json={"virtual_time": "2031-02-03T04:05:06Z",
                                                      "virtual_seconds_per_real_second": 0}).json()
    assert (state["virtual_time"], state["virtual_seconds_per_real_second"]) == ("2031-02-03T04:05:06.000Z", 0.0)
    assert httpx.put(base + "/clock/advance", json={"seconds": 3600}).json()["virtual_time"] == "2031-02-03T05:05:06.000Z"
    assert httpx.put(base + "/clock/advance", json={"virtual_time": "2031-02-04T00:00:00Z"}).status_code == 200
    assert httpx.get(clock.url).json() == {"virtual_time": "2031-02-04T00:00:00.000Z",
                                           "virtual_seconds_per_real_second": 0.0}
    assert httpx.get(base + "/clock/state").json()["env_get_time_url"] == clock.url
    backwards = httpx.put(base + "/clock/advance", json={"virtual_time": "2031-01-01T00:00:00Z"})
    assert backwards.status_code == 400 and "forward" in backwards.json()["error"]
    assert httpx.put(base + "/clock/advance", json={}).status_code == 400
    assert httpx.put(base + "/clock/set-time", content=b"[1]").status_code == 400
    assert httpx.get(base + "/clock/other").status_code == 404


def test_web_is_installed_when_it_has_no_node_modules_and_built_when_dist_is_older_than_a_source(tmp_path):
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    (web / "package.json").write_text("{}")
    (web / "src" / "main.js").write_text("")
    assert website_dev.web_plan(web, watch=False) == (
        [["npm", "install", "--no-audit", "--no-fund"], ["npm", "run", "build"]], None)
    (web / "package-lock.json").write_text("{}")
    (web / "node_modules" / "vite").mkdir(parents=True)
    (web / "dist").mkdir()
    (web / "dist" / "index.html").write_text("")
    for path in [*web.rglob("*")]:
        os.utime(path, (1000, 1000))
    os.utime(web / "dist" / "index.html", (2000, 2000))
    (web / "node_modules" / "vite" / "new.js").write_text("")
    assert not website_dev.web_stale(web)
    assert website_dev.web_plan(web, watch=True) == ([], ["npm", "run", "build", "--", "--watch"])
    os.utime(web / "src" / "main.js", (3000, 3000))
    assert website_dev.web_stale(web)
    assert website_dev.web_plan(web, watch=False)[0] == [["npm", "run", "build"]]
    (web / "node_modules").rename(web / "gone")
    assert website_dev.web_plan(web, watch=False)[0][0] == ["npm", "ci", "--no-audit", "--no-fund"]


def test_priming_syncs_the_site_to_the_clock_and_loads_the_seed_in_place_of_its_state(site_dir, clock, monkeypatch):
    client = TestClient(_app(site_dir, monkeypatch))
    client.post("/api/items", json={"title": "made before priming"})
    seed = site_dir / "data" / "seed.json"
    boot = website_dev.prime(client, "http://testserver", clock.url, seed)
    assert boot == client.get("/healthz").headers[website_dev.BOOT_HEADER]
    assert client.get("/healthz").headers[UNIVERSE_NOW_HEADER].startswith("2030-01-01T09:0")
    assert [i["title"] for i in client.get("/api/items").json()] == ["Try the site"]
    assert client.post("/api/items", json={"title": "now"}).json()["created_at"].startswith("2030-01-01T09:0")


def _card_site(card: dict, sent: list) -> httpx.Client:
    """A client of a site that serves ``card`` and records each request it is sent, answering every other one."""
    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path == WELL_KNOWN_PATH:
            return httpx.Response(200, json=card)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    return httpx.Client(transport=httpx.MockTransport(answer))


def test_priming_loads_the_seed_at_the_data_plane_the_card_names_and_refuses_one_it_doesnt_serve(site_dir):
    seed = site_dir / "data" / "seed.json"
    clock_ext = {"uri": website_dev.CLOCK_EXTENSION_URI,
                 "params": {"methods": {"sync_time": {"endpoint": "/agentenv/ext/clock/sync"}}}}
    card = {"name": "shop", "url": "/svc/shop/rpc", "capabilities": {"extensions": [clock_ext],
                                                                     "operations": ["data/reset", "data/add"]}}
    sent: list[httpx.Request] = []
    website_dev.prime(_card_site(card, sent), "http://site", "http://clock", seed)
    rpcs = [(str(r.url), json.loads(r.content)["method"]) for r in sent
            if r.url.path not in (WELL_KNOWN_PATH, "/agentenv/ext/clock/sync")]
    assert rpcs == [("http://site/svc/shop/rpc", "data/reset"), ("http://site/svc/shop/rpc", "data/add")]

    sent.clear()
    website_dev.prime(_card_site({**card, "url": "http://elsewhere/rpc"}, sent), "http://site", "http://clock", seed)
    assert {str(r.url) for r in sent if r.method == "POST" and "clock" not in r.url.path} == {"http://elsewhere/rpc"}

    sent.clear()
    unserving = {**card, "capabilities": {**card["capabilities"], "operations": ["data/get"]}}
    with pytest.raises(website_dev.DevError, match="doesn't serve data/reset, data/add"):
        website_dev.prime(_card_site(unserving, sent), "http://site", "http://clock", seed)
    assert not any("rpc" in r.url.path for r in sent)


def test_the_supervisor_primes_each_boot_once_and_runs_the_conformance_check_after_the_first(site_dir, clock,
                                                                                             monkeypatch):
    first = TestClient(_app(site_dir, monkeypatch))
    said = []
    supervisor = website_dev.Supervisor("http://testserver", clock.url, site_dir / "data" / "seed.json",
                                        echo=said.append, client=first)
    with patch.object(website_dev, "prime", wraps=website_dev.prime) as primed:
        assert supervisor.step()
        assert "[dev] conformance: 10 passed or skipped, 0 failed" in said and primed.call_count == 2
        assert first.get("/healthz").headers[UNIVERSE_NOW_HEADER].startswith("2030-01-01T09:0")
        assert supervisor.step() and primed.call_count == 2

        reloaded = TestClient(_app(site_dir, monkeypatch))
        supervisor.client = reloaded
        assert supervisor.step() and primed.call_count == 3
    assert sum("conformance:" in line for line in said) == 1
    assert reloaded.get("/healthz").headers[UNIVERSE_NOW_HEADER].startswith("2030-01-01T09:0")


def test_the_supervisor_shows_a_warning_and_syncs_the_site_again_even_when_the_check_raises(site_dir, clock,
                                                                                          monkeypatch):
    client = TestClient(_app(site_dir, monkeypatch))
    said = []
    supervisor = website_dev.Supervisor("http://testserver", clock.url, None, echo=said.append, client=client)
    warned = conformance.Result("clock", "warning", "interrupted by a reload", False)
    with patch.object(conformance, "check", return_value=[warned]), \
            patch.object(website_dev, "prime", wraps=website_dev.prime) as primed:
        assert supervisor.step()
    assert "[dev] WARN clock: interrupted by a reload" in said and primed.call_count == 2

    supervisor.checked = False
    with patch.object(conformance, "check", side_effect=RuntimeError("boom")), \
            patch.object(website_dev, "prime", wraps=website_dev.prime) as primed, pytest.raises(RuntimeError):
        supervisor._conformance()
    assert primed.call_count == 1

    with patch.object(conformance, "check", side_effect=httpx.RemoteProtocolError("reloaded")), \
            patch.object(website_dev, "prime", wraps=website_dev.prime) as primed:
        supervisor._conformance()
    assert "[dev] the conformance check lost the site, likely to a reload: reloaded" in said and primed.call_count == 1


def test_the_supervisor_primes_again_on_the_next_poll_when_priming_after_the_check_fails(site_dir, clock,
                                                                                        monkeypatch):
    client = TestClient(_app(site_dir, monkeypatch))
    said = []
    supervisor = website_dev.Supervisor("http://testserver", clock.url, None, echo=said.append, client=client)
    real_prime = website_dev.prime
    with patch.object(conformance, "check", return_value=[]), \
            patch.object(website_dev, "prime", side_effect=[real_prime(client, "http://testserver", clock.url, None),
                                                            httpx.ConnectError("refused")]):
        assert supervisor.step()
    assert "[dev] couldn't sync the site again after the check: refused" in said and supervisor.primed is None
    with patch.object(website_dev, "prime", wraps=website_dev.prime) as primed:
        assert supervisor.step() and primed.call_count == 1
    assert supervisor.primed == client.get("/healthz").headers[website_dev.BOOT_HEADER]


def test_the_supervisor_reports_a_seed_it_cant_read_and_primes_once_the_seed_is_back(site_dir, clock, monkeypatch):
    client = TestClient(_app(site_dir, monkeypatch))
    said = []
    seed = site_dir / "data" / "seed.json"
    kept = seed.read_text(encoding="utf-8")
    seed.unlink()
    supervisor = website_dev.Supervisor("http://testserver", clock.url, seed, echo=said.append, client=client)
    with patch.object(conformance, "check", return_value=[]):
        assert supervisor.step()
        assert "couldn't sync the site and load its seed" in said[-1] and supervisor.primed is None

        seed.write_text(kept, encoding="utf-8")
        real_prime = website_dev.prime
        with patch.object(website_dev, "prime", side_effect=[real_prime(client, "http://testserver", clock.url, seed),
                                                             FileNotFoundError(str(seed))]):
            assert supervisor.step()
        assert "couldn't sync the site again after the check" in said[-1] and supervisor.primed is None

        assert supervisor.step()
    assert supervisor.primed == client.get("/healthz").headers[website_dev.BOOT_HEADER]
    assert [i["title"] for i in client.get("/api/items").json()] == ["Try the site"]


def test_the_supervisor_waits_for_a_server_that_isnt_up_and_reports_one_it_cant_prime():
    down = MagicMock(spec=httpx.Client)
    down.get.side_effect = httpx.ConnectError("refused")
    assert not website_dev.Supervisor("http://127.0.0.1:1", "http://clock", None, client=down).step()

    app = declare(domains=["x.test"])(type("Bare", (website_dev.WebsiteEnvironment,), {}))().create_app()
    said = []
    supervisor = website_dev.Supervisor("http://testserver", "http://clock", None, client=TestClient(app),
                                        echo=said.append, check=False)
    with patch.object(website_dev, "find_extension_method", return_value=None):
        assert supervisor.step()
    assert "couldn't sync the site" in said[0]


def test_dev_runs_uvicorn_with_reload_beside_the_clock_and_the_supervisor(site_dir, monkeypatch):
    monkeypatch.setenv(website_dev.SITE_ENV, "unset")
    clock, supervisor = MagicMock(url="http://127.0.0.1:5555/clock/time"), MagicMock()
    supervisor.start.return_value = supervisor
    clock.start.return_value = clock
    with patch.object(website_dev, "DevClock", return_value=clock) as made_clock, \
            patch.object(website_dev, "Supervisor", return_value=supervisor) as made_supervisor, \
            patch.object(website_dev, "npm_available", return_value=True), \
            patch.object(website_dev, "run_npm") as npm, \
            patch("uvicorn.run") as run:
        result = CliRunner().invoke(website, ["dev", "--dir", str(site_dir), "--port", "8123", "--no-watch",
                                              "--clock-time", "2030-01-01T09:00:00Z"])
        assert os.environ[website_dev.SITE_ENV] == str(site_dir.resolve() / "site.py")
    assert result.exit_code == 0, result.output
    assert [c.args[0] for c in npm.call_args_list] == [["npm", "install", "--no-audit", "--no-fund"]]
    assert made_clock.call_args.args[0] == parse_time("2030-01-01T09:00:00Z")
    assert made_supervisor.call_args.args == ("http://127.0.0.1:8123", clock.url, site_dir.resolve() / "data" / "seed.json")
    assert run.call_args.args == ("agent_env.cli.env.website_dev:create_app",)
    options = run.call_args.kwargs
    assert (options["factory"], options["reload"], options["port"]) == (True, True, 8123)
    assert options["reload_dirs"] == [str(site_dir.resolve())]
    assert options["reload_includes"] == ["*.py", "*.json", "*.toml"]
    assert options["reload_excludes"] == [str(site_dir.resolve() / "web")]
    assert "http://127.0.0.1:8123/" in result.output and "/clock/advance" in result.output
    supervisor.stop.assert_called_once()
    clock.close.assert_called_once()


def test_dev_refuses_a_folder_without_its_site_and_a_clock_time_that_isnt_one(tmp_path, site_dir):
    missing = CliRunner().invoke(website, ["dev", "--dir", str(tmp_path)])
    assert missing.exit_code == 1 and "no site.py" in missing.output
    bad = CliRunner().invoke(website, ["dev", "--dir", str(site_dir), "--clock-time", "soon"])
    assert bad.exit_code == 1 and "not an RFC3339 time" in bad.output
    (tmp_path / "site.py").write_text("raise RuntimeError('broken')\n")
    broken = CliRunner().invoke(website, ["dev", "--dir", str(tmp_path)])
    assert broken.exit_code == 1 and "broken" in broken.output


def test_dev_reads_the_folders_env_toml_as_a_bundle_would_and_refuses_one_that_isnt_a_websites(tmp_path, site_dir):
    assert website_dev.env_toml(tmp_path) is None
    assert website_dev.env_toml(site_dir) == {"type": "website", "environment_name": "shop",
                                              "backend_image": {"dockerfile": "Dockerfile"}}
    named = tmp_path / "named"
    scaffold(named, "shop", domains=["shop.example"])
    assert website_dev.env_toml(named)["domains"] == ["shop.example"]
    for text, said in (('type = "mcp_server"\n', "not a website"), ('type = "website"\ndomains = "shop.example"\n',
                       "domains is a list"), ('type = "website"\ndomains = ["not a host"]\n', ""), ("type = \n", "can't read")):
        (site_dir / "env.toml").write_text(text)
        refused = CliRunner().invoke(website, ["dev", "--dir", str(site_dir)])
        assert refused.exit_code == 1 and "Error:" in refused.output and said in refused.output, (text, refused.output)
