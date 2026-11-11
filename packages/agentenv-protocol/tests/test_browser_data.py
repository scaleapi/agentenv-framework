"""A website browser's data plane, a file tree: data/get returns its state as browser-state.json, its profile without
caches or locks, and its finished downloads; data/add applies such a state while the browser runs, the start tabs
loaded again after cookies change, a load's tabs in place of the open ones, and a profile restored with the browser
stopped, every part checked first; once an agent has acted, a load can't change the profile or the tabs, a failed one
isn't reset, and one that fails part way says what it applied; a load whose start tabs on one of the env's sites
don't load fails naming them, as does a reset's relaunch, while a live one never does; data/reset ends the addresses,
puts the profile back as installed, empties the downloads and uploads and starts the record over."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from types import SimpleNamespace

import pytest

from agentenv_protocol import website_browser
from agentenv_protocol.website_browser import log as browser_log
from agentenv_protocol.computer import Launch, Ready, app, linux
from agentenv_protocol.computer.app import AGENT_ACTED
from agentenv_protocol.computer.errors import DeviceUnavailable, InvalidArgument
from agentenv_protocol.computer.platform import Platform
from agentenv_protocol.types import DATA_OBJECTS_EXTENSION_URI, INTAKE_EXTENSION_URI, AddDataRequest
from agentenv_protocol.website_browser import WebsiteBrowserEnvironment

PROFILE = ".config/chromium"


@linux(paths=[PROFILE], launch=Launch(command=["chromium/chrome"], window={"class": "Chromium"},
                                      ready=Ready(window=True)))
class Browser(WebsiteBrowserEnvironment):
    version = "156.0.8078.4"


class FakeControl:
    """The browser's control, answering from what a test sets and recording what it's asked to do."""

    def __init__(self):
        self.did, self.fail, self.start_tabs_outcome = [], None, "loaded"
        self.cookie_list = [{"name": "sid", "value": "s1", "domain": "shop.test", "path": "/", "expires": -1,
                             "httpOnly": True, "secure": False, "sameSite": "Lax"}]

    async def cookies(self, urls=None):
        return self.cookie_list

    async def local_storage(self, origins=None):
        return [{"origin": "https://shop.test", "localStorage": [{"name": "cart", "value": "3"}]}]

    async def tab_list(self):
        return [{"id": "t1", "url": "https://shop.test/", "title": "Shop", "opened_at": "t0", "opener": None,
                 "active": True}]

    async def record(self):
        return {"closed_tabs": [{"id": "t0", "url": "about:blank", "title": "", "opened_at": "t0", "opener": None,
                                 "closed_at": "t1", "closed_by": "agentenv"}], "downloads": []}

    async def _done(self, what, value=None):
        if self.fail == what:
            raise RuntimeError(f"{what} failed")
        self.did.append((what, value))

    async def set_cookies(self, cookies):
        await self._done("cookies", [cookie.name for cookie in cookies])

    async def set_local_storage(self, origins):
        await self._done("local_storage", [origin.origin for origin in origins])

    async def replace_tabs(self, tabs):
        await self._done("tabs", [(tab.url, tab.active) for tab in tabs])

    async def load_start_tabs(self, urls, site_of, own_timeout_s, live_timeout_s):
        self.waits = (own_timeout_s, live_timeout_s)
        await self._done("start_tabs", urls)
        return [{"tab": f"t{index}", "url": url, "final_url": url, "status": 503, "error": None, "load_ms": 40,
                 "site": site_of(url), "outcome": self.start_tabs_outcome} for index, url in enumerate(urls)]

    def forget(self):
        self.did.append(("forget", None))


@pytest.fixture
def browser(tmp_path, monkeypatch):
    """A browser on Linux, its home and its app folder, the current directory, under ``tmp_path``, its control a fake
    and its launches recorded."""
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux"))
    monkeypatch.setattr(website_browser.environment, "_shm_bytes", lambda path="/dev/shm": 1 << 30)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "folder" / ".agentenv").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "folder")
    monkeypatch.setenv("ENVIRONMENT_NAME", "chromium")
    monkeypatch.setenv("AGENTENV_APP_SETTINGS", json.dumps({"chromium": {"start_urls": ["https://shop.test/"]}}))
    monkeypatch.setattr(browser_log, "LOG_DIR", tmp_path / "logs")
    made = Browser()
    made.control = FakeControl()
    launches = []

    async def stop_app():
        launches.append("stop")
        return True

    async def launch_app(cause="launch"):
        launches.append(f"launch {cause}")
        return 1

    async def until_ready():
        launches.append("ready")

    @contextlib.contextmanager
    def paused():
        launches.append("pause")
        yield
        launches.append("continue")

    made._computer_app = SimpleNamespace(stop_app=stop_app, launch_app=launch_app, paused=paused, timeout_s=60.0)
    made._until_ready = until_ready
    made.launches = launches
    return made


def _file(path, data: bytes) -> dict:
    return {"kind": "file", "file": {"bytes": base64.b64encode(data).decode(), "name": path.rsplit("/", 1)[-1]},
            "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": path}}}


def _parts(*parts):
    return AddDataRequest(parts=list(parts)).parts


def _files(parts) -> dict:
    return {part.metadata[DATA_OBJECTS_EXTENSION_URI]["path"]: base64.b64decode(part.file.bytes) for part in parts}


@pytest.mark.asyncio
async def test_data_get_is_its_state_its_profile_without_caches_or_locks_and_its_finished_downloads(browser):
    home = website_browser.environment.Path.home()
    for path, data in (("Default/Cookies", b"db"), ("Default/Cookies-journal", b"j"), ("Default/Cache/x", b"c"),
                       ("Default/Code Cache/js/y", b"c"), ("GPUCache/z", b"c"), ("Default/Local Storage/leveldb/CURRENT",
                       b"m"), ("Crashpad/r", b"c")):
        (home / PROFILE / path).parent.mkdir(parents=True, exist_ok=True)
        (home / PROFILE / path).write_bytes(data)
    (home / PROFILE / "SingletonLock").symlink_to("host-123")
    (home / "Downloads" / "report.csv").write_bytes(b"a,b")
    (home / "Downloads" / "big.zip.crdownload").write_bytes(b"part")

    files = _files(await browser.browser_get_data())

    state = json.loads(files.pop("browser-state.json"))
    assert set(state) == {"cookies", "local_storage", "tabs", "closed_tabs", "downloads", "clock", "start_tabs"}
    assert state["cookies"] == FakeControl().cookie_list and "partitionKey" not in state["cookies"][0]
    assert state["clock"]["mode"] == "real" and state["tabs"][0]["id"] == "t1" and state["closed_tabs"][0]["id"] == "t0"
    assert files == {f"{PROFILE}/Default/Cookies": b"db", f"{PROFILE}/Default/Cookies-journal": b"j",
                     f"{PROFILE}/Default/Local Storage/leveldb/CURRENT": b"m", "Downloads/report.csv": b"a,b",
                     "browser-log.jsonl": b""}
    assert not list((home / ".cache").iterdir())


@pytest.mark.asyncio
async def test_data_get_exports_the_log_written_so_far_and_data_add_passes_over_it(browser):
    browser.log.note("reset", {})
    pending = browser.log.reserve()
    files = _files(await browser.browser_get_data())
    browser.log.write(pending, {"reset": {}})

    logged = [json.loads(line) for line in files["browser-log.jsonl"].decode().splitlines()]
    assert [line["seq"] for line in logged] == [1] and logged[0]["reset"] == {}
    await browser.browser_add_data(_parts(_file("browser-state.json", _snapshot_state()),
                                          _file("browser-log.jsonl", files["browser-log.jsonl"])))
    assert ("cookies", ["sid"]) in browser.control.did


@pytest.mark.asyncio
async def test_data_get_reads_the_state_then_copies_the_profile_with_the_browser_paused(browser, monkeypatch):
    stage = website_browser.data.stage_profile

    def staged(*args):
        browser.launches.append("copy")
        stage(*args)

    monkeypatch.setattr(website_browser.data, "stage_profile", staged)
    await browser.browser_get_data()

    assert browser.launches == ["pause", "copy", "continue"]


@pytest.mark.asyncio
async def test_data_get_leaves_out_what_the_browser_removes_while_it_copies(browser, monkeypatch):
    home = website_browser.environment.Path.home()
    for path in ("Default/Cookies", "Default/Cookies-journal", "Default/Session Storage/LOG"):
        (home / PROFILE / path).parent.mkdir(parents=True, exist_ok=True)
        (home / PROFILE / path).write_bytes(b"x")
    copy2 = website_browser.data.shutil.copy2

    def removed_meanwhile(source, target):
        if str(source).endswith("-journal"):
            raise FileNotFoundError(source)
        return copy2(source, target)

    monkeypatch.setattr(website_browser.data.shutil, "copy2", removed_meanwhile)
    files = _files(await browser.browser_get_data())

    assert set(files) == {"browser-state.json", "browser-log.jsonl", f"{PROFILE}/Default/Cookies",
                          f"{PROFILE}/Default/Session Storage/LOG"}


def _snapshot_state(**changes) -> bytes:
    state = {"cookies": FakeControl().cookie_list, "local_storage": [{"origin": "https://shop.test",
             "localStorage": [{"name": "cart", "value": "3"}]}], "closed_tabs": [], "downloads": [],
             "clock": {"mode": "real", "now": "t"}, **changes}
    return json.dumps(state).encode()


@pytest.mark.asyncio
async def test_a_snapshots_state_loads_live_its_cookies_and_storage_then_the_start_tabs_again_and_its_files(browser):
    home = website_browser.environment.Path.home()

    await browser.browser_add_data(_parts(_file("browser-state.json", _snapshot_state()),
                                          _file("Uploads/cv.pdf", b"cv"), _file("Downloads/old.txt", b"o")))

    assert browser.control.did == [("cookies", ["sid"]), ("local_storage", ["https://shop.test"]),
                                   ("start_tabs", ["https://shop.test/"])]
    assert browser.launches == [] and (home / "Uploads/cv.pdf").read_bytes() == b"cv"
    assert (home / "Downloads/old.txt").read_bytes() == b"o"


@pytest.mark.asyncio
async def test_a_loads_tabs_replace_the_open_ones_and_a_data_part_is_a_state_too(browser):
    from agentenv_protocol.types import DataPart
    state = {"tabs": [{"url": "https://a.test/", "active": False}, {"id": "x", "url": "https://b.test/",
                                                                     "active": True, "title": "B"}]}

    await browser.browser_add_data([DataPart(data=state)])

    assert browser.control.did == [("tabs", [("https://a.test/", False), ("https://b.test/", True)])]


@pytest.mark.asyncio
async def test_a_profile_is_restored_with_the_browser_stopped_and_relaunched_before_the_rest(browser):
    home = website_browser.environment.Path.home()
    (home / PROFILE / "Default").mkdir(parents=True)
    (home / PROFILE / "Default" / "Stale").write_bytes(b"old")

    await browser.browser_add_data(_parts(_file(f"{PROFILE}/Default/History", b"h"),
                                          _file("browser-state.json", _snapshot_state(local_storage=[]))))

    assert browser.launches == ["stop", "launch data/add", "ready"]
    assert sorted(p.name for p in (home / PROFILE / "Default").iterdir()) == ["History"]
    assert browser.control.did == [("cookies", ["sid"]), ("start_tabs", ["https://shop.test/"])]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [".ssh/id_rsa", "Documents/x", "../escape", PROFILE, "Downloads", "Uploads"])
async def test_a_file_outside_the_browsers_folders_or_in_place_of_one_is_refused_before_anything_is_applied(browser,
                                                                                                         path):
    with pytest.raises(ValueError, match="none of the browser's|not a normalized relative path|a folder of the browser's"):
        await browser.browser_add_data(_parts(_file("browser-state.json", _snapshot_state()), _file(path, b"x")))
    assert browser.control.did == [] and browser.launches == []


@pytest.mark.asyncio
@pytest.mark.parametrize("link, path", [("Uploads/cv.txt", "Uploads/cv.txt"), ("Uploads/docs", "Uploads/docs/a.txt")])
async def test_a_file_whose_path_a_link_leads_out_of_the_browsers_folders_is_refused_before_any_is_written(
        browser, link, path):
    home = website_browser.environment.Path.home()
    (home / "Documents").mkdir()
    (home / ".bashrc").write_text("mine")
    (home / link).symlink_to(home / (".bashrc" if link.endswith(".txt") else "Documents"))

    with pytest.raises(ValueError, match="through a link"):
        await browser.browser_add_data(_parts(_file("Downloads/ok.txt", b"ok"), _file(path, b"x"),
                                              _file("browser-state.json", _snapshot_state())))

    assert (home / ".bashrc").read_text() == "mine" and not list((home / "Documents").iterdir())
    assert not (home / "Downloads/ok.txt").exists() and browser.control.did == []


@pytest.mark.asyncio
async def test_once_an_agent_has_acted_a_load_sets_cookies_without_the_start_tabs_and_cant_change_profile_or_tabs(
        browser):
    AGENT_ACTED.write_text("t")

    await browser.browser_add_data(_parts(_file("browser-state.json", _snapshot_state(local_storage=[]))))
    for refused in (_file(f"{PROFILE}/Default/History", b"h"),
                    _file("browser-state.json", _snapshot_state(tabs=[{"url": "https://a.test/"}]))):
        with pytest.raises(InvalidArgument, match="an agent has acted"):
            await browser.browser_add_data(_parts(refused))

    assert browser.control.did == [("cookies", ["sid"])] and browser.launches == []
    assert browser._resets_failed_add() is False


@pytest.mark.asyncio
async def test_a_load_that_fails_part_way_says_what_it_had_applied_and_one_before_any_agent_is_reset(browser):
    browser.control.fail = "local_storage"

    with pytest.raises(RuntimeError, match=r"local_storage failed \(the load had applied the cookies\)"):
        await browser.browser_add_data(_parts(_file("browser-state.json", _snapshot_state())))

    assert browser._resets_failed_add() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("sites, refused", [([{"name": "shop", "domains": ["shop.test"]}], True), ([], False)],
                         ids=["own-site", "live"])
async def test_a_load_whose_start_tab_on_an_env_site_doesnt_load_fails_naming_it_and_a_live_one_doesnt(
        browser, sites, refused):
    browser.control.start_tabs_outcome = "failed"
    await browser.browser_sites_set(sites)
    load = browser.browser_add_data(_parts(_file("browser-state.json", _snapshot_state())))

    if refused:
        with pytest.raises(InvalidArgument, match=r"start tab https://shop.test/ on the env's site shop answered 503 "
                                                  r"\(the load had applied the cookies, the local storage\)"):
            await load
    else:
        await load
    assert browser.start_tab_launches[-1]["tabs"][0]["site"] == ("shop" if refused else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("set_up_s, waits", [(None, (60, 30)), (45, (15, 15)), (75, (0, 0))],
                         ids=["a-reopen", "a-launch-whose-setup-took-45-s", "a-launch-past-its-timeout"])
async def test_start_tabs_settle_within_the_ready_timeout_of_their_launch_its_setup_counted(browser, set_up_s, waits):
    loop = asyncio.get_running_loop()

    await browser._load_start_tabs(["https://shop.test/"], "t", None if set_up_s is None else loop.time() - set_up_s)

    own, live = browser.control.waits
    assert (round(own), round(live)) == waits


@pytest.mark.asyncio
async def test_after_a_reset_relaunches_it_a_start_tab_on_an_env_site_that_failed_fails_the_reset(browser):
    def launched(url, site, outcome, **why):
        return {"at": "t", "tabs": [{"tab": "t1", "url": url, "final_url": None, "status": None, "error": None,
                                     "load_ms": 60000, "site": site, "outcome": outcome, **why}]}

    browser._following = asyncio.get_running_loop().create_future()
    browser._following.set_result(None)
    browser.start_tab_launches = [launched("https://slack.com/", None, "timed_out")]
    await browser._relaunched("data/reset")
    browser.start_tab_launches.append(launched("https://shop.test/", "shop", "timed_out"))
    with pytest.raises(InvalidArgument, match="start tab https://shop.test/ on the env's site shop was still loading "
                                              "after 60 s"):
        await browser._relaunched("data/reset")
    browser.start_tab_launches.append(launched("https://shop.test/", "shop", "failed",
                                               error="net::ERR_CONNECTION_REFUSED"))
    with pytest.raises(InvalidArgument, match="didn't load: net::ERR_CONNECTION_REFUSED"):
        await browser._relaunched("data/reset")
    assert browser.launches == ["ready"]


@pytest.mark.asyncio
async def test_after_a_reset_relaunches_it_start_tabs_still_settling_past_its_time_and_margin_fail_the_reset(
        browser, monkeypatch):
    monkeypatch.setattr(website_browser.environment, "START_TABS_MARGIN_S", 0.05)
    browser._computer_app.timeout_s = 0.05
    browser._following = asyncio.get_running_loop().create_future()

    with pytest.raises(DeviceUnavailable, match="start tabs hadn't settled after 0.1 s"):
        await browser._relaunched("data/reset")


@pytest.mark.asyncio
async def test_sites_set_keeps_the_env_sites_a_start_tab_matches_by_name_or_one_label_under_a_wildcard(browser):
    environment = website_browser.environment
    before = environment._read_sites()

    await browser.browser_sites_set([{"name": "shop", "domains": ["shop.test"]},
                                     {"name": "mail", "domains": ["*.mail.test"]}])
    sites = environment._read_sites()
    with pytest.raises(InvalidArgument, match=r"sites\[0\]: domains"):
        await browser.browser_sites_set([{"name": "shop", "domains": []}])

    assert before == [] and [site.name for site in environment._read_sites()] == ["shop", "mail"]
    assert [environment._site_of(url, sites) for url in (
        "https://shop.test/", "https://SHOP.test./cart", "https://www.mail.test/", "https://a.b.mail.test/",
        "https://mail.test/", "https://slack.com/", "about:blank")] == ["shop", "shop", "mail", None, None, None, None]


@pytest.mark.asyncio
async def test_a_state_that_doesnt_fit_is_refused_naming_its_fields_and_never_a_value(browser):
    bad = _snapshot_state(cookies=[{"name": "sid", "value": "SECRET", "maxAge": 5}])
    with pytest.raises(InvalidArgument, match=r"browser-state.json: cookies\.0") as refused:
        await browser.browser_add_data(_parts(_file("browser-state.json", bad)))
    assert "SECRET" not in str(refused.value) and browser.control.did == []


@pytest.mark.asyncio
async def test_data_reset_ends_its_addresses_restores_its_profile_empties_its_folders_and_starts_over(browser):
    home = website_browser.environment.Path.home()
    (home / PROFILE).mkdir(parents=True)
    (home / PROFILE / "Bookmarks").write_text("installed")
    browser._computer_app_data_plane().keep_initial()
    (home / PROFILE / "Bookmarks").write_text("changed")
    (home / PROFILE / "History").write_text("visited")
    (home / "Downloads" / "r.csv").write_text("r")
    (home / "Uploads" / "cv.pdf").write_text("cv")
    token = browser.cdp_tokens.mint("agent", "http://x").http_url.rsplit("/cdp/", 1)[1].strip("/")
    AGENT_ACTED.write_text("t")

    await browser.browser_reset_data()

    assert (home / PROFILE / "Bookmarks").read_text() == "installed" and not (home / PROFILE / "History").exists()
    assert not list((home / "Downloads").iterdir()) and not list((home / "Uploads").iterdir())
    assert browser.cdp_tokens.address(token) is None and not AGENT_ACTED.exists()
    assert browser.control.did == [("forget", None)]


def test_its_card_declares_a_file_tree_it_loads_from_and_a_state_as_a_data_part(browser):
    intake = next(e for e in browser.declared_extensions() if e.uri == INTAKE_EXTENSION_URI)
    assert intake.params["get"] == [{"part": "file", "format": "file-tree"}]
    assert {(entry["part"], entry["format"]) for entry in intake.params["add"]} == {("file", "file-tree"),
                                                                                    ("data", "json")}
