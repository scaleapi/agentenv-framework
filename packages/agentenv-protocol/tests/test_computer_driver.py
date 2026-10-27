"""ComputerDriverEnvironment as a class: the checks at class definition, operations chosen per platform, settings,
the generated card, readiness and how driver failures are reported."""
from __future__ import annotations

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

import asyncio  # noqa: E402
import base64  # noqa: E402
import io  # noqa: E402

from computer_fake import LINUX, FakeDriver, FakeScreen, MinimalDriver, http_client, mcp_session  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from agentenv_protocol.computer import (  # noqa: E402
    SETTINGS_ENV,
    ComputerDriverEnvironment,
    ComputerSettings,
    Display,
    Image,
    InvalidArgument,
    OutOfBounds,
    Platform,
    ReadyCheck,
    capture,
    click,
    computer_driver,
    display,
    drag,
    pointer_move,
    press,
    run_in_session,
    scroll,
    set_display,
    type_text,
)
from agentenv_protocol.computer.keys import canonical  # noqa: E402
from agentenv_protocol.computer.operations import declared, resolve  # noqa: E402
from agentenv_protocol.computer.platform import detect, normalize, supported  # noqa: E402
from agentenv_protocol.computer.settings import load_settings  # noqa: E402
from agentenv_protocol.computer.tools import _modifiers  # noqa: E402

MACOS = Platform(os="macos", arch="arm64")


def _required(namespace: dict) -> dict:
    """The required operations as no-op methods, with ``namespace`` added on top."""
    ops = {
        "screen": display(lambda self: Display(10, 10)),
        "grab": capture(lambda self: PILImage.new("RGB", (10, 10))),
        "click_at": click(lambda self, *a: None),
        "drag_along": drag(lambda self, *a: None),
        "scroll_at": scroll(lambda self, *a: None),
        "type_in": type_text(lambda self, *a: None),
        "press_keys": press(lambda self, *a: None),
    }
    return {**ops, **namespace}


def _define(namespace: dict, **features) -> type:
    return type("Driver", (ComputerDriverEnvironment,), {**_required(namespace), **features})


def test_a_driver_missing_required_operations_is_refused_when_defined():
    with pytest.raises(TypeError, match=r"missing required operation\(s\): @drag, @press"):
        type("Driver", (ComputerDriverEnvironment,),
             {k: v for k, v in _required({}).items() if k not in ("drag_along", "press_keys")})


def test_half_a_group_is_refused():
    with pytest.raises(TypeError, match="the pointer group needs all of .*missing @pointer_down, @pointer_up, "
                                        "@pointer_position"):
        _define({"move": pointer_move(lambda self, x, y: None)})


def test_two_methods_one_platform_could_both_match_are_refused():
    with pytest.raises(TypeError, match="@capture is marked on both"):
        _define({"grab_x11": capture(display="x11")(lambda self: None)})


def test_predicates_choose_one_method_per_platform():
    Driver = _define({
        "grab": capture(display="x11")(lambda self: "x11"),
        "grab_wayland": capture(display="wayland")(lambda self: "wayland"),
    })
    assert resolve(Driver, LINUX)["capture"] == "grab"
    assert resolve(Driver, Platform(os="linux", display="wayland"))["capture"] == "grab_wayland"
    with pytest.raises(RuntimeError, match=r"no method for @capture on this platform \(\{'os': 'macos'"):
        Driver(platform=MACOS)


def test_a_group_with_a_member_that_doesnt_match_the_platform_is_dropped():
    assert FakeDriver(platform=LINUX).has("pointer")
    Driver = type("Driver", (FakeDriver,), {"move": pointer_move(os="windows")(lambda self, x, y: None)})
    driver = Driver()
    assert not driver.has("pointer") and "pointer_down" not in driver._computer.operations


def test_unknown_predicates_are_refused():
    with pytest.raises(TypeError, match=r"unknown predicate\(s\) \['kernel'\]"):
        capture(kernel="6.8")
    assert normalize({"arch": ["amd64", "arm64"], "os": "linux"}) == {"arch": frozenset({"amd64", "arm64"}),
                                                                       "os": frozenset({"linux"})}


@pytest.mark.parametrize("features, error", [
    ({"buttons": ("right",)}, "buttons include left"),
    ({"buttons": ("left", "thumb")}, "buttons include left"),
    ({"max_click_count": 0}, "max_click_count is an integer of at least 1"),
    ({"max_wait_ms": 1000}, "max_wait_ms is an integer of at least 60000"),
    ({"keys": ("enter",)}, "keys are W3C key values in their W3C case"),
    ({"keys": ("Return",)}, "keys: 'Return' is not a W3C key value"),
    ({"display_modes": ((10, 10),)}, "display_modes and @set_display come together"),
])
def test_features_are_checked_when_the_class_is_defined(features, error):
    with pytest.raises(TypeError, match=error):
        _define({}, **features)


def test_set_display_without_modes_is_refused():
    with pytest.raises(TypeError, match="display_modes and @set_display come together"):
        _define({"resize": set_display(lambda self, w, h, s: None)})


def test_a_driver_method_replaces_the_linux_default():
    Driver = _define({"run": run_in_session(lambda self, *a: None)})
    assert declared(Driver)["run_in_session"] == [("run", {})]
    assert declared(MinimalDriver)["run_in_session"] == [("_computer_linux_run_in_session", {"os": frozenset({"linux"})})]


def test_the_platform_is_detected(tmp_path, monkeypatch):
    release = tmp_path / "os-release"
    release.write_text('NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="24.04"\n# comment\nbare\n')
    monkeypatch.setattr("agentenv_protocol.computer.platform.sys.platform", "linux")
    monkeypatch.setattr("agentenv_protocol.computer.platform._platform.machine", lambda: "aarch64")
    assert detect({"DISPLAY": ":0"}, release) == Platform("linux", "ubuntu", "24.04", "arm64", "x11")
    assert detect({"XDG_SESSION_TYPE": "Wayland", "DISPLAY": ":0"}, release).display == "wayland"
    assert detect({"WAYLAND_DISPLAY": "wayland-0"}, release).display == "wayland"
    assert detect({}, tmp_path / "missing").as_dict() == {"os": "linux", "arch": "arm64"}
    monkeypatch.setattr("agentenv_protocol.computer.platform.sys.platform", "darwin")
    assert detect({"DISPLAY": ":0"}, release) == Platform("macos", None, None, "arm64", None)


def test_settings_come_from_one_json_value():
    assert load_settings({}) == ComputerSettings()
    settings = load_settings({SETTINGS_ENV: '{"data_paths": ["notes"], "default_space": "norm", "tap_hold_ms": 20}'})
    assert settings.data_paths == ["notes"] and settings.default_space == "norm" and settings.tap_hold_ms == 20
    for raw in ('{"data_paths": ["../out"]}', '{"tap_hold_ms": -1}', '{"unknown": 1}', "not json"):
        with pytest.raises(ValueError, match=SETTINGS_ENV):
            load_settings({SETTINGS_ENV: raw})


def test_a_driver_reads_its_settings_and_platform_from_the_environment(monkeypatch):
    monkeypatch.setenv(SETTINGS_ENV, '{"queue_wait_s": 2}')
    monkeypatch.setattr("agentenv_protocol.computer.driver.detect", lambda: LINUX)
    driver = _define({})()
    assert driver.settings.queue_wait_s == 2 and driver.platform == LINUX


@pytest.mark.asyncio
async def test_the_card_declares_the_contract_the_driver_implements():
    async with http_client(FakeDriver()) as http:
        card = (await http.get("/.well-known/agent-env.json")).json()
    extensions = {e["uri"]: e for e in card["capabilities"]["extensions"]}
    computer = extensions["urn:agentenv:computer/v1"]["params"]
    assert computer["buttons"] == ["left", "right", "middle"] and computer["max_click_count"] == 3
    assert computer["keys"] == "all" and computer["platform"] == LINUX.as_dict() and computer["delivery"] == "foreground"
    assert extensions["urn:agentenv:computer-pointer/v1"]["params"]["buttons"] == ["left", "right", "middle"]
    assert set(extensions["urn:agentenv:computer-keyboard/v1"]["params"]["methods"]) == {"get", "set"}
    assert set(extensions["urn:agentenv:computer-display/v1"]["params"]["methods"]) == {"get", "set"}
    assert extensions["urn:agentenv:computer-exec/v1"]["params"]["max_timeout_s"] == 3600
    for uri in ("urn:agentenv:computer-ready/v1", "urn:agentenv:computer-spaces/v1", "urn:agentenv:intake/v1",
                "urn:agentenv:data-objects/v1", "urn:agentenv:disable-tool/v1", "urn:agentenv:enable-tool/v1"):
        assert uri in extensions
    assert card["capabilities"]["operations"] == ["data/add", "data/get"]
    assert {t["name"] for t in card["capabilities"]["tools"]} >= {"computer_click", "computer_pointer_list"}
    assert card["additionalInterfaces"] == [{"url": "/mcp", "transport": "mcp"}]


@pytest.mark.asyncio
async def test_a_minimal_driver_declares_only_what_it_has():
    async with http_client(MinimalDriver(platform=MACOS)) as http:
        card = (await http.get("/.well-known/agent-env.json")).json()
        display_set = await http.post("/agentenv/ext/computer-display/set", json={"width": 1, "height": 1})
        exec_run = await http.post("/agentenv/ext/computer-exec/run", json={"argv": ["true"], "request_id": "r"})
    extensions = {e["uri"]: e for e in card["capabilities"]["extensions"]}
    assert "urn:agentenv:computer-pointer/v1" not in extensions and "urn:agentenv:computer-keyboard/v1" not in extensions
    assert "urn:agentenv:computer-exec/v1" not in extensions
    assert set(extensions["urn:agentenv:computer-display/v1"]["params"]["methods"]) == {"get"}
    assert extensions["urn:agentenv:intake/v1"]["params"] == {"get": [{"part": "file", "format": "file-tree"}]}
    assert card["capabilities"]["operations"] == ["data/get"]
    assert {t["name"] for t in card["capabilities"]["tools"]} == {
        "computer_screenshot", "computer_click", "computer_drag", "computer_scroll", "computer_type", "computer_key",
        "computer_wait"}
    assert display_set.status_code == 404 and exec_run.status_code == 404


def test_computer_driver_names_the_driver_and_declares_its_platforms():
    @computer_driver(name="reference", supports=[{"os": {"type": "linux"}, "arch": ["amd64"]}])
    class Reference(MinimalDriver):
        pass

    driver = Reference()
    card = driver._build_card()
    assert card.name == "reference" and Reference.supports[0]["arch"] == ["amd64"]
    assert computer_driver()(Reference) is Reference
    with pytest.raises(TypeError, match="decorates a ComputerDriverEnvironment subclass"):
        computer_driver(name="x")(object)


def test_a_driver_that_skips_super_init_is_told_so():
    class Forgetful(MinimalDriver):
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError, match=r"Forgetful.__init__ must call super\(\).__init__\(\)"):
        Forgetful().create_app()


@pytest.mark.asyncio
async def test_tools_refuse_until_the_ready_checks_pass_once():
    driver = FakeDriver()
    driver.fail["display"] = RuntimeError("no X server")
    async with mcp_session(driver) as session:
        refused = await session.call_tool("computer_screenshot", {})
        served = await session.call_tool("computer_screenshot", {})
    assert refused.isError and refused.structuredContent["code"] == "device_unavailable"
    assert refused.structuredContent["retryable"] is True
    assert "display: @display failed: RuntimeError: no X server" in refused.structuredContent["message"]
    assert not served.isError and driver._computer.ready


@pytest.mark.asyncio
async def test_the_drivers_own_ready_checks_count():
    class Checked(FakeDriver):
        healthy = False

        async def ready_checks(self):
            return [ReadyCheck("session", self.healthy, "logged in" if self.healthy else "no session")]

    class Raising(FakeDriver):
        async def ready_checks(self):
            raise RuntimeError("probe crashed")

    driver = Checked()
    async with http_client(driver) as http:
        before = (await http.get("/agentenv/ext/computer-ready/status")).json()
        driver.healthy = True
        after = (await http.get("/agentenv/ext/computer-ready/status")).json()
    assert before["ready"] is False and before["checks"][-1] == {"name": "session", "ok": False, "detail": "no session"}
    assert after["ready"] is True
    async with http_client(Raising()) as http:
        raised = (await http.get("/agentenv/ext/computer-ready/status")).json()
    assert raised["checks"][-1] == {"name": "driver", "ok": False, "detail": "RuntimeError: probe crashed"}


@pytest.mark.asyncio
@pytest.mark.parametrize("captured, error", [
    (lambda screen: "pixels", "@capture returned str, not an Image"),
    (lambda screen: PILImage.new("RGB", (5, 5)), "@capture returned 5×5, but the display at scale 1 is 1280×800"),
])
async def test_a_bad_capture_is_an_internal_error(captured, error):
    class Bad(FakeDriver):
        @capture
        async def grab(self):
            return captured(self.screen)

    async with http_client(Bad()) as http:
        status = (await http.get("/agentenv/ext/computer-ready/status")).json()
    assert status["ready"] is False and error in status["checks"][1]["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, Display(0, 10), Display(10, 10, 0)])
async def test_a_bad_display_is_an_internal_error(result):
    class Bad(FakeDriver):
        @display
        async def screen_size(self):
            return result

    async with http_client(Bad()) as http:
        status = (await http.get("/agentenv/ext/computer-ready/status")).json()
    assert status["checks"][0]["ok"] is False and "@display returned" in status["checks"][0]["detail"]
    assert status["checks"][1] == {"name": "capture", "ok": False, "detail": "the display didn't answer"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["png", "jpeg", "RGB", "BGRA"])
async def test_a_capture_may_be_encoded_or_raw_pixels(mode):
    screen = FakeScreen(4, 2)
    screen.image.putpixel((1, 0), (10, 20, 30))

    class Raw(FakeDriver):
        @capture
        async def grab(self):
            if mode in ("png", "jpeg"):
                buffer = io.BytesIO()
                self.screen.image.save(buffer, format=mode.upper())
                return Image(4, 2, buffer.getvalue(), mode)
            if mode == "BGRA":
                r, g, b = self.screen.image.split()
                return Image(4, 2, PILImage.merge("RGBA", (b, g, r, r)).tobytes(), "BGRA")
            return Image(4, 2, self.screen.image.tobytes(), "RGB")

    async with mcp_session(Raw(screen)) as session:
        shot = await session.call_tool("computer_screenshot", {})
    assert not shot.isError and shot.structuredContent["width"] == 4
    decoded = PILImage.open(io.BytesIO(base64.b64decode(shot.content[1].data))).convert("RGB")
    assert decoded.size == (4, 2)
    if mode != "jpeg":
        assert decoded.getpixel((1, 0)) == (10, 20, 30)


@pytest.mark.asyncio
async def test_a_failing_input_operation_may_have_been_delivered_and_others_were_not():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        driver.fail["click"] = RuntimeError("XTEST gone")
        crashed = await session.call_tool("computer_click", {"x": 1, "y": 1})
        driver.fail["click"] = OutOfBounds("the window moved")
        refused = await session.call_tool("computer_click", {"x": 1, "y": 1})
        driver.fail["capture"] = RuntimeError("mss failed")
        broken = await session.call_tool("computer_screenshot", {})
    assert crashed.structuredContent == {"code": "delivery_failed", "message": "@click failed: RuntimeError: XTEST gone",
                                         "retryable": False, "applied": "unknown"}
    assert refused.structuredContent["code"] == "out_of_bounds" and refused.structuredContent["applied"] == "no"
    assert broken.structuredContent["code"] == "internal" and broken.structuredContent["applied"] == "unknown"


@pytest.mark.asyncio
async def test_a_failure_after_part_of_a_call_was_delivered_is_applied_unknown():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        driver.fail["pointer_down"] = OutOfBounds("the window moved")
        moved_then_refused = await session.call_tool("computer_pointer_down", {"x": 3, "y": 4})
    assert moved_then_refused.structuredContent["code"] == "out_of_bounds"
    assert moved_then_refused.structuredContent["applied"] == "unknown"
    assert driver.screen.log[-1] == ("move", 3, 4)


@pytest.mark.asyncio
async def test_a_driver_mounted_on_an_app_serves_the_same_tools():
    driver = MinimalDriver()
    app = driver.mount(FastMCP("mounted"))
    async with create_connected_server_and_client_session(app) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        refused = await session.call_tool("computer_click", {})
    assert "required" not in tools["computer_click"].inputSchema and "computer_pointer_move" not in tools
    assert "frame space by default" in tools["computer_scroll"].description
    assert refused.structuredContent["code"] == "invalid_argument"
    assert refused.structuredContent["message"] == "give x and y"


@pytest.mark.asyncio
async def test_concurrent_first_calls_run_the_ready_checks_once():
    driver = FakeDriver()
    driver.slow["display"] = 0.05
    async with mcp_session(driver) as session:
        results = await asyncio.gather(*(session.call_tool("computer_wait", {"ms": 0}) for _ in range(3)))
    assert all(not r.isError for r in results) and driver._computer.ready


def test_keys_and_modifiers_that_arent_lists_of_names_are_refused():
    for key in (3, ""):
        with pytest.raises(InvalidArgument, match="a key is a W3C key value"):
            canonical(key)
    with pytest.raises(InvalidArgument, match="modifiers is a list"):
        _modifiers(FakeDriver(), "Shift")


@pytest.mark.parametrize("name", ["_computer", "_computer_ready_status", "_computer_new", "computer_click", "_application", "_mounted"])
def test_a_driver_member_that_would_replace_the_base_classs_own_is_refused(name):
    with pytest.raises(TypeError, match=rf"Driver redefines {name}, which ComputerDriverEnvironment uses itself"):
        _define({name: lambda self: None})


def test_a_drivers_own_private_helpers_and_the_hooks_are_its_own():
    Driver = _define({"_button": lambda self, d, b, down: None, "_move": lambda self: None,
                      "caller_identity": lambda self, context: "x", "buttons": ("left", "right")})
    assert Driver().caller_identity(None) == "x"


def test_a_driver_class_declares_what_it_serves_on_each_platform(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT_NAME", "deployed-name")

    @computer_driver(name="declared")
    class Declared(FakeDriver):
        def __init__(self):
            raise AssertionError("declaration builds the card without the subclass's __init__")

    linux = Declared.declaration(LINUX)
    elsewhere = MinimalDriver.declaration(Platform(os="macos"))
    uris = lambda card: {e.uri for e in card.capabilities.extensions}  # noqa: E731
    assert linux.name == "declared" and len(linux.capabilities.tools) == 14
    assert {"urn:agentenv:computer-pointer/v1", "urn:agentenv:computer-exec/v1"} <= uris(linux)
    assert elsewhere.name == "MinimalDriver" and len(elsewhere.capabilities.tools) == 7
    assert "urn:agentenv:computer-exec/v1" not in uris(elsewhere)
    computer = next(e for e in elsewhere.capabilities.extensions if e.uri == "urn:agentenv:computer/v1")
    assert computer.params["platform"] == {"os": "macos"}


def test_a_supports_entry_names_each_platform_it_covers():
    entry = {"os": {"type": "linux", "distribution": "ubuntu", "version": ["22.04", "24.04"], "display": "x11"},
             "arch": ["amd64", "arm64"]}
    assert [(p.version, p.arch) for p in supported(entry)] == [("22.04", "amd64"), ("22.04", "arm64"),
                                                               ("24.04", "amd64"), ("24.04", "arm64")]
    assert supported({"os": {"type": "macos"}}) == [Platform(os="macos")]
    with pytest.raises(ValueError, match="names its os as"):
        supported({"os": "linux"})
