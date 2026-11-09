"""A computer app deployed alone through ``agent-env run``: a one-app computer of the computer and driver its env.toml
names, its card named after it, its service running from its folder in the venv its own server runs in, on the
settings the deploy gave it, a secret resolved at boot, its window launched once by its server as the desktop session
comes up, and the computer ready only once its server says its window and its @ready_check are; its apt packages,
one from a PPA, its download, run script and install step on its build setting, fetched at put, installed offline.
And two apps composed into one computer env, each running on its own settings from its own folder, the files their run
scripts wrote compared before the push of this machine's arch, or the image of the same key reused, as one an earlier
run pushed is. And the two apps' data planes through the computer: a snapshot loaded into it reaches each app by its
name, notes's paths and clock's hooks, the computer's snapshot keeps each app's export beside its own files, leaving the
apps' paths out of its own, and each app's data/reset, sent through the driver's route to it, puts the app back as
installed. And through the driver's route from outside the computer, a WebSocket reaches the clock's own server both
ways until it closes, only with the control token, a 20 MB answer streams through whole, and the clock sees the
address the driver was reached at. And the lock's .debs and wheels, mounted into the RUNs that install them, are in
no layer of the composed image. Needs Docker and the local registry on localhost:5000."""

import hashlib
import json
import platform
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError, InvalidStatus

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.bundle import parse_bundle
from agent_env.cli import cli
from agent_env.env import Env
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from tst.integration.computer.conftest import _composed, docker, run_context

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

APP = Path(__file__).parent / "app"
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
MB = 1 << 20


@pytest.fixture(scope="module")
def desk(tmp_path_factory):
    """The app bundle's desk, composed: clock and notes on the ubuntu-24.04 base and the reference driver."""
    bundle = parse_bundle(APP)
    yield from _composed(tmp_path_factory.mktemp("app-desk"), bundle, f"{bundle.id_root}/desk")


def test_an_app_deployed_alone_runs_its_service_on_its_settings_on_a_computer_named_after_it_as_its_driver(state,
                                                                                                         monkeypatch):
    monkeypatch.setenv("NOTES_TOKEN", "t0ken")

    ran = CliRunner().invoke(cli, ["run", str(APP), "--task", "alone"])

    assert ran.exit_code == 0, ran.output
    context = run_context(ran.output)
    (deployed,) = context["deployed_envs"]
    assert deployed["env_id"].endswith("/notes") and deployed["environment_card"]["name"] == "notes-driver"
    service = context["metadata"]["script_results"]["service"]
    version, fastfetch = service.pop("fastfetch")
    assert version.endswith("~noble") and fastfetch.startswith("fastfetch ")
    assert service == {
        "signature": "-- Ada", "token": "t0ken", "edition": "pro", "font": "DejaVu Sans", "card": "notes-driver",
        "app_ready": [{"name": "app notes", "ok": True, "detail": "ready"}],
        "launchers": ["agentenv-app-notes.desktop"], "launched": ["notes"], "hello": "Hello, world!", "license": True,
        "installed": "pro"}


def test_two_apps_composed_into_one_computer_env_each_run_on_their_own_settings(state, monkeypatch):
    monkeypatch.setenv("NOTES_TOKEN", "t0ken")

    ran = CliRunner().invoke(cli, ["run", str(APP), "--task", "together"])

    assert ran.exit_code == 0, ran.output
    context = run_context(ran.output)
    (deployed,) = context["deployed_envs"]
    assert deployed["env_id"].endswith("/desk") and deployed["environment_card"]["name"] == "desk"
    assert context["metadata"]["script_results"]["together"] == {
        "notes": {"signature": "-- Ada", "token": "t0ken"},
        "clock": {"zone": "Europe/Paris", "served": {"zone": "Europe/Paris", "known": True}}, "card": "desk",
        "app_ready": ["app clock", "app notes"], "launched": ["notes"], "folders": ["clock", "notes"], "hello": True,
        "paris": True}
    said = [line for line in ran.output.splitlines() if line.startswith(f"envs/desk: container/{ARCH}: ")]
    assert any(line.endswith(("comparing the files its parts' run scripts and install steps wrote",
                              "reusing the image of the same key, built earlier")) for line in said), said


def test_a_computers_apps_load_export_and_reset_each_through_its_own_data_plane(state, monkeypatch):
    monkeypatch.setenv("NOTES_TOKEN", "t0ken")

    ran = CliRunner().invoke(cli, ["run", str(APP), "--task", "data"])

    assert ran.exit_code == 0, ran.output
    context = run_context(ran.output)
    (deployed,) = context["deployed_envs"]
    assert sorted(child["name"] for child in deployed["environment_card"]["children_environments"]) == [
        "clock", "notes"]
    snapshots = context["metadata"]["env_snapshotted_universes"]
    desk = {"Documents/seed.txt": b"seeded\n", "Desktop/todo.txt": b"water the plants\n"}
    assert _kept(snapshots["loaded"]) == {"desk": desk, "notes": {
        "Notes/welcome.md": b"# Welcome\n", "Notes/today.md": b"# Today\n", "Notes/ideas.md": b"# Ideas\n"},
        "clock": {"alarms": ["07:00"]}}
    assert context["metadata"]["script_results"]["reset"] == {"reset": {"notes": True, "clock": True},
                                                             "notes": ["welcome.md"], "alarms": False}
    assert _kept(snapshots["after_reset"]) == {"desk": desk, "notes": {"Notes/welcome.md": b"# Welcome\n"},
                                               "clock": {"alarms": []}}


@pytest.mark.asyncio
async def test_through_the_drivers_route_a_websocket_reaches_an_apps_server_and_a_20_mb_answer_streams_whole(desk):
    with namespace_routing():
        deployed = await Env.get(desk.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            svc, headers = f"{deployed.environment_url}/svc/clock", deployed.control_headers
            ws = svc.replace("http", "ws", 1) + "/echo"
            async with connect(ws, additional_headers=headers, subprotocols=["cdp"], open_timeout=60) as socket:
                hello = json.loads(await socket.recv())
                await socket.send("Page.navigate")
                text = await socket.recv()
                await socket.send(b"\x01\x02\x03")
                data = await socket.recv()
                await socket.send("bye")
                with pytest.raises(ConnectionClosedError) as closed:
                    await socket.recv()
                subprotocol = socket.subprotocol
            with pytest.raises(InvalidStatus) as refused:
                async with connect(ws, open_timeout=60):
                    pass
            async with httpx.AsyncClient(headers=headers, timeout=120) as http:
                big = await http.get(f"{svc}/big")
                seen = (await http.get(f"{svc}/headers", headers={"x-forwarded-prefix": "/elsewhere"})).json()
        finally:
            report = await teardown_run(context)

    assert hello == {"prefix": "/svc/clock", "authorization": None} and subprotocol == "cdp"
    assert (text, data) == ("Page.navigate", b"\x03\x02\x01") and closed.value.rcvd.code == 4001
    assert refused.value.response.status_code == 403
    assert big.status_code == 200 and len(big.content) == 20 * MB
    assert hashlib.sha256(big.content).digest() == hashlib.sha256(b"".join(bytes([65 + n]) * MB
                                                                            for n in range(20))).digest()
    host = httpx.URL(deployed.environment_url).netloc.decode()
    assert seen["host"] == host and seen["headers"]["x-forwarded-host"] == host
    assert seen["headers"]["x-forwarded-prefix"] == "/svc/clock" and "authorization" not in seen["headers"]
    assert not report.failed


def test_the_locks_debs_and_wheels_install_from_mounts_and_stay_in_no_layer_of_the_image(desk):
    image = desk.metadata["composed"]["container"][ARCH]["image"]
    docker("pull", "-q", image)

    made = docker("history", "--no-trunc", "--format", "{{.CreatedBy}}", image).stdout.decode().splitlines()
    installed = docker("run", "--rm", "--entrypoint", "sh", image, "-c",
                       "dpkg-query -W -f '${Package} ${Status}\\n' hello fastfetch && "
                       "/opt/agentenv/apps/clock/venv/bin/python -c 'import websockets; print(\"websockets\")' && "
                       "if [ -e /tmp/agentenv-packages ]; then echo left; else echo clean; fi"
                       ).stdout.decode().splitlines()

    assert [line for line in made if "packages/" in line and line.startswith("COPY")] == []
    assert any(line.startswith("RUN /bin/sh -c apt-get install -y --no-install-recommends /tmp/agentenv-packages/apt/")
               for line in made)
    assert sorted(installed[:2]) == ["fastfetch install ok installed", "hello install ok installed"]
    assert installed[2:] == ["websockets", "clean"]


def _kept(snapshot: dict) -> dict:
    """Each environment of a snapshot: the computer's and notes's files by their paths, and the clock's export."""
    with namespace_routing():
        universe = EnvironmentUniverseArtifact.get(snapshot["id"], version=snapshot["version"])
        kept = {environment.environment_name: {path: file.load() for path, file in
                                               environment.get_file_artifacts().items()}
                for environment in universe.get_environment_artifacts()}
    (clock,) = kept.pop("clock").values()
    return {**kept, "clock": json.loads(clock)}
