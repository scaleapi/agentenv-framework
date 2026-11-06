"""``agent-env run --set`` and a computer env's ``tools`` on real computers, through the set test bundle's desktop (the
ubuntu-24.04 base with cua-driver) and its task, which lists what the agents' role sees, resets the computer and lists it
again: each ``tools`` value against cua-driver on local Docker, before and after the reset; another driver set for the
run, written and composed under the variant's id while the bundle's own env stays unwritten; and ``tools`` on a Modal VM,
reset in place. Needs Docker and the local registry on localhost:5000; the last test needs Modal credentials too, with
``[sandbox.attribution]`` tags for Modal's apps."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import agent_env
from agent_env.cli import cli
from agent_env.env import Env
from agent_env.store.base import NotFoundError
from agent_env.store.routing import namespace_routing
from tst.integration.computer.conftest import run_context
from tst.util.capabilities import skip_without_remote_sandbox

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BUNDLE = Path(__file__).parent / "set"
MANIFEST = Path(agent_env.__file__).parent / "drivers" / "envs" / "cua-driver" / "mcp.json"
CUA_DRIVER_TOOLS = {tool["name"] for tool in json.loads(MANIFEST.read_text())["tools"]}
COMPUTER_TOOLS = {"computer_screenshot", "computer_click", "computer_drag", "computer_scroll", "computer_type",
                  "computer_key", "computer_wait"}
REFERENCE = "@local/agentenv-framework/drivers/reference"


@pytest.mark.parametrize("tools, computer, own", [
    (None, COMPUTER_TOOLS, CUA_DRIVER_TOOLS),
    ("standard", COMPUTER_TOOLS, set()),
    ("driver", set(), CUA_DRIVER_TOOLS),
], ids=["unset", "standard", "driver"])
def test_a_computer_envs_tools_are_what_its_agents_see_at_deploy_and_after_a_reset(state, tools, computer, own):
    settings = ["--set", f"envs/desktop.tools={tools}"] if tools else []

    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "tools", *settings])

    assert ran.exit_code == 0, ran.output
    context = run_context(ran.output)
    assert _seen(context, "tools") == _seen(context, "again") == (computer, own)


def test_set_runs_the_bundles_computer_env_with_another_driver_under_an_id_of_its_own(state):
    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "tools", "--set", "envs/desktop.driver=reference"])

    assert ran.exit_code == 0, ran.output
    assert ("--set envs/desktop.driver=reference: 1 env and 1 task written under ids ending in (desktop.driver "
            "reference)") in ran.output
    context = run_context(ran.output)
    (deployed,) = context["deployed_envs"]
    variant = deployed["env_id"]
    assert variant.endswith("/desktop(desktop.driver reference)")
    with namespace_routing():
        assert Env.get(variant).driver.id == REFERENCE
        with pytest.raises(NotFoundError):
            Env.get(variant.removesuffix("(desktop.driver reference)"))
    computer, own = _seen(context, "tools")
    assert COMPUTER_TOOLS <= computer and not own


@skip_without_remote_sandbox("modal_vm")
def test_a_computer_envs_tools_hold_on_a_modal_vm_through_a_reset_in_place(state, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(state / "cache"))

    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "tools", "--sandbox", "modal_vm", "--set",
                                   "envs/desktop.tools=standard"])

    assert ran.exit_code == 0, ran.output
    context = run_context(ran.output)
    (deployed,) = context["deployed_envs"]
    assert (deployed["sandbox_type"], deployed["metadata"]["computer"]["form"]) == ("modal_vm", "disk")
    assert deployed["env_id"].endswith("/desktop(desktop.tools standard)")
    assert _seen(context, "tools") == _seen(context, "again") == (COMPUTER_TOOLS, set())


def _seen(context: dict, verifier: str) -> tuple[set[str], set[str]]:
    """The computer_* tools and the driver's own that the verifier ``verifier`` found the agents' role offered."""
    results = {result["id"]: set(result["description"].split()) for result in
               context["metadata"]["verifications"][verifier]["results"]}
    return results["computer"], results["own"]
