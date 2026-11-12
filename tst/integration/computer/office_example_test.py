"""The shipped office example: its check, run against a computer of its desktop, says each file isn't there before
any is, fails the animation of a deck whose list doesn't enter, with no animation or with its title's alone, and passes
the memo and deck LibreOffice saves in Office formats, converted headless from text and from a flat ODP whose second
slide's list enters with an animation; and an agent given LibreOffice's skill writes both through the computer_* tools
under a recording. Needs Docker and the local registry; the agent's task, a model endpoint too."""

import importlib.util
import re

import pytest
from click.testing import CliRunner

from agentenv_protocol import client as protocol_v1

from agent_env.cli import cli
from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.bundle.installed import find_bundle
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from tst.integration.computer.conftest import run_context
from tst.util.capabilities import skip_without_model_endpoint

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

MEMO = "Quarterly Plan\nWe grow the team this quarter.\nWe ship the new release by March.\n"
#: The deck as a flat ODP: two slides titled as the task asks, the second's list of three entering with an animation.
DECK = """<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
 xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
 xmlns:anim="urn:oasis:names:tc:opendocument:xmlns:animation:1.0"
 xmlns:smil="urn:oasis:names:tc:opendocument:xmlns:smil-compatible:1.0"
 office:version="1.3" office:mimetype="application/vnd.oasis.opendocument.presentation">
 <office:body><office:presentation>
  <draw:page draw:name="page1">
   <draw:frame presentation:class="title" svg:x="2cm" svg:y="2cm" svg:width="24cm" svg:height="3cm">
    <draw:text-box><text:p>Quarterly Plan</text:p></draw:text-box></draw:frame>
  </draw:page>
  <draw:page draw:name="page2">
   <draw:frame presentation:class="title" draw:id="heading" xml:id="heading" svg:x="2cm" svg:y="2cm" svg:width="24cm"
    svg:height="3cm"><draw:text-box><text:p>Next Steps</text:p></draw:text-box></draw:frame>
   <draw:frame presentation:class="outline" draw:id="steps" xml:id="steps" svg:x="2cm" svg:y="6cm" svg:width="24cm"
    svg:height="10cm"><draw:text-box><text:list>
     <text:list-item><text:p>Hire</text:p></text:list-item><text:list-item><text:p>Ship</text:p></text:list-item>
     <text:list-item><text:p>Review</text:p></text:list-item></text:list></draw:text-box></draw:frame>
   <anim:par presentation:node-type="timing-root"><anim:seq presentation:node-type="main-sequence">
    <anim:par smil:begin="next"><anim:par smil:begin="0s">
     <anim:par smil:begin="0s" smil:fill="hold" presentation:node-type="on-click" presentation:preset-class="entrance"
      presentation:preset-id="ooo-entrance-appear">
      <anim:set smil:begin="0s" smil:dur="0.001s" smil:fill="hold" smil:targetElement="steps"
       smil:attributeName="visibility" smil:to="visible"/>
     </anim:par></anim:par></anim:par></anim:seq></anim:par>
  </draw:page>
 </office:presentation></office:body>
</office:document>
"""
#: The same deck with its second slide's title entering instead of its list.
TITLE_ANIMATED = DECK.replace('smil:targetElement="steps"', 'smil:targetElement="heading"')
UNANIMATED = re.sub(r'<anim:par presentation:node-type="timing-root">.*?</anim:seq></anim:par>', "", DECK, flags=re.S)
WRITE = ("import pathlib, sys; path = pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True); "
         "path.write_text(sys.argv[2])")
#: Each file converted into Documents, headless, under the name the task asks for.
CONVERT = ("cd /tmp/office && soffice --headless --convert-to docx --outdir ~/Documents "
           "quarterly-plan.txt && soffice --headless --convert-to pptx --outdir ~/Documents quarterly-plan.fodp")


@pytest.mark.asyncio
async def test_the_check_names_what_isnt_there_or_isnt_animated_and_passes_what_libreoffice_saves(office_desktop):
    verify = _verify()
    with namespace_routing():
        deployed = await Env.get(office_desktop.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            with protocol_v1.authorized({deployed.environment_url: deployed.control_headers}):
                missing = await verify(f"{deployed.environment_url}/mcp", deployed.environment_url)
                await _write(deployed, "/tmp/office/quarterly-plan.txt", MEMO)
                await _write(deployed, "/tmp/office/quarterly-plan.fodp", UNANIMATED)
                await _run(deployed, ["sh", "-c", CONVERT])
                unanimated = await verify(f"{deployed.environment_url}/mcp", deployed.environment_url)
                await _write(deployed, "/tmp/office/quarterly-plan.fodp", TITLE_ANIMATED)
                await _run(deployed, ["sh", "-c", CONVERT])
                title_animated = await verify(f"{deployed.environment_url}/mcp", deployed.environment_url)
                await _write(deployed, "/tmp/office/quarterly-plan.fodp", DECK)
                await _run(deployed, ["sh", "-c", CONVERT])
                saved = await verify(f"{deployed.environment_url}/mcp", deployed.environment_url)
        finally:
            report = await teardown_run(context)

    assert {result["id"]: result["result"] for result in missing} == {"memo": False, "slides": False,
                                                                      "animation": False}, missing
    assert "quarterly-plan.docx isn't there" in missing[0]["description"], missing
    assert {result["id"]: result["result"] for result in unanimated} == {"memo": True, "slides": True,
                                                                         "animation": False}, unanimated
    assert {result["id"]: result["result"] for result in title_animated} == {"memo": True, "slides": True,
                                                                             "animation": False}, title_animated
    assert all(result["result"] for result in saved), saved
    assert not report.failed


@skip_without_model_endpoint()
def test_an_agent_given_libreoffices_skill_writes_the_memo_and_the_animated_deck_under_a_recording(state):
    ran = CliRunner().invoke(cli, ["run", "office", "--task", "author"])

    assert ran.exit_code == 0, ran.output
    assert "passed (check: 1)" in ran.output, ran.output
    context = run_context(ran.output)
    entry = context["metadata"]["computer_recordings"]["agent"]
    assert (entry["state"], entry["missing_segments"]) == ("ended", 0)
    skills = context["metadata"]["env_skills"]["default-agent"]["desktop"]
    assert any("Saving in Office formats" in skill for skill in skills.values()), list(skills)


def _verify():
    """The office example's check, loaded from its bundle as the verifier step loads it."""
    path = find_bundle("office").root / "artifacts" / "check_office" / "verify.py"
    spec = importlib.util.spec_from_file_location("check_office", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify


async def _write(deployed, path: str, text: str) -> None:
    await _run(deployed, ["python3", "-c", WRITE, path, text])


async def _run(deployed, argv: list[str]) -> str:
    ran = await inside.run(deployed, argv, timeout_s=120)
    assert ran["exit_code"] == 0, ran
    return ran["stdout"]
