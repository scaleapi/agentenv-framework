"""The osworld_* steps on a computer env: osworld_deploy_controller deploys the controller beside it, in remote mode
against the guest server its card advertises with the env's control token, and records both addresses; it refuses a
computer without the app before it creates a sandbox, and logs no secret. osworld_initialize sets an OSWorld-V2 task up
with a reset and /initialize_task, and a classic one through /setup. osworld_evaluate grades on the controller: an
OSWorld-V2 task by /evaluate, a classic one func by func combined by its conj, a controller error as a score of 0, a
turn that errored as a skip unless the agent ran out of its budget, result files staged first on a re-grade."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent_env.artifact import DockerImageArtifact
from agent_env.osworld import controller, guest
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.osworld import deploy_controller as deploy_module
from agent_env.task_step.task_steps.osworld.deploy_controller import OSWorldDeployControllerTaskStep
from agent_env.task_step.task_steps.osworld.evaluate import OSWorldEvaluateTaskStep
from agent_env.task_step.task_steps.osworld.initialize import OSWorldInitializeTaskStep
from tst.unit.osworld.helpers import CONTROLLER, ENV_URL, TOKEN, Controller, card, computer, route_httpx

LITELLM = "sk-model-key"


def _context(env, *prompts: PromptResponse) -> TaskStepContext:
    context = TaskStepContext()
    context.deployed_envs.append(env)
    context.prompt_responses.extend(prompts)
    return context


def _ready():
    return computer(osworld_controller_url=CONTROLLER, osworld_guest_url=f"{ENV_URL}/svc/osworld-server/osworld")


@pytest.fixture
def ctl(monkeypatch):
    stand_in = Controller()
    route_httpx(monkeypatch, controller, stand_in)
    monkeypatch.setattr(controller.asyncio, "sleep", AsyncMock())
    return stand_in


@pytest.mark.parametrize("data", [
    {"type": "osworld_deploy_controller", "env_id": "desk", "controller_image_artifact_id": "ctl", "ttl_seconds": 60,
     "cpu": 1.0, "memory_mb": 2048, "env_vars": {"JUDGE_MODEL": "m"}},
    {"type": "osworld_initialize", "env_id": "desk", "init_config": [{"type": "sleep", "parameters": {"seconds": 1}}],
     "osworld_v2_task_path": "https://host/t.py"},
    {"type": "osworld_evaluate", "env_id": "desk", "evaluator": {"funcs": "compare_table"}, "verifier_id": "v",
     "osworld_v2_task_path": "https://host/t.py", "judge_model": "m"},
], ids=lambda data: data["type"])
def test_the_steps_are_registered_and_round_trip(data):
    step = get_task_step_registry()[data["type"]].from_dict({"id": "s", "version": 1, **data})
    assert {k: v for k, v in step.to_dict().items() if k in data} == data


def test_the_legacy_task_url_field_is_read_as_the_path():
    step = OSWorldInitializeTaskStep.from_dict({"id": "s", "env_id": "desk", "osworld_v2_task_url": "https://h/t.py"})
    assert step.osworld_v2_task_path == "https://h/t.py"


class _Provider:
    ON_THIS_MACHINE = False

    def __init__(self):
        self.links = [self]
        self.create_container = AsyncMock(return_value=SimpleNamespace(
            sandbox_id="sb-1", mode="container", type="docker", tunnel_urls={8000: CONTROLLER}))


@pytest.fixture
def deploy(monkeypatch):
    provider = _Provider()
    image = DockerImageArtifact(id="cua-controller", version=3, description="controller",
                                image_name="ghcr.io/example/controller:1")
    config = MagicMock()
    config.get_litellm_api_key.return_value = LITELLM
    config.get_litellm_base_url.return_value = "https://model.example"
    ready = AsyncMock()
    get = MagicMock(return_value=image)
    monkeypatch.setattr(deploy_module.Artifact, "get", get)
    monkeypatch.setattr(deploy_module, "get_config", lambda: config)
    monkeypatch.setattr(deploy_module, "get_sandbox_provider", lambda: provider)
    monkeypatch.setattr(deploy_module.controller, "wait_ready", ready)
    return SimpleNamespace(provider=provider, ready=ready, get=get)


@pytest.mark.asyncio
async def test_deploy_runs_the_controller_against_the_card_endpoint_and_records_it(deploy, caplog):
    env = computer(card("/apps/7/osworld"), existing="kept")
    context = _context(env)
    step = OSWorldDeployControllerTaskStep(id="ctl", version=1, env_id="desk", env_vars={"JUDGE_MODEL": "m",
                                                                                         "PROVIDER": "docker"})
    with caplog.at_level(logging.DEBUG):
        await step.execute(context)

    deploy.get.assert_called_once_with("cua-controller")
    kwargs = deploy.provider.create_container.call_args.kwargs
    assert kwargs["image_name"] == "ghcr.io/example/controller:1" and kwargs["port"] == 8000
    assert kwargs["env"] == {"JUDGE_MODEL": "m", "PROVIDER": "remote", "OS_TYPE": "Ubuntu",
                             "REMOTE_VM_IP": f"{ENV_URL}/apps/7/osworld",
                             "REMOTE_SERVER_HEADERS": json.dumps({"Authorization": f"Bearer {TOKEN}"}),
                             "LITELLM_API_KEY": LITELLM, "LITELLM_BASE_URL": "https://model.example"}
    deploy.ready.assert_awaited_once_with(CONTROLLER)
    assert env.metadata == {"existing": "kept", guest.CONTROLLER_URL_KEY: CONTROLLER,
                            guest.CONTROLLER_SANDBOX_ID_KEY: "sb-1", guest.GUEST_URL_KEY: f"{ENV_URL}/apps/7/osworld"}
    [sandbox] = context.deployed_sandboxes
    assert (sandbox.sandbox_name, sandbox.sandbox_id, sandbox.tunnel_urls) == \
        ("osworld-controller-desk", "sb-1", {"8000": CONTROLLER})
    assert TOKEN not in caplog.text and LITELLM not in caplog.text


@pytest.mark.asyncio
async def test_deploy_refuses_a_computer_without_the_app_before_creating_a_sandbox(deploy, monkeypatch):
    monkeypatch.setattr(guest, "CARD_WAIT_S", 0)
    monkeypatch.setattr(guest.protocol_v1, "get_card", AsyncMock(return_value={}))
    env = computer({})
    with pytest.raises(RuntimeError, match="osworld-server computer app"):
        await OSWorldDeployControllerTaskStep(id="ctl", version=1, env_id="desk").execute(_context(env))
    deploy.provider.create_container.assert_not_called()
    assert env.metadata == {}


@pytest.mark.asyncio
async def test_deploy_twice_for_one_env_is_refused(deploy):
    context = _context(computer())
    await OSWorldDeployControllerTaskStep(id="a", version=1, env_id="desk").execute(context)
    with pytest.raises(RuntimeError, match="already deployed"):
        await OSWorldDeployControllerTaskStep(id="b", version=1, env_id="desk").execute(context)


@pytest.mark.asyncio
async def test_initialize_v2_resets_then_initializes_the_task(ctl):
    ctl.answers.update({"/reset": httpx.Response(200, text="ok"), "/initialize_task": {"success": True}})
    await OSWorldInitializeTaskStep(id="init", version=1, env_id="desk",
                                    osworld_v2_task_path="https://host/t.py").execute(_context(_ready()))
    assert ctl.paths() == ["/reset", "/initialize_task"]
    assert ctl.bodies("/initialize_task") == [{"id": "init", "osworld_v2_task_url": "https://host/t.py"}]


@pytest.mark.asyncio
async def test_initialize_v2_fails_on_a_failed_setup(ctl):
    ctl.answers.update({"/reset": {}, "/initialize_task": {"success": False, "error": "no TASK_CLASS"}})
    with pytest.raises(RuntimeError, match="no TASK_CLASS"):
        await OSWorldInitializeTaskStep(id="init", version=1, env_id="desk",
                                        osworld_v2_task_path="https://host/t.py").execute(_context(_ready()))


@pytest.mark.asyncio
async def test_initialize_classic_sends_the_whole_config_to_setup(ctl):
    config = [{"type": "download", "parameters": {"files": [{"url": "https://h/a", "path": "/home/user/a"}]}},
              {"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://h/"]}}, {"type": "launch", "parameters": {"command": ["gedit"]}}]
    ctl.answers["/setup"] = {"success": True, "results": [{"app": "mail", "ok": True}]}
    context = _context(_ready())
    await OSWorldInitializeTaskStep(id="init", version=1, env_id="desk", init_config=config).execute(context)
    assert ctl.bodies("/setup") == [{"config": config}]
    assert context.metadata["osworld_seeded_sites"] == [{"app": "mail", "ok": True}]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, match", [({"success": False, "error": "no such type"}, "no such type"),
                                           ({"success": True, "results": [{"app": "mail", "ok": False,
                                                                           "error": "seed 500"}]}, "seed 500")])
async def test_initialize_classic_fails_on_a_failed_setup_or_site(ctl, answer, match):
    ctl.answers["/setup"] = answer
    with pytest.raises(RuntimeError, match=match):
        await OSWorldInitializeTaskStep(id="init", version=1, env_id="desk",
                                        init_config=[{"type": "sleep"}]).execute(_context(_ready()))


@pytest.mark.asyncio
async def test_a_step_needs_the_controller_deployed_first(ctl):
    with pytest.raises(RuntimeError, match="run osworld_deploy_controller"):
        await OSWorldInitializeTaskStep(id="init", version=1, env_id="desk",
                                        init_config=[{"type": "sleep"}]).execute(_context(computer()))


def _prompt(**fields) -> PromptResponse:
    return PromptResponse(**{"prompt_id": "p", "response": "done", "prompt_text": "Save the sheet", **fields})


@pytest.mark.asyncio
async def test_evaluate_v2_records_the_controllers_score(ctl):
    ctl.answers.update({"/evaluate_async": {"job_id": "j"},
                        "/evaluate_result": {"status": "done", "result": {"score": True, "message": "ok"}}})
    context = _context(_ready(), _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  osworld_v2_task_path="https://host/t.py").execute(context)
    assert ctl.bodies("/evaluate_async") == [{"osworld_v2_task_url": "https://host/t.py"}]
    assert context.metadata["verifications"]["v"] == {
        "results": [{"id": "osworld_v2", "score": 1.0, "result": True, "message": "ok"}], "score": 1.0}


@pytest.mark.asyncio
async def test_evaluate_v2_scores_a_controller_error_0(ctl):
    ctl.answers.update({"/evaluate_async": httpx.Response(404), "/evaluate": httpx.Response(500, text="boom")})
    context = _context(_ready(), _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  osworld_v2_task_path="https://host/t.py").execute(context)
    [result] = context.metadata["verifications"]["v"]["results"]
    assert context.metadata["verifications"]["v"]["score"] == 0.0
    assert result["id"] == "osworld_v2" and result["result"] is False and "HTTP 500" in result["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("v2", [True, False], ids=["v2", "classic"])
async def test_evaluate_skips_a_turn_that_errored(ctl, v2):
    fields = {"osworld_v2_task_path": "https://host/t.py"} if v2 else {"evaluator": {"funcs": "compare_table"}}
    context = _context(_ready(), _prompt(error_type="INFRA_ERROR", error_message="sandbox gone"))
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v", **fields).execute(context)
    assert context.metadata["verifications"]["v"] == {
        "results": [{"id": "prompt_error", "score": 0.0, "result": False,
                     "message": "Skipped: prompt had error_type=INFRA_ERROR"}], "score": 0.0}
    assert "/evaluate_async" not in ctl.paths()


def _scores(ctl, *scores):
    ctl.answers["/evaluate_async"] = httpx.Response(404)
    ctl.answers["/evaluate"] = [httpx.Response(200, json={"score": s, "message": f"got {s}"}) for s in scores]


@pytest.mark.asyncio
@pytest.mark.parametrize("conj, scores, expected", [("and", (1, 1), 1.0), ("and", (1, 0), 0.0), ("or", (0, 1), 1.0),
                                                    ("or", (0, 0), 0.0)])
async def test_evaluate_classic_sends_each_func_and_combines_by_conj(ctl, conj, scores, expected):
    _scores(ctl, *scores)
    evaluator = {"funcs": ["compare_table", "agent_judge"], "conj": conj,
                 "result": [{"type": "vm_file", "path": "/home/user/a.xlsx"}, {"type": "vm_file", "path": "/b"}],
                 "expected": [{"type": "cloud_file", "path": "https://h/gold.xlsx"}, None],
                 "options": [{"model": "copied-by-mistake"}, {}], "rubric": [{"criterion": "x"}]}
    context = _context(_ready(), _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v", evaluator=evaluator,
                                  judge_model="judge-m").execute(context)
    table, judge = ctl.bodies("/evaluate")
    assert table == {"func": "compare_table", "options": {}, "result": evaluator["result"][0],
                     "expected": evaluator["expected"][0], "instruction": "Save the sheet"}
    assert judge == {"func": "agent_judge", "options": {"model": "judge-m"}, "result": evaluator["result"][1],
                     "rubric": [{"criterion": "x"}], "instruction": "Save the sheet"}
    verification = context.metadata["verifications"]["v"]
    assert verification["score"] == expected
    assert [r["func"] for r in verification["results"]] == ["compare_table", "agent_judge"]
    assert verification["results"][0]["results"] == [
        {"id": "compare_table", "score": float(scores[0]), "result": True, "message": f"got {scores[0]}"}]


@pytest.mark.asyncio
async def test_evaluate_classic_one_func_keeps_its_partial_score(ctl):
    _scores(ctl, 0.4)
    context = _context(_ready(), _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "agent_judge"}).execute(context)
    assert context.metadata["verifications"]["v"]["score"] == 0.4


@pytest.mark.asyncio
async def test_evaluate_classic_scores_a_failing_func_0_and_grades_the_rest(ctl):
    ctl.answers["/evaluate_async"] = httpx.Response(404)
    ctl.answers["/evaluate"] = [httpx.Response(500, text="getter failed"), httpx.Response(200, json={"score": 1})]
    context = _context(_ready(), _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": ["compare_table", "file_contains"], "conj": "or"}).execute(context)
    verification = context.metadata["verifications"]["v"]
    assert verification["score"] == 1.0 and verification["debug"]["error"] is True
    assert "getter failed" in verification["results"][0]["results"][0]["message"]


@pytest.mark.asyncio
async def test_evaluate_infeasible_is_graded_on_the_agents_answer(ctl):
    context = _context(_ready(), _prompt(response="[INFEASIBLE] no such app"))
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": ["infeasible"]}).execute(context)
    assert context.metadata["verifications"]["v"]["score"] == 1.0
    await OSWorldEvaluateTaskStep(id="e2", version=1, env_id="desk", verifier_id="w",
                                  evaluator={"funcs": "compare_table"}).execute(context)
    [[result]] = [r["results"] for r in context.metadata["verifications"]["w"]["results"]]
    assert result["id"] == "infeasible_on_feasible" and context.metadata["verifications"]["w"]["score"] == 0.0
    assert ctl.paths() == []


@pytest.mark.asyncio
async def test_evaluate_grades_a_run_that_ran_out_of_turns_and_marks_it(ctl):
    _scores(ctl, 1)
    context = _context(_ready(), _prompt(error_type="AGENT_ERROR",
                                         error_message="Max turns (50) reached without completion."))
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "compare_table"}).execute(context)
    verification = context.metadata["verifications"]["v"]
    assert verification["score"] == 1.0
    assert verification["partial_credit"]["reason"] == "budget_exhausted"
    assert context.metadata["partial_credit"] == {"v": verification["partial_credit"]}


@pytest.mark.asyncio
async def test_evaluate_runs_postconfig_once_and_grades_when_it_fails(ctl):
    _scores(ctl, 1)
    ctl.answers["/setup"] = {"success": False, "error": "window gone"}
    postconfig = [{"type": "activate_window", "parameters": {"window_name": "x"}}]
    context = _context(_ready(), _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "compare_table", "postconfig": postconfig}).execute(context)
    assert ctl.bodies("/setup") == [{"config": postconfig}]
    assert "postconfig" not in ctl.bodies("/evaluate")[0]
    assert context.metadata["verifications"]["v"]["score"] == 1.0


def _regrade(context, files):
    context.metadata["user_overrides"] = {"step_params": {"e": {"stage_files": files}}}
    return context


@pytest.mark.asyncio
async def test_evaluate_stages_result_files_before_grading(ctl):
    _scores(ctl, 1)
    ctl.answers["/setup"] = {"success": True}
    files = [{"url": "https://signed.example/out.xlsx", "path": "/home/user/out.xlsx"}]
    context = _regrade(_context(_ready(), _prompt()), files)
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "compare_table"}).execute(context)
    assert ctl.paths()[0] == "/setup"
    assert ctl.bodies("/setup") == [{"config": [{"type": "download", "parameters": {"files": files}}]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("files, answer", [([{"url": "s3://b/out.xlsx", "path": "/x"}], None),
                                           ([{"url": "https://h/out", "path": "/x"}], {"success": False})],
                         ids=["not-http", "download-failed"])
async def test_evaluate_refuses_to_grade_an_unstaged_computer(ctl, files, answer):
    if answer is not None:
        ctl.answers["/setup"] = answer
    context = _regrade(_context(_ready(), _prompt()), files)
    with pytest.raises((ValueError, RuntimeError)):
        await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                      evaluator={"funcs": "compare_table"}).execute(context)
    assert "/evaluate_async" not in ctl.paths()
