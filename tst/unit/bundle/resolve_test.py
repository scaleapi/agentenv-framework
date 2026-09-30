"""Resolving a parsed bundle: every declared reference rewritten to the id it names, before any write."""

import json

import pytest

from agent_env.bundle import BundleError, BundleKind, parse_bundle
from agent_env.bundle import resolve as resolve_module
from agent_env.bundle.resolve import BuiltImage, Output, Reference, resolve_bundle
from agent_env.entity_refs import EntityKind, EntityRef
from agent_env.env.env import Env
from agent_env.task import Task
from agent_env.task_step.task_step import TaskStep

ROOT = "@local/~/triage"
LAYOUT = {
    "envs/tickets/Dockerfile": "FROM scratch\n",
    "agents/solver/Dockerfile": "FROM scratch\n",
    "artifacts/greeting/hello.txt": "hello\n",
    "artifacts/greeting/check.py": "print('ok')\n",
    "artifacts/base-mcp/Dockerfile": "FROM scratch\n",
    "artifacts/check.py": "print('stray')\n",
    "skills/pdf/SKILL.md": "---\nname: pdf\n---\n",
}


class _Composite(Env):
    """An env type whose toml names other envs and images."""

    type = "composite_test"
    toml_refs = (
        EntityRef.env("mcp_server_envs[]"),
        EntityRef.artifact("image", artifact_type="docker_image"),
        EntityRef.artifact("backend_image", artifact_type="docker_image"),
    )


class _Plugin(TaskStep):
    """A step type that declares no references."""

    type = "plugin_test"


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    envs, steps = resolve_module.get_env_registry(), resolve_module.get_task_step_registry()
    monkeypatch.setattr(resolve_module, "get_env_registry", lambda: {**envs, _Composite.type: _Composite})
    monkeypatch.setattr(resolve_module, "get_task_step_registry", lambda: {**steps, _Plugin.type: _Plugin})


@pytest.fixture
def make(tmp_path, monkeypatch):
    """A bundle at ``~/triage`` with the fixed layout, plus the given tasks, evals and files."""
    monkeypatch.setenv("HOME", str(tmp_path))

    def make(tasks=None, evals=None, files=None):
        spec = {**LAYOUT, **(files or {})}
        spec.update({f"tasks/{name}.json": json.dumps(steps) for name, steps in (tasks or {}).items()})
        spec.update({f"evals/{name}.toml": text for name, text in (evals or {}).items()})
        for rel, text in spec.items():
            path = tmp_path / "triage" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return parse_bundle(tmp_path / "triage")

    return make


def resolved(bundle, name):
    return next(e for e in resolve_bundle(bundle).entries if e.entry.name == name)


def problems(bundle) -> tuple[str, ...]:
    with pytest.raises(BundleError) as caught:
        resolve_bundle(bundle)
    return caught.value.problems


def refs(entry) -> list[tuple]:
    return [(r.kind, r.id, r.version, getattr(r.local, "name", None) or getattr(r.local, "dockerfile", None))
            for r in entry.references]


def test_a_step_naming_bundle_entities_is_rewritten_to_their_ids(make):
    steps = [
        {"id": "deploy", "type": "deploy_env", "env_id": "tickets"},
        {"id": "seed", "type": "load_artifact", "env_id": "tickets", "artifact_id": "greeting"},
        {"id": "agent", "type": "deploy_agent", "env_ids": ["tickets"], "a2a_agent_id": "solver"},
    ]
    bundle = make(tasks={"t": steps})
    entry = resolved(bundle, "t")
    assert entry.config == [
        {"id": "deploy", "type": "deploy_env", "env_id": f"{ROOT}/tickets"},
        {"id": "seed", "type": "load_artifact", "env_id": f"{ROOT}/tickets", "artifact_id": f"{ROOT}/greeting"},
        {"id": "agent", "type": "deploy_agent", "env_ids": [f"{ROOT}/tickets"], "a2a_agent_id": f"{ROOT}/solver"},
    ]
    assert refs(entry) == [
        (EntityKind.ENV, f"{ROOT}/tickets", None, "tickets"),
        (EntityKind.ENV, f"{ROOT}/tickets", None, "tickets"),
        (EntityKind.ARTIFACT, f"{ROOT}/greeting", None, "greeting"),
        (EntityKind.ENV, f"{ROOT}/tickets", None, "tickets"),
        (EntityKind.AGENT, f"{ROOT}/solver", None, "solver"),
    ]
    assert next(e for e in bundle.entries if e.name == "t").config == steps


def test_a_store_id_stays_as_written_with_its_pin(make):
    step = {"id": "deploy", "type": "deploy_env", "env_id": "shared-env", "env_version": 3}
    entry = resolved(make(tasks={"t": [step]}), "t")
    assert entry.config == [step]
    assert entry.references == (Reference(EntityKind.ENV, "shared-env", 3, None, "step 'deploy': env_id", None),)


def test_an_entity_named_by_its_full_id_is_still_this_bundles(make):
    entry = resolved(make(tasks={"t": [{"id": "d", "type": "deploy_env", "env_id": f"{ROOT}/tickets"}]}), "t")
    assert refs(entry) == [(EntityKind.ENV, f"{ROOT}/tickets", None, "tickets")]


def test_artifact_refs_find_skills_and_skill_refs_want_a_skill(make):
    entry = resolved(make(tasks={"t": [
        {"id": "load", "type": "load_artifact", "agent_name": "solver", "artifact_id": "pdf"},
        {"id": "skills", "type": "add_skills", "agent_name": "solver", "skills": [{"skill_artifact_id": "pdf"}]},
    ]}), "t")
    assert entry.config[0]["artifact_id"] == entry.config[1]["skills"][0]["skill_artifact_id"] == f"{ROOT}/pdf"


def test_a_nested_trigger_ref_is_rewritten(make):
    triggers = [{"id": "on-ticket", "when": {"type": "any", "of": [{"type": "env_trigger", "env_id": "tickets",
                                                                    "trigger_id": "new"}]}}]
    entry = resolved(make(tasks={"t": [
        {"id": "trig", "type": "register_agent_triggers", "agent_name": "solver", "triggers": triggers},
    ]}), "t")
    assert entry.config[0]["triggers"][0]["when"]["of"][0]["env_id"] == f"{ROOT}/tickets"


def test_an_output_is_named_under_its_task_and_later_steps_find_it(make):
    entry = resolved(make(tasks={"t": [
        {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "cli_artifact_id": "tickets-cli"},
        {"id": "install", "type": "load_artifact", "agent_name": "solver", "artifact_id": "tickets-cli"},
        {"id": "skills", "type": "add_skills", "agent_name": "solver", "cli_artifact_ids": ["tickets-cli"]},
    ]}), "t")
    output = f"{ROOT}/t/tickets-cli"
    assert [entry.config[0]["cli_artifact_id"], entry.config[1]["artifact_id"],
            entry.config[2]["cli_artifact_ids"][0]] == [output] * 3
    assert entry.outputs == (Output(EntityKind.ARTIFACT, output, 0, "cli", "cli"),)
    assert refs(entry) == [(EntityKind.ENV, f"{ROOT}/tickets", None, "tickets")]


def test_an_output_id_a_step_derives_is_named_under_its_task_too(make):
    entry = resolved(make(tasks={"t": [
        {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "tickets"},
        {"id": "skills", "type": "add_skills", "agent_name": "solver", "cli_artifact_ids": ["cli-tickets"]},
    ]}), "t")
    assert entry.config[0]["cli_artifact_id"] == entry.config[1]["cli_artifact_ids"][0] == f"{ROOT}/t/cli-tickets"
    assert [output.id for output in entry.outputs] == [f"{ROOT}/t/cli-tickets"]


def test_a_resolved_task_still_builds_its_steps(make):
    entry = resolved(make(tasks={"t": [
        {"id": "deploy", "type": "deploy_env", "env_id": "tickets"},
        {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "tickets",
         "cli_artifact_id": "tickets-cli"},
        {"id": "install", "type": "load_artifact", "agent_name": "solver", "artifact_id": "tickets-cli"},
    ]}), "t")
    task = Task.from_dict({"id": entry.entry.id, "steps": entry.config})
    assert [(step.type, getattr(step, "env_id", None)) for step in task.steps[:2]] == [
        ("deploy_env", f"{ROOT}/tickets"), ("build_mcp_cli", f"{ROOT}/tickets"),
    ]
    assert task.steps[1].cli_artifact_id == task.steps[2].artifacts[0]["id"] == f"{ROOT}/t/tickets-cli"


@pytest.mark.parametrize(("step", "problem"), [
    ({"id": "d", "type": "deploy_env", "env_id": "Tickets"},
     "step 'd': env_id: 'Tickets' isn't in this bundle, but 'tickets' is; names are case-sensitive"),
    ({"id": "d", "type": "deploy_env", "env_id": f"{ROOT}/Tickets"},
     f"step 'd': env_id: '{ROOT}/Tickets' isn't in this bundle, but '{ROOT}/tickets' is; names are case-sensitive"),
    ({"id": "l", "type": "load_artifact", "env_id": "shared-env", "artifact_id": "check.py"},
     "step 'l': artifact_id: 'check.py' names artifacts/check.py, which this bundle ignores"),
    ({"id": "l", "type": "load_artifact", "env_id": "shared-env", "artifact_id": "check"},
     "step 'l': artifact_id: 'check' names artifacts/check.py, which this bundle ignores"),
    ({"id": "d", "type": "deploy_env", "env_id": "tickets", "env_version": 2},
     "step 'd': env_id: 'tickets' is defined in this bundle, so it has no fixed version; drop the pin"),
    ({"id": "s", "type": "add_skills", "agent_name": "solver", "skills": [{"skill_artifact_id": "greeting"}]},
     "step 's': skills[0].skill_artifact_id: 'greeting' is this bundle's file_artifact_universe, but this field takes "
     "skill"),
    ({"id": "d", "type": "deploy_env", "env_id": 3}, "step 'd': env_id must be an id, not 3"),
    ({"id": "x", "type": "no_such_step"}, "step 'x': 'no_such_step' is not a known step type"),
])
def test_a_bad_step_reference_is_refused(make, step, problem):
    assert problems(make(tasks={"t": [step]})) == (f"tasks/t.json: {problem}",)


def test_an_output_cannot_share_a_bundle_entitys_name_or_take_a_pin(make):
    assert problems(make(tasks={"t": [
        {"id": "snap", "type": "snapshot_env", "env_id": "tickets", "snapshot_id": "greeting"},
        {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "cli_artifact_id": "tickets-cli"},
        {"id": "install", "type": "load_artifact", "agent_name": "solver", "artifact_id": "tickets-cli",
         "artifact_version": 2},
    ]})) == (
        "tasks/t.json: step 'install': artifact_id: 'tickets-cli' is written by this task, so it has no fixed "
        "version; drop the pin",
        "tasks/t.json: step 'snap': snapshot_id: 'greeting' is written by this step and is also this bundle's "
        "artifact 'greeting'; rename one",
    )


@pytest.mark.parametrize(("steps", "problem"), [
    ([{"id": "install", "type": "load_artifact", "agent_name": "solver", "artifact_id": "tickets-cli"},
      {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "c", "cli_artifact_id": "tickets-cli"}],
     "step 'install': artifact_id: 'tickets-cli' is written by step 'cli', which this step doesn't depend on"),
    ([{"id": "deploy", "type": "deploy_env", "env_id": "tickets"},
      {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "c", "cli_artifact_id": "tickets-cli",
       "depends_on": [{"task_step_id": "deploy"}]},
      {"id": "install", "type": "load_artifact", "agent_name": "solver", "artifact_id": "tickets-cli",
       "depends_on": [{"task_step_id": "deploy"}]}],
     "step 'install': artifact_id: 'tickets-cli' is written by step 'cli', which this step doesn't depend on"),
    ([{"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "c", "cli_artifact_id": "tickets-cli"},
      {"id": "again", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "c", "cli_artifact_id": "tickets-cli"}],
     "step 'again': cli_artifact_id: 'tickets-cli' is also written by step 'cli'; give each output its own name"),
    ([{"id": "snap", "type": "snapshot_env", "env_id": "tickets", "snapshot_id": "snap"},
      {"id": "skills", "type": "add_skills", "agent_name": "solver", "cli_artifact_ids": ["snap"]}],
     "step 'skills': cli_artifact_ids[0]: this task writes 'snap' as environment_universe, but this field takes cli"),
    ([{"id": "cli", "type": "build_mcp_cli", "env_id": "tickets"}],
     "step 'cli': build_mcp_cli can't read it (KeyError: 'command_name')"),
])
def test_an_output_must_come_from_one_earlier_step_of_the_right_type(make, steps, problem):
    assert problems(make(tasks={"t": steps})) == (f"tasks/t.json: {problem}",)


@pytest.mark.parametrize("depends_on", [
    [{"task_step_id": "cli"}],
    [{"task_step_id": "tag"}],
    ["cli"],
    ["tag"],
    [{"task_step_id": "tag"}, "cli"],
])
def test_an_output_resolves_for_a_step_that_depends_on_its_writer(make, depends_on):
    entry = resolved(make(tasks={"t": [
        {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "c", "cli_artifact_id": "tickets-cli"},
        {"id": "tag", "type": "deploy_env", "env_id": "tickets", "depends_on": [{"task_step_id": "cli"}]},
        {"id": "install", "type": "load_artifact", "agent_name": "solver", "artifact_id": "tickets-cli",
         "depends_on": depends_on},
    ]}), "t")
    assert entry.config[2]["artifact_id"] == f"{ROOT}/t/tickets-cli"


@pytest.mark.parametrize(("depends_on", "problem"), [
    ("cli", "depends_on is a list of step ids, not 'cli'"),
    ([7], 'a depends_on entry is a step id or {"task_step_id": "<step id>"}, not 7'),
    ([{"id": "cli"}], 'a depends_on entry is a step id or {"task_step_id": "<step id>"}, not {\'id\': \'cli\'}'),
    (["clii"], "depends_on 'clii' names no step in this task"),
])
def test_a_depends_on_that_cant_be_read_is_named_beside_the_output_it_then_cant_reach(make, depends_on, problem):
    assert set(problems(make(tasks={"t": [
        {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "c", "cli_artifact_id": "tickets-cli"},
        {"id": "install", "type": "load_artifact", "agent_name": "solver", "artifact_id": "tickets-cli",
         "depends_on": depends_on},
    ]}))) == {
        "tasks/t.json: step 'install': artifact_id: 'tickets-cli' is written by step 'cli', which this step doesn't "
        "depend on",
        f"tasks/t.json: step 'install': {problem}",
    }


def test_a_name_differing_only_in_unicode_form_still_resolves(make):
    nfd = "cafe\u0301"
    entry = resolved(make(files={"envs/caf\u00e9/Dockerfile": "FROM scratch\n"}, tasks={"t": [
        {"id": "deploy", "type": "deploy_env", "env_id": nfd},
        {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "c", "cli_artifact_id": f"{nfd}-cli"},
        {"id": "install", "type": "load_artifact", "agent_name": "solver", "artifact_id": "caf\u00e9-cli"},
        {"id": "store", "type": "deploy_env", "env_id": f"{nfd}-store"},
    ]}), "t")
    assert entry.config[0]["env_id"] == f"{ROOT}/caf\u00e9"
    assert entry.config[1]["cli_artifact_id"] == entry.config[2]["artifact_id"] == f"{ROOT}/t/caf\u00e9-cli"
    assert entry.config[3]["env_id"] == f"{nfd}-store"


def test_an_undeclared_step_passes_through_unless_it_repeats_a_bundle_name(make):
    clean = {"id": "tickets", "type": "plugin_test", "note": "hello", "after_step_id": "greeting"}
    assert resolved(make(tasks={"t": [clean], "hello": []}, evals={"regression": 'tasks = ["t"]'}), "t").config == [clean]
    after = {"id": "after", "type": "plugin_test", "depends_on": ["tickets"]}
    assert resolved(make(tasks={"t": [clean, after]}), "t").config == [clean, after]
    assert resolved(make(tasks={"u": [{**clean, "note": "regression"}]}), "u").config == [{**clean, "note": "regression"}]
    assert problems(make(tasks={"u": [{"id": "p", "type": "plugin_test", "target": {"name": "Tickets"}}]})) == (
        "tasks/u.json: step 'p': target.name = 'Tickets' names this bundle's env 'tickets', but plugin_test "
        "doesn't declare its entity_refs, so it can't be rewritten",
    )


def test_an_eval_names_tasks_in_the_bundle_or_the_store(make):
    bundle = make(tasks={"t": []}, evals={"regression": 'tasks = ["t", { task = "shared-task", version = 3 }]\n'})
    entry = resolved(bundle, "regression")
    assert entry.config == {"tasks": [f"{ROOT}/t", {"task": "shared-task", "version": 3}]}
    assert refs(entry) == [(EntityKind.TASK, f"{ROOT}/t", None, "t"), (EntityKind.TASK, "shared-task", 3, None)]


@pytest.mark.parametrize(("toml", "problem"), [
    ('tasks = [{ task = "t", version = 2 }]', "tasks[0].task: 't' is defined in this bundle, so it has no fixed "
     "version; drop the pin"),
    ('tasks = [{ task = "t", extra = 1 }]', "tasks[0].task: expected an id or { task = \"<id>\", version = <n> }, "
     "the version optional, not {'task': 't', 'extra': 1}"),
])
def test_a_bad_eval_reference_is_refused(make, toml, problem):
    assert problems(make(tasks={"t": []}, evals={"regression": toml})) == (f"evals/regression.toml: {problem}",)


def test_an_eval_lists_each_task_once(make):
    toml = f'tasks = ["t", "{ROOT}/t", {{ task = "t" }}, "multi-x", {{ task = "multi-x", version = 3 }}]\n'
    once = "an eval lists each task once (for repeat runs, use agent-env eval run --k)"
    assert problems(make(tasks={"t": []}, evals={"regression": toml})) == (
        f"evals/regression.toml: tasks[1]: this bundle's task 't' is also tasks[0]; {once}",
        f"evals/regression.toml: tasks[2]: this bundle's task 't' is also tasks[0]; {once}",
        f"evals/regression.toml: tasks[4]: 'multi-x' is also tasks[3]; {once}",
    )


def test_an_eval_type_other_than_eval_is_refused(make):
    assert problems(make(tasks={"t": []}, evals={"regression": 'type = "banana"\ntasks = ["t"]\n'})) == (
        "evals/regression.toml: 'banana' is not a known eval type",
    )
    assert resolved(make(tasks={"t": []}, evals={"regression": 'type = "eval"\ntasks = ["t"]\n'}), "regression")


def test_a_toml_names_envs_and_images_in_the_bundle_or_the_store(make):
    toml = ('type = "composite_test"\nmcp_server_envs = ["tickets", { env = "crm", version = 2 }]\n'
            'image = "base-mcp"\n')
    entry = resolved(make(files={"envs/both/env.toml": toml}), "both")
    assert entry.config == {"type": "composite_test", "mcp_server_envs": [f"{ROOT}/tickets", {"env": "crm", "version": 2}],
                            "image": f"{ROOT}/base-mcp"}
    assert refs(entry) == [
        (EntityKind.ENV, f"{ROOT}/tickets", None, "tickets"),
        (EntityKind.ENV, "crm", 2, None),
        (EntityKind.ARTIFACT, f"{ROOT}/base-mcp", None, "base-mcp"),
    ]


def test_an_image_key_left_out_or_given_a_dockerfile_builds_the_folder(make):
    bundle = make(files={
        "envs/web/Dockerfile": "FROM scratch\n",
        "envs/web/Dockerfile.backend": "FROM scratch\n",
        "envs/web/env.toml": 'type = "composite_test"\nbackend_image = { dockerfile = "Dockerfile.backend" }\n',
    })
    result = resolve_bundle(bundle)
    web = next(e for e in result.entries if e.entry.name == "web")
    assert web.config == {"type": "composite_test", "image": f"{ROOT}/web__env_image",
                          "backend_image": f"{ROOT}/web__backend_image"}
    assert [image for image in result.built_images if image.entry == web.entry] == [
        BuiltImage(f"{ROOT}/web__env_image", web.entry, "Dockerfile"),
        BuiltImage(f"{ROOT}/web__backend_image", web.entry, "Dockerfile.backend"),
    ]
    assert refs(web) == [(EntityKind.ARTIFACT, f"{ROOT}/web__env_image", None, "Dockerfile"),
                         (EntityKind.ARTIFACT, f"{ROOT}/web__backend_image", None, "Dockerfile.backend")]


def test_an_agent_names_a_store_image_or_builds_its_folder(make):
    result = resolve_bundle(make(files={"agents/judge/agent.toml": 'image = { artifact = "claude-image", version = 3 }\n'}))
    solver, judge = (next(e for e in result.entries if e.entry.name == name) for name in ("solver", "judge"))
    assert (solver.config, judge.config) == (
        {"image": f"{ROOT}/solver__agent_image"}, {"image": {"artifact": "claude-image", "version": 3}})
    assert result.built_images == (BuiltImage(f"{ROOT}/solver__agent_image", solver.entry, "Dockerfile"),)
    assert (refs(solver), refs(judge)) == (
        [(EntityKind.ARTIFACT, f"{ROOT}/solver__agent_image", None, "Dockerfile")],
        [(EntityKind.ARTIFACT, "claude-image", 3, None)],
    )


@pytest.mark.parametrize(("toml", "problem"), [
    ('image = "base-mcp"\nbackend_image = { dockerfile = "Dockerfile.missing" }', "backend_image: there is no 'Dockerfile.missing' in "
     "this folder to build"),
    ('image = { ref = "ghcr.io/acme/web:1" }', "image: an external image ({ ref = ... }) isn't supported yet"),
    ('image = "greeting"', "image: 'greeting' is this bundle's file_artifact_universe, but this field takes "
     "docker_image"),
    ('image = "base-mcp"\nbackend_image = { dockerfile = "../tickets/Dockerfile" }', "backend_image: there is no "
     "'../tickets/Dockerfile' in this folder to build"),
    ('image = "base-mcp"\nbackend_image = { dockerfile = "/etc/hosts" }', "backend_image: there is no '/etc/hosts' "
     "in this folder to build"),
    ('image = "base-mcp"\nmcp_server_envs = [{ env = "crm", extra = 1 }]', "mcp_server_envs[0].env: expected an id "
     "or { env = \"<id>\", version = <n> }, the version optional, not {'env': 'crm', 'extra': 1}"),
])
def test_a_bad_toml_reference_is_refused(make, toml, problem):
    bundle = make(files={"envs/web/env.toml": f'type = "composite_test"\n{toml}\n'})
    assert problems(bundle) == (f"envs/web: {problem}",)


def test_an_image_left_out_needs_a_dockerfile_to_build(make):
    assert problems(make(files={"envs/web/env.toml": 'type = "composite_test"\n'})) == (
        "envs/web: image: there is no 'Dockerfile' in this folder to build",
    )


@pytest.mark.parametrize(("rel", "toml", "problem"), [
    ("envs/odd/env.toml", 'type = "nope"', "envs/odd: 'nope' is not a known env type"),
    ("agents/odd/agent.toml", 'type = "other"', "agents/odd: 'other' is not a known agent type"),
    ("artifacts/odd/artifact.toml", 'type = "nope"', "artifacts/odd: 'nope' is not a known artifact type"),
])
def test_an_unknown_entity_type_is_refused(make, rel, toml, problem):
    assert problems(make(files={rel: toml})) == (problem,)


@pytest.mark.parametrize(("task", "files", "problem"), [
    ([{"id": "eval", "type": "plugin_evaluate"}], {},
     "tasks/t.json: step 'eval': 'plugin_evaluate' is not a known step type (registered by agentenv-broken 3.0 but "
     "failed to load)"),
    ([], {"envs/odd/env.toml": 'type = "plugin_env"'},
     "envs/odd: 'plugin_env' is not a known env type (registered by agentenv-broken 3.0 but failed to load)"),
    ([], {"artifacts/odd/artifact.toml": 'type = "old_name"'},
     "artifacts/odd: 'old_name' is not a known artifact type (registered by agentenv-broken 3.0 but failed to load)"),
])
def test_an_unknown_type_says_why_when_a_plugin_failed_to_load(make, monkeypatch, task, files, problem):
    monkeypatch.setattr(resolve_module, "canonical_type", lambda name: {"old_name": "new_name"}.get(name, name))
    monkeypatch.setattr(resolve_module._registration, "failure_note", lambda group, name: (
        "" if name == "old_name" else " (registered by agentenv-broken 3.0 but failed to load)"))
    assert problem in problems(make(tasks={"t": task}, files=files))


def test_every_problem_is_reported_together(make):
    assert problems(make(tasks={
        "a": [{"id": "d", "type": "deploy_env", "env_id": "Tickets"}],
        "b": [{"id": "x", "type": "no_such_step"}],
    })) == (
        "tasks/a.json: step 'd': env_id: 'Tickets' isn't in this bundle, but 'tickets' is; names are case-sensitive",
        "tasks/b.json: step 'x': 'no_such_step' is not a known step type",
    )


def test_entities_without_references_resolve_to_a_copy_of_their_config(make):
    result = resolve_bundle(make())
    assert {(e.entry.kind, e.entry.name): e.references for e in result.entries if e.entry.kind is not BundleKind.AGENT} == {
        (BundleKind.ENV, "tickets"): (), (BundleKind.ARTIFACT, "greeting"): (),
        (BundleKind.ARTIFACT, "base-mcp"): (), (BundleKind.SKILL, "pdf"): (),
    }
    assert [image.entry.kind for image in result.built_images] == [BundleKind.AGENT]


@pytest.mark.parametrize(("depends_on", "problem"), [
    ("tickets", "depends_on is a list of step ids, not 'tickets'"),
    ([7], 'a depends_on entry is a step id or {"task_step_id": "<step id>"}, not 7'),
])
def test_a_bad_depends_on_on_a_step_deriving_its_output_is_named_once(make, depends_on, problem):
    assert problems(make(tasks={"t": [
        {"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "command_name": "c", "depends_on": depends_on},
    ]})) == (f"tasks/t.json: step 'cli': {problem}",)
