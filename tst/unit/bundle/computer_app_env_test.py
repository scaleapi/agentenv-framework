"""Computer app envs written from a bundle's env folders: the app's files stored with each version, its class read where
its own dependencies aren't installed, what it declares, refusals before any write, and a deploy refused until it names
a computer and a driver."""

import json

import pytest

from agent_env.bundle import BundleError
from agent_env.bundle.materialize import materialize
from agent_env.env import Env
from agent_env.env.envs.computer.app import ComputerAppEnv
from agent_env.providers.env_providers.env_provider import DeployRefused
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from tst.unit.bundle._support import layout, plan_of
from tst.unit.bundle.computer_driver_env_test import DRIVER, LINUX

ROOT = "@local/~/apps"
APP = '''import not_installed_here
from pydantic import BaseModel, SecretStr

from agentenv_protocol import add_data, get_data
from agentenv_protocol.computer import ComputerAppEnvironment, Launch, Linux, Service, computer_app, install_step


@computer_app(supports={supports}, permissions=["camera"])
class Mail(ComputerAppEnvironment):
    linux = Linux(python="requirements.txt", services=[Service(name="imap", command="venv/bin/python imap.py")])
    launch = Launch(command=["thunderbird"], window={{"class": "thunderbird"}})
    skill = {skill!r}

    class Settings(BaseModel):
        signature: str = ""
        password: SecretStr = SecretStr("")

    @install_step(scope="session")
    def profile(self): not_installed_here.profile()

    @add_data
    async def load(self, parts): pass

    @get_data
    async def export(self): return []
'''
TOML = 'type = "computer_app"\nmodule = "app.py:Mail"\n'


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "apps"


def _bundle(root, toml=TOML, supports=LINUX, skill="skill.md", **extra):
    files = {"envs/mail/env.toml": toml, "envs/mail/app.py": APP.format(supports=supports, skill=skill),
             "envs/mail/requirements.txt": "agentenv-framework-protocol\n", "envs/mail/skill.md": "# Mail\n"}
    return layout(root, {**files, **extra, "tasks/t.json": json.dumps([{"id": "agent", "type": "deploy_agent",
                                                                        "env_ids": ["mail"]}])})


def _problems(root):
    with pytest.raises(BundleError) as caught:
        materialize(plan_of(root))
    return " ".join(caught.value.problems)


@pytest.mark.asyncio
async def test_an_app_folder_is_written_with_its_files_and_what_its_class_declares(bundle_dir):
    _bundle(bundle_dir)

    materialize(plan_of(bundle_dir))
    env = Env.get(f"{ROOT}/mail")

    assert isinstance(env, ComputerAppEnv) and (env.environment_name, env.module) == ("mail", "app.py:Mail")
    assert sorted(env.files.file_artifact_ids) == ["app.py", "requirements.txt", "skill.md"]
    declared = env.declaration
    assert (declared["supports"], declared["permissions"]) == (LINUX, ["camera"])
    assert declared["install"] == {"linux": {"files": ".", "python": "requirements.txt",
                                             "services": [{"name": "imap", "command": "venv/bin/python imap.py"}]}}
    assert declared["launch"] == {"at": "boot", "command": ["thunderbird"], "window": {"class": "thunderbird"}}
    assert declared["settings"]["properties"]["password"]["writeOnly"] is True
    assert (declared["skill"], declared["data"]) == ({"file": "skill.md"}, ["data/add", "data/get"])
    assert declared["install_steps"] == [{"name": "profile", "scope": "session"}]
    assert (env.computer, env.driver) == (None, None)
    assert ComputerAppEnv.from_dict(env.to_dict()).to_dict() == env.to_dict()
    (refusal,) = DeployEnvTaskStep(id="env", version=None, env_id=env.id).preflight()
    assert "names no computer and driver to deploy alone on; name them in its env.toml" in refusal
    with pytest.raises(DeployRefused, match="names no computer and driver to deploy alone on"):
        await env.deploy()


@pytest.mark.parametrize("toml, extra, skill, problem", [
    ('type = "computer_app"\n', {}, "skill.md", "module is required"),
    ('type = "computer_app"\nmodule = "app.py:Desk"\n', {"envs/mail/app.py": DRIVER.format(supports=LINUX,
                                                                                          buttons=("left",))},
     "skill.md", "app.py:Desk is a class a computer_driver env declares, not a computer_app env"),
    (TOML, {"envs/mail/requirements.txt": None}, "skill.md",
     "install.linux.python: 'requirements.txt' isn't in this folder"),
    (TOML, {}, "guide.md", "skill: 'guide.md' isn't in this folder"),
    (f'{TOML}image = "x"\n', {}, "skill.md", "unknown key 'image'"),
], ids=["no-module", "a-driver", "install-python", "skill", "unknown-key"])
def test_an_app_folder_that_cant_be_read_is_refused_before_any_write(bundle_dir, toml, extra, skill, problem):
    _bundle(bundle_dir, toml=toml, skill=skill, **{path: data for path, data in extra.items() if data is not None})
    for path in (path for path, data in extra.items() if data is None):
        (bundle_dir / path).unlink()

    assert problem in _problems(bundle_dir)
    with pytest.raises(Exception):
        Env.get(f"{ROOT}/mail")
