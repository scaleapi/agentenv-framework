"""Whether an agent is offered grants depends on whether they reach its sandbox, and a local store's
objects never go out as an S3-form file:// URL an agent cannot use."""

import pytest

from agent_env.a2a_agent.object_transfer import (
    choose_transfer,
    skill_add_call,
    snapshot_save_call,
    trajectory_mode,
)
from agent_env.store import LocalFilesystemObjectStore
from tst.unit.store.fakes import FakeObjectStore

SDK_TASK_GET = {"request": {"required": ["task_id"], "oneOf": [{}, {"required": ["objects"]}]}}
OBJECTS_ONLY_GET = {"request": {"required": ["task_id", "objects"]}}
SAVE_EITHER = {"request": {"required": ["context_id"], "oneOf": [{"required": ["objects"]}, {"required": ["s3_prefix"]}]}}
SAVE_S3_ONLY = {"request": {"required": ["context_id", "s3_prefix"]}}
SAVE_OBJECTS_ONLY = {"request": {"required": ["context_id", "objects"]}}
SKILL_ANY = {
    "request": {
        "required": ["name", "description"],
        "oneOf": [{"required": ["skill_bundle"]}, {"required": ["skill_s3_url"]}, {"required": ["skill_md"]}],
    }
}


@pytest.fixture
def local(tmp_path):
    return LocalFilesystemObjectStore(str(tmp_path))


@pytest.mark.parametrize("sandbox_type, expected", [("local", "objects"), ("modal", "legacy"), (None, "legacy")])
def test_a_local_store_offers_grants_only_to_local_agents(local, sandbox_type, expected):
    mode = choose_transfer(
        SDK_TASK_GET, objects=("task_id", "objects"), legacy=("task_id",), store=local, sandbox_type=sandbox_type
    )
    assert mode == expected


def test_a_remote_agent_gets_its_trajectory_inline_from_a_local_run(local):
    assert trajectory_mode(SDK_TASK_GET, local, by="task_id", sandbox_type="modal") == "legacy"
    assert trajectory_mode(OBJECTS_ONLY_GET, local, by="task_id", sandbox_type="modal") is None


def test_a_hosted_store_offers_grants_whatever_the_sandbox():
    store = FakeObjectStore()
    store.supports_transfer_grants = True
    mode = choose_transfer(
        SDK_TASK_GET, objects=("task_id", "objects"), legacy=("task_id",), store=store, sandbox_type="modal"
    )
    assert mode == "objects"


def _save(store, method, sandbox_type):
    return snapshot_save_call(
        method, store, agent_name="solver", context_id="ctx", capture_prefix=store.object_url("snap/"),
        sandbox_type=sandbox_type,
    )


def test_a_legacy_only_agent_is_refused_a_local_file_url(local):
    with pytest.raises(RuntimeError, match="Agent 'solver' cannot save a snapshot on a local object store: the agent takes only the S3 form"):
        _save(local, SAVE_S3_ONLY, "local")


def test_a_remote_agent_is_refused_a_local_file_url_and_told_why(local):
    with pytest.raises(RuntimeError, match="grants do not reach agents on the 'modal' sandbox provider"):
        _save(local, SAVE_EITHER, "modal")


def test_with_grants_off_the_refusal_names_the_setting(tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path), grants="off")
    with pytest.raises(RuntimeError, match='grants = "off"'):
        _save(store, SAVE_EITHER, "local")


def test_an_objects_only_agent_out_of_reach_is_told_why(local):
    with pytest.raises(RuntimeError, match="no snapshot save form .* do not reach agents on the 'unknown' sandbox provider"):
        _save(local, SAVE_OBJECTS_ONLY, None)


def test_an_s3_form_url_the_agent_can_use_goes_out_unchanged():
    call = _save(FakeObjectStore(), SAVE_S3_ONLY, "modal")
    assert call.mode == "legacy"
    assert not call.payload["s3_prefix"].startswith("file://")


def test_a_skill_given_inline_still_reaches_a_remote_agent(local):
    call = skill_add_call(SKILL_ANY, local, name="s", description="d", skill_md="# s", sandbox_type="modal")
    assert (call.mode, call.payload["skill_md"]) == ("legacy", "# s")


def test_a_skill_from_local_objects_is_refused_to_a_remote_agent(local):
    local.put("skills/s/SKILL.md", b"# s")
    with pytest.raises(RuntimeError, match="Agent 'solver' cannot add a skill from objects"):
        skill_add_call(
            SKILL_ANY, local, name="s", description="d", object_url=local.object_url("skills/s"),
            sandbox_type="modal", agent_name="solver",
        )
