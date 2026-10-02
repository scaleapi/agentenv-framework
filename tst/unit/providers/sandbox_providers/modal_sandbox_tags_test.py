"""The ``pipeline_step`` sandbox tag: built by deploy steps, stamped on Modal sandboxes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.attribution import PIPELINE_STEP_KEY, RUN_ID_KEY, deploy_attribution
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider, _attribution_tags
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider


def _step(step_id="deploy_env", attribution=None):
    metadata = {"attribution": attribution} if attribution is not None else {}
    return SimpleNamespace(id=step_id, metadata=metadata)


def _context(instance_id=None, **metadata):
    return SimpleNamespace(metadata=metadata, instance_id=instance_id)


# --- deploy_attribution ---------------------------------------------------------


def test_pipeline_step_is_task_id_and_step_id():
    attribution = deploy_attribution(
        _step("deploy_env", {"team": "t"}), _context(task_id="my-pipeline"),
    )
    assert attribution == {"team": "t", PIPELINE_STEP_KEY: "my-pipeline_deploy_env"}


def test_no_task_id_means_no_pipeline_step():
    assert deploy_attribution(_step(), _context()) == {}


def test_run_id_is_the_instance_id():
    attribution = deploy_attribution(_step(), _context(instance_id="inst-1", task_id="t"))
    assert attribution == {PIPELINE_STEP_KEY: "t_deploy_env", RUN_ID_KEY: "inst-1"}


def test_no_instance_id_means_no_run_id():
    assert RUN_ID_KEY not in deploy_attribution(_step(), _context(task_id="t"))


def test_a_task_authored_run_id_wins():
    step = _step(attribution={RUN_ID_KEY: "custom"})
    assert deploy_attribution(step, _context(instance_id="inst-1"))[RUN_ID_KEY] == "custom"


def test_a_task_authored_pipeline_step_wins():
    step = _step(attribution={PIPELINE_STEP_KEY: "custom"})
    assert deploy_attribution(step, _context(task_id="t"))[PIPELINE_STEP_KEY] == "custom"


def test_deploy_attribution_does_not_mutate_the_step():
    step = _step(attribution={"team": "t"})
    deploy_attribution(step, _context(task_id="t"))
    assert step.metadata["attribution"] == {"team": "t"}


# --- _attribution_tags ----------------------------------------------------------


def test_sandbox_tags_carry_every_key():
    assert _attribution_tags({PIPELINE_STEP_KEY: "t_s", RUN_ID_KEY: "inst-1", "team": "t"}) == {
        PIPELINE_STEP_KEY: "t_s", RUN_ID_KEY: "inst-1", "team": "t",
    }


def test_sandbox_tags_sanitize_invalid_characters():
    assert _attribution_tags({PIPELINE_STEP_KEY: "my task/v2_deploy env"}) == {
        PIPELINE_STEP_KEY: "my-task-v2_deploy-env",
    }


def test_sandbox_tags_fit_modal_limit_and_keep_steps_distinct():
    long_task = "t" * 80
    a = _attribution_tags({PIPELINE_STEP_KEY: f"{long_task}_deploy_env"})[PIPELINE_STEP_KEY]
    b = _attribution_tags({PIPELINE_STEP_KEY: f"{long_task}_deploy_agent"})[PIPELINE_STEP_KEY]
    assert len(a) == len(b) == 63
    assert a != b
    assert a.startswith("t" * 54)


def test_sandbox_tags_at_the_limit_are_unchanged():
    assert _attribution_tags({PIPELINE_STEP_KEY: "x" * 63}) == {PIPELINE_STEP_KEY: "x" * 63}


# --- providers pass the tag to Modal --------------------------------------------


class _StopAfterCreate(Exception):
    pass


@pytest.fixture(autouse=True)
def _local_image_store():
    from agent_env.config import get_config
    from tst.unit.store.fakes import FakeImageStore

    get_config().set_image_store(FakeImageStore())
    yield


def _patch_clients(provider):
    provider._get_app = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    provider._get_client = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attribution", "expected"),
    [
        ({PIPELINE_STEP_KEY: "my-pipeline_deploy_env", RUN_ID_KEY: "inst-1"},
         {PIPELINE_STEP_KEY: "my-pipeline_deploy_env", RUN_ID_KEY: "inst-1"}),
        ({}, None),
    ],
)
async def test_container_create_passes_sandbox_tags(attribution, expected):
    provider = _patch_clients(ModalSandboxProvider(app_name="agent-env-test"))
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as create:
        create.aio = AsyncMock(side_effect=_StopAfterCreate())
        with pytest.raises(RuntimeError):
            await provider.create_container(
                image_name="img:latest", port=8000, env={}, attribution=attribution,
            )
    assert create.aio.call_args.kwargs["tags"] == expected


@pytest.mark.asyncio
async def test_gpu_container_create_passes_sandbox_tags():
    provider = _patch_clients(ModalSandboxProvider(app_name="agent-env-test", gpu="H100"))
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox.create") as create:
        create.aio = AsyncMock(side_effect=_StopAfterCreate())
        with pytest.raises(RuntimeError):
            await provider.create_container(
                image_name="img:latest", port=8000, env={},
                attribution={PIPELINE_STEP_KEY: "t_s"},
            )
    assert create.aio.call_args.kwargs["tags"] == {PIPELINE_STEP_KEY: "t_s"}


@pytest.mark.asyncio
async def test_vm_create_passes_sandbox_tags():
    provider = _patch_clients(ModalVmSandboxProvider(app_name="agent-env-test"))
    with patch("agent_env.providers.sandbox_providers.modal_vm_sandbox.modal.Sandbox._experimental_create") as create:
        create.aio = AsyncMock(side_effect=_StopAfterCreate())
        with pytest.raises(RuntimeError):
            await provider.create_vm(attribution={PIPELINE_STEP_KEY: "t_s"})
    assert create.aio.call_args.kwargs["tags"] == {PIPELINE_STEP_KEY: "t_s"}
