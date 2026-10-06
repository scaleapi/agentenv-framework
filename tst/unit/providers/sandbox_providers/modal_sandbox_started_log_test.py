"""The per-sandbox log line joining Modal's container id to the sandbox's run and step tags."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.attribution import PIPELINE_STEP_KEY, RUN_ID_KEY
from agent_env.providers.sandbox_providers.modal_sandbox import (
    SANDBOX_STARTED_EVENT,
    ModalSandboxProvider,
    _log_sandbox_started,
)

_TAGS = {PIPELINE_STEP_KEY: "my-pipeline_deploy_env", RUN_ID_KEY: "inst-1"}


def _fake_sb(task_id="ta-01ABC"):
    sb = MagicMock()
    sb.object_id = "sb-01XYZ"
    sb._get_task_id.aio = (
        AsyncMock(side_effect=task_id) if isinstance(task_id, Exception) else AsyncMock(return_value=task_id)
    )
    sb.wait_until_ready.aio = AsyncMock()
    sb.tunnels.aio = AsyncMock(return_value={8000: MagicMock(host="ta-x.w.modal.host", port=443)})
    return sb


def _started_record(caplog):
    records = [r for r in caplog.records if getattr(r, "event", None) == SANDBOX_STARTED_EVENT]
    assert len(records) == 1
    return records[0]


@pytest.mark.asyncio
async def test_logs_the_container_id_with_the_sandbox_and_its_tags(caplog):
    with caplog.at_level(logging.INFO, logger="agent_env.providers.sandbox_providers.modal_sandbox"):
        await _log_sandbox_started(
            _fake_sb(), app_name="agent-env-p", sandbox_tags=_TAGS, cpu=2.0, memory=4096, gpu="H100",
        )
    record = _started_record(caplog)
    assert record.modal_container_id == "ta-01ABC"
    assert record.modal_sandbox_id == "sb-01XYZ"
    assert record.modal_app_name == "agent-env-p"
    assert record.modal_sandbox_tags == _TAGS
    assert getattr(record, PIPELINE_STEP_KEY) == "my-pipeline_deploy_env"
    assert getattr(record, RUN_ID_KEY) == "inst-1"
    assert (record.cpu, record.memory_mb, record.gpu) == (2.0, 4096, "H100")


@pytest.mark.asyncio
async def test_an_unreadable_container_id_still_logs_and_never_raises(caplog):
    with caplog.at_level(logging.INFO, logger="agent_env.providers.sandbox_providers.modal_sandbox"):
        await _log_sandbox_started(
            _fake_sb(RuntimeError("private API moved")), app_name="a", sandbox_tags={},
            cpu=1.0, memory=1024, gpu=None,
        )
    assert _started_record(caplog).modal_container_id is None


@pytest.mark.asyncio
async def test_a_tag_named_like_a_log_record_field_does_not_clobber_it(caplog):
    tags = {"name": "n", "module": "m", RUN_ID_KEY: "inst-1"}
    with caplog.at_level(logging.INFO, logger="agent_env.providers.sandbox_providers.modal_sandbox"):
        await _log_sandbox_started(_fake_sb(), app_name="a", sandbox_tags=tags, cpu=1.0, memory=1024, gpu=None)
    record = _started_record(caplog)
    assert record.modal_sandbox_tags == tags
    assert record.name == "agent_env.providers.sandbox_providers.modal_sandbox"
    assert getattr(record, RUN_ID_KEY) == "inst-1"


@pytest.fixture(autouse=True)
def _local_image_store():
    from agent_env.config import get_config
    from tst.unit.store.fakes import FakeImageStore

    get_config().set_image_store(FakeImageStore())
    yield


@pytest.mark.asyncio
async def test_create_container_emits_the_started_line(caplog):
    provider = ModalSandboxProvider(app_name="agent-env-test")
    provider._get_app = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    provider._get_client = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as create, \
            caplog.at_level(logging.INFO, logger="agent_env.providers.sandbox_providers.modal_sandbox"):
        create.aio = AsyncMock(return_value=_fake_sb())
        await provider.create_container(image_name="img:latest", port=8000, env={}, attribution=_TAGS)
    record = _started_record(caplog)
    assert record.modal_container_id == "ta-01ABC"
    assert getattr(record, RUN_ID_KEY) == "inst-1"
