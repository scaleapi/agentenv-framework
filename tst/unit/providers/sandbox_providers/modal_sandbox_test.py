"""Unit tests for modal_sandbox container-create error handling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import modal
import pytest

from agent_env.config import get_config
from agent_env.providers.sandbox_providers import modal_sandbox
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox, ModalSandboxProvider
from agent_env.store import set_object_store
from tst.unit.store.fakes import ConfiguredObjectStore


class _BareSandboxTimeoutError(Exception):
    """Stand-in for modal.exception.SandboxTimeoutError — empty-args."""


def test_provider_default_pull_secret_is_none():
    assert ModalSandboxProvider()._ecr_pull_secret_name is None


@pytest.mark.asyncio
async def test_apps_are_looked_up_in_the_configured_environment():
    provider = ModalSandboxProvider(environment_name="rl-prod")
    provider._get_client = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.App.lookup") as lookup:
        lookup.aio = AsyncMock(return_value=MagicMock())
        await provider._get_app("agent-env-p")
    assert lookup.aio.call_args.kwargs["environment_name"] == "rl-prod"


@pytest.fixture
def _patched_provider():
    provider = ModalSandboxProvider(app_name="agent-env-test")
    provider._get_app = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    provider._get_client = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    return provider


@pytest.fixture(autouse=True)
def _local_image_store():
    """Route image-store auth through an in-memory fake so create_container never reaches ECR."""
    from agent_env.config import get_config
    from tst.unit.store.fakes import FakeImageStore

    get_config().set_image_store(FakeImageStore())
    yield


@pytest.mark.asyncio
async def test_create_container_wraps_modal_create_failure_with_context(_patched_provider):
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(side_effect=_BareSandboxTimeoutError())

        with pytest.raises(RuntimeError) as ei:
            await _patched_provider.create_container(
                image_name="123456789012.dkr.ecr.us-west-2.amazonaws.com/example/mcp-server:v1",
                port=18765,
                env={},
                cpu=2.0,
                memory=4096,
                private_network=True,
            )

    msg = str(ei.value)
    assert "Modal sandbox create failed" in msg
    assert "_BareSandboxTimeoutError" in msg
    assert "example/mcp-server:v1" in msg
    assert "port=18765" in msg
    assert "region=us-east-1" in msg
    assert "i6pn=True" in msg
    assert isinstance(ei.value.__cause__, _BareSandboxTimeoutError)


@pytest.mark.asyncio
async def test_create_container_wraps_post_create_failure_and_terminates(_patched_provider):
    fake_sb = MagicMock()
    fake_sb.object_id = "sb-FAKEFAKEFAKEFAKEFAKE"
    fake_sb.tunnels = MagicMock()
    fake_sb.tunnels.aio = AsyncMock(side_effect=_BareSandboxTimeoutError())
    fake_sb.terminate = MagicMock()
    fake_sb.terminate.aio = AsyncMock()

    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(return_value=fake_sb)

        with pytest.raises(RuntimeError) as ei:
            await _patched_provider.create_container(
                image_name="registry.example/mcp-server-example:v1",
                port=18765,
                env={},
            )

    msg = str(ei.value)
    assert "Modal sandbox post-create failed" in msg
    assert "sb_id=sb-FAKEFAKEFAKEFAKEFAKE" in msg
    assert "mcp-server-example:v1" in msg
    fake_sb.terminate.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_container_terminates_a_sandbox_whose_create_is_cancelled(_patched_provider):
    """A chain's deadline cancels a create still waiting on its sandbox, which would otherwise run until its timeout."""
    fake_sb = MagicMock()
    fake_sb.object_id = "sb-FAKEFAKEFAKEFAKEFAKE"
    fake_sb.wait_until_ready.aio = AsyncMock(side_effect=asyncio.CancelledError)
    fake_sb.terminate.aio = AsyncMock()

    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(return_value=fake_sb)

        with pytest.raises(asyncio.CancelledError):
            await _patched_provider.create_container(image_name="registry.example/mcp-server-example:v1", port=18765, env={})

    fake_sb.terminate.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_container_terminates_a_sandbox_modal_creates_after_its_create_is_cancelled(_patched_provider):
    """Cancelled while Modal is still creating the sandbox, as a deploy cancelled mid-batch is: Modal creates it all the
    same, and it would run until its timeout, so the create waits for it and terminates it."""
    fake_sb = MagicMock()
    fake_sb.terminate.aio = AsyncMock()
    asked, release = asyncio.Event(), asyncio.Event()

    async def experimental_create(*args, **kwargs):
        asked.set()
        await release.wait()
        return fake_sb

    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(side_effect=experimental_create)
        create = asyncio.create_task(
            _patched_provider.create_container(image_name="registry.example/mcp-server-example:v1", port=18765, env={}))
        await asyncio.wait_for(asked.wait(), timeout=5)
        create.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await create

    fake_sb.terminate.aio.assert_awaited_once()
    fake_sb.wait_until_ready.aio.assert_not_called()


@pytest.mark.asyncio
async def test_a_sandbox_a_cancelled_create_cant_terminate_is_named_in_a_warning(_patched_provider, caplog):
    """It runs until its timeout, so whoever reads the log can find it."""
    fake_sb = MagicMock(object_id="sb-STILLRUNNING")
    fake_sb.terminate.aio = AsyncMock(side_effect=RuntimeError("Modal is unavailable"))
    asked, release = asyncio.Event(), asyncio.Event()

    async def experimental_create(*args, **kwargs):
        asked.set()
        await release.wait()
        return fake_sb

    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(side_effect=experimental_create)
        create = asyncio.create_task(
            _patched_provider.create_container(image_name="registry.example/mcp-server-example:v1", port=18765, env={}))
        await asyncio.wait_for(asked.wait(), timeout=5)
        create.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await create

    assert any("sb-STILLRUNNING" in r.getMessage() and "Modal is unavailable" in r.getMessage()
               for r in caplog.records if r.levelname == "WARNING")


async def _cancel_a_create_that_hangs(provider, *, cancels: int) -> tuple[MagicMock, asyncio.Event]:
    """Start a create whose request to Modal hangs until the returned event is set, cancel it ``cancels`` times, and
    check it raises CancelledError without the request having ended."""
    fake_sb = MagicMock(object_id="sb-LATE")
    fake_sb.terminate.aio = AsyncMock()
    asked, release = asyncio.Event(), asyncio.Event()

    async def experimental_create(*args, **kwargs):
        asked.set()
        await release.wait()
        return fake_sb

    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(side_effect=experimental_create)
        create = asyncio.create_task(
            provider.create_container(image_name="registry.example/mcp-server-example:v1", port=18765, env={}))
        await asyncio.wait_for(asked.wait(), timeout=5)
        for _ in range(cancels):
            create.cancel()
            await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(create, timeout=5)
    fake_sb.terminate.aio.assert_not_called()
    return fake_sb, release


@pytest.mark.asyncio
@pytest.mark.parametrize("cancels, wait_seconds", [(1, 0.05), (2, 30)], ids=["outlasts-the-wait", "cancelled-again"])
async def test_a_cancelled_create_stops_waiting_on_a_request_that_hangs_and_terminates_what_it_creates_later(
        _patched_provider, monkeypatch, caplog, cancels, wait_seconds):
    """So a hung request can't hold up a chain's deadline or a Ctrl-C, nor leave a sandbox behind when it ends."""
    monkeypatch.setattr(modal_sandbox, "_CANCELLED_CREATE_WAIT_SECONDS", wait_seconds)

    fake_sb, release = await _cancel_a_create_that_hangs(_patched_provider, cancels=cancels)
    release.set()
    for _ in range(50):
        if fake_sb.terminate.aio.await_count:
            break
        await asyncio.sleep(0.01)

    fake_sb.terminate.aio.assert_awaited_once()
    assert any("still waiting on Modal" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")


@pytest.mark.asyncio
async def test_a_sandbox_created_just_as_a_cancelled_creates_wait_gives_up_is_terminated(_patched_provider, monkeypatch):
    """The wait can give up in the same step Modal answers, which leaves the create done but its sandbox running."""
    fake_sb = MagicMock(object_id="sb-JUSTINTIME")
    fake_sb.terminate.aio = AsyncMock()
    asked, release = asyncio.Event(), asyncio.Event()

    async def experimental_create(*args, **kwargs):
        asked.set()
        await release.wait()
        return fake_sb

    async def gives_up_as_the_create_ends(awaitable, timeout):
        await awaitable
        raise TimeoutError

    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(side_effect=experimental_create)
        create = asyncio.create_task(
            _patched_provider.create_container(image_name="registry.example/mcp-server-example:v1", port=18765, env={}))
        await asked.wait()
        monkeypatch.setattr(modal_sandbox.asyncio, "wait_for", gives_up_as_the_create_ends)
        create.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await create
    for _ in range(50):
        if fake_sb.terminate.aio.await_count:
            break
        await asyncio.sleep(0.01)

    fake_sb.terminate.aio.assert_awaited_once()


def test_a_process_that_exits_while_a_cancelled_create_hangs_names_the_create_in_a_warning(
        _patched_provider, monkeypatch, caplog):
    """Exiting cancels the create, so what Modal makes for it runs until its timeout; the warning lets it be found."""
    monkeypatch.setattr(modal_sandbox, "_CANCELLED_CREATE_WAIT_SECONDS", 0.05)

    asyncio.run(_cancel_a_create_that_hangs(_patched_provider, cancels=1))

    assert any("stopped before Modal answered" in r.getMessage() and "mcp-server-example:v1" in r.getMessage()
               for r in caplog.records if r.levelname == "WARNING")


@pytest.mark.asyncio
async def test_create_container_includes_cause_message_in_wrapped_text(_patched_provider):
    class _RemoteErrorWithMessage(Exception):
        pass

    inner_msg = "Image build for im-Xhe66qHlOEXQpwDyiiBuGX failed. See build logs."
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(side_effect=_RemoteErrorWithMessage(inner_msg))

        with pytest.raises(RuntimeError) as ei:
            await _patched_provider.create_container(
                image_name="registry.example/some-image:v1",
                port=18765,
                env={},
            )

    msg = str(ei.value)
    assert "_RemoteErrorWithMessage" in msg
    assert inner_msg in msg


@pytest.mark.asyncio
async def test_create_container_handles_empty_args_exception(_patched_provider):
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(side_effect=_BareSandboxTimeoutError())

        with pytest.raises(RuntimeError) as ei:
            await _patched_provider.create_container(
                image_name="registry.example/some-image:v1",
                port=18765,
                env={},
            )

    msg = str(ei.value)
    assert "_BareSandboxTimeoutError]" in msg
    assert "_BareSandboxTimeoutError: ]" not in msg


@pytest.mark.asyncio
async def test_image_auth_paths(_patched_provider):
    """ECR -> from_aws_ecr + named secret; store-supplied creds -> docker login; else anonymous."""
    from agent_env.config import get_config
    from agent_env.store import (
        EcrCredentials,
        OciRegistryImageStore,
        RegistryAuth,
        OciRegistryCredentials,
    )

    class _Credentials(OciRegistryCredentials):
        def mint(self, host):
            return RegistryAuth(host, "u", "p")

    ecr_host = "123456789012.dkr.ecr.us-west-2.amazonaws.com"
    ecr = "123456789012.dkr.ecr.us-west-2.amazonaws.com/example/x:v1"
    _patched_provider._ecr_pull_secret_name = "agent-env-ecr-reader"
    _patched_provider._environment_name = "rl-prod"
    p = lambda n: patch(f"agent_env.providers.sandbox_providers.modal_sandbox.modal.{n}")  # noqa: E731

    with p("Image.from_aws_ecr") as aws_ecr, p("Image.from_registry") as registry, \
         p("Secret.from_name") as from_name, p("Sandbox._experimental_create") as create:
        create.aio = AsyncMock(side_effect=_BareSandboxTimeoutError())
        cases = (
            (
                OciRegistryImageStore(
                    ecr_host,
                    credentials=EcrCredentials(client=MagicMock()),
                ),
                ecr,
            ),
            (OciRegistryImageStore("priv.example", credentials=_Credentials()), "priv.example/x:v1"),
            (OciRegistryImageStore("pub.example"), "pub.example/x:v1"),
            (
                OciRegistryImageStore(
                    ecr_host,
                    credentials=EcrCredentials(client=MagicMock()),
                ),
                f"{ecr_host}.attacker.example/x:v1",
            ),
        )
        for store, ref in cases:
            get_config().set_image_store(store)
            with pytest.raises(RuntimeError):
                await _patched_provider.create_container(image_name=ref, port=1, env={})

    assert aws_ecr.call_args.args[0] == ecr
    from_name.assert_called_once_with(
        "agent-env-ecr-reader",
        required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
        environment_name="rl-prod",
    )
    assert registry.call_args_list[0].kwargs["secret"] is not None
    assert registry.call_args_list[1].args == ("pub.example/x:v1",)
    assert registry.call_args_list[2].args == (f"{ecr_host}.attacker.example/x:v1",)


@pytest.mark.asyncio
async def test_unset_pull_secret_falls_back_to_minted_credentials(_patched_provider):
    """The named workspace secret is a cache opt-in: without it, ECR images pull via
    a freshly minted token (uncached) instead of erroring."""
    import base64

    from agent_env.config import get_config
    from agent_env.store import EcrCredentials, OciRegistryImageStore

    _patched_provider._ecr_pull_secret_name = None
    ecr_client = MagicMock()
    ecr_client.get_authorization_token.return_value = {
        "authorizationData": [{"authorizationToken": base64.b64encode(b"AWS:tok").decode()}]
    }
    get_config().set_image_store(
        OciRegistryImageStore(
            "123.dkr.ecr.us-west-2.amazonaws.com",
            credentials=EcrCredentials(client=ecr_client),
        )
    )

    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Image.from_aws_ecr") as aws_ecr, \
         patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Image.from_registry") as registry, \
         patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as create:
        create.aio = AsyncMock(side_effect=_BareSandboxTimeoutError())
        with pytest.raises(RuntimeError):
            await _patched_provider.create_container(
                image_name="123.dkr.ecr.us-west-2.amazonaws.com/team/image:v1",
                port=1,
                env={},
            )

    aws_ecr.assert_not_called()
    assert registry.call_args.kwargs["secret"] is not None


@pytest.mark.parametrize(
    "vnc_port, expected_ports, expected_vnc_url",
    [
        (6080, [5000, 6080], "https://ta-x-6080-y.w.modal.host/vnc.html"),
        (None, [5000], None),
    ],
)
@pytest.mark.asyncio
async def test_vnc_port_controls_the_tunnel_set_and_the_vnc_url(
    _patched_provider, vnc_port, expected_ports, expected_vnc_url
):
    fake_sb = MagicMock()
    fake_sb.object_id = "sb-FAKEFAKEFAKEFAKEFAKE"
    fake_sb.wait_until_ready = MagicMock()
    fake_sb.wait_until_ready.aio = AsyncMock()
    fake_sb.tunnels = MagicMock()
    fake_sb.tunnels.aio = AsyncMock(return_value={
        p: MagicMock(host=f"ta-x-{p}-y.w.modal.host", port=443) for p in (5000, 6080)
    })

    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as mock_create:
        mock_create.aio = AsyncMock(return_value=fake_sb)
        sandbox = await _patched_provider.create_container(
            image_name="registry.example/vnc-image:v1", port=5000, env={}, vnc_port=vnc_port,
        )

    assert mock_create.aio.call_args.kwargs["encrypted_ports"] == expected_ports
    assert sandbox.vnc_url == expected_vnc_url


@pytest.mark.asyncio
async def test_write_file_from_object_reads_a_url_from_the_store_that_holds_it(cli_routing):
    set_object_store(ConfiguredObjectStore())
    url = get_config().get_object_store_for("@local/~/t").put("objects/a.json", b"LOCAL")
    sandbox = ModalSandbox(MagicMock(), tunnel_urls={})
    written = []

    async def collect(chunks, destination_path):
        written.append((b"".join([chunk async for chunk in chunks]), destination_path))

    sandbox._write_stream_via_exec = collect
    await sandbox.write_file_from_object(url, "/data/a.json")
    assert written == [(b"LOCAL", "/data/a.json")]


@pytest.mark.asyncio
async def test_terminate_sandbox_terminates_by_id_without_reading_its_tunnels(_patched_provider):
    sb = MagicMock()
    sb.tunnels.aio = AsyncMock(side_effect=modal.exception.ConflictError("Sandbox has already finished with status terminated"))
    sb.terminate.aio = AsyncMock()
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox.from_id") as from_id:
        from_id.aio = AsyncMock(return_value=sb)
        await _patched_provider.terminate_sandbox("sb-finished")

    sb.terminate.aio.assert_awaited_once()
    sb.tunnels.aio.assert_not_called()
    sb.exec.aio.assert_not_called()


@pytest.mark.asyncio
async def test_terminate_sandbox_counts_a_sandbox_modal_no_longer_knows_as_gone(_patched_provider):
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox.from_id") as from_id:
        from_id.aio = AsyncMock(side_effect=modal.exception.NotFoundError("Sandbox not found"))
        await _patched_provider.terminate_sandbox("sb-gone")


@pytest.mark.asyncio
async def test_get_sandbox_still_raises_for_a_finished_sandbox_so_a_reattach_sees_it_dead(_patched_provider):
    sb = MagicMock()
    sb.tunnels.aio = AsyncMock(side_effect=modal.exception.ConflictError("Sandbox has already finished with status terminated"))
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox.from_id") as from_id:
        from_id.aio = AsyncMock(return_value=sb)
        with pytest.raises(modal.exception.ConflictError):
            await _patched_provider.get_sandbox("sb-finished")
