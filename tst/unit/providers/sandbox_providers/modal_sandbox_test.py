"""Unit tests for modal_sandbox container-create error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider


class _BareSandboxTimeoutError(Exception):
    """Stand-in for modal.exception.SandboxTimeoutError — empty-args."""


def test_provider_default_pull_secret_is_none():
    assert ModalSandboxProvider()._ecr_pull_secret_name is None


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
                region="us-east-1",
                i6pn=True,
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
