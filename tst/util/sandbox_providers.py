"""A sandbox provider for unit tests that stand one in with a MagicMock: it declares what a real provider declares, that
it is its own only link, how it runs an agent's, a server's or a gateway's images, and that its sandboxes reach a URL as
it is."""

from unittest.mock import MagicMock

from agent_env.providers.sandbox_providers.sandbox_provider import Runs


def mock_provider(runs: Runs = Runs.IN_VM, *, on_this_machine: bool = False, **attrs) -> MagicMock:
    """A MagicMock provider that runs every image as ``runs``: in a VM it creates, by default."""
    provider = MagicMock(ON_THIS_MACHINE=on_this_machine, **attrs)
    provider.links = (provider,)
    provider.runs.return_value = runs
    provider.creates_vms.return_value = runs is Runs.IN_VM
    provider.url_from_sandbox.side_effect = lambda url: url
    provider.gateway_container_options.return_value = {}
    return provider
