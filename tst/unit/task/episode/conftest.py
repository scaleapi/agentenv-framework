"""An episode's turns reach a proxy agent this process serves on loopback, so these tests allow loopback
connections and nothing else."""

import pytest
from pytest_socket import enable_socket, socket_allow_hosts


@pytest.fixture(autouse=True)
def _disable_network():
    socket_allow_hosts(["127.0.0.1"], allow_unix_socket=True)
    yield
    enable_socket()
