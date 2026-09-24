"""Disable real network access for the unit test suite.

Unit tests must not touch the network: anything that does is either missing a
mock or belongs in tst/integration. This autouse fixture (scoped to tst/unit
via this conftest's location) blocks IP sockets through pytest-socket. AF_UNIX
sockets are allowed (asyncio's self-pipe / local IPC), and subprocesses run in
their own interpreter so they are unaffected (e.g. generated CLI scripts).

A test that genuinely needs a socket can opt out with @pytest.mark.enable_socket.

Unit tests that touch AWS run against moto, which still makes botocore resolve credentials;
without any, botocore probes the instance-metadata endpoint and trips the socket guard. So
placeholder credentials are set for the session when the environment has none.
"""

import os

import pytest
from pytest_socket import disable_socket, enable_socket

_PLACEHOLDER_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_DEFAULT_REGION": "us-west-2",
    "AWS_EC2_METADATA_DISABLED": "true",
}


@pytest.fixture(scope="session", autouse=True)
def _placeholder_aws_credentials():
    if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
        for name, value in _PLACEHOLDER_AWS_ENV.items():
            os.environ.setdefault(name, value)


@pytest.fixture(autouse=True)
def _disable_network():
    disable_socket(allow_unix_socket=True)
    yield
    enable_socket()
