"""The AWS backends' tests, which need the ``aws`` extra; without it the directory is skipped.

They run against moto or stubbed clients, which still make botocore resolve credentials. Without
any, botocore probes the instance-metadata endpoint and trips the socket guard, so each test gets
placeholder credentials when the environment has none.
"""

import os

import pytest

pytest.importorskip("boto3")

_PLACEHOLDER_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_DEFAULT_REGION": "us-west-2",
    "AWS_EC2_METADATA_DISABLED": "true",
}


@pytest.fixture(autouse=True)
def _placeholder_aws_credentials(monkeypatch):
    if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
        for name, value in _PLACEHOLDER_AWS_ENV.items():
            if name not in os.environ:
                monkeypatch.setenv(name, value)
