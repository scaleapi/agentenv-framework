"""Unit tests for the ECR image-store credential provider."""

import base64
from unittest.mock import patch

import pytest

from agent_env.store import EcrCredentials


def test_ecr_mints_a_fresh_authorization_token_each_time():
    class _Client:
        def __init__(self):
            self.passwords = iter(("one", "two"))

        def get_authorization_token(self):
            token = base64.b64encode(f"AWS:{next(self.passwords)}".encode()).decode()
            return {"authorizationData": [{"authorizationToken": token}]}

    credentials = EcrCredentials(client=_Client())

    assert credentials.mint("123.dkr.ecr.us-west-2.amazonaws.com").password == "one"
    assert credentials.mint("123.dkr.ecr.us-west-2.amazonaws.com").password == "two"


def test_ecr_from_config_requires_explicit_aws_credentials():
    with pytest.raises(ValueError, match="must be configured"):
        EcrCredentials.from_config()

    with pytest.raises(ValueError, match="must be configured"):
        EcrCredentials.from_config(
            region="us-west-2", access_key="", secret_key="secret"
        )


def test_ecr_from_config_builds_the_keyed_client():
    with patch(
        "agent_env.store.image_store.ecr_image_store.boto3.client"
    ) as boto_client:
        credentials = EcrCredentials.from_config(
            region="us-west-2",
            access_key="access",
            secret_key="secret",
        )

        assert credentials.client is boto_client.return_value

    boto_client.assert_called_once_with(
        "ecr",
        region_name="us-west-2",
        aws_access_key_id="access",
        aws_secret_access_key="secret",
    )
