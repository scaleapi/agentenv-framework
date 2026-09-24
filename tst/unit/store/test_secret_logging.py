"""A secret fetch must not put the secret in the logs.

botocore logs full HTTP bodies at DEBUG. For GetSecretValue the body *is* the
bundle, so any process that turns DEBUG on — someone debugging an unrelated
timeout, a verbose CI job — writes every key in the account to stdout in
plaintext. This test is the guard, and it failed before the fix.
"""
from __future__ import annotations

import io
import logging
from unittest.mock import MagicMock, patch

from agent_env.store.secret_store.aws_secrets_manager_secret_store import (
    AwsSecretsManagerSecretStore,
    suppress_aws_body_logging,
)

_SECRET_BODY = '{"mongodb_uri": "mongodb://admin:hunter2@example", "token": "ghp_xxxx"}'


def _fetch_under_debug_logging() -> str:
    """Fetch a secret with DEBUG on, returning everything that got logged."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    root = logging.getLogger()
    prior_level, prior_handlers = root.level, root.handlers[:]
    root.handlers = [handler]
    root.setLevel(logging.DEBUG)
    try:
        client = MagicMock()
        client.get_secret_value.return_value = {"SecretString": _SECRET_BODY}

        # Stand in for what botocore does at DEBUG: log the response body.
        def _logging_get_secret_value(**kwargs):
            logging.getLogger("botocore.parsers").debug("Response body:\n%s", _SECRET_BODY)
            return {"SecretString": _SECRET_BODY}

        client.get_secret_value.side_effect = _logging_get_secret_value
        with patch("boto3.client", return_value=client):
            store = AwsSecretsManagerSecretStore("some/secret", "us-west-2")
            store.get("mongodb_uri")
        return buf.getvalue()
    finally:
        root.handlers = prior_handlers
        root.setLevel(prior_level)


def test_secret_body_is_not_logged_at_debug():
    logged = _fetch_under_debug_logging()
    for marker in ("mongodb_uri", "hunter2", "ghp_xxxx"):
        assert marker not in logged, f"the secret leaked into the logs via {marker!r}"


def test_botocore_log_level_is_restored():
    """Silencing is scoped to the call: it must not deafen the caller afterwards."""
    parsers = logging.getLogger("botocore.parsers")
    parsers.setLevel(logging.DEBUG)
    try:
        with suppress_aws_body_logging():
            assert parsers.level == logging.INFO
        assert parsers.level == logging.DEBUG
    finally:
        parsers.setLevel(logging.NOTSET)


def test_level_restored_even_when_the_call_raises():
    parsers = logging.getLogger("botocore.parsers")
    parsers.setLevel(logging.DEBUG)
    try:
        try:
            with suppress_aws_body_logging():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert parsers.level == logging.DEBUG
    finally:
        parsers.setLevel(logging.NOTSET)


def test_an_overlapping_fetch_stays_silenced_when_the_first_finishes():
    """The race Greptile flagged, forced rather than raced for.

    Two fetches overlap and the one that started FIRST finishes first. It
    captured DEBUG on the way in, so a naive restore puts DEBUG back while the
    second fetch is still in flight — and that fetch's SecretString goes to
    whatever handlers are configured.

    Entered by hand instead of with threads because the bug is about ordering,
    not timing: a thread test either reproduces it or does not depending on the
    scheduler, which makes it useless as a regression guard.
    """
    parsers = logging.getLogger("botocore.parsers")
    parsers.setLevel(logging.DEBUG)
    first = suppress_aws_body_logging()
    second = suppress_aws_body_logging()
    try:
        first.__enter__()                      # captures DEBUG, raises to INFO
        second.__enter__()                     # captures INFO, already raised
        first.__exit__(None, None, None)       # first fetch done, second is not

        assert parsers.level == logging.INFO, (
            "logging fell back to DEBUG while a fetch was still in flight — "
            "that fetch's secret would reach the handlers"
        )

        second.__exit__(None, None, None)
        assert parsers.level == logging.DEBUG, (
            "the caller's original level was not restored once all fetches ended"
        )
    finally:
        parsers.setLevel(logging.NOTSET)
