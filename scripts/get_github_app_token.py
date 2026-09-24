#!/usr/bin/env python3
"""Mint a short-lived GitHub App installation access token.

Reads the App credentials from the environment, signs a JWT with the App's
private key, exchanges it for an installation access token, and prints the
token to stdout (and nothing else) so the caller can capture it.

Required environment variables:
  GH_APP_ID                 - Numeric App ID (from the App settings page)
  GH_APP_INSTALLATION_ID    - Installation ID for the target org/repo
  GH_APP_PRIVATE_KEY_B64    - PEM private key, base64-encoded (single line)

Usage (e.g. from CircleCI):
  GH_TOKEN=$(python scripts/get_github_app_token.py)

Tokens are valid for ~1 hour, which is plenty for the release bump job.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

import jwt  # PyJWT[crypto]


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} must be set", file=sys.stderr)
        sys.exit(1)
    return val


def main() -> None:
    app_id = _require("GH_APP_ID")
    installation_id = _require("GH_APP_INSTALLATION_ID")
    private_key_b64 = _require("GH_APP_PRIVATE_KEY_B64")

    try:
        private_key = base64.b64decode(private_key_b64).decode("utf-8")
    except Exception as e:
        print(f"ERROR: GH_APP_PRIVATE_KEY_B64 is not valid base64: {e}", file=sys.stderr)
        sys.exit(1)

    # GitHub allows up to 10 minutes; clock-skew tolerant 9-minute window.
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 9 * 60, "iss": app_id}
    app_jwt = jwt.encode(payload, private_key, algorithm="RS256")

    req = urllib.request.Request(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        method="POST",
        headers={
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "agent-env-circleci-bumper",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(
            f"ERROR: GitHub returned {e.code} when minting installation token: "
            f"{e.read().decode('utf-8', errors='replace')}",
            file=sys.stderr,
        )
        sys.exit(1)

    token = body.get("token")
    if not token:
        print(f"ERROR: response missing 'token': {body}", file=sys.stderr)
        sys.exit(1)

    # Stdout MUST contain only the token so callers can $(...) capture it.
    sys.stdout.write(token)


if __name__ == "__main__":
    main()
