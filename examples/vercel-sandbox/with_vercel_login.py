"""Run a command with project OIDC from the existing Vercel CLI login."""

import argparse
import json
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
parser.add_argument("--scope", required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    parser.error("a command after -- is required")
result = subprocess.run(
    ["vercel", "project", "token", args.project, "--scope", args.scope, "--json"],
    capture_output=True, text=True,
)
if result.returncode:
    raise SystemExit("Vercel project authentication failed. Check your CLI login and project access.")
token = json.loads(result.stdout)["token"]
if not isinstance(token, str) or not token:
    raise SystemExit("The Vercel CLI did not return a token")
environment = dict(os.environ, VERCEL_OIDC_TOKEN=token)
print("Using project OIDC from your Vercel CLI login.", flush=True)
raise SystemExit(subprocess.run(command, env=environment).returncode)
