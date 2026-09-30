#!/usr/bin/env bash
# Bump the release versions (both pyproject files and uv.lock) + push tag,
# called from CircleCI's `bump` job.
#
# Auth: uses a GitHub App installation token ($GH_TOKEN) minted earlier in
# the job by scripts/get_github_app_token.py. The App must be installed on
# the repo with Contents: read & write, and listed in the bypass-actors of
# the ruleset / branch-protection rule covering main.

set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN must be set (minted by get_github_app_token.py)}"
: "${GH_APP_BOT_NAME:=agent-env-bot}"
: "${GH_APP_BOT_EMAIL:=agent-env-bot@users.noreply.github.com}"
: "${CIRCLE_PROJECT_USERNAME:?CIRCLE_PROJECT_USERNAME must be set}"
: "${CIRCLE_PROJECT_REPONAME:?CIRCLE_PROJECT_REPONAME must be set}"

REMOTE="https://x-access-token@github.com/${CIRCLE_PROJECT_USERNAME}/${CIRCLE_PROJECT_REPONAME}.git"
MAIN_REF="refs/remotes/release/main"
MAX_ATTEMPTS=3

git config user.name  "${GH_APP_BOT_NAME}"
git config user.email "${GH_APP_BOT_EMAIL}"
# The token reaches git through a credential helper that reads it from the environment at
# push time, so it is never part of a URL that git or this script could print. The user name
# stays in the URL so CircleCI checkout's global https-to-ssh insteadOf rewrite does not match.
git config credential.helper '!f() { echo username=x-access-token; echo "password=${GH_TOKEN}"; }; f'

pip install --quiet tomlkit

is_protection_error() {
  echo "$1" | grep -qiE 'protected branch|requires a pull request|refusing to allow|status check|GH00[0-9]+|GH01[0-9]+'
}

# Drop any locally-created v*.*.* tags so bump_version.py can't be
# misled by a tag from a prior failed attempt that never reached remote.
drop_local_release_tags() {
  local stale tag
  stale=$(git tag -l 'v*.*.*') || return 1
  if [[ -n "$stale" ]]; then
    while IFS= read -r tag; do
      git tag -d "$tag" >/dev/null || return 1
    done <<< "$stale"
  fi
}

attempt_bump_and_push() {
  local push_output_file=$1
  local new_version

  drop_local_release_tags || return 1

  git fetch "$REMOTE" "+refs/heads/main:${MAIN_REF}" >&2 || return 1
  git fetch "$REMOTE" "+refs/tags/v*:refs/tags/v*" >&2 || return 1
  git reset --hard "$MAIN_REF" >&2 || return 1

  new_version=$(python scripts/bump_version.py) || return 1
  if [[ -z "$new_version" ]]; then
    echo "ERROR: bump_version.py produced empty output" >&2
    return 1
  fi
  echo "computed next version: v${new_version}" >&2

  git add pyproject.toml packages/agentenv-protocol/pyproject.toml uv.lock >&2 || return 1
  git commit -m "bump version to v${new_version}" >&2 || return 1
  git tag -a "v${new_version}" -m "v${new_version}" >&2 || return 1

  # Push branch first. If this fails the tag never reaches the remote,
  # so a retry is safe: the next iteration drops the local tag and
  # recomputes the version against the latest remote tags.
  if ! git push "$REMOTE" "HEAD:refs/heads/main" >"$push_output_file" 2>&1; then
    return 1
  fi
  cat "$push_output_file"

  # Branch push succeeded. We MUST push the tag now, or the release is
  # broken: the bump commit is on main but there's no trigger tag, and
  # bump_version.py will never regenerate this exact version on a
  # retry. Bail loudly with the manual-recovery command.
  if ! git push "$REMOTE" "refs/tags/v${new_version}" 2>&1; then
    {
      echo "ERROR: Branch push for v${new_version} succeeded but tag push failed."
      echo "       The bump commit is already on main; bump_version.py will not"
      echo "       regenerate v${new_version}. To fire the CircleCI publish run:"
      echo "         git push origin refs/tags/v${new_version}"
    } >&2
    exit 2
  fi
  return 0
}

push_output_file=$(mktemp)
trap 'rm -f "$push_output_file"' EXIT

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if attempt_bump_and_push "$push_output_file"; then
    echo "Bump + tag push succeeded on attempt ${attempt}/${MAX_ATTEMPTS}"
    exit 0
  fi
  cat "$push_output_file"
  if is_protection_error "$(cat "$push_output_file")"; then
    {
      echo "ERROR: Push to main rejected by branch protection."
      echo "       Add the GitHub App (GH_APP_ID=${GH_APP_ID:-?}) to the"
      echo "       bypass-actors list of the ruleset / branch-protection"
      echo "       rule covering main."
    } >&2
    exit 1
  fi
  if (( attempt < MAX_ATTEMPTS )); then
    echo "push lost the race on attempt ${attempt}/${MAX_ATTEMPTS}; refreshing main+tags, recomputing version, retrying"
  fi
done

{
  echo "ERROR: Push to main failed after ${MAX_ATTEMPTS} attempts."
  echo "       Likely racing concurrent pushes this job couldn't outrace."
} >&2
exit 1
