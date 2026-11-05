# lib.sh, sourced by a composed VM bundle's provision script, and by the script that resolves its venvs' wheels, in the
# VM, as its session user: root through sudo, with the password the build folder holds, the pinned Python agent-env
# runs on, and TCC permissions granted to an app bundle.

build="/Volumes/My Shared Files/agentenv-build"
export PATH="/opt/agentenv/bin:/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

fail() { echo "agentenv-compose: $*" >&2; exit 1; }

# as_root COMMAND...: COMMAND as root.
as_root() { sudo -S -p '' "$@" < "$build/.password"; }

# The Python agent-env runs on: python-build-standalone's CPython 3.12.16 for Apple silicon, with Tk, by its sha256.
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-aarch64-apple-darwin-install_only.tar.gz"
PYTHON_SHA256=fbf4afef53c2a6b72bb741bc06d5671a9dfaf87bd84e5aa73ac9b3d0d589f14e

# pinned_python: that Python at /opt/agentenv/python, python3 first on agent-env's PATH.
pinned_python() {
    local tarball
    tarball=$(mktemp)
    curl -fsSL --retry 3 -o "$tarball" "$PYTHON_URL"
    echo "$PYTHON_SHA256  $tarball" | shasum -a 256 -c - > /dev/null || fail "the Python downloaded isn't the one pinned"
    as_root mkdir -p /opt/agentenv
    as_root chown "$(id -un):staff" /opt/agentenv
    rm -rf /opt/agentenv/python
    mkdir -p /opt/agentenv/python /opt/agentenv/bin
    tar -xzf "$tarball" -C /opt/agentenv/python --strip-components 1
    rm -f "$tarball"
    ln -sf /opt/agentenv/python/bin/python3 /opt/agentenv/bin/python3
}

# grant APP SERVICE...: each TCC SERVICE (accessibility, post_event, screen_capture) for the app bundle APP, by its
# identifier, bound to the code signature it has now, so another app of that identifier gets none.
grant() {
    local app=$1 id requirement hex service now csreq
    shift
    id=$(defaults read "$app/Contents/Info" CFBundleIdentifier) || fail "$app has no bundle identifier"
    requirement=$(codesign -d -r- "$app" 2>&1 | sed -n 's/^\(# *\)\{0,1\}designated => //p')
    [ -n "$requirement" ] || fail "$app has no designated requirement, so it can't be granted"
    csreq=$(mktemp)
    echo "$requirement" | csreq -r- -b "$csreq"
    hex=$(xxd -p "$csreq" | tr -d '\n')
    rm -f "$csreq"
    now=$(date +%s)
    for service in "$@"; do
        case "$service" in
            accessibility) service=kTCCServiceAccessibility ;;
            post_event) service=kTCCServicePostEvent ;;
            screen_capture) service=kTCCServiceScreenCapture ;;
            *) fail "no TCC service $service" ;;
        esac
        as_root sqlite3 "/Library/Application Support/com.apple.TCC/TCC.db" "INSERT OR REPLACE INTO access (service,
            client, client_type, auth_value, auth_reason, auth_version, csreq, policy_id, indirect_object_identifier_type,
            indirect_object_identifier, indirect_object_code_identity, flags, last_modified, pid, pid_version, boot_uuid,
            last_reminded) VALUES ('$service', '$id', 0, 2, 2, 1, X'$hex', NULL, 0, 'UNUSED', NULL, 0, $now, NULL, NULL,
            'UNUSED', $now);"
        if [ "$service" = kTCCServiceScreenCapture ]; then
            # macOS asks once an app, even granted, before its first direct capture; an approval recorded ahead counts.
            approvals="$HOME/Library/Group Containers/group.com.apple.replayd/ScreenCaptureApprovals.plist"
            mkdir -p "$(dirname "$approvals")"
            defaults write "$approvals" "$id" -dict kScreenCaptureApprovalLastAlerted -date "3024-01-01 00:00:00 +0000" \
                kScreenCaptureApprovalLastUsed -date "3024-01-01 00:00:00 +0000"
        fi
    done
    for pid in $(pgrep -x tccd); do as_root kill "$pid" 2>/dev/null || true; done
}
