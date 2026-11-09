# lib.sh, sourced by a composed VM bundle's provision script, and by the script that resolves its venvs' wheels, in the
# VM, as its session user: root through sudo, with the password the build folder holds, the pinned Python agent-env
# runs on, and TCC permissions granted to an app bundle or an executable.

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
    local app=$1 id
    shift
    id=$(defaults read "$app/Contents/Info" CFBundleIdentifier) || fail "$app has no bundle identifier"
    granted "$id" 0 "$app" "$@"
}

# grant_executable EXECUTABLE SERVICE...: those for the executable file EXECUTABLE, by its path, bound to the code
# signature it has now, so another program at that path gets none. What it starts has them too: macOS charges a process
# to the program launchd started.
grant_executable() {
    local executable=$1
    shift
    [ -f "$executable" ] && [ ! -L "$executable" ] || fail "$executable isn't an executable file of its own"
    granted "$executable" 1 "$executable" "$@"
}

# file_url PATH: the file URL of the folder PATH as Foundation writes a bundle's: letters, digits and "!$&'()*+,-./:=@_~"
# as they are, every other printable ASCII character percent-encoded. A path with any other byte fails, since Foundation
# decomposes such a character before encoding it.
file_url() {
    local LC_ALL=C path=${1%/} url= char i
    for ((i = 0; i < ${#path}; i++)); do
        char=${path:i:1}
        case "$char" in
            [A-Za-z0-9/._~!\$\&\'\(\)\*+,:=@-]) url+=$char ;;
            [[:print:]]) url+=$(printf '%%%02X' "'$char") ;;
            *) fail "$1 isn't printable ASCII, so its file URL can't be written for its screen capture approval" ;;
        esac
    done
    echo "file://$url/"
}

# granted CLIENT TYPE PROGRAM SERVICE...: the TCC rows of CLIENT, a bundle identifier (TYPE 0) or a path (TYPE 1), for
# each SERVICE, bound to PROGRAM's designated requirement.
granted() {
    local client=$1 type=$2 program=$3 requirement hex service now csreq approvals quoted keys key
    shift 3
    quoted=${client//\'/\'\'}
    requirement=$(codesign -d -r- "$program" 2>&1 | sed -n 's/^\(# *\)\{0,1\}designated => //p')
    [ -n "$requirement" ] || fail "$program has no designated requirement, so it can't be granted"
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
            last_reminded) VALUES ('$service', '$quoted', $type, 2, 2, 1, X'$hex', NULL, 0, 'UNUSED', NULL, 0, $now, NULL,
            NULL, 'UNUSED', $now);"
        if [ "$service" = kTCCServiceScreenCapture ]; then
            # macOS asks a program, even granted, before its first direct capture, keyed by the identifier or the path
            # TCC knows it by, and an app by its bundle's file URL too, "to bypass the system private window picker",
            # and again once its privacy hint's date passes; approvals recorded ahead under each, hinted at no sooner
            # than they are, count.
            approvals="$HOME/Library/Group Containers/group.com.apple.replayd/ScreenCaptureApprovals.plist"
            mkdir -p "$(dirname "$approvals")"
            keys=("$client")
            if ((type == 0)); then keys+=("$(file_url "$program")"); fi
            for key in "${keys[@]}"; do
                defaults write "$approvals" "$key" -dict \
                    kScreenCaptureApprovalLastAlerted -date "3024-01-01 00:00:00 +0000" \
                    kScreenCaptureApprovalLastUsed -date "3024-01-01 00:00:00 +0000" \
                    kScreenCapturePrivacyHintDate -date "3024-01-01 00:00:00 +0000" \
                    kScreenCapturePrivacyHintPolicy -int 3153600000
            done
        fi
    done
    for pid in $(pgrep -x tccd); do as_root kill "$pid" 2>/dev/null || true; done
}
