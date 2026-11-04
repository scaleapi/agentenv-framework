# lib.sh, sourced by a composed VM bundle's provision script in the VM, as its session user: root through sudo, with
# the password the build folder holds, and TCC permissions granted to an app bundle.

build="/Volumes/My Shared Files/agentenv-build"
export PATH="/opt/agentenv/bin:/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

fail() { echo "agentenv-compose: $*" >&2; exit 1; }

# as_root COMMAND...: COMMAND as root.
as_root() { sudo -S -p '' "$@" < "$build/.password"; }

# grant APP SERVICE...: each TCC SERVICE (accessibility, post_event, screen_capture) for the app bundle APP, by its
# identifier, bound to the code signature it has now, so another app of that identifier gets none.
grant() {
    local app=$1 id requirement hex service now
    shift
    id=$(defaults read "$app/Contents/Info" CFBundleIdentifier) || fail "$app has no bundle identifier"
    requirement=$(codesign -d -r- "$app" 2>&1 | sed -n 's/^\(# *\)\{0,1\}designated => //p')
    [ -n "$requirement" ] || fail "$app has no designated requirement, so it can't be granted"
    echo "$requirement" | csreq -r- -b "$TMPDIR/agentenv.csreq"
    hex=$(xxd -p "$TMPDIR/agentenv.csreq" | tr -d '\n')
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
