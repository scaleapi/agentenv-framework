#!/bin/bash
# recipe.sh USER DISTRIBUTION VERSION: run once as root over ssh in an Omarchy base's builder, Omarchy's own image booted
# by the QEMU launcher with its session user USER signed in. It refuses an image that isn't the DISTRIBUTION and VERSION
# its artifact declares, then makes it a computer's base: cua-driver's Hyprland plugin, pinned to the package built for
# this Omarchy's Hyprland; agent-env's Hyprland drop-in; the seed unit that hands each boot its deploy's environment;
# SDDM logging back in after a logout; Omarchy's toggles for an unattended desktop; and off, what a trial doesn't want:
# crash toasts, dictation and its keys, housekeeping timers that would run at boot, and the firewall, which would stop
# the computer's own ports. The session user's password is locked, and the machine id left empty, as Omarchy's own
# sealed images leave it.
set -euo pipefail
user=$1 distribution=$2 version=$3
here=$(cd "$(dirname "$0")" && pwd)
home=$(getent passwd "$user" | cut -d: -f6)
#: cua-driver's Hyprland plugin, as Omarchy's own repository builds it for Omarchy 4.5's Hyprland 0.56.2-4.
plugin_url=https://pkgs.omarchy.org/stable/x86_64/cua-hyprland-plugin-0.33.2-3-x86_64.pkg.tar.zst
plugin_sha256=48a25c70dafdd423118c9937d3b469953e89a15cc0f278662f8fd728b523c47e

. /etc/os-release
if [ "$ID" != "$distribution" ] || [ "$VERSION_ID" != "$version" ]; then
    echo "recipe: this image is ${ID:-?} ${VERSION_ID:-?}, not $distribution $version as its artifact declares" >&2
    exit 3
fi
echo "os-release: ID=$ID VERSION_ID=$VERSION_ID"

plugin=$(mktemp --suffix=.pkg.tar.zst)
curl -fsSL --retry 5 -o "$plugin" "$plugin_url"
echo "$plugin_sha256  $plugin" | sha256sum --check --quiet
pacman -U --noconfirm --needed "$plugin"
rm -f "$plugin"

install -d /opt/agentenv
install -m 644 "$here/seed-env" /opt/agentenv/seed-env
install -m 755 "$here/agentenv-seed" "$here/with-seed" /opt/agentenv/
echo "$user" > /opt/agentenv/session-user
install -m 644 "$here/agentenv-seed.service" /etc/systemd/system/agentenv-seed.service
systemctl enable --quiet agentenv-seed.service
install -d /etc/sddm.conf.d
install -m 644 "$here/sddm-relogin.conf" /etc/sddm.conf.d/zz-agentenv.conf

install -m 644 -o "$user" -g "$user" "$here/agentenv.lua" "$home/.config/hypr/agentenv.lua"
grep -q 'require("hypr.agentenv")' "$home/.config/hypr/hyprland.lua" \
    || echo 'require("hypr.agentenv")' >> "$home/.config/hypr/hyprland.lua"

systemctl --global mask --quiet omarchy-crash-watch.service superwhisper.service
systemctl mask --quiet snapper-cleanup.timer plocate-updatedb.timer man-db.timer shadow.timer \
    archlinux-keyring-wkd-sync.timer
# With no dictation backend chosen, Omarchy binds no dictation keys (its bindings/dictation.lua), Alt+Space among them.
rm -f "$home/.config/omarchy/defaults/dictation"
# Omarchy's firewall lets in only ssh and LocalSend. A computer's network is the launcher's user-mode one, which only the
# ports agent-env forwards reach, its driver's and apps', so the firewall would stop just those.
ufw --force disable > /dev/null
systemctl disable --quiet ufw.service

runtime=/run/user/$(id -u "$user")
for _ in $(seq 180); do
    [ -S "$runtime/bus" ] && compgen -G "$runtime/hypr/*/.socket.sock" > /dev/null && break
    sleep 1
done
for _ in $(seq 180); do
    [ -e "$home/.local/state/omarchy/done/first-run-user" ] && break
    sleep 1
done
# A command run in the session user's own systemd manager, with the environment its graphical session gave it.
session() { systemd-run --quiet --machine="$user@.host" --user --wait --pipe --collect -- "$@"; }
session omarchy toggle idle stay-awake
session omarchy toggle animations off
session omarchy toggle notification silencing
session omarchy toggle screensaver
session omarchy toggle suspend

passwd --lock --quiet "$user" 2>/dev/null || passwd -l "$user" > /dev/null
truncate -s 0 /etc/machine-id
echo "recipe: done"
