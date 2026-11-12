#!/bin/bash
# recipe.sh USER DISTRIBUTION VERSION: run once as root over ssh in an Omarchy base's builder, Omarchy's own image booted
# by the QEMU launcher with its session user USER signed in. It refuses an image that isn't the DISTRIBUTION and VERSION
# its artifact declares, then makes it a computer's base: agent-env's Hyprland drop-in; the seed unit that hands each
# boot its deploy's environment; SDDM logging back in after a logout; Omarchy's toggles for an unattended desktop; off,
# what a trial doesn't want: crash toasts, dictation and its keys, housekeeping timers that would run at boot, and the
# firewall, which would stop the computer's own ports; and cua-driver's Hyprland plugin, built by its release's kit on
# what the kit pins. The session user's password is locked, and the machine id left empty, as Omarchy's own sealed
# images leave it.
set -euo pipefail
user=$1 distribution=$2 version=$3
here=$(cd "$(dirname "$0")" && pwd)
home=$(getent passwd "$user" | cut -d: -f6)
#: cua-driver's Hyprland plugin, as its release's source and build kit, which the driver's Hyprland pin matches.
plugin_release=https://github.com/trycua/cua/releases/download/cua-driver-rs-v0.47.4
plugin=cua-hyprland-plugin-0.47.4-09b38ab661aa8b7d46deb5ff98ca00063019f592
plugin_sha256=fd2e7f6426388753c19b8898a920f1f0d19f3ce81d5f7d7ae9ef3a2dee2c3b98
plugin_kit_sha256=4da83523664049e9fa5a5f3269294dfad9602e6c5a02b0534a8c6f5b29920354
#: What the kit builds on, in place of Omarchy's newer builds: Arch's Hyprland 0.56.2-1 with the libraries it was built
#: against, and the GCC 16.1.1 it was built with, from that day of the Arch Linux Archive (arch-day.conf).
pinned=(hyprland aquamarine hyprtoolkit hyprland-guiutils gcc gcc-libs libasan libatomic libgcc libgfortran libgomp
        libhwasan liblsan libobjc libquadmath libstdc++ libtsan libubsan)
#: What the kit builds with that Omarchy lacks, removed once it has.
build_tools=(cmake ninja)

. /etc/os-release
if [ "$ID" != "$distribution" ] || [ "$VERSION_ID" != "$version" ]; then
    echo "recipe: this image is ${ID:-?} ${VERSION_ID:-?}, not $distribution $version as its artifact declares" >&2
    exit 3
fi
echo "os-release: ID=$ID VERSION_ID=$VERSION_ID"

install -d /opt/agentenv
install -m 644 "$here/seed-env" /opt/agentenv/seed-env
install -m 755 "$here/agentenv-seed" "$here/with-seed" /opt/agentenv/
echo "$user" > /opt/agentenv/session-user
install -m 644 "$here/agentenv-seed.service" /etc/systemd/system/agentenv-seed.service
systemctl enable --quiet agentenv-seed.service
install -d /etc/sddm.conf.d
install -m 644 "$here/sddm-relogin.conf" /etc/sddm.conf.d/zz-agentenv.conf

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

# The rest with the desktop session ended, as the plugin's kit says to install it: Hyprland reloads its config whenever
# a file of it changes, and the Hyprland running here can neither load the plugin nor show the error its config naming a
# missing one makes without hanging. The computer boots the pinned Hyprland with the drop-in loading the plugin.
systemctl stop sddm.service
loginctl terminate-user "$user"
for _ in $(seq 60); do pgrep -x Hyprland > /dev/null || break; sleep 1; done
if pgrep -x Hyprland > /dev/null; then
    echo "recipe: Hyprland still runs a minute after the desktop session ended" >&2
    exit 1
fi
install -m 644 -o "$user" -g "$user" "$here/agentenv.lua" "$home/.config/hypr/agentenv.lua"
grep -q 'require("hypr.agentenv")' "$home/.config/hypr/hyprland.lua" \
    || echo 'require("hypr.agentenv")' >> "$home/.config/hypr/hyprland.lua"
# The plugin's kit runs makepkg as an ordinary user, in a folder of its own, and its package goes in from there.
pacman --config "$here/arch-day.conf" -Sy
pacman --config "$here/arch-day.conf" -S --needed --noconfirm "${pinned[@]}" "${build_tools[@]}"
kit=$(mktemp -d) download=$(mktemp -d)
curl -fsSL --retry 5 -o "$kit/$plugin.tar.gz" "$plugin_release/$plugin.tar.gz"
curl -fsSL --retry 5 -o "$download/kit.tar.gz" "$plugin_release/$plugin-build-kit.tar.gz"
echo "$plugin_sha256  $kit/$plugin.tar.gz" | sha256sum --check --quiet
echo "$plugin_kit_sha256  $download/kit.tar.gz" | sha256sum --check --quiet
tar -xzf "$download/kit.tar.gz" -C "$kit"
(cd "$kit" && sha256sum --check --quiet SHA256SUMS)
chown -R "$user:$user" "$kit"
runuser -u "$user" -- env HOME="$home" bash -c 'cd "$1" && makepkg --noconfirm' makepkg "$kit"
pacman -U --noconfirm "$kit"/cua-hyprland-plugin-*.pkg.tar.zst
pacman -Rns --noconfirm "${build_tools[@]}"
rm -rf "$kit" "$download" /var/lib/pacman/sync/core.db* /var/lib/pacman/sync/extra.db*
pacman -Q hyprland gcc cua-hyprland-plugin

passwd --lock --quiet "$user" 2>/dev/null || passwd -l "$user" > /dev/null
truncate -s 0 /etc/machine-id
echo "recipe: done"
