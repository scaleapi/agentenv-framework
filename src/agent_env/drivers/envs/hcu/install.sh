#!/bin/sh
# Turn the desktop's accessibility on from boot, as root, through a dconf system default. Chromium and Electron read the
# accessibility setting only when they start, and a computer's apps start at boot, before HCU's first call would switch
# it on; with org.gnome.desktop.interface toolkit-accessibility true from the start, at-spi-bus-launcher reports
# org.a11y.Status IsEnabled true and every app that starts joins the accessibility bus.
set -eu
mkdir -p bin
tar -xzf computer-use.tar.gz -C bin
rm -f computer-use.tar.gz
mkdir -p /etc/dconf/profile /etc/dconf/db/local.d
printf 'user-db:user\nsystem-db:local\n' > /etc/dconf/profile/user
printf '[org/gnome/desktop/interface]\ntoolkit-accessibility=true\n' > /etc/dconf/db/local.d/00-hcu-accessibility
dconf update
