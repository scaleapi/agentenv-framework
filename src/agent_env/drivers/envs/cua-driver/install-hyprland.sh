#!/bin/sh
# cua-driver's executable on Hyprland, from the release tarball the install downloaded and checked, in the driver's
# cua-driver folder.
set -eu
mkdir -p cua-driver
tar -xzf cua-driver.tar.gz -C cua-driver cua-driver
rm -f cua-driver.tar.gz
