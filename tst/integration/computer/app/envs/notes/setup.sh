#!/bin/sh
# Run as root once the app's packages are in: hello is one of them, and fastfetch the one only its PPA has.
set -e
hello > hello.txt
dpkg-query -W -f '${Version}\n' fastfetch > fastfetch.txt
fastfetch --version >> fastfetch.txt
