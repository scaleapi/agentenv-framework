#!/bin/sh
# Builds agent-env's libfaketime and its TLS shim from ./clock and the downloaded tarball, installs the library's
# settings and the page's writer, and preloads both into every process the computer starts, the library first.
set -eu
sh clock/build.sh /opt/agentenv/clock libfaketime.tar.gz
install -m 0644 clock/clockpage.py /opt/agentenv/clock/clockpage.py
install -D -m 0644 clock/clock.conf /etc/agentenv/clock.conf
printf '%s\n' /opt/agentenv/clock/libfaketimeMT.so.1 /opt/agentenv/clock/libtls_realtime.so > /etc/ld.so.preload
