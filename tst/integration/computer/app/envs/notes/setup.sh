#!/bin/sh
# Run as root once the app's packages are in: hello is one of them, and docker-compose-plugin the one only Docker's apt
# repository has.
set -e
hello > hello.txt
dpkg-query -W -f '${Version}\n' docker-compose-plugin > compose.txt
/usr/libexec/docker/cli-plugins/docker-compose version >> compose.txt
