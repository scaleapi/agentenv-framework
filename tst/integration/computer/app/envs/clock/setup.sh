#!/bin/sh
# Run as root once tzdata is in: the zones it installed, beside the app.
set -e
ls /usr/share/zoneinfo/Europe > zones.txt
