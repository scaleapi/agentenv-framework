#!/bin/sh
# Run as root once the app's packages are in: a copy of the apt JDK under /opt, its links kept but its store a file of
# its own rather than the link to the one the system store's hook builds, as a JDK not from apt has.
set -e
cp -a /usr/lib/jvm/java-21-openjdk-*/. /opt/copied-jdk/
rm /opt/copied-jdk/lib/security/cacerts
cp /etc/ssl/certs/java/cacerts /opt/copied-jdk/lib/security/cacerts
test ! -L /opt/copied-jdk/lib/security/cacerts
