#!/bin/sh
# Builds agent-env's libfaketime: upstream 0.9.12 from its sha256-pinned tarball, libfaketime.patch applied and the two
# engine files beside it copied in, the multithreaded library with stat and utime faking compiled out (file times stay
# real); and the TLS shim beside it (tls_realtime.c). Writes OUT/libfaketimeMT.so.1 and OUT/libtls_realtime.so, which
# /etc/ld.so.preload lists in that order. Needs a C toolchain, make, patch and curl; builds on x86_64 and aarch64.
#   build.sh OUT [TARBALL]   (TARBALL: a local copy of upstream's v0.9.12 tarball, else it is downloaded)
set -eu
out=$1
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tarball=${2:-$work/libfaketime.tar.gz}
[ -f "$tarball" ] || curl -fsSL -o "$tarball" https://github.com/wolfcw/libfaketime/archive/refs/tags/v0.9.12.tar.gz
echo "4fc32218697c052adcdc5ee395581f2554ca56d086ac817ced2be0d6f1f8a9fa  $tarball" | sha256sum -c - >/dev/null
tar xzf "$tarball" -C "$work"
src=$work/libfaketime-0.9.12
patch -s -d "$src" -p1 < "$here/libfaketime.patch"
cp "$here/agentenv_clock_page.c" "$here/agentenv_clock_waits.c" "$src/src/"
make -s -C "$src/src" FAKETIME_COMPILE_CFLAGS="-UFAKE_STAT -UFAKE_UTIME" libfaketimeMT.so.1
mkdir -p "$out"
cp "$src/src/libfaketimeMT.so.1" "$out/"
${CC:-cc} -std=gnu99 -O2 -fPIC -shared -Wall -Wextra -Werror -o "$out/libtls_realtime.so" "$here/tls_realtime.c" -ldl
