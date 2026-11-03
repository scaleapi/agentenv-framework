#!/usr/bin/env bash
# The shell side of a computer env's builder VM (builder.VmBuilder): sent to the VM once, then run as
# `bash builder_in_vm.sh <subcommand> <args>`. Those that don't use Docker run on Linux and macOS alike.
set -euo pipefail

retry=(--retry 5 --retry-all-errors --retry-delay 1)

sum() {
  if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -c1-64
}

# buildx URL SHA256 PLUGINS: give Docker BuildKit's client from URL, checked by SHA256, when it has none.
buildx() {
  docker buildx version >/dev/null 2>&1 && return
  mkdir -p "$3"
  curl -fsSL "${retry[@]}" "$1" -o "$3/docker-buildx"
  [ "$(sum "$3/docker-buildx")" = "$2" ] || { echo "$1 isn't the buildx pinned, $2" >&2; exit 1; }
  chmod 755 "$3/docker-buildx"
}

# emulate ARCH IMAGE: register QEMU for ARCH with IMAGE, the VM's own binfmt_misc mounted first, since a newer kernel
# gives a mount of it in a container an instance of its own.
emulate() {
  mountpoint -q /proc/sys/fs/binfmt_misc || mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc
  docker run --privileged --rm "$2" --install "$1"
}

# files DIR: each plain file under DIR, never a link, as "<sha256> <size> <path under DIR>", each ending in a NUL.
files() {
  cd "$1"
  find . -type f -print0 | while IFS= read -r -d '' f; do
    printf '%s %s %s\0' "$(sum "$f")" "$(wc -c < "$f" | tr -d ' ')" "${f#./}"
  done
}

# link SOURCE TARGET ...: each TARGET a hard link to its SOURCE.
link() {
  while [ $# -gt 0 ]; do
    mkdir -p "$(dirname "$2")"
    ln -f "$1" "$2"
    shift 2
  done
}

# check DIR LIST: each line of LIST, "<sha256>  <path under DIR>", naming each file whose sha256 differs.
check() {
  cd "$1"
  local failed=0 line want path
  while IFS= read -r line; do
    want="${line%%  *}" path="${line#*  }"
    if [ "$(sum "$path")" != "$want" ]; then
      echo "$path: its sha256 is $(sum "$path"), not $want"
      failed=1
    fi
  done < "$2"
  return "$failed"
}

# runnable PATH ...: make each executable that starts as a script or a Linux binary does.
runnable() {
  for f in "$@"; do
    case "$(head -c 4 "$f" | od -An -tx1 | tr -d ' \n')" in
      2321* | 7f454c46) chmod 755 "$f" ;;
    esac
  done
}

# download URL PATH: PATH fetched from URL; its sha256.
download() {
  mkdir -p "$(dirname "$2")"
  curl -fsSL "${retry[@]}" "$1" -o "$2"
  sum "$2"
}

# installed TAG FILE: FILE copied out of TAG's image without running it, on any arch; nothing when it holds none.
installed() {
  local container copied
  container=$(docker create "$1")
  copied=$(mktemp)
  if docker cp "$container:$2" "$copied" >/dev/null 2>&1; then cat "$copied"; fi
  docker rm "$container" >/dev/null
  rm -f "$copied"
}

"$@"
