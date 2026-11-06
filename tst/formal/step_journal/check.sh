#!/usr/bin/env bash
# Model-check StepJournal.tla with TLC and compare every run with its recorded outcome.
#
#   tst/formal/step_journal/check.sh              the baseline configs
#   tst/formal/step_journal/check.sh mutations    each modeled code break must be caught
#   tst/formal/step_journal/check.sh one CFG [INVARIANT] [MUT]   one run, no expectation
#
# Exits nonzero when any run differs from its expectation, or TLC errors (a deadlock is a modeling error).
# Needs Java 11+ and, on the first run, the network to fetch the pinned TLC jar.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
TLA_VERSION=1.7.4
TLA_SHA256=936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88
TLA_URL=https://github.com/tlaplus/tlaplus/releases/download/v$TLA_VERSION/tla2tools.jar
cache=${AGENTENV_TLA_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/agentenv-formal}
[[ $cache == /* ]] || cache=$PWD/$cache
jar=$cache/tla2tools-$TLA_VERSION.jar
out=${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/step-journal-tlc.XXXXXX")}

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

fetch_tlc() {
  if [[ -f $jar && $(sha256 "$jar") == "$TLA_SHA256" ]]; then return; fi
  mkdir -p "$cache"
  local part got; part=$(mktemp "$cache/tla2tools.XXXXXX")
  curl -fsSL -o "$part" "$TLA_URL" || { rm -f "$part"; exit 2; }
  got=$(sha256 "$part")
  if [[ $got != "$TLA_SHA256" ]]; then
    rm -f "$part"
    echo "tla2tools.jar sha256 $got, expected $TLA_SHA256" >&2
    exit 2
  fi
  mv -f "$part" "$jar"
}

die() { echo "check.sh: $*" >&2; exit 2; }

configure() {
  local name=$1 cfg=$2 invariant=$3 mut=$4 dir=$out/$1
  mkdir -p "$dir"
  sed -e "s/Mut = \"none\"/Mut = \"$mut\"/" "$here/$cfg" > "$dir/$name.cfg"
  grep -q "Mut = \"$mut\"" "$dir/$name.cfg" || die "$cfg has no 'Mut = \"none\"' line to set Mut = \"$mut\" in"
  if [[ -n $invariant ]]; then
    sed -i.bak -e "s/^INVARIANT .*/INVARIANT $invariant/" "$dir/$name.cfg" && rm "$dir/$name.cfg.bak"
    grep -qx "INVARIANT $invariant" "$dir/$name.cfg" || die "$cfg has no INVARIANT line to replace with INVARIANT $invariant"
  fi
  cp "$here/StepJournal.tla" "$dir/"
}

run() {
  local name=$1 dir=$out/$1
  (cd "$dir" && java -XX:+UseParallelGC -cp "$jar" tlc2.TLC -workers auto -metadir meta \
    -config "$name.cfg" StepJournal.tla > tlc.out 2>&1) || true
  local states outcome
  states=$(grep -Eo '[0-9,]+ distinct states found' "$dir/tlc.out" | tail -1 || true)
  local violated; violated=$(grep -Eo 'Invariant [A-Za-z]+ is violated' "$dir/tlc.out" | head -1 | awk '{print $2}' || true)
  if [[ -n $violated ]]; then outcome="violates $violated"
  elif grep -q 'Model checking completed. No error has been found.' "$dir/tlc.out"; then outcome=holds
  else outcome=error; fi
  printf '%s\t%s\n' "$outcome" "${states:-no states}"
}

failed=0
expect() {
  local name=$1 cfg=$2 invariant=$3 mut=$4 want=$5 note=${6:-}
  local outcome states mark=ok
  configure "$name" "$cfg" "$invariant" "$mut"
  IFS=$'\t' read -r outcome states < <(run "$name")
  if [[ $outcome != "$want" ]]; then mark=UNEXPECTED; failed=1; fi
  printf '%-11s %-34s %-34s %s%s\n' "$mark" "$name" "$outcome" "$states" "${note:+  ($note)}"
  if [[ $mark != ok ]]; then echo "            expected: $want; TLC output: $out/$name/tlc.out"; fi
}

command -v java >/dev/null || die "java not found on PATH; TLC needs Java 11 or later"
fetch_tlc
case ${1:-baseline} in
  baseline)
    expect single   single.cfg   "" none holds
    expect takeover takeover.cfg "" none holds
    ;;
  mutations)
    expect NoCasEpochFence         single.cfg   NoSupersededWriteLands NoCasEpochFence      "violates NoSupersededWriteLands"
    expect NoUndoEpochBump         single.cfg   NoSupersededWriteLands NoUndoEpochBump      "violates NoSupersededWriteLands"
    expect NoCas                   single.cfg   NoSupersededWriteLands NoCas                "violates NoSupersededWriteLands"
    expect NoUndoFence             takeover.cfg NoSupersededWriteLands NoUndoFence          "violates NoSupersededWriteLands"
    expect NoTerminalFence         takeover.cfg NoSupersededWriteLands NoTerminalFence      "violates NoSupersededWriteLands"
    expect NoJournalSeqGuard       single.cfg   ReplayReproducesStored NoJournalSeqGuard    "violates ReplayReproducesStored"
    expect FenceBeforeSeq          single.cfg   ReplayReproducesStored FenceBeforeSeq       "violates ReplayReproducesStored"
    expect NoSeedDocFence-inv1     takeover.cfg NoSupersededWriteLands NoSeedDocFence       "violates NoSupersededWriteLands"
    expect NoSeedDocFence-inv2     takeover.cfg ReplayReproducesStored NoSeedDocFence       "violates ReplayReproducesStored"
    expect NoSeedJournalFence-inv1 takeover.cfg NoSupersededWriteLands NoSeedJournalFence   "violates NoSupersededWriteLands"
    expect NoSeedJournalFence-inv2 takeover.cfg ReplayReproducesStored NoSeedJournalFence   "violates ReplayReproducesStored"
    expect NoCasGenerationCheck    takeover.cfg ""                     NoCasGenerationCheck holds "redundant: re-register also bumps attempt_epoch"
    ;;
  one)
    configure one "${2:?config file}" "${3:-}" "${4:-none}"
    IFS=$'\t' read -r outcome states < <(run one)
    echo "$outcome, $states; TLC output: $out/one/tlc.out"
    if [[ $outcome == error ]]; then exit 2; fi
    exit 0
    ;;
  *) echo "usage: $0 [baseline|mutations|one CFG [INVARIANT] [MUT]]" >&2; exit 2 ;;
esac
echo "TLC $TLA_VERSION; configs and output in $out"
exit $failed
