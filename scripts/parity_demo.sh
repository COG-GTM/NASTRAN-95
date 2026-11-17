#!/usr/bin/env bash
# Run a delta-wing deck through gfortran NASTRAN-95 and the Python port, then compare.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DECKS="${*:-d01013a_all d01014a_all}"
[ -x bin/nastran.exe ] || make -j"$(nproc)" >/dev/null
mkdir -p out/parity
for d in $DECKS; do
  echo "== $d: gfortran NASTRAN-95"
  INP_DIR=tests/decks OUT_DIR=out/parity scripts/run.sh "$d" >/dev/null
  grep -m1 "SUBTITLE\|NASTRAN DEMONSTRATION PROBLEM" "out/parity/$d.out" | sed 's/^ *//' || true
  echo "== $d: Python port vs gfortran listing"
  python3 -m nastran95py.parity_run "tests/decks/$d.inp" "out/parity/$d.out"
done
