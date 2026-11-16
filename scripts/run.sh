#!/usr/bin/env bash
# Run NASTRAN-95 on one or more input decks: scripts/run.sh d01011a [d01012a ...]
# Decks are read from inp/<name>.inp; output goes to ${OUT_DIR:-out}/<name>.out
# (the Unix equivalent of the interactive csh driver bin/nastran).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$(mkdir -p "${OUT_DIR:-$ROOT/out}" && cd "${OUT_DIR:-$ROOT/out}" && pwd)"
EXE="${NASTRAN_EXE:-$ROOT/bin/nastran.exe}"
[ -n "${NASTRAN_EXE:-}" ] || make -C "$ROOT" -j"$(nproc)" >/dev/null || exit 2
rc_all=0
for name in "$@"; do
  name="$(basename "$name" .inp)"
  work="$(mktemp -d "${TMPDIR:-/tmp}/nastran-$name.XXXX")"
  (
    cd "$work"
    # NASTRAN keeps file names in CHARACTER*44, so use short names relative to the work dir;
    # every logical file gets its own name (gfortran won't connect one file to two units).
    ln -s "$ROOT/build/rf" rf
    env NPTPNM=nptp PLTNM=plt DICTNM=dict PUNCHNM=pch OPTPNM=optp SOF1=sof1 SOF2=sof2 \
        DIRCTY=. LOGNM=log RFDIR=rf \
        FTN11=ftn11 FTN12=ftn12 FTN13=ftn13 FTN14=ftn14 FTN15=ftn15 FTN16=ftn16 FTN17=ftn17 \
        FTN18=ftn18 FTN19=ftn19 FTN20=ftn20 FTN21=ftn21 FTN22=ftn22 FTN23=ftn23 \
        DBMEM=12000000 OCMEM=2000000 \
        timeout "${NASTRAN_TIMEOUT:-300}" "$EXE" < <(tr -d '\r\032' < "$ROOT/inp/$name.inp") > "$OUT_DIR/$name.out"
    rc=$?
    [ -s log ] && cp log "$OUT_DIR/$name.log"
    [ -s pch ] && cp pch "$OUT_DIR/$name.pch"
    exit $rc
  )
  rc=$?
  rm -rf "$work"
  echo "$name rc=$rc $(grep -c . "$OUT_DIR/$name.out") lines"
  [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
