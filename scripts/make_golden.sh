#!/usr/bin/env bash
# Regenerate tests/golden from the gfortran build: the NASA decks and the tests/decks
# variants (same models, every output request set to ALL).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
INP_DIR=tests/decks OUT_DIR=tests/golden scripts/run.sh d01013a_all d01014a_all
OUT_DIR=tests/golden scripts/run.sh d01013a d01014a
rm -f tests/golden/*.log tests/golden/*.pch
