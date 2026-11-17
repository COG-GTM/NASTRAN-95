"""PARITY.md generation (``tests/parity_report.py``)."""

from __future__ import annotations

import pytest

from nastran95py.reference.inventory import RF1_DECKS

from . import parity_report as P


def _run(**over) -> dict:
    records = [
        {"name": d, "n": 1, "max_abs": 0.0, "max_rel": 0.0, "deck": d, "table": "x"}
        for d in RF1_DECKS
    ]
    records += [
        {"name": e, "n": 1, "max_abs": 0.0, "max_rel": 0.0, "element": e, "quantity": q,
         "source": "golden"}
        for e in P.ELEMENTS for q, _ in P.QUANTITIES
    ]  # fmt: skip
    return {"exitstatus": 0, "records": records, "outcomes": [], **over}


def test_complete_run_validates():
    P._validate(_run())


def test_failed_run_is_rejected():
    run = _run(exitstatus=1, outcomes=[{"nodeid": "tests/x.py::t", "outcome": "failed"}])
    with pytest.raises(P.ParityError, match="tests/x.py::t"):
        P._validate(run)


def test_partial_run_is_rejected():
    run = _run()
    run["records"] = [r for r in run["records"] if r.get("deck") != "d01014a"]
    with pytest.raises(P.ParityError, match="d01014a"):
        P._validate(run)


def test_run_without_fortran_harness_is_rejected():
    run = _run()
    run["records"] = [r for r in run["records"] if r.get("element") != "CSHEAR"]
    with pytest.raises(P.ParityError, match="CSHEAR"):
        P._validate(run)


def test_stale_ignores_last_bit_noise_only():
    old = "| d | x | 1 | 10 | 1.00e-04 | 3.00e-16 |"
    assert P.stale("| d | x | 1 | 10 | 1.01e-04 | 7.00e-16 |", old) == []
    assert P.stale("| d | x | 1 | 10 | 2.00e-04 | 3.00e-16 |", old)
    assert P.stale("| d | x | 1 | 11 | 1.00e-04 | 3.00e-16 |", old)
    assert P.stale("| d | y | 1 | 10 | 1.00e-04 | 3.00e-16 |", old)


def test_committed_report_lists_every_deck_and_relaxation():
    text = P.PARITY_MD.read_text()
    for deck in RF1_DECKS:
        assert f"| {deck} |" in text
    for needle in ("d01013a CROD post", "t01241a", "d01161a", "rtol 0.05"):
        assert needle in text
