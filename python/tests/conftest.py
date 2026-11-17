from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

KERNELS_DIR = Path(__file__).parent / "fortran_kernels"
PARITY: list[dict] = []
OUTCOMES: list[dict] = []


def record_parity(name: str, n: int, max_abs: float, max_rel: float, **tags) -> None:
    """Add a row to the "Fortran parity" summary (and ``--parity-json``).

    ``tags`` feed ``tests/parity_report.py`` (PARITY.md): ``deck``/``table`` for a
    listing table, ``element``/``quantity``/``source`` for Python vs the Fortran
    element harness (``source`` "golden" for the checked-in dumps, "live" for this
    machine's build).
    """
    PARITY.append(
        {"name": name, "n": int(n), "max_abs": float(max_abs), "max_rel": float(max_rel), **tags}
    )


def harness_source(label: str) -> str:
    return "golden" if "golden" in label and "live" not in label else "live"


def pytest_addoption(parser):
    parser.addoption(
        "--parity-json",
        metavar="PATH",
        help="write the parity rows and test outcomes to PATH (input of tests/parity_report.py)",
    )


def pytest_runtest_logreport(report):
    if report.when != "call" and report.passed:
        return
    outcome = report.outcome
    if hasattr(report, "wasxfail"):
        outcome = "xfailed" if report.skipped else "xpassed"
    OUTCOMES.append(
        {"nodeid": report.nodeid, "when": report.when, "outcome": outcome,
         "reason": getattr(report, "wasxfail", "")}
    )  # fmt: skip


def pytest_sessionfinish(session, exitstatus):
    path = session.config.getoption("--parity-json", default=None)
    if not path:
        return
    data = {"exitstatus": int(exitstatus), "records": PARITY, "outcomes": OUTCOMES}
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


@pytest.fixture(scope="session")
def fortran_kernels() -> Path:
    """Build (if needed) and return the shared-kernel Fortran harness binary.

    Skips when gfortran is unavailable unless NASTRAN95PY_REQUIRE_FORTRAN=1,
    which CI sets so a missing toolchain fails instead of skipping.
    """
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    subprocess.run(["make", "-s", "-C", str(KERNELS_DIR)], check=True)
    return KERNELS_DIR / "build" / "kernels"


def pytest_terminal_summary(terminalreporter):
    if not PARITY:
        return
    terminalreporter.section("Fortran parity (Python vs mis/*.f)")
    header = f"{'kernel':<34}{'values':>8}{'max abs err':>14}{'max rel err':>14}"
    terminalreporter.write_line(header)
    for r in PARITY:
        terminalreporter.write_line(
            f"{r['name']:<34}{r['n']:>8}{r['max_abs']:>14.3e}{r['max_rel']:>14.3e}"
        )


@pytest.fixture(scope="session")
def manifest():
    from nastran95py.reference.inventory import load_manifest

    return load_manifest()


@pytest.fixture(scope="session")
def decks(manifest):
    return {d["id"]: d for d in manifest["decks"]}
