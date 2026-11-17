from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

KERNELS_DIR = Path(__file__).parent / "fortran_kernels"
PARITY: list[tuple[str, int, float, float]] = []


def record_parity(name: str, n: int, max_abs: float, max_rel: float) -> None:
    PARITY.append((name, n, max_abs, max_rel))


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
    for name, n, max_abs, max_rel in PARITY:
        terminalreporter.write_line(f"{name:<34}{n:>8}{max_abs:>14.3e}{max_rel:>14.3e}")


@pytest.fixture(scope="session")
def manifest():
    from nastran95py.reference.inventory import load_manifest

    return load_manifest()


@pytest.fixture(scope="session")
def decks(manifest):
    return {d["id"]: d for d in manifest["decks"]}
