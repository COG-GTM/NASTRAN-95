"""Parity of BASGLB/GLBBAS/GBTRAN against the unmodified Fortran (mis/basglb.f).

The CSTM records come from random CORD2R/C/S definitions and from every CORD card the
demo decks in ../inp define, solved by ``build_cstm`` and rounded to single precision so
both sides see identical inputs.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from nastran95py.coords import (
    CoordSystem,
    CoordType,
    Cord2,
    Cstm,
    basglb,
    build_cstm,
    cords_from_cards,
    gbtran,
    glbbas,
    grids_from_cards,
)
from nastran95py.reference.cards import read_deck
from nastran95py.reference.inventory import repo_root

from .conftest import record_parity

pytestmark = pytest.mark.fortran

f32 = np.float32
RNG = np.random.default_rng(186)


def _round(cs: CoordSystem, cid: int) -> CoordSystem:
    return CoordSystem(cid, cs.ctype, f32(cs.origin).astype(float), f32(cs.t).astype(float))


def _systems() -> Cstm:
    systems = []
    for k, ctype in enumerate([1, 2, 3] * 4):
        a = RNG.normal(size=3)
        cord = Cord2(100 + k, CoordType(ctype), a, a + RNG.normal(size=3), a + RNG.normal(size=3))
        systems.append(_round(build_cstm([cord])[cord.cid], cord.cid))
    cid = 200
    for path in sorted((repo_root() / "inp").glob("*.inp")):
        text = path.read_text(errors="replace")
        if "\nCORD" not in text:
            continue
        try:
            cards = read_deck(text).bulk
            cstm = build_cstm(cords_from_cards(cards), grids_from_cards(cards))
        except ValueError:
            continue
        for cs in cstm.values():
            systems.append(_round(cs, cid))
            cid += 1
    return Cstm(systems)


def _fmt(values) -> str:
    return " ".join(f"{v:.9e}" if isinstance(v, float) else str(v) for v in values)


def _run(binary, cstm: Cstm, cases) -> list[np.ndarray]:
    lines = [str(len(cstm))]
    for cs in cstm.values():
        words = [*map(float, cs.origin), *map(float, cs.t.ravel())]
        lines.append(_fmt([cs.cid, int(cs.ctype), *words]))
    lines.append(str(len(cases)))
    for kind, cid, p, v in cases:
        lines.append(_fmt([kind, cid, *map(float, p), *map(float, v)]))
    out = subprocess.run(
        [str(binary)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True
    ).stdout
    return [np.array(line.split(), dtype=float) for line in out.splitlines() if line.strip()]


def _errors(name: str, got: np.ndarray, ref: np.ndarray, rtol: float) -> None:
    err = np.abs(got - ref)
    scale = np.maximum(np.abs(ref).max(axis=1, keepdims=True), 1e-30)
    max_abs, max_rel = float(err.max()), float((err / scale).max())
    record_parity(name, ref.size, max_abs, max_rel)
    assert max_rel < rtol, f"{name}: max rel err {max_rel:.3e}"


@pytest.fixture(scope="module")
def parity(fortran_kernels):
    binary = fortran_kernels.parent / "coords"
    cstm = _systems()
    cids = np.array([0, *cstm])
    n = 8 * len(cids)
    cid = np.repeat(cids, 8)
    pts = f32(RNG.uniform(-5, 5, (n, 3))).astype(float)
    vec = f32(RNG.normal(size=(n, 3))).astype(float)
    cases = [(1, c, p, v) for c, p, v in zip(cid, pts, vec, strict=True)]
    cases += [(2, c, p, v) for c, p, v in zip(cid, pts, vec, strict=True)]
    rect = np.array([c == 0 or cstm[c].ctype == CoordType.RECTANGULAR for c in cid])
    cases += [(3, c, p, v) for c, p, v in zip(cid[rect], pts[rect], vec[rect], strict=True)]
    eye = np.eye(3)
    curvilinear = zip(cid[~rect], pts[~rect], strict=True)
    cases += [(2, c, p, eye[j]) for c, p in curvilinear for j in range(3)]
    out = _run(binary, cstm, cases)
    assert len(out) == len(cases)
    return cstm, cid, pts, vec, rect, out


def test_basglb_matches_fortran(parity):
    cstm, cid, pts, vec, _, out = parity
    got = np.array(out[: len(cid)])
    _errors("BASGLB (basic->global)", got, basglb(cstm, cid, pts, vec), 1e-5)


def test_glbbas_matches_fortran(parity):
    cstm, cid, pts, vec, _, out = parity
    got = np.array(out[len(cid) : 2 * len(cid)])
    _errors("GLBBAS (global->basic)", got, glbbas(cstm, cid, pts, vec), 1e-5)


def test_gbtran_matches_fortran(parity):
    cstm, cid, pts, _, rect, out = parity
    n, nr = len(cid), int(rect.sum())
    ta = gbtran(cstm, cid, pts)
    got_rect = np.array(out[2 * n : 2 * n + nr])
    # Curvilinear matrices: column j of TA is GLBBAS applied to unit vector e_j.
    cols = np.array(out[2 * n + nr :]).reshape(-1, 3, 3)
    got = np.empty((n, 9))
    got[rect] = got_rect
    got[~rect] = np.transpose(cols, (0, 2, 1)).reshape(-1, 9)
    assert got.shape == (n, 9) and (~rect).any() and rect.any()
    _errors("GBTRAN (global->basic T)", got, ta.reshape(n, 9), 1e-5)
