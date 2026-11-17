"""Element-level Fortran harness: K, M and stress dumps from the unmodified mis/ routines.

The harness (tests/fortran_harness) is the machine-precision ground truth the
Python element kernels are compared against. These tests check that it builds,
that every RF1 element type produces K and stress output, that the output is
physically right (symmetry, rigid-body modes, closed forms, constant-strain
patch), that the single- and double-precision Fortran paths agree, and that it
reproduces the checked-in golden file.
"""

from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np
import pytest

from tests.conftest import record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import runner as R

pytestmark = pytest.mark.fortran

ELEMENTS = ("CROD", "CONROD", "CSHEAR", "CTRMEM", "CQDMEM", "CQDMEM1", "CQDMEM2")


@pytest.fixture(scope="module")
def results() -> dict[str, R.CaseResult]:
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    return {r.case.name: r for r in R.run()}


def _err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    return max_abs, (max_abs / scale if scale else max_abs)


def _dofs_per_grid(r: R.CaseResult) -> int:
    if r.case.kind == C.SMA1:
        return 6
    k = next(m for m in r.matrices if m["file"] == 1)
    return k["ndof"] // len(r.case.grids)


def _rigid_modes(grids, ndof):
    """Six rigid-body vectors (3 translations, 3 rotations about the origin)."""
    modes = []
    for axis in range(3):
        t = np.zeros(3)
        t[axis] = 1.0
        modes.append([*t, 0, 0, 0])
    for axis in range(3):
        w = np.zeros(3)
        w[axis] = 1.0
        modes.append(w)
    vecs = []
    for i, m in enumerate(modes):
        v = []
        for xyz in grids:
            g = list(m) if i < 3 else [*np.cross(m, np.asarray(xyz, float)), *m]
            v += g[:ndof]
        vecs.append(np.array(v))
    return vecs


def test_every_element_type_dumps_k_and_stress(results):
    for elem in ELEMENTS:
        mine = [r for r in results.values() if r.case.element == elem]
        k_cases = [r for r in mine if r.case.kind in (C.EMG, C.SMA1)]
        s_cases = [r for r in mine if r.case.kind == C.SDR2]
        assert k_cases and s_cases, elem
        for r in k_cases:
            assert not r.nogo, r.case.name
            k = r.stiffness()
            assert np.all(np.isfinite(k)) and np.abs(k).max() > 0, r.case.name
            if r.case.kind == C.EMG:
                assert any(m["file"] == 2 for m in r.matrices), f"{r.case.name}: no mass"
        for r in s_cases:
            stress = r.words["STRESS"]
            assert R.real_word_as_int(stress[0]) == r.case.words[0]
            assert np.abs(stress[1:]).max() > 0, r.case.name
            if r.words["FORCE"].size:
                assert R.real_word_as_int(r.words["FORCE"][0]) == r.case.words[0]


@pytest.mark.parametrize("name", [c.name for c in C.ALL_CASES if c.kind != C.SDR2])
def test_stiffness_symmetric_with_rigid_body_modes(results, name):
    r = results[name]
    k = r.stiffness()
    single = r.case.kind == C.EMG and r.matrices[0]["prec"] == 1
    tol = 2e-6 if single else 1e-10
    scale = np.abs(k).max()
    assert np.abs(k - k.T).max() <= tol * scale
    ndof = _dofs_per_grid(r)
    for i, v in enumerate(_rigid_modes(r.case.grids, ndof)):
        # The shear panel is not exactly rotation-free when its plane is
        # oblique to the basic axes (observed residual ~6e-7 in SHEARD too).
        rtol = 1e-6 if r.case.element == "CSHEAR" and i >= 3 else 10 * tol
        assert np.abs(k @ v).max() <= rtol * scale * np.abs(v).max(), (name, i)


def _f32(*values):
    """The values as the REAL words the Fortran actually receives."""
    return [float(np.float32(v)) for v in values]


def _rod_k(case):
    (a, b) = (np.asarray(_f32(*g), float) for g in case.grids)
    e, g = _f32(*C.MAT1_BY_ID[case.meta["mid"]][:2])
    length = np.linalg.norm(b - a)
    n = (b - a) / length
    nn = np.outer(n, n)
    k = np.zeros((12, 12))
    area, j = _f32(case.meta["area"], case.meta["j"])
    for blk, kk in ((0, e * area / length), (3, g * j / length)):
        for i, j, s in ((0, 0, 1), (6, 6, 1), (0, 6, -1), (6, 0, -1)):
            k[i + blk : i + blk + 3, j + blk : j + blk + 3] = s * kk * nn
    return k


@pytest.mark.parametrize("elem", ["CROD", "CONROD"])
def test_rod_stiffness_and_mass_closed_form(results, elem):
    worst = {"RODS": (0.0, 0.0), "RODD": (0.0, 0.0), "KROD": (0.0, 0.0)}
    count = dict.fromkeys(worst, 0)
    for r in results.values():
        if r.case.element != elem or r.case.kind == C.SDR2:
            continue
        exact = _rod_k(r.case)
        k = r.stiffness()
        if k.shape[0] == 6:  # J = 0: EMG drops the rotations
            exact = exact[np.ix_([0, 1, 2, 6, 7, 8], [0, 1, 2, 6, 7, 8])]
        err = _err(k, exact)
        worst[r.case.routine] = tuple(map(max, worst[r.case.routine], err))
        count[r.case.routine] += k.size
        # Skew rods keep ~3e-9: part of the double path uses REAL intermediates.
        assert err[1] < (1e-6 if r.case.routine == "RODS" else 1e-8), r.case.name
        if r.case.kind == C.EMG:
            m = r.mass()
            rho = C.MAT1_BY_ID[r.case.meta["mid"]][3]
            length = math.dist(*r.case.grids)
            total = (rho * r.case.meta["area"] + r.case.meta["nsm"]) * length
            assert m.sum() == pytest.approx(3 * total, rel=1e-6)
    for routine, (a, rel) in worst.items():
        record_parity(f"{elem} {routine} K vs AE/L,GJ/L", count[routine], a, rel)


@pytest.mark.parametrize(
    "single,double,tol",
    # SHEARS loses ~5 digits to cancellation on the oblique (skew3d) panel.
    [
        ("RODS", "RODD", 2e-6),
        ("SHEARS", "SHEARD", 1e-4),
        ("TRMEMS", "TRMEMD", 2e-6),
        ("QDMM1S", "QDMM1D", 2e-6),
        ("QDMM2S", "QDMM2D", 2e-6),
    ],
)
def test_single_vs_double_precision_paths_agree(results, single, double, tol):
    worst_k, worst_m, n = (0.0, 0.0), (0.0, 0.0), 0
    for name, r in results.items():
        if r.case.routine != single:
            continue
        d = results[name.replace(single, double)]
        for ms, md in zip(r.matrices, d.matrices, strict=True):
            assert [ms[k] for k in ("file", "dict2", "ndof", "lbuf")] == [
                md[k] for k in ("file", "dict2", "ndof", "lbuf")
            ]
            err = _err(ms["values"], md["values"])
            assert err[1] < tol, name
            n += ms["lbuf"]
            if ms["file"] == 1:
                worst_k = tuple(map(max, worst_k, err))
            else:
                worst_m = tuple(map(max, worst_m, err))
    assert n
    record_parity(f"{single} vs {double} K", n, *worst_k)
    record_parity(f"{single} vs {double} M", n, *worst_m)


def _translations(k, ngrid):
    idx = [6 * g + i for g in range(ngrid) for i in range(3)]
    return k[np.ix_(idx, idx)]


def test_sma1_paths_match_emg_double(results):
    pairs = [(n, n.replace("-KROD", "-RODD-m-1")) for n in results if n.endswith("-KROD")]
    pairs += [(n, n.replace("-KTRMEM", "-TRMEMD")) for n in results if n.endswith("-KTRMEM")]
    worst = (0.0, 0.0)
    for sma1, emg in pairs:
        ks = results[sma1].stiffness()
        ke = results[emg].stiffness()
        if ke.shape[0] != ks.shape[0]:
            ks = _translations(ks, len(results[sma1].case.grids))
        err = _err(ks, ke)
        assert err[1] < 1e-6, sma1
        worst = tuple(map(max, worst, err))
    record_parity("SMA1 (KROD/KTRMEM) vs EMG D", len(pairs), *worst)


def test_ktriqd_membrane_is_kqdmem(results):
    for name, r in results.items():
        if r.case.routine == "KTRIQD3":
            np.testing.assert_array_equal(
                r.stiffness(), results[name.replace("KTRIQD3", "KQDMEM")].stiffness()
            )


def test_rod_stress_closed_form(results):
    worst = (0.0, 0.0)
    for r in results.values():
        if r.case.routine != "SROD1":
            continue
        e, g = C.MAT1_BY_ID[r.case.meta["mid"]][:2]
        length = math.dist(*r.case.grids)
        area, j, c = (r.case.meta[k] for k in ("area", "j", "c"))
        sigma = e * 2.0e-3 / length
        torque = g * j * 1.0e-3 / length
        tau = c / j * torque if j else 0.0
        got = [r.words["STRESS"][1], r.words["STRESS"][3], *r.words["FORCE"][1:3]]
        err = _err(got, [sigma, tau, area * sigma, torque])
        assert err[1] < 1e-6, r.case.name
        worst = tuple(map(max, worst, err))
    record_parity("SROD1/SROD2 vs closed form", 4 * len(C.ROD_GEOMS) * 2, *worst)


def test_shear_panel_pure_shear_stress(results):
    r = results["CSHEAR-rect-SPANL"]
    g = C.MAT1_BY_ID[1][1]
    assert r.words["STRESS"][1] == pytest.approx(g * 1.0e-4, rel=1e-6)


def _principal(sx, sy, txy):
    c, rad = 0.5 * (sx + sy), math.hypot(0.5 * (sx - sy), txy)
    return c + rad, c - rad, rad


MEMBRANE_STRESS = [
    c.name
    for c in C.ALL_CASES
    if c.kind == C.SDR2 and c.element not in ("CROD", "CONROD", "CSHEAR")
]


def _element_axes(grids):
    g = np.asarray(grids, float)
    n = np.cross(g[2] - g[0], g[3] - g[1]) if len(g) == 4 else np.cross(g[1] - g[0], g[2] - g[0])
    n /= np.linalg.norm(n)
    x = g[1] - g[0]
    x -= x.dot(n) * n
    x /= np.linalg.norm(x)
    return x, np.cross(n, x)


def _rotate_stress(s, deg):
    c, sn = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array(
        [
            c * c * s[0] + sn * sn * s[1] + 2 * c * sn * s[2],
            sn * sn * s[0] + c * c * s[1] - 2 * c * sn * s[2],
            (s[1] - s[0]) * c * sn + (c * c - sn * sn) * s[2],
        ]
    )


def _plane_stress_d(mid, theta):
    if mid in C.MAT1_BY_ID:
        e, _g, nu = C.MAT1_BY_ID[mid][:3]
        d = e / (1 - nu * nu)
        return np.array([[d, d * nu, 0], [d * nu, d, 0], [0, 0, d * (1 - nu) / 2]])
    g = dict(C.MAT2)[mid]
    d = np.array([[g[0], g[1], g[2]], [g[1], g[3], g[4]], [g[2], g[4], g[5]]])
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    t = np.array(
        [[c * c, s * s, c * s], [s * s, c * c, -c * s], [-2 * c * s, 2 * c * s, c * c - s * s]]
    )
    return t.T @ d @ t


def _patch_stress(case):
    """Plane stress of the uniform basic-frame STRAIN in the element frame (x along G1-G2)."""
    ex, ey, gxy = C.STRAIN.values()
    eps = np.array([[ex, gxy / 2, 0.0], [gxy / 2, ey, 0.0], [0.0, 0.0, 0.0]])
    x, y = _element_axes(case.grids)
    strain = np.array([x @ eps @ x, y @ eps @ y, 2 * x @ eps @ y])
    theta = case.meta.get("theta", 0.0)
    stress = _plane_stress_d(case.meta["mid"], theta) @ strain
    # SQDM22 reports CQDMEM2 stresses in the material axes.
    return _rotate_stress(stress, theta) if case.element == "CQDMEM2" else stress


def test_membrane_constant_strain_patch(results):
    worst = (0.0, 0.0)
    for name in MEMBRANE_STRESS:
        r = results[name]
        sx, sy, txy, _angle, s1, s2, _smax = r.words["STRESS"][1:8]
        p1, p2, _rad = _principal(sx, sy, txy)
        assert (s1, s2) == pytest.approx((p1, p2), rel=1e-5, abs=1e-3), name
        err = _err([sx, sy, txy], _patch_stress(r.case))
        assert err[1] < 1e-6, (name, err)
        worst = tuple(map(max, worst, err))
    record_parity("membrane stress vs D*strain patch", 3 * len(MEMBRANE_STRESS), *worst)


def test_membrane_routines_agree_on_principal_stresses(results):
    worst, n = (0.0, 0.0), 0
    for tag in C.QUAD_GEOMS:
        ref = results[f"CQDMEM-{tag}-SQDME"].words["STRESS"][5:7]
        for elem in ("CQDMEM1", "CQDMEM2"):
            got = results[f"{elem}-{tag}-SQDM"].words["STRESS"][5:7]
            err = _err(got, ref)
            assert err[1] < 1e-5, (elem, tag)
            worst, n = tuple(map(max, worst, err)), n + 2
    record_parity("SQDM12/SQDM22 vs SQDME1 principal", n, *worst)


def test_harness_reproduces_golden(results):
    golden = json.loads(R.GOLDEN.read_text())
    current = R.to_golden(list(results.values()))
    assert set(golden) == set(current)
    worst, n = (0.0, 0.0), 0
    for name, entry in current.items():
        ref = golden[name]
        pairs = []
        for m, mr in zip(entry.get("matrices", []), ref.get("matrices", []), strict=True):
            pairs.append((m["values"], mr["values"]))
        pairs += [(v, ref["parts"][k]) for k, v in entry.get("parts", {}).items()]
        pairs += [(v, ref["words"][k]) for k, v in entry.get("words", {}).items()]
        for got, exp in pairs:
            err = _err(got, exp)
            assert err[1] < 1e-6, name
            worst, n = tuple(map(max, worst, err)), n + len(got)
    record_parity("harness vs golden/harness_golden.json", n, *worst)
