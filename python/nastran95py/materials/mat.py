"""MAT1 isotropic and MAT2 anisotropic material evaluation.

Port of the ``MAT`` entry point of ``mis/premat.f`` for temperature-independent
materials. The ``inflag`` codes follow the Fortran ``/MATIN/`` contract:

* ``1`` - MAT1 properties in MAT1 format (MAT1 only).
* ``2`` - MAT2 format: a 3x3 membrane ``G`` matrix, thermal expansion vector and
  allowables. MAT1 materials are converted to plane-stress ``G``; MAT2 materials
  are rotated by the material angle (``sinth``, ``costh``) as ``U^T G U``.
* ``3`` - transverse shear ``J11, J12, J22`` (``G`` for MAT1, zero for MAT2).
* ``4`` - density only.

As in the Fortran, MAT1 entries are searched before MAT2 entries with the same
ID. Temperature-dependent properties (MATT1/MATT2 + TABLEMi) are not ported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_ANGLE_TOL = 1.0e-4


class MaterialError(ValueError):
    """Invalid material request (Fortran fatal message from ``MAT``)."""


class MaterialNotFoundError(MaterialError, KeyError):
    """Material ID not present in the tables searched (Fortran message 42)."""


@dataclass(frozen=True)
class Mat1:
    """MAT1 bulk data entry as stored on the MPT (fields 2-12)."""

    mid: int
    e: float
    g: float
    nu: float
    rho: float = 0.0
    alpha: float = 0.0
    tref: float = 0.0
    ge: float = 0.0
    st: float = 0.0
    sc: float = 0.0
    ss: float = 0.0


@dataclass(frozen=True)
class Mat2:
    """MAT2 bulk data entry as stored on the MPT (fields 2-16)."""

    mid: int
    g11: float
    g12: float
    g13: float
    g22: float
    g23: float
    g33: float
    rho: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a12: float = 0.0
    tref: float = 0.0
    ge: float = 0.0
    st: float = 0.0
    sc: float = 0.0
    ss: float = 0.0

    @property
    def g(self) -> FloatArray:
        return np.array(
            [
                [self.g11, self.g12, self.g13],
                [self.g12, self.g22, self.g23],
                [self.g13, self.g23, self.g33],
            ]
        )


@dataclass(frozen=True)
class Mat1Out:
    """``/MATOUT/`` words 1-10 for ``inflag = 1``."""

    e: float
    g: float
    nu: float
    rho: float
    alpha: float
    tref: float
    ge: float
    sigt: float
    sigc: float
    sigs: float


@dataclass(frozen=True)
class Mat2Out:
    """``/MATOUT/`` words 1-15 for ``inflag = 2``.

    ``g`` is the symmetric 3x3 material matrix in element axes and ``alpha`` the
    thermal expansion vector ``(a1, a2, a12)``. ``source`` is ``"MAT1"`` or
    ``"MAT2"``.
    """

    g: FloatArray
    rho: float
    alpha: FloatArray
    tref: float
    ge: float
    sigt: float
    sigc: float
    sigs: float
    source: str

    def to_words(self) -> FloatArray:
        """Return the 15 ``/MATOUT/`` words in Fortran order."""
        g = self.g
        return np.array(
            [
                g[0, 0],
                g[0, 1],
                g[0, 2],
                g[1, 1],
                g[1, 2],
                g[2, 2],
                self.rho,
                *self.alpha,
                self.tref,
                self.ge,
                self.sigt,
                self.sigc,
                self.sigs,
            ]
        )


def rotation_matrix(sinth: float, costh: float) -> FloatArray:
    """Stress-strain rotation ``U`` used by ``MAT`` (row-major ``X(1..9)``)."""
    c2 = costh * costh
    s2 = sinth * sinth
    cs = costh * sinth
    return np.array(
        [
            [c2, s2, cs],
            [s2, c2, -cs],
            [-2.0 * cs, 2.0 * cs, c2 - s2],
        ]
    )


class MaterialTable:
    """In-memory MAT1/MAT2 tables, the Python stand-in for ``PREMAT`` + ``MAT``."""

    def __init__(self, mat1: list[Mat1] | None = None, mat2: list[Mat2] | None = None):
        self._mat1: dict[int, Mat1] = {}
        self._mat2: dict[int, Mat2] = {}
        for m in mat1 or []:
            self.add(m)
        for m in mat2 or []:
            self.add(m)

    def add(self, material: Mat1 | Mat2) -> None:
        table = self._mat1 if isinstance(material, Mat1) else self._mat2
        if material.mid in table:
            raise MaterialError(f"duplicate {type(material).__name__} id {material.mid}")
        table[material.mid] = material

    def mat1(self, mid: int) -> Mat1Out:
        """``inflag = 1``: MAT1 properties in MAT1 format."""
        m = self._find1(mid)
        if m is None:
            raise MaterialNotFoundError(mid)
        return Mat1Out(m.e, m.g, m.nu, m.rho, m.alpha, m.tref, m.ge, m.st, m.sc, m.ss)

    def mat2(self, mid: int, sinth: float = 0.0, costh: float = 1.0) -> Mat2Out:
        """``inflag = 2``: MAT2 format in element axes rotated by the material angle."""
        m1 = self._find1(mid)
        if m1 is not None:
            g11 = m1.e / (1.0 - m1.nu**2)
            g = np.array(
                [
                    [g11, m1.nu * g11, 0.0],
                    [m1.nu * g11, g11, 0.0],
                    [0.0, 0.0, m1.g],
                ]
            )
            alpha = np.array([m1.alpha, m1.alpha, 0.0])
            return Mat2Out(g, m1.rho, alpha, m1.tref, m1.ge, m1.st, m1.sc, m1.ss, "MAT1")

        m2 = self._find2(mid)
        if m2 is None:
            raise MaterialNotFoundError(mid)
        g = m2.g
        alpha = np.array([m2.a1, m2.a2, m2.a12])
        # Fortran skips the transform only when SINTH is exactly zero.
        if sinth != 0.0:
            if abs(sinth**2 + costh**2 - 1.0) > _ANGLE_TOL:
                raise MaterialError(
                    f"material {mid}: sin/cos of material angle not normalized "
                    f"(sinth={sinth}, costh={costh})"
                )
            u = rotation_matrix(sinth, costh)
            g = u.T @ g @ u
            u_alpha = u.copy()
            u_alpha[0, 2] = -u_alpha[0, 2]
            u_alpha[1, 2] = -u_alpha[1, 2]
            u_alpha[2, 0] = -u_alpha[2, 0]
            u_alpha[2, 1] = -u_alpha[2, 1]
            alpha = u_alpha @ alpha
        return Mat2Out(g, m2.rho, alpha, m2.tref, m2.ge, m2.st, m2.sc, m2.ss, "MAT2")

    def transverse_shear(self, mid: int) -> tuple[float, float, float]:
        """``inflag = 3``: ``(J11, J12, J22)``."""
        m1 = self._find1(mid)
        if m1 is not None:
            return (m1.g, 0.0, m1.g)
        if self._find2(mid) is None:
            raise MaterialNotFoundError(mid)
        return (0.0, 0.0, 0.0)

    def density(self, mid: int) -> float:
        """``inflag = 4``: mass density."""
        m: Mat1 | Mat2 | None = self._find1(mid) or self._find2(mid)
        if m is None:
            raise MaterialNotFoundError(mid)
        return m.rho

    def mat(
        self, mid: int, inflag: int, sinth: float = 0.0, costh: float = 1.0
    ) -> Mat1Out | Mat2Out | tuple[float, float, float] | float:
        """Dispatch on the Fortran ``inflag`` code (1-4)."""
        if inflag == 1:
            return self.mat1(mid)
        if inflag == 2:
            return self.mat2(mid, sinth, costh)
        if inflag == 3:
            return self.transverse_shear(mid)
        if inflag == 4:
            return self.density(mid)
        raise MaterialError(f"inflag {inflag} is not ported")

    def _find1(self, mid: int) -> Mat1 | None:
        return self._mat1.get(mid)

    def _find2(self, mid: int) -> Mat2 | None:
        return self._mat2.get(mid)
