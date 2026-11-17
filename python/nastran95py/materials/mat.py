"""MAT1 isotropic and MAT2 anisotropic material evaluation.

Port of the ``MAT`` entry point of ``mis/premat.f`` (structural branch).
The ``inflag`` codes follow the Fortran ``/MATIN/`` contract:

* ``1`` - MAT1 properties in MAT1 format (MAT1 only).
* ``2`` - MAT2 format: a 3x3 membrane ``G`` matrix, thermal expansion vector and
  allowables. MAT1 materials are converted to plane-stress ``G``; MAT2 materials
  are rotated by the material angle (``sinth``, ``costh``) as ``U^T G U``.
* ``3`` - transverse shear ``J11, J12, J22`` (``G`` for MAT1, zero for MAT2).
* ``4`` - density only.

As in the Fortran, MAT1 entries are searched before MAT2 entries with the same
ID.

Temperature dependence (``TEMPID != 0`` in the Fortran) is requested by passing
``temp``: every MAT1 property with a MATT1 table is evaluated at that temperature
through its TABLEM1-4 table before ``inflag`` processing (see
:mod:`nastran95py.materials.tables`). MATT2 (MAT2 temperature dependence) is not
ported, so MAT2 properties ignore ``temp``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from nastran95py.materials.tables import TableM

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
class Matt1:
    """MATT1 entry: TABLEMi IDs for the MAT1 properties of ``mid`` (0 = constant).

    Field order follows the MPT record; MATT1 has no TREF table.
    """

    mid: int
    e: int = 0
    g: int = 0
    nu: int = 0
    rho: int = 0
    alpha: int = 0
    ge: int = 0
    st: int = 0
    sc: int = 0
    ss: int = 0

    def table_ids(self) -> dict[str, int]:
        """Nonzero table references keyed by the MAT1 field name."""
        return {f.name: getattr(self, f.name) for f in fields(self)[1:] if getattr(self, f.name)}


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
    """In-memory material tables, the Python stand-in for ``PREMAT`` + ``MAT``.

    ``matt1`` and ``tables`` (TABLEM1-4) add temperature dependence to MAT1
    entries. As in ``PREMAT``, a MATT1 without a matching MAT1 (message 19) or a
    referenced table missing from ``tables`` (message 41) is rejected.
    """

    def __init__(
        self,
        mat1: list[Mat1] | None = None,
        mat2: list[Mat2] | None = None,
        matt1: list[Matt1] | None = None,
        tables: list[TableM] | None = None,
    ):
        self._mat1: dict[int, Mat1] = {}
        self._mat2: dict[int, Mat2] = {}
        self._matt1: dict[int, Matt1] = {}
        self._tables: dict[int, TableM] = {}
        for m in mat1 or []:
            self.add(m)
        for m in mat2 or []:
            self.add(m)
        for t in tables or []:
            self.add_table(t)
        for m in matt1 or []:
            self.add(m)

    def add(self, material: Mat1 | Mat2 | Matt1) -> None:
        if isinstance(material, Matt1):
            self._add_matt1(material)
            return
        table = self._mat1 if isinstance(material, Mat1) else self._mat2
        if material.mid in table:
            raise MaterialError(f"duplicate {type(material).__name__} id {material.mid}")
        table[material.mid] = material

    def add_table(self, table: TableM) -> None:
        if table.tid in self._tables:
            raise MaterialError(f"duplicate TABLEM id {table.tid}")
        self._tables[table.tid] = table

    def _add_matt1(self, matt1: Matt1) -> None:
        if matt1.mid not in self._mat1:
            raise MaterialError(f"MATT1 {matt1.mid} has no matching MAT1 (message 19)")
        if matt1.mid in self._matt1:
            raise MaterialError(f"duplicate MATT1 id {matt1.mid}")
        missing = sorted(set(matt1.table_ids().values()) - self._tables.keys())
        if missing:
            raise MaterialError(f"MATT1 {matt1.mid} references undefined tables {missing}")
        self._matt1[matt1.mid] = matt1

    def mat1_at(self, mid: int, temp: float | None) -> Mat1 | None:
        """MAT1 entry with its MATT1 properties evaluated at ``temp`` (``None``: as input)."""
        m = self._mat1.get(mid)
        if m is None or temp is None or mid not in self._matt1:
            return m
        values = {
            name: self._tables[tid].evaluate(temp, getattr(m, name))
            for name, tid in self._matt1[mid].table_ids().items()
        }
        return replace(m, **values)

    def mat1(self, mid: int, temp: float | None = None) -> Mat1Out:
        """``inflag = 1``: MAT1 properties in MAT1 format."""
        m = self._find1(mid, temp)
        if m is None:
            raise MaterialNotFoundError(mid)
        return Mat1Out(m.e, m.g, m.nu, m.rho, m.alpha, m.tref, m.ge, m.st, m.sc, m.ss)

    def mat2(
        self, mid: int, sinth: float = 0.0, costh: float = 1.0, temp: float | None = None
    ) -> Mat2Out:
        """``inflag = 2``: MAT2 format in element axes rotated by the material angle."""
        m1 = self._find1(mid, temp)
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

    def transverse_shear(self, mid: int, temp: float | None = None) -> tuple[float, float, float]:
        """``inflag = 3``: ``(J11, J12, J22)``."""
        m1 = self._find1(mid, temp)
        if m1 is not None:
            return (m1.g, 0.0, m1.g)
        if self._find2(mid) is None:
            raise MaterialNotFoundError(mid)
        return (0.0, 0.0, 0.0)

    def density(self, mid: int, temp: float | None = None) -> float:
        """``inflag = 4``: mass density."""
        m: Mat1 | Mat2 | None = self._find1(mid, temp) or self._find2(mid)
        if m is None:
            raise MaterialNotFoundError(mid)
        return m.rho

    def mat(
        self,
        mid: int,
        inflag: int,
        sinth: float = 0.0,
        costh: float = 1.0,
        temp: float | None = None,
    ) -> Mat1Out | Mat2Out | tuple[float, float, float] | float:
        """Dispatch on the Fortran ``inflag`` code (1-4); ``temp`` is ``/MATIN/`` TEMP."""
        if inflag == 1:
            return self.mat1(mid, temp)
        if inflag == 2:
            return self.mat2(mid, sinth, costh, temp)
        if inflag == 3:
            return self.transverse_shear(mid, temp)
        if inflag == 4:
            return self.density(mid, temp)
        raise MaterialError(f"inflag {inflag} is not ported")

    def _find1(self, mid: int, temp: float | None = None) -> Mat1 | None:
        return self.mat1_at(mid, temp)

    def _find2(self, mid: int) -> Mat2 | None:
        return self._mat2.get(mid)
