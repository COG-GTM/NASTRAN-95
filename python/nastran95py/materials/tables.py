"""TABLEM1-4 material property tables (port of the table look-up in ``mis/premat.f``).

``MAT`` reads the DIT tables referenced by MATT1 cards and evaluates them at the
element temperature ``T`` (statements 1080-1310 of ``premat.f``):

* ``TABLEM1`` - ``P(T) = f(T)``; the table value replaces the MAT1 property.
* ``TABLEM2`` - ``P(T) = P0 * f(T - X1)``.
* ``TABLEM3`` - ``P(T) = P0 * f((T - X1) / X2)``.
* ``TABLEM4`` - ``P(T) = P0 * sum(A_i * x**i)`` with ``x = (T - X1) / X2`` clamped to
  ``[(X3 - X1) / X2, (X4 - X1) / X2]``.

``P0`` is the MAT1 value. ``f`` is the Fortran piecewise-linear look-up: binary
search over increasing or decreasing abscissae, linear extrapolation from the end
segments, and the average of the two ordinates when the argument hits a repeated
abscissa (a step). The log-scale options of TABLEM1 are not honoured by ``MAT``
and are not ported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from nastran95py.materials.mat import MaterialError


def interpolate(x: Sequence[float], y: Sequence[float], arg: float) -> float:
    """Piecewise-linear look-up ``f(arg)`` exactly as ``premat.f`` statements 1080-1170."""
    n = len(x)
    up = -1.0 if x[0] > x[1] else 1.0
    if (arg - x[0]) * up <= 0.0:
        return _linear(x, y, 0, arg)
    if (arg - x[n - 1]) * up >= 0.0:
        return _linear(x, y, n - 2, arg)
    klo, khi = 1, n
    while True:
        kx = (klo + khi + 1) // 2
        kxx = kx - 1
        d = (arg - x[kxx]) * up
        if d == 0.0:
            return _on_point(x, y, kxx, arg)
        if d < 0.0:
            khi = kx
        else:
            klo = kx
        if khi - klo == 1:
            break
    k1 = klo - 1
    if kxx != k1 and arg == x[k1 + 1]:
        return _on_point(x, y, k1 + 1, arg)
    return _linear(x, y, k1, arg)


def _linear(x: Sequence[float], y: Sequence[float], k: int, arg: float) -> float:
    return (arg - x[k]) * (y[k + 1] - y[k]) / (x[k + 1] - x[k]) + y[k]


def _on_point(x: Sequence[float], y: Sequence[float], k: int, arg: float) -> float:
    # A repeated abscissa is a step: return the mean of the two ordinates. When the
    # search lands on the first point of a step, the Fortran computes this mean at
    # statement 1170 but falls through into 1180 and repeats the same search forever.
    if arg == x[k - 1]:
        return (y[k - 1] + y[k]) / 2.0
    if arg == x[k + 1]:
        return (y[k] + y[k + 1]) / 2.0
    return y[k]


def _check_points(tid: int, x: Sequence[float], y: Sequence[float]) -> None:
    if len(x) != len(y):
        raise MaterialError(f"table {tid}: {len(x)} abscissae but {len(y)} ordinates")
    if len(x) < 2:
        raise MaterialError(f"table {tid}: at least two points are required")
    if x[0] == x[1] or x[-1] == x[-2]:
        raise MaterialError(f"table {tid}: first and last segments need distinct abscissae")
    sign = 1.0 if x[1] > x[0] else -1.0
    if any((b - a) * sign < 0.0 for a, b in zip(x, x[1:], strict=False)):
        raise MaterialError(f"table {tid}: abscissae must be monotonic")


@dataclass(frozen=True)
class TableM1:
    """TABLEM1: ``P(T) = f(T)``."""

    tid: int
    x: tuple[float, ...]
    y: tuple[float, ...]
    type_code: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", tuple(float(v) for v in self.x))
        object.__setattr__(self, "y", tuple(float(v) for v in self.y))
        _check_points(self.tid, self.x, self.y)

    def evaluate(self, temp: float, base: float) -> float:
        return interpolate(self.x, self.y, temp)


@dataclass(frozen=True)
class TableM2:
    """TABLEM2: ``P(T) = P0 * f(T - X1)``."""

    tid: int
    x1: float
    x: tuple[float, ...]
    y: tuple[float, ...]
    type_code: int = field(default=2, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", tuple(float(v) for v in self.x))
        object.__setattr__(self, "y", tuple(float(v) for v in self.y))
        _check_points(self.tid, self.x, self.y)

    def evaluate(self, temp: float, base: float) -> float:
        return base * interpolate(self.x, self.y, temp - self.x1)


@dataclass(frozen=True)
class TableM3:
    """TABLEM3: ``P(T) = P0 * f((T - X1) / X2)``."""

    tid: int
    x1: float
    x2: float
    x: tuple[float, ...]
    y: tuple[float, ...]
    type_code: int = field(default=3, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", tuple(float(v) for v in self.x))
        object.__setattr__(self, "y", tuple(float(v) for v in self.y))
        if self.x2 == 0.0:
            raise MaterialError(f"TABLEM3 {self.tid}: X2 must be nonzero")
        _check_points(self.tid, self.x, self.y)

    def evaluate(self, temp: float, base: float) -> float:
        return base * interpolate(self.x, self.y, (temp - self.x1) / self.x2)


@dataclass(frozen=True)
class TableM4:
    """TABLEM4: ``P(T) = P0 * sum(a[i] * x**i)``, ``x = (T - X1)/X2`` clamped to [X3, X4]."""

    tid: int
    x1: float
    x2: float
    x3: float
    x4: float
    a: tuple[float, ...]
    type_code: int = field(default=4, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", tuple(float(v) for v in self.a))
        if self.x2 == 0.0:
            raise MaterialError(f"TABLEM4 {self.tid}: X2 must be nonzero")
        if not self.a:
            raise MaterialError(f"TABLEM4 {self.tid}: at least one coefficient is required")

    def polynomial(self, x: float) -> float:
        """Horner evaluation, highest coefficient first (statements 1280-1290)."""
        p = self.a[-1]
        for coef in reversed(self.a[:-1]):
            p = p * x + coef
        return p

    def evaluate(self, temp: float, base: float) -> float:
        x = (temp - self.x1) / self.x2
        lo = (self.x3 - self.x1) / self.x2
        hi = (self.x4 - self.x1) / self.x2
        if x <= lo:
            p = self.polynomial(lo)
        elif x >= hi:
            p = self.polynomial(hi)
        else:
            p = self.polynomial(x)
        return base * p


TableM = TableM1 | TableM2 | TableM3 | TableM4
