"""Single-precision word handling for mixed integer/real Fortran records.

NASTRAN tables such as the EST, ECPT and CSTM are arrays of 32-bit words where
integers and reals share storage through EQUIVALENCE. These helpers convert
between Python values and the raw 32-bit patterns (as signed int32).
"""

from __future__ import annotations

import numpy as np


def real_to_bits(value: float) -> int:
    """Return the IEEE single-precision bit pattern of ``value`` as a signed int32."""
    return int(np.array([value], dtype=np.float32).view(np.int32)[0])


def bits_to_real(word: int) -> float:
    """Interpret a signed int32 bit pattern as a single-precision real."""
    return float(np.array([word], dtype=np.int32).view(np.float32)[0])


def int_to_bits(value: int) -> int:
    """Return an INTEGER word as a signed int32 (range checked)."""
    word = int(value)
    if not -(2**31) <= word < 2**31:
        raise OverflowError(f"{value} does not fit in a 32-bit INTEGER word")
    return word


def bits_to_int(word: int) -> int:
    """Interpret a raw word as an INTEGER."""
    return int_to_bits(word)


def fortran_float(text: str) -> float:
    """Parse a Fortran-formatted real, accepting ``D`` exponents (``1.5D+02``)."""
    return float(text.strip().upper().replace("D", "E"))
