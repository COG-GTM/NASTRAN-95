import numpy as np
import pytest

from nastran95py.io import bits_to_int, bits_to_real, fortran_float, int_to_bits, real_to_bits


def test_real_bits_round_trip():
    for x in (0.0, 1.0, -2.5, 1.0e7, 3.0e-6):
        assert bits_to_real(real_to_bits(x)) == float(np.float32(x))
    assert real_to_bits(1.0) == 0x3F800000


def test_int_words():
    assert bits_to_int(int_to_bits(-7)) == -7
    with pytest.raises(OverflowError):
        int_to_bits(2**31)


def test_fortran_float():
    assert fortran_float(" 1.5D+02") == 150.0
    assert fortran_float("-2.0E-3") == -0.002
