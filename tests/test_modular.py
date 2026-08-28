from __future__ import annotations

import pytest

from affine_hints.modular import invert_matrix_mod, matmul_mod, unit_inverse


@pytest.mark.parametrize("q", [17, 31, 32, 256])
def test_matrix_inverse_over_prime_and_prime_power(q: int) -> None:
    matrix = [[1, 2], [0, 1]]
    inverse = invert_matrix_mod(matrix, q)
    assert matmul_mod(matrix, inverse, q) == [[1, 0], [0, 1]]


def test_nonunit_rejected() -> None:
    with pytest.raises(ZeroDivisionError):
        unit_inverse(2, 32)

