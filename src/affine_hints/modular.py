"""Exact modular arithmetic for row-oriented affine hints.

All matrices use Python integers. Public LWE matrices follow the convention
``A: m x n`` and hint matrices follow ``H: r x n``.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence


class NoUnitMinorError(ValueError):
    """Raised when the requested unit minor does not exist."""


def as_int_matrix(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    """Copy a rectangular matrix into Python ``int`` storage."""

    rows = [[int(value) for value in row] for row in matrix]
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix is not rectangular")
    return rows


def mod_vector(vector: Iterable[int], q: int) -> list[int]:
    """Return canonical representatives in ``{0, ..., q-1}``."""

    return [int(value) % q for value in vector]


def centered_lift(value: int, q: int) -> int:
    """Return the centered representative, using ``[-q/2, q/2)`` for even q."""

    residue = int(value) % q
    return residue - q if 2 * residue >= q else residue


def centered_vector(vector: Iterable[int], q: int) -> list[int]:
    """Center-lift every coordinate modulo ``q``."""

    return [centered_lift(value, q) for value in vector]


def unit_inverse(value: int, q: int) -> int:
    """Invert a unit in ``Z/qZ``."""

    value %= q
    if math.gcd(value, q) != 1:
        raise ZeroDivisionError(f"{value} is not a unit modulo {q}")
    return pow(value, -1, q)


def matmul_mod(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]], q: int) -> list[list[int]]:
    """Multiply two small integer matrices exactly modulo ``q``."""

    a = as_int_matrix(left)
    b = as_int_matrix(right)
    if not a:
        return []
    if not b:
        return [[] for _ in a]
    if len(a[0]) != len(b):
        raise ValueError("incompatible matrix dimensions")
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) % q for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def matvec_mod(matrix: Sequence[Sequence[int]], vector: Sequence[int], q: int) -> list[int]:
    """Multiply a small integer matrix by a column vector modulo ``q``."""

    rows = as_int_matrix(matrix)
    values = [int(value) for value in vector]
    if rows and len(rows[0]) != len(values):
        raise ValueError("incompatible matrix/vector dimensions")
    return [sum(a * b for a, b in zip(row, values)) % q for row in rows]


def determinant_bareiss(matrix: Sequence[Sequence[int]]) -> int:
    """Compute an exact determinant using fraction-free Bareiss elimination."""

    a = as_int_matrix(matrix)
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("determinant requires a square matrix")
    if n == 0:
        return 1
    sign = 1
    previous = 1
    for k in range(n - 1):
        pivot = next((i for i in range(k, n) if a[i][k] != 0), None)
        if pivot is None:
            return 0
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            sign *= -1
        pivot_value = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = a[i][j] * pivot_value - a[i][k] * a[k][j]
                a[i][j] = numerator // previous
        previous = pivot_value
        for i in range(k + 1, n):
            a[i][k] = 0
    return sign * a[-1][-1]


def invert_matrix_mod(matrix: Sequence[Sequence[int]], q: int) -> list[list[int]]:
    """Invert a square matrix whose determinant is a unit modulo ``q``."""

    a = as_int_matrix(matrix)
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("inverse requires a square matrix")
    augmented = [[value % q for value in row] + [int(i == j) for j in range(n)] for i, row in enumerate(a)]
    for column in range(n):
        pivot = next((i for i in range(column, n) if math.gcd(augmented[i][column], q) == 1), None)
        if pivot is None:
            # A unit determinant guarantees a unit pivot after suitable row
            # combinations, but a raw row may not expose it. Try adding rows.
            for i in range(column, n):
                for j in range(column, n):
                    if i != j and math.gcd((augmented[i][column] + augmented[j][column]) % q, q) == 1:
                        augmented[i] = [(x + y) % q for x, y in zip(augmented[i], augmented[j])]
                        pivot = i
                        break
                if pivot is not None:
                    break
        if pivot is None:
            raise NoUnitMinorError(f"matrix is not invertible over Z/{q}Z")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = unit_inverse(augmented[column][column], q)
        augmented[column] = [(scale * value) % q for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column] % q
            if factor:
                augmented[row] = [
                    (value - factor * pivot_value) % q
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [row[n:] for row in augmented]


def first_prime_divisor(q: int) -> int:
    """Return the smallest prime divisor of ``q``."""

    if q < 2:
        raise ValueError("modulus must be at least 2")
    divisor = 2
    while divisor * divisor <= q:
        if q % divisor == 0:
            return divisor
        divisor += 1
    return q


def rank_mod_prime(matrix: Sequence[Sequence[int]], p: int) -> int:
    """Compute matrix rank over the prime field ``F_p``."""

    a = [[value % p for value in row] for row in as_int_matrix(matrix)]
    if not a:
        return 0
    row = 0
    for column in range(len(a[0])):
        pivot = next((i for i in range(row, len(a)) if a[i][column] % p), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        inverse = pow(a[row][column], -1, p)
        a[row] = [(inverse * value) % p for value in a[row]]
        for i in range(len(a)):
            if i != row and a[i][column] % p:
                factor = a[i][column]
                a[i] = [(x - factor * y) % p for x, y in zip(a[i], a[row])]
        row += 1
        if row == len(a):
            break
    return row


def find_unit_minor(
    matrix: Sequence[Sequence[int]],
    q: int,
    size: int,
    *,
    row_order: Sequence[int] | None = None,
    column_order: Sequence[int] | None = None,
    max_combinations: int = 200_000,
) -> tuple[list[int], list[int]]:
    """Find a ``size`` square minor with unit determinant modulo ``q``.

    This exhaustive bounded search is intended for ``r <= 5`` correctness and
    toy experiments. It never silently widens its search budget.
    """

    h = as_int_matrix(matrix)
    if size == 0:
        return [], []
    rows = list(row_order) if row_order is not None else list(range(len(h)))
    columns = list(column_order) if column_order is not None else list(range(len(h[0]) if h else 0))
    checked = 0
    for row_indices in itertools.combinations(rows, size):
        for column_indices in itertools.combinations(columns, size):
            checked += 1
            if checked > max_combinations:
                raise NoUnitMinorError("unit-minor search exceeded max_combinations")
            minor = [[h[i][j] for j in column_indices] for i in row_indices]
            if math.gcd(determinant_bareiss(minor), q) == 1:
                return list(row_indices), list(column_indices)
    raise NoUnitMinorError(f"no unit {size}x{size} minor modulo {q}")


def matrix_rank_unit(matrix: Sequence[Sequence[int]], q: int) -> int:
    """Return the largest size of a unit minor, bounded by the row count."""

    h = as_int_matrix(matrix)
    for size in range(min(len(h), len(h[0]) if h else 0), -1, -1):
        try:
            find_unit_minor(h, q, size)
            return size
        except NoUnitMinorError:
            continue
    return 0

