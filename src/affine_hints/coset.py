"""Exact partial elimination for affine/mod-q hints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .modular import invert_matrix_mod, matmul_mod, matvec_mod
from .pivoting import PivotChoice, choose_unit_minor


@dataclass(frozen=True)
class AffineCosetElimination:
    """Represent ``s_I = c - C s_J`` and the remaining modular checks."""

    q: int
    H: tuple[tuple[int, ...], ...]
    ell: tuple[int, ...]
    eliminated_rows: tuple[int, ...]
    remaining_rows: tuple[int, ...]
    pivot_indices: tuple[int, ...]
    residual_indices: tuple[int, ...]
    c: tuple[int, ...]
    C: tuple[tuple[int, ...], ...]
    H_tilde: tuple[tuple[int, ...], ...]
    ell_tilde: tuple[int, ...]
    pivot_choice: PivotChoice

    @classmethod
    def build(
        cls,
        H: Sequence[Sequence[int]],
        ell: Sequence[int],
        q: int,
        r_elim: int,
        *,
        strategy: str = "first_unit_minor",
        rng: np.random.Generator | None = None,
        max_combinations: int = 100_000,
    ) -> "AffineCosetElimination":
        """Build the exact affine substitution from a selected unit minor."""

        hints = [[int(value) % q for value in row] for row in H]
        targets = [int(value) % q for value in ell]
        if len(hints) != len(targets):
            raise ValueError("H and ell have incompatible dimensions")
        if hints and any(len(row) != len(hints[0]) for row in hints):
            raise ValueError("H is not rectangular")
        if not 0 <= r_elim <= len(hints):
            raise ValueError("r_elim is outside 0..rank(H)")
        generator = rng if rng is not None else np.random.default_rng(0)
        choice = choose_unit_minor(hints, q, r_elim, strategy, generator, max_combinations=max_combinations)
        rows = list(choice.rows)
        pivots = list(choice.columns)
        n = len(hints[0]) if hints else 0
        residual = [j for j in range(n) if j not in pivots]
        remaining_rows = [i for i in range(len(hints)) if i not in rows]
        if r_elim:
            minor = [[hints[i][j] for j in pivots] for i in rows]
            inverse = invert_matrix_mod(minor, q)
            c = matvec_mod(inverse, [targets[i] for i in rows], q)
            C = matmul_mod(inverse, [[hints[i][j] for j in residual] for i in rows], q)
        else:
            c = []
            C = []
        h_tilde: list[list[int]] = []
        ell_tilde: list[int] = []
        for i in remaining_rows:
            row_j = [hints[i][j] for j in residual]
            if r_elim:
                row_i = [[hints[i][j] for j in pivots]]
                correction = matmul_mod(row_i, C, q)[0]
                constant = matvec_mod(row_i, c, q)[0]
                h_tilde.append([(x - y) % q for x, y in zip(row_j, correction)])
                ell_tilde.append((targets[i] - constant) % q)
            else:
                h_tilde.append(row_j)
                ell_tilde.append(targets[i])
        return cls(
            q=q,
            H=tuple(tuple(row) for row in hints),
            ell=tuple(targets),
            eliminated_rows=tuple(rows),
            remaining_rows=tuple(remaining_rows),
            pivot_indices=tuple(pivots),
            residual_indices=tuple(residual),
            c=tuple(c),
            C=tuple(tuple(row) for row in C),
            H_tilde=tuple(tuple(row) for row in h_tilde),
            ell_tilde=tuple(ell_tilde),
            pivot_choice=choice,
        )

    def reconstruct(self, residual_secret: Sequence[int]) -> tuple[int, ...]:
        """Reconstruct the full secret in original coordinate order."""

        x = [int(value) % self.q for value in residual_secret]
        if len(x) != len(self.residual_indices):
            raise ValueError("residual secret has the wrong dimension")
        if self.pivot_indices:
            product = matvec_mod(self.C, x, self.q)
            pivot = [(constant - value) % self.q for constant, value in zip(self.c, product)]
        else:
            pivot = []
        full = [0] * (len(self.pivot_indices) + len(self.residual_indices))
        for index, value in zip(self.pivot_indices, pivot):
            full[index] = value
        for index, value in zip(self.residual_indices, x):
            full[index] = value
        return tuple(full)

    def remaining_hints_pass(self, residual_secret: Sequence[int]) -> bool:
        """Check only the hint rows not consumed by elimination."""

        return tuple(matvec_mod(self.H_tilde, residual_secret, self.q)) == self.ell_tilde

    def all_hints_pass(self, residual_secret: Sequence[int]) -> bool:
        """Check the original full hint system after reconstruction."""

        return tuple(matvec_mod(self.H, self.reconstruct(residual_secret), self.q)) == self.ell

    def transform_lwe(
        self, A: Sequence[Sequence[int]], b: Sequence[int]
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
        """Return ``A*`` and ``b*`` with unchanged error, from the task equation."""

        matrix = [[int(value) % self.q for value in row] for row in A]
        targets = [int(value) % self.q for value in b]
        if len(matrix) != len(targets):
            raise ValueError("A and b have incompatible dimensions")
        if matrix and len(matrix[0]) != len(self.pivot_indices) + len(self.residual_indices):
            raise ValueError("A has the wrong secret dimension")
        A_star: list[tuple[int, ...]] = []
        b_star: list[int] = []
        for row, target in zip(matrix, targets):
            row_i = [[row[j] for j in self.pivot_indices]]
            row_j = [row[j] for j in self.residual_indices]
            correction = matmul_mod(row_i, self.C, self.q)[0] if self.pivot_indices else [0] * len(row_j)
            constant = matvec_mod(row_i, self.c, self.q)[0] if self.pivot_indices else 0
            A_star.append(tuple((x - y) % self.q for x, y in zip(row_j, correction)))
            b_star.append((target - constant) % self.q)
        return tuple(A_star), tuple(b_star)

