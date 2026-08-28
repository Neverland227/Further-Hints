"""Bounded unit-minor selection and activation-depth diagnostics."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .modular import determinant_bareiss, invert_matrix_mod, matmul_mod


@dataclass(frozen=True)
class PivotChoice:
    """A unit minor and its induced affine-coupling diagnostics."""

    rows: tuple[int, ...]
    columns: tuple[int, ...]
    fill: int
    mean_last_nonzero: float
    max_last_nonzero: int
    examined_pairs: int = 0
    total_pairs: int = 0
    search_truncated: bool = False


def _choice_metrics(
    H: Sequence[Sequence[int]], q: int, rows: Sequence[int], columns: Sequence[int]
) -> PivotChoice:
    residual = [j for j in range(len(H[0])) if j not in columns]
    minor = [[int(H[i][j]) for j in columns] for i in rows]
    right = [[int(H[i][j]) for j in residual] for i in rows]
    coupling = matmul_mod(invert_matrix_mod(minor, q), right, q)
    fill = sum(value % q != 0 for row in coupling for value in row)
    last_positions = [max((j + 1 for j, value in enumerate(row) if value % q), default=0) for row in coupling]
    return PivotChoice(
        tuple(rows),
        tuple(columns),
        fill,
        float(sum(last_positions) / len(last_positions)) if last_positions else 0.0,
        max(last_positions, default=0),
    )


def choose_unit_minor(
    H: Sequence[Sequence[int]],
    q: int,
    r_elim: int,
    strategy: str,
    rng: np.random.Generator,
    *,
    max_combinations: int = 100_000,
) -> PivotChoice:
    """Select a unit minor with a finite, preregistered search budget."""

    rows_total = len(H)
    columns_total = len(H[0]) if rows_total else 0
    if r_elim == 0:
        return PivotChoice((), (), 0, 0.0, 0, 0, 1, False)
    if strategy not in ("first_unit_minor", "random_unit_minor", "min_fill", "min_activation_depth"):
        raise ValueError(f"unknown pivot strategy: {strategy}")
    total_pairs = math.comb(rows_total, r_elim) * math.comb(columns_total, r_elim)
    if strategy == "random_unit_minor" and total_pairs > max_combinations:
        sampled: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        maximum_attempts = max_combinations * 20
        for _ in range(maximum_attempts):
            rows = tuple(sorted(int(value) for value in rng.choice(rows_total, size=r_elim, replace=False)))
            columns = tuple(sorted(int(value) for value in rng.choice(columns_total, size=r_elim, replace=False)))
            sampled.add((rows, columns))
            if len(sampled) >= max_combinations:
                break
        pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]] = list(sampled)
        rng.shuffle(pairs)
    else:
        iterator = itertools.product(
            itertools.combinations(range(rows_total), r_elim),
            itertools.combinations(range(columns_total), r_elim),
        )
        pairs = list(itertools.islice(iterator, min(total_pairs, max_combinations))) if strategy == "random_unit_minor" else iterator
        if strategy == "random_unit_minor":
            rng.shuffle(pairs)
    best: PivotChoice | None = None
    examined = 0
    for rows, columns in pairs:
        if examined >= max_combinations:
            break
        examined += 1
        minor = [[int(H[i][j]) for j in columns] for i in rows]
        if math.gcd(determinant_bareiss(minor), q) == 1:
            choice = _choice_metrics(H, q, rows, columns)
            if strategy in ("first_unit_minor", "random_unit_minor"):
                return replace(
                    choice,
                    examined_pairs=examined,
                    total_pairs=total_pairs,
                    search_truncated=False,
                )
            if strategy == "min_fill":
                key = lambda value: (value.fill, value.max_last_nonzero, value.columns, value.rows)
            else:
                key = lambda value: (value.mean_last_nonzero, value.max_last_nonzero, value.fill, value.columns, value.rows)
            if best is None or key(choice) < key(best):
                best = choice
    if best is None:
        raise ValueError(f"no unit {r_elim}x{r_elim} minor found")
    return replace(
        best,
        examined_pairs=examined,
        total_pairs=total_pairs,
        search_truncated=total_pairs > examined,
    )
