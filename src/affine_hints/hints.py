"""Synthetic hint-matrix distributions and coded-dual structure diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .modular import first_prime_divisor, matrix_rank_unit, rank_mod_prime


@dataclass(frozen=True)
class HintMatrix:
    """A synthetic hint matrix together with generation metadata."""

    H: tuple[tuple[int, ...], ...]
    hint_class: str
    metadata: dict[str, Any]


def _statistics(matrix: list[list[int]], q: int) -> dict[str, Any]:
    if not matrix:
        return {"rank": 0, "row_weights": [], "column_weights": []}
    row_weights = [sum(value % q != 0 for value in row) for row in matrix]
    column_weights = [sum(row[j] % q != 0 for row in matrix) for j in range(len(matrix[0]))]
    p = first_prime_divisor(q)
    return {
        "unit_rank": matrix_rank_unit(matrix, q),
        "rank_mod_smallest_prime": rank_mod_prime(matrix, p),
        "row_weight_min": min(row_weights),
        "row_weight_mean": float(np.mean(row_weights)),
        "row_weight_max": max(row_weights),
        "column_weight_min": min(column_weights),
        "column_weight_mean": float(np.mean(column_weights)),
        "column_weight_max": max(column_weights),
    }


def _ensure_unit_rank(matrix: list[list[int]], q: int, r: int) -> None:
    if matrix_rank_unit(matrix, q) < r:
        raise ValueError(f"generated hint matrix lacks a unit {r}x{r} minor")


def _kronecker(left: list[list[int]], right: list[list[int]], q: int) -> list[list[int]]:
    return [
        [left[i][j] * right[u][v] % q for j in range(len(left[0])) for v in range(len(right[0]))]
        for i in range(len(left))
        for u in range(len(right))
    ]


def coded_dual_g_transpose(
    *, rng: np.random.Generator, n: int, r: int, q: int, alpha: int = 1, puncturing_rule: str = "prefix"
) -> HintMatrix:
    """Construct ``H = (P K F_I)^T`` from the local paper's Lemma 3.

    The default prefix row-selection is a project diagnostic, not a claim that it
    reproduces Carrier's finite-length polar-code puncturing schedule.
    """

    if math.gcd(alpha, q) != 1:
        raise ValueError("alpha must be a unit modulo q")
    mother = 1
    while mother < n:
        mother *= 2
    kernel = [[1, 1], [alpha % q, 0]]
    transform = [[1]]
    while len(transform) < mother:
        transform = _kronecker(transform, kernel, q)
    if puncturing_rule == "prefix":
        retained = list(range(n))
    elif puncturing_rule == "random_rows":
        retained = sorted(int(value) for value in rng.choice(mother, size=n, replace=False))
    else:
        raise ValueError(f"unsupported puncturing rule: {puncturing_rule}")
    punctured = [[transform[i][j] for j in range(mother)] for i in retained]
    p = first_prime_divisor(q)
    chosen: list[int] = []
    current_rank = 0
    column_order = list(range(mother))
    # A seeded shuffle probes more than one valid information set while keeping
    # every result reproducible.
    rng.shuffle(column_order)
    for column in column_order:
        trial = chosen + [column]
        submatrix = [[row[j] for j in trial] for row in punctured]
        rank = rank_mod_prime(submatrix, p)
        if rank > current_rank:
            chosen.append(column)
            current_rank = rank
        if len(chosen) == r:
            break
    if len(chosen) < r:
        raise ValueError("unable to select a full-rank information set")
    generator = [[row[j] % q for j in chosen] for row in punctured]
    hints = [[generator[i][j] for i in range(n)] for j in range(r)]
    _ensure_unit_rank(hints, q, r)
    metadata = {
        "construction": "G=P K F_I; H=G^T",
        "source_label": "LITERATURE_EXACT algebra / PROJECT_DIAGNOSTIC puncturing distribution",
        "mother_length": mother,
        "puncturing_rule": puncturing_rule,
        "retained_rows": retained,
        "information_set": chosen,
        "alpha": alpha,
        "unit_minor_check": True,
    }
    metadata.update(_statistics(hints, q))
    return HintMatrix(tuple(tuple(row) for row in hints), "coded_dual_G_transpose", metadata)


def generate_hint_matrix(
    rng: np.random.Generator,
    *,
    n: int,
    r: int,
    q: int,
    hint_class: str,
    parameters: dict[str, Any] | None = None,
) -> HintMatrix:
    """Generate one bounded synthetic full-unit-rank hint matrix."""

    parameters = parameters or {}
    if r < 0 or r > n:
        raise ValueError("r must be in 0..n")
    if r == 0:
        return HintMatrix((), hint_class, {"unit_rank": 0})
    name = hint_class.lower()
    if name == "coded_dual_g_transpose":
        return coded_dual_g_transpose(
            rng=rng,
            n=n,
            r=r,
            q=q,
            alpha=int(parameters.get("alpha", 1)),
            puncturing_rule=str(parameters.get("puncturing_rule", "prefix")),
        )
    if name == "dense_random":
        for attempt in range(int(parameters.get("max_attempts", 100))):
            matrix = rng.integers(0, q, size=(r, n), dtype=np.int64).tolist()
            if matrix_rank_unit(matrix, q) == r:
                break
        else:
            raise ValueError("failed to generate a dense unit-rank matrix")
        metadata = {"attempt": attempt + 1}
    elif name == "systematic_random":
        tail = rng.integers(0, q, size=(r, n - r), dtype=np.int64).tolist()
        matrix = [[int(i == j) for j in range(r)] + tail[i] for i in range(r)]
        metadata = {"form": "[I | R]"}
    elif name == "row_sparse":
        weight = int(parameters.get("row_weight", 3))
        if weight not in (3, 5, 8) or weight > n:
            raise ValueError("row_sparse weight must be one of 3,5,8 and at most n")
        matrix = [[0] * n for _ in range(r)]
        for i in range(r):
            matrix[i][i] = 1
            choices = [j for j in range(n) if j != i]
            selected = rng.choice(choices, size=weight - 1, replace=False)
            for j in selected:
                value = 0
                while value == 0:
                    value = int(rng.integers(1, q))
                matrix[i][int(j)] = value
        metadata = {"row_weight_requested": weight, "systematic_unit_anchor": True}
    elif name == "banded":
        bandwidth = int(parameters.get("bandwidth", 3))
        matrix = [[0] * n for _ in range(r)]
        for i in range(r):
            matrix[i][i] = 1
            for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
                if j != i:
                    matrix[i][j] = int(rng.integers(0, q))
        metadata = {"bandwidth": bandwidth, "systematic_unit_anchor": True}
    elif name == "block_local":
        block_size = int(parameters.get("block_size", max(r, 8)))
        matrix = [[0] * n for _ in range(r)]
        for i in range(r):
            matrix[i][i] = 1
            start = (i // max(1, block_size)) * block_size
            for j in range(start, min(start + block_size, n)):
                if j != i:
                    matrix[i][j] = int(rng.integers(0, q))
        metadata = {"block_size": block_size, "systematic_unit_anchor": True}
    else:
        raise ValueError(f"unknown hint class: {hint_class}")
    _ensure_unit_rank(matrix, q, r)
    metadata.update(_statistics(matrix, q))
    metadata["source_label"] = "PROJECT_DIAGNOSTIC"
    return HintMatrix(tuple(tuple(int(value) % q for value in row) for row in matrix), hint_class, metadata)

