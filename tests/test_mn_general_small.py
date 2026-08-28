from __future__ import annotations

import numpy as np
import pytest

from affine_hints.candidates.mn_general import (
    _extract_unique_residual_candidates,
    bounded_schnorr_euchner,
    build_extended_basis,
    construct_modular_sublattice,
    multiply_row_combination,
    target_coefficients_and_vector,
)
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.priors import UniformTernaryPrior


def test_mn_equation20_basis_contains_target_vector() -> None:
    rng = np.random.default_rng(15)
    hint = generate_hint_matrix(rng, n=5, r=2, q=17, hint_class="systematic_random")
    instance = generate_synthetic_lwe(rng, n=5, m=6, q=17, secret_prior=UniformTernaryPrior(), H=hint.H)
    basis = build_extended_basis(A=instance.A, b=instance.b, H=instance.H, ell=instance.ell, q=17)
    coefficients, target = target_coefficients_and_vector(instance)
    assert multiply_row_combination(coefficients, basis) == target
    assert len(basis) == instance.m + len(instance.H) + instance.n + 1


def test_project_enumerator_enforces_node_bound_and_integer_norm() -> None:
    pytest.importorskip("fpylll")
    vectors, metadata = bounded_schnorr_euchner(
        [[1, 0], [0, 1]], radius_squared=2.0, max_solutions=20, node_limit=100
    )
    assert vectors
    assert all(sum(value * value for value in vector) <= 2 for vector in vectors)
    assert metadata["enumeration_nodes"] <= 100
    assert metadata["enumeration_complete_within_bounds"] is True


def test_project_enumerator_reports_hard_node_truncation() -> None:
    pytest.importorskip("fpylll")
    _, metadata = bounded_schnorr_euchner(
        [[1, 0], [0, 1]], radius_squared=100.0, max_solutions=1000, node_limit=2
    )
    assert metadata["enumeration_node_limit_reached"] is True
    assert metadata["enumeration_nodes"] == 2


def test_project_enumerator_treats_solution_budget_as_incomplete() -> None:
    pytest.importorskip("fpylll")
    _, metadata = bounded_schnorr_euchner(
        [[1, 0], [0, 1]], radius_squared=2.0, max_solutions=1, node_limit=100
    )
    assert metadata["enumeration_solution_budget_reached"] is True
    assert metadata["enumeration_complete_within_bounds"] is False


def test_embedding_sign_pair_is_one_residual_candidate() -> None:
    candidates, scores, metadata = _extract_unique_residual_candidates(
        [
            (0, 1, -1, -1),
            (0, -1, 1, 1),
            (0, 0, 1, -1),
        ],
        m=1,
        n=2,
        q=17,
        residual_indices=(0, 1),
    )
    assert candidates == [(1, 16), (0, 1)]
    assert scores == [-3.0, -2.0]
    assert metadata == {
        "eligible_embedding_vector_count": 3,
        "duplicate_residual_count": 1,
        "unique_candidate_count": 2,
    }


def test_mn_algorithm1_adaptation_produces_verified_square_basis() -> None:
    pytest.importorskip("fpylll")
    rng = np.random.default_rng(151)
    hint = generate_hint_matrix(rng, n=3, r=1, q=17, hint_class="systematic_random")
    instance = generate_synthetic_lwe(
        rng,
        n=3,
        m=4,
        q=17,
        secret_prior=UniformTernaryPrior(),
        H=hint.H,
    )
    basis, metadata = construct_modular_sublattice(
        A=instance.A,
        b=instance.b,
        H=instance.H,
        ell=instance.ell,
        q=instance.q,
        scaling_c=16.0,
    )
    assert len(basis) == instance.m + instance.n + 1
    assert all(len(row) == len(basis) for row in basis)
    assert metadata["zero_block_verified"] is True
