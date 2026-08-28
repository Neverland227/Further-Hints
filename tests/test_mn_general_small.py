from __future__ import annotations

import numpy as np
import pytest
from sympy import Matrix

from affine_hints.candidates.mn_general import (
    CertifiedTopKUnavailable,
    MNConstrainedTopKCandidateSource,
    MNGeneralCandidateSource,
    _extract_unique_residual_candidates,
    bounded_cvp_schnorr_euchner,
    bounded_schnorr_euchner,
    build_extended_basis,
    canonical_b2_score_squared,
    constrained_candidates_within_radius,
    construct_constrained_slice_lattice,
    construct_modular_sublattice,
    multiply_row_combination,
    target_coefficients_and_vector,
)
from affine_hints.coset import AffineCosetElimination
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


def _small_full_elimination_instance(seed: int = 812, *, q: int = 5):
    rng = np.random.default_rng(seed)
    hint = generate_hint_matrix(rng, n=3, r=1, q=q, hint_class="systematic_random")
    instance = generate_synthetic_lwe(
        rng,
        n=3,
        m=3,
        q=q,
        secret_prior=UniformTernaryPrior(),
        H=hint.H,
    )
    elimination = AffineCosetElimination.build(
        instance.H,
        instance.ell,
        instance.q,
        1,
        strategy="first_unit_minor",
        rng=rng,
    )
    return instance, elimination


def test_constrained_slice_basis_preserves_every_hint_without_truth_oracle() -> None:
    instance, elimination = _small_full_elimination_instance()
    basis, offset, metadata = construct_constrained_slice_lattice(instance, elimination)
    assert len(basis) == instance.m + instance.n
    assert len(offset) == instance.m + instance.n
    assert metadata["fixed_embedding_coordinate"] == -1
    assert metadata["truth_norm_used_for_radius"] is False
    for coefficients in ((0,) * len(basis), tuple((index % 3) - 1 for index in range(len(basis)))):
        point = [
            offset[column]
            + sum(coefficients[row] * basis[row][column] for row in range(len(basis)))
            for column in range(len(offset))
        ]
        secret = point[instance.m :]
        assert all(
            sum(hint[column] * secret[column] for column in range(instance.n)) % instance.q
            == target
            for hint, target in zip(instance.H, instance.ell)
        )
        assert all(
            (sum(instance.A[row][column] * secret[column] for column in range(instance.n))
             - instance.b[row]
             - point[row])
            % instance.q
            == 0
            for row in range(instance.m)
        )


def test_canonical_b2_score_of_truth_matches_full_secret_error_norm() -> None:
    instance, elimination = _small_full_elimination_instance()
    truth = tuple(instance.s[index] % instance.q for index in elimination.residual_indices)
    score, full_secret, error = canonical_b2_score_squared(instance, elimination, truth)
    assert full_secret == instance.s
    assert tuple(abs(value) for value in error) == tuple(abs(value) for value in instance.e)
    assert score == 1 + sum(value * value for value in instance.s) + sum(
        value * value for value in instance.e
    )


def test_constrained_slice_spans_every_canonical_residual_representative() -> None:
    instance, elimination = _small_full_elimination_instance(seed=913)
    basis, offset, _ = construct_constrained_slice_lattice(instance, elimination)
    basis_matrix = Matrix(basis)
    basis_inverse = basis_matrix.T.inv()
    assert abs(int(basis_matrix.det())) == instance.q ** (
        instance.m + len(instance.H)
    )
    for left in range(instance.q):
        for right in range(instance.q):
            residual = (left, right)
            _, full_secret, error = canonical_b2_score_squared(
                instance,
                elimination,
                residual,
            )
            canonical_point = tuple(error) + tuple(full_secret)
            difference = Matrix(
                [canonical_point[index] - offset[index] for index in range(len(offset))]
            )
            coefficients = basis_inverse * difference
            assert all(value.q == 1 for value in coefficients)


def test_constrained_cvp_enumerator_fixes_one_embedding_sign() -> None:
    pytest.importorskip("fpylll")
    vectors, metadata = bounded_cvp_schnorr_euchner(
        [[2, 0], [0, 2]],
        offset=[1, 0],
        radius_squared=2,
        max_slice_vectors=20,
        node_limit=100,
    )
    assert set(vectors) == {(1, 0, -1), (-1, 0, -1)}
    assert all(vector[-1] == -1 for vector in vectors)
    assert metadata["enumeration_complete_within_bounds"] is True


def test_constrained_raw_vector_cap_marks_ball_incomplete() -> None:
    pytest.importorskip("fpylll")
    _, metadata = bounded_cvp_schnorr_euchner(
        [[1, 0], [0, 1]],
        offset=[0, 0],
        radius_squared=3,
        max_slice_vectors=1,
        node_limit=100,
    )
    assert metadata["enumeration_slice_vector_limit_reached"] is True
    assert metadata["enumeration_complete_within_bounds"] is False


def test_constrained_slice_matches_bruteforce_residual_ball() -> None:
    pytest.importorskip("fpylll")
    instance, elimination = _small_full_elimination_instance(seed=991)
    radius_squared = 25
    constrained = constrained_candidates_within_radius(
        instance,
        elimination,
        radius_squared=radius_squared,
        beta=0,
        beta_max=30,
        reduction_seed=991,
        node_limit=100_000,
        max_slice_vectors=10_000,
    )
    assert constrained.metadata["enumeration_complete_within_bounds"] is True
    expected = set()
    for left in range(instance.q):
        for right in range(instance.q):
            residual = (left, right)
            score, _, _ = canonical_b2_score_squared(instance, elimination, residual)
            if score <= radius_squared:
                expected.add(residual)
    assert set(constrained.candidates) == expected


def test_constrained_slice_equals_complete_legacy_embedding_ball() -> None:
    pytest.importorskip("fpylll")
    instance, elimination = _small_full_elimination_instance(seed=151, q=17)
    legacy = MNGeneralCandidateSource()
    legacy.prepare(
        instance,
        {
            "elimination": elimination,
            "max_n": 80,
            "scaling_c": 16.0,
            "beta": 0,
            "beta_max": 30,
            "radius_multiplier": 1.5,
            "enumeration_node_limit": 100_000,
            "reduction_seed": 151,
        },
    )
    legacy_list = legacy.generate(10_000, np.random.default_rng(151))
    assert legacy_list.metadata["enumeration_complete_within_bounds"] is True
    constrained = constrained_candidates_within_radius(
        instance,
        elimination,
        radius_squared=float(legacy_list.metadata["enumeration_radius_squared"]),
        beta=0,
        beta_max=30,
        reduction_seed=151,
        node_limit=100_000,
        max_slice_vectors=10_000,
    )
    assert constrained.metadata["enumeration_complete_within_bounds"] is True
    assert set(constrained.candidates) == set(legacy_list.candidates)
    assert dict(zip(constrained.candidates, constrained.scores)) == dict(
        zip(legacy_list.candidates, legacy_list.scores)
    )


def test_certified_top_k_is_deterministic_and_uses_no_truth_radius() -> None:
    pytest.importorskip("fpylll")
    instance, elimination = _small_full_elimination_instance(seed=1201)
    config = {
        "elimination": elimination,
        "max_n": 80,
        "beta": 0,
        "beta_max": 30,
        "reduction_seed": 1201,
        "top_k_values": [2, 4, 8],
        "score_radius_squared_grid": [5, 10, 15, 20, 25, 30, 40, 50],
        "max_radius_steps": 8,
        "max_slice_vectors_per_radius": 10_000,
        "enumeration_node_limit": 100_000,
    }
    outputs = []
    for _ in range(2):
        source = MNConstrainedTopKCandidateSource()
        source.prepare(instance, config)
        outputs.append(source.generate(8, np.random.default_rng(1201)))
    assert outputs[0].candidates == outputs[1].candidates
    assert outputs[0].scores == outputs[1].scores
    assert len(outputs[0].candidates) == 8
    assert outputs[0].metadata["top_k_certified"] is True
    assert outputs[0].metadata["radius_selection_uses_true_norm"] is False


def test_certified_top_k_never_returns_a_resource_capped_prefix() -> None:
    pytest.importorskip("fpylll")
    instance, elimination = _small_full_elimination_instance(seed=1277)
    source = MNConstrainedTopKCandidateSource()
    source.prepare(
        instance,
        {
            "elimination": elimination,
            "max_n": 80,
            "beta": 0,
            "beta_max": 30,
            "reduction_seed": 1277,
            "top_k_values": [2],
            "score_radius_squared_grid": [50],
            "max_radius_steps": 1,
            "max_slice_vectors_per_radius": 1,
            "enumeration_node_limit": 100_000,
        },
    )
    with pytest.raises(CertifiedTopKUnavailable, match="censored") as captured:
        source.generate(2, np.random.default_rng(1277))
    assert captured.value.metadata["top_k_certified"] is False
