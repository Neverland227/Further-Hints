from __future__ import annotations

from affine_hints.estimators.beta_sensitivity import beta_sensitivity_band
from affine_hints.estimators.list_capacity import candidate_capacity


def test_capacity_and_beta_sensitivity_have_correct_direction() -> None:
    capacity = candidate_capacity(q=97, r=3, prior="ternary")[0]
    assert capacity.bits > 0
    rows = beta_sensitivity_band(
        capacity_bits=capacity.bits,
        baseline_beta=60,
        dimensions=(("lattice_dimension", 129), ("bkz_beta", 60)),
    )
    assert all(row.predicted_delta_beta_integer >= 0 for row in rows)
    assert all(row.radius_ratio >= 1 for row in rows)

