from __future__ import annotations

import numpy as np

from affine_hints.diagnostics import candidate_correlation_diagnostics, projective_class, uniformity_diagnostics


def test_projective_normalization_and_clustering() -> None:
    q = 17
    assert projective_class((2, 4, 0), q) == (1, 2, 0)
    assert projective_class((4, 8, 0), q) == (1, 2, 0)
    diagnostics = candidate_correlation_diagnostics([(2, 4), (4, 8), (1, 0)], [True, False, False], q)
    assert diagnostics["number_projective_classes"] == 2
    assert diagnostics["largest_class_size"] == 2
    assert diagnostics["fraction_collinear_pairs"] > 0
    assert diagnostics["collinear_joint_pass_rate"] == 0.0
    assert diagnostics["independence_joint_pass_reference"] == (1.0 / 3.0) ** 2


def test_zero_dimensional_uniformity_is_not_applicable() -> None:
    result = uniformity_diagnostics([(), ()], 17, np.random.default_rng(1))
    assert result["applicable"] is False
    assert result["dimension"] == 0
