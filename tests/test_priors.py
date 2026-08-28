from __future__ import annotations

import math

import numpy as np

from affine_hints.priors import CBDPrior, FixedWeightTernaryPrior, UniformTernaryPrior


def test_uniform_ternary_interface() -> None:
    prior = UniformTernaryPrior()
    assert prior.in_support((-1, 0, 1))
    assert not prior.in_support((-2, 0, 1))
    assert len(list(prior.enumerate_support(3, 27))) == 27
    assert math.isclose(prior.second_moment(), 2 / 3)


def test_cbd_exact_pmf_and_sampling() -> None:
    prior = CBDPrior(2)
    assert math.isclose(sum(prior.coordinate_prob(x) for x in prior.support), 1.0)
    assert math.isclose(prior.coordinate_prob(0), 6 / 16)
    sample = prior.sample(np.random.default_rng(1), 100)
    assert prior.in_support(sample)


def test_fixed_weight_tracks_sign_counts() -> None:
    prior = FixedWeightTernaryPrior(2, 1)
    assert prior.in_support((1, 1, -1, 0, 0))
    assert not prior.in_support((1, -1, -1, 0, 0))
    support = list(prior.enumerate_support(5, 1000))
    assert len(support) == math.comb(5, 2) * math.comb(3, 1)

