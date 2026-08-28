"""Information diagnostics for rank-only candidate capacity.

None of the functions in this module is a measured attack speedup. Results are
explicitly tagged as either ``RANK_ONLY_REFERENCE`` or
``INFORMATION_DIAGNOSTIC``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from affine_hints.priors import CBDPrior, FixedWeightTernaryPrior, UniformTernaryPrior


@dataclass(frozen=True)
class CapacityResult:
    """One bounded analytic candidate-capacity calculation."""

    prior: str
    metric: str
    bits: float
    label: str
    assumptions: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fixed_weight_conditional_capacity(
    *,
    q: int,
    r_elim: int,
    h_plus: int,
    h_minus: int,
    residual_plus: int,
    residual_minus: int,
) -> CapacityResult:
    """Exact conditional count ``-log2(D_I / q^r)`` for a residual profile."""

    required_plus = h_plus - residual_plus
    required_minus = h_minus - residual_minus
    required_zero = r_elim - required_plus - required_minus
    if min(required_plus, required_minus, required_zero) < 0:
        count = 0
    else:
        count = math.comb(r_elim, required_plus) * math.comb(r_elim - required_plus, required_minus)
    bits = math.inf if count == 0 else math.log2((q**r_elim) / count)
    return CapacityResult(
        prior=f"fixed_weight(+{h_plus},-{h_minus})",
        metric="exact_residual_profile",
        bits=bits,
        label="RANK_ONLY_REFERENCE",
        assumptions="uniform q-ary pivot image and the stated residual plus/minus counts",
    )


def same_second_moment_entropy_proxy(q: int) -> CapacityResult:
    """Return the required, narrowly labelled ternary-vs-Gaussian proxy."""

    gaussian_entropy = 0.5 * math.log2(2.0 * math.pi * math.e * (2.0 / 3.0))
    difference = gaussian_entropy - math.log2(3.0)
    return CapacityResult(
        prior="uniform_ternary",
        metric="same_second_moment_entropy_proxy",
        bits=difference,
        label="INFORMATION_DIAGNOSTIC",
        assumptions=f"same second moment only; q={q} does not turn this into an attack theorem",
    )


def candidate_capacity(
    *,
    q: int,
    r: int,
    prior: str,
    fixed_weight: dict[str, int] | None = None,
) -> list[CapacityResult]:
    """Compute the preregistered rank/support/entropy diagnostics."""

    name = prior.lower()
    if name in ("ternary", "uniform_ternary"):
        return [
            CapacityResult(
                prior="uniform_ternary",
                metric="hard_support_selectivity",
                bits=r * math.log2(q / 3.0),
                label="RANK_ONLY_REFERENCE",
                assumptions="candidate residual is independent of dense H and uniform in the relevant mod-q image",
            ),
            same_second_moment_entropy_proxy(q),
        ]
    if name in ("cbd2", "cbd3"):
        eta = int(name[-1])
        cbd = CBDPrior(eta)
        return [
            CapacityResult(
                prior=name,
                metric="support_selectivity",
                bits=r * math.log2(q / len(cbd.support)),
                label="INFORMATION_DIAGNOSTIC",
                assumptions="support-only diagnostic, not weighted CBD likelihood",
            ),
            CapacityResult(
                prior=name,
                metric="shannon_deficit",
                bits=r * (math.log2(q) - cbd.entropy_bits()),
                label="INFORMATION_DIAGNOSTIC",
                assumptions="coordinatewise entropy diagnostic",
            ),
            CapacityResult(
                prior=name,
                metric="renyi2_separation",
                bits=r * (math.log2(q) - cbd.collision_entropy_bits()),
                label="INFORMATION_DIAGNOSTIC",
                assumptions="coordinatewise collision-entropy diagnostic",
            ),
        ]
    if name == "fixed_weight":
        if not fixed_weight:
            raise ValueError("fixed_weight parameters are required")
        n = int(fixed_weight["n"])
        h_plus = int(fixed_weight["h_plus"])
        h_minus = int(fixed_weight["h_minus"])
        prior_object = FixedWeightTernaryPrior(h_plus, h_minus)
        global_bits = n * math.log2(q) - prior_object.entropy_bits_for_length(n)
        expected_residual_plus = round(h_plus * (n - r) / n)
        expected_residual_minus = round(h_minus * (n - r) / n)
        conditional = fixed_weight_conditional_capacity(
            q=q,
            r_elim=r,
            h_plus=h_plus,
            h_minus=h_minus,
            residual_plus=expected_residual_plus,
            residual_minus=expected_residual_minus,
        )
        return [
            CapacityResult(
                prior=f"fixed_weight(+{h_plus},-{h_minus})",
                metric="global_combinatorial_deficit",
                bits=global_bits,
                label="INFORMATION_DIAGNOSTIC",
                assumptions="full-n exact multinomial count; not assigned wholesale to r hints",
            ),
            conditional,
        ]
    raise ValueError(f"unknown prior: {prior}")

