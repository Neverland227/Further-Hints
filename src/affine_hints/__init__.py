"""Exact-affine-hint diagnostics for bounded synthetic LWE experiments."""

from .coset import AffineCosetElimination
from .lwe import LWEInstance, generate_synthetic_lwe
from .priors import (
    CBDPrior,
    FixedWeightTernaryPrior,
    UniformTernaryPrior,
)

__all__ = [
    "AffineCosetElimination",
    "CBDPrior",
    "FixedWeightTernaryPrior",
    "LWEInstance",
    "UniformTernaryPrior",
    "generate_synthetic_lwe",
]

