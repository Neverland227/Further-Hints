"""B0 no-hint toy reference construction."""

from __future__ import annotations

from typing import Any

from affine_hints.candidates.mn_general import build_extended_basis, target_coefficients_and_vector


def build_no_hint_embedding(instance: Any) -> tuple[tuple[int, ...], ...]:
    """Return the standard ``(-e,s,-1)`` row embedding (MN Eq. (3) orientation)."""

    return build_extended_basis(A=instance.A, b=instance.b, H=(), ell=(), q=instance.q)


def no_hint_target_certificate(instance: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return exact coefficients and target for correctness checks only."""

    if instance.H:
        # Build a view without hints so the certificate dimensions match B0.
        class View:
            A = instance.A
            b = instance.b
            H = ()
            ell = ()
            q = instance.q
            m = instance.m
            n = instance.n
            s = instance.s
            e = instance.e

        return target_coefficients_and_vector(View())
    return target_coefficients_and_vector(instance)

