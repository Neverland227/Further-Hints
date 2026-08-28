"""Bounded exact top-K feasibility enumeration for tiny structured hints."""

from __future__ import annotations

import heapq
import math
from typing import Sequence

from affine_hints.modular import matvec_mod
from affine_hints.priors import SecretPrior


def top_k_assignments(
    *,
    H: Sequence[Sequence[int]],
    ell: Sequence[int],
    q: int,
    prior: SecretPrior,
    n: int,
    top_k: int,
    max_factor_states: int,
) -> list[tuple[float, tuple[int, ...]]]:
    """Exact bounded reference for factor-graph outputs at tiny dimensions."""

    heap: list[tuple[float, tuple[int, ...]]] = []
    for assignment in prior.enumerate_support(n, max_factor_states):
        if tuple(matvec_mod(H, assignment, q)) != tuple(int(value) % q for value in ell):
            continue
        score = prior.log_prob(assignment)
        item = (score, tuple(assignment))
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return sorted(heap, reverse=True)

