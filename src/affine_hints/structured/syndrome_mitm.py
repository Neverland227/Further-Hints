"""Bounded syndrome-hash MITM for tiny synthetic sanity checks."""

from __future__ import annotations

import collections
import itertools
from typing import Sequence


def syndrome_mitm(
    *,
    H: Sequence[Sequence[int]],
    ell: Sequence[int],
    q: int,
    support: Sequence[int],
    max_states: int,
    max_matches: int,
    h_plus: int | None = None,
    h_minus: int | None = None,
) -> list[tuple[int, ...]]:
    """Return bounded support assignments satisfying hints and weight counts."""

    if (h_plus is None) != (h_minus is None):
        raise ValueError("h_plus and h_minus must be provided together")

    n = len(H[0]) if H else 0
    split = n // 2
    left_count = len(support) ** split
    right_count = len(support) ** (n - split)
    if left_count + right_count > max_states:
        raise RuntimeError("RESOURCE_LIMIT: MITM max_states exceeded")
    table: dict[tuple[tuple[int, ...], int | None, int | None], list[tuple[int, ...]]] = collections.defaultdict(list)
    for left in itertools.product(support, repeat=split):
        syndrome = tuple(sum(int(H[i][j]) * left[j] for j in range(split)) % q for i in range(len(H)))
        plus = left.count(1) if h_plus is not None else None
        minus = left.count(-1) if h_minus is not None else None
        table[(syndrome, plus, minus)].append(tuple(left))
    matches: list[tuple[int, ...]] = []
    for right in itertools.product(support, repeat=n - split):
        partial = tuple(
            sum(int(H[i][split + j]) * right[j] for j in range(n - split)) % q for i in range(len(H))
        )
        needed = tuple((int(ell[i]) - partial[i]) % q for i in range(len(H)))
        if h_plus is None:
            key = (needed, None, None)
        else:
            required_plus = int(h_plus) - right.count(1)
            required_minus = int(h_minus) - right.count(-1)
            if required_plus < 0 or required_minus < 0:
                continue
            key = (needed, required_plus, required_minus)
        for left in table.get(key, ()):
            matches.append(tuple(left) + tuple(right))
            if len(matches) >= max_matches:
                return matches
    return matches
