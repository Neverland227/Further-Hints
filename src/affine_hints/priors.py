"""Exact discrete secret-prior interfaces and bounded support enumeration."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np


class SupportTooLarge(ValueError):
    """Raised before an exact enumeration would exceed its configured limit."""


class SecretPrior(Protocol):
    """Protocol required by every exact secret-prior implementation."""

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray: ...

    def in_support(self, vector: Sequence[int]) -> bool: ...

    def log_prob(self, vector: Sequence[int]) -> float: ...

    def enumerate_support(self, n: int, max_size: int) -> Iterable[tuple[int, ...]]: ...

    def second_moment(self) -> float: ...

    def entropy_bits(self) -> float: ...

    def collision_entropy_bits(self) -> float: ...


@dataclass(frozen=True)
class UniformTernaryPrior:
    """Independent uniform coordinates on ``{-1, 0, 1}``."""

    support: tuple[int, int, int] = (-1, 0, 1)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(np.asarray(self.support, dtype=np.int64), size=n).astype(object)

    def in_support(self, vector: Sequence[int]) -> bool:
        return all(int(value) in self.support for value in vector)

    def log_prob(self, vector: Sequence[int]) -> float:
        return -len(vector) * math.log(3.0) if self.in_support(vector) else -math.inf

    def enumerate_support(self, n: int, max_size: int) -> Iterator[tuple[int, ...]]:
        size = 3**n
        if size > max_size:
            raise SupportTooLarge(f"ternary support {size} exceeds {max_size}")
        return itertools.product(self.support, repeat=n)

    def second_moment(self) -> float:
        return 2.0 / 3.0

    def entropy_bits(self) -> float:
        return math.log2(3.0)

    def collision_entropy_bits(self) -> float:
        return math.log2(3.0)


@dataclass(frozen=True)
class CBDPrior:
    """Independent centered-binomial coordinates with parameter ``eta``."""

    eta: int

    def __post_init__(self) -> None:
        if self.eta <= 0:
            raise ValueError("eta must be positive")

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(range(-self.eta, self.eta + 1))

    def coordinate_prob(self, value: int) -> float:
        if value not in self.support:
            return 0.0
        return math.comb(2 * self.eta, self.eta + value) / float(2 ** (2 * self.eta))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        left = rng.binomial(self.eta, 0.5, size=n)
        right = rng.binomial(self.eta, 0.5, size=n)
        return (left - right).astype(object)

    def in_support(self, vector: Sequence[int]) -> bool:
        return all(int(value) in self.support for value in vector)

    def log_prob(self, vector: Sequence[int]) -> float:
        if not self.in_support(vector):
            return -math.inf
        return sum(math.log(self.coordinate_prob(int(value))) for value in vector)

    def enumerate_support(self, n: int, max_size: int) -> Iterator[tuple[int, ...]]:
        size = len(self.support) ** n
        if size > max_size:
            raise SupportTooLarge(f"CBD support {size} exceeds {max_size}")
        return itertools.product(self.support, repeat=n)

    def second_moment(self) -> float:
        return self.eta / 2.0

    def entropy_bits(self) -> float:
        probabilities = [self.coordinate_prob(value) for value in self.support]
        return -sum(p * math.log2(p) for p in probabilities if p)

    def collision_entropy_bits(self) -> float:
        probabilities = [self.coordinate_prob(value) for value in self.support]
        return -math.log2(sum(p * p for p in probabilities))


@dataclass(frozen=True)
class FixedWeightTernaryPrior:
    """Uniform ternary vectors with exact counts of ``+1`` and ``-1``."""

    h_plus: int
    h_minus: int

    def __post_init__(self) -> None:
        if self.h_plus < 0 or self.h_minus < 0:
            raise ValueError("weights must be non-negative")

    def support_size(self, n: int) -> int:
        if self.h_plus + self.h_minus > n:
            return 0
        return math.comb(n, self.h_plus) * math.comb(n - self.h_plus, self.h_minus)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.h_plus + self.h_minus > n:
            raise ValueError("fixed weights exceed vector length")
        vector = np.zeros(n, dtype=object)
        permutation = rng.permutation(n)
        vector[permutation[: self.h_plus]] = 1
        vector[permutation[self.h_plus : self.h_plus + self.h_minus]] = -1
        return vector

    def in_support(self, vector: Sequence[int]) -> bool:
        values = [int(value) for value in vector]
        return (
            all(value in (-1, 0, 1) for value in values)
            and values.count(1) == self.h_plus
            and values.count(-1) == self.h_minus
        )

    def log_prob(self, vector: Sequence[int]) -> float:
        if not self.in_support(vector):
            return -math.inf
        return -math.log(self.support_size(len(vector)))

    def enumerate_support(self, n: int, max_size: int) -> Iterator[tuple[int, ...]]:
        size = self.support_size(n)
        if size > max_size:
            raise SupportTooLarge(f"fixed-weight support {size} exceeds {max_size}")
        if size == 0:
            return iter(())

        def generate() -> Iterator[tuple[int, ...]]:
            indices = tuple(range(n))
            for plus in itertools.combinations(indices, self.h_plus):
                plus_set = set(plus)
                remaining = tuple(i for i in indices if i not in plus_set)
                for minus in itertools.combinations(remaining, self.h_minus):
                    vector = [0] * n
                    for i in plus:
                        vector[i] = 1
                    for i in minus:
                        vector[i] = -1
                    yield tuple(vector)

        return generate()

    def second_moment_for_length(self, n: int) -> float:
        if n <= 0:
            raise ValueError("length must be positive")
        return (self.h_plus + self.h_minus) / n

    def second_moment(self) -> float:
        raise ValueError("fixed-weight second moment requires an explicit vector length")

    def entropy_bits_for_length(self, n: int) -> float:
        size = self.support_size(n)
        return math.log2(size) if size else -math.inf

    def entropy_bits(self) -> float:
        raise ValueError("fixed-weight entropy requires an explicit vector length")

    def collision_entropy_bits_for_length(self, n: int) -> float:
        return self.entropy_bits_for_length(n)

    def collision_entropy_bits(self) -> float:
        raise ValueError("fixed-weight collision entropy requires an explicit vector length")


def make_prior(specification: str | dict[str, int]) -> SecretPrior:
    """Construct a prior from a compact YAML-friendly specification."""

    if isinstance(specification, str):
        name = specification.lower()
        if name in ("ternary", "uniform_ternary"):
            return UniformTernaryPrior()
        if name == "cbd2":
            return CBDPrior(2)
        if name == "cbd3":
            return CBDPrior(3)
        raise ValueError(f"unknown prior: {specification}")
    kind = str(specification.get("kind", "")).lower()
    if kind == "fixed_weight":
        return FixedWeightTernaryPrior(int(specification["h_plus"]), int(specification["h_minus"]))
    if kind == "cbd":
        return CBDPrior(int(specification["eta"]))
    raise ValueError(f"unknown prior specification: {specification}")

