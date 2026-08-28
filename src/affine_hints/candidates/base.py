"""Candidate-source protocol and immutable candidate-list records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class CandidateList:
    """A clustered candidate list produced for one synthetic instance."""

    candidates: tuple[tuple[int, ...], ...]
    scores: tuple[float, ...]
    true_index: int | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.candidates) != len(self.scores):
            raise ValueError("candidate and score lengths differ")


class CandidateSource(Protocol):
    """Interface shared by synthetic and optional bounded lattice sources."""

    def prepare(self, instance: Any, baseline_config: dict[str, Any]) -> None: ...

    def generate(self, budget: int, rng: np.random.Generator) -> CandidateList: ...

