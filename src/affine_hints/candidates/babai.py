"""Optional bounded randomized-Babai candidate source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import CandidateList
from .mn_general import BackendUnavailable


@dataclass
class RandomizedBabaiSource:
    """Interface placeholder for prepared toy bases; never expands its budget."""

    prepared: dict[str, Any] | None = None

    def prepare(self, instance: Any, baseline_config: dict[str, Any]) -> None:
        try:
            import fpylll  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailable("RandomizedBabaiSource requires optional fpylll") from exc
        if "basis" not in baseline_config or "target" not in baseline_config:
            raise ValueError("a bounded prepared basis and target are required")
        self.prepared = dict(baseline_config)

    def generate(self, budget: int, rng: np.random.Generator) -> CandidateList:
        if self.prepared is None:
            raise RuntimeError("prepare must be called first")
        # The generic basis-to-secret coordinate map is deliberately required
        # from the caller; this prevents exposing an arbitrary LWE-input attack
        # wrapper through this diagnostic class.
        mapper = self.prepared.get("bounded_candidate_mapper")
        if mapper is None:
            raise BackendUnavailable("UNAVAILABLE: no experiment-specific bounded candidate mapper was provided")
        candidates, scores = mapper(budget=budget, rng=rng)
        return CandidateList(tuple(candidates), tuple(scores), None, "RandomizedBabaiSource", {"bounded": True})

