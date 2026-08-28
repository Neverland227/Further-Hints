"""Optional bounded fpylll enumeration adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import CandidateList
from .mn_general import BackendUnavailable


@dataclass
class FpylllEnumerationSource:
    """Fail-closed enumeration interface with explicit radius/node limits."""

    prepared: dict[str, Any] | None = None

    def prepare(self, instance: Any, baseline_config: dict[str, Any]) -> None:
        del instance
        try:
            import fpylll  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailable("FpylllEnumerationSource requires optional fpylll") from exc
        required = ("basis", "radius", "enumeration_node_limit", "bounded_candidate_mapper")
        missing = [key for key in required if key not in baseline_config]
        if missing:
            raise ValueError(f"missing bounded enumeration settings: {missing}")
        self.prepared = dict(baseline_config)

    def generate(self, budget: int, rng: np.random.Generator) -> CandidateList:
        del rng
        if self.prepared is None:
            raise RuntimeError("prepare must be called first")
        # fpylll's node-limit hooks differ across releases. The server campaign
        # therefore supplies an experiment-specific callable that is also
        # enclosed by a process wall-time/RSS guard.
        runner = self.prepared.get("bounded_enumerator")
        if runner is None:
            raise BackendUnavailable("UNAVAILABLE: bounded enumerator adapter not available for this fpylll release")
        vectors, scores, nodes = runner(
            basis=self.prepared["basis"],
            radius=self.prepared["radius"],
            budget=budget,
            node_limit=int(self.prepared["enumeration_node_limit"]),
        )
        if nodes > int(self.prepared["enumeration_node_limit"]):
            raise RuntimeError("RESOURCE_LIMIT: enumeration node limit exceeded")
        mapper = self.prepared["bounded_candidate_mapper"]
        candidates = [mapper(vector) for vector in vectors]
        return CandidateList(tuple(candidates), tuple(scores), None, "FpylllEnumerationSource", {"nodes": nodes})

