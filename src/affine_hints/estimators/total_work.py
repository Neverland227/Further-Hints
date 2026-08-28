"""Complete candidate-generation/predicate/verification work accounting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WorkEstimate:
    """A complete toy measured or heuristic expected-work record."""

    basis_time: float
    reduction_time: float
    list_generation_time: float
    candidate_count: int
    predicate_time_per_candidate: float
    survivor_count: int
    verification_time_per_survivor: float
    success_probability: float
    total_time: float
    expected_work: float
    log2_expected_work: float
    label: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_work(
    *,
    basis_time: float,
    reduction_time: float,
    list_generation_time: float,
    candidate_count: int,
    predicate_time_per_candidate: float,
    survivor_count: int,
    verification_time_per_survivor: float,
    success_probability: float,
    label: str,
) -> WorkEstimate:
    """Evaluate ``W = T_total / P_success`` with all required cost terms."""

    total = (
        basis_time
        + reduction_time
        + list_generation_time
        + candidate_count * predicate_time_per_candidate
        + survivor_count * verification_time_per_survivor
    )
    work = math.inf if success_probability <= 0 else total / success_probability
    return WorkEstimate(
        basis_time=basis_time,
        reduction_time=reduction_time,
        list_generation_time=list_generation_time,
        candidate_count=candidate_count,
        predicate_time_per_candidate=predicate_time_per_candidate,
        survivor_count=survivor_count,
        verification_time_per_survivor=verification_time_per_survivor,
        success_probability=success_probability,
        total_time=total,
        expected_work=work,
        log2_expected_work=math.log2(work) if 0 < work < math.inf else math.inf,
        label=label,
    )


def list_generation_cost(model: str, *, beta: int, measured_seconds: float | None = None) -> dict[str, Any]:
    """Return a strictly labelled list-generation model."""

    if model == "explicit_enumeration":
        return {"model": model, "cost_units": "candidate_nodes", "label": "HEURISTIC ESTIMATOR"}
    if model == "sieve_database_model":
        return {
            "model": model,
            "log2_database_size": 0.2075 * beta,
            "label": "HEURISTIC / NOT EXECUTED",
        }
    if model == "measured_backend":
        if measured_seconds is None:
            raise ValueError("measured backend requires measured_seconds")
        return {"model": model, "seconds": measured_seconds, "label": "TOY MEASURED"}
    raise ValueError(f"unknown list-generation model: {model}")

