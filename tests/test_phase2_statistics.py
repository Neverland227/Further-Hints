from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from phase2_lattice import _comparison_rows, _select_calibration  # noqa: E402


def test_calibration_selects_aggregate_expected_work_without_dropping_failures() -> None:
    rows = []
    for arm in ("baseline", "exact_prior"):
        for seed, success in ((1, 1.0), (2, 0.0)):
            rows.append(
                {
                    "regime_id": "toy",
                    "arm": arm,
                    "theta_id": "fast_partial_success",
                    "total_time": 10.0,
                    "success_probability": success,
                }
            )
        for seed in (1, 2):
            rows.append(
                {
                    "regime_id": "toy",
                    "arm": arm,
                    "theta_id": "slow_full_success",
                    "total_time": 100.0,
                    "success_probability": 1.0,
                }
            )
    selected = _select_calibration(pd.DataFrame(rows))
    assert selected["toy"]["baseline"] == "fast_partial_success"
    assert selected["toy"]["exact_prior"] == "fast_partial_success"


def test_confirmation_expected_work_uses_all_paired_instance_clusters() -> None:
    rows = []
    baseline_successes = (1.0, 0.0, 1.0, 0.0)
    exact_successes = (1.0, 1.0, 0.0, 1.0)
    common = {
        "regime_id": "toy",
        "n": 24,
        "m": 24,
        "q": 97,
        "r": 2,
        "prior": "ternary",
        "H_class": "dense_random",
        "candidate_budget": 100,
        "pivot_strategy": "first_unit_minor",
        "r_elim": 2,
    }
    for seed, baseline_success, exact_success in zip(range(4), baseline_successes, exact_successes):
        rows.append(
            {
                **common,
                "seed": seed,
                "arm": "baseline",
                "theta_id": "baseline_theta",
                "total_time": 10.0,
                "success_probability": baseline_success,
                "beta": 20,
            }
        )
        rows.append(
            {
                **common,
                "seed": seed,
                "arm": "exact_prior",
                "theta_id": "exact_theta",
                "total_time": 11.0,
                "success_probability": exact_success,
                "beta": 10,
            }
        )
    comparison, decision = _comparison_rows(
        pd.DataFrame(rows),
        {"toy": {"baseline": "baseline_theta", "exact_prior": "exact_theta"}},
        bootstrap_seed=7,
        replicates=400,
    )
    assert len(comparison) == 4
    assert decision["paired_instances"] == 4
    assert math.isclose(decision["baseline_success_probability"], 0.5)
    assert math.isclose(decision["exact_success_probability"], 0.75)
    assert math.isclose(decision["expected_work_ratio"], (11.0 / 0.75) / (10.0 / 0.5))
    assert decision["delta_beta"] == 10.0
    assert decision["usable_bootstrap_replicates"] > 300
