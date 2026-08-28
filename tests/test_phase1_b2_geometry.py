from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from affine_hints.config import load_yaml  # noqa: E402
from phase1_b2_geometry import (  # noqa: E402
    _build_formal_gate_config,
    _regimes,
    _select_geometry,
    _summarize_geometry,
    _theta_id,
    _thetas,
    _validate_protocol,
)
from phase1_selectivity import (  # noqa: E402
    _expand_cells,
    _instance_seed_plan,
    _require_geometry_selection_binding,
)


CONFIG_PATH = ROOT / "configs" / "phase1" / "b2_geometry_calibration_v2.yaml"
SELECTION_FIXTURE = ROOT / "tests" / "fixtures" / "geometry_selection_selected.json"


def test_frozen_geometry_protocol_has_eight_points_and_160_tasks() -> None:
    config = load_yaml(CONFIG_PATH)
    regimes = _regimes(config)
    thetas = _thetas(config)
    _validate_protocol(config, regimes, thetas)
    assert len(regimes) == 4
    assert len(thetas) == 8
    assert len(regimes) * len(thetas) * len(config["calibration_seeds"]) == 160
    assert set(config["calibration_seeds"]).isdisjoint(config["confirmation_seeds"])


def test_geometry_summary_rejects_degenerate_and_truncated_lists() -> None:
    regime = {
        "regime_id": "dense_ternary",
        "prior": "ternary",
        "H_class": "dense_random",
    }
    theta = {
        "beta": 10,
        "scaling_c": 1.0,
        "radius_multiplier": 1.5,
        "candidate_budget": 1000,
        "r_elim": "all",
        "pivot_strategy": "min_activation_depth",
    }
    rows = []
    for seed in range(5):
        rows.append(
            {
                "regime_id": regime["regime_id"],
                "theta_id": _theta_id(theta),
                "false_candidate_count": 0,
                "true_candidate_present": True,
                "enumeration_complete_within_bounds": seed != 0,
                "enumeration_node_limit_reached": seed == 0,
                "enumeration_solution_budget_reached": False,
                "candidate_count": 1,
                "generation_time": 0.1,
                "enumeration_nodes": 10,
            }
        )
    summary = _summarize_geometry(
        pd.DataFrame(rows),
        pd.DataFrame(),
        [regime],
        [theta],
        list(range(5)),
        {
            "minimum_rows_per_regime_theta": 5,
            "minimum_true_candidate_presence_rate": 1.0,
            "require_false_candidate_in_every_list": True,
            "minimum_median_false_candidates": 10,
            "maximum_median_false_candidates": 300,
            "minimum_total_false_candidates_per_regime": 50,
        },
    )
    result = summary.iloc[0]
    assert not bool(result["eligible_geometry"])
    assert "incomplete_or_truncated_enumeration" in result["rejection_reasons"]
    assert "at_least_one_list_has_no_false_candidate" in result["rejection_reasons"]
    assert "median_false_candidates_below_requirement" in result["rejection_reasons"]


def test_selection_prefers_lower_beta_before_smaller_radius() -> None:
    regimes = [{"regime_id": f"regime-{index}"} for index in range(4)]
    rows = []
    for beta, radius in ((10, 2.0), (20, 1.25)):
        theta_id = f"beta={beta}-radius={radius}"
        for regime in regimes:
            rows.append(
                {
                    "regime_id": regime["regime_id"],
                    "theta_id": theta_id,
                    "beta": beta,
                    "scaling_c": 1.0,
                    "radius_multiplier": radius,
                    "candidate_budget": 1000,
                    "r_elim": "all",
                    "pivot_strategy": "min_activation_depth",
                    "false_candidate_count_median": 20.0,
                    "generation_time_mean": 1.0,
                    "eligible_geometry": True,
                    "selection_uses_prior_pass_rate": False,
                }
            )
    selected = _select_geometry(pd.DataFrame(rows), regimes)
    assert selected is not None
    assert selected["beta"] == 10
    assert selected["radius_multiplier"] == 2.0


def test_formal_config_freezes_prerequisites_and_reserved_seeds() -> None:
    config = load_yaml(CONFIG_PATH)
    selection_path = SELECTION_FIXTURE
    gate0_path = ROOT / "results" / "phase0" / "test" / "gate0_decision.json"
    phase_minus1_path = (
        ROOT / "results" / "phase_minus1" / "test" / "phase_minus1_decision.json"
    )
    selected = {
        "beta": 10,
        "scaling_c": 1.0,
        "radius_multiplier": 2.0,
        "candidate_budget": 1000,
        "r_elim": "all",
        "pivot_strategy": "min_activation_depth",
    }
    formal = _build_formal_gate_config(
        config,
        selected,
        calibration_run_id="calibration-test",
        selection_path=selection_path,
        gate0_path=gate0_path,
        phase_minus1_path=phase_minus1_path,
    )
    assert formal["instance_seeds"] == config["confirmation_seeds"]
    assert formal["instances"] == 30
    assert set(formal["instance_seeds"]).isdisjoint(formal["calibration_seeds"])
    assert formal["gate0_decision"] == str(gate0_path.resolve())
    assert formal["phase_minus1_decision"] == str(phase_minus1_path.resolve())
    assert formal["radius_multiplier"] == 2.0
    assert _require_geometry_selection_binding(formal) == selection_path.resolve()
    cells = _expand_cells(formal)
    assert len(cells) == 4
    assert len(_instance_seed_plan(formal, len(cells))) == 120


def test_formal_config_rejects_a_changed_geometry_hash() -> None:
    config = load_yaml(CONFIG_PATH)
    selected = {
        "beta": 10,
        "scaling_c": 1.0,
        "radius_multiplier": 2.0,
        "candidate_budget": 1000,
        "r_elim": "all",
        "pivot_strategy": "min_activation_depth",
    }
    formal = _build_formal_gate_config(
        config,
        selected,
        calibration_run_id="calibration-test",
        selection_path=SELECTION_FIXTURE,
        gate0_path=ROOT / "gate0_decision.json",
        phase_minus1_path=ROOT / "gate_minus1_decision.json",
    )
    formal["geometry_selection_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _require_geometry_selection_binding(formal)


def test_explicit_instance_seed_plan_is_deterministic_and_cell_separated() -> None:
    config = {
        "master_seed": 99,
        "instances": 3,
        "calibration_seeds": [5501, 5502],
        "confirmation_seeds": [6501, 6502, 6503],
        "instance_seeds": [6501, 6502, 6503],
    }
    first = _instance_seed_plan(config, 2)
    second = _instance_seed_plan(config, 2)
    assert first == second
    assert [protocol for _, protocol in first] == [6501, 6502, 6503] * 2
    assert len({seed for seed, _ in first}) == 6


def test_explicit_instance_seeds_cannot_reuse_calibration_seed() -> None:
    config = {
        "master_seed": 99,
        "instances": 2,
        "calibration_seeds": [5501],
        "confirmation_seeds": [5501, 6502],
        "instance_seeds": [5501, 6502],
    }
    with pytest.raises(ValueError, match="disjoint"):
        _instance_seed_plan(config, 1)
