from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from phase1_selectivity import _gate_decision, _structured_h_deviations, _summarize  # noqa: E402


def _trial_frame(*, alpha: float, false_candidates: int, false_survivors: int) -> pd.DataFrame:
    rows = []
    for prior in ("ternary", "fixed_weight"):
        for hint_class in ("dense_random", "coded_dual_G_transpose"):
            for instance_index in range(30):
                rows.append(
                    {
                        "n": 12,
                        "m": 12,
                        "q": 17,
                        "r": 2,
                        "r_elim": 2,
                        "r_check": 0,
                        "prior": prior,
                        "H_class": hint_class,
                        "source": "MNGeneralCandidateSource",
                        "source_requested": "mn_general",
                        "evidence_label": "TOY REAL-LATTICE CANDIDATE DISTRIBUTION",
                        "alpha": alpha,
                        "false_candidate_count": false_candidates,
                        "false_survivor_count": false_survivors,
                        "any_false_survivor": bool(false_survivors),
                        "prior_auc_candidate_level_diagnostic": math.nan,
                        "prior_average_precision_candidate_level_diagnostic": math.nan,
                        "uniformity": {"sample_count": 0},
                        "true_candidate_present": True,
                        "true_retention": True,
                        "candidate_count": false_candidates + 1,
                        "generation_time": 0.01,
                        "predicate_time": 0.001,
                        "metadata": {
                            "enumeration_complete_within_bounds": True,
                            "instance_index": instance_index,
                        },
                    }
                )
    return pd.DataFrame(rows)


def _formal_config() -> dict[str, object]:
    return {
        "eligible_for_gate": True,
        "smoke": False,
        "gate1_alpha_upper_max": 0.95,
        "gate1_min_H_classes": 2,
        "gate1_min_valid_alpha_lists": 30,
        "gate1_min_false_candidates_per_group": 300,
    }


def _summarize_formal(frame: pd.DataFrame) -> pd.DataFrame:
    return _summarize(
        frame,
        bootstrap_seed=17,
        replicates=200,
        minimum_valid_alpha_lists=30,
        minimum_false_candidates_per_group=300,
    )


def test_zero_false_candidates_are_missing_information_not_infinite_bits() -> None:
    summary = _summarize_formal(
        _trial_frame(alpha=math.nan, false_candidates=0, false_survivors=0)
    )
    assert len(summary) == 4
    assert (summary["valid_alpha_lists"] == 0).all()
    assert (summary["false_candidates_total"] == 0).all()
    assert (~summary["alpha_estimable"].astype(bool)).all()
    assert summary["underpowered"].astype(bool).all()
    assert summary["mean_alpha_by_candidate_list"].isna().all()
    assert summary["cluster_bootstrap_alpha_high"].isna().all()
    assert summary["marginal_bits_lower_from_cluster_ci"].isna().all()
    assert summary["probability_any_false_survivor"].isna().all()


def test_gate_blocks_degenerate_b2_lists_instead_of_claiming_no_effect() -> None:
    summary = _summarize_formal(
        _trial_frame(alpha=math.nan, false_candidates=0, false_survivors=0)
    )
    deviations = _structured_h_deviations(summary)
    decision = _gate_decision(
        _formal_config(),
        summary,
        deviations,
        [],
        phase_minus1_verified=True,
        phase_minus1_path=None,
    )
    assert decision["status"] == "BLOCKED_B2_NO_FALSE_CANDIDATES"
    assert decision["b2_no_false_candidate_group_count"] == 4
    assert decision["b2_all_alpha_estimable"] is False
    assert decision["b2_structured_h_quantified"] is False
    assert not deviations["comparison_estimable"].astype(bool).any()


def test_gate_reports_no_effect_only_for_finite_powered_comparisons() -> None:
    summary = _summarize_formal(
        _trial_frame(alpha=1.0, false_candidates=10, false_survivors=10)
    )
    deviations = _structured_h_deviations(summary)
    decision = _gate_decision(
        _formal_config(),
        summary,
        deviations,
        [],
        phase_minus1_verified=True,
        phase_minus1_path=None,
    )
    assert decision["status"] == "FAIL_NO_MARGINAL_EFFECT"
    assert decision["b2_all_alpha_estimable"] is True
    assert decision["b2_underpowered_group_count"] == 0
    assert decision["b2_structured_h_quantified"] is True


def test_gate_blocks_finite_but_underpowered_b2_groups() -> None:
    summary = _summarize_formal(
        _trial_frame(alpha=0.0, false_candidates=1, false_survivors=0)
    )
    deviations = _structured_h_deviations(summary)
    decision = _gate_decision(
        _formal_config(),
        summary,
        deviations,
        [],
        phase_minus1_verified=True,
        phase_minus1_path=None,
    )
    assert decision["status"] == "BLOCKED_B2_UNDERPOWERED"
    assert decision["b2_no_false_candidate_group_count"] == 0
    assert decision["b2_underpowered_group_count"] == 4


def test_gate_can_pass_finite_powered_selectivity() -> None:
    summary = _summarize_formal(
        _trial_frame(alpha=0.9, false_candidates=10, false_survivors=9)
    )
    deviations = _structured_h_deviations(summary)
    decision = _gate_decision(
        _formal_config(),
        summary,
        deviations,
        [],
        phase_minus1_verified=True,
        phase_minus1_path=None,
    )
    assert decision["status"] == "PASS"
    assert decision["b2_structured_h_quantified"] is True
