from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

from affine_hints.config import ConfigurationError, load_yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from phase1c_backend_audit import (  # noqa: E402
    _build_confirmation_config,
    _regimes as audit_regimes,
    _validate_protocol as validate_audit_protocol,
)
from phase1c_topk import (  # noqa: E402
    _derived_seed,
    _gate_decision,
    _regimes as confirmation_regimes,
    _require_backend_audit,
    _summarize,
    _validate_protocol as validate_confirmation_protocol,
)


AUDIT_CONFIG = ROOT / "configs" / "phase1" / "gate1c_backend_audit.yaml"
AUDIT_PASS_FIXTURE = ROOT / "tests" / "fixtures" / "backend_audit_pass.json"


def _formal_config() -> dict:
    audit = load_yaml(AUDIT_CONFIG)
    formal = _build_confirmation_config(
        audit,
        audit_run_id="fixture-audit",
        audit_decision_path=AUDIT_PASS_FIXTURE,
        gate0_path=AUDIT_PASS_FIXTURE,
        phase_minus1_path=AUDIT_PASS_FIXTURE,
    )
    return formal


def _synthetic_rows(*, survivors: int, true_retained: bool = True) -> pd.DataFrame:
    rows = []
    regimes = (
        ("dense_ternary", "ternary", "dense_random"),
        ("coded_ternary", "ternary", "coded_dual_G_transpose"),
        ("dense_fixed", "fixed", "dense_random"),
        ("coded_fixed", "fixed", "coded_dual_G_transpose"),
    )
    for regime_id, prior, h_class in regimes:
        for instance_index in range(30):
            rows.append(
                {
                    "regime_id": regime_id,
                    "prior": prior,
                    "H_class": h_class,
                    "top_k": 128,
                    "top_k_certified": True,
                    "true_candidate_present": True,
                    "true_retention": true_retained,
                    "false_candidate_count": 127,
                    "false_survivor_count": survivors,
                    "any_false_survivor": bool(survivors),
                    "true_rank_before_prior": 5,
                    "true_rank_after_prior": 1 if true_retained else None,
                    "certification_radius_squared": 64.0,
                    "enumeration_nodes_total_across_attempts": 1000,
                    "generation_time": 0.5,
                    "raw_slice_vector_count_final": 150,
                    "canonical_unique_candidates_in_certification_ball": 140,
                    "correlations": {
                        "fraction_collinear_pairs": 0.1,
                        "largest_class_size": 3,
                    },
                    "task_id": f"{regime_id}:{instance_index}",
                }
            )
    return pd.DataFrame(rows)


def test_audit_and_confirmation_protocols_are_frozen_and_disjoint() -> None:
    audit = load_yaml(AUDIT_CONFIG)
    regimes = audit_regimes(audit)
    validate_audit_protocol(audit, regimes)
    assert len(regimes) * len(audit["audit_radius_multipliers"]) * len(
        audit["calibration_seeds"]
    ) == 40

    formal = _formal_config()
    confirmation = confirmation_regimes(formal)
    validate_confirmation_protocol(formal, confirmation)
    assert _require_backend_audit(formal) == AUDIT_PASS_FIXTURE.resolve()
    derived = [
        _derived_seed(seed, regime_index)
        for regime_index in range(len(confirmation))
        for seed in formal["confirmation_seeds"]
    ]
    assert len(derived) == 120
    assert len(set(derived)) == 120
    assert not set(formal["calibration_seeds"]) & set(formal["confirmation_seeds"])


def test_confirmation_rejects_audit_hash_mismatch() -> None:
    formal = _formal_config()
    formal["backend_audit_sha256"] = "0" * 64
    with pytest.raises(ConfigurationError, match="SHA-256 mismatch"):
        _require_backend_audit(formal)


def test_gate1c_pass_means_marginal_selectivity_only() -> None:
    config = _formal_config()
    regimes = confirmation_regimes(config)
    summary = _summarize(_synthetic_rows(survivors=8), bootstrap_seed=11, replicates=200)
    decision = _gate_decision(config, summary, [], regimes)
    assert decision["status"] == "PASS"
    assert decision["classification"] == "PASS_MARGINAL_SELECTIVITY_ONLY"
    assert decision["phase2_permission"] == "REOPTIMIZED_WORK_FRONTIER_ONLY"
    assert decision["leaf_postfilter_gain_not_claimed"] is True


def test_gate1c_valid_no_effect_is_distinct_from_backend_censoring() -> None:
    config = _formal_config()
    regimes = confirmation_regimes(config)
    summary = _summarize(_synthetic_rows(survivors=127), bootstrap_seed=12, replicates=200)
    decision = _gate_decision(config, summary, [], regimes)
    assert decision["status"] == "FAIL_NO_MARGINAL_EFFECT"
    assert decision["classification"] == "STOP_TESTED_TOPK_IDEA"

    censored = _gate_decision(
        config,
        summary,
        [{"classification": "CENSORED_RESOURCE_LIMIT"}],
        regimes,
    )
    assert censored["status"] == "BLOCKED_BACKEND_CENSORED"
    assert censored["classification"] == "STOP_CURRENT_PROTOCOL"


def test_gate1c_true_retention_is_a_hard_requirement() -> None:
    config = _formal_config()
    regimes = confirmation_regimes(config)
    summary = _summarize(
        _synthetic_rows(survivors=8, true_retained=False),
        bootstrap_seed=13,
        replicates=200,
    )
    decision = _gate_decision(config, summary, [], regimes)
    assert decision["status"] == "FAIL_TRUE_RETENTION"
    assert decision["true_presence_or_retention_requirement_failed"] is True


def test_confirmation_requires_frozen_true_retention_threshold() -> None:
    formal = _formal_config()
    del formal["minimum_true_retention_rate"]
    with pytest.raises(ConfigurationError, match="missing Gate-1c rate fields"):
        validate_confirmation_protocol(formal, confirmation_regimes(formal))


def test_confirmation_rejects_prior_aware_parameter_selection() -> None:
    formal = copy.deepcopy(_formal_config())
    formal["confirmation_uses_prior_for_parameter_selection"] = True
    with pytest.raises(ConfigurationError, match="prior-blind"):
        validate_confirmation_protocol(formal, confirmation_regimes(formal))
