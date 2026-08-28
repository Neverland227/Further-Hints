"""One-shot held-out Gate 1c on certified constrained B2 top-K lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from affine_hints.candidates.mn_general import (
    BackendUnavailable,
    CertifiedTopKUnavailable,
    MNConstrainedTopKCandidateSource,
)
from affine_hints.config import (
    ConfigurationError,
    RunContext,
    append_jsonl,
    atomic_write_text,
    load_yaml,
    process_peak_rss_bytes,
    read_jsonl,
    require_common_limits,
)
from affine_hints.coset import AffineCosetElimination
from affine_hints.diagnostics import candidate_correlation_diagnostics, wilson_interval
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.modular import centered_vector
from affine_hints.priors import make_prior
from affine_hints.resources import apply_unix_task_limits, bounded_pool_imap_unordered
from phase1_selectivity import _require_gate0, _resolve_phase_minus1


EXPOSED_AUDIT_SEEDS = [5501, 5502, 5503, 5504, 5505]
RESERVED_CONFIRMATION_SEEDS = list(range(6501, 6531))
CERTIFIED_TOP_K_VALUES = [16, 32, 64, 128]


def _require_backend_audit(config: dict[str, Any]) -> Path:
    artifact_value = config.get("backend_audit_artifact")
    expected_hash = config.get("backend_audit_sha256")
    if artifact_value is None or expected_hash is None:
        raise ConfigurationError("Gate-1c confirmation requires a bound backend audit")
    artifact = Path(str(artifact_value)).resolve()
    try:
        payload = artifact.read_bytes()
        decision = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read Gate-1c backend audit: {artifact}") from exc
    if hashlib.sha256(payload).hexdigest() != str(expected_hash):
        raise ConfigurationError("Gate-1c backend audit SHA-256 mismatch")
    if decision.get("status") != "PASS":
        raise ConfigurationError("Gate-1c backend audit is not PASS")
    if decision.get("prior_selectivity_evaluated") is not False:
        raise ConfigurationError("backend audit must not use prior-selectivity outcomes")
    if decision.get("confirmation_seeds_used") is not False:
        raise ConfigurationError("backend audit reports that confirmation seeds were exposed")
    return artifact


def _regimes(config: dict[str, Any]) -> list[dict[str, Any]]:
    regimes: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(config["regimes"]):
        regime = dict(raw)
        regime_id = str(regime.get("regime_id", f"r{index:03d}"))
        if regime_id in identifiers:
            raise ConfigurationError(f"duplicate Gate-1c regime_id: {regime_id}")
        identifiers.add(regime_id)
        regime["regime_id"] = regime_id
        regime["n"] = int(regime["n"])
        regime["m"] = int(regime.get("m", regime["n"]))
        regime["q"] = int(regime["q"])
        regime["r"] = int(regime["r"])
        if regime["n"] > int(config.get("real_lattice_n_max", 80)):
            raise ConfigurationError(f"Gate-1c regime exceeds toy dimension cap: {regime_id}")
        regimes.append(regime)
    if not regimes:
        raise ConfigurationError("Gate-1c requires at least one regime")
    return regimes


def _validate_protocol(config: dict[str, Any], regimes: list[dict[str, Any]]) -> None:
    require_common_limits(config)
    if config.get("protocol_revision") != "ONE_SHOT_GATE1C_CERTIFIED_CONSTRAINED_TOP_K":
        raise ConfigurationError("unexpected Gate-1c protocol revision")
    if not bool(config.get("eligible_for_gate", False)) or bool(config.get("smoke", False)):
        raise ConfigurationError("Gate-1c confirmation must be gate-eligible and non-smoke")
    calibration = [int(value) for value in config["calibration_seeds"]]
    confirmation = [int(value) for value in config["confirmation_seeds"]]
    explicit = [int(value) for value in config["instance_seeds"]]
    if explicit != confirmation:
        raise ConfigurationError("instance_seeds must exactly match confirmation_seeds")
    if len(set(confirmation)) != len(confirmation):
        raise ConfigurationError("Gate-1c confirmation seeds must be unique")
    if set(calibration) & set(confirmation):
        raise ConfigurationError("Gate-1c calibration and confirmation seeds overlap")
    if calibration != EXPOSED_AUDIT_SEEDS:
        raise ConfigurationError("Gate-1c calibration seeds differ from the exposed audit seeds")
    if confirmation != RESERVED_CONFIRMATION_SEEDS:
        raise ConfigurationError("Gate-1c confirmation seeds differ from the reserved held-out seeds")
    required_instances = int(config["minimum_confirmation_instances_per_regime"])
    if len(confirmation) != required_instances or int(config["instances"]) != required_instances:
        raise ConfigurationError("Gate-1c confirmation instance count is not frozen correctly")
    top_k_values = [int(value) for value in config["top_k_values"]]
    primary_top_k = int(config["primary_top_k"])
    if sorted(set(top_k_values)) != top_k_values or top_k_values[-1] != primary_top_k:
        raise ConfigurationError("invalid nested Gate-1c top-K values")
    if top_k_values != CERTIFIED_TOP_K_VALUES:
        raise ConfigurationError("Gate-1c nested top-K is frozen to 16,32,64,128")
    if primary_top_k != int(config["max_candidates"]):
        raise ConfigurationError("primary_top_k must equal max_candidates")
    thresholds = [float(value) for value in config["score_radius_squared_grid"]]
    if thresholds != sorted(set(thresholds)) or thresholds[0] <= 1:
        raise ConfigurationError("score-radius grid must be strictly increasing and public")
    if len(thresholds) > int(config["max_radius_steps"]):
        raise ConfigurationError("score-radius grid exceeds max_radius_steps")
    if config.get("r_elim") != "all":
        raise ConfigurationError("primary Gate-1c confirmation requires r_elim=all")
    if len(regimes) * len(confirmation) > 120:
        raise ConfigurationError("Gate-1c confirmation exceeds the frozen 120-task cap")
    if bool(config.get("confirmation_uses_prior_for_parameter_selection", True)):
        raise ConfigurationError("Gate-1c parameter selection must be prior-blind")
    rate_keys = (
        "minimum_certified_top_k_rate",
        "minimum_true_presence_rate",
        "minimum_true_retention_rate",
        "gate1c_alpha_upper_max",
    )
    missing_rates = [key for key in rate_keys if key not in config]
    if missing_rates:
        raise ConfigurationError(f"missing Gate-1c rate fields: {missing_rates}")
    for key in rate_keys:
        value = float(config[key])
        if not 0.0 <= value <= 1.0:
            raise ConfigurationError(f"{key} must be in [0, 1]")
    if int(config["minimum_false_candidates_per_regime"]) <= 0:
        raise ConfigurationError("minimum_false_candidates_per_regime must be positive")
    if float(config["maximum_candidate_alpha_ci_width"]) <= 0.0:
        raise ConfigurationError("maximum_candidate_alpha_ci_width must be positive")


def _derived_seed(protocol_seed: int, regime_index: int) -> int:
    return int(
        np.random.SeedSequence([int(protocol_seed), int(regime_index), 0x473143]).generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )


def _confirmation_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    regime = task["regime"]
    protocol_seed = int(task["protocol_seed"])
    seed = int(task["seed"])
    try:
        apply_unix_task_limits(
            max_wall_seconds=int(config["max_wall_time"]),
            max_address_space=config.get("max_RSS", config["max_memory"]),
        )
        rng = np.random.default_rng(seed)
        prior = make_prior(regime["prior"])
        hint = generate_hint_matrix(
            rng,
            n=int(regime["n"]),
            r=int(regime["r"]),
            q=int(regime["q"]),
            hint_class=str(regime["H_class"]),
            parameters=dict(regime.get("H_parameters", {})),
        )
        instance = generate_synthetic_lwe(
            rng,
            n=int(regime["n"]),
            m=int(regime["m"]),
            q=int(regime["q"]),
            secret_prior=prior,
            H=hint.H,
            instance_id=f"{regime['regime_id']}-gate1c-{protocol_seed}",
        )
        elimination = AffineCosetElimination.build(
            instance.H,
            instance.ell,
            instance.q,
            int(regime["r"]),
            strategy=str(config["pivot_strategy"]),
            rng=rng,
            max_combinations=int(config.get("max_pivot_combinations", 100_000)),
        )
        source = MNConstrainedTopKCandidateSource()
        source.prepare(
            instance,
            {
                "elimination": elimination,
                "max_n": int(config.get("real_lattice_n_max", 80)),
                "beta": int(config["beta"]),
                "beta_max": int(config["beta_max"]),
                "reduction_seed": seed,
                "top_k_values": list(config["top_k_values"]),
                "score_radius_squared_grid": list(config["score_radius_squared_grid"]),
                "max_radius_steps": int(config["max_radius_steps"]),
                "max_slice_vectors_per_radius": int(config["max_slice_vectors_per_radius"]),
                "enumeration_node_limit": int(config["enumeration_node_limit"]),
            },
        )
        generation_started = time.monotonic()
        candidate_list = source.generate(int(config["primary_top_k"]), rng)
        generation_time = time.monotonic() - generation_started
        if not bool(candidate_list.metadata.get("top_k_certified", False)):
            raise RuntimeError("RESOURCE_LIMIT: constrained top-K was not certified")
        truth = tuple(instance.s[index] % instance.q for index in elimination.residual_indices)
        true_index = next(
            (index for index, candidate in enumerate(candidate_list.candidates) if candidate == truth),
            None,
        )
        rows: list[dict[str, Any]] = []
        for top_k in (int(value) for value in config["top_k_values"]):
            candidates = candidate_list.candidates[:top_k]
            exact_passes: list[bool] = []
            differences: list[tuple[int, ...]] = []
            false_passes: list[bool] = []
            predicate_started = time.monotonic()
            for candidate in candidates:
                full = tuple(centered_vector(elimination.reconstruct(candidate), instance.q))
                passed = bool(
                    prior.in_support(full) and elimination.remaining_hints_pass(candidate)
                )
                exact_passes.append(passed)
                if tuple(candidate) != truth:
                    false_passes.append(passed)
                    differences.append(
                        tuple(
                            (truth[index] - int(candidate[index])) % instance.q
                            for index in range(len(truth))
                        )
                    )
            predicate_time = time.monotonic() - predicate_started
            true_present = true_index is not None and true_index < top_k
            true_retained = bool(true_present and exact_passes[true_index])
            false_count = len(false_passes)
            false_survivors = int(sum(false_passes))
            passing_indices = [index for index, value in enumerate(exact_passes) if value]
            true_rank_after = (
                passing_indices.index(true_index) + 1
                if true_present and true_index in passing_indices
                else None
            )
            correlations = candidate_correlation_diagnostics(
                differences,
                false_passes,
                instance.q,
            )
            metadata = candidate_list.metadata
            rows.append(
                {
                    "task_id": task["task_id"],
                    "protocol_role": "ONE_SHOT_GATE1C_HELD_OUT_CONFIRMATION",
                    "regime_id": regime["regime_id"],
                    "protocol_seed": protocol_seed,
                    "derived_seed": seed,
                    "n": int(regime["n"]),
                    "m": int(regime["m"]),
                    "q": int(regime["q"]),
                    "r": int(regime["r"]),
                    "r_elim": int(regime["r"]),
                    "prior": (
                        json.dumps(regime["prior"], sort_keys=True)
                        if isinstance(regime["prior"], dict)
                        else regime["prior"]
                    ),
                    "H_class": regime["H_class"],
                    "top_k": top_k,
                    "primary_top_k": int(config["primary_top_k"]),
                    "candidate_protocol": "CERTIFIED_CONSTRAINED_TOP_K",
                    "top_k_certified": True,
                    "candidate_count": len(candidates),
                    "false_candidate_count": false_count,
                    "false_survivor_count": false_survivors,
                    "candidate_weighted_alpha_list": (
                        false_survivors / false_count if false_count else math.nan
                    ),
                    "any_false_survivor": bool(false_survivors),
                    "true_candidate_present": true_present,
                    "true_retention": true_retained,
                    "true_rank_before_prior": true_index + 1 if true_present else None,
                    "true_rank_after_prior": true_rank_after,
                    "certification_radius_squared": float(
                        metadata["certification_radius_squared"]
                    ),
                    "kth_candidate_score_squared": float(
                        -candidate_list.scores[top_k - 1]
                    ),
                    "canonical_unique_candidates_in_certification_ball": int(
                        metadata["canonical_unique_candidates_in_certification_ball"]
                    ),
                    "enumeration_nodes_final": int(metadata["enumeration_nodes"]),
                    "enumeration_nodes_total_across_attempts": int(
                        metadata["enumeration_nodes_total_across_attempts"]
                    ),
                    "raw_slice_vector_count_final": int(metadata["raw_slice_vector_count"]),
                    "raw_slice_vectors_total_across_attempts": int(
                        metadata["raw_slice_vectors_total_across_attempts"]
                    ),
                    "embedding_slice_vector_count_final": int(
                        metadata["embedding_slice_vector_count"]
                    ),
                    "duplicate_residual_count_final": int(
                        metadata["duplicate_residual_count"]
                    ),
                    "radius_attempt_count": len(metadata["radius_attempts"]),
                    "generation_time": generation_time,
                    "predicate_time": predicate_time,
                    "peak_memory": process_peak_rss_bytes(),
                    "predicate_integration": "leaf_postfilter",
                    "radius_selection_uses_true_norm": False,
                    "radius_selection_uses_prior_pass_rate": False,
                    "correlations": correlations,
                }
            )
        return {"task_id": task["task_id"], "rows": rows, "failure": None}
    except Exception as exc:
        resource_limited = (
            isinstance(exc, (TimeoutError, MemoryError, CertifiedTopKUnavailable))
            or "RESOURCE_LIMIT" in str(exc)
        )
        censoring_metadata = exc.metadata if isinstance(exc, CertifiedTopKUnavailable) else None
        return {
            "task_id": task["task_id"],
            "rows": [],
            "failure": {
                "task_id": task["task_id"],
                "protocol_role": "ONE_SHOT_GATE1C_HELD_OUT_CONFIRMATION",
                "regime_id": regime.get("regime_id"),
                "protocol_seed": protocol_seed,
                "derived_seed": seed,
                "classification": (
                    "CENSORED_RESOURCE_LIMIT"
                    if resource_limited
                    else ("UNAVAILABLE" if isinstance(exc, BackendUnavailable) else "FAILED_CONFIRMATION_TASK")
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "censoring_metadata": censoring_metadata,
                "traceback": traceback.format_exc(),
            },
        }


def _ratio_cluster_interval(
    false_counts: np.ndarray,
    survivor_counts: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    if len(false_counts) != len(survivor_counts) or not len(false_counts):
        return math.nan, math.nan
    estimates: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, len(false_counts), size=len(false_counts))
        denominator = float(false_counts[indices].sum())
        if denominator > 0:
            estimates.append(float(survivor_counts[indices].sum()) / denominator)
    if not estimates:
        return math.nan, math.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _mean_cluster_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return math.nan, math.nan
    estimates = [
        float(finite[rng.integers(0, len(finite), size=len(finite))].mean())
        for _ in range(replicates)
    ]
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _summarize(
    frame: pd.DataFrame,
    *,
    bootstrap_seed: int,
    replicates: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["regime_id", "prior", "H_class", "top_k"]
    for group_index, (values, group) in enumerate(frame.groupby(keys, dropna=False)):
        false_counts = group["false_candidate_count"].to_numpy(dtype=float)
        survivor_counts = group["false_survivor_count"].to_numpy(dtype=float)
        false_total = int(false_counts.sum())
        survivor_total = int(survivor_counts.sum())
        candidate_alpha = survivor_total / false_total if false_total else math.nan
        list_alpha = np.divide(
            survivor_counts,
            false_counts,
            out=np.full_like(survivor_counts, np.nan),
            where=false_counts > 0,
        )
        rng = np.random.default_rng(bootstrap_seed + group_index)
        candidate_low, candidate_high = _ratio_cluster_interval(
            false_counts,
            survivor_counts,
            rng,
            replicates,
        )
        instance_low, instance_high = _mean_cluster_interval(
            list_alpha,
            rng,
            replicates,
        )
        lists_with_survivor = int(group["any_false_survivor"].astype(bool).sum())
        any_low, any_high = wilson_interval(lists_with_survivor, len(group))
        diagnostic_low, diagnostic_high = wilson_interval(survivor_total, false_total)
        correlation_rows = [
            value for value in group["correlations"] if isinstance(value, dict)
        ]
        rows.append(
            {
                **dict(zip(keys, values)),
                "instances": len(group),
                "certified_top_k_rate": float(group["top_k_certified"].astype(float).mean()),
                "true_candidate_presence_rate": float(
                    group["true_candidate_present"].astype(float).mean()
                ),
                "true_retention_rate": float(group["true_retention"].astype(float).mean()),
                "false_candidates_total": false_total,
                "false_survivors_total": survivor_total,
                "candidate_weighted_alpha": candidate_alpha,
                "candidate_weighted_alpha_cluster_ci_low": candidate_low,
                "candidate_weighted_alpha_cluster_ci_high": candidate_high,
                "candidate_weighted_alpha_cluster_ci_width": (
                    candidate_high - candidate_low
                    if math.isfinite(candidate_low) and math.isfinite(candidate_high)
                    else math.nan
                ),
                "instance_weighted_alpha": float(np.nanmean(list_alpha)),
                "instance_weighted_alpha_cluster_ci_low": instance_low,
                "instance_weighted_alpha_cluster_ci_high": instance_high,
                "probability_any_false_survivor": lists_with_survivor / len(group),
                "probability_any_false_survivor_wilson_low": any_low,
                "probability_any_false_survivor_wilson_high": any_high,
                "zero_survivor_conservative_list_level_upper": (
                    any_high if survivor_total == 0 else None
                ),
                "candidate_level_wilson_low_diagnostic": diagnostic_low,
                "candidate_level_wilson_high_diagnostic": diagnostic_high,
                "candidate_level_interval_label": (
                    "diagnostic only; candidates are dependent and primary uncertainty clusters by list"
                ),
                "median_true_rank_before_prior": float(
                    group["true_rank_before_prior"].dropna().median()
                ) if group["true_rank_before_prior"].notna().any() else None,
                "median_true_rank_after_prior": float(
                    group["true_rank_after_prior"].dropna().median()
                ) if group["true_rank_after_prior"].notna().any() else None,
                "median_certification_radius_squared": float(
                    group["certification_radius_squared"].median()
                ),
                "median_enumeration_nodes_total": float(
                    group["enumeration_nodes_total_across_attempts"].median()
                ),
                "median_generation_time": float(group["generation_time"].median()),
                "median_raw_slice_vectors_final": float(
                    group["raw_slice_vector_count_final"].median()
                ),
                "median_unique_candidates_in_certification_ball": float(
                    group["canonical_unique_candidates_in_certification_ball"].median()
                ),
                "mean_fraction_collinear_pairs": (
                    float(
                        np.mean(
                            [
                                float(value["fraction_collinear_pairs"])
                                for value in correlation_rows
                                if value.get("fraction_collinear_pairs") is not None
                            ]
                        )
                    )
                    if correlation_rows
                    else None
                ),
                "maximum_projective_class_size": max(
                    (int(value.get("largest_class_size", 0)) for value in correlation_rows),
                    default=None,
                ),
            }
        )
    return pd.DataFrame(rows)


def _gate_decision(
    config: dict[str, Any],
    summary: pd.DataFrame,
    failures: list[dict[str, Any]],
    regimes: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_k = int(config["primary_top_k"])
    primary = summary[summary["top_k"].astype(int) == primary_k] if not summary.empty else summary
    required_regimes = {str(value["regime_id"]) for value in regimes}
    observed_regimes = set(primary["regime_id"].astype(str)) if not primary.empty else set()
    required_instances = int(config["minimum_confirmation_instances_per_regime"])
    incomplete = bool(
        failures
        or observed_regimes != required_regimes
        or primary.empty
        or (primary["instances"].astype(int) < required_instances).any()
        or (
            primary["certified_top_k_rate"].astype(float)
            < float(config["minimum_certified_top_k_rate"])
        ).any()
    )
    true_failure = bool(
        not primary.empty
        and (
            (
                primary["true_candidate_presence_rate"].astype(float)
                < float(config["minimum_true_presence_rate"])
            )
            | (
                primary["true_retention_rate"].astype(float)
                < float(config["minimum_true_retention_rate"])
            )
        ).any()
    )
    underpowered = bool(
        primary.empty
        or (
            primary["false_candidates_total"].astype(int)
            < int(config["minimum_false_candidates_per_regime"])
        ).any()
    )
    imprecise = bool(
        primary.empty
        or (~np.isfinite(primary["candidate_weighted_alpha_cluster_ci_width"].astype(float))).any()
        or (
            primary["candidate_weighted_alpha_cluster_ci_width"].astype(float)
            > float(config["maximum_candidate_alpha_ci_width"])
        ).any()
    )
    measurable = (
        primary[
            (primary["candidate_weighted_alpha_cluster_ci_high"].astype(float)
             < float(config["gate1c_alpha_upper_max"]))
        ]
        if not primary.empty
        else primary
    )
    measurable_enough = bool(
        len(measurable) >= int(config["minimum_measurable_regimes"])
        and measurable["H_class"].nunique() >= int(config["minimum_measurable_H_classes"])
    )
    if incomplete:
        status = "BLOCKED_BACKEND_CENSORED"
        classification = "STOP_CURRENT_PROTOCOL"
    elif true_failure:
        status = "FAIL_TRUE_RETENTION"
        classification = "STOP_SCIENTIFIC_VALIDITY"
    elif underpowered:
        status = "BLOCKED_UNDERPOWERED"
        classification = "STOP_CURRENT_PROTOCOL"
    elif imprecise:
        status = "BLOCKED_IMPRECISE"
        classification = "STOP_CURRENT_PROTOCOL"
    elif measurable_enough:
        status = "PASS"
        classification = "PASS_MARGINAL_SELECTIVITY_ONLY"
    else:
        status = "FAIL_NO_MARGINAL_EFFECT"
        classification = "STOP_TESTED_TOPK_IDEA"
    return {
        "gate": "Gate 1c",
        "status": status,
        "classification": classification,
        "candidate_protocol": "CERTIFIED_CONSTRAINED_TOP_K",
        "primary_top_k": primary_k,
        "required_regimes": sorted(required_regimes),
        "observed_primary_regimes": sorted(observed_regimes),
        "failure_count": len(failures),
        "backend_censored": incomplete,
        "true_presence_or_retention_requirement_failed": true_failure,
        "underpowered": underpowered,
        "imprecise": imprecise,
        "measurable_regime_count": len(measurable),
        "measurable_H_class_count": (
            int(measurable["H_class"].nunique()) if not measurable.empty else 0
        ),
        "selection_uses_prior_pass_rate": False,
        "confirmation_parameters_frozen_before_outcomes": True,
        "leaf_postfilter_gain_not_claimed": True,
        "list_level_survivor_probability_is_reporting_not_gate_endpoint": True,
        "phase2_permission": (
            "REOPTIMIZED_WORK_FRONTIER_ONLY" if status == "PASS" else "NO"
        ),
        "rule": (
            "certified held-out top-K completeness, true retention, precision, and stable marginal selectivity across H classes"
        ),
    }


def _refuse_duplicate_confirmation(config: dict[str, Any]) -> None:
    if config.get("_resume_dir"):
        return
    requested_seeds = [int(value) for value in config["confirmation_seeds"]]
    for recorded_path in sorted((PROJECT_ROOT / "results" / "phase1c").glob("*/config.yaml")):
        try:
            recorded = load_yaml(recorded_path)
        except (OSError, ConfigurationError):
            continue
        recorded_seeds = [int(value) for value in recorded.get("confirmation_seeds", [])]
        if (
            recorded.get("protocol_revision")
            == "ONE_SHOT_GATE1C_CERTIFIED_CONSTRAINED_TOP_K"
            and recorded_seeds == requested_seeds
        ):
            raise RuntimeError(
                "one-shot Gate-1c confirmation already has a run directory for these held-out seeds; use --resume on that directory"
            )


def run(config_path: str | Path | dict[str, Any]) -> Path:
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    regimes = _regimes(config)
    _validate_protocol(config, regimes)
    audit_path = _require_backend_audit(config)
    gate0_path = _require_gate0(config)
    phase_minus1_verified, phase_minus1_path = _resolve_phase_minus1(config)
    if not phase_minus1_verified or phase_minus1_path is None:
        raise RuntimeError("Phase -1 PASS decision not found; Gate-1c fails closed")
    _refuse_duplicate_confirmation(config)
    context = RunContext.create("phase1c", config, PROJECT_ROOT)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    tasks: list[dict[str, Any]] = []
    derived_seeds: list[int] = []
    for regime_index, regime in enumerate(regimes):
        for protocol_seed in config["confirmation_seeds"]:
            seed = _derived_seed(int(protocol_seed), regime_index)
            derived_seeds.append(seed)
            task_id = f"gate1c:{regime['regime_id']}:{int(protocol_seed)}:{seed}"
            tasks.append(
                {
                    "task_id": task_id,
                    "config": serializable,
                    "regime": regime,
                    "protocol_seed": int(protocol_seed),
                    "seed": seed,
                }
            )
    if len(set(derived_seeds)) != len(derived_seeds):
        raise ConfigurationError("Gate-1c derived seed collision")
    completed = {
        str(row["task_id"])
        for row in read_jsonl(context.run_dir / "checkpoints.jsonl")
    }
    pending = [task for task in tasks if task["task_id"] not in completed]
    for result in bounded_pool_imap_unordered(
        _confirmation_task,
        pending,
        workers=min(int(config.get("workers", 1)), max(1, len(pending))),
        max_tasks_per_child=int(config.get("max_tasks_per_child", 1)),
    ):
        append_jsonl(context.run_dir / "trials.jsonl", result["rows"])
        if result["failure"]:
            append_jsonl(context.run_dir / "failures.jsonl", [result["failure"]])
        append_jsonl(context.run_dir / "checkpoints.jsonl", [{"task_id": result["task_id"]}])
    trial_frame = pd.DataFrame(read_jsonl(context.run_dir / "trials.jsonl"))
    failure_rows = read_jsonl(context.run_dir / "failures.jsonl")
    failure_frame = pd.DataFrame(failure_rows)
    if not trial_frame.empty and {"task_id", "top_k"}.issubset(trial_frame.columns):
        trial_frame = trial_frame.drop_duplicates(["task_id", "top_k"], keep="first")
    if not failure_frame.empty and "task_id" in failure_frame:
        failure_frame = failure_frame.drop_duplicates("task_id", keep="first")
        failure_rows = failure_frame.to_dict("records")
    summary = (
        _summarize(
            trial_frame,
            bootstrap_seed=int(config["bootstrap_seed"]),
            replicates=int(config["bootstrap_replicates"]),
        )
        if not trial_frame.empty
        else pd.DataFrame()
    )
    summary.to_csv(context.run_dir / "summary.csv", index=False)
    decision = _gate_decision(config, summary, failure_rows, regimes)
    decision.update(
        {
            "planned_confirmation_tasks": len(tasks),
            "completed_confirmation_tasks": (
                int(trial_frame["task_id"].nunique()) if not trial_frame.empty else 0
            ),
            "backend_audit_artifact": str(audit_path),
            "gate0_decision": str(gate0_path),
            "phase_minus1_decision": str(phase_minus1_path),
            "protocol_deviation": None,
        }
    )
    decision_path = context.run_dir / "gate1c_decision.json"
    atomic_write_text(decision_path, json.dumps(decision, indent=2) + "\n")
    if decision["status"] == "PASS":
        atomic_write_text(
            context.run_dir / "NEXT.md",
            "# Gate 1c PASS\n\n"
            "Marginal selectivity is measurable on the certified held-out top-K lists. Do not run the existing Phase-2 script unchanged; only a separately preregistered reoptimized work-frontier experiment is permitted. No expected-work gain is claimed here.\n",
        )
    else:
        atomic_write_text(
            context.run_dir / "STOP.md",
            "# STOP\n\n"
            f"- Gate 1c status: {decision['status']}\n"
            f"- classification: {decision['classification']}\n"
            "- action: do not enter Phase 2 and do not revise this one-shot candidate protocol after confirmation\n",
        )
    atomic_write_text(
        context.run_dir / "report.md",
        "# Gate-1c certified top-K report\n\n"
        f"- planned held-out tasks: {len(tasks)}\n"
        f"- completed held-out tasks: {decision['completed_confirmation_tasks']}\n"
        f"- censored/failed tasks: {len(failure_rows)}\n"
        f"- Gate 1c: {decision['status']}\n\n"
        "This stage measures marginal information only. Leaf-postfilter selectivity is not an expected-work improvement.\n",
    )
    context.write_manifest(
        status="COMPLETED",
        derived_seeds={
            "protocol_confirmation_seeds": config["confirmation_seeds"],
            "derived_regime_task_seeds": derived_seeds,
        },
    )
    return context.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="existing phase1c run directory to resume")
    arguments = parser.parse_args()
    if arguments.resume:
        config = load_yaml(arguments.config)
        config["_resume_dir"] = arguments.resume
        print(run(config))
        return
    print(run(arguments.config))


if __name__ == "__main__":
    main()
