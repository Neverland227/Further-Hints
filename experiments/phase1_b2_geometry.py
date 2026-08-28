"""Bounded, prior-blind B2 geometry calibration before formal Gate 1.

This stage selects a non-degenerate MN-general candidate-list geometry using
only list size, truth presence, enumeration completeness, and resource use.
It never evaluates exact-prior pass rates and never runs formal Gate 1
automatically.  A selected geometry produces a frozen Gate-1b YAML artifact
that must be launched separately.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
import yaml

from affine_hints.candidates.mn_general import BackendUnavailable, MNGeneralCandidateSource
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
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.priors import make_prior
from affine_hints.resources import apply_unix_task_limits, bounded_pool_imap_unordered
from phase1_selectivity import _require_gate0, _resolve_phase_minus1


GRID_KEYS = (
    "beta",
    "scaling_c",
    "radius_multiplier",
    "candidate_budget",
    "r_elim",
    "pivot_strategy",
)


def _theta_id(theta: dict[str, Any]) -> str:
    return "-".join(f"{key}={theta[key]}" for key in GRID_KEYS)


def _regimes(config: dict[str, Any]) -> list[dict[str, Any]]:
    regimes: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, value in enumerate(config["regimes"]):
        regime = dict(value)
        regime_id = str(regime.get("regime_id", f"r{index:03d}"))
        if regime_id in identifiers:
            raise ConfigurationError(f"duplicate geometry regime_id: {regime_id}")
        identifiers.add(regime_id)
        regime["regime_id"] = regime_id
        regime["n"] = int(regime["n"])
        regime["m"] = int(
            regime.get(
                "m",
                round(float(regime.get("m_over_n", 1.0)) * regime["n"]),
            )
        )
        regime["q"] = int(regime["q"])
        regime["r"] = int(regime["r"])
        if regime["r"] > regime["n"]:
            raise ConfigurationError(f"r exceeds n in geometry regime {regime_id}")
        if regime["n"] > int(config.get("real_lattice_n_max", 80)):
            raise ConfigurationError(f"geometry regime exceeds toy dimension cap: {regime_id}")
        regimes.append(regime)
    if not regimes:
        raise ConfigurationError("at least one B2 geometry regime is required")
    return regimes


def _thetas(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["grid"]
    missing = [key for key in GRID_KEYS if key not in grid or not grid[key]]
    if missing:
        raise ConfigurationError(f"missing or empty B2 geometry grid fields: {missing}")
    thetas = [
        dict(zip(GRID_KEYS, values))
        for values in itertools.product(*(grid[key] for key in GRID_KEYS))
    ]
    maximum_grid_points = int(config.get("max_geometry_grid_points", 8))
    if len(thetas) > maximum_grid_points:
        raise ConfigurationError(
            f"geometry grid has {len(thetas)} points, above frozen cap {maximum_grid_points}"
        )
    radius_max = float(config.get("radius_multiplier_max", 2.5))
    for theta in thetas:
        beta = int(theta["beta"])
        radius = float(theta["radius_multiplier"])
        budget = int(theta["candidate_budget"])
        scaling = float(theta["scaling_c"])
        if beta < 0 or beta > int(config["beta_max"]):
            raise ConfigurationError("geometry beta is outside the frozen bound")
        if not (0.0 < radius <= radius_max):
            raise ConfigurationError("geometry radius is outside the frozen bound")
        if not (0 < budget <= int(config["max_candidates"])):
            raise ConfigurationError("geometry candidate budget is outside the frozen bound")
        if scaling <= 0:
            raise ConfigurationError("geometry scaling_c must be positive")
    return thetas


def _validate_protocol(
    config: dict[str, Any],
    regimes: list[dict[str, Any]],
    thetas: list[dict[str, Any]],
) -> None:
    require_common_limits(config)
    calibration_seeds = [int(value) for value in config["calibration_seeds"]]
    confirmation_seeds = [int(value) for value in config["confirmation_seeds"]]
    if not calibration_seeds or len(set(calibration_seeds)) != len(calibration_seeds):
        raise ConfigurationError("calibration seeds must be non-empty and unique")
    if not confirmation_seeds or len(set(confirmation_seeds)) != len(confirmation_seeds):
        raise ConfigurationError("confirmation seeds must be non-empty and unique")
    if set(calibration_seeds) & set(confirmation_seeds):
        raise ConfigurationError("calibration and confirmation seeds must be disjoint")
    planned_tasks = len(regimes) * len(thetas) * len(calibration_seeds)
    maximum_tasks = int(config.get("max_geometry_calibration_tasks", planned_tasks))
    if planned_tasks > maximum_tasks:
        raise ConfigurationError(
            f"geometry calibration has {planned_tasks} tasks, above frozen cap {maximum_tasks}"
        )
    selection = config["selection"]
    if int(selection["minimum_rows_per_regime_theta"]) != len(calibration_seeds):
        raise ConfigurationError(
            "minimum_rows_per_regime_theta must equal the number of calibration seeds"
        )
    if float(selection["minimum_median_false_candidates"]) <= 0:
        raise ConfigurationError("the geometry calibration must require false candidates")
    if (
        float(selection["maximum_median_false_candidates"])
        < float(selection["minimum_median_false_candidates"])
    ):
        raise ConfigurationError("invalid false-candidate calibration interval")
    if len(confirmation_seeds) != int(config["formal_gate"]["instances"]):
        raise ConfigurationError(
            "formal_gate.instances must equal the number of reserved confirmation seeds"
        )


def _resolved_r_elim(theta: dict[str, Any], regime: dict[str, Any]) -> int:
    value = theta["r_elim"]
    result = int(regime["r"]) if value == "all" else int(value)
    if result < 0 or result > int(regime["r"]):
        raise ConfigurationError("geometry r_elim is outside the regime hint rank")
    return result


def _geometry_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    regime = task["regime"]
    theta = task["theta"]
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
            instance_id=f"{regime['regime_id']}-geometry-{seed}",
        )
        r_elim = _resolved_r_elim(theta, regime)
        elimination = AffineCosetElimination.build(
            instance.H,
            instance.ell,
            instance.q,
            r_elim,
            strategy=str(theta["pivot_strategy"]),
            rng=rng,
            max_combinations=int(config.get("max_pivot_combinations", 100_000)),
        )
        source = MNGeneralCandidateSource()
        source.prepare(
            instance,
            {
                "elimination": elimination,
                "max_n": int(config.get("real_lattice_n_max", 80)),
                "scaling_c": float(theta["scaling_c"]),
                "beta": int(theta["beta"]),
                "beta_max": int(config["beta_max"]),
                "radius_multiplier": float(theta["radius_multiplier"]),
                "enumeration_node_limit": int(config["enumeration_node_limit"]),
                "reduction_seed": seed,
            },
        )
        started = time.monotonic()
        candidate_list = source.generate(int(theta["candidate_budget"]), rng)
        generation_time = time.monotonic() - started
        truth = tuple(
            instance.s[index] % instance.q for index in elimination.residual_indices
        )
        true_candidate_count = sum(
            tuple(candidate) == truth for candidate in candidate_list.candidates
        )
        false_candidate_count = len(candidate_list.candidates) - true_candidate_count
        metadata = candidate_list.metadata
        row = {
            "task_id": task["task_id"],
            "protocol_role": "B2_GEOMETRY_CALIBRATION_ONLY",
            "regime_id": regime["regime_id"],
            "theta_id": _theta_id(theta),
            "seed": seed,
            "n": int(regime["n"]),
            "m": int(regime["m"]),
            "q": int(regime["q"]),
            "r": int(regime["r"]),
            "prior": (
                json.dumps(regime["prior"], sort_keys=True)
                if isinstance(regime["prior"], dict)
                else regime["prior"]
            ),
            "H_class": regime["H_class"],
            "beta": int(theta["beta"]),
            "scaling_c": float(theta["scaling_c"]),
            "radius_multiplier": float(theta["radius_multiplier"]),
            "candidate_budget": int(theta["candidate_budget"]),
            "r_elim": r_elim,
            "pivot_strategy": theta["pivot_strategy"],
            "candidate_count": len(candidate_list.candidates),
            "false_candidate_count": false_candidate_count,
            "true_candidate_count": true_candidate_count,
            "true_candidate_present": bool(true_candidate_count),
            "eligible_embedding_vector_count": int(
                metadata.get("eligible_embedding_vector_count", len(candidate_list.candidates))
            ),
            "duplicate_residual_count": int(metadata.get("duplicate_residual_count", 0)),
            "unique_candidate_count": int(
                metadata.get("unique_candidate_count", len(candidate_list.candidates))
            ),
            "enumeration_complete_within_bounds": bool(
                metadata.get("enumeration_complete_within_bounds", False)
            ),
            "enumeration_node_limit_reached": bool(
                metadata.get("enumeration_node_limit_reached", False)
            ),
            "enumeration_solution_budget_reached": bool(
                metadata.get("enumeration_solution_budget_reached", False)
            ),
            "enumeration_nodes": int(metadata.get("enumeration_nodes", 0)),
            "enumeration_radius_squared": float(
                metadata.get("enumeration_radius_squared", math.nan)
            ),
            "generation_time": generation_time,
            "peak_memory": process_peak_rss_bytes(),
            "prior_selectivity_evaluated": False,
        }
        return {"task_id": task["task_id"], "row": row, "failure": None}
    except Exception as exc:
        resource_limited = isinstance(exc, (TimeoutError, MemoryError)) or "RESOURCE_LIMIT" in str(exc)
        return {
            "task_id": task["task_id"],
            "row": None,
            "failure": {
                "task_id": task["task_id"],
                "protocol_role": "B2_GEOMETRY_CALIBRATION_ONLY",
                "regime_id": regime.get("regime_id"),
                "theta_id": _theta_id(theta),
                "seed": seed,
                "classification": (
                    "RESOURCE_LIMIT"
                    if resource_limited
                    else ("UNAVAILABLE" if isinstance(exc, BackendUnavailable) else "FAILED_TASK")
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def _summarize_geometry(
    frame: pd.DataFrame,
    failure_frame: pd.DataFrame,
    regimes: list[dict[str, Any]],
    thetas: list[dict[str, Any]],
    calibration_seeds: list[int],
    selection: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required_rows = int(selection["minimum_rows_per_regime_theta"])
    minimum_median_false = float(selection["minimum_median_false_candidates"])
    maximum_median_false = float(selection["maximum_median_false_candidates"])
    minimum_total_false = int(selection["minimum_total_false_candidates_per_regime"])
    require_false_every_list = bool(selection.get("require_false_candidate_in_every_list", True))
    required_true_rate = float(selection.get("minimum_true_candidate_presence_rate", 1.0))
    for regime in regimes:
        for theta in thetas:
            theta_id = _theta_id(theta)
            if frame.empty:
                group = pd.DataFrame()
            else:
                group = frame[
                    (frame["regime_id"] == regime["regime_id"])
                    & (frame["theta_id"] == theta_id)
                ]
            if failure_frame.empty:
                failures = pd.DataFrame()
            else:
                failures = failure_frame[
                    (failure_frame["regime_id"] == regime["regime_id"])
                    & (failure_frame["theta_id"] == theta_id)
                ]
            false_counts = (
                group["false_candidate_count"].astype(float).to_numpy()
                if not group.empty
                else np.asarray([], dtype=float)
            )
            complete = bool(
                len(group) == required_rows
                and group["enumeration_complete_within_bounds"].astype(bool).all()
                and not group["enumeration_node_limit_reached"].astype(bool).any()
                and not group["enumeration_solution_budget_reached"].astype(bool).any()
            ) if not group.empty else False
            true_rate = (
                float(group["true_candidate_present"].astype(float).mean())
                if not group.empty
                else 0.0
            )
            lists_with_false = int(np.sum(false_counts > 0))
            median_false = float(np.median(false_counts)) if len(false_counts) else math.nan
            total_false = int(np.sum(false_counts)) if len(false_counts) else 0
            rejection_reasons: list[str] = []
            if len(group) != required_rows or len(failures):
                rejection_reasons.append("missing_or_failed_calibration_rows")
            if not complete:
                rejection_reasons.append("incomplete_or_truncated_enumeration")
            if true_rate < required_true_rate:
                rejection_reasons.append("true_candidate_presence_below_requirement")
            if require_false_every_list and lists_with_false != required_rows:
                rejection_reasons.append("at_least_one_list_has_no_false_candidate")
            if not math.isfinite(median_false) or median_false < minimum_median_false:
                rejection_reasons.append("median_false_candidates_below_requirement")
            if math.isfinite(median_false) and median_false > maximum_median_false:
                rejection_reasons.append("median_false_candidates_above_requirement")
            if total_false < minimum_total_false:
                rejection_reasons.append("total_false_candidates_below_requirement")
            rows.append(
                {
                    "regime_id": regime["regime_id"],
                    "prior": (
                        json.dumps(regime["prior"], sort_keys=True)
                        if isinstance(regime["prior"], dict)
                        else regime["prior"]
                    ),
                    "H_class": regime["H_class"],
                    "theta_id": theta_id,
                    "beta": int(theta["beta"]),
                    "scaling_c": float(theta["scaling_c"]),
                    "radius_multiplier": float(theta["radius_multiplier"]),
                    "candidate_budget": int(theta["candidate_budget"]),
                    "r_elim": theta["r_elim"],
                    "pivot_strategy": theta["pivot_strategy"],
                    "planned_rows": len(calibration_seeds),
                    "completed_rows": len(group),
                    "failure_count": len(failures),
                    "all_enumerations_complete": complete,
                    "true_candidate_presence_rate": true_rate,
                    "lists_with_false_candidates": lists_with_false,
                    "false_candidate_count_min": (
                        int(np.min(false_counts)) if len(false_counts) else None
                    ),
                    "false_candidate_count_median": median_false,
                    "false_candidate_count_max": (
                        int(np.max(false_counts)) if len(false_counts) else None
                    ),
                    "false_candidates_total": total_false,
                    "candidate_count_median": (
                        float(group["candidate_count"].median()) if not group.empty else None
                    ),
                    "generation_time_mean": (
                        float(group["generation_time"].astype(float).mean())
                        if not group.empty
                        else None
                    ),
                    "enumeration_nodes_max": (
                        int(group["enumeration_nodes"].astype(int).max())
                        if not group.empty
                        else None
                    ),
                    "eligible_geometry": not rejection_reasons,
                    "rejection_reasons": ";".join(rejection_reasons),
                    "selection_uses_prior_pass_rate": False,
                }
            )
    return pd.DataFrame(rows)


def _select_geometry(
    summary: pd.DataFrame,
    regimes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if summary.empty:
        return None
    eligible: list[dict[str, Any]] = []
    required_regimes = {str(regime["regime_id"]) for regime in regimes}
    for theta_id, group in summary.groupby("theta_id"):
        observed_regimes = set(group["regime_id"].astype(str))
        if observed_regimes != required_regimes or not group["eligible_geometry"].astype(bool).all():
            continue
        first = group.iloc[0]
        eligible.append(
            {
                "theta_id": str(theta_id),
                "beta": int(first["beta"]),
                "scaling_c": float(first["scaling_c"]),
                "radius_multiplier": float(first["radius_multiplier"]),
                "candidate_budget": int(first["candidate_budget"]),
                "r_elim": first["r_elim"],
                "pivot_strategy": str(first["pivot_strategy"]),
                "minimum_regime_median_false_candidates": float(
                    group["false_candidate_count_median"].astype(float).min()
                ),
                "maximum_regime_median_false_candidates": float(
                    group["false_candidate_count_median"].astype(float).max()
                ),
                "mean_generation_time_across_regimes": float(
                    group["generation_time_mean"].astype(float).mean()
                ),
            }
        )
    if not eligible:
        return None
    # Prefer the least expensive reduction first, then the smallest radius
    # that already yields an informative list.  No prior-pass statistic enters
    # this ordering.
    return min(
        eligible,
        key=lambda row: (
            int(row["beta"]),
            float(row["radius_multiplier"]),
            int(row["candidate_budget"]),
            float(row["scaling_c"]),
            str(row["pivot_strategy"]),
        ),
    )


def _build_formal_gate_config(
    config: dict[str, Any],
    selected: dict[str, Any],
    *,
    calibration_run_id: str,
    selection_path: Path,
    gate0_path: Path,
    phase_minus1_path: Path,
) -> dict[str, Any]:
    formal = dict(config["formal_gate"])
    for key in (
        "max_wall_time",
        "max_memory",
        "max_RSS",
        "max_exact_states",
        "beta_max",
        "enumeration_node_limit",
        "gate0_decision",
        "phase_minus1_decision",
        "workers",
        "max_tasks_per_child",
        "max_pivot_combinations",
        "real_lattice_n_max",
    ):
        if key in config and key not in formal:
            formal[key] = config[key]
    formal["max_candidates"] = int(selected["candidate_budget"])
    formal["calibration_seeds"] = [int(value) for value in config["calibration_seeds"]]
    formal["confirmation_seeds"] = [int(value) for value in config["confirmation_seeds"]]
    formal["instance_seeds"] = [int(value) for value in config["confirmation_seeds"]]
    formal["instances"] = len(formal["instance_seeds"])
    formal["beta"] = int(selected["beta"])
    formal["mn_scaling_c"] = float(selected["scaling_c"])
    formal["radius_multiplier"] = float(selected["radius_multiplier"])
    formal["r_elim"] = [selected["r_elim"]]
    formal["pivot_strategy"] = str(selected["pivot_strategy"])
    formal["gate0_decision"] = str(gate0_path.resolve())
    formal["phase_minus1_decision"] = str(phase_minus1_path.resolve())
    formal["geometry_calibration_run_id"] = calibration_run_id
    formal["geometry_selection_artifact"] = str(selection_path)
    formal["geometry_selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    require_common_limits(formal)
    return formal


def run(config_path: str | Path | dict[str, Any]) -> Path:
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    regimes = _regimes(config)
    thetas = _thetas(config)
    _validate_protocol(config, regimes, thetas)
    gate0_path = _require_gate0(config)
    phase_minus1_verified, phase_minus1_path = _resolve_phase_minus1(config)
    if not phase_minus1_verified:
        raise RuntimeError("Phase -1 PASS decision not found; B2 geometry calibration fails closed")
    context = RunContext.create("phase1_geometry", config, PROJECT_ROOT)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    tasks: list[dict[str, Any]] = []
    for regime in regimes:
        for theta in thetas:
            for seed in config["calibration_seeds"]:
                task_id = f"geometry:{regime['regime_id']}:{_theta_id(theta)}:{int(seed)}"
                tasks.append(
                    {
                        "task_id": task_id,
                        "config": serializable,
                        "regime": regime,
                        "theta": theta,
                        "seed": int(seed),
                    }
                )
    completed = {str(row["task_id"]) for row in read_jsonl(context.run_dir / "checkpoints.jsonl")}
    pending = [task for task in tasks if task["task_id"] not in completed]
    for result in bounded_pool_imap_unordered(
        _geometry_task,
        pending,
        workers=min(int(config.get("workers", 1)), max(1, len(pending))),
        max_tasks_per_child=int(config.get("max_tasks_per_child", 1)),
    ):
        if result["row"]:
            append_jsonl(context.run_dir / "trials.jsonl", [result["row"]])
        if result["failure"]:
            append_jsonl(context.run_dir / "failures.jsonl", [result["failure"]])
        append_jsonl(context.run_dir / "checkpoints.jsonl", [{"task_id": result["task_id"]}])
    trial_frame = pd.DataFrame(read_jsonl(context.run_dir / "trials.jsonl"))
    failure_frame = pd.DataFrame(read_jsonl(context.run_dir / "failures.jsonl"))
    # A crash can occur after a result row is flushed but before its checkpoint
    # is appended. On resume that task is safely rerun; collapse only exact
    # task-id duplicates for the analysis while retaining the raw append log.
    if not trial_frame.empty and "task_id" in trial_frame:
        trial_frame = trial_frame.drop_duplicates(subset=["task_id"], keep="first")
    if not failure_frame.empty and "task_id" in failure_frame:
        failure_frame = failure_frame.drop_duplicates(subset=["task_id"], keep="first")
    summary = _summarize_geometry(
        trial_frame,
        failure_frame,
        regimes,
        thetas,
        [int(value) for value in config["calibration_seeds"]],
        dict(config["selection"]),
    )
    summary.to_csv(context.run_dir / "geometry_summary.csv", index=False)
    selected = _select_geometry(summary, regimes)
    selection = {
        "protocol_role": "B2_GEOMETRY_CALIBRATION_ONLY",
        "status": "SELECTED" if selected else "NO_ELIGIBLE_GEOMETRY",
        "selected_theta": selected,
        "selection_rule": (
            "all regimes satisfy the frozen geometry requirements; then prefer lowest beta and smallest radius"
        ),
        "selection_uses_prior_pass_rate": False,
        "prior_selectivity_evaluated": False,
        "calibration_seeds": [int(value) for value in config["calibration_seeds"]],
        "reserved_confirmation_seeds": [int(value) for value in config["confirmation_seeds"]],
        "planned_tasks": len(tasks),
        "completed_rows": len(trial_frame),
        "failure_count": len(failure_frame),
        "gate0_decision": str(gate0_path),
        "phase_minus1_decision": str(phase_minus1_path),
        "protocol_deviation": None,
    }
    selection_path = context.run_dir / "geometry_selection.json"
    atomic_write_text(selection_path, json.dumps(selection, indent=2) + "\n")
    if selected:
        formal = _build_formal_gate_config(
            config,
            selected,
            calibration_run_id=context.run_id,
            selection_path=selection_path,
            gate0_path=gate0_path,
            phase_minus1_path=phase_minus1_path,
        )
        formal_path = context.run_dir / "gate1b_config.yaml"
        atomic_write_text(formal_path, yaml.safe_dump(formal, sort_keys=False))
        atomic_write_text(
            context.run_dir / "RUN_NEXT.md",
            "# Next step\n\n"
            "Inspect geometry_selection.json and geometry_summary.csv. If the recorded selection matches the frozen rule, run formal Gate 1 separately:\n\n"
            f"```bash\npython experiments/phase1_selectivity.py --config \"{formal_path}\"\n```\n\n"
            "Do not edit gate1b_config.yaml after seeing any formal Gate-1 output.\n",
        )
    else:
        atomic_write_text(
            context.run_dir / "STOP.md",
            "# STOP\n\n"
            "- failed stage: B2 geometry calibration\n"
            "- evidence: no theta satisfied the frozen geometry requirements across every regime\n"
            "- action: do not enlarge this grid after observing the result; do not run formal Gate 1b or Phase 2\n",
        )
    atomic_write_text(
        context.run_dir / "report.md",
        "# B2 geometry calibration report\n\n"
        f"- planned tasks: {len(tasks)}\n"
        f"- completed rows: {len(trial_frame)}\n"
        f"- failures/resource limits: {len(failure_frame)}\n"
        f"- selection status: {selection['status']}\n\n"
        "This stage uses no exact-prior pass rate and cannot pass Gate 1 by itself.\n",
    )
    context.write_manifest(
        status="COMPLETED",
        derived_seeds={
            "calibration": config["calibration_seeds"],
            "reserved_confirmation": config["confirmation_seeds"],
        },
    )
    return context.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="existing phase1_geometry run directory to resume")
    arguments = parser.parse_args()
    if arguments.resume:
        config = load_yaml(arguments.config)
        config["_resume_dir"] = arguments.resume
        print(run(config))
        return
    print(run(arguments.config))


if __name__ == "__main__":
    main()
