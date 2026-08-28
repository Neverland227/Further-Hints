"""Phase 2: gate-locked toy real-lattice candidate-distribution calibration."""

from __future__ import annotations

import argparse
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

from affine_hints.candidates.mn_general import MNGeneralCandidateSource
from affine_hints.config import RunContext, append_jsonl, atomic_write_text, load_yaml, process_peak_rss_bytes, read_jsonl, validate_real_lattice_bounds
from affine_hints.coset import AffineCosetElimination
from affine_hints.estimators.total_work import expected_work
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.modular import centered_vector
from affine_hints.posterior import lwe_error
from affine_hints.priors import make_prior
from affine_hints.resources import apply_unix_task_limits, bounded_pool_imap_unordered


def _latest_gate1_decision() -> Path | None:
    candidates = sorted((PROJECT_ROOT / "results" / "phase1").glob("*/gate1_decision.json"), reverse=True)
    return candidates[0] if candidates else None


def _require_gate1(config: dict[str, Any]) -> Path:
    specified = config.get("gate1_decision", "AUTO_LATEST")
    path = _latest_gate1_decision() if specified == "AUTO_LATEST" else Path(str(specified)).resolve()
    if path is None or not path.exists():
        raise RuntimeError("Gate 1 PASS decision not found; Phase 2 fails closed")
    if json.loads(path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Gate 1 is not PASS")
    return path


def _theta_id(theta: dict[str, Any]) -> str:
    return "-".join(f"{key}={theta[key]}" for key in sorted(theta))


def _task(task: dict[str, Any]) -> dict[str, Any]:
    config, regime, theta, seed, split = task["config"], task["regime"], task["theta"], task["seed"], task["split"]
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
            instance_id=f"{regime['regime_id']}-{split}-{seed}",
        )
        elimination = AffineCosetElimination.build(
            instance.H,
            instance.ell,
            instance.q,
            int(theta["r_elim"]),
            strategy=str(theta["pivot_strategy"]),
            rng=rng,
            max_combinations=int(config.get("max_pivot_combinations", 100_000)),
        )
        source = MNGeneralCandidateSource()
        source.prepare(
            instance,
            {
                "elimination": elimination,
                "max_n": 80,
                "scaling_c": float(theta["scaling_c"]),
                "beta": int(theta["beta"]),
                "beta_max": int(config["beta_max"]),
                "radius_multiplier": float(theta["radius_multiplier"]),
                "enumeration_node_limit": int(config["enumeration_node_limit"]),
                "reduction_seed": seed,
            },
        )
        generation_started = time.monotonic()
        candidate_list = source.generate(int(theta["candidate_budget"]), rng)
        if not bool(candidate_list.metadata.get("enumeration_complete_within_bounds", False)):
            raise RuntimeError("RESOURCE_LIMIT: bounded enumeration node limit reached")
        generation_time = time.monotonic() - generation_started
        A_star, b_star = elimination.transform_lwe(instance.A, instance.b)
        verification_started = time.monotonic()
        errors = [lwe_error(A_star, b_star, candidate, instance.q) for candidate in candidate_list.candidates]
        verification_time = time.monotonic() - verification_started
        predicate_started = time.monotonic()
        exact_passes: list[bool] = []
        for candidate in candidate_list.candidates:
            full = tuple(centered_vector(elimination.reconstruct(candidate), instance.q))
            exact_passes.append(prior.in_support(full) and elimination.remaining_hints_pass(candidate))
        predicate_time = time.monotonic() - predicate_started
        survivor_count = sum(exact_passes)
        truth = tuple(instance.s[i] % instance.q for i in elimination.residual_indices)
        true_indices = [i for i, candidate in enumerate(candidate_list.candidates) if tuple(candidate) == truth]
        baseline_success = bool(true_indices)
        exact_success = any(exact_passes[i] for i in true_indices)
        construction_reduction = float(candidate_list.metadata.get("reduction_time", 0.0))
        candidate_reduction = float(candidate_list.metadata.get("candidate_reduction_time", 0.0))
        reduction_time = construction_reduction + candidate_reduction
        basis_time = float(candidate_list.metadata.get("basis_time", 0.0))
        list_generation_time = max(0.0, generation_time - basis_time - reduction_time)
        per_verify = verification_time / len(candidate_list.candidates) if candidate_list.candidates else 0.0
        per_predicate = predicate_time / len(candidate_list.candidates) if candidate_list.candidates else 0.0
        baseline = expected_work(
            basis_time=basis_time,
            reduction_time=reduction_time,
            list_generation_time=list_generation_time,
            candidate_count=len(candidate_list.candidates),
            predicate_time_per_candidate=0.0,
            survivor_count=len(candidate_list.candidates),
            verification_time_per_survivor=per_verify,
            success_probability=float(baseline_success),
            label="TOY REAL-LATTICE CALIBRATION",
        )
        exact = expected_work(
            basis_time=basis_time,
            reduction_time=reduction_time,
            list_generation_time=list_generation_time,
            candidate_count=len(candidate_list.candidates),
            predicate_time_per_candidate=per_predicate,
            survivor_count=survivor_count,
            verification_time_per_survivor=per_verify,
            success_probability=float(exact_success),
            label="TOY REAL-LATTICE CALIBRATION / leaf_postfilter",
        )
        common = {
            "regime_id": regime["regime_id"],
            "theta_id": _theta_id(theta),
            "split": split,
            "seed": seed,
            "n": regime["n"],
            "m": regime["m"],
            "q": regime["q"],
            "r": regime["r"],
            "prior": json.dumps(regime["prior"], sort_keys=True) if isinstance(regime["prior"], dict) else regime["prior"],
            "H_class": regime["H_class"],
            **theta,
            "candidate_count": len(candidate_list.candidates),
            "survivor_count": survivor_count,
            "true_candidate_present": baseline_success,
            "true_retention": exact_success,
            "basis_time": basis_time,
            "reduction_time": reduction_time,
            "candidate_generation_time": generation_time,
            "list_generation_time_measured": list_generation_time,
            "predicate_time": predicate_time,
            "verification_time": verification_time,
            "peak_memory": process_peak_rss_bytes(),
            "predicate_integration": "leaf_postfilter",
            "predicate_activation_depth_mean": elimination.pivot_choice.mean_last_nonzero,
            "predicate_activation_depth_max": elimination.pivot_choice.max_last_nonzero,
            "constraint_fill": elimination.pivot_choice.fill,
            "pivot_examined_pairs": elimination.pivot_choice.examined_pairs,
            "pivot_total_pairs": elimination.pivot_choice.total_pairs,
            "pivot_search_truncated": elimination.pivot_choice.search_truncated,
        }
        return {
            "task_id": task["task_id"],
            "rows": [
                {**common, "arm": "baseline", **baseline.as_dict()},
                {**common, "arm": "exact_prior", **exact.as_dict()},
            ],
            "failure": None,
        }
    except Exception as exc:
        resource_limited = isinstance(exc, (TimeoutError, MemoryError)) or "RESOURCE_LIMIT" in str(exc)
        return {
            "task_id": task["task_id"],
            "rows": [],
            "failure": {
                "regime_id": regime.get("regime_id"),
                "theta_id": _theta_id(theta),
                "split": split,
                "seed": seed,
                "classification": "RESOURCE_LIMIT" if resource_limited else "UNAVAILABLE_OR_FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def _regimes(config: dict[str, Any]) -> list[dict[str, Any]]:
    regimes = []
    for index, value in enumerate(config["regimes"]):
        regime = dict(value)
        regime["regime_id"] = regime.get("regime_id", f"r{index:03d}")
        regime["m"] = int(regime.get("m", math.floor(float(regime.get("m_over_n", 1.0)) * int(regime["n"]))))
        regimes.append(regime)
    return regimes


def _thetas(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["grid"]
    keys = ("beta", "scaling_c", "radius_multiplier", "candidate_budget", "r_elim", "pivot_strategy")
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def _select_calibration(frame: pd.DataFrame, minimum_rows_per_theta: int = 1) -> dict[str, dict[str, str]]:
    """Select by aggregate W=mean(total time)/mean(success), not per-run infinities."""

    selected: dict[str, dict[str, str]] = {}
    for regime_id, regime in frame.groupby("regime_id"):
        selected[regime_id] = {}
        for arm, group in regime.groupby("arm"):
            scores: dict[str, float] = {}
            for theta_id, theta_rows in group.groupby("theta_id"):
                if len(theta_rows) < minimum_rows_per_theta:
                    continue
                success = float(theta_rows["success_probability"].astype(float).mean())
                mean_time = float(theta_rows["total_time"].astype(float).mean())
                if success > 0 and mean_time > 0 and math.isfinite(mean_time):
                    scores[str(theta_id)] = math.log2(mean_time / success)
            if scores:
                selected[regime_id][arm] = min(scores, key=scores.get)
    return selected


def _comparison_rows(frame: pd.DataFrame, selected: dict[str, dict[str, str]], bootstrap_seed: int, replicates: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build paired instance clusters and bootstrap aggregate expected work."""

    pairs: list[dict[str, Any]] = []
    for regime_id, choices in selected.items():
        if set(choices) != {"baseline", "exact_prior"}:
            continue
        baseline = frame[
            (frame["regime_id"] == regime_id)
            & (frame["arm"] == "baseline")
            & (frame["theta_id"] == choices["baseline"])
        ].set_index("seed")
        exact = frame[
            (frame["regime_id"] == regime_id)
            & (frame["arm"] == "exact_prior")
            & (frame["theta_id"] == choices["exact_prior"])
        ].set_index("seed")
        for seed in sorted(set(baseline.index) & set(exact.index)):
            baseline_row = baseline.loc[seed]
            exact_row = exact.loc[seed]
            pairs.append(
                {
                    "regime_id": regime_id,
                    "seed": seed,
                    "n": int(baseline_row["n"]),
                    "m": int(baseline_row["m"]),
                    "q": int(baseline_row["q"]),
                    "r": int(baseline_row["r"]),
                    "prior": baseline_row["prior"],
                    "H_class": baseline_row["H_class"],
                    "baseline_theta": choices["baseline"],
                    "exact_theta": choices["exact_prior"],
                    "baseline_r_elim": int(baseline_row["r_elim"]),
                    "exact_r_elim": int(exact_row["r_elim"]),
                    "baseline_candidate_budget": int(baseline_row["candidate_budget"]),
                    "exact_candidate_budget": int(exact_row["candidate_budget"]),
                    "baseline_pivot_strategy": baseline_row["pivot_strategy"],
                    "exact_pivot_strategy": exact_row["pivot_strategy"],
                    "baseline_total_time": float(baseline_row["total_time"]),
                    "exact_total_time": float(exact_row["total_time"]),
                    "baseline_success": float(baseline_row["success_probability"]),
                    "exact_success": float(exact_row["success_probability"]),
                    "baseline_beta": int(baseline_row["beta"]),
                    "exact_beta": int(exact_row["beta"]),
                    "wall_clock_ratio": (
                        float(exact_row["total_time"]) / float(baseline_row["total_time"])
                        if float(baseline_row["total_time"]) > 0
                        else math.inf
                    ),
                }
            )
    comparison = pd.DataFrame(pairs)
    if comparison.empty:
        return comparison, {"classification": "STOP", "reason": "no paired confirmation records"}

    def aggregate(rows: pd.DataFrame) -> dict[str, float] | None:
        baseline_success = float(rows["baseline_success"].mean())
        exact_success = float(rows["exact_success"].mean())
        baseline_time = float(rows["baseline_total_time"].mean())
        exact_time = float(rows["exact_total_time"].mean())
        if min(baseline_success, exact_success, baseline_time, exact_time) <= 0:
            return None
        baseline_work = baseline_time / baseline_success
        exact_work = exact_time / exact_success
        ratio = exact_work / baseline_work
        return {
            "baseline_success_probability": baseline_success,
            "exact_success_probability": exact_success,
            "success_probability_difference": exact_success - baseline_success,
            "baseline_expected_work": baseline_work,
            "exact_expected_work": exact_work,
            "expected_work_ratio": ratio,
            "log2_expected_work_difference": -math.log2(ratio),
            "median_wall_clock_ratio": float(np.median(rows["wall_clock_ratio"].to_numpy(float))),
            "delta_beta": float(np.mean(rows["baseline_beta"].to_numpy(float) - rows["exact_beta"].to_numpy(float))),
        }

    def bootstrap(rows: pd.DataFrame, seed: int) -> tuple[dict[str, float], int]:
        rng = np.random.default_rng(seed)
        sampled: list[dict[str, float]] = []
        for _ in range(replicates):
            indices = rng.integers(0, len(rows), size=len(rows))
            value = aggregate(rows.iloc[indices])
            if value is not None and all(math.isfinite(number) for number in value.values()):
                sampled.append(value)
        if not sampled:
            return {}, 0
        result: dict[str, float] = {}
        for key in (
            "expected_work_ratio",
            "log2_expected_work_difference",
            "median_wall_clock_ratio",
            "success_probability_difference",
            "delta_beta",
        ):
            values = np.asarray([row[key] for row in sampled], dtype=float)
            result[f"{key}_ci_low"] = float(np.quantile(values, 0.025))
            result[f"{key}_ci_high"] = float(np.quantile(values, 0.975))
        return result, len(sampled)

    regime_effects: list[dict[str, Any]] = []
    for regime_index, (regime_id, rows) in enumerate(comparison.groupby("regime_id")):
        point = aggregate(rows)
        intervals, usable = bootstrap(rows, bootstrap_seed + regime_index + 1)
        first = rows.iloc[0]
        regime_effects.append(
            {
                "regime_id": regime_id,
                "n": int(first["n"]),
                "m": int(first["m"]),
                "q": int(first["q"]),
                "r": int(first["r"]),
                "prior": first["prior"],
                "H_class": first["H_class"],
                "baseline_theta": first["baseline_theta"],
                "exact_theta": first["exact_theta"],
                "baseline_r_elim": int(first["baseline_r_elim"]),
                "exact_r_elim": int(first["exact_r_elim"]),
                "baseline_candidate_budget": int(first["baseline_candidate_budget"]),
                "exact_candidate_budget": int(first["exact_candidate_budget"]),
                "paired_instances": len(rows),
                **(point or {}),
                **intervals,
                "usable_bootstrap_replicates": usable,
            }
        )

    point = aggregate(comparison)
    intervals, usable = bootstrap(comparison, bootstrap_seed)
    if point is None:
        classification = "STOP"
        reason = "zero confirmation success probability in at least one arm"
    elif usable < max(100, int(0.8 * replicates)):
        classification = "STOP"
        reason = "too few finite paired bootstrap replicates"
    elif intervals["log2_expected_work_difference_ci_high"] <= 0:
        classification = "STOP"
        reason = "upper 95% CI of log2 expected-work effect is non-positive"
    elif intervals["expected_work_ratio_ci_high"] < 1:
        classification = "MEASURABLE SECURITY-ESTIMATOR EFFECT"
        reason = "upper 95% CI of expected-work ratio is below one"
    else:
        classification = "MODELING RESULT"
        reason = "direction/selectivity may be measurable but complete expected work is not confirmed lower"
    strong_effects = [
        row
        for row in regime_effects
        if row.get("expected_work_ratio", math.inf) <= 2.0 / 3.0 or row.get("delta_beta", -math.inf) >= 4.0
    ]
    strong_signal_across_two_classes = (
        len({row["H_class"] for row in strong_effects}) >= 2
        or len({row["prior"] for row in strong_effects}) >= 2
    )
    return comparison, {
        "classification": classification,
        "reason": reason,
        "paired_instances": len(comparison),
        **(point or {}),
        **intervals,
        "usable_bootstrap_replicates": usable,
        "requested_bootstrap_replicates": replicates,
        "regime_effects": regime_effects,
        "strong_toy_signal_threshold": "expected-work ratio <= 2/3 or delta_beta >= 4",
        "strong_signal_across_two_H_or_prior_classes": strong_signal_across_two_classes,
        "label": "TOY REAL-LATTICE CALIBRATION",
    }


def run(config_path: str | Path | dict[str, Any]) -> Path:
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    validate_real_lattice_bounds(config)
    gate1 = _require_gate1(config)
    context = RunContext.create("phase2", config, PROJECT_ROOT)
    regimes, thetas = _regimes(config), _thetas(config)
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    calibration_tasks = []
    for regime in regimes:
        for theta in thetas:
            for seed in config["calibration_seeds"]:
                task_id = f"calibration:{regime['regime_id']}:{_theta_id(theta)}:{int(seed)}"
                calibration_tasks.append(
                    {
                        "task_id": task_id,
                        "config": serializable,
                        "regime": regime,
                        "theta": theta,
                        "seed": int(seed),
                        "split": "calibration",
                    }
                )
    completed = {str(row["task_id"]) for row in read_jsonl(context.run_dir / "checkpoints.jsonl")}
    pending_calibration = [task for task in calibration_tasks if task["task_id"] not in completed]
    for result in bounded_pool_imap_unordered(
        _task,
        pending_calibration,
        workers=min(int(config.get("workers", 1)), max(1, len(pending_calibration))),
        max_tasks_per_child=1,
    ):
        append_jsonl(context.run_dir / "trials.jsonl", result["rows"])
        if result["failure"]:
            append_jsonl(context.run_dir / "failures.jsonl", [result["failure"]])
        append_jsonl(context.run_dir / "checkpoints.jsonl", [{"task_id": result["task_id"]}])
    calibration_rows = [row for row in read_jsonl(context.run_dir / "trials.jsonl") if row.get("split") == "calibration"]
    failures = read_jsonl(context.run_dir / "failures.jsonl")
    calibration_frame = pd.DataFrame(calibration_rows)
    preregistration_path = context.run_dir / "confirmation_preregistration.json"
    if preregistration_path.exists():
        preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
        selected = preregistration["selected_parameters"]
    else:
        selected = (
            _select_calibration(
                calibration_frame,
                minimum_rows_per_theta=int(config.get("minimum_calibration_rows_per_theta", len(config["calibration_seeds"]))),
            )
            if not calibration_frame.empty
            else {}
        )
        preregistration = {
            "created_after_calibration_before_confirmation": True,
            "selected_parameters": selected,
            "confirmation_seeds": config["confirmation_seeds"],
            "selection_rule": "minimum mean calibration log2 expected work separately for each arm within the finite YAML grid",
            "protocol_deviation": None,
        }
        atomic_write_text(preregistration_path, json.dumps(preregistration, indent=2) + "\n")
    theta_lookup = {_theta_id(theta): theta for theta in thetas}
    confirmation_tasks: list[dict[str, Any]] = []
    for regime in regimes:
        choices = selected.get(regime["regime_id"], {})
        for theta_id in sorted(set(choices.values())):
            for seed in config["confirmation_seeds"]:
                theta = theta_lookup[theta_id]
                task_id = f"confirmation:{regime['regime_id']}:{theta_id}:{int(seed)}"
                confirmation_tasks.append(
                    {
                        "task_id": task_id,
                        "config": serializable,
                        "regime": regime,
                        "theta": theta,
                        "seed": int(seed),
                        "split": "confirmation",
                    }
                )
    completed = {str(row["task_id"]) for row in read_jsonl(context.run_dir / "checkpoints.jsonl")}
    pending_confirmation = [task for task in confirmation_tasks if task["task_id"] not in completed]
    for result in bounded_pool_imap_unordered(
        _task,
        pending_confirmation,
        workers=min(int(config.get("workers", 1)), max(1, len(pending_confirmation))),
        max_tasks_per_child=1,
    ):
        append_jsonl(context.run_dir / "trials.jsonl", result["rows"])
        if result["failure"]:
            append_jsonl(context.run_dir / "failures.jsonl", [result["failure"]])
        append_jsonl(context.run_dir / "checkpoints.jsonl", [{"task_id": result["task_id"]}])
    all_rows = read_jsonl(context.run_dir / "trials.jsonl")
    failures = read_jsonl(context.run_dir / "failures.jsonl")
    confirmation_rows = [row for row in all_rows if row.get("split") == "confirmation"]
    pd.DataFrame(all_rows).to_csv(context.run_dir / "summary.csv", index=False)
    confirmation_frame = pd.DataFrame(confirmation_rows)
    comparison, decision = _comparison_rows(
        confirmation_frame,
        selected,
        int(config.get("bootstrap_seed", config["master_seed"] + 1)),
        int(config.get("bootstrap_replicates", 2000)),
    ) if not confirmation_frame.empty else (pd.DataFrame(), {"classification": "STOP", "reason": "no confirmation data"})
    paired_counts = comparison.groupby("regime_id").size().to_dict() if not comparison.empty else {}
    minimum_confirmation_pairs = int(config.get("minimum_confirmation_pairs", 30))
    underpowered_regimes = [
        regime["regime_id"]
        for regime in regimes
        if int(paired_counts.get(regime["regime_id"], 0)) < minimum_confirmation_pairs
    ]
    if underpowered_regimes:
        decision["classification"] = "STOP"
        decision["reason"] = "fewer than the preregistered minimum paired confirmation instances"
    decision.update(
        {
            "planned_calibration_tasks": len(calibration_tasks),
            "completed_calibration_tasks": len(calibration_rows) // 2,
            "planned_confirmation_tasks": len(confirmation_tasks),
            "completed_confirmation_tasks": len(confirmation_rows) // 2,
            "failure_count": len(failures),
            "minimum_confirmation_pairs": minimum_confirmation_pairs,
            "underpowered_regimes": underpowered_regimes,
        }
    )
    comparison.to_csv(context.run_dir / "confirmation_comparisons.csv", index=False)
    decision.update({"gate": "Gate 2", "gate1_decision": str(gate1), "selected_parameters": selected})
    pd.DataFrame(decision.get("regime_effects", [])).to_csv(
        context.run_dir / "confirmation_effects.csv",
        index=False,
    )
    atomic_write_text(context.run_dir / "gate2_decision.json", json.dumps(decision, indent=2) + "\n")
    if decision["classification"] == "STOP":
        atomic_write_text(
            context.run_dir / "STOP.md",
            "# STOP\n\n- failed gate: Gate 2\n- evidence: gate2_decision.json and confirmation_comparisons.csv\n- interpretation: no confirmed complete expected-work improvement in the bounded toy regime\n- action: do not develop a higher-performance solver\n",
        )
    atomic_write_text(
        context.run_dir / "report.md",
        "# Phase 2 report\n\n"
        f"- calibration tasks: {len(calibration_tasks)}\n"
        f"- confirmation tasks: {len(confirmation_tasks)}\n"
        f"- failures/resource limits: {len(failures)}\n"
        f"- Gate 2 classification: {decision['classification']}\n\n"
        "All results are bounded synthetic toy calibration and use leaf postfiltering; they are not practical attack measurements.\n",
    )
    context.write_manifest(status="COMPLETED", derived_seeds={"calibration": config["calibration_seeds"], "confirmation": config["confirmation_seeds"]})
    return context.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="existing phase2 run directory to resume")
    arguments = parser.parse_args()
    if arguments.resume:
        config = load_yaml(arguments.config)
        config["_resume_dir"] = arguments.resume
        print(run(config))
        return
    print(run(arguments.config))


if __name__ == "__main__":
    main()
