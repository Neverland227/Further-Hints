"""One-shot Gate-1c audit: legacy complete balls versus constrained slices.

This stage uses only previously exposed geometry-calibration seeds.  It never
evaluates the exact-prior predicate and never launches held-out confirmation.
Only an all-pass equivalence audit materializes a frozen Gate-1c confirmation
configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from affine_hints.candidates.mn_general import (
    BackendUnavailable,
    MNGeneralCandidateSource,
    constrained_candidates_within_radius,
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
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.priors import make_prior
from affine_hints.resources import apply_unix_task_limits, bounded_pool_imap_unordered
from phase1_selectivity import _require_gate0, _resolve_phase_minus1


EXPOSED_AUDIT_SEEDS = [5501, 5502, 5503, 5504, 5505]
RESERVED_CONFIRMATION_SEEDS = list(range(6501, 6531))
CERTIFIED_TOP_K_VALUES = [16, 32, 64, 128]


def _regimes(config: dict[str, Any]) -> list[dict[str, Any]]:
    regimes: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(config["regimes"]):
        regime = dict(raw)
        regime_id = str(regime.get("regime_id", f"r{index:03d}"))
        if regime_id in identifiers:
            raise ConfigurationError(f"duplicate Gate-1c audit regime_id: {regime_id}")
        identifiers.add(regime_id)
        regime["regime_id"] = regime_id
        regime["n"] = int(regime["n"])
        regime["m"] = int(regime.get("m", regime["n"]))
        regime["q"] = int(regime["q"])
        regime["r"] = int(regime["r"])
        if regime["r"] > regime["n"]:
            raise ConfigurationError(f"r exceeds n in audit regime {regime_id}")
        if regime["n"] > int(config.get("real_lattice_n_max", 80)):
            raise ConfigurationError(f"audit regime exceeds the toy cap: {regime_id}")
        regimes.append(regime)
    if not regimes:
        raise ConfigurationError("Gate-1c audit requires at least one regime")
    return regimes


def _validate_protocol(config: dict[str, Any], regimes: list[dict[str, Any]]) -> None:
    require_common_limits(config)
    calibration = [int(value) for value in config["calibration_seeds"]]
    confirmation = [int(value) for value in config["confirmation_seeds"]]
    if not calibration or len(set(calibration)) != len(calibration):
        raise ConfigurationError("audit calibration seeds must be non-empty and unique")
    if not confirmation or len(set(confirmation)) != len(confirmation):
        raise ConfigurationError("confirmation seeds must be non-empty and unique")
    if set(calibration) & set(confirmation):
        raise ConfigurationError("audit and confirmation seeds must be disjoint")
    if calibration != EXPOSED_AUDIT_SEEDS:
        raise ConfigurationError("Gate-1c audit must use only the five exposed seeds")
    if confirmation != RESERVED_CONFIRMATION_SEEDS:
        raise ConfigurationError("Gate-1c must preserve the thirty reserved held-out seeds")
    radii = [float(value) for value in config["audit_radius_multipliers"]]
    if not radii or any(value <= 0 for value in radii):
        raise ConfigurationError("audit radius multipliers must be positive")
    if radii != [1.25, 1.5]:
        raise ConfigurationError("Gate-1c audit is frozen to complete radii 1.25 and 1.5")
    planned = len(regimes) * len(radii) * len(calibration)
    if planned > int(config["max_backend_audit_tasks"]):
        raise ConfigurationError("Gate-1c backend audit exceeds its frozen task cap")
    if config.get("r_elim") != "all":
        raise ConfigurationError("primary Gate-1c audit requires full hint elimination")
    if int(config["legacy_raw_solution_limit"]) <= 0:
        raise ConfigurationError("legacy_raw_solution_limit must be positive")
    if int(config["constrained_slice_vector_limit"]) <= 0:
        raise ConfigurationError("constrained_slice_vector_limit must be positive")
    if int(config["beta"]) != 10 or float(config["scaling_c"]) != 1.0:
        raise ConfigurationError("Gate-1c audit is frozen to beta=10 and scaling_c=1")
    formal = config["formal_gate1c"]
    if int(formal["primary_top_k"]) != int(formal["max_candidates"]):
        raise ConfigurationError("formal primary_top_k must equal max_candidates")
    top_k_values = [int(value) for value in formal["top_k_values"]]
    if not top_k_values or sorted(set(top_k_values)) != top_k_values:
        raise ConfigurationError("formal top_k_values must be sorted and unique")
    if top_k_values[-1] != int(formal["primary_top_k"]):
        raise ConfigurationError("formal top_k_values must end at primary_top_k")
    if top_k_values != CERTIFIED_TOP_K_VALUES:
        raise ConfigurationError("Gate-1c nested top-K is frozen to 16,32,64,128")
    thresholds = [float(value) for value in formal["score_radius_squared_grid"]]
    if len(thresholds) > int(formal["max_radius_steps"]):
        raise ConfigurationError("formal score-radius grid exceeds max_radius_steps")
    if thresholds != sorted(set(thresholds)) or thresholds[0] <= 1:
        raise ConfigurationError("formal score-radius grid must be increasing and above one")
    if len(confirmation) != int(formal["minimum_confirmation_instances_per_regime"]):
        raise ConfigurationError(
            "held-out seeds must match minimum_confirmation_instances_per_regime"
        )


def _audit_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    regime = task["regime"]
    seed = int(task["seed"])
    radius_multiplier = float(task["radius_multiplier"])
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
            instance_id=f"{regime['regime_id']}-gate1c-audit-{seed}",
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
        legacy_source = MNGeneralCandidateSource()
        legacy_source.prepare(
            instance,
            {
                "elimination": elimination,
                "max_n": int(config.get("real_lattice_n_max", 80)),
                "scaling_c": float(config["scaling_c"]),
                "beta": int(config["beta"]),
                "beta_max": int(config["beta_max"]),
                "radius_multiplier": radius_multiplier,
                "enumeration_node_limit": int(config["enumeration_node_limit"]),
                "reduction_seed": seed,
            },
        )
        legacy = legacy_source.generate(
            int(config["legacy_raw_solution_limit"]),
            np.random.default_rng(seed),
        )
        if not bool(legacy.metadata.get("enumeration_complete_within_bounds", False)):
            raise RuntimeError("RESOURCE_LIMIT: legacy audit ball was not complete")
        absolute_radius_squared = float(legacy.metadata["enumeration_radius_squared"])
        constrained = constrained_candidates_within_radius(
            instance,
            elimination,
            radius_squared=absolute_radius_squared,
            beta=int(config["beta"]),
            beta_max=int(config["beta_max"]),
            reduction_seed=seed,
            node_limit=int(config["enumeration_node_limit"]),
            max_slice_vectors=int(config["constrained_slice_vector_limit"]),
        )
        if not bool(constrained.metadata.get("enumeration_complete_within_bounds", False)):
            raise RuntimeError("RESOURCE_LIMIT: constrained audit ball was not complete")
        legacy_set = set(legacy.candidates)
        constrained_set = set(constrained.candidates)
        legacy_scores = {
            candidate: int(round(-float(score)))
            for candidate, score in zip(legacy.candidates, legacy.scores)
        }
        constrained_scores = {
            candidate: int(round(-float(score)))
            for candidate, score in zip(constrained.candidates, constrained.scores)
        }
        score_mismatches = sorted(
            candidate
            for candidate in legacy_set & constrained_set
            if legacy_scores[candidate] != constrained_scores[candidate]
        )
        truth = tuple(instance.s[index] % instance.q for index in elimination.residual_indices)
        row = {
            "task_id": task["task_id"],
            "protocol_role": "GATE1C_BACKEND_EQUIVALENCE_AUDIT_ONLY",
            "regime_id": regime["regime_id"],
            "seed": seed,
            "radius_multiplier": radius_multiplier,
            "absolute_radius_squared": absolute_radius_squared,
            "beta": int(config["beta"]),
            "legacy_complete": True,
            "constrained_complete": True,
            "candidate_sets_equal": legacy_set == constrained_set,
            "canonical_scores_equal": not score_mismatches and legacy_set == constrained_set,
            "legacy_only_count": len(legacy_set - constrained_set),
            "constrained_only_count": len(constrained_set - legacy_set),
            "score_mismatch_count": len(score_mismatches),
            "legacy_only_examples": [list(value) for value in sorted(legacy_set - constrained_set)[:3]],
            "constrained_only_examples": [
                list(value) for value in sorted(constrained_set - legacy_set)[:3]
            ],
            "score_mismatch_examples": [list(value) for value in score_mismatches[:3]],
            "legacy_candidate_count": len(legacy.candidates),
            "constrained_candidate_count": len(constrained.candidates),
            "legacy_truth_present": truth in legacy_set,
            "constrained_truth_present": truth in constrained_set,
            "legacy_raw_vector_count": len(legacy.metadata.get("enumerated_exact_norms", [])),
            "legacy_embedding_vector_count": int(
                legacy.metadata.get("eligible_embedding_vector_count", 0)
            ),
            "constrained_raw_slice_vector_count": int(
                constrained.metadata.get("raw_slice_vector_count", 0)
            ),
            "constrained_duplicate_residual_count": int(
                constrained.metadata.get("duplicate_residual_count", 0)
            ),
            "legacy_enumeration_nodes": int(legacy.metadata.get("enumeration_nodes", 0)),
            "constrained_enumeration_nodes": int(
                constrained.metadata.get("enumeration_nodes", 0)
            ),
            "fixed_embedding_coordinate": int(
                constrained.metadata.get("fixed_embedding_coordinate", 0)
            ),
            "prior_selectivity_evaluated": False,
            "peak_memory": process_peak_rss_bytes(),
        }
        return {"task_id": task["task_id"], "row": row, "failure": None}
    except Exception as exc:
        resource_limited = isinstance(exc, (TimeoutError, MemoryError)) or "RESOURCE_LIMIT" in str(exc)
        return {
            "task_id": task["task_id"],
            "row": None,
            "failure": {
                "task_id": task["task_id"],
                "protocol_role": "GATE1C_BACKEND_EQUIVALENCE_AUDIT_ONLY",
                "regime_id": regime.get("regime_id"),
                "seed": seed,
                "radius_multiplier": radius_multiplier,
                "classification": (
                    "RESOURCE_LIMIT"
                    if resource_limited
                    else ("UNAVAILABLE" if isinstance(exc, BackendUnavailable) else "FAILED_AUDIT_TASK")
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def _build_confirmation_config(
    config: dict[str, Any],
    *,
    audit_run_id: str,
    audit_decision_path: Path,
    gate0_path: Path,
    phase_minus1_path: Path,
) -> dict[str, Any]:
    formal = dict(config["formal_gate1c"])
    formal["calibration_seeds"] = [int(value) for value in config["calibration_seeds"]]
    formal["confirmation_seeds"] = [int(value) for value in config["confirmation_seeds"]]
    formal["instance_seeds"] = list(formal["confirmation_seeds"])
    formal["instances"] = len(formal["confirmation_seeds"])
    formal["regimes"] = [dict(value) for value in config["regimes"]]
    formal["gate0_decision"] = str(gate0_path.resolve())
    formal["phase_minus1_decision"] = str(phase_minus1_path.resolve())
    formal["backend_audit_run_id"] = audit_run_id
    formal["backend_audit_artifact"] = str(audit_decision_path.resolve())
    formal["backend_audit_sha256"] = hashlib.sha256(
        audit_decision_path.read_bytes()
    ).hexdigest()
    formal["protocol_revision"] = "ONE_SHOT_GATE1C_CERTIFIED_CONSTRAINED_TOP_K"
    formal["confirmation_uses_prior_for_parameter_selection"] = False
    require_common_limits(formal)
    return formal


def run(config_path: str | Path | dict[str, Any]) -> Path:
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    regimes = _regimes(config)
    _validate_protocol(config, regimes)
    gate0_path = _require_gate0(config)
    phase_minus1_verified, phase_minus1_path = _resolve_phase_minus1(config)
    if not phase_minus1_verified or phase_minus1_path is None:
        raise RuntimeError("Phase -1 PASS decision not found; Gate-1c audit fails closed")
    context = RunContext.create("phase1c_audit", config, PROJECT_ROOT)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    tasks: list[dict[str, Any]] = []
    for regime in regimes:
        for radius_multiplier in config["audit_radius_multipliers"]:
            for seed in config["calibration_seeds"]:
                task_id = (
                    f"gate1c-audit:{regime['regime_id']}:"
                    f"radius={float(radius_multiplier)}:{int(seed)}"
                )
                tasks.append(
                    {
                        "task_id": task_id,
                        "config": serializable,
                        "regime": regime,
                        "radius_multiplier": float(radius_multiplier),
                        "seed": int(seed),
                    }
                )
    completed = {
        str(row["task_id"])
        for row in read_jsonl(context.run_dir / "checkpoints.jsonl")
    }
    pending = [task for task in tasks if task["task_id"] not in completed]
    for result in bounded_pool_imap_unordered(
        _audit_task,
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
    if not trial_frame.empty and "task_id" in trial_frame:
        trial_frame = trial_frame.drop_duplicates("task_id", keep="first")
    if not failure_frame.empty and "task_id" in failure_frame:
        failure_frame = failure_frame.drop_duplicates("task_id", keep="first")
    trial_frame.to_csv(context.run_dir / "audit_summary.csv", index=False)
    all_equivalent = bool(
        len(trial_frame) == len(tasks)
        and failure_frame.empty
        and trial_frame["candidate_sets_equal"].astype(bool).all()
        and trial_frame["canonical_scores_equal"].astype(bool).all()
        and trial_frame["legacy_complete"].astype(bool).all()
        and trial_frame["constrained_complete"].astype(bool).all()
        and (trial_frame["fixed_embedding_coordinate"].astype(int) == -1).all()
    ) if not trial_frame.empty else False
    if all_equivalent:
        status = "PASS"
    elif not failure_frame.empty or len(trial_frame) != len(tasks):
        status = "BLOCKED_AUDIT_INCOMPLETE"
    else:
        status = "FAIL_BACKEND_EQUIVALENCE"
    decision = {
        "gate": "Gate 1c backend correctness audit",
        "status": status,
        "protocol_classification": (
            "AUDIT_PASS" if status == "PASS" else "STOP_CURRENT_PROTOCOL"
        ),
        "planned_tasks": len(tasks),
        "completed_rows": len(trial_frame),
        "failure_count": len(failure_frame),
        "candidate_set_mismatch_count": (
            int((~trial_frame["candidate_sets_equal"].astype(bool)).sum())
            if not trial_frame.empty
            else None
        ),
        "canonical_score_mismatch_count": (
            int((~trial_frame["canonical_scores_equal"].astype(bool)).sum())
            if not trial_frame.empty
            else None
        ),
        "prior_selectivity_evaluated": False,
        "confirmation_seeds_used": False,
        "gate0_decision": str(gate0_path),
        "phase_minus1_decision": str(phase_minus1_path),
        "rule": (
            "all complete legacy radius balls equal the constrained t=-1 canonical candidate balls, including B2 scores"
        ),
        "protocol_deviation": None,
    }
    decision_path = context.run_dir / "backend_audit_decision.json"
    atomic_write_text(decision_path, json.dumps(decision, indent=2) + "\n")
    if status == "PASS":
        confirmation = _build_confirmation_config(
            config,
            audit_run_id=context.run_id,
            audit_decision_path=decision_path,
            gate0_path=gate0_path,
            phase_minus1_path=phase_minus1_path,
        )
        formal_path = context.run_dir / "gate1c_confirmation_config.yaml"
        atomic_write_text(formal_path, yaml.safe_dump(confirmation, sort_keys=False))
        atomic_write_text(
            context.run_dir / "RUN_CONFIRMATION.md",
            "# Gate-1c confirmation\n\n"
            "First verify that the complete server test suite executes every fpylll test and that backend_audit_decision.json is PASS. Then launch the held-out confirmation separately:\n\n"
            f"```bash\npython experiments/phase1c_topk.py --config \"{formal_path}\"\n```\n\n"
            "Do not edit the generated YAML and do not run confirmation more than once.\n",
        )
    else:
        atomic_write_text(
            context.run_dir / "STOP.md",
            "# STOP_CURRENT_PROTOCOL\n\n"
            "- failed stage: Gate-1c backend equivalence audit\n"
            "- action: do not use the constrained source for confirmation and do not enter Phase 2\n",
        )
    atomic_write_text(
        context.run_dir / "report.md",
        "# Gate-1c backend audit report\n\n"
        f"- planned tasks: {len(tasks)}\n"
        f"- completed equivalence rows: {len(trial_frame)}\n"
        f"- failures/resource limits: {len(failure_frame)}\n"
        f"- audit status: {status}\n\n"
        "No exact-prior pass outcome is evaluated in this stage.\n",
    )
    context.write_manifest(
        status="COMPLETED",
        derived_seeds={
            "audit_only": config["calibration_seeds"],
            "reserved_confirmation": config["confirmation_seeds"],
        },
    )
    return context.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="existing phase1c_audit run directory to resume")
    arguments = parser.parse_args()
    if arguments.resume:
        config = load_yaml(arguments.config)
        config["_resume_dir"] = arguments.resume
        print(run(config))
        return
    print(run(arguments.config))


if __name__ == "__main__":
    main()
