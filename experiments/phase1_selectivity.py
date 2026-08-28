"""Phase 1: baseline-relative selectivity, uniformity, and correlation."""

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

from affine_hints.candidates.mn_general import BackendUnavailable, MNGeneralCandidateSource
from affine_hints.candidates.synthetic import NormShellProxySource, PriorSupportedSyntheticSource, SyntheticUniformSource
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
from affine_hints.diagnostics import (
    candidate_correlation_diagnostics,
    difference_valuation,
    projective_class,
    uniformity_diagnostics,
    wilson_interval,
)
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.modular import centered_vector, first_prime_divisor
from affine_hints.posterior import lwe_error
from affine_hints.priors import CBDPrior, FixedWeightTernaryPrior, UniformTernaryPrior, make_prior
from affine_hints.resources import apply_unix_task_limits, bounded_pool_imap_unordered
from affine_hints.statistics import cluster_bootstrap_interval


def _latest_gate0_decision() -> Path | None:
    candidates = sorted((PROJECT_ROOT / "results" / "phase0").glob("*/gate0_decision.json"), reverse=True)
    return candidates[0] if candidates else None


def _require_gate0(config: dict[str, Any]) -> Path:
    configured = config.get("gate0_decision", "AUTO_LATEST")
    path = _latest_gate0_decision() if configured == "AUTO_LATEST" else Path(str(configured)).resolve()
    if path is None or not path.exists():
        raise RuntimeError("Gate 0 PASS decision not found; Phase 1 fails closed")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("status") != "PASS":
        raise RuntimeError(f"Gate 0 is not PASS: {path}")
    return path


def _resolve_phase_minus1(config: dict[str, Any]) -> tuple[bool, Path | None]:
    """Resolve a real Phase -1 PASS artifact instead of trusting a config flag."""

    configured = config.get("phase_minus1_decision", "AUTO_LATEST")
    if configured == "AUTO_LATEST":
        candidates = sorted(
            (PROJECT_ROOT / "results" / "phase_minus1").glob("*/gate_minus1_decision.json"),
            reverse=True,
        )[:1]
    else:
        candidates = [Path(str(configured)).resolve()]
    for path in candidates:
        try:
            decision = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if decision.get("status") == "PASS":
            return True, path
    return False, None


def _require_geometry_selection_binding(config: dict[str, Any]) -> Path | None:
    """Verify that a generated formal config still matches its calibration."""

    artifact_value = config.get("geometry_selection_artifact")
    expected_hash = config.get("geometry_selection_sha256")
    if artifact_value is None and expected_hash is None:
        return None
    if artifact_value is None or expected_hash is None:
        raise ConfigurationError(
            "formal geometry binding requires both artifact path and SHA-256"
        )
    artifact = Path(str(artifact_value)).resolve()
    try:
        payload = artifact.read_bytes()
        selection = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"cannot read the frozen geometry selection: {artifact}"
        ) from exc
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != str(expected_hash):
        raise ConfigurationError("frozen geometry selection SHA-256 mismatch")
    if selection.get("status") != "SELECTED":
        raise ConfigurationError("formal Gate 1 requires a SELECTED geometry artifact")
    if selection.get("selection_uses_prior_pass_rate") is not False:
        raise ConfigurationError("geometry selection must be prior-pass-rate blind")
    selected = selection.get("selected_theta")
    if not isinstance(selected, dict):
        raise ConfigurationError("geometry selection has no selected_theta mapping")
    r_elim_values = config.get("r_elim")
    if not isinstance(r_elim_values, list) or len(r_elim_values) != 1:
        raise ConfigurationError("formal geometry binding requires exactly one r_elim")
    exact_fields = {
        "beta": int(config["beta"]),
        "candidate_budget": int(config["max_candidates"]),
        "r_elim": r_elim_values[0],
        "pivot_strategy": str(config["pivot_strategy"]),
    }
    for key, expected in exact_fields.items():
        if selected.get(key) != expected:
            raise ConfigurationError(f"formal config differs from selected geometry: {key}")
    float_fields = {
        "scaling_c": float(config["mn_scaling_c"]),
        "radius_multiplier": float(config["radius_multiplier"]),
    }
    for key, expected in float_fields.items():
        try:
            observed = float(selected[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid selected geometry field: {key}") from exc
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError(f"formal config differs from selected geometry: {key}")
    return artifact


def _source_factory(name: str, prior: Any, elimination: AffineCosetElimination, cell: dict[str, Any]) -> Any:
    if name == "prior_supported":
        return PriorSupportedSyntheticSource(prior)
    if name == "uniform_residual":
        return SyntheticUniformSource()
    if name == "norm_shell_proxy":
        if isinstance(prior, FixedWeightTernaryPrior):
            second_moment = prior.second_moment_for_length(int(cell["n"]))
        else:
            second_moment = prior.second_moment()
        return NormShellProxySource(second_moment=second_moment, shell_tolerance=float(cell.get("shell_tolerance", 0.20)))
    if name == "mn_general":
        return MNGeneralCandidateSource()
    raise ValueError(f"unknown candidate source: {name}")


def _coordinate_support(prior: Any) -> tuple[int, ...]:
    if isinstance(prior, FixedWeightTernaryPrior):
        return (-1, 0, 1)
    return tuple(int(value) for value in prior.support)


def _evaluate_list(
    *,
    candidate_list: Any,
    instance: Any,
    elimination: AffineCosetElimination,
    prior: Any,
    rng: np.random.Generator,
    generation_time: float,
    record_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    A_star, b_star = elimination.transform_lwe(instance.A, instance.b)
    true_residual = tuple(instance.s[i] % instance.q for i in elimination.residual_indices)
    support = set(_coordinate_support(prior))
    passes: list[bool] = []
    differences: list[tuple[int, ...]] = []
    pivot_images: list[tuple[int, ...]] = []
    records: list[dict[str, Any]] = []
    false_count = 0
    false_survivors = 0
    prime_modulus = first_prime_divisor(instance.q) == instance.q
    if isinstance(prior, UniformTernaryPrior):
        constant_supported_log_prior = -instance.n * math.log(3.0)
    elif isinstance(prior, FixedWeightTernaryPrior):
        constant_supported_log_prior = -math.log(prior.support_size(instance.n))
    else:
        constant_supported_log_prior = None
    predicate_started = time.monotonic()
    true_positions_before: list[int] = []
    true_positions_after: list[int] = []
    prior_scores: list[float] = []
    true_flags: list[bool] = []
    sorted_indices = sorted(range(len(candidate_list.candidates)), key=lambda i: candidate_list.scores[i], reverse=True)
    before_rank = {index: rank + 1 for rank, index in enumerate(sorted_indices)}
    surviving_indices: list[int] = []
    for candidate_id, residual in enumerate(candidate_list.candidates):
        full_mod = elimination.reconstruct(residual)
        full = tuple(centered_vector(full_mod, instance.q))
        pivot = tuple(full[i] for i in elimination.pivot_indices)
        support_pass = prior.in_support(full)
        exact_log_prior = (
            (constant_supported_log_prior if constant_supported_log_prior is not None else prior.log_prob(full))
            if support_pass
            else -math.inf
        )
        pivot_support_pass = all(value in support for value in pivot)
        remaining_pass = elimination.remaining_hints_pass(residual)
        fixed_weight_pass = support_pass if isinstance(prior, FixedWeightTernaryPrior) else True
        exact_pass = support_pass and remaining_pass
        is_true = tuple(int(value) % instance.q for value in residual) == true_residual
        prior_scores.append(float(exact_log_prior))
        true_flags.append(is_true)
        difference = tuple((true_residual[j] - int(residual[j])) % instance.q for j in range(len(residual)))
        if exact_pass:
            surviving_indices.append(candidate_id)
        if is_true:
            true_positions_before.append(before_rank[candidate_id])
        else:
            false_count += 1
            false_survivors += int(exact_pass)
            passes.append(exact_pass)
            differences.append(difference)
            pivot_images.append(tuple(value % instance.q for value in pivot))
        if candidate_id < record_limit:
            residual_error = lwe_error(A_star, b_star, residual, instance.q)
            residual_centered = tuple(centered_vector(residual, instance.q))
            residual_plus = residual_centered.count(1)
            residual_minus = residual_centered.count(-1)
            if isinstance(prior, FixedWeightTernaryPrior):
                required_plus = prior.h_plus - residual_plus
                required_minus = prior.h_minus - residual_minus
                required_zero = len(pivot) - required_plus - required_minus
                residual_supported = all(value in (-1, 0, 1) for value in residual_centered)
                if not residual_supported or min(required_plus, required_minus, required_zero) < 0:
                    conditional_count = 0
                else:
                    conditional_count = math.comb(len(pivot), required_plus) * math.comb(
                        len(pivot) - required_plus,
                        required_minus,
                    )
                conditional_probability = conditional_count / (instance.q ** len(pivot))
            else:
                required_plus = required_minus = required_zero = conditional_count = None
                conditional_probability = None
            records.append(
                {
                    "instance_id": instance.instance_id,
                    "candidate_id": candidate_id,
                    "baseline": {
                        "MNGeneralCandidateSource": "B2/B4",
                        "PriorSupportedSyntheticSource": "B1/B3_SYNTHETIC_REFERENCE",
                        "NormShellProxySource": "B2_PROXY",
                        "SyntheticUniformSource": "UNIFORM_RESIDUAL_DIAGNOSTIC",
                        "PosteriorWeightedSyntheticSource": "POSTERIOR_DIAGNOSTIC",
                    }.get(candidate_list.source, "SYNTHETIC_DIAGNOSTIC"),
                    "source": candidate_list.source,
                    "is_true": is_true,
                    "baseline_score": candidate_list.scores[candidate_id],
                    "candidate_vector_norm_squared": (
                        -float(candidate_list.scores[candidate_id])
                        if candidate_list.source == "MNGeneralCandidateSource"
                        else None
                    ),
                    "lwe_residual_norm": math.sqrt(sum(value * value for value in residual_error)),
                    "residual_secret": list(residual),
                    "reconstructed_pivot": list(pivot),
                    "exact_log_prior": exact_log_prior,
                    "support_pass": support_pass,
                    "pivot_support_pass": pivot_support_pass,
                    "fixed_weight_pass": fixed_weight_pass,
                    "remaining_hint_pass": remaining_pass,
                    "residual_plus_count": residual_plus,
                    "residual_minus_count": residual_minus,
                    "required_pivot_plus_count": required_plus,
                    "required_pivot_minus_count": required_minus,
                    "required_pivot_zero_count": required_zero,
                    "D_I": conditional_count,
                    "D_I_over_q_to_r_elim": conditional_probability,
                    "difference_valuation": difference_valuation(difference, instance.q) if not is_true else None,
                    "projective_class": projective_class(difference, instance.q) if (not is_true and prime_modulus) else None,
                    "generation_time": generation_time,
                }
            )
    surviving_set = set(surviving_indices)
    after_sorted = [index for index in sorted_indices if index in surviving_set]
    after_rank = {index: rank + 1 for rank, index in enumerate(after_sorted)}
    for index in surviving_indices:
        if tuple(candidate_list.candidates[index]) == true_residual:
            true_positions_after.append(after_rank[index])
    true_prior_ranks: list[int] = []
    prior_auc = math.nan
    prior_average_precision = math.nan
    if isinstance(prior, CBDPrior) and prior_scores:
        finite_floor = min((value for value in prior_scores if math.isfinite(value)), default=0.0) - 1.0
        rankable_prior_scores = np.asarray(
            [value if math.isfinite(value) else finite_floor for value in prior_scores],
            dtype=float,
        )
        prior_sorted = sorted(range(len(prior_scores)), key=lambda i: rankable_prior_scores[i], reverse=True)
        prior_rank = {index: rank + 1 for rank, index in enumerate(prior_sorted)}
        true_prior_ranks = [prior_rank[index] for index, flag in enumerate(true_flags) if flag]
        positive_count = sum(true_flags)
        negative_count = len(true_flags) - positive_count
    else:
        rankable_prior_scores = np.asarray([], dtype=float)
        prior_sorted = []
        positive_count = negative_count = 0
    if positive_count and negative_count:
        from scipy.stats import rankdata

        ascending_ranks = rankdata(rankable_prior_scores, method="average")
        rank_sum = float(sum(ascending_ranks[index] for index, flag in enumerate(true_flags) if flag))
        prior_auc = (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
        positives_seen = 0
        precision_terms: list[float] = []
        for rank, index in enumerate(prior_sorted, start=1):
            if true_flags[index]:
                positives_seen += 1
                precision_terms.append(positives_seen / rank)
        prior_average_precision = float(np.mean(precision_terms))
    predicate_time = time.monotonic() - predicate_started
    predicate_time_per_candidate = predicate_time / len(candidate_list.candidates) if candidate_list.candidates else math.nan
    for record in records:
        record["predicate_time"] = predicate_time_per_candidate
    interval = wilson_interval(false_survivors, false_count)
    alpha = false_survivors / false_count if false_count else math.nan
    if prime_modulus:
        correlations = candidate_correlation_diagnostics(differences, passes, instance.q)
    else:
        correlations = {
            "number_projective_classes": None,
            "largest_class_size": None,
            "fraction_collinear_pairs": None,
            "difference_valuation_min": min(
                (
                    min(
                        (
                            _valuation(value, first_prime_divisor(instance.q), instance.q)
                            for value in difference
                            if value % instance.q
                        ),
                        default=0,
                    )
                    for difference in differences
                ),
                default=None,
            ),
        }
    uniformity = (
        uniformity_diagnostics(
            pivot_images,
            instance.q,
            rng,
            projections=16,
            include_histograms=instance.q >= 1000,
        )
        if pivot_images
        else {"sample_count": 0}
    )
    summary = {
        "candidate_count": len(candidate_list.candidates),
        "false_candidate_count": false_count,
        "false_survivor_count": false_survivors,
        "alpha": alpha,
        "alpha_wilson_low": interval[0],
        "alpha_wilson_high": interval[1],
        "alpha_interval_method": "Wilson 95% diagnostic",
        "marginal_bits": -math.log2(alpha) if alpha > 0 else None,
        "marginal_bits_lower_95": -math.log2(interval[1]) if interval[1] > 0 else math.inf,
        "marginal_bits_upper_95": -math.log2(interval[0]) if interval[0] > 0 else math.inf,
        "any_false_survivor": bool(false_survivors),
        "true_candidate_present": bool(true_positions_before),
        "true_rank_before": min(true_positions_before, default=None),
        "true_rank_after": min(true_positions_after, default=None),
        "true_rank_prior_only": min(true_prior_ranks, default=None),
        "prior_auc_candidate_level_diagnostic": prior_auc,
        "prior_average_precision_candidate_level_diagnostic": prior_average_precision,
        "true_retention": bool(true_positions_after),
        "generation_time": generation_time,
        "predicate_time": predicate_time,
        "predicate_time_per_candidate": predicate_time_per_candidate,
        "metadata": candidate_list.metadata,
        "uniformity": uniformity,
        "correlations": correlations,
    }
    return summary, records


def _valuation(value: int, p: int, q: int) -> int:
    value %= q
    if not value:
        return 0
    result = 0
    while value % p == 0:
        result += 1
        value //= p
    return result


def _run_instance(task: dict[str, Any]) -> dict[str, Any]:
    config, cell, instance_index, seed = task["config"], task["cell"], task["instance_index"], task["seed"]
    protocol_seed = task.get("protocol_seed")
    started = time.monotonic()
    if "mn_general" in cell["sources"]:
        apply_unix_task_limits(
            max_wall_seconds=int(config["max_wall_time"]),
            max_address_space=config.get("max_RSS", "10GB"),
        )
    rng = np.random.default_rng(seed)
    n, m, q, r = (int(cell[key]) for key in ("n", "m", "q", "r"))
    prior = make_prior(cell["prior"])
    hint = generate_hint_matrix(
        rng,
        n=n,
        r=r,
        q=q,
        hint_class=str(cell["H_class"]),
        parameters=dict(cell.get("H_parameters", {})),
    )
    instance = generate_synthetic_lwe(
        rng,
        n=n,
        m=m,
        q=q,
        secret_prior=prior,
        H=hint.H,
        instance_id=f"{cell['cell_id']}-i{instance_index:04d}",
    )
    outputs: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    child_sequences = np.random.SeedSequence(seed).spawn(len(cell["r_elim"]) * len(cell["sources"]))
    child_index = 0
    for r_elim in (int(value) for value in cell["r_elim"]):
        elimination = AffineCosetElimination.build(
            instance.H,
            instance.ell,
            q,
            r_elim,
            strategy=str(cell.get("pivot_strategy", "first_unit_minor")),
            rng=rng,
            max_combinations=int(config.get("max_pivot_combinations", 100_000)),
        )
        for source_name in cell["sources"]:
            source_seed = int(child_sequences[child_index].generate_state(1, dtype=np.uint64)[0])
            child_index += 1
            source_rng = np.random.default_rng(source_seed)
            try:
                source = _source_factory(source_name, prior, elimination, cell)
                baseline_config = {
                    "elimination": elimination,
                    "max_n": int(config.get("real_lattice_n_max", 80)),
                    "scaling_c": float(cell.get("mn_scaling_c", 1.0)),
                    "beta": int(cell.get("beta", 0)),
                    "beta_max": int(config["beta_max"]),
                    "radius_multiplier": float(cell.get("radius_multiplier", 1.05)),
                    "enumeration_node_limit": int(config["enumeration_node_limit"]),
                    "reduction_seed": source_seed,
                }
                source.prepare(instance, baseline_config)
                generation_started = time.monotonic()
                candidate_list = source.generate(int(config["max_candidates"]), source_rng)
                generation_time = time.monotonic() - generation_started
                evaluated, records = _evaluate_list(
                    candidate_list=candidate_list,
                    instance=instance,
                    elimination=elimination,
                    prior=prior,
                    rng=source_rng,
                    generation_time=generation_time,
                    record_limit=int(config.get("candidate_record_limit_per_list", 0)),
                )
                outputs.append(
                    {
                        "instance_id": instance.instance_id,
                        "instance_index": instance_index,
                        "seed": seed,
                        "protocol_seed": protocol_seed,
                        "source_seed": source_seed,
                        "n": n,
                        "m": m,
                        "q": q,
                        "r": r,
                        "r_elim": r_elim,
                        "r_check": r - r_elim,
                        "prior": json.dumps(cell["prior"], sort_keys=True) if isinstance(cell["prior"], dict) else cell["prior"],
                        "H_class": cell["H_class"],
                        "H_metadata": hint.metadata,
                        "pivot_strategy": cell.get("pivot_strategy", "first_unit_minor"),
                        "pivot_fill": elimination.pivot_choice.fill,
                        "pivot_mean_last_nonzero": elimination.pivot_choice.mean_last_nonzero,
                        "pivot_max_last_nonzero": elimination.pivot_choice.max_last_nonzero,
                        "pivot_examined_pairs": elimination.pivot_choice.examined_pairs,
                        "pivot_total_pairs": elimination.pivot_choice.total_pairs,
                        "pivot_search_truncated": elimination.pivot_choice.search_truncated,
                        "source": candidate_list.source,
                        "source_requested": source_name,
                        "evidence_label": (
                            "TOY REAL-LATTICE CANDIDATE DISTRIBUTION"
                            if candidate_list.source == "MNGeneralCandidateSource"
                            else candidate_list.metadata.get("label", "SYNTHETIC STATISTICAL EXPERIMENT")
                        ),
                        "peak_memory": process_peak_rss_bytes(),
                        **evaluated,
                    }
                )
                for record in records:
                    record.update({"r_elim": r_elim, "H_class": cell["H_class"], "prior": str(cell["prior"])})
                candidate_records.extend(records)
            except Exception as exc:
                resource_limited = isinstance(exc, (TimeoutError, MemoryError)) or "RESOURCE_LIMIT" in str(exc)
                failures.append(
                    {
                        "instance_id": instance.instance_id,
                        "source_requested": source_name,
                        "r_elim": r_elim,
                        "seed": seed,
                        "protocol_seed": protocol_seed,
                        "source_seed": source_seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "classification": (
                            "RESOURCE_LIMIT"
                            if resource_limited
                            else ("UNAVAILABLE" if isinstance(exc, BackendUnavailable) else "FAILED_INSTANCE")
                        ),
                    }
                )
        if time.monotonic() - started > float(config["max_wall_time"]):
            failures.append(
                {
                    "instance_id": instance.instance_id,
                    "classification": "RESOURCE_LIMIT",
                    "error": "per-instance max_wall_time exceeded",
                }
            )
            break
    return {
        "task_id": task["task_id"],
        "trials": outputs,
        "candidates": candidate_records,
        "failures": failures,
        "seed": seed,
        "protocol_seed": protocol_seed,
    }


def _instance_seed_plan(
    config: dict[str, Any],
    cell_count: int,
) -> list[tuple[int, int | None]]:
    """Return per-cell task seeds while preserving explicit protocol seeds.

    Legacy configurations retain their prior SeedSequence behavior. A formal
    configuration can instead provide one reserved ``instance_seed`` per
    instance; each is domain-separated by cell index before it reaches the
    instance generator. The original protocol seed is recorded alongside the
    derived task seed in every output row.
    """

    instance_count = int(config["instances"])
    if instance_count <= 0:
        raise ConfigurationError("instances must be positive")
    if cell_count <= 0:
        raise ConfigurationError("Phase 1 expands to no experiment cells")
    explicit = config.get("instance_seeds")
    if explicit is None:
        sequence = np.random.SeedSequence(int(config["master_seed"]))
        children = sequence.spawn(cell_count * instance_count)
        return [
            (int(child.generate_state(1, dtype=np.uint64)[0]), None)
            for child in children
        ]

    protocol_seeds = [int(value) for value in explicit]
    if len(protocol_seeds) != instance_count:
        raise ConfigurationError(
            "instance_seeds length must equal instances for a formal seeded run"
        )
    if len(set(protocol_seeds)) != len(protocol_seeds):
        raise ConfigurationError("instance_seeds must be unique")
    calibration_seeds = {int(value) for value in config.get("calibration_seeds", [])}
    if calibration_seeds & set(protocol_seeds):
        raise ConfigurationError("instance_seeds must be disjoint from calibration_seeds")
    if "confirmation_seeds" in config:
        confirmation_seeds = [int(value) for value in config["confirmation_seeds"]]
        if protocol_seeds != confirmation_seeds:
            raise ConfigurationError(
                "instance_seeds must exactly match the frozen confirmation_seeds"
            )

    plan: list[tuple[int, int | None]] = []
    derived_seen: set[int] = set()
    for cell_index in range(cell_count):
        for protocol_seed in protocol_seeds:
            seed = int(
                np.random.SeedSequence([protocol_seed, cell_index]).generate_state(
                    1,
                    dtype=np.uint64,
                )[0]
            )
            if seed in derived_seen:
                raise ConfigurationError("derived instance seed collision")
            derived_seen.add(seed)
            plan.append((seed, protocol_seed))
    return plan


def _expand_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    index = 0
    for n in config["n"]:
        for q in config["q"]:
            for r in config["r"]:
                if int(r) > int(n):
                    continue
                for prior in config["priors"]:
                    for hint_class in config["H_classes"]:
                        index += 1
                        cells.append(
                            {
                                "cell_id": f"c{index:04d}",
                                "n": int(n),
                                "m": int(round(float(config.get("m_over_n", 1.0)) * int(n))),
                                "q": int(q),
                                "r": int(r),
                                "prior": prior,
                                "H_class": hint_class,
                                "H_parameters": config.get("H_parameters", {}).get(hint_class, {}),
                                "r_elim": [int(r) if value == "all" else int(value) for value in config["r_elim"] if (value == "all" or int(value) <= int(r))],
                                "sources": list(config["sources"]),
                                "pivot_strategy": config.get("pivot_strategy", "first_unit_minor"),
                                "shell_tolerance": config.get("shell_tolerance", 0.20),
                                "mn_scaling_c": config.get("mn_scaling_c", 1.0),
                                "beta": config.get("beta", 0),
                                "radius_multiplier": config.get("radius_multiplier", 1.05),
                            }
                        )
    return cells


def _marginal_bits(value: Any) -> float | None:
    """Convert a finite probability to bits without turning missing data into infinity."""

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        return None
    if probability == 0.0:
        return math.inf
    return -math.log2(probability)


def _summarize(
    frame: pd.DataFrame,
    bootstrap_seed: int,
    replicates: int,
    *,
    minimum_valid_alpha_lists: int = 30,
    minimum_false_candidates_per_group: int = 300,
) -> pd.DataFrame:
    keys = ["n", "m", "q", "r", "r_elim", "r_check", "prior", "H_class", "source", "source_requested", "evidence_label"]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for values, group in frame.groupby(keys, dropna=False):
        alpha_array = group["alpha"].astype(float).to_numpy()
        valid_alpha_mask = np.isfinite(alpha_array)
        finite_alphas = alpha_array[valid_alpha_mask]
        valid_alpha_lists = int(valid_alpha_mask.sum())
        low, high = cluster_bootstrap_interval(finite_alphas, rng, replicates=replicates)
        mean_alpha = float(np.mean(finite_alphas)) if valid_alpha_lists else math.nan
        all_false_counts = group["false_candidate_count"].astype(float).to_numpy()
        false_candidates_total = int(np.sum(all_false_counts))
        lists_with_false_candidates = int(np.sum(all_false_counts > 0))
        valid_group = group.loc[valid_alpha_mask]
        false_counts = valid_group["false_candidate_count"].astype(float).to_numpy()
        survivor_counts = valid_group["false_survivor_count"].astype(float).to_numpy()
        any_survivor_count = int(valid_group["any_false_survivor"].astype(bool).sum())
        if valid_alpha_lists:
            any_probability = any_survivor_count / valid_alpha_lists
            any_low, any_high = wilson_interval(any_survivor_count, valid_alpha_lists)
        else:
            any_probability = math.nan
            any_low = any_high = math.nan
        observed_survivor_variance = (
            float(np.var(survivor_counts, ddof=1))
            if len(survivor_counts) > 1
            else (0.0 if len(survivor_counts) == 1 else math.nan)
        )
        # Under independent Bernoulli passes and possibly unequal list sizes,
        # total variance is E[N]a(1-a) + Var[N]a^2.
        count_variance = (
            float(np.var(false_counts, ddof=1))
            if len(false_counts) > 1
            else (0.0 if len(false_counts) == 1 else math.nan)
        )
        independent_survivor_variance = math.nan
        if valid_alpha_lists:
            independent_survivor_variance = (
                float(np.mean(false_counts)) * mean_alpha * (1.0 - mean_alpha)
                + count_variance * mean_alpha * mean_alpha
            )
        alpha_estimable = bool(
            valid_alpha_lists
            and false_candidates_total > 0
            and math.isfinite(low)
            and math.isfinite(high)
        )
        underpowered = bool(
            valid_alpha_lists < minimum_valid_alpha_lists
            or false_candidates_total < minimum_false_candidates_per_group
        )
        auc_values = [
            float(value)
            for value in group["prior_auc_candidate_level_diagnostic"]
            if value is not None and math.isfinite(float(value))
        ]
        ap_values = [
            float(value)
            for value in group["prior_average_precision_candidate_level_diagnostic"]
            if value is not None and math.isfinite(float(value))
        ]
        auc_low, auc_high = cluster_bootstrap_interval(auc_values, rng, replicates=replicates)
        ap_low, ap_high = cluster_bootstrap_interval(ap_values, rng, replicates=replicates)
        uniformity_aggregates: dict[str, Any] = {}
        for metric in ("mean_coordinate_tv", "character_bias_max", "character_bias_q95", "collision_rate"):
            metric_values = [
                float(value[metric])
                for value in group["uniformity"]
                if isinstance(value, dict) and value.get(metric) is not None and math.isfinite(float(value[metric]))
            ]
            metric_low, metric_high = cluster_bootstrap_interval(metric_values, rng, replicates=replicates)
            uniformity_aggregates[f"uniformity_{metric}_cluster_mean"] = (
                float(np.mean(metric_values)) if metric_values else None
            )
            uniformity_aggregates[f"uniformity_{metric}_cluster_ci_low"] = metric_low
            uniformity_aggregates[f"uniformity_{metric}_cluster_ci_high"] = metric_high
        rows.append(
            {
                **dict(zip(keys, values)),
                "instances": len(group),
                "valid_alpha_lists": valid_alpha_lists,
                "lists_with_false_candidates": lists_with_false_candidates,
                "false_candidates_total": false_candidates_total,
                "minimum_valid_alpha_lists": minimum_valid_alpha_lists,
                "minimum_false_candidates_per_group": minimum_false_candidates_per_group,
                "alpha_estimable": alpha_estimable,
                "mean_alpha_by_candidate_list": mean_alpha,
                "cluster_bootstrap_alpha_low": low,
                "cluster_bootstrap_alpha_high": high,
                "marginal_bits_from_cluster_mean": _marginal_bits(mean_alpha),
                "marginal_bits_lower_from_cluster_ci": _marginal_bits(high),
                "marginal_bits_upper_from_cluster_ci": _marginal_bits(low),
                "probability_any_false_survivor": any_probability,
                "probability_any_false_survivor_wilson_low": any_low,
                "probability_any_false_survivor_wilson_high": any_high,
                "survivor_variance_across_candidate_lists": observed_survivor_variance,
                "variance_under_independence": independent_survivor_variance,
                "overdispersion_factor": (
                    observed_survivor_variance / independent_survivor_variance
                    if independent_survivor_variance > 0
                    else None
                ),
                "mean_prior_auc_list_cluster": float(np.mean(auc_values)) if auc_values else None,
                "prior_auc_cluster_ci_low": auc_low,
                "prior_auc_cluster_ci_high": auc_high,
                "mean_prior_average_precision_list_cluster": float(np.mean(ap_values)) if ap_values else None,
                "prior_average_precision_cluster_ci_low": ap_low,
                "prior_average_precision_cluster_ci_high": ap_high,
                "weighted_prior_ranking_label": "candidate-level diagnostic; uncertainty clustered by complete list",
                **uniformity_aggregates,
                "all_true_candidates_present": bool(group["true_candidate_present"].all()),
                "true_candidate_presence_rate": float(group["true_candidate_present"].astype(float).mean()),
                "all_present_true_candidates_retained": bool(group.loc[group["true_candidate_present"], "true_retention"].all()),
                "median_candidate_count": float(group["candidate_count"].median()),
                "median_false_candidate_count": float(group["false_candidate_count"].median()),
                "median_generation_time": float(group["generation_time"].median()),
                "median_predicate_time": float(group["predicate_time"].median()),
                "underpowered": underpowered,
                "all_enumerations_complete": all(
                    not isinstance(value, dict) or bool(value.get("enumeration_complete_within_bounds", True))
                    for value in group["metadata"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _structured_h_deviations(summary: pd.DataFrame) -> pd.DataFrame:
    """Contrast each structured-H list cluster with its dense-H reference."""

    if summary.empty or "dense_random" not in set(summary["H_class"]):
        return pd.DataFrame()
    keys = ["n", "m", "q", "r", "r_elim", "r_check", "prior", "source", "source_requested", "evidence_label"]
    rows: list[dict[str, Any]] = []
    for values, group in summary.groupby(keys, dropna=False):
        dense = group[group["H_class"] == "dense_random"]
        if dense.empty:
            continue
        reference = dense.iloc[0]
        for _, structured in group[group["H_class"] != "dense_random"].iterrows():
            structured_alpha = float(structured["mean_alpha_by_candidate_list"])
            dense_alpha = float(reference["mean_alpha_by_candidate_list"])
            structured_low = float(structured["cluster_bootstrap_alpha_low"])
            structured_high = float(structured["cluster_bootstrap_alpha_high"])
            dense_low = float(reference["cluster_bootstrap_alpha_low"])
            dense_high = float(reference["cluster_bootstrap_alpha_high"])
            comparison_estimable = all(
                math.isfinite(value)
                for value in (
                    structured_alpha,
                    dense_alpha,
                    structured_low,
                    structured_high,
                    dense_low,
                    dense_high,
                )
            )
            rows.append(
                {
                    **dict(zip(keys, values)),
                    "H_class": structured["H_class"],
                    "comparison_estimable": comparison_estimable,
                    "dense_alpha": dense_alpha if comparison_estimable else None,
                    "structured_alpha": structured_alpha if comparison_estimable else None,
                    "alpha_difference_structured_minus_dense": (
                        structured_alpha - dense_alpha if comparison_estimable else None
                    ),
                    "alpha_ratio_structured_over_dense": (
                        structured_alpha / dense_alpha
                        if comparison_estimable and dense_alpha > 0
                        else None
                    ),
                    "difference_ci_low_conservative": (
                        structured_low - dense_high if comparison_estimable else None
                    ),
                    "difference_ci_high_conservative": (
                        structured_high - dense_low if comparison_estimable else None
                    ),
                    "comparison_design": "independent list clusters; conservative interval-endpoint contrast",
                }
            )
    return pd.DataFrame(rows)


def _gate_decision(
    config: dict[str, Any],
    summary: pd.DataFrame,
    structured_deviations: pd.DataFrame,
    failures: list[dict[str, Any]],
    *,
    phase_minus1_verified: bool,
    phase_minus1_path: Path | None,
) -> dict[str, Any]:
    eligible = bool(config.get("eligible_for_gate", False)) and not bool(config.get("smoke", False))
    b2 = summary[summary["source"] == "MNGeneralCandidateSource"] if not summary.empty else summary
    b2_failures = [row for row in failures if row.get("source_requested") == "mn_general"]
    unavailable_b2 = bool(b2_failures)
    incomplete_b2_groups = (
        int((~b2["all_enumerations_complete"].astype(bool)).sum()) if not b2.empty else 0
    )
    no_false_candidate_groups = (
        int((b2["false_candidates_total"].astype(float) <= 0).sum()) if not b2.empty else 0
    )
    all_alpha_estimable = bool(b2["alpha_estimable"].astype(bool).all()) if not b2.empty else False
    underpowered_groups = int(b2["underpowered"].astype(bool).sum()) if not b2.empty else 0
    enough_h_classes = (
        b2["H_class"].nunique() >= int(config.get("gate1_min_H_classes", 2))
        if not b2.empty
        else False
    )
    structured_rows = (
        structured_deviations[
            structured_deviations["source"] == "MNGeneralCandidateSource"
        ]
        if not structured_deviations.empty and "source" in structured_deviations
        else pd.DataFrame()
    )
    expected_structured_rows = 0
    if not b2.empty:
        comparison_keys = [
            "n",
            "m",
            "q",
            "r",
            "r_elim",
            "r_check",
            "prior",
            "source",
            "source_requested",
            "evidence_label",
        ]
        for _, group in b2.groupby(comparison_keys, dropna=False):
            if "dense_random" in set(group["H_class"]):
                expected_structured_rows += int((group["H_class"] != "dense_random").sum())
    structured_h_quantified = bool(
        enough_h_classes
        and expected_structured_rows > 0
        and len(structured_rows) == expected_structured_rows
        and structured_rows["comparison_estimable"].astype(bool).all()
    )
    if not eligible:
        status = "NOT_EVALUATED_SMOKE"
    elif not phase_minus1_verified:
        status = "BLOCKED_PHASE_MINUS1"
    elif unavailable_b2 or incomplete_b2_groups:
        status = "BLOCKED_B2_INCOMPLETE"
    elif b2.empty:
        status = "BLOCKED_B2_UNAVAILABLE"
    elif no_false_candidate_groups:
        status = "BLOCKED_B2_NO_FALSE_CANDIDATES"
    elif not all_alpha_estimable or underpowered_groups:
        status = "BLOCKED_B2_UNDERPOWERED"
    else:
        measurable = bool((b2["cluster_bootstrap_alpha_high"] < float(config.get("gate1_alpha_upper_max", 0.95))).any())
        retention = bool(b2["all_present_true_candidates_retained"].all())
        protocol_ready = retention and enough_h_classes and structured_h_quantified
        if not protocol_ready:
            status = "FAIL_PROTOCOL_REQUIREMENT"
        elif measurable:
            status = "PASS"
        else:
            status = "FAIL_NO_MARGINAL_EFFECT"
    return {
        "gate": "Gate 1",
        "eligible_run": eligible,
        "status": status,
        "b2_groups": int(len(b2)),
        "b2_unavailable": unavailable_b2,
        "b2_failure_count": len(b2_failures),
        "b2_incomplete_group_count": incomplete_b2_groups,
        "b2_no_false_candidate_group_count": no_false_candidate_groups,
        "b2_valid_alpha_lists_min": int(b2["valid_alpha_lists"].min()) if not b2.empty else None,
        "b2_false_candidates_total_min": int(b2["false_candidates_total"].min()) if not b2.empty else None,
        "b2_all_alpha_estimable": all_alpha_estimable,
        "b2_underpowered_group_count": underpowered_groups,
        "b2_H_classes": int(b2["H_class"].nunique()) if not b2.empty else 0,
        "b2_structured_h_quantified": structured_h_quantified,
        "b2_true_candidate_presence_rate_min": float(b2["true_candidate_presence_rate"].min()) if not b2.empty else None,
        "b2_true_retention_conditional": bool(b2["all_present_true_candidates_retained"].all()) if not b2.empty else None,
        "gate1_min_valid_alpha_lists": int(config.get("gate1_min_valid_alpha_lists", 30)),
        "gate1_min_false_candidates_per_group": int(config.get("gate1_min_false_candidates_per_group", 300)),
        "theoretical_true_retention_of_exact_predicate": 1.0,
        "requires_phase_minus1_consistency": True,
        "phase_minus1_gate_verified": phase_minus1_verified,
        "phase_minus1_decision": str(phase_minus1_path) if phase_minus1_path else None,
        "headline_proxy_excluded": True,
        "rule": "measurable B2/B4 marginal selectivity, true retention, structured-H quantification, and Phase -1 direction",
    }


def run(config_path: str | Path | dict[str, Any]) -> Path:
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    require_common_limits(config)
    geometry_selection_path = _require_geometry_selection_binding(config)
    gate0_path = _require_gate0(config)
    phase_minus1_verified, phase_minus1_path = _resolve_phase_minus1(config)
    context = RunContext.create("phase1", config, PROJECT_ROOT)
    cells = _expand_cells(config)
    instance_count = int(config["instances"])
    seed_plan = _instance_seed_plan(config, len(cells))
    tasks: list[dict[str, Any]] = []
    seeds: list[int] = []
    index = 0
    serializable_config = {k: v for k, v in config.items() if not k.startswith("_")}
    for cell in cells:
        for instance_index in range(instance_count):
            seed, protocol_seed = seed_plan[index]
            index += 1
            seeds.append(seed)
            task_id = f"{cell['cell_id']}:{instance_index}:{seed}"
            tasks.append(
                {
                    "task_id": task_id,
                    "config": serializable_config,
                    "cell": cell,
                    "instance_index": instance_index,
                    "seed": seed,
                    "protocol_seed": protocol_seed,
                }
            )
    completed = {str(row["task_id"]) for row in read_jsonl(context.run_dir / "checkpoints.jsonl")}
    pending = [task for task in tasks if task["task_id"] not in completed]
    for result in bounded_pool_imap_unordered(
        _run_instance,
        pending,
        workers=min(int(config.get("workers", 1)), max(1, len(pending))),
        max_tasks_per_child=int(config.get("max_tasks_per_child", 1)),
    ):
        append_jsonl(context.run_dir / "trials.jsonl", result["trials"])
        append_jsonl(context.run_dir / "candidate_records.jsonl", result["candidates"])
        append_jsonl(context.run_dir / "failures.jsonl", result["failures"])
        append_jsonl(
            context.run_dir / "checkpoints.jsonl",
            [
                {
                    "task_id": result["task_id"],
                    "seed": result["seed"],
                    "protocol_seed": result["protocol_seed"],
                }
            ],
        )
    trials = read_jsonl(context.run_dir / "trials.jsonl")
    failures = read_jsonl(context.run_dir / "failures.jsonl")
    frame = pd.DataFrame(trials)
    summary = (
        _summarize(
            frame,
            int(config.get("bootstrap_seed", config["master_seed"] + 1)),
            int(config.get("bootstrap_replicates", 2000)),
            minimum_valid_alpha_lists=int(config.get("gate1_min_valid_alpha_lists", 30)),
            minimum_false_candidates_per_group=int(
                config.get("gate1_min_false_candidates_per_group", 300)
            ),
        )
        if not frame.empty
        else pd.DataFrame()
    )
    summary.to_csv(context.run_dir / "summary.csv", index=False)
    structured_deviations = _structured_h_deviations(summary)
    structured_deviations.to_csv(context.run_dir / "structured_h_deviations.csv", index=False)
    decision = _gate_decision(
        config,
        summary,
        structured_deviations,
        failures,
        phase_minus1_verified=phase_minus1_verified,
        phase_minus1_path=phase_minus1_path,
    )
    decision["gate0_decision"] = str(gate0_path)
    decision["geometry_selection_artifact"] = (
        str(geometry_selection_path) if geometry_selection_path else None
    )
    atomic_write_text(context.run_dir / "gate1_decision.json", json.dumps(decision, indent=2) + "\n")
    if decision["status"] == "FAIL_NO_MARGINAL_EFFECT":
        atomic_write_text(
            context.run_dir / "STOP.md",
            "# STOP\n\n- failed gate: Gate 1\n- evidence: see summary.csv cluster-level intervals\n- interpretation: exact discrete prior appears largely absorbed by norm-aware geometry in the tested regime\n- action: do not enter a more complex solver phase\n",
        )
    elif decision["status"] in {
        "BLOCKED_B2_NO_FALSE_CANDIDATES",
        "BLOCKED_B2_UNDERPOWERED",
        "BLOCKED_B2_INCOMPLETE",
        "BLOCKED_B2_UNAVAILABLE",
    }:
        atomic_write_text(
            context.run_dir / "BLOCKED.md",
            "# BLOCKED\n\n"
            f"- blocked gate: Gate 1\n- status: {decision['status']}\n"
            "- evidence: see summary.csv and gate1_decision.json\n"
            "- interpretation: the B2 data do not support an estimable, sufficiently powered marginal-selectivity conclusion\n"
            "- action: do not enter Phase 2; amend and version the measurement protocol before any rerun\n",
        )
    report = (
        "# Phase 1 report\n\n"
        f"- independent synthetic instances: {len(tasks)}\n"
        f"- list-level trial rows: {len(trials)}\n"
        f"- failures/unavailable rows: {len(failures)}\n"
        f"- Gate 1: {decision['status']}\n\n"
        "Primary intervals resample complete candidate lists/instances. Candidate-level Wilson intervals are diagnostic only.\n"
        "`norm_shell_proxy` is excluded from headline B2/B4 conclusions.\n"
    )
    atomic_write_text(context.run_dir / "report.md", report)
    derived_seed_manifest: Any = seeds
    if "instance_seeds" in config:
        derived_seed_manifest = {
            "protocol_instance_seeds": [int(value) for value in config["instance_seeds"]],
            "derived_task_seeds": seeds,
        }
    context.write_manifest(status="COMPLETED", derived_seeds=derived_seed_manifest)
    return context.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="existing phase1 run directory to resume")
    arguments = parser.parse_args()
    if arguments.resume:
        config = load_yaml(arguments.config)
        config["_resume_dir"] = arguments.resume
        print(run(config))
        return
    print(run(arguments.config))


if __name__ == "__main__":
    main()
