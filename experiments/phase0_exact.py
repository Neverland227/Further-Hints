"""Phase 0: exact algebra, posterior, and representation-invariance checks."""

from __future__ import annotations

import argparse
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

from affine_hints.config import RunContext, append_jsonl, atomic_write_text, load_yaml, read_jsonl, require_common_limits
from affine_hints.coset import AffineCosetElimination
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.modular import centered_vector, matvec_mod
from affine_hints.posterior import original_log_posterior, reduced_log_posterior
from affine_hints.priors import CBDPrior, FixedWeightTernaryPrior, UniformTernaryPrior, make_prior


def _center_secret(values: tuple[int, ...], q: int) -> tuple[int, ...]:
    return tuple(centered_vector(values, q))


def _supported_solution_set(instance: Any, prior: Any, elimination: AffineCosetElimination, max_states: int) -> set[tuple[int, ...]]:
    support = tuple(getattr(prior, "support", (-1, 0, 1)))
    dimension = len(elimination.residual_indices)
    if len(support) ** dimension > max_states:
        raise RuntimeError("RESOURCE_LIMIT: residual representation enumeration exceeds max_exact_states")
    import itertools

    results: set[tuple[int, ...]] = set()
    for residual_signed in itertools.product(support, repeat=dimension):
        residual = tuple(value % instance.q for value in residual_signed)
        if not elimination.remaining_hints_pass(residual):
            continue
        full = _center_secret(elimination.reconstruct(residual), instance.q)
        if prior.in_support(full):
            results.add(full)
    return results


def _run_case(config: dict[str, Any], case: dict[str, Any], seed: int, case_index: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n, m, q, r = (int(case[key]) for key in ("n", "m", "q", "r"))
    prior = make_prior(case["prior"])
    error_prior = UniformTernaryPrior()
    hint = generate_hint_matrix(rng, n=n, r=r, q=q, hint_class=str(case.get("H_class", "dense_random")))
    instance = generate_synthetic_lwe(
        rng,
        n=n,
        m=m,
        q=q,
        secret_prior=prior,
        error_prior=error_prior,
        H=hint.H,
        instance_id=f"phase0-{case_index}-{seed}",
    )
    rows: list[dict[str, Any]] = []
    reference_set: set[tuple[int, ...]] | None = None
    for r_elim in range(r + 1):
        elimination = AffineCosetElimination.build(
            instance.H,
            instance.ell,
            q,
            r_elim,
            strategy=str(case.get("pivot_strategy", "first_unit_minor")),
            rng=rng,
            max_combinations=int(config.get("max_pivot_combinations", 100_000)),
        )
        true_residual = tuple(instance.s[i] % q for i in elimination.residual_indices)
        reconstructed = _center_secret(elimination.reconstruct(true_residual), q)
        hint_ok = elimination.all_hints_pass(true_residual)
        A_star, b_star = elimination.transform_lwe(instance.A, instance.b)
        original_error = tuple(centered_vector(((instance.b[i] - matvec_mod(instance.A, instance.s, q)[i]) % q for i in range(m)), q))
        reduced_error = tuple(centered_vector(((b_star[i] - matvec_mod(A_star, true_residual, q)[i]) % q for i in range(m)), q))
        residual_identity = original_error == reduced_error == instance.e
        original_log = original_log_posterior(
            A=instance.A,
            b=instance.b,
            H=instance.H,
            ell=instance.ell,
            q=q,
            secret=instance.s,
            secret_prior=prior,
            error_prior=error_prior,
        )
        reduced_log = reduced_log_posterior(
            A_star=A_star,
            b_star=b_star,
            elimination=elimination,
            residual_secret=true_residual,
            secret_prior=prior,
            error_prior=error_prior,
        )
        true_posterior_ok = (math.isinf(original_log) and math.isinf(reduced_log)) or abs(original_log - reduced_log) <= 1e-12
        solution_set = _supported_solution_set(instance, prior, elimination, int(config["max_exact_states"]))
        original_logs: list[float] = []
        reduced_logs: list[float] = []
        pointwise_ok = True
        max_log_discrepancy = 0.0
        for full_secret in sorted(solution_set):
            residual_secret = tuple(full_secret[i] % q for i in elimination.residual_indices)
            left = original_log_posterior(
                A=instance.A,
                b=instance.b,
                H=instance.H,
                ell=instance.ell,
                q=q,
                secret=full_secret,
                secret_prior=prior,
                error_prior=error_prior,
            )
            right = reduced_log_posterior(
                A_star=A_star,
                b_star=b_star,
                elimination=elimination,
                residual_secret=residual_secret,
                secret_prior=prior,
                error_prior=error_prior,
            )
            original_logs.append(left)
            reduced_logs.append(right)
            if math.isfinite(left) and math.isfinite(right):
                discrepancy = abs(left - right)
                max_log_discrepancy = max(max_log_discrepancy, discrepancy)
                pointwise_ok &= discrepancy <= 1e-12
            else:
                pointwise_ok &= math.isinf(left) and math.isinf(right) and ((left < 0) == (right < 0))
        finite_indices = [index for index, value in enumerate(original_logs) if math.isfinite(value)]
        if finite_indices:
            from scipy.special import logsumexp

            left_normalizer = float(logsumexp([original_logs[index] for index in finite_indices]))
            right_normalizer = float(logsumexp([reduced_logs[index] for index in finite_indices]))
            total_variation = 0.5 * sum(
                abs(
                    math.exp(original_logs[index] - left_normalizer)
                    - math.exp(reduced_logs[index] - right_normalizer)
                )
                for index in finite_indices
            )
        else:
            total_variation = math.inf
        posterior_ok = true_posterior_ok and pointwise_ok and total_variation <= 1e-12
        if reference_set is None:
            reference_set = solution_set
        representation_ok = solution_set == reference_set
        row = {
            "case_index": case_index,
            "seed": seed,
            "instance_id": instance.instance_id,
            "prior": str(case["prior"]),
            "H_class": hint.hint_class,
            "q": q,
            "n": n,
            "m": m,
            "r": r,
            "r_elim": r_elim,
            "reconstruction_ok": reconstructed == instance.s,
            "hint_invariant_ok": hint_ok,
            "lwe_residual_identity_ok": residual_identity,
            "posterior_equivalence_ok": posterior_ok,
            "posterior_max_log_discrepancy": max_log_discrepancy,
            "posterior_normalized_total_variation": total_variation,
            "representation_invariance_ok": representation_ok,
            "solution_count": len(solution_set),
            "pivot_indices": elimination.pivot_indices,
            "eliminated_rows": elimination.eliminated_rows,
            "status": "PASS",
            "label": "EXACT THEOREM CHECK",
        }
        if not all(
            row[key]
            for key in (
                "reconstruction_ok",
                "hint_invariant_ok",
                "lwe_residual_identity_ok",
                "posterior_equivalence_ok",
                "representation_invariance_ok",
            )
        ):
            row["status"] = "FAIL"
        rows.append(row)
    return rows


def run(config_path: str | Path | dict[str, Any]) -> Path:
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    require_common_limits(config)
    context = RunContext.create("phase0", config, PROJECT_ROOT)
    seed_sequence = np.random.SeedSequence(int(config["master_seed"]))
    tasks = [(case_index, case, child) for case_index, (case, child) in enumerate(zip(config["cases"], seed_sequence.spawn(len(config["cases"]))))]
    derived: list[int] = []
    completed = {int(row["case_index"]) for row in read_jsonl(context.run_dir / "checkpoints.jsonl")}
    for case_index, case, child in tasks:
        seed = int(child.generate_state(1, dtype=np.uint64)[0])
        derived.append(seed)
        if case_index in completed:
            continue
        try:
            case_rows = _run_case(config, case, seed, case_index)
            append_jsonl(context.run_dir / "trials.jsonl", case_rows)
        except Exception as exc:  # fail closed and preserve the counterexample seed
            append_jsonl(
                context.run_dir / "failures.jsonl",
                [{
                    "case_index": case_index,
                    "case": case,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }],
            )
        append_jsonl(context.run_dir / "checkpoints.jsonl", [{"case_index": case_index, "seed": seed}])
    trials = read_jsonl(context.run_dir / "trials.jsonl")
    failures = read_jsonl(context.run_dir / "failures.jsonl")
    frame = pd.DataFrame(trials)
    if not frame.empty:
        frame.to_csv(context.run_dir / "summary.csv", index=False)
    passed = not failures and bool(trials) and all(row["status"] == "PASS" for row in trials)
    decision = {
        "gate": "Gate 0",
        "status": "PASS" if passed else "FAIL",
        "checks": len(trials),
        "failures": len(failures) + sum(row["status"] != "PASS" for row in trials),
        "label": "EXACT THEOREM CHECK",
    }
    atomic_write_text(context.run_dir / "gate0_decision.json", json.dumps(decision, indent=2) + "\n")
    if not passed:
        atomic_write_text(
            context.run_dir / "BUG_REPORT.md",
            "# BUG REPORT\n\nGate 0 failed. See `failures.jsonl` and retained seeds. Later lattice phases must not run.\n",
        )
        atomic_write_text(
            context.run_dir / "STOP.md",
            "# STOP\n\n- failed gate: Gate 0\n- interpretation: implementation correctness unresolved\n- action: do not run Phase 1/2/3 until the minimal counterexample is fixed\n",
        )
    report = (
        "# Phase 0 report\n\n"
        f"- exact checks: {len(trials)}\n"
        f"- recorded exceptions: {len(failures)}\n"
        f"- Gate 0: {decision['status']}\n\n"
        "Every result is labelled `EXACT THEOREM CHECK`; no sampling confidence interval is applicable.\n"
    )
    atomic_write_text(context.run_dir / "report.md", report)
    context.write_manifest(status="COMPLETED" if passed else "BUG", derived_seeds=derived)
    return context.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="existing phase0 run directory to resume")
    arguments = parser.parse_args()
    if arguments.resume:
        config = load_yaml(arguments.config)
        config["_resume_dir"] = arguments.resume
        print(run(config))
        return
    print(run(arguments.config))


if __name__ == "__main__":
    main()
