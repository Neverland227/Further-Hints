"""Phase -1: analytic candidate-capacity and beta-sensitivity scan."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from affine_hints.config import RunContext, atomic_write_jsonl, atomic_write_text, load_yaml, require_common_limits
from affine_hints.estimators.beta_sensitivity import beta_sensitivity_band
from affine_hints.estimators.list_capacity import candidate_capacity


def _r_values(n: int, config: dict[str, Any]) -> list[int]:
    maximum = min(int(config.get("r_max", 64)), int(math.floor(float(config.get("r_fraction_max", 0.15)) * n)))
    values = set(range(1, maximum + 1)) if config.get("all_r", True) else set()
    values.update(int(value) for value in config.get("r_explicit", [1, 2, 3, 4, 5]))
    for fraction in config.get("r_fractions", [0.01, 0.025, 0.05, 0.10, 0.15]):
        values.add(max(1, int(math.floor(float(fraction) * n))))
    return sorted(value for value in values if 1 <= value <= maximum)


def run(config_path: str | Path | dict[str, Any]) -> Path:
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    require_common_limits(config)
    context = RunContext.create("phase_minus1", config, PROJECT_ROOT)
    if context.is_resume and (context.run_dir / "manifest.json").exists():
        manifest = json.loads((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") == "COMPLETED":
            return context.run_dir
    rows: list[dict[str, Any]] = []
    analytic_beta = int(config.get("analytic_baseline_beta", 60))
    beta_cost_slope = float(config.get("heuristic_beta_cost_slope", 0.292))
    for n in (int(value) for value in config["n"]):
        m = int(round(float(config.get("m_over_n", 1.0)) * n))
        lattice_dimension = m + n + 1
        for q in (int(value) for value in config["q"]):
            for r in _r_values(n, config):
                for prior_spec in config["priors"]:
                    if isinstance(prior_spec, str):
                        prior_name = prior_spec
                        fixed = None
                    else:
                        prior_name = str(prior_spec["kind"])
                        fixed = dict(prior_spec)
                        fixed["n"] = n
                    capacities = candidate_capacity(q=q, r=r, prior=prior_name, fixed_weight=fixed)
                    for capacity in capacities:
                        if capacity.label != "RANK_ONLY_REFERENCE" or not math.isfinite(capacity.bits) or capacity.bits < 0:
                            sensitivities = []
                        else:
                            sensitivities = beta_sensitivity_band(
                                capacity_bits=capacity.bits,
                                baseline_beta=analytic_beta,
                                dimensions=(
                                    ("lattice_dimension", lattice_dimension),
                                    ("bkz_beta", analytic_beta),
                                    ("sieve_dimension_proxy", int(config.get("sieve_dimension_proxy", analytic_beta))),
                                ),
                            )
                        if not sensitivities:
                            rows.append(
                                {
                                    "n": n,
                                    "m": m,
                                    "q": q,
                                    "r": r,
                                    **capacity.as_dict(),
                                    "model": None,
                                    "label": capacity.label,
                                }
                            )
                        for sensitivity in sensitivities:
                            record = {
                                "n": n,
                                "m": m,
                                "q": q,
                                "r": r,
                                **capacity.as_dict(),
                                **sensitivity.as_dict(),
                            }
                            record["capacity_label"] = capacity.label
                            record["predicted_log2_work_gain"] = max(
                                0.0,
                                sensitivity.predicted_delta_beta_integer * beta_cost_slope
                                - float(config.get("predicate_overhead_bits", 0.05)),
                            )
                            record["work_model_label"] = "HEURISTIC ESTIMATOR / beta-cost proxy"
                            rows.append(record)
    frame = pd.DataFrame(rows)
    atomic_write_jsonl(context.run_dir / "trials.jsonl", rows)
    frame.to_csv(context.run_dir / "summary.csv", index=False)
    primary = frame[
        (frame["metric"].isin(["hard_support_selectivity", "support_selectivity", "exact_residual_profile"]))
        & (frame["model"] == "bkz_beta")
    ].copy()
    if not primary.empty:
        _plot_scatter(
            primary,
            x="r",
            y="predicted_delta_beta_integer",
            path=context.run_dir / "figures" / "contours_delta_beta.png",
            ylabel="Predicted integer beta reduction (blocks)",
        )
        _plot_scatter(
            primary,
            x="r",
            y="predicted_log2_work_gain",
            path=context.run_dir / "figures" / "contours_work_gain.png",
            ylabel="Predicted log2 work gain (heuristic bits)",
        )
        small = primary[primary["r"] <= 5]
        _plot_scatter(
            small,
            x="r",
            y="predicted_delta_beta_integer",
            path=context.run_dir / "figures" / "coded_dual_r_le_5.png",
            ylabel="Predicted integer beta reduction (blocks)",
        )
    measurable = frame[
        frame["model"].isin(["lattice_dimension", "bkz_beta", "sieve_dimension_proxy"])
        & frame["predicted_delta_beta_integer"].fillna(0).gt(0)
        & frame["n"].le(int(config.get("toy_n_max", 64)))
    ]
    grouped = measurable.groupby(["n", "q", "r", "prior", "metric"])["model"].nunique() if not measurable.empty else pd.Series(dtype=int)
    predicted_regions = int((grouped >= 2).sum())
    eligible = bool(config.get("eligible_for_gate", False)) and not bool(config.get("smoke", False))
    decision = {
        "gate": "Gate -1",
        "eligible_run": eligible,
        "status": "PASS" if eligible and predicted_regions else ("NO_MEASURABLE_REGION" if eligible else "NOT_EVALUATED_SMOKE"),
        "regions_with_two_or_more_same-direction_models": predicted_regions,
        "rule": "two or more d_eff models predict a positive integer beta change in a toy-scale region",
        "evidence_label": "HEURISTIC ESTIMATOR",
    }
    atomic_write_text(context.run_dir / "gate_minus1_decision.json", json.dumps(decision, indent=2) + "\n")
    report = (
        "# Phase -1 report\n\n"
        "All rows are analytic diagnostics. No lattice backend was executed.\n\n"
        f"- rows: {len(frame)}\n"
        f"- gate-eligible: {eligible}\n"
        f"- two-model toy regions: {predicted_regions}\n"
        f"- decision: {decision['status']}\n\n"
        "The rank-only formula is conditional on candidate/hint independence and uniformity in the relevant mod-q image.\n"
    )
    atomic_write_text(context.run_dir / "report.md", report)
    atomic_write_text(context.run_dir / "failures.jsonl", "")
    context.write_manifest(status="COMPLETED", derived_seeds={"master_seed": config["master_seed"]})
    return context.run_dir


def _plot_scatter(frame: pd.DataFrame, *, x: str, y: str, path: Path, ylabel: str) -> None:
    figure, axis = plt.subplots()
    for label, group in frame.groupby("prior"):
        axis.scatter(group[x], group[y], label=label, alpha=0.65)
    axis.set_xlabel("Number of exact hint rows r")
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    atomic_write_text(path.with_suffix(".md"), f"{ylabel} versus r; points are HEURISTIC ESTIMATOR outputs.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="existing phase_minus1 run directory to resume")
    arguments = parser.parse_args()
    if arguments.resume:
        config = load_yaml(arguments.config)
        config["_resume_dir"] = arguments.resume
        print(run(config))
        return
    print(run(arguments.config))


if __name__ == "__main__":
    main()
