"""Phase 4: empirical calibration model and finite full-flow estimator integration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from affine_hints.config import RunContext, append_jsonl, atomic_write_text, load_yaml, require_common_limits


def _latest_phase2() -> Path | None:
    candidates = sorted((PROJECT_ROOT / "results" / "phase2").glob("*/confirmation_effects.csv"), reverse=True)
    return candidates[0] if candidates else None


def _log2add(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log2(2.0 ** (left - maximum) + 2.0 ** (right - maximum))


def run(config_path: str | Path | dict[str, Any]) -> Path:
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    require_common_limits(config)
    context = RunContext.create("phase4", config, PROJECT_ROOT)
    if context.is_resume and (context.run_dir / "manifest.json").exists():
        manifest = json.loads((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") == "COMPLETED":
            return context.run_dir
    specified = config.get("phase2_comparisons", "AUTO_LATEST")
    source = _latest_phase2() if specified == "AUTO_LATEST" else Path(str(specified)).resolve()
    failures = []
    if source is None or not source.exists():
        comparisons = pd.DataFrame()
        failures.append({"classification": "UNAVAILABLE", "component": "empirical_calibration", "reason": "Phase 2 comparisons not found"})
    else:
        try:
            comparisons = pd.read_csv(source)
        except pd.errors.EmptyDataError:
            comparisons = pd.DataFrame()
            failures.append(
                {
                    "classification": "UNAVAILABLE",
                    "component": "empirical_calibration",
                    "reason": "Phase 2 effects file is empty",
                }
            )
    finite = comparisons[
        np.isfinite(comparisons.get("log2_expected_work_difference", pd.Series(dtype=float)))
    ] if not comparisons.empty else comparisons
    if finite.empty:
        model = {
            "status": "UNAVAILABLE",
            "reason": "no finite Phase 2 paired calibration effects",
            "calibration_source": str(source) if source else None,
            "validity_range": None,
        }
    else:
        effects = finite["log2_expected_work_difference"].astype(float)
        regimes = sorted(finite["regime_id"].unique().tolist())
        gain = float(effects.mean())
        validity_records = finite[
            [
                key
                for key in (
                    "regime_id",
                    "n",
                    "m",
                    "q",
                    "r",
                    "prior",
                    "H_class",
                    "baseline_r_elim",
                    "exact_r_elim",
                    "baseline_candidate_budget",
                    "exact_candidate_budget",
                )
                if key in finite.columns
            ]
        ].drop_duplicates().to_dict(orient="records")
        model = {
            "status": "CALIBRATED_TOY_ONLY",
            "inputs": ["n", "m", "q", "r", "r_elim", "prior", "H_class", "baseline", "candidate_backend", "beta", "list_budget"],
            "predicted_residual_expected_work_gain_bits": gain,
            "predicted_residual_expected_work_ratio": 2.0 ** (-gain),
            "uncertainty_interval_empirical_quantiles": [float(effects.quantile(0.025)), float(effects.quantile(0.975))],
            "validity_range": {"regime_ids": regimes, "measured_points": validity_records},
            "calibration_source": str(source),
            "label": "TOY REAL-LATTICE CALIBRATION",
        }
    calibration_dir = PROJECT_ROOT / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(calibration_dir / "residual_prior_gain_model.json", json.dumps(model, indent=2) + "\n")
    rows = []
    gain = float(model.get("predicted_residual_expected_work_gain_bits", 0.0))
    low, high = model.get("uncertainty_interval_empirical_quantiles", [math.nan, math.nan])
    measured_points = (model.get("validity_range") or {}).get("measured_points", [])
    for point in config.get("finite_estimator_grid", []):
        stage1 = float(point["stage1_log2_work"])
        residual = float(point["standard_residual_log2_work"])
        standard_total = _log2add(stage1, residual)
        exact_residual = residual - gain
        exact_total = _log2add(stage1, exact_residual)
        within = any(
            all(
                measured.get(key) == point.get(key)
                for key in ("n", "m", "q", "r", "prior", "H_class")
            )
            and int(point.get("r_elim", -1))
            in {
                int(measured.get("baseline_r_elim", -2)),
                int(measured.get("exact_r_elim", -3)),
            }
            for measured in measured_points
        )
        rows.append(
            {
                **point,
                "within_empirical_range_requested": bool(point.get("within_empirical_range", False)),
                "within_empirical_range_verified": within,
                "standard_total_log2_work": standard_total,
                "exact_prior_residual_log2_work": exact_residual,
                "exact_prior_total_log2_work": exact_total,
                "full_flow_gain_bits": standard_total - exact_total,
                "residual_gain_interval_low": low,
                "residual_gain_interval_high": high,
                "label": "HEURISTIC ESTIMATOR" if within else "HEURISTIC ESTIMATOR / EXTRAPOLATION",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(context.run_dir / "summary.csv", index=False)
    append_jsonl(context.run_dir / "trials.jsonl", rows)
    failures.append(
        {
            "classification": "UNAVAILABLE",
            "component": "current_working_paper_estimator_direct_adapter",
            "reason": "no stable callable API was identified in this repository; finite common-prefix integration is reported separately and not renamed as the paper estimator",
        }
    )
    append_jsonl(context.run_dir / "failures.jsonl", failures)
    atomic_write_text(
        context.run_dir / "report.md",
        "# Phase 4 report\n\n"
        f"Calibration status: {model['status']}\n\n"
        "This phase executes no cryptographic-scale lattice attack. The finite-grid common-prefix calculation is a HEURISTIC ESTIMATOR. "
        "Points outside the toy calibration range are additionally labelled EXTRAPOLATION.\n",
    )
    context.write_manifest(status="COMPLETED", derived_seeds={"bootstrap": config.get("bootstrap_seed")})
    return context.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="existing phase4 run directory to resume")
    arguments = parser.parse_args()
    if arguments.resume:
        config = load_yaml(arguments.config)
        config["_resume_dir"] = arguments.resume
        print(run(config))
        return
    print(run(arguments.config))


if __name__ == "__main__":
    main()
