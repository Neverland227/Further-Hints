"""Phase 3: tiny gate-locked structured-H factor/MITM cross-check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from affine_hints.config import RunContext, append_jsonl, atomic_write_text, load_yaml, require_common_limits
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.priors import FixedWeightTernaryPrior, make_prior
from affine_hints.structured.factor_graph import min_fill_order
from affine_hints.structured.syndrome_mitm import syndrome_mitm
from affine_hints.structured.variable_elimination import top_k_assignments


def _require_gate2(config):  # noqa: ANN001
    configured = config.get("gate2_decision", "AUTO_LATEST")
    if configured == "AUTO_LATEST":
        candidates = sorted((PROJECT_ROOT / "results" / "phase2").glob("*/gate2_decision.json"), reverse=True)
        path = candidates[0] if candidates else None
    else:
        path = Path(str(configured)).resolve()
    if path is None or not path.exists():
        raise RuntimeError("Gate 2 decision not found")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("classification") != "MEASURABLE SECURITY-ESTIMATOR EFFECT":
        raise RuntimeError("Phase 3 requires Gate 2 = MEASURABLE SECURITY-ESTIMATOR EFFECT")
    return path


def run(config_path: str | Path | dict) -> Path:
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    require_common_limits(config)
    gate2 = _require_gate2(config)
    context = RunContext.create("phase3", config, PROJECT_ROOT)
    if context.is_resume and (context.run_dir / "manifest.json").exists():
        manifest = json.loads((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") == "COMPLETED":
            return context.run_dir
    trials = []
    failures = []
    for index, case in enumerate(config["cases"]):
        try:
            n = int(case["n"])
            if n > 28:
                raise RuntimeError("DEFERRED_OUT_OF_SCOPE: Phase 3 n exceeds 28")
            rng = np.random.default_rng(int(case["seed"]))
            prior = make_prior(case["prior"])
            hint = generate_hint_matrix(
                rng,
                n=n,
                r=int(case["r"]),
                q=int(case["q"]),
                hint_class=str(case["H_class"]),
                parameters=dict(case.get("H_parameters", {})),
            )
            instance = generate_synthetic_lwe(
                rng,
                n=n,
                m=int(case.get("m", n)),
                q=int(case["q"]),
                secret_prior=prior,
                H=hint.H,
                instance_id=f"phase3-{index}",
            )
            order, width = min_fill_order(instance.H, instance.q, int(config["max_treewidth"]))
            top = top_k_assignments(
                H=instance.H,
                ell=instance.ell,
                q=instance.q,
                prior=prior,
                n=n,
                top_k=int(config["max_assignments"]),
                max_factor_states=int(config["max_factor_states"]),
            )
            support = tuple(getattr(prior, "support", (-1, 0, 1)))
            mitm = syndrome_mitm(
                H=instance.H,
                ell=instance.ell,
                q=instance.q,
                support=support,
                max_states=int(config["max_factor_states"]),
                max_matches=int(config["max_assignments"]),
                h_plus=prior.h_plus if isinstance(prior, FixedWeightTernaryPrior) else None,
                h_minus=prior.h_minus if isinstance(prior, FixedWeightTernaryPrior) else None,
            )
            top_set = {assignment for _, assignment in top}
            mitm_set = set(mitm)
            trials.append(
                {
                    "case": index,
                    "n": n,
                    "q": instance.q,
                    "r": len(instance.H),
                    "H_class": case["H_class"],
                    "min_fill_order": order,
                    "induced_width": width,
                    "factor_top_count": len(top),
                    "mitm_count": len(mitm),
                    "bounded_sets_agree": top_set == mitm_set if len(top) < int(config["max_assignments"]) and len(mitm) < int(config["max_assignments"]) else None,
                    "label": "SYNTHETIC STATISTICAL EXPERIMENT / tiny exact cross-check",
                }
            )
        except Exception as exc:
            failures.append({"case": index, "error_type": type(exc).__name__, "error": str(exc)})
    append_jsonl(context.run_dir / "trials.jsonl", trials)
    append_jsonl(context.run_dir / "failures.jsonl", failures)
    pd.DataFrame(trials).to_csv(context.run_dir / "summary.csv", index=False)
    atomic_write_text(
        context.run_dir / "report.md",
        f"# Phase 3 report\n\nGate 2 source: `{gate2}`\n\nCompleted cases: {len(trials)}; bounded/deferred cases: {len(failures)}.\n",
    )
    context.write_manifest(status="COMPLETED", derived_seeds=[case["seed"] for case in config["cases"]])
    return context.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="existing phase3 run directory to resume")
    arguments = parser.parse_args()
    if arguments.resume:
        config = load_yaml(arguments.config)
        config["_resume_dir"] = arguments.resume
        print(run(config))
        return
    print(run(arguments.config))


if __name__ == "__main__":
    main()
