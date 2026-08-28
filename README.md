# Exact affine hints and discrete LWE priors

This directory contains a synthetic-only, falsification-first experiment framework.
It separates four evidence levels:

- exact modular algebra and posterior checks;
- analytic/heuristic estimator diagnostics;
- synthetic candidate-distribution experiments;
- optional bounded real-lattice calibration at `n <= 80`.

It does **not** contain a general LWE attack interface, does not accept captured or
third-party samples, does not use G6K, and does not execute cryptographic-scale
BKZ or sieving. Inputs are generated inside the experiment scripts.

The core package does not depend on Sage. Optional `fpylll` work is gate-locked
and is intended for a controlled Linux server environment. See
`docs/SERVER_RUNBOOK.md` before running a campaign.

## Local correctness check

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Gate order

```bash
python experiments/phase_minus1_estimator.py --config configs/phase_minus1/default.yaml
python experiments/phase0_exact.py --config configs/phase0/default.yaml
python experiments/phase1_selectivity.py --config configs/phase1/dense_smallq.yaml
python experiments/phase1_selectivity.py --config configs/phase1/coded_dual_q3329.yaml
python experiments/phase1_b2_geometry.py --config configs/phase1/b2_geometry_calibration_v2.yaml
```

The B2 geometry command searches exactly eight preregistered, prior-blind
list geometries. It never runs formal Gate 1 automatically. If it records
`SELECTED`, review the run directory and launch the frozen command in
`RUN_NEXT.md`; if it records `NO_ELIGIBLE_GEOMETRY`, stop without enlarging
that legacy grid. The only authorized revision after that stop is the
one-shot Gate-1c constrained-slice protocol:

```bash
python experiments/phase1c_backend_audit.py --config configs/phase1/gate1c_backend_audit.yaml
```

This first uses only exposed calibration seeds to compare complete legacy
balls with complete fixed-`t=-1` balls. It does not evaluate prior outcomes.
Only an all-pass audit writes a hash-bound held-out confirmation config and a
separate command in `RUN_CONFIRMATION.md`. A resource-capped prefix is never
called certified top-K.

Phase 2 and Phase 3 require explicit gate-decision files from a non-smoke,
gate-eligible run. They fail closed when the prerequisite is absent or does not
pass. `AUTO_LATEST` never searches backward for an older PASS. Phase 4 is estimator-only and labels any point outside its empirical
calibration range as `EXTRAPOLATION`.
