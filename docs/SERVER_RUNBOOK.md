# AMD EPYC Turin server runbook

Target: 64 cores, 128 GiB RAM. The core package does not require Sage.

## 1. Environment

```bash
cd research/affine_prior
git rev-parse HEAD
git status --short
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,lattice]"
python experiments/server_preflight.py --config configs/server_turin.yaml
python -m pytest -q
```

Run the formal campaign from a Git checkout so every manifest can record the
commit and dirty state. The current Windows source directory is not itself a
Git checkout; do not transfer only loose files without first placing them in a
versioned server checkout. Each run also records `source_tree_sha256` as an
independent source/configuration fingerprint.

The preflight should report `ready_for_core=true` and
`ready_for_phase2=true`. The latter requires fpylll. Do not substitute Sage for
missing fpylll.

With fpylll installed, the four locally skipped MN/enumeration tests must execute;
do not start B2 if the test run still reports an fpylll-related skip or failure.

## 2. Resource allocation

- Synthetic/statistical Phase 1: 48 spawn workers, one BLAS thread each, two
  tasks per process. Estimated 48 GiB worker RSS plus 24 GiB reserve.
- B2 geometry calibration, formal B2, and Phase 2: 6 workers, one task per
  process, 10 GiB address-space cap per worker, a 600-second task alarm for
  B2, and a hard enumeration node limit.
- Sage: disabled. If a later isolated check genuinely needs Sage, use at most 4
  concurrent processes, one task per process, 12 GiB/process, 600 seconds, and
  discard the process after that one task. Never run a long-lived Sage pool.

The Python worker initializer fixes OpenMP/OpenBLAS/MKL/NumExpr to one thread,
preventing 48 processes from each creating a 64-thread BLAS team. The caps are
set in the parent before spawn, so they are already active while each child is
importing NumPy/fpylll.

Frozen workload bounds:

- `dense_smallq`: 8,640 instance tasks and at most about 1.04e8 synthetic
  candidates;
- `coded_dual_q3329`: 600 instance tasks and at most 2.4e7 candidates;
- B2 geometry calibration: exactly 160 one-process lattice tasks (4 regimes,
  8 frozen geometries, and 5 calibration seeds);
- formal B2 Gate-1b: 120 one-process lattice tasks, launched separately only
  after one geometry passes the calibration rule;
- Phase 2: 320 calibration tasks, then at most 200 confirmation tasks.

The Phase-2 grid is intentionally finite: 2 beta values, 1 scaling value, 2
radius values, 2 budgets, 2 elimination choices, and 2 pivot strategies. Do
not enlarge it after seeing confirmation output.

## 3. Gate-ordered execution

```bash
python experiments/phase_minus1_estimator.py --config configs/phase_minus1/default.yaml
python experiments/phase0_exact.py --config configs/phase0/default.yaml
python experiments/phase1_selectivity.py --config configs/phase1/dense_smallq.yaml
python experiments/phase1_selectivity.py --config configs/phase1/coded_dual_q3329.yaml
python experiments/phase1_b2_geometry.py --config configs/phase1/b2_geometry_calibration_v2.yaml
```

The geometry stage evaluates exactly eight preregistered choices: beta 10 or
20 and radius multiplier 1.25, 1.5, 2.0, or 2.5. It observes list geometry,
truth presence, completeness, and resource use, but it does not evaluate the
prior predicate or Gate-1 effect. The five calibration seeds and thirty
reserved confirmation seeds are disjoint.

Inspect `geometry_selection.json` and `geometry_summary.csv` in the printed
run directory. If the status is `SELECTED`, run the command recorded in that
directory's `RUN_NEXT.md`; it uses the generated, frozen `gate1b_config.yaml`.
Do not edit that YAML after seeing formal output. If the status is
`NO_ELIGIBLE_GEOMETRY`, stop and do not enlarge the grid.

The earlier `configs/phase1/b2_gate_toy.yaml` is retained only for historical
reproduction; it is not the new formal protocol.

After the separately launched formal B2 run, inspect its
`gate1_decision.json`. Run Phase 2 only if its status is exactly `PASS`:

```bash
python experiments/phase2_lattice.py --config configs/phase2/toy_confirmation.yaml
```

Run Phase 3 only if `gate2_decision.json` says exactly
`MEASURABLE SECURITY-ESTIMATOR EFFECT`:

```bash
python experiments/phase3_structured.py --config configs/phase3/tiny_structured.yaml
```

Estimator integration is last:

```bash
python experiments/phase4_full_flow.py --config configs/phase4/default.yaml
```

`AUTO_LATEST` means the newest decision, not the newest historical PASS. A
newer failed or smoke gate therefore blocks continuation. To bind a campaign
to a specific audited artifact, replace `AUTO_LATEST` in a copied YAML config
with that decision file's absolute path; the copied config is preserved in the
run directory.

## 4. Resume after interruption

Every command prints its versioned run directory. Re-run the same config and
pass that directory:

```bash
python experiments/phase1_selectivity.py \
  --config configs/phase1/dense_smallq.yaml \
  --resume results/phase1/<run_id>
```

For an interrupted geometry calibration, use its original config and directory:

```bash
python experiments/phase1_b2_geometry.py \
  --config configs/phase1/b2_geometry_calibration_v2.yaml \
  --resume results/phase1_geometry/<run_id>
```

The same `--resume` option exists for all phase scripts. Geometry, Phase 1,
and Phase 2 skip checkpointed task IDs; Phase 2 never reselects confirmation
parameters after `confirmation_preregistration.json` exists.

## 5. Stop conditions

- Any Gate-0 failure: stop all later phases.
- B2 unavailable, truncated, or resource-limited: Gate 1 is blocked, not passed
  using the proxy.
- B2 lists with no false candidates: Gate 1 is
  `BLOCKED_B2_NO_FALSE_CANDIDATES`; an empty alpha denominator is not evidence
  of zero marginal effect.
- Too few finite alpha lists or fewer than the configured number of false
  candidates: Gate 1 is `BLOCKED_B2_UNDERPOWERED`.
- No geometry satisfying the frozen completeness, truth-presence, and
  false-candidate requirements across all four regimes: stop before formal
  Gate 1b and do not add new parameter points after observing the result.
- No norm-aware marginal information: write STOP and do not enlarge beta,
  radius, dimension, or worker count.
- No confirmed total expected-work improvement: keep the result as STOP or a
  modeling result; do not develop a faster solver.
- Fewer than 30 paired confirmation instances in either preregistered regime:
  Gate 2 is `STOP` and the regime is marked underpowered.
