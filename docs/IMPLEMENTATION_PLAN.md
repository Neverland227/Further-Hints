# Implementation and gate plan

## Frozen scope

- Inputs are generated synthetic LWE instances only.
- Exact algebra uses `A: m x n`, column secrets, and Python integers.
- Real-lattice work is capped at `n <= 80`, `beta <= 30`, a finite YAML grid,
  explicit radius/candidate/node/time/RSS limits, and no automatic expansion.
- G6K and cryptographic-scale recovery are out of scope.

## Gate order

1. Phase -1 writes information and beta/work sensitivity diagnostics. Only
   `RANK_ONLY_REFERENCE` capacity enters radius/beta conversion.
2. Phase 0 checks reconstruction, residual identity, posterior equality,
   representation invariance, and prime-power unit pivots. Any failure writes
   `BUG_REPORT.md` and `STOP.md`.
3. Phase 1 measures list-cluster selectivity, structured-H deviations,
   uniformity effects, projective clustering, valuations, and true retention.
   A B2 proxy can never pass Gate 1. Gate 1 resolves an actual latest Phase -1
   decision artifact; a YAML boolean cannot certify it.
4. Phase 2 reads an explicit Gate-1 PASS file, selects each arm only on
   calibration seeds, freezes `confirmation_preregistration.json`, and then
   runs confirmation seeds. Resume never rewrites that selection. Parameter
   selection and confirmation use aggregate success probability, retaining
   failed candidate-recovery instances in expected-work accounting.
5. Phase 3 runs only if Gate 2 is exactly
   `MEASURABLE SECURITY-ESTIMATOR EFFECT`.
6. Phase 4 is estimator-only and labels every point outside empirical range as
   `EXTRAPOLATION`.

## Checkpoint behavior

Phase 0 checkpoints each exact case. Phase 1 and Phase 2 checkpoint each
complete instance/list task immediately as workers finish. Resume uses
`--resume <existing-run-dir>`, validates the recorded YAML, and schedules only
missing task IDs. Failures remain in `failures.jsonl`. Phase 2 preserves an
existing confirmation preregistration verbatim.

## Completion boundary in this checkout

Local work is limited to unit tests and smoke runs. Formal 30-instance and
50-instance results are not generated on this Windows machine. Optional
fpylll enumeration tests are skipped locally and must pass on the Linux server
before B2/B4 can contribute to a gate decision.
