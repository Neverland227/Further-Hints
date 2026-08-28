# Model assumptions and interpretation limits

## Exact statements checked by code

- A selected unit minor gives `s_I = c - C s_J (mod q)`.
- Substitution preserves the LWE error exactly.
- Remaining hint rows become the derived residual system.
- Evaluating the full signed secret prior after reconstruction preserves CBD
  likelihood and fixed-weight plus/minus coupling.
- MN Eq. (20) contains `(-e,0,s,-1)` for a consistent modular-hint instance.

## Conditional references

`(|S|/q)^r` is used only when the candidate distribution is sufficiently
independent of a dense hint image and uniform in that image. Structured-H,
norm-aware, and correlated candidate lists are measured separately. A zero
survivor count is reported with an interval upper bound, never as probability
zero.

## Heuristics

- Root-Hermite sensitivity is continuous and restricted to beta >= 40.
- Integer beta changes use a finite difference after the continuous solve.
- The sieve database expression is analytic and marked `NOT EXECUTED`.
- The matched-second-moment shell is a project proxy and cannot support a B2/B4
  headline conclusion.
- The MN scaling parameter is selected only from a preregistered finite grid.

## Candidate dependence

Primary intervals resample complete instances/candidate lists. Candidate-level
Wilson intervals are diagnostics. Projective classes, collinear-pair fraction,
collinear/non-collinear joint pass rates, cross-list overdispersion, and
prime-power valuations are recorded to expose dependence. CBD AUC/average
precision are diagnostic values whose uncertainty is clustered by complete
candidate list.

## Cost interpretation

An estimator decrease is not a measured speedup. Complete toy work includes
basis, reduction, list generation, predicate, verification, and success
probability. Expected work is estimated across confirmation clusters as
`mean(total time)/mean(success)`, so zero-success instances remain in the
denominator rather than being discarded. Phase-2 findings are labelled
`TOY REAL-LATTICE CALIBRATION` and
cannot establish a Kyber/ML-KEM attack, asymptotic speedup, or real-world break.
