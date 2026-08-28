# Model assumptions and interpretation limits

## Exact statements checked by code

- A selected unit minor gives `s_I = c - C s_J (mod q)`.
- Substitution preserves the LWE error exactly.
- Remaining hint rows become the derived residual system.
- Evaluating the full signed secret prior after reconstruction preserves CBD
  likelihood and fixed-weight plus/minus coupling.
- MN Eq. (20) contains `(-e,0,s,-1)` for a consistent modular-hint instance.
- With full hint elimination, the fixed embedding slice `t=-1` is exactly the
  affine lattice of `(A*s-b+q*k, s, -1)` with `H*s=ell (mod q)`.
- A residual candidate is canonicalized by centered full-secret and centered
  LWE-error lifts. Its intrinsic B2 score is `1+||s||^2+||e||^2`; ordering uses
  score first and residual-secret lexicographic order second.

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
- BKZ beta changes only efficiency for a complete fixed-radius candidate set;
  it is not treated as a candidate-distribution parameter in Gate-1c.

## Gate-1c candidate definition

The primary endpoint is a certified unique residual-secret top-K, not a raw
lattice-vector prefix. Each public score threshold is enumerated completely in
the constrained `t=-1` slice. The first complete ball containing at least K
canonical candidates certifies nested `K={16,32,64,128}`. A node, wall-time,
memory, or raw-vector guard hit censors the whole instance. Threshold selection
does not use realized true-secret norm or prior-pass outcomes.

## Candidate dependence

Primary intervals resample complete instances/candidate lists. Candidate-level
Wilson intervals are diagnostics. Projective classes, collinear-pair fraction,
collinear/non-collinear joint pass rates, cross-list overdispersion, and
prime-power valuations are recorded to expose dependence. CBD AUC/average
precision are diagnostic values whose uncertainty is clustered by complete
candidate list.

Gate-1c reports candidate-weighted alpha, instance-weighted alpha, and the
list-level probability of any false survivor. The first is the primary marginal
selectivity estimand; the list-level value is reported but is not substituted
for it. Zero-survivor results retain Wilson upper bounds. Bootstrap uncertainty
resamples complete lists, not individual dependent candidates.

## Cost interpretation

An estimator decrease is not a measured speedup. Complete toy work includes
basis, reduction, list generation, predicate, verification, and success
probability. Expected work is estimated across confirmation clusters as
`mean(total time)/mean(success)`, so zero-success instances remain in the
denominator rather than being discarded. Phase-2 findings are labelled
`TOY REAL-LATTICE CALIBRATION` and
cannot establish a Kyber/ML-KEM attack, asymptotic speedup, or real-world break.
Gate-1c leaf-postfilter selectivity is information evidence only. It is not an
expected-work improvement and cannot authorize the existing Phase-2 grid
unchanged.
