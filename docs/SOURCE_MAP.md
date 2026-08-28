# Source-to-code map

This map separates exact literature constructions, heuristics, project
derivations, proxies, and diagnostics. A module is not promoted to a named
baseline merely because it has a superficially similar matrix shape.

## Local source inventory

| Material | Local artifact | Inspection/use | Status |
|---|---|---|---|
| May--Nowakowski, *Too Many Hints* (ASIACRYPT 2023) | `D:/Further Hints/References/Too_many_hints_MN.pdf` | Section 3, Theorem 3.3 and Eq. (7); Section 4.3 Algorithm 1; Section 5.1 Eqs. (18)--(22), Theorem 5.1 | available and used |
| DDGR, *LWE with Side Information* (CRYPTO 2020) | `D:/Further Hints/References/2020-292_3_27.pdf` | Framework/baseline identity and applicability context only | available; no DDGR implementation claimed |
| Local working-paper snapshot | `D:/Further Hints/References/Working_Paper_1_Version_3 (2).pdf` | Section 5.3, Lemma 3 (`G=P K F_I`) and Proposition 4 (unit-minor elimination) | available and used for the synthetic coded-dual matrix generator only |
| Cao--Feng--Pan, *Refined Attack on LWE with Hints* | `D:/Further Hints/References/Refined_Attack_on_LWE .pdf` | candidate cross-check source identified | available; Cao Z-SIS-LWE baseline not implemented |
| Carrier/MATZOV-related local papers | `D:/Further Hints/References/FullVersion_Assessing the Impact of a Variant of MATZOV’s Dual Attack on Kyber.pdf`, `D:/Further Hints/References/2020-292_3_27.pdf` | scope and cost-model context | available; no G6K/sieve execution |
| Hhan et al., *From Perfect to Approximate Hints* | `D:/Further Hints/Cycle1To3/tmp/pdfs/2026-1081.pdf` and extracted text | sparse-secret approximate/perfect-hint correlation/peeling and empirical/GAA context | inspected, context only; not an exact-affine posterior or MN/Babai baseline |
| Meet-LWE local artifact | `D:/Further Hints/Cycle1To3/tmp/pdfs/acisp2026-meet-lwe-hints.pdf` | header/content check | `UNAVAILABLE`: file is an HTML client-challenge page, not a PDF; no result relies on it |
| Existing Carrier polar implementation | `D:/Further Hints/ExternalCarrierCode/CodedDualAttack` | generator/decoder interfaces and puncturing dependencies | inspected; not reused because the exposed path depends on `polar.o`, randomized decoder construction, and `galois`, and is not a stable pure synthetic-H generator API |
| Bundled older estimator snapshot | `D:/Further Hints/ExternalCarrierCode/CodedDualAttack/OptimizeCodedDualAttack/estimator` | callable/API inventory | available as an older snapshot; not renamed as the current working-paper estimator |
| prompt path `/mnt/data/4444.txt` | no matching local artifact | none | `UNAVAILABLE` |
| prompt path `faster_Dual_Attack.pdf` | no matching local artifact | none | `UNAVAILABLE` |

The local working-paper PDF is a dated snapshot. It is not treated as the
authoritative current manuscript source; no manuscript claim was edited from
this copy.

## Core mapping

| Code | Source or derivation | Label | Fidelity/status |
|---|---|---|---|
| `modular.py`, `coset.py` | MN Section 3, Eq. (7)/Theorem 3.3 for mod-q elimination; the task's explicit partial-elimination equations for arbitrary `r_elim` | `LITERATURE_EXACT` plus `PROJECT_DERIVATION` for the unified partial form | implemented; exact tests over prime and prime-power moduli |
| `posterior.py` | exact change of variables under the affine substitution; full prior is evaluated after reconstruction | `PROJECT_DERIVATION` | implemented; retains fixed-weight coupling and CBD likelihood |
| `priors.py` | standard uniform ternary, centered-binomial, and fixed-weight distributions | `PROJECT_DERIVATION` interface | implemented |
| `hints.py:coded_dual_g_transpose` | local working-paper Section 5.3, Lemma 3: `G=P K F_I`, `H=G^T`, unit maximal minor | algebra: `LITERATURE_EXACT`; chosen prefix/random puncturing distribution: `PROJECT_DIAGNOSTIC` | implemented; does not claim to reproduce Carrier's production puncturing schedule |
| other generators in `hints.py` | finite synthetic matrix families specified by the task | `PROJECT_DIAGNOSTIC` | implemented |
| `candidates/mn_general.py:build_extended_basis` | MN Section 5.1, Eq. (20) | `LITERATURE_EXACT` | implemented and target-membership tested |
| `construct_modular_sublattice` | MN Section 4.3 Algorithm 1, adapted as Section 5 directs to the modular block | basis logic: `LITERATURE_EXACT`; finite scaling grid: `LITERATURE_HEURISTIC` | optional fpylll path; zero block and transform recorded |
| `bounded_schnorr_euchner` | standard GSO coefficient enumeration implemented locally to enforce an in-loop node cap | `PROJECT_DERIVATION` | optional fpylll GSO; exact integer norm recheck; node/radius/solution limits recorded |
| `baselines/no_hints.py` | standard no-hint `(-e,s,-1)` row embedding (MN Eq. (3) orientation) | `LITERATURE_EXACT` construction | B0 construction/certificate implemented; full measured solver remains optional |
| `candidates/synthetic.py` | task-defined synthetic sources and matched-moment shell | synthetic sources: `PROJECT_DIAGNOSTIC`; shell: `PROJECT_PROXY / B2_PROXY` | implemented; proxy excluded from Gate 1 headline logic |
| `diagnostics.py`, Phase-1 list summaries | task-defined large-q, projective, valuation, fixed-weight, and dependence checks | `PROJECT_DIAGNOSTIC` | coordinate histograms for q=3329; character/collision effect intervals; collinear joint rates; list-level overdispersion; CBD AUC/AP remain diagnostic |
| `estimators/list_capacity.py` | conditional rank-only uniform-image reference and exact combinatorial counts | `PROJECT_DIAGNOSTIC` / `RANK_ONLY_REFERENCE` | implemented; not universal candidate reduction |
| `same_second_moment_entropy_proxy` | entropy of a Gaussian with variance `2/3` minus uniform-ternary entropy | `PROJECT_PROXY` | retained as information diagnostic only; never converted to beta/work |
| `estimators/beta_sensitivity.py` | continuous root-Hermite-factor sensitivity and integer finite difference | `LITERATURE_HEURISTIC` / `HEURISTIC ESTIMATOR` | restricted to beta >= 40; scans three `d_eff` conventions |
| `estimators/total_work.py` | task definition of complete expected work | `PROJECT_DERIVATION` | includes generation, predicate, verification, and success probability |
| Phase-2 aggregate comparison | task definition `W=T_total/P_success` applied to paired confirmation clusters | `PROJECT_DERIVATION` / `TOY REAL-LATTICE CALIBRATION` | uses `mean(T)/mean(success)` and retains zero-success instances; five requested endpoints and bootstrap intervals are emitted |
| `structured/*` | bounded factor-graph/min-fill/top-K and syndrome MITM requested by the task | `PROJECT_DIAGNOSTIC` | implemented, Gate-2 locked, `n <= 28` |
| `candidates/g6k_backend.py` | task boundary | `DEFERRED_OUT_OF_SCOPE` | always raises; no G6K import |

## Named-baseline ledger

| Baseline | Status | Reason |
|---|---|---|
| B0 no hints | `partial` | exact embedding/certificate exists; a full measured LLL/enumeration campaign is server-optional |
| B1 rank-only elimination | `implemented` | exact affine elimination, with rank-only assumptions stated |
| B2 MN general full-secret modular hints | `implemented_optional_backend` | exact basis, Algorithm-1 adaptation, bounded project enumeration, LLL/small BKZ, extraction, and costs; requires fpylll and complete resource-guarded runs |
| B2 proxy | `implemented` | explicitly excluded from headline conclusions |
| B3 rank-only plus exact prior | `implemented` | leaf predicate retains full prior coupling |
| B4 norm-aware plus exact prior | `implemented_optional_backend` | paired postfilter on the same B2 candidate list |
| B5 toy MITM | `implemented` | hard state/match bounds and tiny dimensions only |
| DDGR | `UNAVAILABLE` | no faithful local implementation was established |
| Cao Z-SIS-LWE cross-check | `UNAVAILABLE` | not implemented rather than renamed from a weaker construction |
| G6K sieve | `DEFERRED_OUT_OF_SCOPE` | prohibited by protocol |
