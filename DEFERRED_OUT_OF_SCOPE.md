# Deferred and unavailable components

## DEFERRED_OUT_OF_SCOPE

- G6K and cryptographic-scale sieving: prohibited by the experiment protocol;
  the backend always raises before execution.
- Cryptographic-scale BKZ, secret recovery against standard parameters, and
  arbitrary external LWE input: not implemented.
- Phase 3 before a measured Gate-2 effect: fail-closed by the runner.
- Sage: not required or invoked. Any future local Sage check must use a fresh,
  single-task bounded process.

## UNAVAILABLE

- Generic randomized-Babai campaign: the bounded adapter exists, but no
  experiment-specific basis-to-secret mapper is supplied. It raises
  `UNAVAILABLE` rather than pretending to be B2. The faithful small B2 path is
  the MN construction plus project-owned bounded enumeration.
- Faithful DDGR baseline: no faithful implementation was established in this
  checkout, so no weaker baseline is renamed DDGR.
- Cao Z-SIS-LWE cross-check: not implemented.
- Direct callable adapter to the current working-paper estimator: no stable API
  was identified here. Phase 4 reports a separate finite common-prefix
  calculation and records this gap.
- Prompt materials `/mnt/data/4444.txt` and `faster_Dual_Attack.pdf`: no matching
  local files were found.

These gaps limit baseline breadth and external-estimator integration. They do
not affect the exact affine algebra tests, but they must remain visible when
interpreting any later server result.
